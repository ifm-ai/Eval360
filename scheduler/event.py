import asyncio
import copy
import pydantic
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from .model import ModelInstance
from .task import Task, ImportedDatasetTask, AsyncGenerationTask

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import logging
logger = logging.getLogger("Event")
logging.basicConfig(level=logging.INFO)


class EventInstance(pydantic.BaseModel):
    """Base class for all evaluation events."""
    model_config = pydantic.ConfigDict(frozen=True)
    uuid: str
    parent_uuid: str
    model: str
    task_uuid: str
    path_to_scores: str


class GradingEventInstance(EventInstance):
    """Event for the standard generate→grade pipeline."""
    path_to_generations: str
    path_to_grades: str
    grader_type: str
    parser_type: str | None = None

    @pydantic.model_validator(mode='after')
    def check_grade_generation_different(self):
        if self.path_to_generations == self.path_to_grades:
            raise ValueError(f'Generation and grade paths are not different: {self.path_to_grades}')
        return self

    @pydantic.model_validator(mode='after')
    def check_score_generation_different(self):
        if self.path_to_generations == self.path_to_scores:
            raise ValueError(f'Generation and scores paths are not different: {self.path_to_scores}')
        return self

    @pydantic.model_validator(mode='after')
    def check_score_grades_different(self):
        if self.path_to_grades == self.path_to_scores:
            raise ValueError(f'Grades and scores paths are not different: {self.path_to_scores}')
        return self


class ImportedDatasetEventInstance(EventInstance):
    """Event for imported datasets that run external benchmark code directly."""
    pass


class ParentEvent(pydantic.BaseModel):
    uuid: str
    model: str
    task_uuids: list[str]
    base_path: str


@dataclass
class DeploymentInfo:
    model: ModelInstance
    # TODO: don't ignore this
    priority: int
    generation_events: set[EventInstance]
    grader_events: set[EventInstance]


class EventManager:
    def __init__(self, db_manager):
        # TODO: also store the asyncio tasks here
        self.desired_models_dict: dict[str, DeploymentInfo] = {}
        self.asyncio_task_dict: dict[str, list] = {} # map from event object to asyncio tasks
        self._queued_events: set = set()  # events enqueued but not yet picked up by handle_event
        self._db_manager = db_manager
        self._launch_queue = asyncio.Queue()
        self._deferred_enqueues: ContextVar[
            list[EventInstance] | None
        ] = ContextVar("eval360_deferred_enqueues", default=None)

    @contextmanager
    def defer_enqueues(self):
        """Stage queue entries until their surrounding DB transaction commits."""
        pending: list[EventInstance] = []
        token = self._deferred_enqueues.set(pending)
        try:
            yield pending
        finally:
            self._deferred_enqueues.reset(token)

    async def create_events_for_model_and_tasks(self, model_instance: ModelInstance,
                                                tasks: list[Task]):
        logger.info(f"model_instance: {model_instance}")
        base_path = model_instance.output_path
        base_path = Path(base_path)
        parent_uuid = ""
        # TODO: replace model name with model uuid (that depends on model config as well)
        for task in tasks:
            if not self._tags_match(model_instance.tag, task.tag):
                logger.debug(
                    "Skipping task %s for model %s due to incompatible tags (model=%s, task=%s)",
                    task.uuid, model_instance.name, model_instance.tag, task.tag
                )
                continue
            existing_events = self._db_manager.get_event_by_model_and_task(model_instance.name, task.uuid)
            if existing_events:
                for event_instance in existing_events:
                    if event_instance in self.asyncio_task_dict:
                        logger.debug(f"Event {event_instance} already in asyncio_task_dict, skipping re-enqueue")
                        continue
                    await self.enqueue(event_instance)
            elif self._db_manager.event_exists_for_model_and_task(
                model_instance.name,
                task.uuid,
            ):
                # A completed event is intentionally absent from
                # get_event_by_model_and_task so it cannot be re-enqueued.
                # Its durable row still makes exact registration replay a
                # no-op instead of creating duplicate work.
                logger.debug(
                    "Completed event already exists for model %s and task %s",
                    model_instance.name,
                    task.uuid,
                )
            else:
                path_to_scores = base_path / f"{task.dataset_name}_scores.yaml"
                if isinstance(task, ImportedDatasetTask):
                    event_instance = ImportedDatasetEventInstance(
                        uuid=str(uuid.uuid4())[:8],
                        parent_uuid=parent_uuid,
                        model=model_instance.name,
                        task_uuid=task.uuid,
                        path_to_scores=str(path_to_scores),
                    )
                else:
                    path_to_generations = base_path / f"{task.dataset_name}_generations.jsonl"
                    path_to_grades = base_path / f"{task.dataset_name}_grades.jsonl"
                    event_instance = GradingEventInstance(
                        uuid=str(uuid.uuid4())[:8],
                        parent_uuid=parent_uuid,
                        model=model_instance.name,
                        task_uuid=task.uuid,
                        path_to_generations=str(path_to_generations),
                        path_to_grades=str(path_to_grades),
                        path_to_scores=str(path_to_scores),
                        grader_type=task.grader.type,
                        parser_type=model_instance.parser_type,
                    )
                self._db_manager.register_event(event_instance)
                assert (event_instance not in self.asyncio_task_dict)
                await self.enqueue(event_instance)

    async def create_events_for_new_model(self, model_instance: ModelInstance):
        logger.info(f"creating events for new model: {model_instance}")
        tasks = self._db_manager.get_all_tasks()
        await self.create_events_for_model_and_tasks(model_instance, tasks)

    async def create_events_for_new_tasks(self, tasks: list[Task]):
        for model_name, model_instance in self._db_manager.get_all_models().items():
            await self.create_events_for_model_and_tasks(model_instance, tasks)

    @staticmethod
    def _tags_match(model_tag: str, task_tag: str) -> bool:
        return model_tag == "any" or task_tag == "any" or model_tag == task_tag

    def remove_dead_async_tasks(self, event_instance=None):
        if event_instance and event_instance in self.asyncio_task_dict:
            tasks = self.asyncio_task_dict[event_instance]
            self.asyncio_task_dict[event_instance] = [(ty, task) for ty, task in tasks if not task.done()]
        else:
            for event_instance, tasks in self.asyncio_task_dict.items():
                self.asyncio_task_dict[event_instance] = [(ty, task) for ty, task in tasks if not task.done()]
            self.asyncio_task_dict = {
                event_instance: tasks for event_instance, tasks in self.asyncio_task_dict.items() if tasks}

    def get_async_tasks(self, event_instance):
        if event_instance not in self.asyncio_task_dict:
            return []
        return copy.copy(self.asyncio_task_dict[event_instance])

    def add_async_task(self, event_instance, task_type, task):
        self._queued_events.discard(event_instance)  # now actively tracked
        if event_instance not in self.asyncio_task_dict:
            self.asyncio_task_dict[event_instance] = []
        self.asyncio_task_dict[event_instance].append((task_type, task))

    def get_desired_models(self):
        return copy.copy(self.desired_models_dict)

    def remove_desired_model(self, event, is_grading):
        if not is_grading:
            # this condition can be true if the model fails
            if event.model not in self.desired_models_dict:
                return
            deployment_info = self.desired_models_dict[event.model]
            if event in deployment_info.generation_events:
                deployment_info.generation_events.remove(event)
            if not deployment_info.generation_events and not deployment_info.grader_events:
                del self.desired_models_dict[event.model]
        else:
            task = self._db_manager.get_task(event.task_uuid)
            if not isinstance(task, AsyncGenerationTask) or task.grader.llm_as_judge is None:
                model_key = event.model
            else:
                model_key = task.grader.llm_as_judge.name
            if model_key not in self.desired_models_dict:
                return
            deployment_info = self.desired_models_dict[model_key]
            if event in deployment_info.grader_events:
                deployment_info.grader_events.remove(event)
            if not deployment_info.generation_events and not deployment_info.grader_events:
                del self.desired_models_dict[model_key]

    def add_desired_model(self, event_instance, is_grading):
        if event_instance.model not in self.desired_models_dict:
            self.desired_models_dict[event_instance.model] = DeploymentInfo(
                model=self._db_manager.get_model(event_instance.model),
                priority=0,
                generation_events=set(),
                grader_events=set())
        if not is_grading:
            self.desired_models_dict[event_instance.model].generation_events.add(event_instance)
        else:
            task = self._db_manager.get_task(event_instance.task_uuid)
            if not isinstance(task, AsyncGenerationTask) or task.grader.llm_as_judge is None:
                self.desired_models_dict[event_instance.model].grader_events.add(event_instance)
                return
            model = task.grader.llm_as_judge
            if model.name not in self.desired_models_dict:
                self.desired_models_dict[model.name] = DeploymentInfo(
                    model=model,
                    priority=0,
                    generation_events=set(),
                    grader_events=set())
            self.desired_models_dict[model.name].grader_events.add(event_instance)

    # TODO REFACTOR: add a occasional job that removes desired models if there are no active tasks for it

    async def register_processing_event(self, event_instance, phase):
        # phase = 0: generation not yet complete
        # phase = 1: generation complete, grading not yet complete
        # phase = 2: grading complete
        # phase = -1: failed
        # TODO: move this to an enum
        logger.info(f"registering event {event_instance} with phase {phase}")
        if phase == 0:
            self._processing_events[event_instance] = 0
        elif phase == 1:
            self._db_manager.update_event(event_instance.uuid, 1)
        elif phase == 2:
            self._db_manager.update_event(event_instance.uuid, 2)
        elif phase == -1:
            logger.info(f"registering failed event: {event_instance}")
            self._db_manager.update_event(event_instance.uuid, -1)

    async def fail_all_events_with_models(self, dead_models, progress_manager=None):
        dead_models = set(dead_models)
        for event_instance, asyncio_tasks in self.asyncio_task_dict.items():
            if event_instance.model in dead_models:
                if progress_manager is not None:
                    progress_manager.set_status(event_instance, "Failed")
                self.desired_models_dict.pop(event_instance.model, None)
                await self.register_processing_event(event_instance, -1)
                # TODO POSTPOC: have a set of active async tasks for this event, and cancel them here

    def enqueue_nowait(self, generation_event):
        if (
            generation_event in self._queued_events
            or generation_event in self.asyncio_task_dict
        ):
            logger.debug(f"Enqueue: skipping already-queued/active event {generation_event}")
            return
        self._queued_events.add(generation_event)
        logger.info(f"Enqueue: {generation_event}")
        self._launch_queue.put_nowait(generation_event)

    async def enqueue(self, generation_event):
        pending = self._deferred_enqueues.get()
        if pending is not None:
            if (
                generation_event not in pending
                and generation_event not in self._queued_events
                and generation_event not in self.asyncio_task_dict
            ):
                pending.append(generation_event)
            return
        self.enqueue_nowait(generation_event)

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._launch_queue.get()
