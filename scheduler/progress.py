from tqdm import tqdm
import numpy as np
import shutil

from .task import AsyncGenerationTask, ImportedDatasetTask

SHADES = np.array(list(" ▁▂▃▄▅▆▇█"))


class ProgressManager:
    def __init__(self, db_manager):
        self._events = {}
        self._db_manager = db_manager

    def _name_widths(self):
        """Compute column widths from the longest active model/dataset names, min 10."""
        if not self._events:
            return 10, 10
        model_w = max(len(e.model) for e in self._events)
        dataset_w = max(
            len(self._db_manager.get_task(e.task_uuid).dataset_name)
            for e in self._events
        )
        return max(model_w, 10), max(dataset_w, 10)

    def _to_desc(self, event, task, mode, status=""):
        model_w, dataset_w = self._name_widths()
        return (
            f"{event.model[:model_w]:<{model_w}} | "
            f"{task.dataset_name[:dataset_w]:<{dataset_w}} | "
            f"{mode:<10} | "
            f"{status:<10}"
        )

    def _bar_width(self, desc):
        """Compute bar width so the full line fits exactly within the terminal.

        Fixed overhead = len(desc) + separators around the bar (2) +
        percentage field (5: ' 100%') + counts (16, covers up to 99999/99999) +
        elapsed/rate (' [MM:SS<MM:SS, 9999.99it/s]').
        Using a fixed count width keeps all bars the same width regardless of total.
        """
        term_width = shutil.get_terminal_size().columns
        overhead = len(desc) + 5 + 2 + 16 + 28
        return max(10, term_width - overhead)

    def _bar_format(self, bar_width):
        return (
            "{desc} {percentage:3.0f}%|{bar:" + str(bar_width) + "}| "
            "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

    def _create_bar(self, event, mode):
        task = self._db_manager.get_task(event.task_uuid)
        desc = self._to_desc(event, task, mode)
        idx = len(self._events) - 1  # stub already added in initialize()
        if mode == "Generation":
            total = (task.num_generations or 0) * max(task.average_over + task.pass_at)
            bar = tqdm(total=total,
                       desc=desc,
                       bar_format=self._bar_format(self._bar_width(desc)),
                       position=2*idx)
        if mode == "Grading":
            total = task.num_generations or 0
            bar = tqdm(total=total,
                       desc=desc,
                       bar_format=self._bar_format(self._bar_width(desc)),
                       position=2*idx+1)
        return bar

    def update_bar(self, bar, generation_chunks, width):
        width = min(width, len(generation_chunks)) if len(generation_chunks) > 0 else width
        chunks = np.array_split(generation_chunks, width)
        densities = np.array([c.mean() if len(c) > 0 else 0.0 for c in chunks])
        np.nan_to_num(densities, copy=False, nan=0.0)
        idx = (densities * (len(SHADES) - 1)).astype(int)
        custom = ''.join(SHADES[idx])
        bar.bar_format = (
            "{desc} {percentage:3.0f}%|" + custom + "| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        )

    def _create_status_bar(self, event):
        task = self._db_manager.get_task(event.task_uuid)
        desc = self._to_desc(event, task, "Job")
        idx = len(self._events) - 1  # stub already added in initialize()
        return tqdm(total=1,
                    desc=desc,
                    bar_format=self._bar_format(self._bar_width(desc)),
                    position=2*idx)

    def initialize(self, event):
        task = self._db_manager.get_task(event.task_uuid)
        if event in self._events:
            return
        # Register the event before creating bars so _name_widths() includes
        # this event's name lengths when computing column widths for the new bars.
        self._events[event] = {}
        if isinstance(task, ImportedDatasetTask):
            bar = self._create_status_bar(event)
            self._events[event] = {"Generation": bar, "Grading": bar,
                                   "status_Generation": "", "status_Grading": ""}
        elif isinstance(task, AsyncGenerationTask):
            generation_bar = self._create_bar(event, "Generation")
            grading_bar = self._create_bar(event, "Grading")
            self._events[event] = {
                "Generation": generation_bar,
                "Grading": grading_bar,
                "generation_chunks": np.zeros(
                    (task.num_generations or 0) * max(task.average_over + task.pass_at),
                    dtype=bool),
                "status_Generation": "",
                "status_Grading": "",
            }
        self._refresh_all_existing_bars()

    def _refresh_all_existing_bars(self):
        """Refresh all bar descriptions to use the current (max) column widths."""
        for ev, bars in self._events.items():
            if not bars:
                continue  # stub not yet populated
            task = self._db_manager.get_task(ev.task_uuid)
            if isinstance(task, ImportedDatasetTask):
                continue  # single job bar, not affected by model/dataset column widths
            for mode in ("Generation", "Grading"):
                if mode in bars:
                    status = bars.get(f"status_{mode}", "")
                    bars[mode].set_description_str(self._to_desc(ev, task, mode, status))
                    bars[mode].refresh()

    def get_status(self, event, mode="Generation") -> str:
        """Return the current status string for the given event and mode, or '' if unknown."""
        bars = self._events.get(event)
        if not bars:
            return ""
        return bars.get(f"status_{mode}", "")

    def set_status(self, event, status, mode="Generation"):
        """Update the status column in the given mode bar's description.

        The status (e.g. 'Queued', 'Deploying', 'Complete', 'Failed') is
        displayed as a fixed-width column alongside the model, dataset, and
        mode columns, e.g.:
            qwen3-4b-it      | mmlu            | Generation | Complete   100%|...|
        The actual tqdm bar always shows real progress — no text is placed
        inside the fill area.
        """
        self.initialize(event)
        if event not in self._events:
            return
        task = self._db_manager.get_task(event.task_uuid)
        self._events[event][f"status_{mode}"] = status
        bar = self._events[event][mode]
        bar.set_description_str(self._to_desc(event, task, mode, status))
        bar.refresh()

    def update_generation_total(self, event, task):
        """Update the generation bar's total after num_generations becomes known (e.g. post-download)."""
        if event not in self._events:
            return
        if isinstance(task, ImportedDatasetTask):
            return
        total = task.num_generations * max(task.average_over + task.pass_at)
        bar = self._events[event]["Generation"]
        bar.total = total
        bar.refresh()
        grading_bar = self._events[event]["Grading"]
        grading_bar.total = task.num_generations
        grading_bar.refresh()
        self._events[event]["generation_chunks"] = np.zeros(total, dtype=bool)

    def update(self, event, index, completed, new_elems, mode):
        task = self._db_manager.get_task(event.task_uuid)
        self.initialize(event)
        if event not in self._events:
            return
        if isinstance(task, ImportedDatasetTask):
            return

        if not new_elems:
            new_elems = completed

        bar = self._events[event][mode]
        if mode == "Generation":
            total = task.num_generations * max(task.average_over + task.pass_at)
            desc = bar.desc
            bar_width = self._bar_width(desc)
            ind = max(task.average_over + task.pass_at) * index
            self._events[event]["generation_chunks"][ind:ind+completed] = True
            self.update_bar(bar, self._events[event]["generation_chunks"], bar_width)
            bar.update(new_elems)
        else:
            total = task.num_generations
            desc = bar.desc
            bar_width = min(self._bar_width(desc), total) if total else self._bar_width(desc)
            bar.bar_format = self._bar_format(bar_width)
            bar.update(new_elems)
