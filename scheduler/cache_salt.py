from __future__ import annotations

import copy
import secrets
from typing import Any, Literal

import pydantic


class CacheSaltConfig(pydantic.BaseModel):
    """Cache-key salt behavior for OpenAI-compatible providers."""
    model_config = pydantic.ConfigDict(extra="forbid", validate_assignment=True)

    mode: Literal["disabled", "static", "unique"] = "disabled"
    salt: str | None = None

    @pydantic.field_validator("mode", mode="before")
    @classmethod
    def normalize_mode(cls, v: Any) -> str:
        if v is None:
            return "disabled"
        if not isinstance(v, str):
            raise ValueError(f"Cannot parse cache_salt.mode: {v}")
        return v.strip().lower()

    @pydantic.field_validator("salt", mode="before")
    @classmethod
    def normalize_salt(cls, v: Any) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError(f"Cannot parse cache_salt.salt: {v}")
        return v.strip()

    @pydantic.model_validator(mode="after")
    def check_mode_salt_combination(self):
        if self.mode == "static":
            if not self.salt:
                raise ValueError('cache_salt.salt is required when cache_salt.mode is "static"')
        elif self.salt is not None:
            raise ValueError(
                'cache_salt.salt may only be set when cache_salt.mode is "static"'
            )
        return self


def request_kwargs_with_cache_salt(
    openai_kwargs: dict[str, Any] | None,
    cache_salt: CacheSaltConfig,
) -> dict[str, Any]:
    if cache_salt is None:
        raise ValueError("cache_salt must be a CacheSaltConfig; use mode='disabled'")

    request_kwargs = copy.deepcopy(openai_kwargs or {})
    if "cache_salt" in request_kwargs:
        raise ValueError(
            "openai kwargs must not set cache_salt directly; use model.cache_salt"
        )

    extra_body = request_kwargs.get("extra_body", {})
    if extra_body is None:
        request_kwargs.pop("extra_body", None)
        extra_body = {}
    elif not isinstance(extra_body, dict):
        raise ValueError("openai kwargs extra_body must be a mapping")
    elif "cache_salt" in extra_body:
        raise ValueError(
            "openai kwargs must not set extra_body.cache_salt directly; "
            "use model.cache_salt"
        )

    if cache_salt.mode == "disabled":
        return request_kwargs

    extra_body = copy.deepcopy(extra_body)
    if cache_salt.mode == "static":
        extra_body["cache_salt"] = cache_salt.salt
    elif cache_salt.mode == "unique":
        extra_body["cache_salt"] = secrets.token_urlsafe(32)
    else:
        raise ValueError(f"Unsupported cache_salt mode: {cache_salt.mode}")
    request_kwargs["extra_body"] = extra_body
    return request_kwargs
