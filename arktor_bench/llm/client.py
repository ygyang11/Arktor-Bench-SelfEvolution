from __future__ import annotations

import asyncio
import json
import random
from types import TracebackType
from typing import TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from arktor_bench.config import ModelEndpoint

T = TypeVar("T", bound=BaseModel)

_RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)
_FATAL = (AuthenticationError, PermissionDeniedError, NotFoundError)


class FatalLLMError(RuntimeError):
    """A config/auth error that recurs identically for every call — abort the
    whole run rather than contain it per cell"""


class StructuredLLM:
    def __init__(self, ep: ModelEndpoint, *, max_retries: int = 4, timeout: float = 180.0) -> None:
        self._ep = ep
        self._max_retries = max_retries
        self._client = AsyncOpenAI(
            base_url=ep.base_url, api_key=ep.api_key or "-", timeout=timeout, max_retries=0,
        )

    async def __aenter__(self) -> StructuredLLM:
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None,
                        exc: BaseException | None, tb: TracebackType | None) -> None:
        await self._client.close()

    async def complete(self, prompt: str, out: type[T], *, system: str | None = None,
                       context: dict[str, object] | None = None) -> T:
        schema = json.dumps(out.model_json_schema(), ensure_ascii=False)
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": (
            f"{prompt}\n\n"
            "Respond with a single JSON object that conforms exactly to the JSON Schema "
            "below: include every required field, give each value the specified type and "
            "keep it within any stated constraint (enum, minimum, maximum), and add no "
            "fields beyond the schema. Output only the raw JSON object — no explanation, "
            f"no markdown, no code fences.\n\nJSON Schema:\n{schema}"
        )})
        last: Exception | None = None
        for attempt in range(self._max_retries + 1):
            content = ""
            try:
                resp = await self._client.chat.completions.create(  # type: ignore[call-overload]
                    model=self._ep.model, messages=msgs,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or ""
                return out.model_validate_json(content, context=context)
            except _FATAL as e:
                raise FatalLLMError(f"{self._ep.base_url} ({self._ep.model}): {e}") from e
            except _RETRYABLE as e:
                last = e
            except ValidationError as e:
                last = e
                msgs.append({"role": "assistant", "content": content})
                msgs.append({"role": "user", "content": (
                    f"Your previous response failed schema validation:\n{e}\n\n"
                    "Return the corrected, complete JSON object: fix exactly the reported "
                    "problems, keep every other field unchanged, and output only the raw "
                    "JSON — no explanation, no markdown, no code fences."
                )})
            if attempt < self._max_retries:
                await asyncio.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
        raise RuntimeError(
            f"structured completion failed after {self._max_retries + 1} attempts "
            f"(model={self._ep.model}): {last}"
        )
