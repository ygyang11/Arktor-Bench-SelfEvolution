from __future__ import annotations

from functools import lru_cache

import tiktoken

_ENCODING = "o200k_base"


@lru_cache(maxsize=2)
def _encoding(name: str = _ENCODING) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def clip_tokens(text: str, limit: int) -> str:
    """Keep head + tail within `limit` tokens, marking the elided middle."""
    enc = _encoding()
    toks = enc.encode(text)
    if len(toks) <= limit:
        return text
    head, tail = limit * 2 // 3, limit // 3
    return (f"{enc.decode(toks[:head])}\n"
            f"[...truncated {len(toks) - head - tail} tokens...]\n"
            f"{enc.decode(toks[-tail:])}")
