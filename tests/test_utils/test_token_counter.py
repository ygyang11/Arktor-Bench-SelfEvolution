from __future__ import annotations

from arktor_bench.utils.token_counter import clip_tokens, count_tokens


def test_count_tokens() -> None:
    assert count_tokens("") == 0
    assert count_tokens("hello world") >= 2
    assert count_tokens("a b c d e f g") > count_tokens("a b")


def test_clip_keeps_head_tail() -> None:
    text = " ".join(f"word{i}" for i in range(500))
    clipped = clip_tokens(text, 30)
    assert "[...truncated" in clipped
    assert count_tokens(clipped) < count_tokens(text)
    assert clipped.startswith("word0")
    assert clipped.rstrip().endswith("word499")


def test_passthrough_when_short() -> None:
    text = "just a short string"
    assert clip_tokens(text, 1000) == text
