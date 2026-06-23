from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace


@check(id="c_ok")
async def c_ok(ws: Workspace) -> tuple[float, str]:
    text = ws.read_text("answer.txt") or ""
    return (1.0, "") if "ok" in text else (0.0, "answer.txt missing 'ok'")
