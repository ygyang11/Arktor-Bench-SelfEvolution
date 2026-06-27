from __future__ import annotations

import json
from pathlib import Path

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

# The candidate's server runs inside the sandbox, whose network the host cannot reach; the HTTP
# probing therefore happens inside the sandbox. Each check ships driver.py via a heredoc and runs
# one scenario against a fresh `python app.py`. driver.py is not a workspace_file: the candidate
# never sees it.
_DRIVER = (Path(__file__).parent / "driver.py").read_text(encoding="utf-8")
_EOF = "ARKTOR_T001_DRIVER_EOF"


async def _run(ws: Workspace, scenario: str) -> tuple[float, str]:
    cmd = (f"cat > /tmp/_t001_drv.py <<'{_EOF}'\n{_DRIVER}\n{_EOF}\n"
           f"python /tmp/_t001_drv.py {scenario}")
    out = (await ws.run(cmd, check=False)).stdout
    line = next((ln for ln in reversed(out.splitlines()) if ln.strip().startswith("{")), "")
    try:
        verdict = json.loads(line)
        return float(verdict["score"]), str(verdict.get("msg", ""))
    except (ValueError, KeyError, TypeError):
        return 0.0, "the service could not be exercised — did it start and listen on $PORT?"


@check(id="shorten_valid_201")
async def shorten_valid_201(ws: Workspace) -> tuple[float, str]:
    return await _run(ws, "shorten_valid_201")


@check(id="code_deterministic")
async def code_deterministic(ws: Workspace) -> tuple[float, str]:
    return await _run(ws, "code_deterministic")


@check(id="shorten_idempotent_200")
async def shorten_idempotent_200(ws: Workspace) -> tuple[float, str]:
    return await _run(ws, "shorten_idempotent_200")


@check(id="alias_create_and_idempotent")
async def alias_create_and_idempotent(ws: Workspace) -> tuple[float, str]:
    return await _run(ws, "alias_create_and_idempotent")


@check(id="alias_collision_409")
async def alias_collision_409(ws: Workspace) -> tuple[float, str]:
    return await _run(ws, "alias_collision_409")


@check(id="alias_rejected_400")
async def alias_rejected_400(ws: Workspace) -> tuple[float, str]:
    return await _run(ws, "alias_rejected_400")


@check(id="redirect_and_hit_counting")
async def redirect_and_hit_counting(ws: Workspace) -> tuple[float, str]:
    return await _run(ws, "redirect_and_hit_counting")


@check(id="stats_known")
async def stats_known(ws: Workspace) -> tuple[float, str]:
    return await _run(ws, "stats_known")


@check(id="list_pagination")
async def list_pagination(ws: Workspace) -> tuple[float, str]:
    return await _run(ws, "list_pagination")


@check(id="list_boundaries")
async def list_boundaries(ws: Workspace) -> tuple[float, str]:
    return await _run(ws, "list_boundaries")


@check(id="list_sort_orders")
async def list_sort_orders(ws: Workspace) -> tuple[float, str]:
    return await _run(ws, "list_sort_orders")


@check(id="invalid_query_400")
async def invalid_query_400(ws: Workspace) -> tuple[float, str]:
    return await _run(ws, "invalid_query_400")


@check(id="delete_then_gone")
async def delete_then_gone(ws: Workspace) -> tuple[float, str]:
    return await _run(ws, "delete_then_gone")


@check(id="unknown_code_404")
async def unknown_code_404(ws: Workspace) -> tuple[float, str]:
    return await _run(ws, "unknown_code_404")


@check(id="top_n_ranking")
async def top_n_ranking(ws: Workspace) -> tuple[float, str]:
    return await _run(ws, "top_n_ranking")


@check(id="bad_request_matrix")
async def bad_request_matrix(ws: Workspace) -> tuple[float, str]:
    return await _run(ws, "bad_request_matrix")


@check(id="method_and_reserved_paths")
async def method_and_reserved_paths(ws: Workspace) -> tuple[float, str]:
    return await _run(ws, "method_and_reserved_paths")


@check(id="json_content_type")
async def json_content_type(ws: Workspace) -> tuple[float, str]:
    return await _run(ws, "json_content_type")
