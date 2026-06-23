from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

_TODO = "python todo.py"           # pinned entry: no PATH / install / chmod corner cases


async def _reset(ws: Workspace) -> None:
    await ws.run("rm -f todo.json", check=False)        # each check starts from empty state


def _item(ws: Workspace, text: str) -> dict | None:
    """The persisted item (ground truth from todo.json) whose text matches, else None."""
    data = ws.read_json("todo.json")
    if not isinstance(data, list):
        return None
    return next((i for i in data if isinstance(i, dict) and i.get("text") == text), None)


@check(id="add_persists")
async def add_persists(ws: Workspace) -> tuple[float, str]:
    await _reset(ws)
    await ws.run(f"{_TODO} add milk", check=False)
    item = _item(ws, "milk")                             # state from todo.json, not from `list`
    if item is None:
        return 0.0, "`add` did not persist the item to todo.json"
    return (1.0, "") if not item.get("done") \
        else (0.1, "the item is persisted but its `done` flag is not false on creation")


@check(id="done_persists")
async def done_persists(ws: Workspace) -> tuple[float, str]:
    await _reset(ws)
    await ws.run(f"{_TODO} add milk", check=False)       # id 1
    await ws.run(f"{_TODO} add eggs", check=False)        # id 2
    if _item(ws, "milk") is None or _item(ws, "eggs") is None:
        return 0.0, "precondition failed: the items were not persisted"
    await ws.run(f"{_TODO} done 1", check=False)          # must mark id 1 (milk) only, keep both
    milk, eggs = _item(ws, "milk"), _item(ws, "eggs")
    if milk is None or eggs is None:
        return 0.0, "`done` removed an item instead of marking it"
    if milk.get("done") and not eggs.get("done"):
        return 1.0, ""
    if milk.get("done"):
        return 0.1, "`done` marked the target but also marked the other item"
    return 0.0, "`done` did not set the target item's done flag"


@check(id="rm_persists")
async def rm_persists(ws: Workspace) -> tuple[float, str]:
    await _reset(ws)
    await ws.run(f"{_TODO} add milk", check=False)       # id 1
    await ws.run(f"{_TODO} add eggs", check=False)        # id 2
    if _item(ws, "milk") is None or _item(ws, "eggs") is None:
        return 0.0, "precondition failed: the items were not persisted"
    await ws.run(f"{_TODO} rm 1", check=False)
    gone, kept = _item(ws, "milk") is None, _item(ws, "eggs") is not None
    if gone and kept:
        return 1.0, ""
    if gone:
        return 0.1, "`rm` removed the target but also dropped the other item"
    return 0.0, "`rm` did not remove the target item"


@check(id="list_displays")
async def list_displays(ws: Workspace) -> tuple[float, str]:
    await _reset(ws)
    await ws.run(f"{_TODO} add milk", check=False)       # id 1
    await ws.run(f"{_TODO} add eggs", check=False)        # id 2
    await ws.run(f"{_TODO} done 1", check=False)          # milk done, eggs not
    if _item(ws, "milk") is None or _item(ws, "eggs") is None:
        return 0.0, "precondition failed: items were not persisted before listing"
    lines = (await ws.run(f"{_TODO} list", check=False)).stdout.splitlines()
    milk_line = next((ln for ln in lines if "milk" in ln), "")
    eggs_line = next((ln for ln in lines if "eggs" in ln), "")
    if not (milk_line and eggs_line):
        return 0.0, "`list` does not show items"
    if "[x]" in milk_line and "[x]" not in eggs_line:    # marker on the done item only
        return 1.0, ""
    return 0.1, "`list` shows the items but `[x]` is on the wrong item or missing"


@check(id="bad_input_exit2")
async def bad_input_exit2(ws: Workspace) -> tuple[float, str]:
    await _reset(ws)
    if not ws.exists("todo.py"):                         # no program -> file-not-found also exits 2; don't credit it
        return 0.0, "todo.py was not created"
    cases = {
        "add without text": f"{_TODO} add",
        "done without an id": f"{_TODO} done",
        "rm without an id": f"{_TODO} rm",
        "done with a non-integer id": f"{_TODO} done abc",
        "rm with a non-integer id": f"{_TODO} rm xyz",
    }
    bad: list[str] = []
    for label, cmd in cases.items():
        try:  # one flaky invocation must not sink the rest
            code = (await ws.run(cmd, check=False)).exit_code
        except Exception as e:  # noqa: BLE001
            bad.append(f"{label} could not run ({e})")
            continue
        if code != 2:
            bad.append(f"{label} exited {code}")
    if not bad:
        return 1.0, ""
    return round((len(cases) - len(bad)) / len(cases), 2), "expected exit 2 — " + "; ".join(bad)
