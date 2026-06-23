---
id: T001_todo_cli
name: Todo CLI
labels:
  domain: software_engineering
  subdomain: cli_tool
  model_capability: [code]
  harness_focus: [instructions, tools]
  complexity: composite
  multimodal: false
  online: false
workspace_files: []
---

## Prompt

Implement a command-line todo manager, runnable as `python todo.py <subcommand>`:

- `add <text>` — add an item; items get a stable integer id starting at 1.
- `list` — print each item with its id and text; mark done items with `[x]`.
- `done <id>` — mark the item done (it stays in the list).
- `rm <id>` — remove the item.

Persist state as a JSON array in `./todo.json`, one object per item:
`{"id": <int>, "text": <str>, "done": <bool>}`. Exit with code 2 on invalid input
(e.g. `add` with no text, or a missing / non-integer id).

## Auto Checks

- {id: add_persists, desc: "add persists the item to todo.json with done=false", weight: 2}
- {id: done_persists, desc: "done sets that item's done flag in todo.json", weight: 1}
- {id: rm_persists, desc: "rm removes that item from todo.json while others remain", weight: 1}
- {id: list_displays, desc: "list shows the items, marking done ones with [x]", weight: 2}
- {id: bad_input_exit2, desc: "invalid input exits with code 2", weight: 1}

## Judge Rubric

- id: code_organization
  desc: "Code organization, regardless of file count: clear command dispatch with persistence kept separate from command logic"
  weight: 1
  levels:
    - {score: 0.0, desc: "One function does it all — argument parsing, dispatch, logic, and file I/O tangled together"}
    - {score: 0.25, desc: "A long logic over the subcommand, with command logic and file reads/writes inlined throughout"}
    - {score: 0.5, desc: "Each subcommand has its own handler, but load/save is duplicated or interleaved inside them"}
    - {score: 0.75, desc: "Dispatch is clean and a load/save helper exists, but the boundary leaks, e.g. handlers re-read or re-write the store themselves instead of receiving the loaded state"}
    - {score: 1.0, desc: "Clean subcommand-to-handler dispatch over an isolated load/save layer; responsibilities cleanly separated"}

- id: error_handling
  desc: "How invalid input and edge cases are handled — controlled exits, useful diagnostics, and not corrupting stored state"
  weight: 1
  levels:
    - {score: 0.0, desc: "Invalid input crashes with an uncaught exception or exits with the wrong code"}
    - {score: 0.25, desc: "Invalid input exits 2 instead of crashing, but with no diagnostic message, and an unknown or out-of-range id still crashes"}
    - {score: 0.5, desc: "Invalid input exits 2 with a clear diagnostic, but an unknown or out-of-range id on done/rm still crashes"}
    - {score: 0.75, desc: "Invalid input and an unknown or out-of-range id are both handled with clear diagnostics and correct exits; only a missing or malformed todo.json can still crash"}
    - {score: 1.0, desc: "All of the above, and a missing or malformed todo.json is handled without crashing or corrupting existing data"}
