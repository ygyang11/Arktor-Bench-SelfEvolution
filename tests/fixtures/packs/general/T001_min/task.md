---
id: T001_min
name: Min Task
labels:
  domain: software_engineering
  subdomain: cli_tool
  model_capability: [code]
  harness_focus: [instructions]
  complexity: linear
  multimodal: false
  online: false
workspace_files: []
---

## Prompt

Write a file `answer.txt` whose content is the word ok.

## Auto Checks

- {id: c_ok, desc: "answer.txt contains ok", weight: 2}

## Judge Rubric

- id: c_quality
  desc: "Overall quality of the deliverable"
  weight: 1
  levels:
    - {score: 0.0, desc: "missing or wrong"}
    - {score: 1.0, desc: "correct and clean"}
