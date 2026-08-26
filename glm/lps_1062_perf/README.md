# LPS-1062 — GLM-5.2 B300 throughput optimization

## Working conventions

- Write human-authored dates and times in PDT using 24-hour time.
- Do not convert timestamps embedded in traces, logs, or other generated
artifacts.
- Do experimental work inside `runs/` by default.

## Layout

- `glm52_moe_layer_diagrams/` — diagrams retained for personal learning.
- `traces/` — the most important and current traces. Copy traces to here. Jack decides which traces are promoted here.
- `tools/` — a lean collection of actual, reusable files used across multiple
runs. Do not place symlinks here.
- `configs/` — a lean collection of actual, important configurations reused
across multiple runs. Do not place symlinks here. Jack decides which configs
are promoted here.
- `runs/` — the default workspace for experiments and investigations. Intended primarily for agent use. You can choose yourself what you want to keep/delete in here.

## Run structure

- Create one self-contained directory per run:
`runs/<task_description>_YYYYMMDD/`.
- Keep all files associated with a run inside its directory.
- Every run must contain a `WORKLOG.md`.
- Prefix each worklog entry with `YYYYMMDD HH:MM PDT`.
- The internal structure of a run is otherwise flexible.

## Promotion rule

Work freely inside `runs/`. Do not create, modify, rename, or delete anything
inside `traces/`, `tools/`, or `configs/` unless Jack explicitly requests it.
Artifacts move from a run into those curated directories only through explicit
promotion.