# Iteration 09 / task 2 — TabPFN-TS status (adamex-svk)

**Status: UNRESOLVED (no complete forecasts).**

## Local check

A search of this worktree's `iterations/` found no `tabpfn-ts-v2` parquet files at all: no
`tabpfn-ts-v2__test__cut{0..3}__padded.parquet` and no `tabpfn-ts-v2__test__final__padded.parquet`.
(The only artifact known from T5 was a 200-article debug run in another checkout, which is not a usable
forecast.)

## Remote check

The GPU machine is reached only through a person-provided SSH host. No host or key is available to this
session, and credentials must never be written into the repo. The five old T5 processes were therefore
not inspected or restarted. T10 already reported them as gone; the status remains unknown, not "running".

## Consequence

TabPFN could not be scored alone, added as a fourth member, or substituted for any D40 member. It remains
untested and does not block the other approaches. If the full padded parquets appear later, the standard
evaluation is:

```bash
.venv/bin/python src/foundation/score_all.py
.venv/bin/python src/meta/ensemble_recipe.py evaluate --models moirai-2.0-small,tirex,chronos-t5-small,tabpfn-ts-v2
```

The Prior Labs licence restriction for newer weights still applies: local/ungated-v2 only, no accounts,
tokens or hosted inference.
