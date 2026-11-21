# Iteration 09 / task 2 — TabPFN-TS status (adamex-svk)

**Status: UNRESOLVED (no complete forecasts).**

## Local check

A search of this worktree's `iterations/` found no `tabpfn-ts-v2` parquet files at all: no
`tabpfn-ts-v2__test__cut{0..3}__padded.parquet` and no `tabpfn-ts-v2__test__final__padded.parquet`.
(The only artifact known from T5 was a 200-article debug run in another checkout, which is not a usable
forecast.)

## Remote check

The GPU machine is reachable via a person-provided SSH host. Access was obtained late in the iteration
and used for the distribution-pooling quantile export (task 3), but TabPFN was not re-run: no complete
`tabpfn-ts-v2` forecasts exist in this worktree. The it08 verdict stands (rejected, live -5.2); this task
remains unresolved here rather than "running".

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
