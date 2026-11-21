# T5 — TabPFN-TS as ensemble member: BLOCKED

Checked at 2026-11-21 14:33 UTC (GPU machine via SSH tunnel).

## Verdict

BLOCKED: TabPFN jobs still running. No usable forecasts exist yet, so no
scoring, no `ensemble_recipe.py evaluate`, no adoption decision was possible.
The jobs were left untouched (not restarted, not killed) as instructed.

## Job state

`ps -eo pid,etime,time,pcpu,args` (all five alive, state `Rl`):

| PID | cmd | elapsed | CPU time | %CPU |
|---|---|---|---|---|
| 38996 | `run_model.py tabpfn-ts-v2 test --cut-index 0` | 03:03:51 | 08:42:06 | 283 |
| 38998 | `run_model.py tabpfn-ts-v2 test --final` | 03:03:51 | 08:42:49 | 284 |
| 39001 | `run_model.py tabpfn-ts-v2 test --cut-index 3` | 03:03:51 | 08:52:41 | 289 |
| 39002 | `run_model.py tabpfn-ts-v2 test --cut-index 1` | 03:03:51 | 08:43:49 | 284 |
| 39007 | `run_model.py tabpfn-ts-v2 test --cut-index 2` | 03:03:51 | 08:43:17 | 284 |

`~/hack/logs/tabpfn_pids.txt` = `38996 38998 39001 39002 39007` (matches).
`~/hack/logs/tabpfn_resume.txt` = `resumed 12:24:12` (jobs were re-launched/resumed).
All five per-job logs (`run_tabpfn_cut{0..3}.txt`, `run_tabpfn_final.txt`) are 0 bytes
(stdout likely buffered), so there is no per-step progress signal.

## Forecast artifacts

`iterations/03-foundation-model/forecasts/` contains only the debug run from 11:28:

```
tabpfn-ts-v2__test__cut0__padded__limit200.json
tabpfn-ts-v2__test__cut0__padded__limit200.parquet   (48 KB, 200 articles only)
```

- No `tabpfn-ts-v2__test__cut{0..3}__padded.parquet` (full 2,904-article files).
- No `tabpfn-ts-v2__test__final__padded.parquet` in `forecasts_final/`.
- No partial/temp `*tabpfn*` files modified in the last 180 minutes.

## Throughput warning

`nvidia-smi` reports GPU utilization 0% with 2.1 GiB used, while each process
burns ~284% CPU. Inference is CPU-bound in local mode; after 3h03m wall the
aggregate CPU burn is ~43.7 CPU-hours with zero completed output file. At this
rate the full run does not look like it will finish soon.

## Anything else needed from this task

Re-run T5 once the full padded parquets exist: copy them down, check
2,904 x 13 completeness, `score_all.py` alone, then `ensemble_recipe.py evaluate`
with tabpfn-ts-v2 added and substituted for each of the three members.
