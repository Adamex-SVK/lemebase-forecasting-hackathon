# Forecast validation contract

Version 1, established during phase 1 on 2026-11-21. Machine-readable rules:
`iterations/10-benchmark-diagnosis/validation.json`. Changes to this contract
must be explicit and recorded before further candidate selection.

## Frozen benchmark

Use the incumbent: 70% equal average of Moirai 2.0-small, TiRex and
Chronos-T5-small, 30% original LightGBM. Zero the foundation average for products
with no sales in the previous eight weeks; blend raw predictions, round once
with NumPy, then clip at zero. Preserve the original winning CSV and the pinned
source commit `71b82f33f0fe5647a13d3786cd17d11b4c4e5493`.

Primary historical offsets (weeks removed before constructing a 13-week
holdout): **0, 13, 26, 39**. Reproduced mean score: **-13.6659469669**.
These are repeatedly used development periods, not untouched tests.

The reported -4.8 is public leaderboard performance on a different period. Do
not compare it numerically with a local -13.67 as if they were the same test.

## Metric and reporting

Keep `src/validate.py` unchanged: equal-weight series MASE, adjacent-week scale
over the pre-cut history, existing leading-zero padding, constant/undefined
scales excluded and counted, score `(1-MASE)*100`. Report full 13 weeks, weeks
1–7 and 8–13 separately, numbers excluded, and scored-unit coverage.

Compare candidate and benchmark on exactly the same series, weeks, scales and
processing. Require complete finite forecasts; do not silently fill failed
predictions. Average scores equally across the four primary periods. Per-product
bootstrap intervals describe product uncertainty, not independent temporal
replications; they do not erase repeated-search bias.

## Additional forecasts and chronological purging

Complete source batch at commit `5592b3593c794dd21b441b0fdd564eea90f78248`
contains offsets 0,4,...,48 for the necessary base models. Its 13 windows overlap;
only 12 origins are additional to the four original cuts (16 unique total).
Some batch windows already appear in donor meta-model results. Do not call them
13 new independent or untouched tests.

Use additional windows for training/calibration diagnostics. For learned
corrections or blend weights, test offsets **0,16,32** provide nonoverlapping
target periods. A training label window at offset `W` is eligible only if
`W >= test_offset + 13`; its entire target must finish before the test target
starts. Hyperparameter selection must obey the same rule. The offset-32 fold
has only one eligible existing batch window (48), so complex meta-model claims
there are especially weak. Existing LightGBM window training retains its own
purged inner early stopping; never use outer holdout labels for tree counts.

`check_protocol.py` materializes these label-window pairs and asserts disjoint
targets. Using the same dates with a new candidate is a controlled forward
check, but it is not a pristine holdout after previous development has seen them.

## Forecast versions are part of the benchmark

Do not replace original saved foundation forecasts with regenerated forecasts
while crediting a candidate for the resulting score difference. The duplicate
offset-0 Chronos-T5-small batch differs from the original, while Moirai, TiRex
and LightGBM match. Composing the same recipe can therefore yield different
predictions. Cause is not established by this audit.

For primary periods use original pinned forecasts. For additional periods use
the additional pinned batch and reconstruct both benchmark and candidate from
the same files. Store their hashes and preserve generation settings/seeds when
new forecasts are generated.

## Confirmation limits and reserved stress input

**No untouched forward confirmation is currently available.** Do not relabel
known periods as unseen. Reserve offset **65**, 2021-10-25 through 2022-01-17,
as a closed, older regime stress check after candidate settings are frozen.
No component forecasts exist for it in the audited batch, so these must be
generated before it can be used. It has 65 pre-target weeks and is an older
stress period, not prospective confirmation after development on later dates.

Offset 52 overlaps offset 48's already examined target weeks. Offset 78 leaves
only one purged inner-training origin for the current own-shop LightGBM recipe;
it is not a strong primary confirmation period. Actual untouched future or
private competition outcomes remain necessary for an independent forward claim.

## Next experiment and promotion review

Run one prespecified feature-group ablation first. Freeze its feature definitions,
training settings, data, weights and processing before scoring candidates. Do
not optimize on public week-by-week feedback or hand-pick high-error product IDs.

Review threshold: primary average gain at least **+0.5 points**, gains on at least
**3 of 4** primary periods, and no mean regression on weeks 8–13. Report behavior
for recent activations, rare sellers, inactive products and stock groups. Check
additional forward folds for consistency. Passing these development thresholds
supports review, not a guarantee of private accuracy or automatic submission.

A-D and unseen-shop claims require their own feasible temporal/shop-held-out
evaluations. For D, always show zero-history exclusions and scored-unit coverage.
No target-shop-only improvement proves cross-shop transfer. Uploads require the
user's approval.
