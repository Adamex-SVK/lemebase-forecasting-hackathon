# Iteration 09 — angle C: within-shop analog donors (Kimi)

**Status: NOT RUN.** The session was stopped after the step-0 audit, the A selector diagnostic,
and the B pooled-level model; no analog donor code or results exist.

The planned C design (from `plan_kimi.md`) remains the last ranked new-information test:

- Eligible donors are other test-shop articles with `mean26 > 0`, at least 26 origin-history weeks,
  a complete 13-week outcome before the target origin (`W_donor >= W_target + 13`), and no shared
  article ID with the target.
- Match on `log1p(mean4/13/26)`, zero share, recency, CV, tenure, and stock arrival; robust-scale
  with the eligible-donor median/IQR; Euclidean distance in target chunks of 256; fixed 50
  neighbours with `(distance, article_id, origin)` tie-break.
- Predict `target_mean26 × median(650 donor weekly y[h] / donor_mean26)`; D40 fallback otherwise;
  no blend.
- Falsifier: analog cannot beat mean26 on eligible past cuts, matches are unstable, or the
  canonical exact score fails +0.5 / 3-of-4 / both-horizon.

Do not read the absence of C as evidence for or against it.
