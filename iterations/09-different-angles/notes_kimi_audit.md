# Iteration 09 — Step 0 audit and angle A selector (Kimi)

Date: 2026-11-21. Worktree: `../hack-kimi`, branch `it09/kimi/different-angles`, based on
`it03/adamex-svk/foundation-models`. Command:

```bash
bash scripts/check.sh
.venv/bin/python src/meta/ensemble_recipe.py evaluate \
  --models moirai-2.0-small,tirex,chronos-t5-small --lgbm-weight 0.3
.venv/bin/python src/experiments_different_angles_kimi.py audit
```

Results: `results_kimi_audit.csv`. No upload, no change to `src/validate.py`,
`src/prepare.py`, `src/features.py`, or D40.

## Step 0 reproduction

Canonical four cuts (D26, integer action, zero rule applied to the foundation ensemble before the
LightGBM blend as in D39/D40):

| cut | D40 |
|---|---|
| 0 | −3.72 |
| 1 | −25.91 |
| 2 | −11.01 |
| 3 | −14.02 |
| mean | **−13.67** |

Thirteen overlapping cuts `W=0,4,...,48` (cached bases, `fforma.load_cut`; these are correlated and
diagnostic only):

| quantity | mean |
|---|---|
| D40 | −12.74 (brief-reported −12.67; 0.07 discrepancy) |
| flat D40 to own mean | −12.65 |
| flat D40 to own median | −12.68 |
| oracle holdout level + D40 shape | +0.92 |
| oracle flat holdout median | +1.84 |
| O_sel2 (D40 vs mean26, hindsight) | −9.23 |
| O_sel11 (11 levels, hindsight) | **−3.95** |
| O_sel11 − D40 | **+8.78** |
| O_sel2 − D40 | +3.51 |

Flattening changes D40 by at most 0.77 (mean) / 0.81 (median); oracle level + D40 shape is 0.92
from the unrestricted oracle flat. D2 is reproduced. O_sel11 = −3.95 and O_sel2 = −9.23 match the
brief exactly; D40 differs by 0.07, which is immaterial to the +8.7 interpretation.

D4 (canonical, no post-blend zero): D40 w1–7 mean **−12.29**, w8–13 mean **−15.27**, matching the
brief.

D5 at `W=40`: article `A139270` scale 0.0112, holdout 79 units, per-article MASE 540.85,
**20.83 score points**. Reproduced exactly.

D1 tail concentration (canonical): top 1% of articles = 25.6–33.2% of total MASE, top 5% =
47.3–53.9%, top 10% = 59.2–63.7%. Worst-set stability disagrees with the brief's 6–37% range:
using each cut's own scored set, overlap is 12.3–40.1% (`/|A|`) or 6.8–25.6% (Jaccard); restricting
to the common scored intersection gives 33.1–54.6% (`/|A|`) or 19.8–37.6% (Jaccard). The qualitative
claim (unstable identities) survives; the exact range does not reproduce under one convention.
This is a measurement discrepancy, not a reason to rerun the tail experiment.

### Winner's-curse null

Each non-D40 candidate's forecasts were permuted independently within origin-defined
dead8 × scale-quintile strata, with D40 fixed, and O_sel11 recomputed 200 times per cut.
Observed O_sel11 mean −3.95; null mean **−5.54**; the observed value lies inside the null 95%
interval on only 5/13 cuts. The oracle retains about +7.2 over D40 even after article alignment is
destroyed, and only ≈+1.6 beyond the null. Most of the +8.8 is therefore consistent with hindsight
selection over level spread, not a stable per-article signal.

## Angle A — level selector diagnostic (cached cuts only)

Predeclared: choose D40 level vs mean26 level with **one** stump; features `trend`
(log1p mean4 − log1p mean13), `zshare13`, `cv26`; thresholds at training-feature quantiles
0.1,…,0.9; training cuts satisfy `W >= W_target + 13`; objective is the exact scaled-absolute loss
of the rounded, zero-ruled action. Also run the previous-cut-best diagnostic and a training-feature
shuffle control within dead8 × scale-quintile strata.

| W_target | train cuts | split chosen | gain vs D40 | switched to mean26 | previous-cut-best gain | C2 | C11 |
|---|---|---|---|---|---|---|---|
| 0 | 9 | none | +0.01 | 0 | −0.87 | 0.003 | 0.001 |
| 4 | 8 | none | −0.44 | 0 | −1.32 | −0.141 | −0.051 |
| 8 | 7 | none | −0.51 | 0 | −1.68 | −0.149 | −0.061 |
| 12 | 6 | none | −0.53 | 0 | −1.46 | −0.166 | −0.061 |

No feature/threshold/direction improved training loss over always-D40 on any target; mean26 wins on
only 10.6–14.0% of articles and its mean flat loss is 0.04–0.08 higher than D40's on every training
cut. The shuffle control is identical because the real fit never produced a split.

**Verdict: A fails the predeclared screen (<+0.5, no 3-of-4 wins, unstable sign). Close the
selector family.** A positive result here would still have been descriptive only.

## Caveats

- Overlapping-cut diagnostics apply the zero rule to the final forecast (`ff.post`), matching the
  D3 oracle construction. Canonical adoption comparisons keep the frozen D40 convention (zero rule
  only on the foundation ensemble, before the 0.3 LightGBM blend).
- `W=0` overlapping D40 is −4.18 versus canonical −3.72 because they are separate GPU runs
  (stochastic Moirai); use overlapping cuts only for relative comparisons.
- All figures condition on the four windows / 13 correlated offsets; they do not establish
  season-to-season significance.
