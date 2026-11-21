# Modest recency weighting — fixed setting

User authorized testing modest recency weighting with existing features, L1 loss,
ensemble and preprocessing held fixed. Senior engineer skill was requested but
not installed in user skills, repo skills or plugin skill cache. Applied the
available senior-data-scientist skill and temporal evaluation workflow instead.

One prespecified setting, no search: 104-week half-life on training-window origin.
For each training matrix, factor = 2 ** ((origin-max_origin)/104); multiply original
sample weights and normalize to preserve their total. Thus rows two years apart
have a 2:1 weight ratio; all horizons/products at the same origin share the factor.
Normalization avoids changing global sample-weight scale and regularization merely
through rescaling. No training labels are used to choose weights or half-life.

Inner validation remains uniform so early stopping reflects the original metric.
Original seed-0 parameters, origins, 13-week purge, 3000-tree cap, patience 50,
seven validation origins and int(best*1.1)+1 final refit remain fixed. Tree counts
are selected inside each arm independently. Compare freshly fitted control to
saved original LightGBM to detect drift. No Phase-2 features or learned blend.

Use original pinned foundation forecasts and 70/30 ensemble on primary offsets
0,13,26,39; matched additional batch on 16 and 32. Additional forward checks are
0,16,32. Leading zeros, scale exclusions, foundation-only zero rule, rounding and
clipping remain unchanged. Evaluate official MASE and weeks 8–13, coverage and
history-defined groups. Reused target-shop own-only history cannot establish
A-D transfer or unseen-shop generalization. Offset 65 stays unopened.

Review thresholds fixed before scoring: primary mean gain >=0.5, >=3/4 primary
wins, nonnegative mean late-horizon gain, no loss on either additional check.
No outer-fold half-life tuning or automatic promotion. Preserve incumbent CSV;
no final forecasts, uploads, commits or pushes. If worse, retire challenger code
from active project and archive locally with a short result note.

Run `.venv/bin/python iterations/13-recency-weighting/run.py`;
use `--output NEW_DIRECTORY` for reproduction without overwriting.

Artifact verification: `.venv/bin/python iterations/13-recency-weighting/verify.py`.
This checks locked data/submission/code hashes, six exact control reproductions,
group partition totals and candidate model serialization/prediction round trip.
Observed runtime versions are recorded in requirements-runtime.txt and metadata.
