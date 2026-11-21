"""Build submissions/ensemble_top3_gate_v1.csv: the gated foundation blend, for the REAL weeks 2023-04-24 .. 2023-07-17.

    .venv/bin/python src/make_submission_gated.py

Recipe (4-cut evidence in iterations/02-robust-eval/notes_adamex_loss-and-nn-plan.md; NOT adopted by the bar there):
  E = mean(moirai-2.0-small, tirex, chronos-t5-small final forecasts), 0 where no sale in the last 8 weeks
  L = LightGBM L1 (3 seeds, trained on ALL test-shop history, early-stopped trees x1.1), P = classifier P(sale>0) (same seeds)
  F = round( gate( 0.3 * L + 0.7 * E ) ),  gate: 0 where P < 0.5.   Clipped at 0.
Uploading is done by a person (TEAM-RULES.md section 6)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments_loss as el, features as f, make_submission as ms, make_submission_foundation as mf, validate as v  # noqa: E401

MODELS = ["moirai-2.0-small", "tirex", "chronos-t5-small"]
H = f.H
full = v.load_weekly("test")
M = f.ShopMatrix(full)
origins = f.make_origins(M.W); val_o = origins[-7:]; tr_o = origins[origins <= val_o[0] - H]
P = dict(M=M, w_tr=f.build_windows(M, tr_o), w_va=f.build_windows(M, val_o), w_all=f.build_windows(M, origins), inf=f.inference_features(M))
assert M.keys["store_id"].nunique() == 1 and M.keys["article_id"].is_unique
Fs, Ps = [], []
for s in range(3):
    F, Pr, b1, b2 = el.fit_pair(P, s); Fs.append(F); Ps.append(Pr)
    print(f"seed {s}: trees reg {b1} clf {b2}", flush=True)
L, Pr = np.mean(Fs, 0), np.mean(Ps, 0)
ids = ms.history_ids()
order = pd.Index(M.keys["article_id"]).get_indexer(ids)
assert (order >= 0).all()
L, Pr = L[order], Pr[order]
E = np.mean([mf.final_matrix(m, ids) for m in MODELS], axis=0); E[mf.dead_mask(ids)] = 0.0
B = 0.3 * L + 0.7 * E
F = np.clip(np.round(np.where(Pr < 0.5, 0.0, B)), 0, None)
path = ms.SUB_DIR / "ensemble_top3_gate_v1.csv"
ms.write_submission(F, ids, path); ms.validate_submission(path)
old = pd.read_csv(ms.SUB_DIR / "ensemble_top3_blend_v1.csv").set_index("article_id").loc[ids].to_numpy(float)
print(f"{path.name}: mean forecast {F.mean():.2f}; articles exactly 0 all weeks {100*(F.max(axis=1)==0).mean():.1f}%; "
      f"gated-to-zero cells {100*((Pr<0.5)&(np.round(B)>0)).mean():.1f}%; mean |diff| vs ensemble_top3_blend_v1 (live -4.8): {np.abs(F-old).mean():.2f}; "
      f"corr of article means {np.corrcoef(F.mean(1), old.mean(1))[0,1]:.3f}")
