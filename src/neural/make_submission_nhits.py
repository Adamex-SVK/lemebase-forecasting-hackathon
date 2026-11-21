"""Build submissions/ensemble_top3_nhits_v1.csv: D40 with NHITS taking half the LightGBM weight, for the REAL weeks.

    .venv-nf/bin/python src/neural/make_submission_nhits.py      (neuralforecast env; laptop CPU is fine, ~8 min)

  N = NHITS own arm, 3 seeds, trained on ALL test-shop history (same settings as run_nf.py defaults)
  L = submissions/lgbm_own_v1.csv (the LightGBM used in the live D40 file)
  E = mean(moirai-2.0-small, tirex, chronos-t5-small final forecasts)
  0 for articles with no sale in the last 8 weeks (on E and N);  F = round(0.15 L + 0.15 N + 0.7 E), clipped at 0.
Local 4-cut evidence (src/neural/nhits_in_d40.py): mean -13.62 vs D40 -13.67, 2/4 cuts, below the adoption bar.
NHITS final forecasts are saved to iterations/06-neural/forecasts_final/. Uploading is done by a person."""
import argparse
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "neural"))
import run_nf as r, validate as v, make_submission as ms, make_submission_foundation as mf  # noqa: E401

H = r.H
OUTF = r.OUT / "forecasts_final"; OUTF.mkdir(parents=True, exist_ok=True)
args = argparse.Namespace(max_steps=1000, lr=1e-3, series_batch=32, windows_batch=1024, patience=300, val_size=H,
                          start_padding=False, input_size=52, accelerator="cpu", devices=1)
full = v.load_weekly("test")
cut = full["week"].max() + pd.Timedelta(weeks=1)
M, scale, valid, cut, train, hold = r.shop_matrix(full, cut=cut)
assert M.W == full["week"].nunique()
panel, idx = r.panel_for(M, scale, valid)
P = dict(panel=panel, M=M, scale=scale, valid=valid, idx=idx, cut=cut)
weeks = pd.date_range(cut, periods=H, freq="7D")
Fs = []
for s in range(3):
    fp = OUTF / f"NHITS__test__final__padded__seed{s}.parquet"
    if fp.exists():
        F = r.matrix_from_long(pd.read_parquet(fp), M, weeks)
    else:
        F, inf = r.run_one(P, "NHITS", args, s); r.to_long(M, F, weeks).to_parquet(fp, index=False)
        print(f"seed {s}: {inf}", flush=True)
    Fs.append(F)
N = np.mean(Fs, 0)
r.to_long(M, N, weeks).to_parquet(OUTF / "NHITS__test__final__padded.parquet", index=False)
ids = ms.history_ids()
order = pd.Index(M.keys["article_id"]).get_indexer(ids); assert (order >= 0).all() and M.keys["store_id"].nunique() == 1
N = N[order]
dead = mf.dead_mask(ids)
E = np.mean([mf.final_matrix(m, ids) for m in ["moirai-2.0-small", "tirex", "chronos-t5-small"]], 0)
E[dead] = 0.0; N[dead] = 0.0
L = pd.read_csv(ms.SUB_DIR / "lgbm_own_v1.csv").set_index("article_id").loc[ids].to_numpy(float)
F = np.clip(np.round(0.15 * L + 0.15 * N + 0.7 * E), 0, None)
path = ms.SUB_DIR / "ensemble_top3_nhits_v1.csv"
ms.write_submission(F, ids, path); ms.validate_submission(path)
old = pd.read_csv(ms.SUB_DIR / "ensemble_top3_blend_v1.csv").set_index("article_id").loc[ids].to_numpy(float)
print(f"{path.name}: mean forecast {F.mean():.2f} (live file {old.mean():.2f}); NHITS mean {N.mean():.2f}, LightGBM mean {L.mean():.2f}; "
      f"mean |diff| vs ensemble_top3_blend_v1 (-4.8): {np.abs(F-old).mean():.3f}; cells changed {100*(F!=old).mean():.1f}%")
