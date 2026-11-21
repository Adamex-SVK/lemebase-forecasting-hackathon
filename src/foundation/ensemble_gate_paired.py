"""Paired per-article intervals for the gate on the final foundation blend (follow-up to ensemble_base_swap.py).
Blend = 0.3 * LightGBM(3-seed) + 0.7 * top3 ensemble (8-week zero rule), rounded. Compared per cut:
  A  same blend without the gate (3-seed base)       B  gate on the final blend (P<thr -> 0)
  S  submitted recipe (seed-0 base, no gate)         B vs S
Score points, 95% bootstrap over articles. Threshold sensitivity (0.3-0.7) shown for information only."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "foundation"))
import experiments_02 as e, features as f, score_all as sa, validate as v  # noqa: E401
FM = ROOT / "iterations/03-foundation-model/forecasts"; FC = ROOT / "iterations/02-robust-eval/forecasts"
MODELS = ["moirai-2.0-small", "tirex", "chronos-t5-small"]; H = f.H
full = v.load_weekly("test"); rng = np.random.default_rng(0)
def sc(M, F, c): return e.score_F(M, np.round(F), c["train"], c["hold"], c["weeks"])
tab = {t: [] for t in (0.3, 0.4, 0.5, 0.6, 0.7)}; out = {"B vs A (gate vs no gate)": [], "B vs S (vs submitted)": []}
for k in range(4):
    df = e.cut_frame(full, k); train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut]); c = dict(train=train, hold=hold, weeks=pd.date_range(cut, periods=H, freq="7D"))
    _, _, dk, _ = sa.prepare_cut("test", k)
    dead = np.array([(a, b) in dk for a, b in zip(M.keys["store_id"], M.keys["article_id"])])
    idx = pd.MultiIndex.from_frame(M.keys)
    E = np.mean([pd.read_parquet(FM / f"{m}__test__cut{k}__padded.parquet").pivot_table(index=v.KEYS, columns="week", values="forecast", aggfunc="sum").reindex(idx).fillna(0.0).to_numpy(float) for m in MODELS], 0)
    E[dead] = 0.0
    L3 = np.mean([np.load(FC / f"loss_F_cut{k}_seed{s}.npy") for s in range(3)], 0)
    Pr = np.mean([np.load(FC / f"loss_P_cut{k}_seed{s}.npy") for s in range(3)], 0)
    L0 = np.load(FC / f"base_cut{k}_seed0.npy")
    A = 0.3 * L3 + 0.7 * E; S = 0.3 * L0 + 0.7 * E
    B = np.where(Pr < 0.5, 0.0, A)
    ra, rb, rs = (sc(M, X, c)["detail"]["mase"] for X in (A, B, S))
    for name, x, y in [("B vs A (gate vs no gate)", rb, ra), ("B vs S (vs submitted)", rb, rs)]:
        d = (x - y).dropna().to_numpy(); bs = [-100 * rng.choice(d, len(d)).mean() for _ in range(1000)]
        out[name].append(f"{-100*d.mean():+.2f} [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]")
    for t in tab: tab[t].append(sc(M, np.where(Pr < t, 0.0, A), c)["score"])
for n, r in out.items(): print(f"{n:26s} " + "  ".join(r))
print("\nthreshold sensitivity (score per cut, mean):")
for t, r in tab.items(): print(f"  thr {t}: " + " ".join(f"{x:7.2f}" for x in r) + f"   mean {np.mean(r):.2f}")
