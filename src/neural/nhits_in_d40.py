"""Does NHITS add to the live D40 recipe? Fixed weights set in advance (no tuning on the cuts):
  D40          0.3 LightGBM(seed 0) + 0.7 FM
  D40+NHITS    0.15 LightGBM + 0.15 NHITS + 0.7 FM
  thirds       1/3 each
FM = mean(moirai-2.0-small, tirex, chronos-t5-small) with the 8-week zero rule; final blend rounded (as D40).
Paired per-article intervals against D40. Writes iterations/06-neural/results_adamex_nhits_d40.csv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "foundation"))
import experiments_02 as e, features as f, score_all as sa, validate as v  # noqa: E401
FM = ROOT / "iterations/03-foundation-model/forecasts"; NF = ROOT / "iterations/06-neural/forecasts"
MODELS = ["moirai-2.0-small", "tirex", "chronos-t5-small"]; H = f.H
full = v.load_weekly("test"); rng = np.random.default_rng(0); rows, pair = [], {}
def mat(p, idx): return pd.read_parquet(p).pivot_table(index=v.KEYS, columns="week", values="forecast", aggfunc="sum").reindex(idx).fillna(0.0).to_numpy(float)
for k in range(4):
    df = e.cut_frame(full, k); train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut]); weeks = pd.date_range(cut, periods=H, freq="7D"); idx = pd.MultiIndex.from_frame(M.keys)
    _, _, dk, _ = sa.prepare_cut("test", k)
    dead = np.array([(a, b) in dk for a, b in zip(M.keys["store_id"], M.keys["article_id"])])
    E = np.mean([mat(FM / f"{m}__test__cut{k}__padded.parquet", idx) for m in MODELS], 0); E[dead] = 0.0
    N = mat(NF / f"NHITS__test__cut{k}__padded.parquet", idx); N[dead] = 0.0
    L = np.load(ROOT / f"iterations/02-robust-eval/forecasts/base_cut{k}_seed0.npy")
    V = {"D40": 0.3 * L + 0.7 * E, "D40+NHITS (0.15/0.15/0.7)": 0.15 * L + 0.15 * N + 0.7 * E, "thirds": (L + N + E) / 3}
    for n, F in V.items():
        r = e.score_F(M, np.round(F), train, hold, weeks); rows.append({"cut": k, "variant": n, "score": r["score"]}); pair[(k, n)] = r["detail"]["mase"]
d = pd.DataFrame(rows); d.to_csv(ROOT / "iterations/06-neural/results_adamex_nhits_d40.csv", index=False)
pt = d.pivot(index="variant", columns="cut", values="score"); pt["mean"] = pt.mean(1); print(pt.round(2).to_string())
for n in ["D40+NHITS (0.15/0.15/0.7)", "thirds"]:
    s = []
    for k in range(4):
        x = (pair[(k, n)] - pair[(k, "D40")]).dropna().to_numpy(); bs = [-100 * rng.choice(x, len(x)).mean() for _ in range(1000)]
        s.append(f"{-100*x.mean():+.2f} [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]")
    print(f"{n} vs D40: " + "  ".join(s))
