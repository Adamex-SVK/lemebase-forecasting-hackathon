"""X1f: is the own-history fine-tuned Chronos-2 really better? 13 overlapping cuts (offsets 0, 4, ..., 48 weeks), paired against D40.

    .venv/bin/python src/improve/ft_overlap_eval.py [SEEDS ...]    (default 0 1 2 3 4; files forecasts_oos-ftT1W{seed}/)

D40 at each cut: 0.3 x LightGBM (base_forecasts/lgbm_w{N}.npy) + 0.7 x mean of moirai-2.0-small, tirex, chronos-t5-small (forecasts_cuts), zero rule, rounding.
Variants: fine-tuned model alone (seed average), and as a 4th ensemble member. Cuts overlap (13-week holdouts every 4 weeks), so neighbouring cuts are
correlated: the interval below treats cuts as exchangeable and is therefore optimistic; the number of effectively independent windows is about 4.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import features as f  # noqa: E402
import validate as v  # noqa: E402
warnings.filterwarnings("ignore")
H = 13
IT3 = ROOT / "iterations" / "03-foundation-model"
BASE = ROOT / "iterations" / "05-meta-ensemble" / "base_forecasts"
FMS = ["moirai-2.0-small", "tirex", "chronos-t5-small"]
OFFS = list(range(0, 49, 4))


def cut_data(full, W):
    df = full[full["week"] <= full["week"].max() - pd.Timedelta(weeks=W)]
    train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut]); idx = pd.MultiIndex.from_frame(M.keys)
    Y = hold.pivot_table(index=v.KEYS, columns="week", values="units", aggfunc="sum").reindex(idx).fillna(0.0).to_numpy(float)
    s = v.history_scale(train, "full").reindex(idx).to_numpy(float)
    return idx, Y, s, M.U[:, -8:].sum(axis=1) == 0


def load(p, idx, dead):
    F = pd.read_parquet(p).pivot_table(index=v.KEYS, columns="week", values="forecast", aggfunc="sum").reindex(idx).fillna(0.0).to_numpy(float)
    F[dead] = 0.0
    return F


def score(F, Y, s, ok):
    return 100 * (1 - np.mean((np.abs(np.clip(np.round(F), 0, None) - Y).mean(axis=1) / s)[ok]))


def main(seeds):
    full = v.load_weekly("test"); rows = []
    for W in OFFS:
        idx, Y, s, dead = cut_data(full, W); ok = np.isfinite(s) & (s > 0)
        L = np.load(BASE / f"lgbm_w{W}.npy")
        E = np.mean([load(IT3 / "forecasts_cuts" / f"{m}__test__w{W}__padded.parquet", idx, dead) for m in FMS], axis=0)
        C = np.mean([load(IT3 / f"forecasts_oos-ftT1W{sd}" / f"chronos-2-ftT1W{sd}__test__w{W}__padded.parquet", idx, dead) for sd in seeds], axis=0)
        r = {"offset": W, "D40": score(0.3 * L + 0.7 * E, Y, s, ok), "ft_alone": score(C, Y, s, ok),
             "D40+ft_4th": score(0.3 * L + 0.7 * (3 * E + C) / 4, Y, s, ok), "D40,half_ft": score(0.3 * L + 0.35 * E + 0.35 * C, Y, s, ok)}
        rows.append(r); print({k: round(x, 2) for k, x in r.items()}, flush=True)
    R = pd.DataFrame(rows).round(2)
    print("\nmean over 13 overlapping cuts:"); print(R.drop(columns="offset").mean().round(2).to_string())
    rng = np.random.default_rng(0)
    for c in ["ft_alone", "D40+ft_4th", "D40,half_ft"]:
        d = (R[c] - R["D40"]).to_numpy(); bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        print(f"{c:14s} vs D40: mean {d.mean():+.2f}  cuts won {(d > 0).sum()}/13  bootstrap 95% [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}] (optimistic, cuts overlap)")
    R.to_csv(ROOT / "iterations" / "10-external-info" / "x1" / "results_adamex-svk_ft_overlap13.csv", index=False)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [0, 1, 2, 3, 4])
