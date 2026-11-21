"""Ensembles of foundation models (averaged), alone and blended with the LightGBM, on the 4 test-shop cuts (D26).

    .venv/bin/python src/foundation/ensemble_eval.py

The model sets are FIXED IN ADVANCE (listed below) so the choice is not tuned on the cuts. For each set:
  1. average the models' raw forecasts, apply the 8-week zero rule, round to whole units  -> "alone"
  2. blend w * LightGBM + (1 - w) * ensemble (w chosen on cuts 1-3), round the blend       -> "blend"
Reported as score by cut, mean over the 4 cuts and the gain over the rounded LightGBM alone.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "foundation"))
import experiments_02 as e  # noqa: E402
import features as f  # noqa: E402
import score_all as sa  # noqa: E402
import validate as v  # noqa: E402

FM_DIR = ROOT / "iterations" / "03-foundation-model" / "forecasts"
H = f.H
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)
SETS = {
    "moirai-2.0-small alone": ["moirai-2.0-small"],
    "top2: moirai-2.0 + tirex": ["moirai-2.0-small", "tirex"],
    "top3: + chronos-t5-small": ["moirai-2.0-small", "tirex", "chronos-t5-small"],
    "top5: + chronos-t5-base + moirai-1.1-base": ["moirai-2.0-small", "tirex", "chronos-t5-small", "chronos-t5-base", "moirai-1.1-base"],
    "diverse5: moirai-2.0, tirex, timesfm, t5-base, chronos-2": ["moirai-2.0-small", "tirex", "timesfm-2.5", "chronos-t5-base", "chronos-2"],
    "all good (<= -20 alone): 9 models": ["moirai-2.0-small", "tirex", "chronos-t5-small", "chronos-t5-base", "moirai-1.1-base", "timesfm-2.5",
                                          "moirai-moe-small", "chronos-t5-large", "moirai-1.1-large"],
}


def main(lgbm_variant="base", extra_sets=None):
    sets = dict(SETS)
    sets.update(extra_sets or {})
    full = v.load_weekly("test")
    ctx = {}
    for k in range(4):
        df = e.cut_frame(full, k)
        train, hold, cut = v.split(df)
        M = f.ShopMatrix(df[df["week"] < cut])
        _, _, dead_keys, _ = sa.prepare_cut("test", k)
        dead = np.array([(a, b) in dead_keys for a, b in zip(M.keys["store_id"], M.keys["article_id"])])
        weeks = pd.date_range(cut, periods=H, freq="7D")
        Fl = np.load(ROOT / "iterations" / "02-robust-eval" / "forecasts" / f"{lgbm_variant}_cut{k}_seed0.npy")
        ctx[k] = (M, train, hold, weeks, dead, Fl)

    cache = {}

    def raw(model, k):
        if (model, k) not in cache:
            M = ctx[k][0]
            fc = pd.read_parquet(FM_DIR / f"{model}__test__cut{k}__padded.parquet")
            cache[(model, k)] = fc.pivot_table(index=v.KEYS, columns="week", values="forecast", aggfunc="sum").reindex(pd.MultiIndex.from_frame(M.keys)).fillna(0.0).to_numpy(float)
        return cache[(model, k)]

    def score(k, F):
        M, train, hold, weeks, *_ = ctx[k]
        return e.score_F(M, np.round(F), train, hold, weeks)["score"]

    ref = [score(k, ctx[k][5]) for k in range(4)]
    print(f"LightGBM ({lgbm_variant}) rounded alone: " + " ".join(f"{x:7.2f}" for x in ref) + f"   mean {np.mean(ref):.2f}\n")
    rows = []
    for name, models in sets.items():
        E = {k: np.mean([raw(m, k) for m in models], axis=0) for k in range(4)}
        for k in range(4):
            E[k][ctx[k][4]] = 0.0                                  # 8-week zero rule
        alone = [score(k, E[k]) for k in range(4)]
        sc = {w: [score(k, w * ctx[k][5] + (1 - w) * E[k]) for k in range(4)] for w in WEIGHTS}
        wbest = max(WEIGHTS, key=lambda w: np.mean(sc[w][1:]))
        bl = sc[wbest]
        rows.append((name, np.mean(alone), wbest, np.mean(bl), np.mean(bl) - np.mean(ref), sum(b > r for b, r in zip(bl, ref))))
        print(f"{name}\n   ensemble alone : " + " ".join(f"{x:7.2f}" for x in alone) + f"   mean {np.mean(alone):.2f}")
        print(f"   blend w={wbest}    : " + " ".join(f"{x:7.2f}" for x in bl) + f"   mean {np.mean(bl):.2f}   gain over rounded LightGBM {np.mean(bl) - np.mean(ref):+.2f}, cuts won {rows[-1][5]}/4")
        print("   mean by weight on LightGBM: " + "  ".join(f"{w:.1f}:{np.mean(sc[w]):.2f}" for w in WEIGHTS[::2]) + "\n")
    return rows


if __name__ == "__main__":
    main()
