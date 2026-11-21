"""T4: flatten the D40 blend over the 13-week horizon (iteration 08).

Hypothesis: the errors grow with the horizon and Moirai's week-to-week wiggle is sampling noise,
so replacing each article's 13 values by one flat level (median / mean) may score better.

Per standard cut k (D26/D32), take the *un-rounded* D40 blend
    F = 0.3 * LightGBM + 0.7 * mean(moirai-2.0-small, tirex, chronos-t5-small)   (zero rule applied)
and compare three parameter-free flattenings:
    flat_median      repeat median(F over the 13 weeks) for every week
    flat_mean        repeat mean(F over the 13 weeks) for every week
    flat_median_fm   repeat median(foundation part) but keep the LightGBM part week-by-week
Then round (as the scored submissions are rounded) and score. No tuning.

Run: .venv/bin/python src/improve/t4_flatten.py
Writes iterations/08-improvements/t4/results_adamex-svk_t4.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "meta"))
import ensemble_recipe as er  # noqa: E402
import validate as v  # noqa: E402

OUT = ROOT / "iterations" / "08-improvements" / "t4"
H = er.H
W = er.REFERENCE_W
MODELS = er.REFERENCE_MODELS
N_CUTS = er.N_CUTS


def components(k, ctx, cache):
    """(LightGBM part, foundation part with zero rule, D40 blend), all un-rounded."""
    M, _, _, _, dead, Fl = ctx[k]
    mod = np.mean([er.raw_practice(m, k, ctx, cache) for m in MODELS], axis=0)
    mod[dead] = 0.0
    return Fl, mod, W * Fl + (1 - W) * mod


def flat(A, how):
    level = np.median(A, axis=1, keepdims=True) if how == "median" else A.mean(axis=1, keepdims=True)
    return np.repeat(level, A.shape[1], axis=1)


def variants(k, ctx, cache):
    Fl, mod, blend = components(k, ctx, cache)
    return {
        "D40": blend,
        "flat_median": flat(blend, "median"),
        "flat_mean": flat(blend, "mean"),
        "flat_median_fm": W * Fl + (1 - W) * flat(mod, "median"),
    }


def horizon_mae(F, ctx, k):
    """Mean absolute error by horizon week (all series), to see whether error grows with the horizon."""
    M, _, hold, weeks, _, _ = ctx[k]
    fc = pd.DataFrame({
        "store_id": np.repeat(M.keys["store_id"].to_numpy(), H),
        "article_id": np.repeat(M.keys["article_id"].to_numpy(), H),
        "week": np.tile(np.asarray(weeks), M.n),
        "forecast": np.round(F).reshape(-1),
    })
    m = hold.merge(fc, on=v.KEYS + ["week"], how="left", validate="one_to_one")
    m["ae"] = (m["units"] - m["forecast"]).abs()
    return m.groupby("week")["ae"].mean().to_numpy()


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ctx, cache = er.cut_contexts(), {}
    names = ["D40", "flat_median", "flat_mean", "flat_median_fm"]
    scores = {n: np.zeros(N_CUTS) for n in names}
    profile = {n: np.zeros(H) for n in names}
    for k in range(N_CUTS):
        for name, F in variants(k, ctx, cache).items():
            scores[name][k] = er.score_cut(k, F, ctx)
            profile[name] += horizon_mae(F, ctx, k) / N_CUTS
        print(f"cut {k}: " + "  ".join(f"{n} {scores[n][k]:7.2f}" for n in names), flush=True)

    ref = scores["D40"]
    rows = []
    for name in names:
        s = scores[name]
        d = s - ref
        won = int((d > 0).sum())
        ok = d.mean() >= er.ADOPT_MEAN and won >= er.ADOPT_CUTS
        rows.append({
            "variant": name, **{f"cut{k}": s[k] for k in range(N_CUTS)},
            "mean": s.mean(), "gain_vs_D40": d.mean(), "cuts_won": won,
            "verdict": "ADOPT" if ok else "keep D40",
        })
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results_adamex-svk_t4.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nScores by cut (higher is better):")
    print(res.round(2).to_string(index=False))
    print("\nMean absolute error by horizon week (average over the 4 cuts):")
    print(pd.DataFrame(profile).round(2).to_string())

    # horizon growth: last 4 weeks vs first 4 weeks of the error profile
    prof = pd.DataFrame(profile)
    grow = (prof.iloc[-4:].mean() / prof.iloc[:4].mean()).rename("late/early MAE")
    print("\n" + grow.round(3).to_string())

    print(f"\nelapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
