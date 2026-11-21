"""D32: rolling-origin evaluation on the test shop. Score at 4 cuts: the last 13 weeks of the
history, then the 13 weeks before that, and so on (cut k drops the last 13*k weeks).

For each cut: zero, naive, mean13, median13, seasonal naive, Top-down MSTL, AutoARIMA and the
LightGBM `own` arm (trained only on weeks before that cut). Scores use the D26 convention.

Run: .venv/bin/python src/rolling.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as b  # noqa: E402
import train_gbm as g  # noqa: E402
import validate as v  # noqa: E402

warnings.filterwarnings("ignore")
OUT = v.ROOT / "iterations" / "02-robust-eval"
N_CUTS = 4


def main(shop="test"):
    OUT.mkdir(parents=True, exist_ok=True)
    full = v.load_weekly(shop)
    rows, det = [], {}
    for k in range(N_CUTS):
        df = full[full["week"] <= full["week"].max() - pd.Timedelta(weeks=13 * k)]
        res = g.own_arm(shop, df=df)
        train, hold, cut = res["train"], res["hold"], res["cut"]
        weeks = res["weeks"]
        ev = g.evaluate(shop, res)
        scores = {"lgbm_own": ev["plain"][1]}
        P = b.wide(train[v.KEYS + ["week", "units"]])
        mk = b.baselines(P, weeks)
        for name, make in mk.items():
            fc = b.complete(make(), hold)
            scores[name] = v.score(train, hold, fc, detail=True)
        for name, r in scores.items():
            rows.append({"shop": shop, "cut_index": k, "first_holdout_week": str(cut.date()), "model": name,
                         "score": r["score"], "n_scored": r["n_scored"], "train_weeks": int(train["week"].nunique())})
            det[(k, name)] = r["detail"]["mase"]
        print(f"cut {k} (holdout starts {cut.date()}): " + "  ".join(f"{n} {r['score']:.1f}" for n, r in scores.items()), flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"results_rolling_{shop}.csv", index=False)
    df = pd.DataFrame(rows)
    pt = df.pivot(index="model", columns="cut_index", values="score")
    pt["mean"] = pt.mean(axis=1)
    pt = pt.sort_values("mean", ascending=False)
    print("\nScore by cut (higher is better):"); print(pt.round(1).to_string())
    rng = np.random.default_rng(0)
    print("\nLightGBM own minus median13, paired over articles (score points):")
    gains = []
    for k in range(N_CUTS):
        a, m = det[(k, "lgbm_own")], det[(k, "median13")]
        d = (a - m).dropna()
        bs = [-100 * rng.choice(d.to_numpy(), len(d)).mean() for _ in range(1000)]
        gains.append(-100 * d.mean())
        print(f"  cut {k}: {-100*d.mean():+.2f}  95% interval [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]")
    print(f"  average over cuts: {np.mean(gains):+.2f}; cuts where LightGBM beats median13: {sum(x > 0 for x in gains)} of {N_CUTS}")


if __name__ == "__main__":
    main(*sys.argv[1:])
