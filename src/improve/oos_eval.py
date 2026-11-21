"""X1: does masking out-of-stock weeks in the context help the D40 foundation models? Scored on the 4 standard cuts.

    .venv/bin/python src/improve/oos_eval.py

Forecasts come from `src/foundation/run_model.py --oos {interp,short}` (iterations/03-foundation-model/forecasts_oos-<mode>/).
Variants replace the three D40 foundation members (moirai-2.0-small, tirex, chronos-t5-small) by their masked-context
versions; LightGBM, the zero rule (from raw sales), blend weights and rounding stay as in D40.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "improve"))
import ens_search as es  # noqa: E402
import validate as v  # noqa: E402

warnings.filterwarnings("ignore")
IT3 = ROOT / "iterations" / "03-foundation-model"
OUT = ROOT / "iterations" / "10-external-info" / "x1"


def masked(c, mode, k):
    out = {}
    for m in es.D40_FM:
        p = IT3 / f"forecasts_oos-{mode}" / f"{m}__test__cut{k}__padded.parquet"
        idx = pd.MultiIndex.from_frame(c["M"].keys)
        F = pd.read_parquet(p).pivot_table(index=v.KEYS, columns="week", values="forecast", aggfunc="sum").reindex(idx).fillna(0.0).to_numpy(float)
        F[c["M"].U[:, -8:].sum(axis=1) == 0] = 0.0                      # zero rule from raw sales (D39)
        out[m] = F[c["valid"]]
    return out


def main():
    full = v.load_weekly("test")
    cuts = [es.load_cut(full, k) for k in range(4)]
    ref = np.array([es.score(c, es.d40(c)) for c in cuts])
    rows = []

    def report(name, scores):
        g = np.array(scores) - ref
        verdict = "ADOPT" if g.mean() >= 0.5 and (g > 0).sum() >= 3 else "reject"
        rows.append({"variant": name, **{f"cut{k}": round(x, 2) for k, x in enumerate(scores)}, "mean": round(np.mean(scores), 2),
                     "gain_vs_D40": round(g.mean(), 2), "cuts_won": int((g > 0).sum()), "verdict": verdict})
        print(f"{name:46s} " + " ".join(f"{x:7.2f}" for x in scores) + f"  mean {np.mean(scores):7.2f}  gain {g.mean():+.2f}  won {(g > 0).sum()}/4  {verdict}")

    print(f"{'D40 (reference)':46s} " + " ".join(f"{x:7.2f}" for x in ref) + f"  mean {ref.mean():7.2f}")
    for mode in ["interp", "short"]:
        per = [masked(c, mode, k) for k, c in enumerate(cuts)]
        report(f"{mode}: all three members masked, D40 blend", [es.score(c, 0.3 * c["mem"]["lgbm"] + 0.7 * np.mean([p[m] for m in es.D40_FM], axis=0)) for c, p in zip(cuts, per)])
        report(f"{mode}: masked ensemble alone", [es.score(c, np.mean([p[m] for m in es.D40_FM], axis=0)) for c, p in zip(cuts, per)])
        for m in es.D40_FM:
            report(f"{mode}: only {m} masked, D40 blend", [es.score(c, 0.3 * c["mem"]["lgbm"] + 0.7 * np.mean([p[x] if x == m else c["mem"][x] for x in es.D40_FM], axis=0)) for c, p in zip(cuts, per)])
        report(f"{mode}: masked + unmasked members together (6)", [es.score(c, 0.3 * c["mem"]["lgbm"] + 0.7 * np.mean([p[m] for m in es.D40_FM] + [c["mem"][m] for m in es.D40_FM], axis=0)) for c, p in zip(cuts, per)])
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "results_adamex-svk_oos_mask.csv", index=False)


if __name__ == "__main__":
    main()
