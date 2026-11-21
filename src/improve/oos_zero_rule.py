"""X1b: zero rule from stock status. Forecast 0 where the article is out of stock at the origin and has sold little recently.

    .venv/bin/python src/improve/oos_zero_rule.py

No fitted parameter; the variants are fixed in advance and all reported. On top of D40 (8-week zero rule already inside).
  V1: out of stock in the last observed week and no sale in the last 2 weeks
  V2: out of stock in each of the last 2 observed weeks
  V3: out of stock in the last observed week and no sale in the last 4 weeks
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "improve"))
import ens_search as es, validate as v  # noqa: E402
warnings.filterwarnings("ignore")
OUT = ROOT / "iterations" / "10-external-info" / "x1"


def main():
    full = v.load_weekly("test")
    cuts = [es.load_cut(full, k) for k in range(4)]
    ref = np.array([es.score(c, es.d40(c)) for c in cuts])
    flags = []
    for k, c in enumerate(cuts):
        cut = c["weeks"][0]
        O = (full[full["week"] < cut].pivot_table(index=v.KEYS, columns="week", values="out_of_stock_flag", aggfunc="max")
             .reindex(index=pd.MultiIndex.from_frame(c["M"].keys)).fillna(False).to_numpy(bool))
        flags.append(O[c["valid"]])
    U = [c["M"].U[c["valid"]] for c in cuts]
    rules = {"V1 OOS last week & no sale 2w": lambda O, u: O[:, -1] & (u[:, -2:].sum(1) == 0),
             "V2 OOS last 2 weeks": lambda O, u: O[:, -1] & O[:, -2],
             "V3 OOS last week & no sale 4w": lambda O, u: O[:, -1] & (u[:, -4:].sum(1) == 0)}
    print(f"{'D40':36s} " + " ".join(f"{x:7.2f}" for x in ref) + f"  mean {ref.mean():7.2f}")
    rows = []
    for name, r in rules.items():
        sc = []
        for c, O, u in zip(cuts, flags, U):
            F = es.d40(c).copy(); hit = r(O, u); F[hit] = 0.0; sc.append((es.score(c, F), hit.mean()))
        s = np.array([x for x, _ in sc]); g = s - ref
        verdict = "ADOPT" if g.mean() >= 0.5 and (g > 0).sum() >= 3 else "reject"
        print(f"{name:36s} " + " ".join(f"{x:7.2f}" for x in s) + f"  mean {s.mean():7.2f}  gain {g.mean():+.2f}  won {(g > 0).sum()}/4  zeroed {np.mean([h for _, h in sc])*100:.1f}%  {verdict}")
        rows.append({"variant": name, **{f"cut{k}": round(x, 2) for k, x in enumerate(s)}, "mean": round(s.mean(), 2), "gain_vs_D40": round(g.mean(), 2), "cuts_won": int((g > 0).sum()), "verdict": verdict})
    pd.DataFrame(rows).to_csv(OUT / "results_adamex-svk_oos_zero_rule.csv", index=False)


if __name__ == "__main__":
    main()
