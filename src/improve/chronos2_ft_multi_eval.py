"""X1d: averages of several fine-tuned Chronos-2 runs, and fine-tunes that also see the test shop's own pre-cut history.

    .venv/bin/python src/improve/chronos2_ft_multi_eval.py
Groups are fixed in advance and all reported; blends: D40 + the group as a 4th ensemble member, and a 50/50 split of the foundation part.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "improve"))
import ens_search as es, validate as v  # noqa: E402
warnings.filterwarnings("ignore")
IT3 = ROOT / "iterations" / "03-foundation-model"
GROUPS = {"siblings, 1 run (seed 0)": ["ft"], "siblings, 3-run avg": ["ft", "ftS1", "ftS2"], "siblings, 6-run avg": ["ft", "ftS1", "ftS2", "ftS3", "ftS4", "ftS5"],
          "siblings + own history x1": ["ftT1"], "siblings + own history x3": ["ftT3"],
          "siblings + own history x1, seed 1": ["ftT1S1"], "siblings + own history x1, seed 2": ["ftT1S2"], "siblings + own history x1, seed 3": ["ftT1S3"], "siblings + own history x1, seed 4": ["ftT1S4"],
          "siblings + own history x1, 5-seed avg": ["ftT1", "ftT1S1", "ftT1S2", "ftT1S3", "ftT1S4"], "siblings + own history x1, 1000 steps": ["ftT1L1000"]}


def get(c, tag, k):
    idx = pd.MultiIndex.from_frame(c["M"].keys)
    F = pd.read_parquet(IT3 / f"forecasts_oos-{tag}" / f"chronos-2-{tag}__test__cut{k}__padded.parquet").pivot_table(
        index=v.KEYS, columns="week", values="forecast", aggfunc="sum").reindex(idx).fillna(0.0).to_numpy(float)
    F[c["M"].U[:, -8:].sum(axis=1) == 0] = 0.0
    return F[c["valid"]]


def main():
    cuts = [es.load_cut(v.load_weekly("test"), k) for k in range(4)]
    ref = np.array([es.score(c, es.d40(c)) for c in cuts]); rows = []
    def rep(name, s):
        s = np.array(s); g = s - ref
        verdict = "ADOPT" if g.mean() >= 0.5 and (g > 0).sum() >= 3 else "reject"
        print(f"{name:62s} " + " ".join(f"{x:7.2f}" for x in s) + f"  mean {s.mean():7.2f}  vs D40 {g.mean():+.2f} won {(g > 0).sum()}/4  {verdict}")
        rows.append({"variant": name, **{f"cut{k}": round(x, 2) for k, x in enumerate(s)}, "mean": round(s.mean(), 2), "gain_vs_D40": round(g.mean(), 2), "cuts_won": int((g > 0).sum()), "verdict": verdict})
    print(f"{'D40':62s} " + " ".join(f"{x:7.2f}" for x in ref) + f"  mean {ref.mean():7.2f}")
    for name, tags in GROUPS.items():
        C = [np.mean([get(c, t, k) for t in tags], axis=0) for k, c in enumerate(cuts)]
        rep(f"[{name}] alone", [es.score(c, x) for c, x in zip(cuts, C)])
        rep(f"[{name}] D40 + it as 4th member", [es.score(c, 0.3 * c["mem"]["lgbm"] + 0.7 * np.mean([c["mem"][m] for m in es.D40_FM] + [x], axis=0)) for c, x in zip(cuts, C)])
        rep(f"[{name}] D40, half of the foundation part is it", [es.score(c, 0.3 * c["mem"]["lgbm"] + 0.35 * np.mean([c["mem"][m] for m in es.D40_FM], axis=0) + 0.35 * x) for c, x in zip(cuts, C)])
    pd.DataFrame(rows).to_csv(ROOT / "iterations" / "10-external-info" / "x1" / "results_adamex-svk_chronos2_ft_multi.csv", index=False)


if __name__ == "__main__":
    main()
