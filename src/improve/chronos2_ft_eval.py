"""X1c scoring: fine-tuned Chronos-2 (sibling shops only) alone and in the D40 blend. Usage: chronos2_ft_eval.py TAG [TAG ...]"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "improve"))
import ens_search as es, validate as v  # noqa: E402
warnings.filterwarnings("ignore")
IT3 = ROOT / "iterations" / "03-foundation-model"


def main(tags):
    cuts = [es.load_cut(v.load_weekly("test"), k) for k in range(4)]
    ref = np.array([es.score(c, es.d40(c)) for c in cuts]); rows = []
    def rep(name, s):
        s = np.array(s); g = s - ref
        print(f"{name:58s} " + " ".join(f"{x:7.2f}" for x in s) + f"  mean {s.mean():7.2f}  vs D40 {g.mean():+.2f} won {(g > 0).sum()}/4")
        rows.append({"variant": name, **{f"cut{k}": round(x, 2) for k, x in enumerate(s)}, "mean": round(s.mean(), 2), "gain_vs_D40": round(g.mean(), 2), "cuts_won": int((g > 0).sum())})
    print(f"{'D40':58s} " + " ".join(f"{x:7.2f}" for x in ref) + f"  mean {ref.mean():7.2f}")
    rep("chronos-2 zero-shot alone", [es.score(c, c["mem"]["chronos-2"]) for c in cuts])
    for tag in tags:
        C = []
        for k, c in enumerate(cuts):
            idx = pd.MultiIndex.from_frame(c["M"].keys)
            F = pd.read_parquet(IT3 / f"forecasts_oos-{tag}" / f"chronos-2-{tag}__test__cut{k}__padded.parquet").pivot_table(
                index=v.KEYS, columns="week", values="forecast", aggfunc="sum").reindex(idx).fillna(0.0).to_numpy(float)
            F[c["M"].U[:, -8:].sum(axis=1) == 0] = 0.0
            C.append(F[c["valid"]])
        rep(f"[{tag}] fine-tuned chronos-2 alone", [es.score(c, x) for c, x in zip(cuts, C)])
        rep(f"[{tag}] D40 with it as 4th member", [es.score(c, 0.3 * c["mem"]["lgbm"] + 0.7 * np.mean([c["mem"][m] for m in es.D40_FM] + [x], axis=0)) for c, x in zip(cuts, C)])
        rep(f"[{tag}] D40 with chronos-t5-small replaced", [es.score(c, 0.3 * c["mem"]["lgbm"] + 0.7 * np.mean([c["mem"][m] for m in ["moirai-2.0-small", "tirex"]] + [x], axis=0)) for c, x in zip(cuts, C)])
    out = ROOT / "iterations" / "10-external-info" / "x1"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"results_adamex-svk_chronos2_ft_{'_'.join(tags)}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
