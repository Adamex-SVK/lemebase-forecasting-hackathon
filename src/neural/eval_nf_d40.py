"""Does a neural forecaster (iteration 06) improve the D40 recipe? Fixed weights only, no tuning on the cuts.

    .venv/bin/python src/neural/eval_nf_d40.py [--model NHITS] [--topic nhits-baseline]

Uses the D40 recipe code in src/meta/ensemble_recipe.py unchanged (same cuts, zero rule, rounding), with
the neural seed-average forecasts from iterations/06-neural/forecasts/{model}__test__cut{k}__padded.parquet.
Variants (all with weights fixed in advance):
  d40              0.3 * LightGBM + 0.7 * mean(moirai-2.0-small, tirex, chronos-t5-small)   (reference)
  nn_alone         the neural model alone (rounded)
  nn_4th_member    0.3 * LightGBM + 0.7 * mean(3 foundation models + neural)
  nn_replaces_lgbm 0.3 * neural   + 0.7 * mean(3 foundation models)
  nn_split_slot    0.15 * LightGBM + 0.15 * neural + 0.7 * mean(3 foundation models)
  equal5           mean(LightGBM, 3 foundation models, neural), zero rule on the whole mean
Adoption bar (TEAM-RULES.md section 2): mean gain over d40 >= +0.5 AND better on >= 3 of 4 cuts.
Writes iterations/06-neural/results_<handle>_<topic>.csv (default handle adamex-svk).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "meta"))
sys.path.insert(0, str(ROOT / "src" / "foundation"))
import ensemble_recipe as er  # noqa: E402
import experiments_02 as e  # noqa: E402
import validate as v  # noqa: E402

NF = ROOT / "iterations" / "06-neural"


def nn_matrix(model, k, ctx, suffix=""):
    M = ctx[k][0]
    fc = pd.read_parquet(NF / "forecasts" / f"{model}__test__cut{k}__padded{suffix}.parquet")
    return fc.pivot_table(index=v.KEYS, columns="week", values="forecast", aggfunc="sum").reindex(
        pd.MultiIndex.from_frame(M.keys)).fillna(0.0).to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NHITS")
    ap.add_argument("--topic", default="nhits-baseline")
    ap.add_argument("--handle", default="adamex-svk")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2], help="single seeds to report alone")
    a = ap.parse_args()
    ctx, cache = er.cut_contexts(), {}
    fms = er.REFERENCE_MODELS
    rows, scores = [], {}

    def blend(name, k, F):
        s = er.score_cut(k, F, ctx)
        scores.setdefault(name, [None] * er.N_CUTS)[k] = s
        rows.append({"variant": name, "model": a.model, "cut": k, "score": round(s, 4)})

    for k in range(er.N_CUTS):
        M, _, _, _, dead, Fl = ctx[k]
        Ffm = [er.raw_practice(m, k, ctx, cache) for m in fms]
        N = nn_matrix(a.model, k, ctx)
        fm3 = np.mean(Ffm, axis=0); fm3[dead] = 0.0
        fm4 = np.mean(Ffm + [N], axis=0); fm4[dead] = 0.0
        eq5 = np.mean(Ffm + [N, Fl], axis=0); eq5[dead] = 0.0
        blend("d40", k, 0.3 * Fl + 0.7 * fm3)
        blend("lgbm_alone", k, Fl)
        blend("nn_alone", k, N)
        for s in a.seeds:
            p = NF / "forecasts" / f"{a.model}__test__cut{k}__padded__seed{s}.parquet"
            if p.exists():
                blend(f"nn_alone_seed{s}", k, nn_matrix(a.model, k, ctx, f"__seed{s}"))
        blend("nn_4th_member", k, 0.3 * Fl + 0.7 * fm4)
        blend("nn_replaces_lgbm", k, 0.3 * N + 0.7 * fm3)
        blend("nn_split_slot", k, 0.15 * Fl + 0.15 * N + 0.7 * fm3)
        blend("equal5", k, eq5)

    ref = np.array(scores["d40"])
    lg = np.array(scores["lgbm_alone"])
    print(f"neural model {a.model}; D40 members {fms}; fixed weights, D40 code path (src/meta/ensemble_recipe.py)\n")
    print(f"{'variant':18s} {'cut0':>7s} {'cut1':>7s} {'cut2':>7s} {'cut3':>7s} {'mean':>7s}   vs d40 (won)   vs lgbm (won)")
    for name, sc in scores.items():
        sc = np.array(sc); d = sc - ref; dl = sc - lg
        ok = d.mean() >= er.ADOPT_MEAN and (d > 0).sum() >= er.ADOPT_CUTS
        print(f"{name:18s} " + " ".join(f"{x:7.2f}" for x in sc) + f" {sc.mean():7.2f}"
              f"   {d.mean():+6.2f} ({int((d > 0).sum())}/4)   {dl.mean():+6.2f} ({int((dl > 0).sum())}/4)"
              + ("   PASSES BAR" if ok and name != "d40" else ""))

    # paired per-article interval of the best non-reference blend vs d40
    cands = [n for n in scores if n not in ("d40", "lgbm_alone") and not n.startswith("nn_alone")]
    best = max(cands, key=lambda n: np.mean(scores[n]))
    rng = np.random.default_rng(0)
    print(f"\nbest blend {best} vs d40, paired over articles (score points, 95% bootstrap interval):")
    for k in range(er.N_CUTS):
        M, train, hold, weeks, dead, Fl = ctx[k]
        Ffm = [er.raw_practice(m, k, ctx, cache) for m in fms]
        N = nn_matrix(a.model, k, ctx)
        fm3 = np.mean(Ffm, axis=0); fm3[dead] = 0.0
        fm4 = np.mean(Ffm + [N], axis=0); fm4[dead] = 0.0
        eq5 = np.mean(Ffm + [N, Fl], axis=0); eq5[dead] = 0.0
        F = {"nn_4th_member": 0.3 * Fl + 0.7 * fm4, "nn_replaces_lgbm": 0.3 * N + 0.7 * fm3,
             "nn_split_slot": 0.15 * Fl + 0.15 * N + 0.7 * fm3, "equal5": eq5}[best]
        db = e.score_F(M, np.round(F), train, hold, weeks)["detail"]["mase"]
        dr = e.score_F(M, np.round(0.3 * Fl + 0.7 * fm3), train, hold, weeks)["detail"]["mase"]
        d = (db - dr).dropna().to_numpy()
        bs = [-100 * rng.choice(d, len(d)).mean() for _ in range(1000)]
        print(f"  cut {k}: {-100 * d.mean():+.2f}  [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")

    out = NF / f"results_{a.handle}_{a.topic}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
