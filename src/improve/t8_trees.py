"""T8 (iteration 08): replace the 0.3 LightGBM slot of recipe D40 with another trained tree model.

Recipe under test (D40):
    mean of the foundation forecasts (moirai-2.0-small, tirex, chronos-t5-small)
    -> 0 for articles with no sale in the last 8 weeks
    -> 0.3 * tree_model + 0.7 * ensemble
    -> round -> clip at 0

`tree_model` is one of the complete forecast sets in iterations/02-robust-eval/forecasts/:
    loss_F  LightGBM l1 (the D40 slot; seed 0 == base_cut{k}_seed0.npy)
    cb_F    CatBoost MAE
    loss_P  LightGBM P(units > 0); used only as the 0.5 zero gate on loss_F (item 2.3)
    d33     LightGBM with the D33 features (extra, complete)
    loss2_F_*  huber/fair/l2/tweedie/poisson; INCOMPLETE across the cuts -> diagnostic only

Fixed weights, no tuning. Scoring: src/validate.py D26 on the 4 standard cuts (src/experiments_02.py).

    .venv/bin/python src/improve/t8_trees.py evaluate
    .venv/bin/python src/improve/t8_trees.py build --variant catboost

Outputs: iterations/08-improvements/t8/results_adamex-svk_t8.csv
         iterations/08-improvements/t8/notes_adamex-svk_t8.md (evaluate)
         submissions/t8_v1.csv (build; only after an adopted variant)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "meta"))
import ensemble_recipe as er  # noqa: E402
import experiments_catboost as ec  # noqa: E402
import experiments_02 as e  # noqa: E402
import features as f  # noqa: E402
import make_submission as ms  # noqa: E402
import validate as v  # noqa: E402

H = f.H
FC = Path("/Users/adomi/Documents/hackathlon/iterations/02-robust-eval/forecasts")  # read-only, shared artifacts
OUT = ROOT / "iterations" / "08-improvements" / "t8"
W = er.REFERENCE_W
N_CUTS = er.N_CUTS
SEEDS = (0, 1, 2)
GATE = 0.5

# name -> forecast file prefix. Complete across 4 cuts x 3 seeds.
COMPLETE = {"loss_F": "loss_F", "cb_F": "cb_F", "loss_P": "loss_P", "d33": "d33"}
# Incomplete diagnostics: prefix + available cuts (see the notes file).
LOSS2 = {
    "loss2_F_huber": (0, 1),       # cut 0 (3 seeds), cut 1 (seed 0 only)
    "loss2_F_fair": (0, 1),
    "loss2_F_l2": (0, 1),
    "loss2_F_tweedie1.2": (0, 1),
    "loss2_F_tweedie1.5": (0,),    # cut 0 only
    "loss2_F_poisson": (0,),       # cut 0 only
}


def load_avg(prefix, k):
    """Mean forecast over the seeds that exist for cut k, and the seed list."""
    seeds = [s for s in SEEDS if (FC / f"{prefix}_cut{k}_seed{s}.npy").exists()]
    if not seeds:
        return None, seeds
    return np.mean([np.load(FC / f"{prefix}_cut{k}_seed{s}.npy") for s in seeds], axis=0), seeds


def load_seed(prefix, k, seed=0):
    return np.load(FC / f"{prefix}_cut{k}_seed{seed}.npy")


# ------------------------------------------------------------------ evaluation
def build_ctx():
    ctx = er.cut_contexts()
    cache, ens = {}, {}
    for k in range(N_CUTS):
        M, _, _, _, dead, _ = ctx[k]
        E = np.mean([er.raw_practice(m, k, ctx, cache) for m in er.REFERENCE_MODELS], axis=0)
        E[dead] = 0.0                      # D39 zero rule on the ensemble part only
        ens[k] = E
    return ctx, ens


def score_slot(T, k, ctx, ens):
    M, train, hold, weeks, dead, Fl = ctx[k]
    return er.score_cut(k, W * T + (1 - W) * ens[k], ctx)


def slot_forecast(name, k):
    """Forecast that replaces the 0.3 LightGBM slot, for cut k. None if unavailable."""
    if name == "lgbm_seed0":
        return load_seed("loss_F", k, 0), SEEDS[:1]
    if name == "lgbm_seedavg":
        return load_avg("loss_F", k)
    if name == "lgbm_gate":
        Fl, sl = load_avg("loss_F", k)
        Pr, _ = load_avg("loss_P", k)
        if Fl is None or Pr is None:
            return None, sl
        return np.where(Pr < GATE, 0.0, Fl), sl
    if name == "catboost":
        return load_avg("cb_F", k)
    if name == "lgbm+cb":
        Fl, sl = load_avg("loss_F", k)
        Fc, _ = load_avg("cb_F", k)
        return (0.5 * Fl + 0.5 * Fc), sl
    if name == "lgbm+cb_seed0":
        return 0.5 * load_seed("loss_F", k, 0) + 0.5 * load_seed("cb_F", k, 0), SEEDS[:1]
    if name == "d33":
        return load_avg("d33", k)
    if name in LOSS2:
        return load_avg(name, k)
    raise SystemExit(f"unknown slot {name!r}")


def variant_cuts(name):
    """Cuts this variant can be scored on."""
    if name in LOSS2:
        return list(LOSS2[name])
    return list(range(N_CUTS))


def evaluate():
    t0 = time.time()
    ctx, ens = build_ctx()
    ref = np.array([score_slot(load_seed("loss_F", k, 0), k, ctx, ens) for k in range(N_CUTS)])
    print(f"reference D40 by cut: " + " ".join(f"{x:7.2f}" for x in ref) + f"   mean {ref.mean():7.2f}")

    specs = [
        ("D40 reference (loss_F seed0 slot)", "lgbm_seed0", "sanity: must reproduce D40 exactly"),
        ("D40 with LGBM seed-avg slot (control)", "lgbm_seedavg", "isolates the seed-averaging gain"),
        ("LightGBM l1 + zero gate (loss_P<0.5)", "lgbm_gate", "item 2.3 gate as the tree member"),
        ("CatBoost MAE (cb_F)", "catboost", "3-seed average"),
        ("LightGBM + CatBoost (1/2 each)", "lgbm+cb", "3-seed averages, equal weights"),
        ("LightGBM + CatBoost (1/2 each, seed0)", "lgbm+cb_seed0", "single seed, no seed averaging"),
        ("0.3 slot split 0.15 LGBM + 0.15 CatBoost", "lgbm+cb", "identical to the 1/2 average above"),
        ("d33 LightGBM (extra)", "d33", "complete 3-seed set, different features"),
    ] + [(f"loss2 {n}", n, "INCOMPLETE: scored only on the cuts that exist") for n in LOSS2]

    rows = []
    for label, name, note in specs:
        cuts = variant_cuts(name)
        scores, used = {}, {}
        for k in cuts:
            T, seeds = slot_forecast(name, k)
            if T is None:
                continue
            scores[k] = score_slot(T, k, ctx, ens)
            used[k] = len(seeds)
        if not scores:
            continue
        s = np.array([scores[k] for k in sorted(scores)])
        r = ref[[k for k in sorted(scores)]]
        gain = s - r
        rows.append({
            "variant": label, "slot": name, "n_cuts": len(s),
            **{f"cut{k}": round(float(scores[k]), 4) for k in sorted(scores)},
            "mean": round(float(s.mean()), 4),
            "D40_mean_same_cuts": round(float(r.mean()), 4),
            "gain_vs_D40": round(float(gain.mean()), 4),
            "cuts_won": int((gain > 0).sum()),
            "cuts_total": len(s),
            "seeds_per_cut": " ".join(str(used[k]) for k in sorted(used)),
            "verdict": ("ADOPT" if len(s) == N_CUTS and gain.mean() >= er.ADOPT_MEAN and (gain > 0).sum() >= er.ADOPT_CUTS
                        else ("keep" if len(s) == N_CUTS else "incomplete")),
            "note": note,
        })
        print(f"{label:42s} " + " ".join(f"{scores.get(k, float('nan')):7.2f}" for k in range(N_CUTS))
              + f"   mean {s.mean():7.2f}  gain {gain.mean():+6.2f}  won {int((gain>0).sum())}/{len(s)}  {rows[-1]['verdict']}")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results_adamex-svk_t8.csv"
    df.to_csv(path, index=False)
    print(f"\nwrote {path}")
    print(f"runtime {time.time() - t0:.0f}s")
    return df, ref


# ------------------------------------------------------------------ final submission
def catboost_final(seeds=SEEDS):
    df = v.load_weekly("test")
    M = f.ShopMatrix(df)                                   # all weeks, no holdout (like make_lgbm_final.py)
    origins = f.make_origins(M.W)
    val_o = origins[-7:]
    tr_o = origins[origins <= val_o[0] - H]
    P = dict(M=M, origins=origins,
             w_tr=f.build_windows(M, tr_o), w_va=f.build_windows(M, val_o),
             w_all=f.build_windows(M, origins), inf=f.inference_features(M))
    Fs = []
    for s in seeds:
        t = time.time()
        F, best = ec.fit_cb(P, s)
        Fs.append(F)
        print(f"  catboost final seed {s}: {best} trees ({time.time() - t:.0f}s)", flush=True)
    return M.keys["article_id"].to_numpy(), np.mean(Fs, axis=0)


def build(variant):
    ids = ms.history_ids()
    if variant == "catboost":
        keys, T = catboost_final()
    elif variant in ("lgbm+cb", "lgbm+cb_seed0"):
        keys, Tc = catboost_final(SEEDS if variant == "lgbm+cb" else SEEDS[:1])
        L = pd.read_csv(ms.SUB_DIR / "lgbm_own_v1.csv").set_index("article_id").loc[ids].to_numpy(float)
        T = 0.5 * (L + Tc)
    else:
        raise SystemExit(f"build not implemented for {variant!r} (only catboost and lgbm+cb)")

    order = pd.Index(keys).get_indexer(ids)
    assert (order >= 0).all(), "final forecast is missing article ids"
    T = T[order]
    E = np.mean([er.final_matrix(m, ids) for m in er.REFERENCE_MODELS], axis=0)
    E[er.dead_mask(ids)] = 0.0
    F = np.clip(np.round(W * T + (1 - W) * E), 0, None)
    out = ROOT / "submissions" / "t8_v1.csv"
    ms.write_submission(F, ids, out)
    ms.validate_submission(out)
    print(f"wrote {out} and it passes the format checks")
    print(f"  variant={variant} | {len(ids)} articles | mean {F.mean():.2f} | "
          f"{100 * (F.max(axis=1) == 0).mean():.1f}% forecast 0 every week")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("evaluate", help="score the D40 recipe with other tree models in the 0.3 slot")
    p.set_defaults(func=lambda a: evaluate())
    b = sub.add_parser("build", help="train the adopted tree model on all history and write submissions/t8_v1.csv")
    b.add_argument("--variant", required=True, choices=["catboost", "lgbm+cb", "lgbm+cb_seed0"])
    b.set_defaults(func=lambda a: build(a.variant))
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
