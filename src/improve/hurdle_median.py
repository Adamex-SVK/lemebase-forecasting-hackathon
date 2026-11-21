"""Zero-inflated hurdle model that estimates the conditional MEDIAN of weekly units.

Occurrence stage:  LightGBM binary classifier for P(units > 0) at week o+h-1 (h = 1..13).
Magnitude stage:   positive-only LightGBM L1 model (conditional median of units | units > 0)
                   plus a positive-only quantile model at alpha = 0.25.
Decision rule:     zero-inflated median via linear quantile matching between the positive
                   0.25-quantile and the positive median (see `decision`).

New origin-time features (computed here, features.py is NOT edited):
    cover_last     log1p(on_hand[o-1] / (mean(units[o-13:o]) + 1))   (0 when on_hand missing)
    cover_missing  1 when on_hand[o-1] is missing
    arrival        1 when on_hand[o-1] > 0 (known) and on_hand is 0/missing on [o-8, o-1)

Leakage: `extra_at(M, o)` reads ONLY columns < o of M.U / M.OH (units[o-13:o], on_hand[o-8:o-1],
on_hand[o-1]). `_leakage_check` randomises every column >= o and asserts the extras do not change.

Run (from the repo root):
    .venv/bin/python src/improve/hurdle_median.py --cuts 0 1 2 3 --seeds 0 1 2
Outputs:
    iterations/09-different-angles/forecasts/hurdle_cut{K}_seed{s}.npy      (raw decision forecast, n x 13)
    iterations/09-different-angles/forecasts/hurdle_cut{K}_seedavg.npy     (mean over seeds)
    iterations/09-different-angles/results_hurdle_median.csv
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "meta"))
import experiments_02 as e  # noqa: E402
import features as f  # noqa: E402
import train_gbm as g  # noqa: E402
import validate as v  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "iterations" / "09-different-angles"
FC = OUT / "forecasts"
H = f.H
N_CUTS = 4

# extra feature column order appended after `features.build_windows`'s columns (which end with h)
EXTRA = ["cover_last", "cover_missing", "arrival"]
CLF = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
           verbose=-1, num_threads=8)


def extra_at(M, o):
    """Origin-time features for every series at origin o. Reads ONLY weeks < o."""
    U = M.U[:, :o].astype(np.float64)        # every slice below is within columns < o
    OH = M.OH[:, :o].astype(np.float64)
    oh_last = OH[:, o - 1]
    missing = np.isnan(oh_last)

    level = U[:, o - 13:o].mean(axis=1) + 1.0
    cover_last = np.log1p(np.clip(oh_last, 0, None) / level)
    cover_last = np.where(missing, 0.0, cover_last)

    prior = OH[:, max(0, o - 8):o - 1]        # weeks [o-8, o-1), 0/missing == "dead stock"
    prior_dead = np.all((prior == 0) | np.isnan(prior), axis=1)
    arrival = ((~missing) & (oh_last > 0) & prior_dead).astype(np.float64)
    return np.column_stack([cover_last, missing.astype(np.float64), arrival]).astype(np.float32)


def _leakage_check(M, o):
    """Randomise all columns >= o in copies of U and OH; extras at o must be unchanged."""
    a = extra_at(M, o)
    U0, OH0 = M.U.copy(), M.OH.copy()
    try:
        rng = np.random.default_rng(123)
        M.U[:, o:] = rng.random(M.U[:, o:].shape) * 100
        M.OH[:, o:] = rng.random(M.OH[:, o:].shape) * 100
        b = extra_at(M, o)
    finally:
        M.U, M.OH = U0, OH0
    assert np.allclose(a, b, equal_nan=True), f"leakage check failed at origin {o}"


def extra_matrix_window(M, origins):
    """extras keyed by origin, shape (n, 3), for the build_windows rows."""
    return {int(o): extra_at(M, o) for o in origins}


def extend_windows(win, extras, n_extra=len(EXTRA)):
    """Append the origin-time extras to a build_windows dict's X (row-aligned via series/origin)."""
    if len(win["X"]) == 0:
        return np.zeros((0, win["X"].shape[1] + n_extra), np.float32)
    E = np.zeros((len(win["X"]), n_extra), dtype=np.float32)
    for o in np.unique(win["origin"]):
        rows = win["origin"] == o
        E[rows] = extras[int(o)][win["series"][rows]]
    return np.hstack([win["X"], E])


def extend_inference(M):
    X, scale, valid = f.inference_features(M)
    ex = extra_at(M, M.W)
    E = np.column_stack([np.repeat(ex[:, i], H) for i in range(ex.shape[1])]).astype(np.float32)
    return np.hstack([X, E]), scale, valid


def _fit(params, Xtr, ytr, Xva, yva, extra_name, max_rounds=3000):
    names = list(f.FEATURES) + EXTRA
    dtr = lgb.Dataset(Xtr, ytr, feature_name=names, free_raw_data=True)
    if Xva is not None and len(Xva) > 0 and len(np.unique(yva)) > 1:
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        m1 = lgb.train(params, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
        best = int(m1.best_iteration)
        m_all = lgb.train(params, lgb.Dataset(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]), feature_name=names),
                          num_boost_round=min(max_rounds, int(best * 1.1) + 1))
        return m_all, best
    m1 = lgb.train(params, dtr, num_boost_round=min(max_rounds, 300))
    return m1, min(max_rounds, 300)


def prepare_cut(full, k):
    df = e.cut_frame(full, k)
    train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut])
    weeks = pd.date_range(cut, periods=H, freq="7D")
    origins = f.make_origins(M.W)
    val_o = origins[-7:]
    tr_o = origins[origins <= val_o[0] - H]
    extras = extra_matrix_window(M, origins)
    _leakage_check(M, int(origins[-1]))
    w_tr = f.build_windows(M, tr_o)
    w_va = f.build_windows(M, val_o)
    w_all = f.build_windows(M, origins)
    X_tr, X_va, X_all = extend_windows(w_tr, extras), extend_windows(w_va, extras), extend_windows(w_all, extras)
    X_inf, scale, valid = extend_inference(M)
    has_pos_series = np.zeros(M.n, dtype=bool)
    has_pos_series[w_all["series"][w_all["y"] > 0]] = True
    return dict(M=M, train=train, hold=hold, weeks=weeks, origins=origins, tr_o=tr_o, val_o=val_o,
                w_tr=w_tr, w_va=w_va, w_all=w_all, X_tr=X_tr, X_va=X_va, X_all=X_all,
                X_inf=X_inf, scale=scale, valid=valid, has_pos=has_pos_series)


def fit_hurdle(P, seed, verbose=True, mag_rounds=400):
    """Returns raw decision forecast F [n x 13] and the three stage models' best iters."""
    t0 = time.time()
    M = P["M"]
    # 1) occurrence
    clf, b_clf = _fit(dict(CLF, seed=seed), P["X_tr"], (P["w_tr"]["y"] > 0).astype(np.float32),
                      P["X_va"], (P["w_va"]["y"] > 0).astype(np.float32), "clf")
    p = clf.predict(P["X_inf"]).reshape(M.n, H)

    # 2) magnitude on positive cells only
    Xpos = P["X_all"][P["w_all"]["y"] > 0]
    ypos = P["w_all"]["y"][P["w_all"]["y"] > 0]
    Xpos_va = P["X_va"][P["w_va"]["y"] > 0]
    ypos_va = P["w_va"]["y"][P["w_va"]["y"] > 0]
    med, b_med = _fit(dict(g.PARAMS, seed=seed), Xpos, ypos, Xpos_va, ypos_va, "med", max_rounds=mag_rounds)
    q25, b_q = _fit(dict(g.PARAMS, objective="quantile", alpha=0.25, seed=seed),
                    Xpos, ypos, Xpos_va, ypos_va, "q25", max_rounds=mag_rounds)
    m_pos = np.clip(med.predict(P["X_inf"]).reshape(M.n, H), 0, None) * P["scale"][:, None]
    m_q25 = np.clip(q25.predict(P["X_inf"]).reshape(M.n, H), 0, None) * P["scale"][:, None]
    m_pos[~P["has_pos"]] = 0.0
    m_q25[~P["has_pos"]] = 0.0
    F = decision(p, m_pos, m_q25, P["valid"], P["has_pos"])
    if verbose:
        print(f"    seed {seed}: clf {b_clf} med {b_med} q25 {b_q} ({time.time()-t0:.0f}s, "
              f"pos rows {len(ypos):,}/{len(P['X_all']):,})", flush=True)
    return F, dict(clf=b_clf, med=b_med, q25=b_q)


def decision(p, m_pos, m_q25, valid, has_pos):
    """Zero-inflated median: forecast 0 if P(units=0) >= 0.5, else the (matching) positive quantile."""
    p = np.clip(p, 0.0, 1.0)
    p0 = 1.0 - p
    zero = p0 >= 0.5
    denom = np.maximum(1.0 - p0, 1e-12)
    q = np.where(zero, 0.0, (0.5 - p0) / denom)           # in [0, 0.5)
    interp = m_q25 + (m_pos - m_q25) * (q - 0.25) / 0.25
    m_q = np.where(q <= 0.25, m_q25, interp)
    m_q = np.where(q >= 0.5, m_pos, m_q)
    F = np.clip(m_q, 1.0, np.maximum(m_pos, 1.0))         # nonzero median is at least 1
    F[zero] = 0.0
    F[~has_pos] = 0.0
    F[~valid] = 0.0
    return F


def variants(F_raw, M):
    r = np.clip(np.round(F_raw), 0, None)
    z = r.copy()
    z[M.U[:, -8:].sum(axis=1) == 0] = 0.0
    return {"hurdle_raw": F_raw, "hurdle_round": r, "hurdle_round_zero8": z}


def score_variants(P, F, rows, cut, seed, tag=""):
    for name, Fv in variants(F, P["M"]).items():
        r = e.score_F(P["M"], Fv, P["train"], P["hold"], P["weeks"])
        rows.append({"variant": name, "cut": cut, "seed": seed, "score": r["score"]})
        print(f"    {tag}{name:20s} cut {cut} seed {seed}: {r['score']:7.2f}", flush=True)


def run_cuts(full, cuts, seeds, force=False, mag_rounds=400):
    FC.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in cuts:
        print(f"cut {k}: preparing windows", flush=True)
        P = prepare_cut(full, k)
        print(f"  n={P['M'].n} W={P['M'].W} train rows {len(P['X_tr']):,} all rows {len(P['X_all']):,}", flush=True)
        for s in seeds:
            fF = FC / f"hurdle_cut{k}_seed{s}.npy"
            if fF.exists() and not force:
                F = np.load(fF)
                print(f"  cut {k} seed {s}: loaded {fF.name}", flush=True)
            else:
                F, bi = fit_hurdle(P, s, mag_rounds=mag_rounds)
                np.save(fF, F)
            score_variants(P, F, rows, k, s)
        paths = sorted(FC.glob(f"hurdle_cut{k}_seed[0-9]*.npy"))
        paths = [p for p in paths if "seedavg" not in p.name]
        if paths:
            Fa = np.mean([np.load(p) for p in paths], axis=0)
            np.save(FC / f"hurdle_cut{k}_seedavg.npy", Fa)
            score_variants(P, Fa, rows, k, -1, tag="avg ")
    return rows


def evaluation(full):
    """References (L1 alone, median13) + D40 reproduction + D40 with the hurdle in the LightGBM slot."""
    import ensemble_recipe as er
    ctx, cache = er.cut_contexts(), {}
    ref, swap, l1r, med13 = [], [], [], []
    for k in range(N_CUTS):
        M, train, hold, weeks, dead, Fl = ctx[k]
        l1r.append(e.score_F(M, np.clip(np.round(Fl), 0, None), train, hold, weeks)["score"])
        Fmed = np.repeat(np.median(M.U[:, -13:], axis=1, keepdims=True), H, axis=1).astype(float)
        med13.append(e.score_F(M, Fmed, train, hold, weeks)["score"])
        mod = np.mean([er.raw_practice(m, k, ctx, cache) for m in er.REFERENCE_MODELS], axis=0)
        mod[dead] = 0.0
        ref.append(er.score_cut(k, er.REFERENCE_W * Fl + (1 - er.REFERENCE_W) * mod, ctx))
        Hf = np.load(FC / f"hurdle_cut{k}_seedavg.npy")
        Hf = np.clip(np.round(Hf), 0, None)
        Hf[dead] = 0.0
        swap.append(er.score_cut(k, er.REFERENCE_W * Hf + (1 - er.REFERENCE_W) * mod, ctx))
    return np.array(l1r), np.array(med13), np.array(ref), np.array(swap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-adoption", action="store_true")
    ap.add_argument("--mag-rounds", type=int, default=400)
    a = ap.parse_args()
    full = v.load_weekly("test")
    rows = run_cuts(full, a.cuts, a.seeds, force=a.force, mag_rounds=a.mag_rounds)

    if not a.skip_adoption:
        l1r, med13, ref, swap = evaluation(full)
        for k in range(N_CUTS):
            rows += [{"variant": "l1_round_seed0", "cut": k, "seed": 0, "score": l1r[k]},
                     {"variant": "median13", "cut": k, "seed": 0, "score": med13[k]},
                     {"variant": "d40_reference", "cut": k, "seed": -1, "score": ref[k]},
                     {"variant": "d40_hurdle_swap", "cut": k, "seed": -1, "score": swap[k]}]
        print("\nreference / adoption:")
        print("  L1 rounded     " + " ".join(f"{x:7.2f}" for x in l1r) + f"   mean {l1r.mean():7.2f}")
        print("  median13       " + " ".join(f"{x:7.2f}" for x in med13) + f"   mean {med13.mean():7.2f}")
        print("  D40 ref        " + " ".join(f"{x:7.2f}" for x in ref) + f"   mean {ref.mean():7.2f}")
        print("  D40 + hurdle   " + " ".join(f"{x:7.2f}" for x in swap) + f"   mean {swap.mean():7.2f}")
        d = swap - ref
        print("  gain           " + " ".join(f"{x:+7.2f}" for x in d) + f"   mean {d.mean():+.2f}, cuts won {int((d > 0).sum())}/4")

    res = pd.DataFrame(rows, columns=["variant", "cut", "seed", "score"])
    res.to_csv(OUT / "results_hurdle_median.csv", index=False)
    pt = res.pivot_table(index=["variant", "seed"], columns="cut", values="score", aggfunc="mean")
    pt["mean"] = pt.mean(axis=1)
    print("\nall rows:\n" + pt.round(2).to_string())


if __name__ == "__main__":
    main()
