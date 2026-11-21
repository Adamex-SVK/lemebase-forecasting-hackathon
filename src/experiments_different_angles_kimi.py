"""Iteration 09: different-angles portfolio (Kimi worktree).

Subcommands
-----------
audit        Step 0 reproduction + the A-level selector diagnostic (cached bases only).
pooled-level Horizon-pooled L1 LightGBM level model (angle B), canonical 4 cuts.
analog       Within-shop analog level donors (angle C), sanity test + canonical 4 cuts.

Evidence protocol (from the plan):
  * every score is the D26 metric on integer, zero-ruled, clipped actions;
  * overlapping W=0,4,...,48 cuts are correlated and diagnostic only;
  * canonical adoption uses the four standard cuts, >= +0.5 mean and >= 3/4 cut wins
    with a paired per-article interval and non-negative w1-7 / w8-13 changes;
  * no feature, donor outcome, or fit may use data at or after the target origin.

Usage
-----
  .venv/bin/python src/experiments_different_angles_kimi.py audit
  .venv/bin/python src/experiments_different_angles_kimi.py pooled-level
  .venv/bin/python src/experiments_different_angles_kimi.py analog
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "meta"))
sys.path.insert(0, str(ROOT / "src" / "foundation"))
import fforma as ff  # noqa: E402
import features as f  # noqa: E402
import validate as v  # noqa: E402

H = v.HORIZON
KEYS = v.KEYS
OUT_DIR = ROOT / "iterations" / "09-different-angles"
MODELS = ["moirai-2.0-small", "tirex", "chronos-t5-small"]
OFFSETS = list(range(0, 49, 4))
DIAG_OFFSETS = [0, 4, 8, 12]
CANONICAL_W = [0, 13, 26, 39]
SELECTOR_FEATURES = ["trend", "zshare13", "cv26"]
QUANTILES = np.round(np.arange(0.1, 1.0, 0.1), 1)


# ------------------------------------------------------------------ scoring helpers
def post(F, c, dead_zero=True):
    """Zero rule (optional) -> round -> clip, the scored integer action."""
    G = np.array(F, dtype=float, copy=True)
    if dead_zero:
        G[c["dead"]] = 0.0
    return np.clip(np.round(G), 0, None)


def score_G(c, G, cols=None):
    """Exact score of an already-posted action matrix; cols selects weeks (w1-7 = 0:7)."""
    A, s, ok = c["A"], c["s"], c["valid"]
    if cols is not None:
        A, G = A[:, cols], G[:, cols]
    per = np.abs(G - A).mean(axis=1) / s
    return float((1.0 - per[ok].mean()) * 100.0)


def per_article_loss_G(c, G, cols=None):
    A, s = c["A"], c["s"]
    if cols is not None:
        A, G = A[:, cols], G[:, cols]
    return np.abs(G - A).mean(axis=1) / s


def flat_action(level, c, dead_zero=True):
    F = np.repeat(np.asarray(level, dtype=float)[:, None], H, axis=1)
    return post(F, c, dead_zero)


def bootstrap_interval(per_article_diff, n_boot=2000, seed=0):
    d = np.asarray(per_article_diff, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = -100.0 * d[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_summary(cut_diffs):
    """Mean gain and interval for a dict {cut: per-article (candidate - ref) MASE}.

    Article IDs are resampled once across all cuts (protocol item 5)."""
    diffs = pd.concat(cut_diffs, names=["cut"]).dropna()
    n = diffs.shape[1]
    rng = np.random.default_rng(0)
    articles = np.array([idx for idx, _ in diffs.index], dtype=object)
    uniq = pd.unique(articles)
    by_article = {a: diffs.loc[a].to_numpy() for a in uniq}
    means = []
    for _ in range(2000):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        vals = np.array([by_article[a] for a in draw])
        means.append(-100.0 * np.nanmean(vals))
    return float(-100.0 * np.nanmean(diffs.to_numpy())), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


# ------------------------------------------------------------------ cut contexts
def set_models(models):
    ff.MODELS = list(models)
    ff.set_ff_bases("lgbm,median13,zero")


def context_from_frames(df, extra=None):
    """Build the fforma-style context for an arbitrary truncated history frame."""
    train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut])
    weeks = pd.date_range(cut, periods=H, freq="7D")
    keys = M.keys
    A = hold.pivot_table(index=KEYS, columns="week", values="units").reindex(
        pd.MultiIndex.from_frame(keys)).to_numpy(float)
    scale = np.abs(np.diff(M.U.astype(np.float64), axis=1)).mean(axis=1)
    valid = np.isfinite(scale) & (scale > 0)
    s = np.where(valid, scale, 1.0)
    dead = M.U[:, -8:].sum(axis=1) == 0
    out = {"M": M, "train": train, "hold": hold, "weeks": weeks, "keys": keys,
           "A": A, "s": s, "valid": valid, "dead": dead}
    if extra:
        out.update(extra)
    return out


def canonical_contexts():
    """The four standard cuts with the frozen D40 foundation forecasts."""
    import ensemble_recipe as er
    ctx, cache = er.cut_contexts(), {}
    out = {}
    for k in range(4):
        M, train, hold, weeks, dead, Fl = ctx[k]
        E = np.mean([er.raw_practice(m, k, ctx, cache) for m in MODELS], axis=0)
        E[dead] = 0.0
        raw = 0.3 * Fl + 0.7 * E
        c = context_from_frames(er.e.cut_frame(v.load_weekly("test"), k))
        assert np.allclose(c["M"].U, M.U)
        assert np.allclose(c["dead"], dead)
        c["raw"] = raw
        c["d40"] = raw
        out[k] = c
    return out


def overlapping_context(W):
    c = ff.load_cut(W)
    c["raw"] = ff.simple_blend(c, 0.3)
    c["d40"] = c["raw"]
    return c


def d40_level(c):
    return c["raw"].mean(axis=1)


def canonical_d40_G(c):
    """The frozen D40 action: the zero rule is applied to the foundation ensemble *before*
    the LightGBM blend (D39/D40), so dead rows can keep 0.3 x LightGBM."""
    return np.clip(np.round(c["raw"]), 0, None)


# ------------------------------------------------------------------ oracle candidates
def candidate_levels(c):
    """The predeclared eleven history-only flat levels (D3)."""
    U = c["M"].U.astype(np.float64)
    W = U.shape[1]

    def mean(w):
        w = min(w, W)
        return U[:, -w:].mean(axis=1)

    def med(w):
        w = min(w, W)
        return np.median(U[:, -w:], axis=1)

    raw = c["raw"]
    return {
        "mean4": mean(4), "mean8": mean(8), "mean13": mean(13),
        "mean26": mean(26), "mean52": mean(52),
        "median8": med(8), "median13": med(13), "median26": med(26),
        "lgbm": ff.simple_blend(c, 1.0).mean(axis=1),
        "ensemble": ff.ensemble_raw(c).mean(axis=1),
        "d40": raw.mean(axis=1),
    }


CANDIDATES = ["mean4", "mean8", "mean13", "mean26", "mean52",
              "median8", "median13", "median26", "lgbm", "ensemble", "d40"]


def oracle_actions(levels, c, names, dead_zero=True):
    """Flat actions for each candidate: shape (n, K, H), posted and rounded."""
    L = np.stack([np.asarray(levels[nm], dtype=float) for nm in names], axis=1)
    G = np.repeat(L[:, :, None], H, axis=2)
    if dead_zero:
        G[c["dead"], :, :] = 0.0
    return np.clip(np.round(G), 0, None)


def oracle_select(c, levels, names, dead_zero=True):
    G = oracle_actions(levels, c, names, dead_zero)
    loss = np.abs(G - c["A"][:, None, :]).mean(axis=2) / c["s"][:, None]
    pick = loss.argmin(axis=1)
    Gsel = G[np.arange(len(pick)), pick]
    ref = names.index("d40") if "d40" in names else 0
    return {"G": Gsel, "score": score_G(c, Gsel), "pick": pick,
            "switch_share": float((pick != ref).mean()), "loss": loss}


def oracle_null(c, levels, names, n_rep=200, seed=0):
    """Winner's-curse null: shuffle each non-D40 candidate's forecasts within
    origin-defined dead8 x scale-quintile strata, keep D40 fixed, recompute O_sel11."""
    G = oracle_actions(levels, c, names, True)
    di = names.index("d40")
    n = G.shape[0]
    q = pd.qcut(c["s"], 5, labels=False, duplicates="drop")
    strata = np.array([f"d{int(d)}q{int(x)}" for d, x in zip(c["dead"], np.nan_to_num(q, nan=0))])
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_rep):
        Gp = G.copy()
        for k in range(G.shape[1]):
            if k == di:
                continue
            for st in np.unique(strata):
                idx = np.flatnonzero(strata == st)
                Gp[idx, k] = G[idx[rng.permutation(len(idx))], k]
        loss = np.abs(Gp - c["A"][:, None, :]).mean(axis=2) / c["s"][:, None]
        pick = loss.argmin(axis=1)
        Gsel = Gp[np.arange(n), pick]
        scores.append(score_G(c, Gsel))
    return np.asarray(scores)


# ------------------------------------------------------------------ Step 0 diagnostics
def audit_oracles(cuts):
    rows = []
    for W, c in cuts.items():
        lev = candidate_levels(c)
        d40 = score_G(c, post(c["raw"], c))
        flat_mean = flat_action(c["raw"].mean(axis=1), c)
        flat_med = flat_action(np.median(c["raw"], axis=1), c)
        medA = np.median(c["A"], axis=1)
        medA0 = medA.copy()
        medA0[c["dead"]] = 0.0
        oracle_level = medA[:, None] + (c["raw"] - c["raw"].mean(axis=1)[:, None])
        oracle_level_G = post(oracle_level, c)
        oracle_flat = flat_action(medA, c, dead_zero=False)
        oracle_flat0 = flat_action(medA, c, dead_zero=True)
        sel2_0 = oracle_select(c, lev, ["mean26", "d40"], True)
        sel11_0 = oracle_select(c, lev, CANDIDATES, True)
        sel2_n = oracle_select(c, lev, ["mean26", "d40"], False)
        sel11_n = oracle_select(c, lev, CANDIDATES, False)
        base = {"section": "oracle", "cut": W, "scope": "all13",
                "d40": d40, "flat_mean": score_G(c, flat_mean),
                "flat_median_d40": score_G(c, flat_med),
                "oracle_level_d40shape": score_G(c, oracle_level_G),
                "oracle_flat_median": score_G(c, oracle_flat),
                "oracle_flat_median_zero": score_G(c, oracle_flat0),
                "o_sel2_zero": sel2_0["score"], "o_sel11_zero": sel11_0["score"],
                "o_sel2_nozero": sel2_n["score"], "o_sel11_nozero": sel11_n["score"],
                "sel11_switch_share": sel11_0["switch_share"]}
        rows.append(base)
        for scope, cols in [("w1_7", np.arange(0, 7)), ("w8_13", np.arange(7, H))]:
            rows.append({"section": "oracle_horizon", "cut": W, "scope": scope,
                         "d40": score_G(c, post(c["raw"], c), cols),
                         "o_sel2_zero": score_G(c, sel2_0["G"], cols),
                         "o_sel11_zero": score_G(c, sel11_0["G"], cols),
                         "oracle_flat_median_zero": score_G(c, oracle_flat0, cols)})
    return pd.DataFrame(rows)


def audit_d1(canon):
    rows, details = [], {}
    for k, c in canon.items():
        G = canonical_d40_G(c)
        det = pd.Series(per_article_loss_G(c, G), index=pd.MultiIndex.from_frame(c["keys"]))
        det = det[c["valid"]]
        details[k] = det
        tot = det.sum()
        for pct in (1, 5, 10):
            q = np.percentile(det, 100 - pct)
            rows.append({"section": "D1_share", "cut": k, "metric": f"top{pct}pct_mase_share",
                         "value": float(det[det >= q].sum() / tot)})
        rows.append({"section": "D1", "cut": k, "metric": "n_scored", "value": float(len(det))})
        rows.append({"section": "D1", "cut": k, "metric": "mean_mase", "value": float(det.mean())})
    common = None
    for det in details.values():
        idx = det.index
        common = idx if common is None else common.intersection(idx)
    for a in range(4):
        for b in range(a + 1, 4):
            for scope, dfs in [("own", (details[a], details[b])),
                               ("common", (details[a].loc[common], details[b].loc[common]))]:
                ta = set(dfs[0].nlargest(max(1, int(round(0.05 * len(dfs[0]))))).index)
                tb = set(dfs[1].nlargest(max(1, int(round(0.05 * len(dfs[1]))))).index)
                rows.append({"section": "D1_overlap", "cut": f"{a}-{b}",
                             "metric": f"worst5pct_overlap_{scope}_A",
                             "value": len(ta & tb) / max(1, len(ta))})
                rows.append({"section": "D1_overlap", "cut": f"{a}-{b}",
                             "metric": f"worst5pct_jaccard_{scope}",
                             "value": len(ta & tb) / max(1, len(ta | tb))})
    rows.append({"section": "D1_overlap", "cut": "common", "metric": "n_common_scored",
                 "value": float(len(common))})
    return pd.DataFrame(rows)


def audit_d5(c):
    keys = c["keys"]
    aid = keys["article_id"].to_numpy()
    hit = np.flatnonzero(aid == "A139270")
    if len(hit) == 0:
        return pd.DataFrame()
    i = hit[0]
    G = post(c["raw"], c)
    mase = float(np.abs(G[i] - c["A"][i]).mean() / c["s"][i])
    n_scored = int(c["valid"].sum())
    return pd.DataFrame([{"section": "D5", "cut": c.get("W", 40), "metric": "A139270_scale",
                          "value": float(c["s"][i])},
                         {"section": "D5", "cut": c.get("W", 40), "metric": "A139270_holdout_units",
                          "value": float(c["A"][i].sum())},
                         {"section": "D5", "cut": c.get("W", 40), "metric": "A139270_mase",
                          "value": mase},
                         {"section": "D5", "cut": c.get("W", 40), "metric": "A139270_score_points",
                          "value": 100.0 * mase / n_scored}])


# ------------------------------------------------------------------ A. selector diagnostic
def selector_features(c):
    U = c["M"].U.astype(np.float64)
    mean4 = U[:, -4:].mean(axis=1)
    mean13 = U[:, -13:].mean(axis=1)
    mean26 = U[:, -26:].mean(axis=1)
    trend = np.log1p(mean4) - np.log1p(mean13)
    zshare13 = (U[:, -13:] == 0).mean(axis=1)
    cv26 = U[:, -26:].std(axis=1) / (mean26 + 1e-9)
    return np.column_stack([trend, zshare13, cv26])


def selector_losses(c):
    lev = candidate_levels(c)
    G = oracle_actions(lev, c, ["d40", "mean26"], True)
    loss = np.abs(G - c["A"][:, None, :]).mean(axis=2) / c["s"][:, None]
    return loss


def selector_training_data(c):
    return {"X": selector_features(c), "loss": selector_losses(c), "valid": c["valid"]}


def fit_stump(train_data):
    """One predeclared split: feature in {trend,zshare13,cv26}, threshold at training
    quantiles 0.1..0.9, direction chosen by the exact scaled-absolute loss."""
    xs = np.vstack([d["X"] for d in train_data])
    base = float(np.mean([d["loss"][d["valid"], 0].mean() for d in train_data]))
    best = {"train_loss": base, "feature": None, "threshold": None, "direction": None,
            "baseline": base}
    for fi in range(xs.shape[1]):
        for t in np.nanquantile(xs[:, fi], QUANTILES):
            for direction in (0, 1):
                total = 0.0
                for d in train_data:
                    left = d["X"][:, fi] <= t
                    choose_left = direction
                    sel = np.where(left, choose_left, 1 - choose_left)
                    loss = d["loss"][np.arange(len(sel)), sel]
                    total += loss[d["valid"]].mean()
                total /= len(train_data)
                if total < best["train_loss"] - 1e-12:
                    best = {"train_loss": total, "feature": fi, "threshold": float(t),
                            "direction": direction, "baseline": base}
    return best


def apply_stump(X, stump):
    if stump["feature"] is None:
        return np.zeros(len(X), dtype=int)
    left = X[:, stump["feature"]] <= stump["threshold"]
    return np.where(left, stump["direction"], 1 - stump["direction"]).astype(int)


def selector_action(c, stump):
    lev = candidate_levels(c)
    levels = np.stack([lev["d40"], lev["mean26"]], axis=1)
    pick = apply_stump(selector_features(c), stump)
    level = levels[np.arange(len(pick)), pick]
    return {"G": flat_action(level, c), "pick": pick, "level": level}


def shuffle_within_strata(X, c, rng):
    Xs = X.copy()
    q = pd.qcut(c["s"], 5, labels=False, duplicates="drop")
    strata = np.array([f"d{int(d)}q{int(x)}" for d, x in zip(c["dead"], np.nan_to_num(q, nan=0))])
    for st in np.unique(strata):
        idx = np.flatnonzero(strata == st)
        Xs[idx] = X[idx[rng.permutation(len(idx))]]
    return Xs


def previous_cut_best(c, prior_cuts):
    """Pick the lower-loss level at the smallest eligible prior W, D40 fallback."""
    p = prior_cuts[0]
    p_loss = selector_losses(p)
    p_pick = (p_loss[:, 1] < p_loss[:, 0]) & p["valid"]
    pick_df = pd.Series(p_pick.astype(int), index=pd.MultiIndex.from_frame(p["keys"]))
    target_idx = pd.MultiIndex.from_frame(c["keys"])
    pick = pick_df.reindex(target_idx).fillna(0).to_numpy(int)
    lev = candidate_levels(c)
    levels = np.stack([lev["d40"], lev["mean26"]], axis=1)
    level = levels[np.arange(len(pick)), pick]
    return {"G": flat_action(level, c), "pick": pick, "level": level}


def run_selector(cuts):
    rows = []
    valid_targets = [W for W in DIAG_OFFSETS if W in cuts]
    base = {W: score_G(cuts[W], post(cuts[W]["raw"], cuts[W])) for W in valid_targets}
    oracle = {W: oracle_select(cuts[W], candidate_levels(cuts[W]), CANDIDATES, True) for W in valid_targets}
    sel2 = {W: oracle_select(cuts[W], candidate_levels(cuts[W]), ["mean26", "d40"], True) for W in valid_targets}
    for W in valid_targets:
        c = cuts[W]
        prior = [cuts[p] for p in OFFSETS if p >= W + 13 and p in cuts]
        if len(prior) < 1:
            continue
        train = [selector_training_data(pc) for pc in prior]
        stump = fit_stump(train)
        act = selector_action(c, stump)
        prev = previous_cut_best(c, prior)
        d40 = base[W]
        flat_d40 = score_G(c, flat_action(d40_level(c), c))
        o2 = sel2[W]["score"]
        o11 = oracle[W]["score"]
        gain = score_G(c, act["G"]) - d40
        gain_vs_flat_d40 = score_G(c, act["G"]) - flat_d40
        gain_prev = score_G(c, prev["G"]) - d40
        c2 = gain / (o2 - d40) if abs(o2 - d40) > 1e-9 else np.nan
        c11 = gain / (o11 - d40) if abs(o11 - d40) > 1e-9 else np.nan
        rows.append({"section": "A_selector", "cut": W, "n_train_cuts": len(prior),
                     "feature": (SELECTOR_FEATURES[stump["feature"]] if stump["feature"] is not None else "none"),
                     "threshold": stump["threshold"], "direction": stump["direction"],
                     "train_loss": stump["train_loss"], "train_d40_loss": stump["baseline"],
                     "d40": d40, "flat_d40": flat_d40, "selector": score_G(c, act["G"]),
                     "gain": gain, "gain_vs_flat_d40": gain_vs_flat_d40,
                     "switched_to_mean26": int(act["pick"].sum()),
                     "previous_cut_best": score_G(c, prev["G"]), "gain_previous": gain_prev,
                     "o_sel2": o2, "o_sel11": o11, "gain_o_sel2": o2 - d40, "gain_o_sel11": o11 - d40,
                     "C2": c2, "C11": c11})
        for scope, cols in [("w1_7", np.arange(0, 7)), ("w8_13", np.arange(7, H))]:
            rows.append({"section": "A_selector_horizon", "cut": W, "scope": scope,
                         "d40": score_G(c, post(c["raw"], c), cols),
                         "selector": score_G(c, act["G"], cols),
                         "gain": score_G(c, act["G"], cols) - score_G(c, post(c["raw"], c), cols)})
        # permutation control: shuffle training features only, keep losses and target fixed
        rng = np.random.default_rng(1000 + W)
        ctrl = []
        for _ in range(20):
            train_s = [{"X": shuffle_within_strata(d["X"], pc, rng),
                        "loss": d["loss"], "valid": d["valid"]} for d, pc in zip(train, prior)]
            st = fit_stump(train_s)
            ctrl.append(score_G(c, selector_action(c, st)["G"]) - d40)
        rows.append({"section": "A_selector_shuffle", "cut": W, "n": len(ctrl),
                     "gain_mean": float(np.mean(ctrl)),
                     "gain_p2_5": float(np.percentile(ctrl, 2.5)),
                     "gain_p97_5": float(np.percentile(ctrl, 97.5)),
                     "real_gain": gain,
                     "p_value": float(np.mean(np.asarray(ctrl) >= gain))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B. pooled L1 level
def pooled_windows(M, origins):
    """Horizon-pooled training rows: identical origin features, no h column, weight 1/13."""
    w = f.build_windows(M, origins)
    return {"X": w["X"][:, :-1], "y": w["y"], "w": w["w"] / H, "n_windows": w["n_windows"]}


def fit_pooled_level(M, seed=0):
    import lightgbm as lgb
    import train_gbm as g
    origins = f.make_origins(M.W)
    val_o = origins[-7:]
    tr_o = origins[origins <= val_o[0] - H]
    w_tr = pooled_windows(M, tr_o)
    w_va = pooled_windows(M, val_o)
    params = dict(g.PARAMS)
    params["seed"] = seed
    names = f.FEATURES[:-1]
    dtr = lgb.Dataset(w_tr["X"], w_tr["y"], weight=w_tr["w"], feature_name=names)
    dva = lgb.Dataset(w_va["X"], w_va["y"], weight=w_va["w"], reference=dtr)
    m1 = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    best = m1.best_iteration
    w_all = pooled_windows(M, origins)
    m2 = lgb.train(params, lgb.Dataset(w_all["X"], w_all["y"], weight=w_all["w"],
                                      feature_name=names), num_boost_round=int(best * 1.1) + 1)
    return {"model": m2, "best_iter": int(best), "n_train_windows": w_tr["n_windows"]}


def pooled_prediction(M, model):
    X, scale, valid = f.features_at(M, M.W)
    pred = model.predict(X)
    level = np.clip(pred * scale, 0, None)
    level[~valid] = 0.0
    return level, scale, valid


def paired_interval(diffs_by_cut, n_boot=2000, seed=0):
    """Article-level MASE differences for {cut: Series indexed like the article keys};
    resample article IDs once across all cuts."""
    df = pd.concat(diffs_by_cut, axis=1)
    vals = df.to_numpy(float)
    observed = -100.0 * np.nanmean(vals)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(vals), len(vals))
        means[b] = -100.0 * np.nanmean(vals[idx])
    return float(observed), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cmd_pooled_level(args):
    set_models(args.models)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    c0 = canonical_contexts()
    rows, losses, diffs, diffs_17, diffs_813 = [], {}, {}, {}, {}
    cand_actions = {}
    for k, c in c0.items():
        t0 = time.time()
        fit = fit_pooled_level(c["M"], seed=args.seed)
        level, scale, valid = pooled_prediction(c["M"], fit["model"])
        Gc = post(np.repeat(level[:, None], H, axis=1), c, dead_zero=True)
        Gd = canonical_d40_G(c)
        cand_actions[k] = Gc
        idx = pd.MultiIndex.from_frame(c["keys"])
        det_c = pd.Series(per_article_loss_G(c, Gc), index=idx)
        det_d = pd.Series(per_article_loss_G(c, Gd), index=idx)
        # flat-level loss: candidate level vs flattened D40 level (same zero rule)
        flat_d = pd.Series(per_article_loss_G(c, flat_action(d40_level(c), c)), index=idx)
        common = det_c.index[det_c.index.isin(flat_d.index)]
        valid_idx = idx[c["valid"]]
        losses[k] = {"candidate": det_c.loc[valid_idx].mean(), "flat_d40": flat_d.loc[valid_idx].mean()}
        d = (det_c - det_d).loc[valid_idx]
        diffs[k] = d
        diffs_17[k] = (pd.Series(per_article_loss_G(c, Gc, np.arange(0, 7)), index=idx)
                       - pd.Series(per_article_loss_G(c, Gd, np.arange(0, 7)), index=idx)).loc[valid_idx]
        diffs_813[k] = (pd.Series(per_article_loss_G(c, Gc, np.arange(7, H)), index=idx)
                        - pd.Series(per_article_loss_G(c, Gd, np.arange(7, H)), index=idx)).loc[valid_idx]
        sc = score_G(c, Gc)
        sd = score_G(c, Gd)
        rows.append({"section": "pooled_level", "cut": k,
                     "candidate": sc, "d40": sd, "gain": sc - sd,
                     "level_loss_candidate": losses[k]["candidate"],
                     "level_loss_flat_d40": losses[k]["flat_d40"],
                     "level_gain": 100 * (losses[k]["flat_d40"] - losses[k]["candidate"]),
                     "w1_7_candidate": score_G(c, Gc, np.arange(0, 7)),
                     "w1_7_d40": score_G(c, Gd, np.arange(0, 7)),
                     "w8_13_candidate": score_G(c, Gc, np.arange(7, H)),
                     "w8_13_d40": score_G(c, Gd, np.arange(7, H)),
                     "best_iter": fit["best_iter"], "n_train_windows": fit["n_train_windows"],
                     "seconds": round(time.time() - t0)})
        print(f"cut {k}: candidate {sc:7.2f}  D40 {sd:7.2f}  gain {sc - sd:+6.2f}  "
              f"best_iter {fit['best_iter']}  ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res["section"] = "pooled_level"
    gain = float(res["gain"].mean())
    wins = int((res["gain"] > 0).sum())
    w17 = float((res["w1_7_candidate"] - res["w1_7_d40"]).mean())
    w813 = float((res["w8_13_candidate"] - res["w8_13_d40"]).mean())
    obs, lo, hi = paired_interval(diffs)
    obs17, lo17, hi17 = paired_interval(diffs_17)
    obs813, lo813, hi813 = paired_interval(diffs_813)
    summary = pd.DataFrame([
        {"section": "summary", "metric": "mean_gain", "value": gain},
        {"section": "summary", "metric": "cuts_won", "value": wins},
        {"section": "summary", "metric": "paired_interval_lo", "value": lo},
        {"section": "summary", "metric": "paired_interval_hi", "value": hi},
        {"section": "summary", "metric": "w1_7_mean_gain", "value": w17},
        {"section": "summary", "metric": "w8_13_mean_gain", "value": w813},
        {"section": "summary", "metric": "w1_7_interval_lo", "value": lo17},
        {"section": "summary", "metric": "w1_7_interval_hi", "value": hi17},
        {"section": "summary", "metric": "w8_13_interval_lo", "value": lo813},
        {"section": "summary", "metric": "w8_13_interval_hi", "value": hi813},
        {"section": "summary", "metric": "level_gain_mean", "value": float(res["level_gain"].mean())},
        {"section": "summary", "metric": "adopt", "value": float(gain >= 0.5 and wins >= 3 and w17 >= 0 and w813 >= 0)},
    ])
    out = pd.concat([res, summary], ignore_index=True, sort=False)
    path = OUT_DIR / "results_kimi_pooled-level.csv"
    out.to_csv(path, index=False)
    print(f"\nmean gain {gain:+.2f}, cuts won {wins}/4, paired 95% [{lo:+.2f}, {hi:+.2f}]")
    print(f"w1-7 gain {w17:+.2f} [{lo17:+.2f}, {hi17:+.2f}], w8-13 gain {w813:+.2f} [{lo813:+.2f}, {hi813:+.2f}]")
    print(f"flat-level gain (candidate - flat D40 level) {res['level_gain'].mean():+.2f}")
    print("verdict:", "ADOPT" if (gain >= 0.5 and wins >= 3 and w17 >= 0 and w813 >= 0)
          else "keep D40 (fails the +0.5 / 3-of-4 / both-horizon bar)")
    print(f"wrote {path}")
    return out


# ------------------------------------------------------------------ audit driver
def cmd_audit(args):
    set_models(args.models)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cuts = {W: overlapping_context(W) for W in OFFSETS}
    for W in cuts:
        cuts[W]["W"] = W
    canon = canonical_contexts()
    canon_scores = {k: score_G(c, canonical_d40_G(c)) for k, c in canon.items()}
    print("Step 0: canonical D40 by cut: " +
          " ".join(f"{x:7.2f}" for x in canon_scores.values()) +
          f"   mean {np.mean(list(canon_scores.values())):7.2f}", flush=True)
    ref = pd.DataFrame([{"section": "canonical", "cut": k, "d40": x}
                        for k, x in canon_scores.items()])

    orc = audit_oracles(cuts)
    sel = run_selector(cuts)
    d1 = audit_d1(canon)
    d5 = audit_d5(cuts[40])

    # D4: w1-7 vs w8-13 on the standard cuts
    d4 = []
    for k, c in canon.items():
        G = canonical_d40_G(c)
        d4.append({"section": "D4", "cut": k,
                   "w1_7": score_G(c, G, np.arange(0, 7)),
                   "w8_13": score_G(c, G, np.arange(7, H))})
    d4 = pd.DataFrame(d4)

    # oracle null on the 13 cached cuts (O_sel11), 200 permutations
    null_rows = []
    for W, c in cuts.items():
        lev = candidate_levels(c)
        obs = oracle_select(c, lev, CANDIDATES, True)["score"]
        sims = oracle_null(c, lev, CANDIDATES, n_rep=args.reps, seed=W)
        null_rows.append({"section": "oracle_null", "cut": W, "observed": obs,
                          "null_mean": float(sims.mean()),
                          "null_p2_5": float(np.percentile(sims, 2.5)),
                          "null_p97_5": float(np.percentile(sims, 97.5)),
                          "p_value": float(np.mean(sims >= obs))})
    null = pd.DataFrame(null_rows)

    long = []
    for name, df in [("canonical", ref), ("oracle", orc), ("D1", d1), ("D5", d5),
                     ("D4", d4), ("null", null), ("A", sel)]:
        if len(df) == 0:
            continue
        tmp = df.copy()
        tmp["block"] = name
        long.append(tmp)
    long = pd.concat(long, ignore_index=True, sort=False)
    path = OUT_DIR / "results_kimi_audit.csv"
    long.to_csv(path, index=False)

    # readable summary
    o = orc[orc["scope"] == "all13"]
    print("\nOverlapping cuts (13): means")
    for col in ["d40", "flat_mean", "flat_median_d40", "oracle_level_d40shape",
                "oracle_flat_median", "oracle_flat_median_zero",
                "o_sel2_zero", "o_sel11_zero", "o_sel2_nozero", "o_sel11_nozero"]:
        print(f"  {col:26s} {o[col].mean():7.2f}")
    print(f"  O_sel11 - D40 mean gain   {o['o_sel11_zero'].mean() - o['d40'].mean():+7.2f}")
    print(f"  O_sel2  - D40 mean gain   {o['o_sel2_zero'].mean() - o['d40'].mean():+7.2f}")
    print(f"  flat-mean max |change|    {(o['flat_mean'] - o['d40']).abs().max():.2f}")
    print(f"  flat-median max |change|  {(o['flat_median_d40'] - o['d40']).abs().max():.2f}")
    oh = orc[orc["scope"] != "all13"].pivot_table(index="scope", values=["d40", "o_sel11_zero"])
    print("\nHorizon split (mean over 13 cuts):")
    for scope in ["w1_7", "w8_13"]:
        sub = orc[orc["scope"] == scope]
        print(f"  {scope:5s} D40 {sub['d40'].mean():7.2f}  O_sel11 {sub['o_sel11_zero'].mean():7.2f}")
    print("\nOracle null (O_sel11):")
    print(f"  observed mean {null['observed'].mean():7.2f}; null mean {null['null_mean'].mean():7.2f}; "
          f"observed inside null interval on {int(((null['observed'] >= null['null_p2_5']) & (null['observed'] <= null['null_p97_5'])).sum())}/{len(null)} cuts")
    print("\nD5 at W=40:")
    if len(d5):
        for _, r in d5.iterrows():
            print(f"  {r['metric']:26s} {r['value']:.4f}")
    print("\nD1 tail concentration and instability:")
    sh = d1[d1["section"] == "D1_share"]
    for m in sorted(sh["metric"].unique()):
        vals = sh[sh["metric"] == m]["value"]
        print(f"  {m:22s} {vals.min()*100:5.1f}% .. {vals.max()*100:5.1f}%")
    ov = d1[d1["section"] == "D1_overlap"]
    for m in sorted(ov["metric"].unique()):
        if m == "n_common_scored":
            continue
        vals = ov[ov["metric"] == m]["value"]
        print(f"  {m:28s} {vals.min()*100:5.1f}% .. {vals.max()*100:5.1f}%")
    print("\nA selector (diagnostic only):")
    a = sel[sel["section"] == "A_selector"]
    if len(a):
        print(a[["cut", "n_train_cuts", "feature", "threshold", "gain",
                 "switched_to_mean26", "gain_previous", "gain_o_sel2", "gain_o_sel11",
                 "C2", "C11"]].to_string(index=False))
        sh = sel[sel["section"] == "A_selector_shuffle"]
        if len(sh):
            print("\n  shuffle control (training features permuted within strata):")
            print(sh[["cut", "gain_mean", "gain_p2_5", "gain_p97_5", "real_gain", "p_value"]].to_string(index=False))
    print(f"\nwrote {path}  ({time.time() - t0:.0f}s)")
    return long


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="Step 0 reproduction + A selector diagnostic")
    a.add_argument("--models", default=",".join(MODELS))
    a.add_argument("--reps", type=int, default=200)
    a.set_defaults(func=cmd_audit)
    b = sub.add_parser("pooled-level", help="angle B: horizon-pooled L1 level model")
    b.add_argument("--models", default=",".join(MODELS))
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_pooled_level)
    args = ap.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.func(args)


if __name__ == "__main__":
    main()
