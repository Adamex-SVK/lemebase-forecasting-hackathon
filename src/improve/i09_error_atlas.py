"""Iteration 09, task 1: error and headroom atlas for D40.

Reproduces the unrounded D40 matrices exactly as `src/meta/ensemble_recipe.py` builds them, then:
  * decomposes the scaled-error contribution (MASE, not raw MAE) by cut, horizon week, scale
    quartile, zero share, weeks since sale, forecast level and calendar week;
  * reports the share of excess error carried by the top 1%, 5% and 10% of articles;
  * compares D40 with its members, median13, zero, NHITS, Sundial, q40 variants and last52;
  * computes the non-adoptable per-article oracle and tests whether a pre-origin selector can
    capture it with the purged protocol (donors only where the whole 13-week holdout ends before
    the target origin, i.e. donor offset W >= target offset N + 13);
  * checks whether padded and last52 D40 errors are complementary.

Output: iterations/09-forecast-approaches/results_adamex-svk_error-atlas.csv
Run:    .venv/bin/python src/improve/i09_error_atlas.py
"""
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import features as f  # noqa: E402
import validate as v  # noqa: E402

warnings.filterwarnings("ignore")

H = f.H
OUT = ROOT / "iterations" / "09-forecast-approaches"
FC = ROOT / "iterations" / "03-foundation-model" / "forecasts"
FCC = ROOT / "iterations" / "03-foundation-model" / "forecasts_cuts"
BASE = ROOT / "iterations" / "05-meta-ensemble" / "base_forecasts"
N_CUTS = 4
OFFSETS = list(range(0, 49, 4))
REF_MODELS = ["moirai-2.0-small", "tirex", "chronos-t5-small"]
REF_W = 0.3
Q40 = {"moirai-q40": "moirai-2.0-small-q40", "tirex-q40": "tirex-q40", "chronos-q40": "chronos-t5-small-q40"}


# ------------------------------------------------------------------ matrices
def matrix_from_parquet(path, M):
    fc = pd.read_parquet(path)
    P = fc.pivot_table(index=v.KEYS, columns="week", values="forecast", aggfunc="sum")
    return P.reindex(pd.MultiIndex.from_frame(M.keys)).fillna(0.0).to_numpy(float)


def actual_matrix(hold, M, weeks):
    P = hold.pivot_table(index=v.KEYS, columns="week", values="units", aggfunc="sum")
    return P.reindex(pd.MultiIndex.from_frame(M.keys)).reindex(columns=list(weeks)).fillna(0.0).to_numpy(float)


def dead_mask(train, M):
    Wd = train.pivot_table(index=v.KEYS, columns="week", values="units", aggfunc="sum", fill_value=0.0)
    dead = (Wd.iloc[:, -8:].sum(axis=1) == 0).reindex(pd.MultiIndex.from_frame(M.keys)).fillna(True).to_numpy()
    return dead


def scale_vector(train, M):
    sc = v.history_scale(train, "full").reindex(pd.MultiIndex.from_frame(M.keys))
    return sc.to_numpy(float)


def mase_matrix(F, A, scale, dead=None, zero8=True, rounding=True):
    F = np.asarray(F, float).copy()
    if zero8 and dead is not None:
        F[dead] = 0.0
    if rounding:
        F = np.round(F)
    F = np.clip(F, 0.0, None)
    err = np.abs(A - F).mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return err / scale


def score_of(mase, ok):
    return float((1.0 - np.nanmean(mase[ok])) * 100.0)


# ------------------------------------------------------------------ contexts
def standard_ctx(k):
    full = v.load_weekly("test")
    df = full[full["week"] <= full["week"].max() - pd.Timedelta(weeks=13 * k)]
    train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut])
    weeks = pd.date_range(cut, periods=H, freq="7D")
    L = np.load(ROOT / "iterations" / "02-robust-eval" / "forecasts" / f"base_cut{k}_seed0.npy")
    return dict(k=k, full=full, df=df, train=train, hold=hold, cut=cut, M=M, weeks=weeks, L=L)


def overlap_ctx(w):
    full = v.load_weekly("test")
    df = full[full["week"] <= full["week"].max() - pd.Timedelta(weeks=w)]
    train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut])
    weeks = pd.date_range(cut, periods=H, freq="7D")
    L = np.load(BASE / f"lgbm_w{w}.npy")
    return dict(w=w, full=full, df=df, train=train, hold=hold, cut=cut, M=M, weeks=weeks, L=L)


def standard_candidates(ctx):
    M, L = ctx["M"], ctx["L"]
    dead = dead_mask(ctx["train"], M)
    raw = {m: matrix_from_parquet(FC / f"{m}__test__cut{ctx['k']}__padded.parquet", M) for m in REF_MODELS}
    E = np.mean([raw[m] for m in REF_MODELS], axis=0)
    E[dead] = 0.0
    cand = {"D40": REF_W * L + (1 - REF_W) * E}
    for m in REF_MODELS:
        cand[m] = raw[m]
    for name, m in Q40.items():
        cand[name] = matrix_from_parquet(FC / f"{m}__test__cut{ctx['k']}__padded.parquet", M)
    # NHITS forecasts live in another session's untracked iteration-06 folder; include them when
    # present, but keep the atlas runnable on a fresh checkout without that folder.
    nhits = ROOT / "iterations" / "06-neural" / "forecasts" / f"NHITS__test__cut{ctx['k']}__padded.parquet"
    if nhits.exists():
        cand["NHITS"] = matrix_from_parquet(nhits, M)
    cand["Sundial"] = matrix_from_parquet(FC / f"sundial-base-128m__test__cut{ctx['k']}__padded.parquet", M)
    # last52 D40 and its members
    raw52 = {m: matrix_from_parquet(FC / f"{m}__test__cut{ctx['k']}__last52.parquet", M) for m in REF_MODELS}
    E52 = np.mean([raw52[m] for m in REF_MODELS], axis=0)
    E52[dead] = 0.0
    cand["D40-last52"] = REF_W * L + (1 - REF_W) * E52
    for m in REF_MODELS:
        cand[f"{m}-last52"] = raw52[m]
    med = np.median(ctx["M"].U[:, -13:], axis=1).astype(float)
    cand["median13"] = np.repeat(med[:, None], H, axis=1)
    cand["zero"] = np.zeros_like(E)
    return cand, dead


def overlap_candidates(ctx):
    M, L = ctx["M"], ctx["L"]
    dead = dead_mask(ctx["train"], M)
    raw = {m: matrix_from_parquet(FCC / f"{m}__test__w{ctx['w']}__padded.parquet", M) for m in REF_MODELS}
    E = np.mean([raw[m] for m in REF_MODELS], axis=0)
    E[dead] = 0.0
    cand = {"D40": REF_W * L + (1 - REF_W) * E}
    cand.update(raw)
    cand["median13"] = np.load(BASE / f"median13_w{ctx['w']}.npy")
    cand["zero"] = np.zeros_like(E)
    return cand, dead


def per_article_tables(ctx, cand, dead, zero8_models=()):
    A = actual_matrix(ctx["hold"], ctx["M"], ctx["weeks"])
    scale = scale_vector(ctx["train"], ctx["M"])
    ok = np.isfinite(scale) & (scale > 0)
    mases = {}
    for name, F in cand.items():
        # D40 / D40-last52 already apply the zero rule inside the ensemble before blending,
        # so they must not be zeroed a second time after the LightGBM mix.
        z8 = name not in ("D40", "D40-last52", "median13", "zero")
        mases[name] = mase_matrix(F, A, scale, dead if z8 else None, zero8=z8, rounding=True)
    return A, scale, ok, mases


# ------------------------------------------------------------------ decomposition
def decompose(ctx, F, A, scale, ok):
    """Tidy scaled-error contribution for the rounded D40 forecast of one cut."""
    M = ctx["M"]
    keys = pd.MultiIndex.from_frame(M.keys)
    Fr = np.clip(np.round(F), 0, None)
    err = np.abs(A - Fr)
    U = M.U
    zshare = (U[:, -26:] == 0).mean(axis=1)
    rev = U[:, ::-1] > 0
    any_sale = rev.any(axis=1)
    wss = np.where(any_sale, np.minimum(rev.argmax(axis=1), 104), 104)
    level = Fr.mean(axis=1)
    rows = []
    base = {"analysis": "decomposition", "cut": ctx["k"], "cut_kind": "standard", "first_holdout_week": str(ctx["weeks"][0].date())}
    # per-article contributions (sum over the 13 weeks)
    per_article = err.sum(axis=1) / scale
    total = per_article[ok].sum()
    arts = pd.DataFrame({
        "scale": scale[ok], "mase": per_article[ok] / 13.0, "sum_scaled_err": per_article[ok],
        "zshare26": zshare[ok], "weeks_since_sale": wss[ok], "level": level[ok],
        "key0": keys.get_level_values(0)[ok], "key1": keys.get_level_values(1)[ok],
    })
    for gt, col, bins, labels in [
        ("scale_quartile", "scale", None, ["Q1 smallest", "Q2", "Q3", "Q4 largest"]),
        ("zero_share26", "zshare26", [-1e-9, 1e-9, 0.25, 0.5, 0.75, 1.0001], ["0", "(0,0.25]", "(0.25,0.5]", "(0.5,0.75]", "(0.75,1]"]),
        ("weeks_since_sale", "weeks_since_sale", [-0.1, 0.1, 4.1, 8.1, 13.1, 26.1, 200.1],
         ["0", "1-4", "5-8", "9-13", "14-26", ">26"]),
        ("forecast_level", "level", [-0.1, 0.1, 1.1, 3.1, 10.1, 1e9], ["0", "(0,1]", "(1,3]", "(3,10]", ">10"]),
    ]:
        if gt == "scale_quartile":
            g = pd.qcut(arts[col].rank(method="first"), 4, labels=labels)
        else:
            g = pd.cut(arts[col], bins=bins, labels=labels, include_lowest=True)
        for name, sub in arts.groupby(g, observed=True):
            rows.append({**base, "group_type": gt, "group": str(name), "n": int(len(sub)),
                         "sum_scaled_err": float(sub["sum_scaled_err"].sum()),
                         "share_of_total": float(sub["sum_scaled_err"].sum() / total),
                         "mean_mase": float(sub["mase"].mean()),
                         "score_equiv": float((1 - sub["mase"].mean()) * 100)})
    # horizon week
    art_ok_keys = [k for k, o in zip(keys, ok) if o]
    for h in range(1, H + 1):
        contrib = err[:, h - 1] / scale
        rows.append({**base, "group_type": "horizon_week", "group": str(h), "n": int(ok.sum()),
                     "sum_scaled_err": float(contrib[ok].sum()),
                     "share_of_total": float(contrib[ok].sum() / total),
                     "mean_mase": np.nan, "score_equiv": np.nan})
    # calendar week of the holdout
    for i, wk in enumerate(ctx["weeks"]):
        contrib = err[:, i] / scale
        rows.append({**base, "group_type": "calendar_week", "group": str(wk.date()), "n": int(ok.sum()),
                     "sum_scaled_err": float(contrib[ok].sum()),
                     "share_of_total": float(contrib[ok].sum() / total),
                     "mean_mase": np.nan, "score_equiv": np.nan})
    return rows, per_article, ok


def top_share_rows(ctx, mase, ok):
    total = mase[ok].sum()
    order = np.argsort(mase[ok])[::-1]
    n = ok.sum()
    rows = []
    for pct in (0.01, 0.05, 0.10):
        k = max(1, int(round(pct * n)))
        rows.append({"analysis": "top_share", "cut": ctx["k"], "cut_kind": "standard",
                     "group_type": f"top_{int(pct*100)}pct", "group": "articles", "n": k,
                     "sum_scaled_err": float(mase[ok][order[:k]].sum()),
                     "share_of_total": float(mase[ok][order[:k]].sum() / total),
                     "mean_mase": float(mase[ok][order[:k]].mean()), "score_equiv": np.nan})
    return rows


# ------------------------------------------------------------------ oracle and selector
def oracle_rows(mases, ok, label, cut_kind, cut):
    Mx = np.column_stack([mases[n] for n in mases])[ok, :]   # ok articles x models
    best = Mx.min(axis=1)
    d40 = mases["D40"][ok]
    rng = np.random.default_rng(0)

    def interval(diff):
        bs = np.array([-100 * rng.choice(diff, len(diff)).mean() for _ in range(500)])
        return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    rows = []
    for name, j in zip(mases, range(len(mases))):
        lo, hi = interval(Mx[:, j] - d40)
        rows.append({"analysis": "candidate_score", "cut": cut, "cut_kind": cut_kind,
                     "group_type": "model", "group": name, "n": int(ok.sum()), "sum_scaled_err": np.nan,
                     "share_of_total": np.nan, "mean_mase": float(np.mean(Mx[:, j])),
                     "score_equiv": float((1 - np.mean(Mx[:, j])) * 100),
                     "gain_vs_d40": float((np.mean(d40) - np.mean(Mx[:, j])) * 100),
                     "note": f"interval=[{lo:+.2f},{hi:+.2f}]"})
    lo, hi = interval(best - d40)
    rows.append({"analysis": "oracle", "cut": cut, "cut_kind": cut_kind, "group_type": label,
                 "group": "per-article best model", "n": int(ok.sum()), "sum_scaled_err": np.nan,
                 "share_of_total": np.nan, "mean_mase": float(np.mean(best)),
                 "score_equiv": float((1 - np.mean(best)) * 100),
                 "gain_vs_d40": float((np.mean(d40) - np.mean(best)) * 100),
                 "note": f"non-adoptable oracle; interval=[{lo:+.2f},{hi:+.2f}]"})
    return rows, float((np.mean(d40) - np.mean(best)) * 100)


def purged_donor_offsets(target, offsets=OFFSETS, horizon=H):
    """Offsets whose whole holdout ends before the target origin: W >= target + horizon."""
    return [w for w in offsets if w >= target + horizon]


def model_features(M):
    X, scale, valid = f.features_at(M, M.W)
    return X


def selector_eval(overlap_data, target_w, names):
    """Purged pre-origin selector: train on donors W >= target+13, test at target."""
    donors = purged_donor_offsets(target_w)
    if len(donors) < 2:
        return None
    Xs, ys = [], []
    for w in donors:
        d = overlap_data[w]
        X = model_features(d["ctx"]["M"])
        lab = np.argmin(np.column_stack([d["mases"][n] for n in names]), axis=1)
        Xs.append(X[d["ok"]])
        ys.append(lab[d["ok"]])
    Xtr = np.vstack(Xs)
    ytr = np.concatenate(ys)
    nclass = len(names)
    params = dict(objective="multiclass", num_class=nclass, learning_rate=0.05, num_leaves=15,
                  min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=5.0, verbose=-1, seed=0, num_threads=8)
    dtr = lgb.Dataset(Xtr, ytr, feature_name=list(f.FEATURES[:-1]))
    booster = lgb.train(params, dtr, num_boost_round=300)
    d = overlap_data[target_w]
    Xt = model_features(d["ctx"]["M"])[d["ok"]]
    pred = booster.predict(Xt).argmax(axis=1)
    Mx = np.column_stack([d["mases"][n] for n in names])[d["ok"]]
    chosen = Mx[np.arange(len(pred)), pred]
    d40 = d["mases"]["D40"][d["ok"]]
    acc = float((pred == np.argmin(Mx, axis=1)).mean())
    maj = float(pd.Series(np.argmin(Mx, axis=1)).value_counts(normalize=True).iloc[0])
    rng = np.random.default_rng(0)
    diff = chosen - d40
    bs = np.array([-100 * rng.choice(diff, len(diff)).mean() for _ in range(1000)])
    return {"w": target_w, "n_donors": len(donors), "donor_offsets": ",".join(map(str, donors)),
            "selector_score": float((1 - chosen.mean()) * 100), "d40_score": float((1 - d40.mean()) * 100),
            "gain": float((d40.mean() - chosen.mean()) * 100), "accuracy": acc, "majority": maj,
            "oracle_gain": float((d40.mean() - Mx.min(axis=1).mean()) * 100),
            "gain_lo": float(np.percentile(bs, 2.5)), "gain_hi": float(np.percentile(bs, 97.5)),
            "chosen_mase": chosen, "d40_mase": d40}


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    std = {}
    oracle_gains = []
    print("standard cuts: loading and scoring candidates", flush=True)
    for k in range(N_CUTS):
        ctx = standard_ctx(k)
        cand, dead = standard_candidates(ctx)
        A, scale, ok, mases = per_article_tables(ctx, cand, dead)
        std[k] = dict(ctx=ctx, cand=cand, dead=dead, A=A, scale=scale, ok=ok, mases=mases)
        d40 = mases["D40"]
        rows.append({"analysis": "by_cut", "cut": k, "cut_kind": "standard", "group_type": "all",
                     "group": "D40", "n": int(ok.sum()), "sum_scaled_err": float(d40[ok].sum()),
                     "share_of_total": 1.0, "mean_mase": float(d40[ok].mean()),
                     "score_equiv": score_of(d40, ok)})
        dec, per_article, _ = decompose(ctx, cand["D40"], A, scale, ok)
        rows += dec
        rows += top_share_rows(ctx, d40, ok)
        orows, og = oracle_rows(mases, ok, "standard_all", "standard", k)
        rows += orows
        oracle_gains.append(og)
        print(f"  cut {k}: D40 {score_of(d40, ok):7.2f}  oracle gain {og:+.2f}", flush=True)

    print("overlapping cuts: loading and scoring D40 + members + median13 + zero", flush=True)
    ov = {}
    ov_oracle = []
    for w in OFFSETS:
        ctx = overlap_ctx(w)
        cand, dead = overlap_candidates(ctx)
        A, scale, ok, mases = per_article_tables(ctx, cand, dead)
        ov[w] = dict(ctx=ctx, cand=cand, dead=dead, A=A, scale=scale, ok=ok, mases=mases)
        orows, og = oracle_rows(mases, ok, "overlap_core", "overlap", w)
        rows += orows
        ov_oracle.append(og)
        print(f"  w{w:02d}: D40 {score_of(mases['D40'], ok):7.2f}  oracle gain {og:+.2f}", flush=True)

    # selector with purged donors; targets need W >= N+13 donors
    names = list(ov[0]["mases"].keys())
    sel = []
    for w in OFFSETS:
        r = selector_eval(ov, w, names)
        if r is not None:
            sel.append(r)
            rows.append({"analysis": "selector", "cut": w, "cut_kind": "overlap",
                         "group_type": "purged_pre_origin", "group": f"donors={r['donor_offsets']}",
                         "n": int(ov[w]["ok"].sum()), "sum_scaled_err": np.nan, "share_of_total": np.nan,
                         "mean_mase": np.nan, "score_equiv": r["selector_score"],
                         "gain_vs_d40": r["gain"],
                         "note": f"d40={r['d40_score']:.2f};acc={r['accuracy']:.3f};maj={r['majority']:.3f};"
                                 f"interval=[{r['gain_lo']:+.2f},{r['gain_hi']:+.2f}]"})
    sel_df = pd.DataFrame(sel)
    if len(sel_df):
        wins = int((sel_df["gain"] > 0).sum())
        pooled_chosen = np.concatenate([r["chosen_mase"] for r in sel])
        pooled_d40 = np.concatenate([r["d40_mase"] for r in sel])
        rng = np.random.default_rng(0)
        pbs = np.array([-100 * rng.choice(pooled_chosen - pooled_d40, len(pooled_chosen)).mean() for _ in range(1000)])
        rows.append({"analysis": "selector_summary", "cut": "mean", "cut_kind": "overlap",
                     "group_type": "purged_pre_origin", "group": "mean over purged targets",
                     "n": len(sel_df), "sum_scaled_err": np.nan, "share_of_total": np.nan,
                     "mean_mase": np.nan, "score_equiv": float(sel_df["selector_score"].mean()),
                     "gain_vs_d40": float(sel_df["gain"].mean()),
                     "note": f"wins={wins}/{len(sel_df)};accuracy={sel_df['accuracy'].mean():.3f};majority={sel_df['majority'].mean():.3f};"
                             f"pooled_interval=[{np.percentile(pbs,2.5):+.2f},{np.percentile(pbs,97.5):+.2f}]"})
        print(f"\nselector (purged): mean gain {sel_df['gain'].mean():+.2f}, wins {wins}/{len(sel_df)}", flush=True)

    # persistence: adjacent overlapping cuts and target vs older donors
    print("persistence of per-article error", flush=True)
    pers = []
    for i in range(1, len(OFFSETS)):
        a, b = OFFSETS[i], OFFSETS[i - 1]
        common = ov[a]["ok"] & ov[b]["ok"]
        x = ov[a]["mases"]["D40"][common]
        y = ov[b]["mases"]["D40"][common]
        rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
        ka = max(1, int(round(0.1 * common.sum())))
        top_a = set(np.argsort(x)[::-1][:ka])
        top_b = set(np.argsort(y)[::-1][:ka])
        overlap = len(top_a & top_b) / ka
        pers.append({"cut": a, "prev": b, "rho": rho, "decile_overlap": overlap})
        rows.append({"analysis": "persistence", "cut": a, "cut_kind": "overlap",
                     "group_type": "adjacent_cuts", "group": f"w{a} vs w{b}", "n": int(common.sum()),
                     "sum_scaled_err": np.nan, "share_of_total": np.nan, "mean_mase": np.nan,
                     "score_equiv": np.nan, "gain_vs_d40": np.nan,
                     "note": f"spearman={rho:+.3f};top_decile_overlap={overlap:.3f}"})
    pers_df = pd.DataFrame(pers)
    if len(pers_df):
        rows.append({"analysis": "persistence_summary", "cut": "mean", "cut_kind": "overlap",
                     "group_type": "adjacent_cuts", "group": "mean", "n": len(pers_df),
                     "sum_scaled_err": np.nan, "share_of_total": np.nan, "mean_mase": np.nan,
                     "score_equiv": np.nan, "gain_vs_d40": np.nan,
                     "note": f"spearman={pers_df['rho'].mean():+.3f};top_decile_overlap={pers_df['decile_overlap'].mean():.3f}"})

    print("padded vs last52 complementarity", flush=True)
    comp = []
    for k in range(N_CUTS):
        d = std[k]
        a = d["mases"]["D40"]
        b = d["mases"]["D40-last52"]
        common = d["ok"]
        x, y = a[common], b[common]
        rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
        frac52 = float(np.mean(y < x))
        gain52 = float((x.mean() - y.mean()) * 100)
        comp.append({"cut": k, "rho": rho, "frac_last52_better": frac52, "gain_last52": gain52})
        rows.append({"analysis": "padded_vs_last52", "cut": k, "cut_kind": "standard",
                     "group_type": "per_article_mase", "group": "D40 padded vs D40 last52",
                     "n": int(common.sum()), "sum_scaled_err": np.nan, "share_of_total": np.nan,
                     "mean_mase": np.nan, "score_equiv": np.nan, "gain_vs_d40": gain52,
                     "note": f"spearman={rho:+.3f};frac_last52_better={frac52:.3f}"})
    comp_df = pd.DataFrame(comp)

    # gate summary
    oracle_mean = float(np.mean(ov_oracle))
    oracle_std_mean = float(np.mean(oracle_gains))
    sel_wins = int((sel_df["gain"] > 0).sum()) if len(sel_df) else 0
    gate_open = bool(oracle_mean >= 1.0 and sel_wins >= 3)
    rows.append({"analysis": "gate", "cut": "mean", "cut_kind": "overlap",
                 "group_type": "dynamic_blending", "group": "gate_open" if gate_open else "gate_closed",
                 "n": len(sel_df), "sum_scaled_err": np.nan, "share_of_total": np.nan,
                 "mean_mase": np.nan, "score_equiv": np.nan, "gain_vs_d40": oracle_mean,
                 "note": f"oracle_overlap={oracle_mean:+.2f};oracle_standard={oracle_std_mean:+.2f};selector_wins={sel_wins}/{len(sel_df)};"
                         f"selector_mean_gain={(sel_df['gain'].mean() if len(sel_df) else float('nan')):+.2f}"})

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results_adamex-svk_error-atlas.csv", index=False)
    print(f"\nwrote {OUT / 'results_adamex-svk_error-atlas.csv'} ({len(res)} rows)")
    print(f"oracle headroom: overlap mean {oracle_mean:+.2f}, standard mean {oracle_std_mean:+.2f}")
    print(f"selector wins {sel_wins}/{len(sel_df)}; gate {'OPEN' if gate_open else 'CLOSED'}")
    if len(comp_df):
        print(comp_df.round(3).to_string(index=False))
    if len(pers_df):
        print(pers_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
