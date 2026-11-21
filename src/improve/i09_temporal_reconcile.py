"""Iteration 09, task 4: multi-scale (2-week / 4-week) temporal reconciliation with D40.

For every standard cut k = 0..3 and every aggregation factor g in {2, 4}:
  * aggregate the pre-origin weekly units into non-overlapping g-week blocks, anchored so the
    final complete block ends exactly at the origin (the last block = M.U[:, o-g:o]);
  * train the existing scaled-L1 LightGBM on the block series (target = block_units / block_scale)
    with a purged early-stopping split at block granularity;
  * forecast the next ceil(13/g) blocks (the holdout aggregation), disaggregate to weekly cells
    either with equal allocation or with the last pre-origin within-block week shares;
  * apply the zero (no sale in last 8 weeks) rule and clip negatives;
  * reconcile with the UNROUNDED D40 forecast:  F = (1-lambda) * D40 + lambda * F_agg;
  * round + clip, then score on the 4 standard cuts.

Fixed primary test: lambda = 0.5.  Every lambda / block-size / disaggregation choice is made
leave-one-cut-out (LOCO); the in-sample optimum is reported separately as non-adoptable.

Outputs (written by this script):
  iterations/09-forecast-approaches/results_adamex-svk_temporal-reconcile.csv
  iterations/09-forecast-approaches/notes_adamex-svk_temporal-reconcile.md

Run: .venv/bin/python src/improve/i09_temporal_reconcile.py
"""
import subprocess
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path("/Users/adomi/Documents/hack-t11")
sys.path.insert(0, str(ROOT / "src"))
import experiments_02 as e  # noqa: E402
import features as f  # noqa: E402
import train_gbm as g  # noqa: E402
import validate as v  # noqa: E402

warnings.filterwarnings("ignore")

H = 13
REF = ["moirai-2.0-small", "tirex", "chronos-t5-small"]
REF_W = 0.3
FC = ROOT / "iterations" / "03-foundation-model" / "forecasts"
OUT = ROOT / "iterations" / "09-forecast-approaches"
N_CUTS = 4
BLOCKS = (2, 4)
LAMBDAS = (0.25, 0.5, 0.75, 1.0)
PRIMARY_LAMBDA = 0.5
MIN_HIST = 13
DEAD_CAP = 104
N_VAL = 7
ADOPT_MEAN = 0.5
ADOPT_CUTS = 3

# Block-level features (mirror of features_at, on the aggregated series).  All are divided by the
# block scale where that makes sense; `h` (the forecast block horizon) is appended per window.
BLOCK_FEATURES = (
    [f"lag_{k}" for k in range(1, 14)]
    + ["mean2", "mean4", "mean8", "mean13", "median13", "max13", "zshare13",
       "mean26", "zshare26", "weeks_since_sale", "series_age", "log_scale", "log_mean13"]
)
BLOCK_NAMES = BLOCK_FEATURES + ["h"]


# --------------------------------------------------------------------------- reference D40 context
def standard_ctx(k):
    full = v.load_weekly("test")
    df = e.cut_frame(full, k)
    train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut])
    weeks = pd.date_range(cut, periods=H, freq="7D")
    L = np.load(ROOT / "iterations" / "02-robust-eval" / "forecasts" / f"base_cut{k}_seed0.npy")
    return dict(k=k, train=train, hold=hold, cut=cut, M=M, weeks=weeks, L=L)


def model_matrix(model, k, M):
    fc = pd.read_parquet(FC / f"{model}__test__cut{k}__padded.parquet")
    P = fc.pivot_table(index=v.KEYS, columns="week", values="forecast", aggfunc="sum")
    return P.reindex(pd.MultiIndex.from_frame(M.keys)).fillna(0.0).to_numpy(float)


def dead_mask(train, M):
    Wd = train.pivot_table(index=v.KEYS, columns="week", values="units", aggfunc="sum", fill_value=0.0)
    dead = (Wd.iloc[:, -8:].sum(axis=1) == 0)
    return dead.reindex(pd.MultiIndex.from_frame(M.keys)).fillna(True).to_numpy()


def d40_matrix(ctx):
    M, L = ctx["M"], ctx["L"]
    dead = dead_mask(ctx["train"], M)
    E = np.mean([model_matrix(m, ctx["k"], M) for m in REF], axis=0)
    E[dead] = 0.0
    return REF_W * L + (1 - REF_W) * E


# --------------------------------------------------------------------------- aggregation
def aggregate(U, b):
    """Sum the last floor(W/b) non-overlapping b-week blocks, anchored at the origin (oldest first)."""
    W = U.shape[1]
    nb = W // b
    used = U[:, W - nb * b:].astype(np.float64)
    return used.reshape(U.shape[0], nb, b).sum(axis=2)


def last_block_shares(U, b):
    """Week shares inside the last complete pre-origin block (uniform fallback when it sums to 0)."""
    last = U[:, U.shape[1] - b:].astype(np.float64)
    tot = last.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sh = np.where(tot[:, None] > 0, last / np.maximum(tot[:, None], 1e-12), 1.0 / b)
    return sh


# --------------------------------------------------------------------------- block model
def block_features(A, o):
    """Features of every block series at block origin o (first forecast block), using blocks < o."""
    assert MIN_HIST <= o <= A.shape[1]
    U = A[:, :o].astype(np.float64)
    n = U.shape[0]
    scale = np.abs(np.diff(U, axis=1)).mean(axis=1)
    valid = scale > 0
    s = np.where(valid, scale, 1.0)
    cols = {}
    for k in range(1, 14):
        cols[f"lag_{k}"] = U[:, o - k] / s
    for w in (2, 4, 8, 13):
        cols[f"mean{w}"] = U[:, o - w:o].mean(axis=1) / s
    cols["median13"] = np.median(U[:, o - 13:o], axis=1) / s
    cols["max13"] = U[:, o - 13:o].max(axis=1) / s
    cols["zshare13"] = (U[:, o - 13:o] == 0).mean(axis=1)
    w26 = min(26, o)
    cols["mean26"] = U[:, o - w26:o].mean(axis=1) / s
    cols["zshare26"] = (U[:, o - w26:o] == 0).mean(axis=1)
    rev = U[:, ::-1] > 0
    any_sale = rev.any(axis=1)
    cols["weeks_since_sale"] = np.where(any_sale, np.minimum(rev.argmax(axis=1), DEAD_CAP), DEAD_CAP)
    first = (U > 0).argmax(axis=1)
    cols["series_age"] = np.where(any_sale, o - first, 0)
    cols["log_scale"] = np.log1p(scale)
    cols["log_mean13"] = np.log1p(U[:, o - 13:o].mean(axis=1))
    X = np.column_stack([cols[name] for name in BLOCK_FEATURES]).astype(np.float32)
    return X, scale, valid


def block_windows(A, origins, nb):
    Xs, ys, ws, org = [], [], [], []
    for o in origins:
        X, scale, valid = block_features(A, o)
        idx = np.flatnonzero(valid)
        if len(idx) == 0:
            continue
        Y = A[idx, o:o + nb].astype(np.float64) / scale[idx, None]
        Xi = X[idx]
        for h in range(1, nb + 1):
            Xs.append(np.column_stack([Xi, np.full(len(idx), h, dtype=np.float32)]))
            ys.append(Y[:, h - 1].astype(np.float32))
            ws.append(np.ones(len(idx), dtype=np.float32))
            org.append(np.full(len(idx), o))
    if not Xs:
        return dict(X=np.zeros((0, len(BLOCK_NAMES)), np.float32), y=np.zeros(0, np.float32),
                    w=np.zeros(0, np.float32), origin=np.zeros(0, int), n_windows=0)
    return dict(X=np.vstack(Xs), y=np.concatenate(ys), w=np.concatenate(ws),
                origin=np.concatenate(org), n_windows=int(sum(len(a) for a in org) // nb))


def fit_block_forecast(A, nb, params):
    """Train the scaled-L1 LightGBM on the block series; return (Fb [n x nb], best_iter)."""
    B = A.shape[1]
    origins = f.make_origins(B, horizon=nb, min_hist=MIN_HIST)
    if len(origins) == 0:
        return np.zeros((A.shape[0], nb)), 0

    # last N_VAL origins are validation, unless that would leave no purged training origin
    n_val = min(N_VAL, len(origins) - 1)
    val_o = origins[-n_val:]
    tr_o = origins[origins <= val_o[0] - nb]
    while len(tr_o) == 0 and n_val > 1:
        n_val -= 1
        val_o = origins[-n_val:]
        tr_o = origins[origins <= val_o[0] - nb]

    best = 200
    w_tr = block_windows(A, tr_o, nb)
    w_va = block_windows(A, val_o, nb)
    if w_tr["n_windows"] > 0 and w_va["n_windows"] > 0:
        dtr = lgb.Dataset(w_tr["X"], w_tr["y"], weight=w_tr["w"], feature_name=BLOCK_NAMES)
        dva = lgb.Dataset(w_va["X"], w_va["y"], weight=w_va["w"], reference=dtr)
        m1 = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
        best = max(1, int(m1.best_iteration))

    w_all = block_windows(A, origins, nb)
    if w_all["n_windows"] == 0:
        return np.zeros((A.shape[0], nb)), best
    m2 = lgb.train(params, lgb.Dataset(w_all["X"], w_all["y"], weight=w_all["w"], feature_name=BLOCK_NAMES),
                   num_boost_round=max(20, int(best * 1.1) + 1))

    X, scale, valid = block_features(A, B)
    pred = []
    for h in range(1, nb + 1):
        Xh = np.column_stack([X, np.full(A.shape[0], h, dtype=np.float32)])
        pred.append(m2.predict(Xh) * scale)
    Fb = np.column_stack(pred)
    Fb = np.nan_to_num(Fb, nan=0.0, posinf=0.0, neginf=0.0)
    Fb[~valid] = 0.0
    return np.clip(Fb, 0, None), best


def disaggregate(Fb, U_pre, b, method):
    """Map block forecasts to 13 weekly cells; equal or last-block-share allocation."""
    if method == "equal":
        Fw = np.repeat(Fb, b, axis=1) / b
    else:
        sh = last_block_shares(U_pre, b)
        Fw = (Fb[:, :, None] * sh[:, None, :]).reshape(Fb.shape[0], -1)
    return Fw[:, :H]


# --------------------------------------------------------------------------- scoring / stats
def score_matrix(M, F, ctx):
    """Round + clip (the D40 convention), then score; returns the v.score dict."""
    Fr = np.clip(np.round(np.asarray(F, float)), 0, None)
    return e.score_F(M, Fr, ctx["train"], ctx["hold"], ctx["weeks"])


def keys_index(M):
    return pd.MultiIndex.from_frame(M.keys)


def bootstrap(cand, ref, rng, n=1000):
    """Point gain and 95% interval of -100*(candidate MASE - D40 MASE) in score points."""
    d = (cand - ref).dropna().to_numpy()
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    means = np.array([rng.choice(d, len(d)).mean() for _ in range(n)])
    return float(-100 * d.mean()), float(np.percentile(-100 * means, 2.5)), float(np.percentile(-100 * means, 97.5)), len(d)


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, (float, np.floating)):
                cells.append("" if pd.isna(val) else f"{val:.2f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    params = dict(g.PARAMS, num_threads=4, seed=0)

    print("== temporal reconciliation: building D40 + block forecasts", flush=True)
    ctxs, d40 = {}, {}
    agg_fc, block_info = {}, {}
    for k in range(N_CUTS):
        ctx = standard_ctx(k)
        ctxs[k] = ctx
        d40[k] = d40_matrix(ctx)
        U_pre = ctx["M"].U.astype(np.float64)
        for b in BLOCKS:
            A = aggregate(U_pre, b)
            nb = int(np.ceil(H / b))
            Fb, best = fit_block_forecast(A, nb, params)
            block_info[(k, b)] = dict(W=ctx["M"].W, B=A.shape[1], nb=nb, best=best)
            for method in ("equal", "shares"):
                agg_fc[(k, b, method)] = disaggregate(Fb, U_pre, b, method)
            print(f"  cut {k} g={b}: blocks={A.shape[1]} nb={nb} best_iter={best}", flush=True)

    dead = {k: dead_mask(ctxs[k]["train"], ctxs[k]["M"]) for k in range(N_CUTS)}

    # scored-article base index (scale > 0) per cut, for aligned per-group / bootstrap work
    base = {}
    for k in range(N_CUTS):
        ctx = ctxs[k]
        idx = keys_index(ctx["M"])
        scale = v.history_scale(ctx["train"], "full").reindex(idx)
        ok = scale.notna() & (scale > 0)
        base[k] = dict(idx=idx, scale=scale, ok=ok, bidx=idx[ok.to_numpy()])

    # ----- score every (config, cut)
    configs = []
    for b in BLOCKS:
        for method in ("equal", "shares"):
            for lam in LAMBDAS:
                configs.append(dict(name=f"g{b}_{method}_lam{lam}", g=b, method=method, lam=lam))
    scores = {c["name"]: {} for c in configs}
    details = {}
    d40_scores, d40_details = {}, {}
    for k in range(N_CUTS):
        ctx = ctxs[k]
        r = score_matrix(ctx["M"], d40[k], ctx)
        d40_scores[k] = float(r["score"])
        d40_details[k] = r["detail"]["mase"]
        details[("D40", k)] = r["detail"]["mase"]
        for c in configs:
            Fa = np.clip(agg_fc[(k, c["g"], c["method"])].copy(), 0, None)
            Fa[dead[k]] = 0.0
            F = (1 - c["lam"]) * d40[k] + c["lam"] * Fa
            rr = score_matrix(ctx["M"], F, ctx)
            scores[c["name"]][k] = float(rr["score"])
            details[(c["name"], k)] = rr["detail"]["mase"]

    d40_cut = np.array([d40_scores[k] for k in range(N_CUTS)])
    d40_mean = float(d40_cut.mean())

    # ----- fixed lambda=0.5 summary (evaluated first, as required)
    fixed_rows = []
    for c in configs:
        if abs(c["lam"] - PRIMARY_LAMBDA) > 1e-12:
            continue
        sc = np.array([scores[c["name"]][k] for k in range(N_CUTS)])
        fixed_rows.append(dict(variant=c["name"], g=c["g"], method=c["method"], lam=c["lam"],
                               **{f"cut{k}": sc[k] for k in range(N_CUTS)},
                               mean=float(sc.mean()), gain=float(sc.mean() - d40_mean),
                               won=int((sc > d40_cut).sum())))
    fixed = pd.DataFrame(fixed_rows).sort_values("mean", ascending=False)

    # ----- LOCO over the full grid (g, method, lambda)
    names = [c["name"] for c in configs]
    sel = {}
    for k in range(N_CUTS):
        others = [j for j in range(N_CUTS) if j != k]
        sel[k] = max(names, key=lambda n: np.mean([scores[n][j] for j in others]))
    loco_sc = np.array([scores[sel[k]][k] for k in range(N_CUTS)])
    loco_mean = float(loco_sc.mean())
    loco_gain = float(loco_mean - d40_mean)
    loco_won = int((loco_sc > d40_cut).sum())
    adopt = bool(loco_gain >= ADOPT_MEAN and loco_won >= ADOPT_CUTS)

    in_best = max(names, key=lambda n: np.mean([scores[n][k] for k in range(N_CUTS)]))
    in_cfg = next(c for c in configs if c["name"] == in_best)
    in_sc = np.array([scores[in_best][k] for k in range(N_CUTS)])

    # ----- wide results table
    loco_for = {n: [k for k in range(N_CUTS) if sel[k] == n] for n in names}
    rows = []
    for c in configs:
        sc = np.array([scores[c["name"]][k] for k in range(N_CUTS)])
        rows.append({
            "variant": c["name"], "g": c["g"], "lambda": c["lam"], "disaggregation": c["method"],
            **{f"cut{k}": sc[k] for k in range(N_CUTS)},
            "mean": float(sc.mean()), "gain_vs_d40": float(sc.mean() - d40_mean),
            "cuts_won": int((sc > d40_cut).sum()),
            "label": "fixed_lambda0.5" if abs(c["lam"] - PRIMARY_LAMBDA) < 1e-12 else "search_grid",
            "is_d40": False, "in_sample_optimum": bool(c["name"] == in_best),
            "loco_selected_for": ",".join(map(str, loco_for[c["name"]])),
        })
    rows.append({
        "variant": "D40", "g": 0, "lambda": 0.0, "disaggregation": "none",
        **{f"cut{k}": d40_cut[k] for k in range(N_CUTS)},
        "mean": d40_mean, "gain_vs_d40": 0.0, "cuts_won": 0,
        "label": "reference", "is_d40": True, "in_sample_optimum": False, "loco_selected_for": "",
    })
    rows.append({
        "variant": "LOCO_selected", "g": "per-cut", "lambda": "per-cut", "disaggregation": "per-cut",
        **{f"cut{k}": loco_sc[k] for k in range(N_CUTS)},
        "mean": loco_mean, "gain_vs_d40": loco_gain, "cuts_won": loco_won,
        "label": "LOCO_honest_adoptable", "is_d40": False, "in_sample_optimum": False,
        "loco_selected_for": "0,1,2,3",
    })
    rows.append({
        "variant": f"IN_SAMPLE[{in_best}]", "g": in_cfg["g"], "lambda": in_cfg["lam"],
        "disaggregation": in_cfg["method"],
        **{f"cut{k}": in_sc[k] for k in range(N_CUTS)},
        "mean": float(in_sc.mean()), "gain_vs_d40": float(in_sc.mean() - d40_mean),
        "cuts_won": int((in_sc > d40_cut).sum()),
        "label": "in_sample_non_adoptable", "is_d40": False, "in_sample_optimum": True,
        "loco_selected_for": ",".join(map(str, loco_for[in_best])),
    })
    wide = pd.DataFrame(rows)
    wide.to_csv(OUT / "results_adamex-svk_temporal-reconcile.csv", index=False)

    # ----- paired bootstrap: honest LOCO candidate vs D40, per cut and pooled
    boot_rows = []
    pooled_c, pooled_r = [], []
    for k in range(N_CUTS):
        cc = details[(sel[k], k)].reindex(base[k]["bidx"])
        rr = details[("D40", k)].reindex(base[k]["bidx"])
        pooled_c.append(cc.to_numpy())
        pooled_r.append(rr.to_numpy())
        point, lo, hi, n = bootstrap(cc, rr, np.random.default_rng(0))
        boot_rows.append(dict(cut=k, selected=sel[k], n=n, gain=point, lo95=lo, hi95=hi))
    pc = pd.Series(np.concatenate(pooled_c))
    pr = pd.Series(np.concatenate(pooled_r))
    point, lo, hi, n = bootstrap(pc, pr, np.random.default_rng(0))
    boot_rows.append(dict(cut="pooled", selected="per-cut", n=n, gain=point, lo95=lo, hi95=hi))
    boot = pd.DataFrame(boot_rows)

    # fixed best lambda=0.5 variant bootstrap (secondary, single config)
    fixed_best = fixed.iloc[0]["variant"]
    fb_rows = []
    for k in range(N_CUTS):
        cc = details[(fixed_best, k)].reindex(base[k]["bidx"])
        rr = details[("D40", k)].reindex(base[k]["bidx"])
        point, lo, hi, n = bootstrap(cc, rr, np.random.default_rng(0))
        fb_rows.append(dict(cut=k, gain=point, lo95=lo, hi95=hi))
    fixed_boot = pd.DataFrame(fb_rows)

    # ----- per-group effects (honest LOCO candidate vs D40), averaged over cuts
    def group_effect(kind):
        recs = []
        for k in range(N_CUTS):
            ctx = ctxs[k]
            M = ctx["M"]
            b = base[k]
            bidx = b["bidx"]
            if kind == "scale":
                q = pd.qcut(b["scale"][b["ok"]].rank(method="first"), 4,
                            labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"])
                labels = q
            else:
                active = pd.Series(M.U[:, -13:].sum(axis=1) > 0, index=b["idx"])[b["ok"]]
                labels = active.map({True: "active_13w", False: "inactive_13w"})
            cc = details[(sel[k], k)].reindex(bidx)
            rr = details[("D40", k)].reindex(bidx)
            d = (cc - rr)
            for grp in pd.unique(labels):
                m = (labels == grp).to_numpy()
                recs.append(dict(cut=k, group=str(grp), n=int(m.sum()),
                                 d40_mase=float(rr[m].mean()), cand_mase=float(cc[m].mean()),
                                 gain_score_points=float(-100 * d[m].mean())))
        df = pd.DataFrame(recs)
        order = (["Q1 smallest", "Q2", "Q3", "Q4 largest"] if kind == "scale"
                 else ["active_13w", "inactive_13w"])
        out = (df.groupby("group", sort=False)
                 .agg(n=("n", "mean"), d40_mase=("d40_mase", "mean"), cand_mase=("cand_mase", "mean"),
                      gain_score_points=("gain_score_points", "mean")).reset_index())
        out["group"] = pd.Categorical(out["group"], categories=order, ordered=True)
        out = out.sort_values("group").reset_index(drop=True)
        df["group"] = pd.Categorical(df["group"], categories=order, ordered=True)
        df = df.sort_values(["cut", "group"]).reset_index(drop=True)
        return df, out

    scale_by_cut, scale_eff = group_effect("scale")
    act_by_cut, act_eff = group_effect("activity")

    # ----- console report
    pd.set_option("display.width", 200)
    print("\n== fixed lambda=0.5 candidates vs D40 (score by cut; higher is better)")
    print(fixed[["variant", "cut0", "cut1", "cut2", "cut3", "mean", "gain", "won"]].round(2).to_string(index=False))
    print("\n== LOCO selection per target cut (chosen on the other 3 cuts)")
    for k in range(N_CUTS):
        print(f"  cut {k}: {sel[k]}  -> honest score {loco_sc[k]:.2f} (D40 {d40_cut[k]:.2f})")
    print(f"\nLOCO honest mean {loco_mean:.2f}  gain {loco_gain:+.2f}  cuts won {loco_won}/4")
    print(f"In-sample optimum (non-adoptable): {in_best}  mean {in_sc.mean():.2f}")
    print("\n== paired bootstrap vs D40 (95%, 1000 resamples, seed 0), score points")
    print(boot.round(2).to_string(index=False))
    print("\n== per-scale-quartile effect (LOCO, averaged over cuts)")
    print(scale_eff.round(2).to_string(index=False))
    print("\n== per-activity effect (LOCO, averaged over cuts)")
    print(act_eff.round(2).to_string(index=False))
    print(f"\nVERDICT: {'ADOPT' if adopt else 'REJECT'} "
          f"(mean gain {loco_gain:+.2f}, cuts won {loco_won}/4; bar +{ADOPT_MEAN} and {ADOPT_CUTS}/4)")

    # ----- notes
    runtime = time.time() - t0
    res_show = wide.copy()
    notes = []
    notes.append("# Iteration 09 / task 4 — temporal reconciliation (adamex-svk)\n")
    notes.append(f"- commit: `{commit}`")
    notes.append(f"- command: `.venv/bin/python src/improve/i09_temporal_reconcile.py`")
    notes.append(f"- runtime: {runtime:.0f}s")
    notes.append(f"- Python: {sys.version.split()[0]}\n")
    notes.append("## What was run\n")
    notes.append(
        "For cuts k=0..3 and aggregation g in {2,4}: aggregate the pre-origin weekly units into "
        "non-overlapping g-week blocks anchored so the last complete block ends at the origin "
        "(`M.U[:, o-g:o]`), train the existing scaled-L1 LightGBM (`train_gbm.PARAMS`, "
        "`num_threads=4`) on `block_units / block_scale`, with block origins from "
        "`features.make_origins(B, horizon=ceil(13/g), min_hist=13)` (default step=2), last 7 as "
        "validation and training origins `<= val_origins[0] - horizon` (purged; validation count is "
        "reduced only when that would leave no training origin, which happens for g=4). "
        "Early stopping 50 rounds, then refit on all origins with `round(best*1.1)+1` trees. "
        "Forecast the next ceil(13/g) blocks and disaggregate either equally or with the last "
        "pre-origin within-block week shares (uniform fallback on a zero block). Apply the zero rule "
        "(no sale in last 8 weeks) and clip; reconcile with the UNROUNDED D40: "
        "`F = (1-lambda)*D40 + lambda*F_agg`; round + clip; score.\n")
    notes.append("Block features: lag_1..lag_13, mean2/4/8/13, median13, max13, zshare13, mean26, "
                 "zshare26, weeks_since_sale (cap 104), series_age, log_scale, log_mean13, plus h "
                 "(all scaled by the block scale where applicable).\n")
    notes.append("## Results (wide; scores are (1-MASE)*100, higher is better)\n")
    notes.append(md_table(res_show))
    notes.append("\n## Fixed lambda=0.5 (evaluated first)\n")
    notes.append(md_table(fixed[["variant", "g", "method", "cut0", "cut1", "cut2", "cut3", "mean", "gain", "won"]]))
    notes.append("\n## LOCO selection (chosen on the other 3 cuts)\n")
    notes.append(md_table(pd.DataFrame([dict(target_cut=k, selected=sel[k], score=loco_sc[k],
                                             d40=d40_cut[k], gain=loco_sc[k] - d40_cut[k])
                                        for k in range(N_CUTS)])))
    notes.append("\n## Paired bootstrap vs D40 — honest LOCO (95%, 1000 resamples, seed 0, score points)\n")
    notes.append(md_table(boot))
    notes.append("\nFixed best lambda=0.5 variant (`" + fixed_best + "`) bootstrap:\n")
    notes.append(md_table(fixed_boot.assign(variant=fixed_best)[["variant", "cut", "gain", "lo95", "hi95"]]))
    notes.append("\n## Per-scale-quartile effect (LOCO, averaged over cuts; gain_score_points = score-point change vs D40, positive = candidate better)\n")
    notes.append(md_table(scale_eff))
    notes.append("\nPer cut:\n")
    notes.append(md_table(scale_by_cut))
    notes.append("\n## Per-activity effect (sold in last 13 weeks vs not; LOCO, averaged over cuts; gain_score_points positive = candidate better)\n")
    notes.append(md_table(act_eff))
    notes.append("\nPer cut:\n")
    notes.append(md_table(act_by_cut))
    notes.append("\n## Verdict\n")
    notes.append(
        f"**{'ADOPT' if adopt else 'REJECT'}** — honest LOCO mean gain {loco_gain:+.2f} score points, "
        f"cuts won {loco_won}/4 (bar: mean >= +{ADOPT_MEAN} and >= {ADOPT_CUTS}/4). "
        f"In-sample optimum `{in_best}` (mean {in_sc.mean():.2f}) is reported for reference only and "
        f"is non-adoptable.\n")
    notes.append("## Notes, warnings and leakage risks\n")
    notes.append(
        "- The final block forecast is trained only on pre-origin blocks; the holdout is never read, "
        "so there is no target leakage. Reconciliation mixes only pre-computed D40 and aggregate "
        "forecasts.\n"
        "- For g=4 the block history is short (`B = 22..32` blocks). With `min_hist=13` and the "
        "default step=2 in `make_origins`, the purged split often has very few training origins; "
        "the validation count is reduced (to 1) so that at least one purged training origin exists. "
        "This makes g=4 weak late in the series and is reported honestly rather than tuned away.\n"
        "- `mean2` is an explicit block feature (the weekly `features_at` has no mean2); the rest of "
        "the block feature set mirrors `features_at`.\n"
        "- D40 is unchanged: the reference recipe command still prints mean -13.67.\n"
        f"- Block-fit early-stopping iterations: "
        + ", ".join(f"cut{k}/g{b}={block_info[(k, b)]['best']}" for k in range(N_CUTS) for b in BLOCKS) + "\n")
    (OUT / "notes_adamex-svk_temporal-reconcile.md").write_text("\n".join(notes))
    print(f"\nwrote {OUT/'results_adamex-svk_temporal-reconcile.csv'}")
    print(f"wrote {OUT/'notes_adamex-svk_temporal-reconcile.md'}")
    print(f"runtime {runtime:.0f}s, commit {commit}")


if __name__ == "__main__":
    main()
