"""Iteration 05: FFORMA-style meta-ensemble over the overlapping cuts (D37).

    .venv/bin/python src/meta/fforma.py [--models m1,m2,...] [--epochs N] [--lr X] [--l2 X]

What it does, per overlapping cut offset W in 0,4,...,48 (holdout = the 13 weeks after the cut):
  * assembles the base forecasts that already exist on disk:
      - LightGBM   iterations/05-meta-ensemble/base_forecasts/lgbm_w{W}.npy   (ShopMatrix key order)
      - median13   iterations/05-meta-ensemble/base_forecasts/median13_w{W}.npy
      - zero       the constant-0 forecast
      - foundation iterations/03-foundation-model/forecasts_cuts/{model}__test__w{W}__padded.parquet
  * actuals and the per-article MASE scale come from validate.split / validate.history_scale (D26);
  * the 8-week zero rule is applied to every forecast before scoring (D39) and training rows that
    the zero rule sends to 0 are dropped from the loss (their error is constant in the weights).

Meta-learners compared on the purged protocol (iterations/05-meta-ensemble/plan.md):
  1. simple blend (reference): average the foundation models, zero rule, then w * LightGBM +
     (1 - w) * ensemble, w chosen on the training cuts only;
  2. FFORMA-linear: per-article softmax(X @ theta) weights over ALL base methods, trained on the
     exact metric objective mean_i mean_w |blend - actual| / scale, L2 + early stopping.

Purged protocol: a test cut at offset W_t may only use cuts with W >= W_t + 13 for training/weight
selection, so the whole training holdout lies before the test holdout. Test cuts W_t = 0, 4, 8, 12.

Outputs iterations/05-meta-ensemble/results_meta_fforma.csv and a summary on stdout.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import features as f  # noqa: E402
import validate as v  # noqa: E402

H = v.HORIZON
KEYS = v.KEYS
OFFSETS = list(range(0, 49, 4))
TEST_OFFSETS = [0, 4, 8, 12]
PURGE = 13                                   # a training cut must be at least this many weeks older
BASE_DIR = ROOT / "iterations" / "05-meta-ensemble" / "base_forecasts"
CUTS_DIR = ROOT / "iterations" / "03-foundation-model" / "forecasts_cuts"
OUT = ROOT / "iterations" / "05-meta-ensemble"
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)
HANDLE = "meta"


# ----------------------------------------------------------------- data assembly
def foundation_matrix(model, W, keys):
    p = CUTS_DIR / f"{model}__test__w{W}__padded.parquet"
    if not p.exists():
        return None
    fc = pd.read_parquet(p)
    P = fc.pivot_table(index=KEYS, columns="week", values="forecast", aggfunc="sum")
    P = P.reindex(pd.MultiIndex.from_frame(keys))
    if P.shape != (len(keys), H) or P.isna().any().any():
        return None
    return P.to_numpy(float)


def article_features(M):
    """Compact per-article features from the training history (weeks < cut)."""
    U = M.U.astype(np.float64)
    o = M.W
    scale = np.abs(np.diff(U, axis=1)).mean(axis=1)
    s = np.where(scale > 0, scale, 1.0)
    last4, last8, last13, last26 = U[:, -4:], U[:, -8:], U[:, -13:], U[:, -26:]
    rev = U[:, ::-1] > 0
    any_sale = rev.any(axis=1)
    wss = np.where(any_sale, np.minimum(rev.argmax(axis=1), 104), 104)
    first = (U > 0).argmax(axis=1)
    age = np.where(any_sale, o - first, 0)
    oh = np.nan_to_num(M.OH[:, -1].astype(np.float64), nan=0.0)
    feat = np.column_stack([
        np.log1p(scale),
        np.log1p(last4.mean(axis=1)), np.log1p(last13.mean(axis=1)), np.log1p(last26.mean(axis=1)),
        last4.mean(axis=1) / s, last13.mean(axis=1) / s,
        (last8 == 0).mean(axis=1), (last13 == 0).mean(axis=1), (last26 == 0).mean(axis=1),
        wss, age,
        np.log1p(np.clip(oh, 0, None)), np.isnan(M.OH[:, -1]).astype(float), M.OOS[:, -1].astype(float),
    ])
    return feat.astype(np.float64), scale


def load_cut(W):
    full = v.load_weekly("test")
    df = full[full["week"] <= full["week"].max() - pd.Timedelta(weeks=W)]
    train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut])
    weeks = pd.date_range(cut, periods=H, freq="7D")
    assert list(weeks) == sorted(hold["week"].unique()), f"week grid mismatch at W={W}"
    keys = M.keys
    hold_keys = hold[KEYS].drop_duplicates()
    assert set(map(tuple, keys.to_numpy())) == set(map(tuple, hold_keys.to_numpy())), \
        f"train/hold series differ at W={W}"

    A = hold.pivot_table(index=KEYS, columns="week", values="units").reindex(pd.MultiIndex.from_frame(keys)).to_numpy(float)
    X, scale = article_features(M)
    valid = np.isfinite(scale) & (scale > 0)
    s = np.where(valid, scale, 1.0)
    dead = M.U[:, -8:].sum(axis=1) == 0
    train_mask = valid & ~dead                      # zero rule makes dead rows constant in the weights

    bases = {}
    lgbm = np.load(BASE_DIR / f"lgbm_w{W}.npy")
    med = np.load(BASE_DIR / f"median13_w{W}.npy")
    assert lgbm.shape == (M.n, H) and med.shape == (M.n, H), f"base shape mismatch at W={W}"
    bases["lgbm"] = lgbm.astype(float)
    bases["median13"] = med.astype(float)
    bases["zero"] = np.zeros((M.n, H))
    for model in MODELS:
        fm = foundation_matrix(model, W, keys)
        if fm is not None:
            bases[model] = fm

    return {"W": W, "M": M, "train": train, "hold": hold, "weeks": weeks, "keys": keys,
            "A": A, "s": s, "valid": valid, "dead": dead, "train_mask": train_mask,
            "X": X, "bases": bases}


def post(F, c):
    """Zero rule -> round -> clip (D38/D39)."""
    G = np.array(F, dtype=float, copy=True)
    G[c["dead"]] = 0.0
    return np.clip(np.round(G), 0, None)


def score(c, F):
    G = post(F, c)
    n, _ = G.shape
    fc = pd.DataFrame({
        "store_id": np.repeat(c["keys"]["store_id"].to_numpy(), H),
        "article_id": np.repeat(c["keys"]["article_id"].to_numpy(), H),
        "week": np.tile(np.asarray(c["weeks"]), n),
        "forecast": G.reshape(-1)})
    return v.score(c["train"], c["hold"], fc)["score"]


# ----------------------------------------------------------------- simple blend
def foundation_names(c):
    return [m for m in MODELS if m in c["bases"]]


def ensemble_raw(c):
    names = foundation_names(c)
    assert names, f"no foundation forecasts at W={c['W']}"
    return np.mean([c["bases"][m] for m in names], axis=0)


def simple_blend(c, w):
    return w * c["bases"]["lgbm"] + (1 - w) * ensemble_raw(c)


def choose_weight(train_cuts):
    scores = {w: np.mean([score(c, simple_blend(c, w)) for c in train_cuts]) for w in WEIGHTS}
    wbest = max(WEIGHTS, key=lambda w: scores[w])
    return wbest, scores


# ----------------------------------------------------------------- FFORMA-linear
FF_NAMES = ["lgbm", "median13", "zero"]


def ff_names(c):
    return [nm for nm in FF_NAMES if nm in c["bases"]] + foundation_names(c)


def set_ff_bases(spec):
    global FF_NAMES
    FF_NAMES = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [s for s in FF_NAMES if s not in ("lgbm", "median13", "zero")]
    assert not unknown, f"unknown base methods {unknown}"


def stack_bases(c):
    return np.stack([c["bases"][nm] for nm in ff_names(c)], axis=1)   # (n, K, H)


def standardise(cuts):
    X = np.vstack([c["X"] for c in cuts])
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    return np.nan_to_num(mu), sd


def prep_X(c, mu, sd):
    X = np.nan_to_num(c["X"], nan=0.0)
    Z = (X - mu) / sd
    return np.hstack([Z, np.ones((len(Z), 1))])


def forward(X, theta, F):
    z = X @ theta.T
    z -= z.max(axis=1, keepdims=True)
    w = np.exp(z)
    w /= w.sum(axis=1, keepdims=True)
    pred = np.einsum("nk,nkw->nw", w, F)
    return w, pred


def masked_loss(c, theta, mu, sd):
    X = prep_X(c, mu, sd)
    F = stack_bases(c)
    _, pred = forward(X, theta, F)
    m = c["train_mask"]
    per = (np.abs(pred - c["A"]) / c["s"][:, None]).mean(axis=1)
    return float(per[m].sum() / max(m.sum(), 1))


def fit_fforma(cuts, val_cut, epochs=1500, lr=0.1, l2=1e-2, early_stop=False, min_epochs=200):
    mu, sd = standardise(cuts)
    data = [(prep_X(c, mu, sd), stack_bases(c), c["A"], c["s"], c["train_mask"]) for c in cuts]
    K, d = data[0][1].shape[1], data[0][0].shape[1]
    theta = np.zeros((K, d))
    m_adam = np.zeros_like(theta)
    v_adam = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best, best_theta, best_epoch = np.inf, theta.copy(), epochs
    for ep in range(1, epochs + 1):
        grad = np.zeros_like(theta)
        cnt = 0
        for X, F, A, s, mask in data:
            w, pred = forward(X, theta, F)
            coef = np.sign(pred - A) / (H * s[:, None])
            coef *= mask[:, None]
            fp = np.einsum("nw,nkw->nk", coef, F)
            cp = (coef * pred).sum(axis=1, keepdims=True)
            gz = w * (fp - cp)
            grad += gz.T @ X
            cnt += int(mask.sum())
        grad /= max(cnt, 1)
        grad += l2 * theta
        m_adam = b1 * m_adam + (1 - b1) * grad
        v_adam = b2 * v_adam + (1 - b2) * grad * grad
        mh = m_adam / (1 - b1 ** ep)
        vh = v_adam / (1 - b2 ** ep)
        theta -= lr * mh / (np.sqrt(vh) + eps)
        if early_stop and val_cut is not None and ep >= min_epochs:
            l = masked_loss(val_cut, theta, mu, sd)
            if l < best:
                best, best_theta, best_epoch = l, theta.copy(), ep
    if not (early_stop and val_cut is not None):
        best_theta, best_epoch = theta.copy(), epochs
    return {"theta": best_theta, "mu": mu, "sd": sd, "val_epoch": best_epoch, "val_loss": best}


def fforma_forecast(c, fitted):
    X = prep_X(c, fitted["mu"], fitted["sd"])
    F = stack_bases(c)
    _, pred = forward(X, fitted["theta"], F)
    return pred


def hedge(c, ff_raw, w_simple, h):
    """h * FFORMA + (1 - h) * simple blend, both raw (post-processing is applied by score/post)."""
    return h * ff_raw + (1 - h) * simple_blend(c, w_simple)


def fit_cuts_for(train_offsets, cuts, fit_all):
    """The tested protocol holds the newest training cut out as a (by default unused) validation cut;
    fit_all=True trains on every training cut instead."""
    val_W = min(train_offsets)
    fit = [cuts[W] for W in train_offsets if fit_all or W != val_W]
    return fit, cuts[val_W]


# ----------------------------------------------------------------- final weeks (submission)
FM_FINAL = ROOT / "iterations" / "03-foundation-model" / "forecasts_final"


def load_final():
    """The real forecast origin: features from the whole test-shop history, bases = the real-week
    forecasts (LightGBM lgbm_own_v1.csv + foundation forecasts_final). Same dict layout as load_cut,
    without actuals."""
    import make_submission as ms
    full = v.load_weekly("test")
    M = f.ShopMatrix(full)
    keys = M.keys
    ids = keys["article_id"].tolist()
    assert keys["store_id"].nunique() == 1 and sorted(ids) == ms.history_ids(), "final keys differ from the history"
    X, scale = article_features(M)
    dead = M.U[:, -8:].sum(axis=1) == 0
    bases = {"zero": np.zeros((M.n, H))}
    L = pd.read_csv(ms.SUB_DIR / "lgbm_own_v1.csv").set_index("article_id").loc[ids]
    bases["lgbm"] = L.to_numpy(float)
    if "median13" in FF_NAMES:
        bases["median13"] = np.repeat(np.median(M.U[:, -13:], axis=1)[:, None], H, axis=1).astype(float)
    for model in MODELS:
        fc = pd.read_parquet(FM_FINAL / f"{model}__test__final__padded.parquet")
        P = fc.pivot_table(index="article_id", columns="week", values="forecast", aggfunc="sum").reindex(ids)
        assert P.shape == (M.n, H) and P.notna().all().all(), f"{model}: final forecast incomplete"
        bases[model] = P.to_numpy(float)
    return {"W": "final", "M": M, "keys": keys, "ids": ids, "X": X, "dead": dead, "bases": bases}


def write_final(c, F, path):
    import make_submission as ms
    G = post(F, c)
    order = pd.Index(c["ids"]).get_indexer(ms.history_ids())
    out = Path(path)
    if not out.is_absolute():
        out = ROOT / out
    ms.write_submission(G[order], ms.history_ids(), out)
    ms.validate_submission(out)
    print(f"wrote {out} (format checks pass) | mean forecast {G.mean():.3f} | "
          f"{100 * (G.max(axis=1) == 0).mean():.1f}% all-zero articles")
    return G[order]


# ----------------------------------------------------------------- driver
MODELS = []


def main():
    global MODELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="moirai-2.0-small,tirex,chronos-t5-small",
                    help="comma-separated foundation models to blend (must exist at the cuts)")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--early-stop", action="store_true", help="early-stop on the adjacent training cut (min-epochs first)")
    ap.add_argument("--min-epochs", type=int, default=200)
    ap.add_argument("--ff-bases", default="lgbm,median13,zero",
                    help="non-foundation base methods FFORMA may weight (subset of lgbm,median13,zero)")
    ap.add_argument("--out", default=None, help="results CSV name (default results_meta_fforma.csv)")
    ap.add_argument("--hedge-weight", type=float, default=0.5,
                    help="weight of FFORMA in the hedge h*FFORMA + (1-h)*simple blend (also scored)")
    ap.add_argument("--fit-all", action="store_true",
                    help="train FFORMA on every training cut (default: hold the newest one out, as tested in D37)")
    ap.add_argument("--final", action="store_true",
                    help="instead of evaluating, train on the overlapping cuts and write submissions for the real weeks")
    ap.add_argument("--final-lgbm-weight", type=float, default=0.3,
                    help="LightGBM weight of the simple blend inside the final hedge (D40 uses 0.3)")
    ap.add_argument("--out-pure", default=None, help="final mode: path of the pure FFORMA submission")
    ap.add_argument("--out-hedge", default=None, help="final mode: path of the hedge submission")
    a = ap.parse_args()
    MODELS = [m.strip() for m in a.models.split(",") if m.strip()]
    set_ff_bases(a.ff_bases)
    if a.final:
        return run_final(a)

    print(f"FFORMA meta-ensemble: foundation models {MODELS}; offsets {OFFSETS}; test offsets {TEST_OFFSETS}")
    print("loading cuts ...", flush=True)
    cuts = {W: load_cut(W) for W in OFFSETS}
    available = [W for W in OFFSETS if all(foundation_matrix(m, W, cuts[W]["keys"]) is not None for m in MODELS)]
    print(f"cuts with every base model forecast available: {available}")

    rows = []
    for W_t in TEST_OFFSETS:
        if W_t not in cuts:
            continue
        train_offsets = [W for W in OFFSETS if W >= W_t + PURGE and W in available]
        if len(train_offsets) < 2:
            print(f"W_t={W_t}: only {len(train_offsets)} training cut(s), skipped")
            continue
        train_cuts = [cuts[W] for W in train_offsets]
        wbest, wsc = choose_weight(train_cuts)
        ref = score(cuts[W_t], simple_blend(cuts[W_t], wbest))

        fit_cuts, val_cut = fit_cuts_for(train_offsets, cuts, a.fit_all)
        fitted = fit_fforma(fit_cuts, val_cut, epochs=a.epochs, lr=a.lr, l2=a.l2,
                            early_stop=a.early_stop, min_epochs=a.min_epochs)
        ff_raw = fforma_forecast(cuts[W_t], fitted)
        ff = score(cuts[W_t], ff_raw)
        hg = score(cuts[W_t], hedge(cuts[W_t], ff_raw, wbest, a.hedge_weight))

        Xt = prep_X(cuts[W_t], fitted["mu"], fitted["sd"])
        wt, _ = forward(Xt, fitted["theta"], stack_bases(cuts[W_t]))
        mean_w = wt.mean(axis=0)
        wstr = " ".join(f"{nm}={x:.2f}" for nm, x in zip(ff_names(cuts[W_t]), mean_w))

        lgbm = score(cuts[W_t], cuts[W_t]["bases"]["lgbm"])
        ens = score(cuts[W_t], ensemble_raw(cuts[W_t]))
        rows.append({"test_offset_W": W_t, "n_train_cuts": len(train_offsets), "simple_weight": wbest,
                     "lgbm_alone": round(lgbm, 2), "ensemble_alone": round(ens, 2),
                     "simple_blend": round(ref, 2), "fforma": round(ff, 2),
                     "gain_fforma_minus_simple": round(ff - ref, 2),
                     "fforma_val_epoch": fitted["val_epoch"],
                     "hedge": round(hg, 2), "gain_hedge_minus_simple": round(hg - ref, 2)})
        print(f"W_t={W_t:2d} | train {len(train_offsets)} cuts, simple w={wbest:.1f} | "
              f"LightGBM {lgbm:6.2f}  ensemble {ens:6.2f}  simple {ref:6.2f}  FFORMA {ff:6.2f}  "
              f"gain {ff - ref:+.2f} | hedge {hg:6.2f} gain {hg - ref:+.2f} "
              f"(val_epoch {fitted['val_epoch']})", flush=True)
        print(f"      mean FFORMA weights: {wstr}", flush=True)

    res = pd.DataFrame(rows)
    out_path = OUT / (a.out or f"results_{HANDLE}_fforma.csv")
    res.to_csv(out_path, index=False)
    if len(res):
        ms, mf = res["simple_blend"].mean(), res["fforma"].mean()
        won = int((res["fforma"] > res["simple_blend"]).sum())
        print(f"\nmean over {len(res)} test cuts: simple blend {ms:.2f}, FFORMA {mf:.2f}, "
              f"gain {mf - ms:+.2f}, FFORMA better on {won}/{len(res)} cuts")
        print("adoption bar: >= +0.50 mean AND better on >= 3/4 cuts ->",
              "ADOPT FFORMA" if (mf - ms >= 0.5 and won >= 3) else "KEEP SIMPLE BLEND")
        mh = res["hedge"].mean()
        wonh = int((res["hedge"] > res["simple_blend"]).sum())
        print(f"hedge (h={a.hedge_weight}): {mh:.2f}, gain {mh - ms:+.2f}, better on {wonh}/{len(res)} cuts")
    print(f"\nwrote {out_path}")


def run_final(a):
    """Train FFORMA on the overlapping cuts (all lie before the real forecast weeks) and write the
    pure and hedge submissions for the real weeks."""
    print(f"FFORMA final: foundation models {MODELS}; FFORMA bases {FF_NAMES}; offsets {OFFSETS}")
    cuts = {W: load_cut(W) for W in OFFSETS}
    available = [W for W in OFFSETS if all(m in cuts[W]["bases"] for m in MODELS)]
    print(f"training cuts: {available} ({'all' if a.fit_all else 'newest held out, as tested'})")
    fit_cuts, val_cut = fit_cuts_for(available, cuts, a.fit_all)
    fitted = fit_fforma(fit_cuts, val_cut, epochs=a.epochs, lr=a.lr, l2=a.l2,
                        early_stop=a.early_stop, min_epochs=a.min_epochs)
    c = load_final()
    ff_raw = fforma_forecast(c, fitted)
    wt, _ = forward(prep_X(c, fitted["mu"], fitted["sd"]), fitted["theta"], stack_bases(c))
    print("mean FFORMA weights at the real origin: " +
          " ".join(f"{nm}={x:.2f}" for nm, x in zip(ff_names(c), wt.mean(axis=0))))
    out = {}
    if a.out_pure:
        out["pure"] = write_final(c, ff_raw, a.out_pure)
    if a.out_hedge:
        out["hedge"] = write_final(c, hedge(c, ff_raw, a.final_lgbm_weight, a.hedge_weight), a.out_hedge)
    d40 = post(simple_blend(c, a.final_lgbm_weight), c)
    for k, G in out.items():
        order = pd.Index(c["ids"]).get_indexer(sorted(c["ids"]))
        diff = np.abs(G - d40[order])
        print(f"{k}: differs from the D40-style blend in {100 * (diff.max(axis=1) > 0).mean():.1f}% of articles, "
              f"mean |diff| {diff.mean():.3f} units")


if __name__ == "__main__":
    main()
