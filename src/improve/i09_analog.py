"""Iteration 09, task 5: analog / nearest-neighbour conditional-median forecaster (local CPU).

For a target origin O, a donor window at origin o is eligible only if its whole 13-week future ends
before O (o + 13 <= O). Donors are all article-origin windows (same article allowed), represented by
the trailing 26-week sequence normalised by the D26 scale plus log level, 26-week zero share and weeks
since sale. The forecast is the inverse-distance weighted median of the 25 nearest donor futures,
rescaled by the target scale; median13 is the deterministic fallback.

Evaluated alone, as a fourth D40 member, and in the fixed 0.15 LightGBM + 0.15 analog + 0.7 foundation
recipe. The standard 4-cut adoption bar applies.

Output: iterations/09-forecast-approaches/results_adamex-svk_analog.csv
Run:    .venv/bin/python src/improve/i09_analog.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "improve"))
import features as f  # noqa: E402
import validate as v  # noqa: E402
import i09_error_atlas as atlas  # noqa: E402

warnings.filterwarnings("ignore")
H = f.H
OUT = ROOT / "iterations" / "09-forecast-approaches"
N_CUTS = 4
KS = (10, 25, 50)
PRIMARY_K = 25
REP_LEN = 26
STEP = 2
ADOPT_MEAN, ADOPT_CUTS, STOP_GAIN = 0.5, 3, 0.25


# ------------------------------------------------------------------ representation
def series_features(U, o, rep_len=REP_LEN):
    """Representation of every series at origin o using weeks < o only."""
    n = U.shape[0]
    Uo = U[:, :o].astype(float)
    scale = np.abs(np.diff(Uo, axis=1)).mean(axis=1)
    s = np.where(scale > 0, scale, np.nan)
    shape = np.full((n, rep_len), np.nan)
    if o >= rep_len:
        shape = Uo[:, o - rep_len:o] / s[:, None]
    level = np.log1p(Uo[:, max(0, o - 13):o].mean(axis=1))
    zshare = (Uo[:, max(0, o - REP_LEN):o] == 0).mean(axis=1)
    rev = Uo[:, ::-1] > 0
    any_sale = rev.any(axis=1)
    wss = np.where(any_sale, np.minimum(rev.argmax(axis=1), 104), 104)
    D = np.column_stack([shape, level, zshare, wss / 13.0]).astype(np.float32)
    return D, scale


def donor_origins(W, target_o, rep_len=REP_LEN, step=STEP):
    last = target_o - H
    return np.arange(rep_len, last + 1, step, dtype=int)


def analog_forecast(U, k=PRIMARY_K, rep_len=REP_LEN, step=STEP, fallback_weeks=13, origin=None):
    """Forecast the 13 weeks after `origin` (default: the last column of U).

    Passing `origin` lets a caller hand in a longer frame and check that columns at/after the
    origin cannot change the forecast (the leakage test)."""
    U = np.asarray(U, dtype=np.float32)
    n, Wfull = U.shape
    W = int(origin) if origin is not None else Wfull
    target, tscale = series_features(U, W, rep_len)
    med13 = np.median(U[:, W - fallback_weeks:W], axis=1).astype(float)
    fb = np.repeat(med13[:, None], H, axis=1)
    out = fb.copy()
    origins = donor_origins(W, W, rep_len, step)
    if len(origins) == 0:
        return out, fb, {"n_donors": 0, "fallback": 1.0}
    Ds, Fs = [], []
    for o in origins:
        D, sc = series_features(U, o, rep_len)
        ok = np.isfinite(sc) & (sc > 0) & np.isfinite(D).all(axis=1)
        Ds.append(D[ok])
        Fs.append((U[ok, o:o + H].astype(float) / sc[ok, None]))
    Dd = np.vstack(Ds)
    Fd = np.vstack(Fs)
    ok_t = np.isfinite(tscale) & (tscale > 0) & np.isfinite(target).all(axis=1)
    if ok_t.sum() == 0 or len(Dd) < k:
        return out, fb, {"n_donors": len(Dd), "fallback": 1.0}
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1).fit(Dd)
    dist, idx = nn.kneighbors(target[ok_t])
    ratio = np.zeros((int(ok_t.sum()), H))
    for i in range(len(idx)):
        vals = Fd[idx[i]]
        w = 1.0 / (dist[i] + 1e-6)
        order = np.argsort(vals, axis=0)
        vv = np.take_along_axis(vals, order, axis=0)
        ww = w[order]
        cw = np.cumsum(ww, axis=0)
        pick = (cw >= 0.5 * cw[-1]).argmax(axis=0)
        ratio[i] = vv[pick, np.arange(H)]
    out[ok_t] = tscale[ok_t, None] * ratio
    out = np.clip(np.nan_to_num(out, nan=0.0), 0, None)
    return out, fb, {"n_donors": len(Dd), "fallback": 0.0, "n_target_valid": int(ok_t.sum())}


# ------------------------------------------------------------------ recipes
def score_long(M, train, hold, F, weeks):
    fc = pd.DataFrame({
        "store_id": np.repeat(M.keys["store_id"].to_numpy(), H),
        "article_id": np.repeat(M.keys["article_id"].to_numpy(), H),
        "week": np.tile(np.asarray(weeks), M.n),
        "forecast": np.clip(np.round(F), 0, None).reshape(-1)})
    return v.score(train, hold, fc, detail=True)


def overlap_rows(cut_atlas, k):
    M, L = cut_atlas["M"], cut_atlas["L"]
    dead = atlas.dead_mask(cut_atlas["train"], M)
    raw = {m: atlas.matrix_from_parquet(atlas.FC / f"{m}__test__cut{k}__padded.parquet", M)
           for m in atlas.REF_MODELS}
    return M, L, dead, raw


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, detail = [], {}
    rng = np.random.default_rng(0)
    print("building analog forecasts", flush=True)
    ctxs, analog, fallback_stats = {}, {}, []
    for k in range(N_CUTS):
        ctx = atlas.standard_ctx(k)
        ctxs[k] = ctx
        U = ctx["M"].U.astype(np.float32)
        F, fb, info = analog_forecast(U, k=PRIMARY_K)
        analog[k] = (F, fb)
        fallback_stats.append({"cut": k, **info})
        print(f"  cut {k}: {info}", flush=True)

    for k in range(N_CUTS):
        ctx = ctxs[k]
        M, L, dead, raw = overlap_rows(ctx, k)
        d40_cand, _ = atlas.standard_candidates(ctx)
        d40 = d40_cand["D40"]
        med13 = np.repeat(np.median(M.U[:, -13:], axis=1)[:, None], H, axis=1)
        Fa, fb = analog[k]
        # analog alone with the zero rule
        A0 = Fa.copy()
        A0[dead] = 0.0
        # fourth member: mean of four raw forecasts, zero rule, 0.3 LightGBM + 0.7 ensemble
        E4 = np.mean(list(raw.values()) + [A0], axis=0)
        E4[dead] = 0.0
        fourth = 0.3 * L + 0.7 * E4
        # fixed slot split: 0.15 L + 0.15 analog + 0.7 foundation
        E3 = np.mean(list(raw.values()), axis=0)
        E3[dead] = 0.0
        slot = 0.15 * L + 0.15 * A0 + 0.7 * E3
        cands = {"D40": d40, "analog": Fa, "median13": med13, "fourth_member": fourth,
                 "slot_0.15_0.15_0.7": slot}
        for name, F in cands.items():
            z8 = name in ("analog",)
            if z8:
                F = F.copy()
                F[dead] = 0.0
            r = score_long(M, ctx["train"], ctx["hold"], F, ctx["weeks"])
            detail[(name, k)] = r["detail"]["mase"]
            rows.append({"variant": name, "cut": k, "score": r["score"], "mean_mase": r["mase"],
                         "n_scored": r["n_scored"]})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results_adamex-svk_analog.csv", index=False)
    piv = res.pivot_table(index="variant", columns="cut", values="score")
    piv["mean"] = piv.mean(axis=1)
    print("\nAnalog results (score by cut):")
    print(piv.round(2).to_string())

    best = "analog"
    if piv.loc["analog", "mean"] < piv.loc["median13", "mean"]:
        print("\nSTOP: analog does not beat median13 on the 4-cut mean.")
    print("\nGains vs D40:")
    for name in piv.index:
        if name == "D40":
            continue
        d = piv.loc[name, list(range(N_CUTS))] - piv.loc["D40", list(range(N_CUTS))]
        a = pd.concat([detail[(name, k)] for k in range(N_CUTS)]).to_numpy()
        b = pd.concat([detail[("D40", k)] for k in range(N_CUTS)]).to_numpy()
        bs = [-100 * rng.choice(b - a, len(a)).mean() for _ in range(1000)]
        print(f"  {name:20s} mean {d.mean():+6.2f}  won {(d > 0).sum()}/4  "
              f"interval [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]")
    pd.DataFrame(fallback_stats).to_csv(OUT / "results_adamex-svk_analog-donors.csv", index=False)
    print("\ndone")


if __name__ == "__main__":
    main()
