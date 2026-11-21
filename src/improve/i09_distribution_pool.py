"""Iteration 09, task 3: pool the predictive distributions of the three D40 members, then take the median.

The members' quantile forecasts must first be exported on the GPU machine with

    python src/foundation/run_model.py MODEL test --cut-index K --emit-quantiles
    python src/foundation/run_model.py MODEL test --final --emit-quantiles

for MODEL in moirai-2.0-small, tirex, chronos-t5-small.  The additive exporter writes
iterations/09-forecast-approaches/quantiles/{model}__test__cutK__padded.npz (and ...__final__...).
This script pools the three CDFs with fixed equal model weights, reconstructs the mixture median by
monotone inverse-CDF interpolation, checks quantile monotonicity and interval coverage, and evaluates
the pooled median alone and inside the exact D40 recipe.

Output: iterations/09-forecast-approaches/results_adamex-svk_distribution-pool.csv
        (and, only when the comparison runs, the pseudo-model `mixture-median-top3`)
Run:    .venv/bin/python src/improve/i09_distribution_pool.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "improve"))
import features as f  # noqa: E402
import validate as v  # noqa: E402
import i09_error_atlas as atlas  # noqa: E402

warnings.filterwarnings("ignore")
H = f.H
OUT = ROOT / "iterations" / "09-forecast-approaches"
QDIR = OUT / "quantiles"
FC = ROOT / "iterations" / "03-foundation-model" / "forecasts"
FCF = ROOT / "iterations" / "03-foundation-model" / "forecasts_final"
MODELS = list(atlas.REF_MODELS)
N_CUTS = 4
ADOPT_MEAN, ADOPT_CUTS, KILL_MEAN = 0.5, 3, 0.25


def enforce_monotone(q):
    return np.maximum.accumulate(np.asarray(q, float), axis=-1)


def mixture_quantile(quantiles, levels_list, p):
    """Quantile at level p of the equal-weight linear pool of CDFs.

    quantiles: list of [..., Q_m] arrays (monotone along the last axis); levels_list: matching
    probability grids. Returns (...), tail_dependent (...) where tail_dependent is True when the
    crossing needs a beyond-observed tail (CDF(p) is not bracketed inside the observed quantile range)."""
    qs = [enforce_monotone(q) for q in quantiles]
    shape = qs[0].shape[:-1]
    grid = np.sort(np.concatenate(qs, axis=-1), axis=-1)
    flat_grid = grid.reshape(-1, grid.shape[-1])
    flat_qs = [q.reshape(-1, q.shape[-1]) for q in qs]
    nrow, ng = flat_grid.shape
    out = np.full(nrow, np.nan)
    tail = np.zeros(nrow, dtype=bool)
    for i in range(nrow):
        g = flat_grid[i]
        F = np.zeros(ng)
        for q, lv in zip(flat_qs, levels_list):
            lv = np.asarray(lv, float)
            # np.interp is linear in x=quantile value, which is the piecewise-linear inverse CDF
            F += np.interp(g, q[i], lv, left=0.0, right=1.0)
        F /= len(flat_qs)
        if F[-1] < p or F[0] > p:
            # the requested level is outside the observed 0.1..0.9 range for every member
            out[i] = g[-1] if F[-1] < p else g[0]
            tail[i] = True
            continue
        k = int(np.searchsorted(F, p))
        if k == 0:
            out[i] = g[0]
        elif k >= ng:
            out[i] = g[-1]
        else:
            x0, x1 = g[k - 1], g[k]
            f0, f1 = F[k - 1], F[k]
            out[i] = x0 if f1 == f0 else x0 + (p - f0) * (x1 - x0) / (f1 - f0)
    return out.reshape(shape), tail.reshape(shape)


def mixture_median(quantiles, levels_list):
    return mixture_quantile(quantiles, levels_list, 0.5)


def interval_coverage(q, levels, actual, lo=0.1, hi=0.9):
    q = enforce_monotone(q)
    lv = np.asarray(levels, float)
    shp = q.shape[:-1]
    flat = q.reshape(-1, q.shape[-1])
    ql = np.array([np.interp(lo, lv, row) for row in flat]).reshape(shp)
    qh = np.array([np.interp(hi, lv, row) for row in flat]).reshape(shp)
    inside = (actual >= ql) & (actual <= qh)
    return float(inside.mean()), float((qh - ql).mean())


def load_quantiles(model, tag):
    p = QDIR / f"{model}__test__{tag}__padded.npz"
    if not p.exists():
        return None, None, None, None
    z = np.load(p, allow_pickle=True)
    return z["quantiles"], z["levels"], z["article_id"], z["store_id"]


def available_tags():
    tags = [f"cut{k}" for k in range(N_CUTS)] + ["final"]
    return [t for t in tags if all((QDIR / f"{m}__test__{t}__padded.npz").exists() for m in MODELS)]


def align(q, keys, weeks):
    """q [n,H,Q] -> [len(keys),H,Q] using the article order stored in the exporter (same W.index)."""
    if q.shape[0] == len(keys):
        return q
    raise ValueError("quantile array and article keys differ in length")


def poisson_bootstrap_interval(a, b, rng, n=1000):
    d = (a - b).dropna().to_numpy()
    bs = [-100 * rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tags = available_tags()
    status = []
    if not tags:
        status.append({"status": "BLOCKED", "reason": "no complete quantile exports for the three D40 members",
                       "quantile_dir": str(QDIR), "models": ",".join(MODELS)})
        pd.DataFrame(status).to_csv(OUT / "results_adamex-svk_distribution-pool.csv", index=False)
        print("BLOCKED: no complete quantile exports in", QDIR)
        print("Run run_model.py --emit-quantiles for each model/cut on the GPU machine first.")
        return

    rows, detail = [], {}
    rng = np.random.default_rng(0)
    pooled_points = {}
    for tag in tags:
        loaded = [load_quantiles(m, tag) for m in MODELS]
        qs, lvs = [x[0] for x in loaded], [x[1] for x in loaded]
        nonmono = [float(np.mean(np.any(np.diff(q, axis=-1) < -1e-6, axis=-1))) for q in qs]
        point, tails = mixture_median(qs, lvs)
        qlo, _ = mixture_quantile(qs, lvs, 0.1)
        qhi, _ = mixture_quantile(qs, lvs, 0.9)
        pooled_points[tag] = point
        rows.append({"analysis": "monotonicity", "tag": tag, "model": "each",
                     "value": float(np.mean(nonmono)), "note": "share of article-weeks with a crossing quantile"})
        rows.append({"analysis": "tail_dependence", "tag": tag, "model": "pooled",
                     "value": float(np.mean(tails)), "note": "share whose 0.5 crossing needs an unobserved tail"})
        if tag.startswith("cut"):
            k = int(tag.replace("cut", ""))
            ctx = atlas.standard_ctx(k)
            M = ctx["M"]
            for m, art in zip(MODELS, [x[2] for x in loaded]):
                if not np.array_equal(np.asarray(art).astype(str), M.keys["article_id"].to_numpy().astype(str)):
                    raise ValueError(f"quantile article order for {m} {tag} does not match the cut matrix")
            A = atlas.actual_matrix(ctx["hold"], M, ctx["weeks"])
            for m, q, lv in zip(MODELS, qs, lvs):
                cov, width = interval_coverage(q, lv, A)
                rows.append({"analysis": "coverage_80", "tag": tag, "model": m, "value": cov,
                             "note": f"mean interval width {width:.3f}"})
            # pooled 80% coverage from the reconstructed 0.1/0.9 mixture quantiles
            inside = ((A >= qlo) & (A <= qhi)).mean()
            rows.append({"analysis": "coverage_80", "tag": tag, "model": "pooled", "value": float(inside),
                         "note": f"mean 80% width {float((qhi-qlo).mean()):.3f}"})
            dead = atlas.dead_mask(ctx["train"], M)
            L = ctx["L"]
            E = point.copy()
            E[dead] = 0.0
            cands = {"pooled_alone": point, "pooled_in_D40": 0.3 * L + 0.7 * E}
            d40 = atlas.standard_candidates(ctx)[0]["D40"]
            for name, F in cands.items():
                fc = pd.DataFrame({"store_id": np.repeat(M.keys["store_id"].to_numpy(), H),
                                   "article_id": np.repeat(M.keys["article_id"].to_numpy(), H),
                                   "week": np.tile(np.asarray(ctx["weeks"]), M.n),
                                   "forecast": np.clip(np.round(F), 0, None).reshape(-1)})
                r = v.score(ctx["train"], ctx["hold"], fc, detail=True)
                detail[(name, k)] = r["detail"]["mase"]
                rows.append({"analysis": "score", "tag": tag, "model": name, "value": r["score"],
                             "cut": k})
            r0 = v.score(ctx["train"], ctx["hold"], pd.DataFrame({
                "store_id": np.repeat(M.keys["store_id"].to_numpy(), H),
                "article_id": np.repeat(M.keys["article_id"].to_numpy(), H),
                "week": np.tile(np.asarray(ctx["weeks"]), M.n),
                "forecast": np.clip(np.round(d40), 0, None).reshape(-1)}), detail=True)
            detail[("D40", k)] = r0["detail"]["mase"]
            rows.append({"analysis": "score", "tag": tag, "model": "D40", "value": r0["score"],
                         "cut": k})

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results_adamex-svk_distribution-pool.csv", index=False)
    sc = res[res.analysis == "score"].pivot(index="model", columns="cut", values="value")
    if len(sc):
        sc["mean"] = sc.mean(axis=1)
        print(sc.round(2).to_string())
        for name in [n for n in sc.index if n != "D40"]:
            d = sc.loc[name, list(range(N_CUTS))] - sc.loc["D40", list(range(N_CUTS))]
            lo, hi = poisson_bootstrap_interval(pd.concat([detail[(name, k)] for k in range(N_CUTS)]),
                                                pd.concat([detail[("D40", k)] for k in range(N_CUTS)]), rng)
            print(f"  {name}: mean {d.mean():+.2f}, won {(d > 0).sum()}/4, interval [{lo:+.2f}, {hi:+.2f}]")
        best = max([n for n in sc.index if n != "D40"], key=lambda n: sc.loc[n, "mean"])
        gain = float(sc.loc[best, "mean"] - sc.loc["D40", "mean"])
        verdict = "KILL (< +0.25)" if gain < KILL_MEAN else (
            "ADOPT" if gain >= ADOPT_MEAN and (sc.loc[best, list(range(N_CUTS))] > sc.loc["D40", list(range(N_CUTS))]).sum() >= ADOPT_CUTS
            else "REJECT")
        print(f"pooled best={best} gain {gain:+.2f} -> {verdict}")

    # save the pooled point forecasts only (small; quantile arrays stay out of git)
    if len(sc):
        for tag, point in pooled_points.items():
            if tag.startswith("cut"):
                ctx = atlas.standard_ctx(int(tag.replace("cut", "")))
                keys, weeks = ctx["M"].keys, ctx["weeks"]
                out = FC / f"mixture-median-top3__test__{tag}__padded.parquet"
            else:
                ids = pd.read_csv(ROOT / "submissions" / "lgbm_own_v1.csv")["article_id"]
                keys = pd.DataFrame({"store_id": "T", "article_id": ids})
                weeks = pd.date_range(pd.Timestamp("2023-04-24"), periods=H, freq="7D")
                out = FCF / f"mixture-median-top3__test__final__padded.parquet"
            if point.shape[0] != len(keys):
                print(f"  not saving {tag}: article count {point.shape[0]} != {len(keys)}")
                continue
            df = pd.DataFrame({"store_id": np.repeat(keys["store_id"].to_numpy(), H),
                               "article_id": np.repeat(keys["article_id"].to_numpy(), H),
                               "week": np.tile(np.asarray(weeks), len(keys)),
                               "forecast": np.clip(point, 0, None).reshape(-1)})
            df.to_parquet(out, index=False)
            print(f"  saved {out}")


if __name__ == "__main__":
    main()
