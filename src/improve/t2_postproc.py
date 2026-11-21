"""T2 (iteration 08): post-processing tuning on top of the un-rounded D40 blend.

For each of the 4 standard cuts we rebuild the un-rounded D40 blend exactly as
``src/meta/ensemble_recipe.py`` does it:

    mean(raw forecasts of moirai-2.0-small, tirex, chronos-t5-small)
    -> 0 for articles with no sale in the last 8 weeks (zero rule)
    -> 0.3 * LightGBM + 0.7 * ensemble

and then score one post-processing variant at a time. Parameters are chosen
leave-one-cut-out (for cut K the parameter is picked on the other three cuts);
the in-sample optimum is reported separately and clearly labelled.

Run:  .venv/bin/python src/improve/t2_postproc.py
Writes iterations/08-improvements/t2/results_adamex-svk_t2.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "foundation"))
import experiments_02 as e  # noqa: E402
import features as f  # noqa: E402
import validate as v  # noqa: E402

H = f.H
N_CUTS = 4
REFERENCE_MODELS = ["moirai-2.0-small", "tirex", "chronos-t5-small"]
REFERENCE_W = 0.3
D40_WINDOW = 8
ADOPT_MEAN, ADOPT_CUTS = 0.5, 3
FM = ROOT / "iterations" / "03-foundation-model" / "forecasts"
OUT = ROOT / "iterations" / "08-improvements" / "t2"

# parameter grids from the task brief
OFFSETS = [round(0.30 + 0.05 * i, 2) for i in range(9)]        # 0.30 .. 0.70
SHRINKS = [round(0.80 + 0.05 * i, 2) for i in range(6)]        # 0.80 .. 1.05
WINDOWS = [4, 6, 8, 10, 13, 26]
SMALL_CUTS = [0.5, 0.75, 1.0]
SEG_NAMES = ["mean13<1", "1<=mean13<=5", "mean13>5"]


# ------------------------------------------------------------------ inputs
def load_cut_contexts():
    """Rebuild exactly the context used by ensemble_recipe.evaluate."""
    full = v.load_weekly("test")
    ctx = {}
    for k in range(N_CUTS):
        df = e.cut_frame(full, k)
        train, hold, cut = v.split(df)
        M = f.ShopMatrix(df[df["week"] < cut])
        weeks = pd.date_range(cut, periods=H, freq="7D")
        Fl = np.load(ROOT / "iterations" / "02-robust-eval" / "forecasts" / f"base_cut{k}_seed0.npy")
        raws = {}
        for m in REFERENCE_MODELS:
            p = FM / f"{m}__test__cut{k}__padded.parquet"
            fc = pd.read_parquet(p)
            raws[m] = fc.pivot_table(index=v.KEYS, columns="week", values="forecast",
                                     aggfunc="sum").reindex(
                pd.MultiIndex.from_frame(M.keys)).fillna(0.0).to_numpy(float)
        ctx[k] = {"M": M, "train": train, "hold": hold, "weeks": weeks, "Fl": Fl, "raws": raws}
    return ctx


def dead_mask(M, window):
    """No sale in the last `window` training weeks (before the holdout)."""
    w = min(window, M.W)
    return M.U[:, -w:].sum(axis=1) == 0


def ensemble_part(ctx, k, window):
    r = ctx[k]["raws"]
    mod = np.mean([r[m] for m in REFERENCE_MODELS], axis=0)
    mod[dead_mask(ctx[k]["M"], window)] = 0.0
    return mod


def blend(ctx, k, window, lgbm_w=REFERENCE_W):
    """Un-rounded D40-style blend for cut k with a given zero-rule window."""
    return lgbm_w * ctx[k]["Fl"] + (1.0 - lgbm_w) * ensemble_part(ctx, k, window)


def score_forecast(ctx, k, F):
    c = ctx[k]
    return e.score_F(c["M"], F, c["train"], c["hold"], c["weeks"])


# ------------------------------------------------------------------ variants
def apply_offset(F, c):
    return np.clip(np.floor(F + c), 0, None)


def apply_shrink(F, a):
    return np.clip(np.round(a * F), 0, None)


def apply_small_cut(F, t):
    G = F.copy()
    G[G < t] = 0.0
    return np.clip(np.round(G), 0, None)


def segment_labels(ctx, k):
    M = ctx[k]["M"]
    mean13 = M.U[:, -min(13, M.W):].mean(axis=1)
    lab = np.zeros(M.n, dtype=int)
    lab[(mean13 >= 1.0) & (mean13 <= 5.0)] = 1
    lab[mean13 > 5.0] = 2
    return lab


def apply_segment_offset(ctx, k, cvec):
    F = blend(ctx, k, D40_WINDOW)
    return np.clip(np.floor(F + cvec[:, None]), 0, None)


# ------------------------------------------------------------------ honest selection
def mean_of(scores, cuts):
    return float(np.mean([scores[j] for j in cuts]))


def loocv_pick(scores_by_param, params):
    """For each cut pick the param with the best mean on the other 3 cuts."""
    honest, picks = [], []
    for k in range(N_CUTS):
        others = [j for j in range(N_CUTS) if j != k]
        best = max(params, key=lambda p: mean_of(scores_by_param[p], others))
        picks.append(best)
        honest.append(scores_by_param[best][k])
    return np.array(honest, dtype=float), picks


def row_from_scores(name, kind, param, scores, ref, extra=None):
    scores = np.asarray(scores, dtype=float)
    gain = scores.mean() - ref.mean()
    won = int((scores > ref).sum())
    verdict = "ADOPT" if (gain >= ADOPT_MEAN and won >= ADOPT_CUTS) else "reject"
    row = {"variant": name, "kind": kind, "param": param,
           "cut0": round(float(scores[0]), 4), "cut1": round(float(scores[1]), 4),
           "cut2": round(float(scores[2]), 4), "cut3": round(float(scores[3]), 4),
           "mean": round(float(scores.mean()), 4), "gain_vs_D40": round(float(gain), 4),
           "cuts_won": won, "verdict": verdict}
    if extra:
        row.update(extra)
    return row


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = load_cut_contexts()

    # D40 reference exactly as ensemble_recipe.evaluate scores it (np.round, no extra clip)
    d40 = np.array([score_forecast(ctx, k, np.clip(np.round(blend(ctx, k, D40_WINDOW)), 0, None))["score"]
                    for k in range(N_CUTS)])
    d40_plain = np.array([score_forecast(ctx, k, np.round(blend(ctx, k, D40_WINDOW)))["score"]
                          for k in range(N_CUTS)])
    print("reproduced D40 by cut:", " ".join(f"{x:7.2f}" for x in d40), f" mean {d40.mean():7.2f}")
    if not np.allclose(d40, d40_plain):
        print("  (clip changed D40: no-clip mean", f"{d40_plain.mean():.2f})")
    rows = [row_from_scores("D40 (reference)", "reference", "round, W=8",
                            d40, d40)]

    # ---- 1. rounding offset floor(x + c)
    off_scores = {c: np.array([score_forecast(ctx, k, apply_offset(blend(ctx, k, D40_WINDOW), c))["score"]
                               for k in range(N_CUTS)]) for c in OFFSETS}
    for c in OFFSETS:
        rows.append(row_from_scores(f"offset floor(x+{c:.2f})", "in_sample_grid", f"c={c:.2f}",
                                    off_scores[c], d40))
    hon, picks = loocv_pick(off_scores, OFFSETS)
    rows.append(row_from_scores("offset (honest LOOCV)", "honest_loocv",
                                "picks " + "/".join(f"{p:.2f}" for p in picks), hon, d40))
    cbest = max(OFFSETS, key=lambda c: off_scores[c].mean())
    rows.append(row_from_scores(f"offset in-sample optimum c={cbest:.2f}", "in_sample_opt",
                                f"c={cbest:.2f}", off_scores[cbest], d40))

    # ---- 2. shrink before rounding round(a * x)
    shr_scores = {a: np.array([score_forecast(ctx, k, apply_shrink(blend(ctx, k, D40_WINDOW), a))["score"]
                               for k in range(N_CUTS)]) for a in SHRINKS}
    for a in SHRINKS:
        rows.append(row_from_scores(f"shrink round({a:.2f}*x)", "in_sample_grid", f"a={a:.2f}",
                                    shr_scores[a], d40))
    hon, picks = loocv_pick(shr_scores, SHRINKS)
    rows.append(row_from_scores("shrink (honest LOOCV)", "honest_loocv",
                                "picks " + "/".join(f"{p:.2f}" for p in picks), hon, d40))
    abest = max(SHRINKS, key=lambda a: shr_scores[a].mean())
    rows.append(row_from_scores(f"shrink in-sample optimum a={abest:.2f}", "in_sample_opt",
                                f"a={abest:.2f}", shr_scores[abest], d40))

    # ---- 3. zero-rule window W
    win_scores = {W: np.array([score_forecast(ctx, k, np.clip(np.round(blend(ctx, k, W)), 0, None))["score"]
                               for k in range(N_CUTS)]) for W in WINDOWS}
    for W in WINDOWS:
        rows.append(row_from_scores(f"zero window W={W}", "in_sample_grid", f"W={W}",
                                    win_scores[W], d40))
    hon, picks = loocv_pick(win_scores, WINDOWS)
    rows.append(row_from_scores("zero window (honest LOOCV)", "honest_loocv",
                                "picks " + "/".join(str(p) for p in picks), hon, d40))
    Wbest = max(WINDOWS, key=lambda W: win_scores[W].mean())
    rows.append(row_from_scores(f"zero window in-sample optimum W={Wbest}", "in_sample_opt",
                                f"W={Wbest}", win_scores[Wbest], d40))

    # ---- 4. small-value cut t (on the D40 blend, before rounding)
    cut_scores = {t: np.array([score_forecast(ctx, k, apply_small_cut(blend(ctx, k, D40_WINDOW), t))["score"]
                               for k in range(N_CUTS)]) for t in SMALL_CUTS}
    for t in SMALL_CUTS:
        rows.append(row_from_scores(f"small cut t={t}", "in_sample_grid", f"t={t}",
                                    cut_scores[t], d40))
    hon, picks = loocv_pick(cut_scores, SMALL_CUTS)
    rows.append(row_from_scores("small cut (honest LOOCV)", "honest_loocv",
                                "picks " + "/".join(f"{p}" for p in picks), hon, d40))
    tbest = max(SMALL_CUTS, key=lambda t: cut_scores[t].mean())
    rows.append(row_from_scores(f"small cut in-sample optimum t={tbest}", "in_sample_opt",
                                f"t={tbest}", cut_scores[tbest], d40))

    # ---- 5. per-segment offset: three parameters c for three mean13 segments.
    # The metric is a mean of per-article MASE, so segment contributions are
    # separable: a segment's c can be chosen on its own mean MASE over the other cuts.
    seg = {k: segment_labels(ctx, k) for k in range(N_CUTS)}
    seg_mase = {k: {} for k in range(N_CUTS)}
    for k in range(N_CUTS):
        for c in OFFSETS:
            r = score_forecast(ctx, k, apply_offset(blend(ctx, k, D40_WINDOW), c))
            d = r["detail"]["mase"]
            lab = pd.Series(seg[k], index=pd.MultiIndex.from_frame(ctx[k]["M"].keys)).reindex(d.index)
            seg_mase[k][c] = {s: float(d[lab == s].mean()) for s in range(3)}
    # honest: for each cut pick each segment's c on the other three cuts
    seg_honest = np.zeros(N_CUTS)
    seg_picks = []
    for k in range(N_CUTS):
        others = [j for j in range(N_CUTS) if j != k]
        cs = []
        for s in range(3):
            c_s = max(OFFSETS, key=lambda c: np.mean([seg_mase[j][c][s] for j in others]))
            cs.append(c_s)
        seg_picks.append(cs)
        cvec = np.array([cs[s] for s in seg[k]])
        seg_honest[k] = score_forecast(ctx, k, apply_segment_offset(ctx, k, cvec))["score"]
    rows.append(row_from_scores("per-segment offset (honest LOOCV)", "honest_loocv",
                                "picks " + " | ".join("/".join(f"{c:.2f}" for c in cs) for cs in seg_picks),
                                seg_honest, d40))
    # in-sample optimum triple
    cs_ins = [max(OFFSETS, key=lambda c: np.mean([seg_mase[k][c][s] for k in range(N_CUTS)]))
              for s in range(3)]
    seg_ins = np.array([score_forecast(ctx, k, apply_segment_offset(
        ctx, k, np.array([cs_ins[s] for s in seg[k]])))["score"] for k in range(N_CUTS)])
    rows.append(row_from_scores("per-segment offset in-sample optimum", "in_sample_opt",
                                "c=" + "/".join(f"{c:.2f}" for c in cs_ins), seg_ins, d40))

    res = pd.DataFrame(rows)
    path = OUT / "results_adamex-svk_t2.csv"
    res.to_csv(path, index=False)

    pd.set_option("display.width", 250)
    print("\nHonest (leave-one-cut-out) table vs D40:")
    show = res[res["kind"].isin(["reference", "honest_loocv"])].copy()
    print(show.to_string(index=False))
    print("\nIn-sample optima (NOT honest; upper bound only):")
    print(res[res["kind"] == "in_sample_opt"].to_string(index=False))
    print("\nFull parameter grid is in", path)
    print(f"\nruntime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
