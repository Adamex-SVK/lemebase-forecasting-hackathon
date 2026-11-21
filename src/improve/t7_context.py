"""T7: context variants (padded / first_sale / last52) for the D40 three-model blend.

Evaluates the three-member blend under each context and a per-model context chosen
leave-one-cut-out (mixed LOCO). Writes iterations/08-improvements/t7/results_adamex-svk_t7.csv.
Scoring matches ensemble_recipe.evaluate: mean of raw forecasts -> zero rule (D39) ->
0.3*LightGBM + 0.7*ensemble -> round -> clip (via score_cut/e.score_F).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "foundation"))
sys.path.insert(0, str(ROOT / "src" / "meta"))
import ensemble_recipe as er  # noqa: E402

MODELS = er.REFERENCE_MODELS
W = er.REFERENCE_W
CONTEXTS = ["padded", "first_sale", "last52"]
OUTDIR = ROOT / "iterations" / "08-improvements" / "t7"


def solo_scores(ctx, cache):
    """solo[(model, context)] -> 4 cut scores after round + zero rule (no LightGBM)."""
    solo = {}
    for m in MODELS:
        for c in CONTEXTS:
            row = []
            for k in range(er.N_CUTS):
                dead = ctx[k][4]
                F = er.raw_practice(m, k, ctx, cache, c).copy()
                F[dead] = 0.0
                row.append(er.score_cut(k, F, ctx))
            solo[(m, c)] = np.array(row)
    return solo


def blend_scores(per_model_context, ctx, cache):
    """D40 blend with a per-(model, cut) context choice."""
    out = []
    for k in range(er.N_CUTS):
        dead, Fl = ctx[k][4], ctx[k][5]
        mod = np.mean([er.raw_practice(m, k, ctx, cache, per_model_context[(m, k)]) for m in MODELS], axis=0)
        mod[dead] = 0.0
        out.append(er.score_cut(k, W * Fl + (1 - W) * mod, ctx))
    return np.array(out)


def mixed_loco(solo, ctx, cache):
    """Pick each model's context on the other three cuts; score on the held-out cut."""
    chosen, per_cut = {}, {k: {} for k in range(er.N_CUTS)}
    for k in range(er.N_CUTS):
        others = [j for j in range(er.N_CUTS) if j != k]
        for m in MODELS:
            means = {c: float(solo[(m, c)][others].mean()) for c in CONTEXTS}
            best = max(CONTEXTS, key=lambda c: means[c])
            chosen[(m, k)] = best
            per_cut[k][m] = best
    return blend_scores(chosen, ctx, cache), per_cut


def row(variant, sc, ref):
    d = sc - ref
    won = int((d > 0).sum())
    ok = d.mean() >= er.ADOPT_MEAN and won >= er.ADOPT_CUTS
    return dict(variant=variant, cut0=sc[0], cut1=sc[1], cut2=sc[2], cut3=sc[3],
                mean=sc.mean(), gain_vs_D40=d.mean(), cuts_won=won,
                verdict="ADOPT" if ok else "keep D40")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ctx, cache = er.cut_contexts(), {}
    ref = er.evaluate_scores(MODELS, W, ctx, cache, "padded")
    rows = [row("D40_padded(reference)", ref, ref)]
    for c in ["first_sale", "last52"]:
        rows.append(row(c, er.evaluate_scores(MODELS, W, ctx, cache, c), ref))
    solo = solo_scores(ctx, cache)
    msc, per_cut = mixed_loco(solo, ctx, cache)
    rows.append(row("mixed_LOCO", msc, ref))
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "results_adamex-svk_t7.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print("\nmixed LOCO chosen contexts per target cut:")
    for k in range(er.N_CUTS):
        print(" cut", k, {m: per_cut[k][m] for m in MODELS})
    print("\nsolo scores (round + zero8, no LightGBM):")
    for m in MODELS:
        for c in CONTEXTS:
            s = solo[(m, c)]
            print(f"  {m:20s} {c:10s} " + " ".join(f"{x:7.2f}" for x in s) + f"  mean {s.mean():7.2f}")


if __name__ == "__main__":
    main()
