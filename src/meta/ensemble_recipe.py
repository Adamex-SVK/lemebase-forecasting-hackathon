"""Reusable ensemble recipe: evaluate a candidate model set on the 4 standard cuts, then build the final CSV.

    # is a new model (or model set) better than the current recipe?
    .venv/bin/python src/meta/ensemble_recipe.py evaluate --models moirai-2.0-small,tirex,chronos-t5-small,new-model
    # build a validated submission from the accepted model set + weight
    .venv/bin/python src/meta/ensemble_recipe.py build --models moirai-2.0-small,tirex,chronos-t5-small \
                                                        --lgbm-weight 0.3 --out submissions/my_new_file.csv

The blend is always:
    mean(candidate models' raw forecasts)
    -> 0 for articles with no sale in the last 8 weeks  (the zero rule, D39)
    -> lgbm_weight * LightGBM + (1 - lgbm_weight) * ensemble
    -> round to whole units -> clip at 0  (D38)

`evaluate` scores the fixed-weight blend on the same 4 rolling cuts used everywhere (D26/D32) and compares
it with the current reference recipe (D40) on *those* cuts. Adoption needs >= +0.5 points on the mean AND
better on >= 3 of 4 cuts (TEAM-RULES.md section 2); smaller differences are noise.

Inputs:
  * practice cuts  iterations/03-foundation-model/forecasts/{model}__test__cut{0..3}__padded.parquet
  * final weeks    iterations/03-foundation-model/forecasts_final/{model}__test__final__padded.parquet
  * LightGBM        iterations/02-robust-eval/forecasts/base_cut{k}_seed0.npy (cuts) and
                    submissions/lgbm_own_v1.csv (final weeks)
Write a new model's forecasts with src/foundation/run_model.py (see iterations/03-foundation-model/gpu-machine.md).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "foundation"))
import experiments_02 as e  # noqa: E402
import features as f  # noqa: E402
import make_submission as ms  # noqa: E402
import score_all as sa  # noqa: E402
import validate as v  # noqa: E402

H = f.H
FM = ROOT / "iterations" / "03-foundation-model" / "forecasts"
FM_FINAL = ROOT / "iterations" / "03-foundation-model" / "forecasts_final"
REFERENCE_MODELS = ["moirai-2.0-small", "tirex", "chronos-t5-small"]
REFERENCE_W = 0.3
ADOPT_MEAN, ADOPT_CUTS, N_CUTS = 0.5, 3, 4


def parse_models(s):
    models = [m.strip() for m in s.split(",") if m.strip()]
    if not models:
        raise SystemExit("--models is empty")
    return models


# ------------------------------------------------------------------ 4-cut evaluation
def cut_contexts():
    full = v.load_weekly("test")
    ctx = {}
    for k in range(N_CUTS):
        df = e.cut_frame(full, k)
        train, hold, cut = v.split(df)
        M = f.ShopMatrix(df[df["week"] < cut])
        _, _, dead_keys, _ = sa.prepare_cut("test", k)
        dead = np.array([(a, b) in dead_keys for a, b in zip(M.keys["store_id"], M.keys["article_id"])])
        weeks = pd.date_range(cut, periods=H, freq="7D")
        Fl = np.load(ROOT / "iterations" / "02-robust-eval" / "forecasts" / f"base_cut{k}_seed0.npy")
        ctx[k] = (M, train, hold, weeks, dead, Fl)
    return ctx


def raw_practice(model, k, ctx, cache, context="padded"):
    if (model, k, context) not in cache:
        M = ctx[k][0]
        p = FM / f"{model}__test__cut{k}__{context}.parquet"
        if not p.exists():
            raise SystemExit(f"missing practice-cut forecast for {model!r} cut {k}: {p}\n"
                             f"run src/foundation/run_model.py {model} test --cut-index {k} on the GPU machine")
        fc = pd.read_parquet(p)
        cache[(model, k, context)] = fc.pivot_table(index=v.KEYS, columns="week", values="forecast",
                                                    aggfunc="sum").reindex(
            pd.MultiIndex.from_frame(M.keys)).fillna(0.0).to_numpy(float)
    return cache[(model, k, context)]


def score_cut(k, F, ctx):
    M, train, hold, weeks, *_ = ctx[k]
    return e.score_F(M, np.round(F), train, hold, weeks)["score"]


def evaluate_scores(models, w, ctx, cache, context="padded"):
    out = []
    for k in range(N_CUTS):
        M, _, _, _, dead, Fl = ctx[k]
        mod = np.mean([raw_practice(m, k, ctx, cache, context) for m in models], axis=0)
        mod[dead] = 0.0                       # zero rule (D39)
        out.append(score_cut(k, w * Fl + (1 - w) * mod, ctx))
    return np.array(out)


def cmd_evaluate(args):
    models = parse_models(args.models)
    ctx, cache = cut_contexts(), {}
    ref = evaluate_scores(REFERENCE_MODELS, REFERENCE_W, ctx, cache, "padded")
    cand = evaluate_scores(models, args.lgbm_weight, ctx, cache, args.context)
    print(f"reference D40   models={REFERENCE_MODELS} lgbm_weight={REFERENCE_W}")
    print("  by cut: " + " ".join(f"{x:7.2f}" for x in ref) + f"   mean {ref.mean():7.2f}")
    print(f"candidate       models={models} lgbm_weight={args.lgbm_weight} context={args.context}")
    print("  by cut: " + " ".join(f"{x:7.2f}" for x in cand) + f"   mean {cand.mean():7.2f}")
    d = cand - ref
    won = int((d > 0).sum())
    print("  gain  : " + " ".join(f"{x:+7.2f}" for x in d) + f"   mean {d.mean():+.2f}, better on {won}/{N_CUTS}")
    ok = d.mean() >= ADOPT_MEAN and won >= ADOPT_CUTS
    print("verdict:", "ADOPT the candidate -> build it" if ok
          else "keep the current recipe (below the +0.5 mean / 3-of-4 bar)")


# ------------------------------------------------------------------ final submission
def dead_mask(ids):
    h = pd.read_parquet(ROOT / "data" / "interim" / "weekly_test.parquet")
    last8 = h[h["week"] > h["week"].max() - pd.Timedelta(weeks=8)].groupby("article_id")["units"].sum()
    return last8.reindex(ids).fillna(0).to_numpy() == 0


def final_matrix(model, ids, context="padded"):
    p = FM_FINAL / f"{model}__test__final__{context}.parquet"
    if not p.exists():
        raise SystemExit(f"missing final forecast for {model!r}: {p}\n"
                         f"run src/foundation/run_model.py {model} test --final on the GPU machine")
    fc = pd.read_parquet(p)
    P = fc.pivot_table(index="article_id", columns="week", values="forecast", aggfunc="sum").reindex(ids)
    assert P.shape == (len(ids), H) and P.notna().all().all(), f"{model}: final forecast incomplete"
    return P.to_numpy(float)


def cmd_build(args):
    models = parse_models(args.models)
    ids = ms.history_ids()
    dead = dead_mask(ids)
    E = np.mean([final_matrix(m, ids, args.context) for m in models], axis=0)
    E[dead] = 0.0
    if args.lgbm_weight > 0:
        L = pd.read_csv(ms.SUB_DIR / "lgbm_own_v1.csv").set_index("article_id").loc[ids].to_numpy(float)
        F = args.lgbm_weight * L + (1 - args.lgbm_weight) * E
    else:
        F = E
    F = np.clip(np.round(F), 0, None)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    ms.write_submission(F, ids, out)
    ms.validate_submission(out)
    print(f"wrote {out} and it passes the format checks")
    print(f"  models={models} lgbm_weight={args.lgbm_weight} context={args.context} | {len(ids)} articles | "
          f"mean forecast {F.mean():.2f} | {100 * (F.max(axis=1) == 0).mean():.1f}% forecast 0 every week")
    print("  uploading is done by a person (TEAM-RULES.md section 6)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("evaluate", help="score a candidate model set on the 4 standard cuts")
    p.add_argument("--models", required=True, help="comma-separated foundation models")
    p.add_argument("--lgbm-weight", type=float, default=REFERENCE_W)
    p.add_argument("--context", default="padded", choices=["padded", "first_sale", "last52"],
                   help="forecast context variant to read (default: padded, D40)")
    p.set_defaults(func=cmd_evaluate)
    b = sub.add_parser("build", help="build and validate the final submission CSV")
    b.add_argument("--models", required=True)
    b.add_argument("--lgbm-weight", type=float, default=REFERENCE_W)
    b.add_argument("--out", default="submissions/ensemble_recipe.csv")
    b.add_argument("--context", default="padded", choices=["padded", "first_sale", "last52"],
                   help="forecast context variant to read (default: padded, D40)")
    b.set_defaults(func=cmd_build)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
