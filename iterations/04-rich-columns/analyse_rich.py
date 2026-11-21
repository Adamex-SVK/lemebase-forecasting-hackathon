"""Analysis for iteration 04 (reference + dropout): score tables, cuts won, paired bootstrap intervals.

Reads a results CSV (arm, cut_index, seed, score, ...) and the per-article MASE detail parquets
written by src/run_rich.py (detail/<arm>_cut<k>_seed<s>.parquet, MultiIndex store_id, article_id,
columns err, n_weeks, scale, mase). Judges only on the 4 rolling cuts.

Paired per-article 95% bootstrap interval of the score difference (arm - reference), per cut:
per-article score contribution = (1 - mase) * 100; average over seeds per article; take the
arm - reference difference per article; bootstrap the mean over articles (1000 resamples).
The scored article set is identical across arms for a cut (same train/holdout), so alignment is exact.

Run examples:
  .venv/bin/python iterations/04-rich-columns/analyse_rich.py \
      --results iterations/04-rich-columns/results_adamex-svk_reference.csv \
      --arms pooled_norich --refs own --seeds 0,1
  .venv/bin/python iterations/04-rich-columns/analyse_rich.py \
      --results iterations/04-rich-columns/results_adamex-svk_dropout.csv \
      --arms pooled_rich_nodrop,pooled_rich_drop50,pooled_rich_drop80 --refs pooled_norich,own --seeds 0,1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
DETAIL = OUT / "detail"


def boot_ci(x, n=1000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_detail(arm, cut, seed):
    return pd.read_parquet(DETAIL / f"{arm}_cut{cut}_seed{seed}.parquet")


def per_article_score(arm, cut, seeds):
    """Mean over seeds of (1 - mase) * 100 per article (index = store_id, article_id)."""
    parts = []
    for s in seeds:
        d = load_detail(arm, cut, s)
        parts.append((1.0 - d["mase"]) * 100.0)
    return pd.concat(parts, axis=1).mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--also", default="", help="comma list of extra results CSVs (e.g. reference arms)")
    ap.add_argument("--arms", required=True, help="comma list of arms to evaluate")
    ap.add_argument("--refs", required=True, help="comma list of references to compare against")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--no-detail", action="store_true", help="skip bootstrap intervals")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    refs = [x for x in a.refs.split(",") if x]
    seeds = [int(x) for x in a.seeds.split(",") if x]

    paths = [a.results] + [p for p in a.also.split(",") if p]
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df.drop_duplicates(["arm", "cut_index", "seed"], keep="last")
    df = df[df["seed"].isin(seeds)].copy()
    cuts = sorted(df["cut_index"].unique())
    all_arms = sorted(df["arm"].unique())

    pd.set_option("display.width", 200)
    print("== scores by arm x cut x seed ==")
    print(df.sort_values(["arm", "cut_index", "seed"])[["arm", "cut_index", "seed", "score", "trees", "seconds"]]
          .to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

    print("\n== mean score by arm x cut (over seeds) ==")
    piv = df.groupby(["arm", "cut_index"])["score"].mean().unstack()
    print(piv.round(2).to_string())

    print("\n== mean over 4 cuts, per seed ==")
    print(df.groupby(["arm", "seed"])["score"].mean().unstack().round(2).to_string())
    print("\n== mean over 4 cuts and all seeds ==")
    print(df.groupby("arm")["score"].mean().round(2).to_string())

    print("\n== seed spread per arm x cut (max-min over seeds) ==")
    spread = df.groupby(["arm", "cut_index"])["score"].agg(lambda s: s.max() - s.min()).unstack()
    print(spread.round(2).to_string())

    for k in cuts:
        print(f"\n== DETAIL cut {k}: paired per-article 95% bootstrap interval (mean diff over articles) ==")
        rows = []
        for arm in arms:
            if arm not in all_arms:
                continue
            sc = per_article_score(arm, k, seeds)
            for ref in refs:
                if ref not in all_arms:
                    continue
                rc = per_article_score(ref, k, seeds)
                idx = sc.index.intersection(rc.index)
                diff = (sc.loc[idx] - rc.loc[idx]).to_numpy()
                lo, hi = boot_ci(diff, 1000, seed=1000 + k)
                score_arm = df[(df.arm == arm) & (df.cut_index == k)]["score"].mean()
                score_ref = df[(df.arm == ref) & (df.cut_index == k)]["score"].mean()
                rows.append({"arm": arm, "ref": ref, "cut": k, "arm_mean": score_arm, "ref_mean": score_ref,
                             "diff": score_arm - score_ref, "ci_lo": lo, "ci_hi": hi,
                             "excludes0": (lo > 0) or (hi < 0), "n_articles": len(idx)})
        if rows:
            print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

    print("\n== cuts won and mean diff over cuts (per arm vs each ref, mean over seeds per cut) ==")
    for arm in arms:
        if arm not in all_arms:
            continue
        for ref in refs:
            if ref not in all_arms:
                continue
            wins, diffs = 0, []
            for k in cuts:
                sa = df[(df.arm == arm) & (df.cut_index == k)]["score"].mean()
                sr = df[(df.arm == ref) & (df.cut_index == k)]["score"].mean()
                diffs.append(sa - sr)
                wins += int(sa > sr)
            print(f"  {arm:20s} vs {ref:14s}: mean over all cut/seed {df[df.arm==arm]['score'].mean():7.2f} "
                  f"vs {df[df.arm==ref]['score'].mean():7.2f}  diff {df[df.arm==arm]['score'].mean()-df[df.arm==ref]['score'].mean():+6.2f}  "
                  f"cuts won {wins}/{len(cuts)}  per-cut diffs {['%+.2f' % d for d in diffs]}")


if __name__ == "__main__":
    main()
