"""Build the three upload candidates requested after the clean checks (2026-11-21).

    .venv/bin/python src/meta/build_upload_candidates.py

All files use the real 13 weeks (2023-04-24 .. 2023-07-17), rounded to whole units and clipped at 0:
  A   zero16            mean(moirai-2.0-small, tirex, chronos-t5-small), 16-week zero rule, 30% own LGBM
  C70 zero8 + 30% own   the current D40 recipe (identical to ensemble_top3_blend_v1.csv)
  C55 zero8 + 20% own + 25% cross-shop
                        cross-shop = pooled LightGBM (own + shops A-D, equal weights), trained on all history
                        with src/meta/cross_shop_forecasts.py --final

Uploading is done by a person (TEAM-RULES.md section 6); log it in submissions/UPLOADS.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "meta"))
import ensemble_recipe as er  # noqa: E402
import make_submission as ms  # noqa: E402

H = 13
MODELS = ["moirai-2.0-small", "tirex", "chronos-t5-small"]
CROSS = ROOT / "iterations" / "03-foundation-model" / "forecasts_lgbm_cross" / "pooled_equal__test__final__seed0.parquet"


def dead_mask(ids, weeks):
    h = pd.read_parquet(ROOT / "data" / "interim" / "weekly_test.parquet")
    last = h[h["week"] > h["week"].max() - pd.Timedelta(weeks=weeks)].groupby("article_id")["units"].sum()
    return last.reindex(ids).fillna(0).to_numpy() == 0


def final_matrix(model, ids):
    fc = pd.read_parquet(er.FM_FINAL / f"{model}__test__final__padded.parquet")
    P = fc.pivot_table(index="article_id", columns="week", values="forecast", aggfunc="sum").reindex(ids)
    assert P.shape == (len(ids), H) and P.notna().all().all(), f"{model}: final forecast incomplete"
    return P.to_numpy(float)


def cross_matrix(ids):
    assert CROSS.exists(), f"missing {CROSS}; run src/meta/cross_shop_forecasts.py --final"
    d = pd.read_parquet(CROSS)
    P = d.pivot_table(index="article_id", columns="week", values="forecast", aggfunc="sum").reindex(ids)
    assert P.shape == (len(ids), H) and P.notna().all().all(), "cross-shop: final forecast incomplete"
    return P.to_numpy(float)


def write(F, name, ids, note):
    F = np.clip(np.round(F), 0, None)
    path = ms.SUB_DIR / name
    ms.write_submission(F, ids, path)
    ms.validate_submission(path)
    print(f"{name:42s} mean {F.mean():5.2f} | {100 * (F.max(axis=1) == 0).mean():5.1f}% all-zero | {note}")
    return F


def main():
    ids = ms.history_ids()
    E = np.mean([final_matrix(m, ids) for m in MODELS], axis=0)
    d8, d16 = dead_mask(ids, 8), dead_mask(ids, 16)
    print(f"dead series: 8-week rule {int(d8.sum())}, 16-week rule {int(d16.sum())} of {len(ids)}")
    L = pd.read_csv(ms.SUB_DIR / "lgbm_own_v1.csv").set_index("article_id").loc[ids].to_numpy(float)
    X = cross_matrix(ids)

    E8 = E.copy(); E8[d8] = 0.0
    E16 = E.copy(); E16[d16] = 0.0

    A = write(0.30 * L + 0.70 * E16, "ensemble_zero16_v1.csv", ids,
              "A: 16-week zero rule, 30% own LGBM")
    C70 = write(0.30 * L + 0.70 * E8, "ensemble_c70_own30_v1.csv", ids,
                "C: 70% pretrained + 30% own LGBM (D40)")
    C55 = write(0.20 * L + 0.55 * E8 + 0.25 * X, "ensemble_c55_own20_cross25_v1.csv", ids,
                "C: 55% pretrained + 20% own + 25% cross-shop")

    d40 = pd.read_csv(ms.SUB_DIR / "ensemble_top3_blend_v1.csv").set_index("article_id").loc[ids].to_numpy(float)
    print(f"\nC70 vs ensemble_top3_blend_v1.csv (live -4.8): max |diff| {np.abs(C70 - d40).max():.6f}, "
          f"changed {100 * (C70 != d40).mean():.1f}% of weeks")
    for name, F in [("A zero16", A), ("C55 cross-shop", C55)]:
        print(f"{name:16s} vs C70: mean |diff| {np.abs(F - C70).mean():.3f} units/article-week, "
              f"changed {100 * (F != C70).mean():.1f}% of weeks")


if __name__ == "__main__":
    main()
