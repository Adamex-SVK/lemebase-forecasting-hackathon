"""Build submission CSVs from foundation-model forecasts of the REAL future weeks (2023-04-24 .. 2023-07-17).

    .venv/bin/python src/make_submission_foundation.py

Inputs: iterations/03-foundation-model/forecasts_final/{model}__test__final__padded.parquet (written by
src/foundation/run_model.py --final on the GPU machine) and submissions/lgbm_own_v1.csv (LightGBM, all history).
Recipe (validated on the 4 local cuts, see iterations/03-foundation-model/log.md):
    average the models' raw forecasts -> 0 for articles with no sale in the last 8 weeks of history -> optional blend
    with the LightGBM -> round to whole units -> clip at 0.
Uploading is done by a person (docs/TEAM-RULES.md section 6).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import make_submission as ms  # noqa: E402

FIN = ROOT / "iterations" / "03-foundation-model" / "forecasts_final"
H = 13


def final_matrix(model, ids):
    fc = pd.read_parquet(FIN / f"{model}__test__final__padded.parquet")
    P = fc.pivot_table(index="article_id", columns="week", values="forecast", aggfunc="sum").reindex(ids)
    assert P.shape == (len(ids), H) and P.notna().all().all(), f"{model}: forecast missing for some articles or weeks"
    assert P.columns.min() == pd.Timestamp("2023-04-24") and P.columns.max() == pd.Timestamp("2023-07-17")
    return P.to_numpy(float)


def dead_mask(ids):
    h = pd.read_parquet(ROOT / "data" / "interim" / "weekly_test.parquet")
    last8 = h[h["week"] > h["week"].max() - pd.Timedelta(weeks=8)].groupby("article_id")["units"].sum()
    return (last8.reindex(ids).fillna(0).to_numpy() == 0)


def build(models, lgbm_weight, name, ids, dead):
    E = np.mean([final_matrix(m, ids) for m in models], axis=0)
    E[dead] = 0.0
    if lgbm_weight > 0:
        L = pd.read_csv(ms.SUB_DIR / "lgbm_own_v1.csv").set_index("article_id").loc[ids].to_numpy(float)
        F = lgbm_weight * L + (1 - lgbm_weight) * E
    else:
        F = E
    F = np.clip(np.round(F), 0, None)
    path = ms.SUB_DIR / name
    ms.write_submission(F, ids, path)
    ms.validate_submission(path)
    v1 = pd.read_csv(ms.SUB_DIR / "lgbm_own_v1.csv").set_index("article_id").loc[ids].to_numpy(float)
    print(f"{name}: {len(ids)} articles | models {models} | LightGBM weight {lgbm_weight} | mean forecast {F.mean():.2f} "
          f"| articles forecast exactly 0 every week: {100 * (F.max(axis=1) == 0).mean():.1f}% | mean |diff| vs the uploaded lgbm_own_v1 (-7.6 live): {np.abs(F - v1).mean():.2f} units")
    return F


if __name__ == "__main__":
    ids = ms.history_ids()
    dead = dead_mask(ids)
    print(f"{int(dead.sum())} of {len(ids)} articles have no sale in the last 8 weeks of history (forecast 0)")
    build(["moirai-2.0-small"], 0.0, "moirai2_small_v1.csv", ids, dead)
    if all((FIN / f"{m}__test__final__padded.parquet").exists() for m in ["moirai-2.0-small", "tirex", "chronos-t5-small"]):
        build(["moirai-2.0-small", "tirex", "chronos-t5-small"], 0.3, "ensemble_top3_blend_v1.csv", ids, dead)
