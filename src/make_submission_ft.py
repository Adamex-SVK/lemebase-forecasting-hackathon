"""Submission files from the sibling-shop fine-tuned Chronos-2 (X1c, config: 300 steps, lr 1e-5, 2000 series per shop of Sets A, C, D).

    .venv/bin/python src/make_submission_ft.py

Input: iterations/03-foundation-model/forecasts_oos-ft/chronos-2-ft__test__final__padded.parquet (src/foundation/x1_chronos2_ft.py --final on the GPU).
Writes (zero rule on 8 weeks, round, clip at 0; format validated; uploading is done by a person):
  submissions/chronos2_ft_siblings_v1.csv          the fine-tuned model alone (local 4-cut -13.73, cut 0 -2.73)
  submissions/ensemble_d40_plus_chronos2ft_v1.csv  D40 with it as a 4th ensemble member (local 4-cut -13.65; live -4.5)
  submissions/ensemble_d40_half_chronos2ft_own_v1.csv  D40, half of the foundation part replaced by the 5-seed own-history fine-tune (13-cut +0.24)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import make_submission as ms  # noqa: E402
import make_submission_foundation as mf  # noqa: E402

FT = ROOT / "iterations" / "03-foundation-model" / "forecasts_oos-ft" / "chronos-2-ft__test__final__padded.parquet"


def own_history_avg(ids, dead):
    """5-seed average of the Chronos-2 fine-tuned on siblings + the shop's own full history (x1_chronos2_ft.py --final --with-test 1)."""
    Cs = []
    for sd in range(5):
        P = pd.read_parquet(ROOT / "iterations" / "03-foundation-model" / f"forecasts_oos-ftT1F{sd}" / f"chronos-2-ftT1F{sd}__test__final__padded.parquet").pivot_table(
            index="article_id", columns="week", values="forecast", aggfunc="sum").reindex(ids)
        assert P.shape == (len(ids), 13) and P.notna().all().all()
        Cs.append(P.to_numpy(float))
    C = np.mean(Cs, axis=0); C[dead] = 0.0
    return C


def main():
    ids = ms.history_ids(); dead = mf.dead_mask(ids)
    P = pd.read_parquet(FT).pivot_table(index="article_id", columns="week", values="forecast", aggfunc="sum").reindex(ids)
    assert P.shape == (len(ids), 13) and P.notna().all().all()
    assert P.columns.min() == pd.Timestamp("2023-04-24") and P.columns.max() == pd.Timestamp("2023-07-17")
    C = P.to_numpy(float); C[dead] = 0.0
    L = pd.read_csv(ms.SUB_DIR / "lgbm_own_v1.csv").set_index("article_id").loc[ids].to_numpy(float)
    E3 = np.mean([mf.final_matrix(m, ids) for m in ["moirai-2.0-small", "tirex", "chronos-t5-small"]], axis=0); E3[dead] = 0.0
    d40 = np.clip(np.round(0.3 * L + 0.7 * E3), 0, None)
    Co = own_history_avg(ids, dead)
    half = 0.3 * L + 0.35 * E3 + 0.35 * Co                         # D40 with half of the foundation part replaced (13-cut gain +0.24)
    for name, F in [("ensemble_d40_half_chronos2ft_own_v1.csv", half), ("chronos2_ft_siblings_v1.csv", C), ("ensemble_d40_plus_chronos2ft_v1.csv", 0.3 * L + 0.7 * (3 * E3 + C) / 4)]:
        F = np.clip(np.round(F), 0, None)
        path = ms.SUB_DIR / name; ms.write_submission(F, ids, path); ms.validate_submission(path)
        print(f"{name}: mean forecast {F.mean():.2f}, zero rows {100 * (F.max(axis=1) == 0).mean():.1f}%, mean |diff| vs D40 {np.abs(F - d40).mean():.2f} units")


if __name__ == "__main__":
    main()
