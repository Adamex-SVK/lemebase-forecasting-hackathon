"""Check candidate global blend weights on the standard 4 non-overlapping cuts (the honest yardstick).

    .venv/bin/python src/meta/eval_global_weights_4cut.py

The FFORMA meta-learner (src/meta/fforma.py) only learns near-global weights on the overlapping
cuts. This script asks whether any of those global weightings beats the current recipe (D40: 0.30 *
LightGBM + equal average of Moirai 2.0-small / TiRex / Chronos-T5-small, zero rule on the models,
rounded) on the 4 standard cuts of iterations 01-03. Adoption needs >= +0.5 points on the mean AND
better on >= 3 of 4 cuts (TEAM-RULES.md section 2); anything smaller is noise.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "foundation"))
import experiments_02 as e  # noqa: E402
import features as f  # noqa: E402
import score_all as sa  # noqa: E402
import validate as v  # noqa: E402

FM_DIR = ROOT / "iterations" / "03-foundation-model" / "forecasts"
H = f.H
TOP3 = ["moirai-2.0-small", "tirex", "chronos-t5-small"]
SCHEMES = {
    "D40 (0.30 + equal top3)": (0.30, {m: 0.70 / 3 for m in TOP3}),
    "0.20 + equal top3": (0.20, {m: 0.80 / 3 for m in TOP3}),
    "0.40 + equal top3": (0.40, {m: 0.60 / 3 for m in TOP3}),
    "0.00 + equal top3": (0.00, {m: 1.0 / 3 for m in TOP3}),
    "FFORMA-global 3 (0.34/moirai .24/tirex .25/chronos .17)":
        (0.34, {"moirai-2.0-small": 0.24, "tirex": 0.25, "chronos-t5-small": 0.17}),
    "FFORMA-global 5 (0.25 + 5 models)":
        (0.25, {"moirai-2.0-small": 0.17, "tirex": 0.18, "chronos-t5-small": 0.13,
                "chronos-t5-base": 0.12, "moirai-1.1-base": 0.15}),
}


def main():
    full = v.load_weekly("test")
    ctx = {}
    for k in range(4):
        df = e.cut_frame(full, k)
        train, hold, cut = v.split(df)
        M = f.ShopMatrix(df[df["week"] < cut])
        _, _, dead_keys, _ = sa.prepare_cut("test", k)
        dead = np.array([(a, b) in dead_keys for a, b in zip(M.keys["store_id"], M.keys["article_id"])])
        weeks = pd.date_range(cut, periods=H, freq="7D")
        Fl = np.load(ROOT / "iterations" / "02-robust-eval" / "forecasts" / f"base_cut{k}_seed0.npy")
        ctx[k] = (M, train, hold, weeks, dead, Fl)

    cache = {}

    def raw(model, k):
        if (model, k) not in cache:
            M = ctx[k][0]
            fc = pd.read_parquet(FM_DIR / f"{model}__test__cut{k}__padded.parquet")
            cache[(model, k)] = fc.pivot_table(index=v.KEYS, columns="week", values="forecast",
                                               aggfunc="sum").reindex(
                pd.MultiIndex.from_frame(M.keys)).fillna(0.0).to_numpy(float)
        return cache[(model, k)]

    def score(k, F):
        M, train, hold, weeks, *_ = ctx[k]
        return e.score_F(M, np.round(F), train, hold, weeks)["score"]

    print(f"4 standard cuts; models available: {sorted({p.stem.split('__')[0] for p in FM_DIR.glob('*__test__cut0__padded.parquet')})}\n")
    out = {}
    for name, (wl, wm) in SCHEMES.items():
        sc = []
        for k in range(4):
            M, _, _, _, dead, Fl = ctx[k]
            mod = np.zeros((M.n, H))
            for m, w in wm.items():
                mod += w * raw(m, k)
            mod[dead] = 0.0                     # zero rule on the model part (as in make_submission_foundation)
            F = wl * Fl + mod
            sc.append(score(k, F))
        out[name] = sc
        print(f"{name:52s} " + " ".join(f"{x:7.2f}" for x in sc) + f"   mean {np.mean(sc):7.2f}")

    ref = np.array(out["D40 (0.30 + equal top3)"])
    print(f"\nvs D40 (mean {ref.mean():.2f}):")
    for name, sc in out.items():
        if name.startswith("D40"):
            continue
        d = np.array(sc) - ref
        won = int((d > 0).sum())
        adopt = d.mean() >= 0.5 and won >= 3
        print(f"  {name:52s} gain {d.mean():+.2f}, better on {won}/4 -> "
              f"{'ADOPT' if adopt else 'keep D40'}")


if __name__ == "__main__":
    main()
