"""The four clean checks requested before an upload, on the 4 rolling test-shop cuts (D26/D32).

    .venv/bin/python src/meta/clean_checks.py

Baseline: the current winning blend D40 = mean(moirai-2.0-small, tirex, chronos-t5-small) raw forecasts,
8-week zero rule on the ensemble, 0.3 x own LightGBM + 0.7 x ensemble, rounded (live -4.8).

Test A  no rounding and/or a 16-week zero rule instead of 8 weeks.
Test B  chronos-t5-small -> chronos-2, models weighted by their solo local validation strength
        (inverse-MASE, and inverse-|score| as a sensitivity).
Test C  70/30 pretrained/own LightGBM versus 55/20/25 pretrained/own/cross-shop LightGBM; the cross-shop
        arm is `src/meta/cross_shop_forecasts.py` (own + shops A-D, pooled, equal weights).

Writes iterations/03-foundation-model/results_kilo_clean_checks.csv.
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
import experiments_02 as e  # noqa: E402
import validate as v  # noqa: E402

H = e.H
CROSS = ROOT / "iterations" / "03-foundation-model" / "forecasts_lgbm_cross"
OUT = ROOT / "iterations" / "03-foundation-model" / "results_kilo_clean_checks.csv"
D40_MODELS = ["moirai-2.0-small", "tirex", "chronos-t5-small"]
LGBM_W = 0.3

ctx = er.cut_contexts()
cache = {}


def raw(model, k):
    return er.raw_practice(model, k, ctx, cache)


def dead_mask(k, weeks):
    """Series with no sale in the last `weeks` weeks of that cut's training history."""
    return ctx[k][0].U[:, -weeks:].sum(axis=1) == 0.0


def score(k, F, rounded=True):
    M, train, hold, weeks, *_ = ctx[k]
    return e.score_F(M, np.round(F) if rounded else F, train, hold, weeks)["score"]


def ensemble(k, models, weights=None):
    mats = [raw(m, k) for m in models]
    if weights is None:
        return np.mean(mats, axis=0)
    return np.average(np.stack(mats), axis=0, weights=np.asarray(weights, float))


def blend(k, P, lgbm_weight=LGBM_W, extra=None, extra_weight=0.0):
    """lgbm_weight * own LightGBM + remainder * P + (extra_weight * cross-shop)."""
    L = ctx[k][5]
    if extra is None:
        return lgbm_weight * L + (1 - lgbm_weight) * P
    return lgbm_weight * L + (1 - lgbm_weight - extra_weight) * P + extra_weight * extra


def cross_shop(k):
    p = CROSS / f"pooled_equal__test__cut{k}__seed0.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    return d.pivot_table(index=v.KEYS, columns="week", values="forecast", aggfunc="sum").reindex(
        pd.MultiIndex.from_frame(ctx[k][0].keys)).fillna(0.0).to_numpy(float)


rows = []


def add(name, by_cut, note=""):
    rows.append({"check": name, "cut0": by_cut[0], "cut1": by_cut[1], "cut2": by_cut[2], "cut3": by_cut[3],
                 "mean": float(np.mean(by_cut)), "note": note})
    print(f"{name:62s} " + " ".join(f"{x:8.2f}" for x in by_cut) + f"   mean {np.mean(by_cut):8.2f}")


# ---------------------------------------------------------------- baseline
P8 = {}
for k in range(4):
    P = ensemble(k, D40_MODELS)
    P[dead_mask(k, 8)] = 0.0
    P8[k] = P
base = [score(k, blend(k, P8[k])) for k in range(4)]
add("baseline D40: 8-week zero, rounded, 0.3 own LGBM", base, "the current winning recipe")

# ---------------------------------------------------------------- Test A
A16 = [score(k, blend(k, np.where(dead_mask(k, 16)[:, None], 0.0, ensemble(k, D40_MODELS)))) for k in range(4)]
add("A1 16-week zero rule, rounded", A16, "zero rule relaxed 8 -> 16 weeks")
A_raw8 = [score(k, blend(k, P8[k]), rounded=False) for k in range(4)]
add("A2 no rounding, 8-week zero rule", A_raw8, "final blend kept fractional")
A_raw16 = []
for k in range(4):
    P = ensemble(k, D40_MODELS)
    P[dead_mask(k, 16)] = 0.0
    A_raw16.append(score(k, blend(k, P), rounded=False))
add("A3 no rounding + 16-week zero rule", A_raw16, "both changes together")

# ---------------------------------------------------------------- Test B
TOP = ["moirai-2.0-small", "tirex", "chronos-2"]
solo = {}
for m in TOP:
    solo[m] = float(np.mean([score(k, np.where(dead_mask(k, 8)[:, None], 0.0, raw(m, k))) for k in range(4)]))
print("\nsolo local scores (rounded + 8-week zero): " + ", ".join(f"{m} {solo[m]:.2f}" for m in TOP))
mase = {m: 1 - solo[m] / 100.0 for m in TOP}
w_imase = np.array([1 / mase[m] for m in TOP]); w_imase /= w_imase.sum()
w_iabs = np.array([1 / abs(solo[m]) for m in TOP]); w_iabs /= w_iabs.sum()
print("weights inverse-MASE: " + ", ".join(f"{m} {w:.3f}" for m, w in zip(TOP, w_imase)))
print("weights inverse-|score|: " + ", ".join(f"{m} {w:.3f}" for m, w in zip(TOP, w_iabs)))

B_eq = []
B_imase = []
B_iabs = []
for k in range(4):
    d = dead_mask(k, 8)
    Eeq = ensemble(k, TOP); Eeq[d] = 0.0
    Eim = ensemble(k, TOP, weights=w_imase); Eim[d] = 0.0
    Eia = ensemble(k, TOP, weights=w_iabs); Eia[d] = 0.0
    B_eq.append(score(k, blend(k, Eeq)))
    B_imase.append(score(k, blend(k, Eim)))
    B_iabs.append(score(k, blend(k, Eia)))
add("B1 chronos-2 swap, equal weights", B_eq, "chronos-t5-small -> chronos-2")
add("B2 chronos-2 swap, inverse-MASE weights", B_imase,
    "weights " + ", ".join(f"{m} {w:.2f}" for m, w in zip(TOP, w_imase)))
add("B3 chronos-2 swap, inverse-|score| weights", B_iabs,
    "weights " + ", ".join(f"{m} {w:.2f}" for m, w in zip(TOP, w_iabs)))

# ---------------------------------------------------------------- Test C
X = [cross_shop(k) for k in range(4)]
if any(x is None for x in X):
    print("\nTest C pending: cross-shop forecasts missing; run src/meta/cross_shop_forecasts.py")
else:
    own_alone = [score(k, ctx[k][5]) for k in range(4)]
    cross_alone = [score(k, X[k]) for k in range(4)]
    add("C0 own LightGBM alone (rounded)", own_alone, "own shops only")
    add("C0 cross-shop LightGBM alone (rounded)", cross_alone, "own + shops A-D, equal weights")
    c70 = [score(k, blend(k, P8[k], lgbm_weight=0.3)) for k in range(4)]
    c55 = [score(k, blend(k, P8[k], lgbm_weight=0.20, extra=X[k], extra_weight=0.25)) for k in range(4)]
    c55z = []
    for k in range(4):
        xz = X[k].copy(); xz[dead_mask(k, 8)] = 0.0
        c55z.append(score(k, blend(k, P8[k], lgbm_weight=0.20, extra=xz, extra_weight=0.25)))
    add("C1 70% pretrained + 30% own LGBM", c70, "baseline weights")
    add("C2 55% pretrained + 20% own + 25% cross-shop", c55, "cross-shop as-is")
    add("C3 55% pretrained + 20% own + 25% cross-shop (zero8)", c55z, "8-week zero rule also on cross-shop")

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"\nwrote {OUT}")
