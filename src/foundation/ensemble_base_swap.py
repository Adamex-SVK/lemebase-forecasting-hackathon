"""Swap the LightGBM in the winning blend (top3 foundation ensemble + LightGBM, rounded) for the new bases from part 2.

    .venv/bin/python src/foundation/ensemble_base_swap.py

Winning blend (ensemble_eval.py, live -4.8): w * LightGBM + (1 - w) * mean(moirai-2.0-small, tirex, chronos-t5-small)
with the 8-week zero rule on the ensemble, whole-unit rounding of the blend. Here the LightGBM term is replaced by:
  lgbm            seed-0 forecasts as in ensemble_eval (reproduces -13.67 at the chosen weight)
  lgbm 3-seed     average of 3 seeds (loss_F files)
  lgbm+gate       3-seed average with the zero gate (P < 0.5 -> 0)
  lgbm+med+gate   gate((3-seed lgbm + median13) / 2)
  blend+gate      gate applied to the final blend instead (base = 3-seed lgbm)
Weight on the base chosen on cuts 1-3 (as before); also shown: fixed w = 0.3 (the submitted file) and the mean by weight.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "foundation"))
import experiments_02 as e, features as f, score_all as sa, validate as v  # noqa: E401
FM_DIR = ROOT / "iterations" / "03-foundation-model" / "forecasts"
FC = ROOT / "iterations" / "02-robust-eval" / "forecasts"
MODELS = ["moirai-2.0-small", "tirex", "chronos-t5-small"]
H, W = f.H, np.round(np.arange(0, 1.01, 0.1), 1)
full = v.load_weekly("test")
ctx = {}
for k in range(4):
    df = e.cut_frame(full, k); train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut])
    _, _, dead_keys, _ = sa.prepare_cut("test", k)
    dead = np.array([(a, b) in dead_keys for a, b in zip(M.keys["store_id"], M.keys["article_id"])])
    weeks = pd.date_range(cut, periods=H, freq="7D")
    idx = pd.MultiIndex.from_frame(M.keys)
    E = np.mean([pd.read_parquet(FM_DIR / f"{m}__test__cut{k}__padded.parquet").pivot_table(index=v.KEYS, columns="week", values="forecast", aggfunc="sum").reindex(idx).fillna(0.0).to_numpy(float) for m in MODELS], axis=0)
    E[dead] = 0.0
    L1 = np.load(FC / f"base_cut{k}_seed0.npy")
    L3 = np.mean([np.load(FC / f"loss_F_cut{k}_seed{s}.npy") for s in range(3)], 0)
    Pr = np.mean([np.load(FC / f"loss_P_cut{k}_seed{s}.npy") for s in range(3)], 0)
    med = np.repeat(np.median(M.U[:, -13:], axis=1, keepdims=True), H, axis=1).astype(float)
    gate = lambda F, Pr=Pr: np.where(Pr < 0.5, 0.0, F)
    bases = {"lgbm (seed 0, as submitted)": L1, "lgbm 3-seed": L3, "lgbm+gate": gate(L3), "lgbm+med+gate": gate((L3 + med) / 2)}
    ctx[k] = (M, train, hold, weeks, E, bases, L3, gate)
def sc(k, F):
    M, train, hold, weeks = ctx[k][:4]
    return e.score_F(M, np.round(F), train, hold, weeks)["score"]
print("ensemble alone (rounded):", " ".join(f"{sc(k, ctx[k][4]):7.2f}" for k in range(4)))
names = list(ctx[0][5]) + ["3-seed lgbm, gate on final blend"]
res = {}
for n in names:
    def blend(k, w):
        M, _, _, _, E, bases, L3, gate = ctx[k]
        if n in bases: return w * bases[n] + (1 - w) * E
        return gate(w * L3 + (1 - w) * E)
    S = {w: [sc(k, blend(k, w)) for k in range(4)] for w in W}
    wb = max(W, key=lambda w: np.mean(S[w][1:]))
    res[n] = (wb, S)
    print(f"\n{n}\n  w={wb} (chosen on cuts 1-3): " + " ".join(f"{x:7.2f}" for x in S[wb]) + f"  mean {np.mean(S[wb]):.2f}")
    print(f"  fixed w=0.3:               " + " ".join(f"{x:7.2f}" for x in S[0.3]) + f"  mean {np.mean(S[0.3]):.2f}")
    print("  mean by weight on base: " + "  ".join(f"{w:.1f}:{np.mean(S[w]):.2f}" for w in W[::2]))
ref = res["lgbm (seed 0, as submitted)"]
print("\nGain over the submitted recipe (same weight rule), per cut and cuts won:")
for n in names[1:]:
    d = [a - b for a, b in zip(res[n][1][res[n][0]], ref[1][ref[0]])]
    print(f"  {n:36s} " + " ".join(f"{x:+6.2f}" for x in d) + f"  mean {np.mean(d):+.2f}  won {sum(x > 0 for x in d)}/4")
