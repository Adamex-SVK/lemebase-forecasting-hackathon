"""T3: robust combination of the foundation-model members (iteration 08).

Per standard cut K (the last 13*K weeks removed), combine the raw member forecasts with
fixed, parameter-free rules and keep the rest of D40 unchanged:

    combine members -> 0 for articles with no sale in the last 8 weeks (zero rule)
    -> 0.3 * LightGBM + 0.7 * combination -> round -> clip at 0

Tested combinations (no tuning on the scored cuts):
    mean            D40 baseline (plain average)
    median          per-cell median of the members
    trimmed mean    drop the min and max member per cell when >= 4 members, else mean
    geometric       expm1(mean(log1p(member))) -- geometric-style mean of log1p
Additionally, the same combinations are run with the LightGBM added as a member and the
0.3 blend removed (the combination output is used directly).

Run: .venv/bin/python src/improve/t3_combine.py
Outputs: iterations/08-improvements/t3/results_adamex-svk_t3.csv
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

H = f.H
FM = ROOT / "iterations" / "03-foundation-model" / "forecasts"
LGBM = ROOT / "iterations" / "02-robust-eval" / "forecasts"
OUT = ROOT / "iterations" / "08-improvements" / "t3"
REF3 = ["moirai-2.0-small", "tirex", "chronos-t5-small"]
FIVE = REF3 + ["chronos-t5-base", "moirai-1.1-base"]
W = 0.3                       # fixed LightGBM weight, D40 (no tuning)
N_CUTS = 4
ADOPT_MEAN, ADOPT_CUTS = 0.5, 3


# ------------------------------------------------------------------ combination rules
def comb_mean(A):
    return A.mean(axis=0)


def comb_median(A):
    return np.median(A, axis=0)


def comb_trimmed(A):
    if A.shape[0] < 4:            # trimming needs at least 4 members; fall back to mean
        return A.mean(axis=0)
    S = np.sort(A, axis=0)
    return S[1:-1].mean(axis=0)   # drop the min and the max per cell


def comb_geometric(A):
    return np.expm1(np.mean(np.log1p(np.clip(A, 0.0, None)), axis=0))


COMBOS = {
    "mean": comb_mean,
    "median": comb_median,
    "trimmed": comb_trimmed,
    "geometric": comb_geometric,
}


# ------------------------------------------------------------------ evaluation plumbing
def cut_contexts():
    full = v.load_weekly("test")
    ctx = {}
    for k in range(N_CUTS):
        df = e.cut_frame(full, k)
        train, hold, cut = v.split(df)
        M = f.ShopMatrix(df[df["week"] < cut])
        _, _, dead_keys, _ = sa.prepare_cut("test", k)
        dead = np.array([(a, b) in dead_keys for a, b in
                         zip(M.keys["store_id"], M.keys["article_id"])])
        weeks = pd.date_range(cut, periods=H, freq="7D")
        Fl = np.load(LGBM / f"base_cut{k}_seed0.npy")
        ctx[k] = (M, train, hold, weeks, dead, Fl)
    return ctx


def raw_practice(model, k, ctx, cache):
    key = (model, k)
    if key not in cache:
        M = ctx[k][0]
        p = FM / f"{model}__test__cut{k}__padded.parquet"
        if not p.exists():
            raise SystemExit(f"missing forecast for {model!r} cut {k}: {p}")
        fc = pd.read_parquet(p)
        cache[key] = fc.pivot_table(index=v.KEYS, columns="week", values="forecast",
                                    aggfunc="sum").reindex(
            pd.MultiIndex.from_frame(M.keys)).fillna(0.0).to_numpy(float)
    return cache[key]


def score_cut(k, F, ctx):
    M, train, hold, weeks, *_ = ctx[k]
    return e.score_F(M, np.round(F), train, hold, weeks)["score"]


def evaluate_recipe(key, members, with_lgbm_member, ctx, cache):
    """Return scores per cut for one combination rule over one member pool."""
    rule = COMBOS[key]
    out = []
    for k in range(N_CUTS):
        M, _, _, _, dead, Fl = ctx[k]
        mats = [raw_practice(m, k, ctx, cache) for m in members]
        if with_lgbm_member:
            mats.append(Fl)
        C = rule(np.stack(mats, axis=0))
        C[dead] = 0.0                                   # zero rule (D39)
        F = C if with_lgbm_member else W * Fl + (1 - W) * C
        out.append(score_cut(k, F, ctx))
    return np.array(out)


# ------------------------------------------------------------------ recipes under test
def recipes():
    rows = []
    # fixed 0.3 LightGBM blend, D40 members plus the 5-member pool
    for key in ("mean", "median", "geometric"):
        rows.append(dict(mode="fixed_blend", combination=key, members=",".join(REF3),
                         n_members=3, key=key, pool=REF3, lgbm_member=False))
    # median of 5 = median over the five available models
    rows.append(dict(mode="fixed_blend", combination="median", members=",".join(FIVE),
                     n_members=5, key="median", pool=FIVE, lgbm_member=False))
    rows.append(dict(mode="fixed_blend", combination="trimmed", members=",".join(FIVE),
                     n_members=5, key="trimmed", pool=FIVE, lgbm_member=False))
    rows.append(dict(mode="fixed_blend", combination="mean", members=",".join(FIVE),
                     n_members=5, key="mean", pool=FIVE, lgbm_member=False))
    rows.append(dict(mode="fixed_blend", combination="geometric", members=",".join(FIVE),
                     n_members=5, key="geometric", pool=FIVE, lgbm_member=False))
    # LightGBM as a member, no fixed blend
    for key in COMBOS:
        rows.append(dict(mode="lgbm_member", combination=key, members=",".join(REF3) + ",LightGBM",
                         n_members=4, key=key, pool=REF3, lgbm_member=True))
        rows.append(dict(mode="lgbm_member", combination=key, members=",".join(FIVE) + ",LightGBM",
                         n_members=6, key=key, pool=FIVE, lgbm_member=True))
    return rows


def main():
    ctx, cache = cut_contexts(), {}
    ref = evaluate_recipe("mean", REF3, False, ctx, cache)
    print("reference D40 (mean of REF3, 0.3 LightGBM):")
    print("  by cut: " + " ".join(f"{x:7.2f}" for x in ref) + f"   mean {ref.mean():7.2f}")
    assert np.allclose(ref, [-3.72, -25.91, -11.01, -14.02], atol=0.01), f"D40 mismatch: {ref}"

    results = []
    for r in recipes():
        sc = evaluate_recipe(r["key"], r["pool"], r["lgbm_member"], ctx, cache)
        gain = sc - ref
        won = int((gain > 0).sum())
        ok = gain.mean() >= ADOPT_MEAN and won >= ADOPT_CUTS
        results.append({
            "mode": r["mode"], "combination": r["combination"], "n_members": r["n_members"],
            "members": r["members"],
            "cut0": sc[0], "cut1": sc[1], "cut2": sc[2], "cut3": sc[3], "mean": sc.mean(),
            "gain_cut0": gain[0], "gain_cut1": gain[1], "gain_cut2": gain[2], "gain_cut3": gain[3],
            "gain_mean": gain.mean(), "cuts_won": won, "verdict": "ADOPT" if ok else "reject",
        })

    res = pd.DataFrame(results).sort_values("mean", ascending=False).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results_adamex-svk_t3.csv"
    res.to_csv(path, index=False, float_format="%.4f")
    print(f"\nwrote {path}")
    cols = ["mode", "combination", "n_members", "cut0", "cut1", "cut2", "cut3", "mean",
            "gain_mean", "cuts_won", "verdict"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
