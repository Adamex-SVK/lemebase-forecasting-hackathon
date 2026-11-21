"""Try to break the current winner (D40: 0.3 LightGBM + 0.7 mean of 3 foundation models).

    .venv/bin/python src/neural/blend_probe.py            # evaluate every attempt
    .venv/bin/python src/neural/blend_probe.py --build    # also write submission CSVs for the best honest attempts

Protocol (TEAM-RULES.md section 2, no tuning on the scored cuts):
  * practice cuts = the 4 standard rolling cuts, D26 metric, rounded + 8-week zero rule on the ensemble part;
  * a blend is chosen by the mean on cuts 1-3 and reported on cut 0 and on the 4-cut mean;
  * every weight grid also gets a leave-one-cut-out (LOCO) number: pick the weights on the other three cuts,
    score the held-out cut; that mean is the honest estimate of a searched blend.
The reference is reproduced exactly as in src/meta/ensemble_recipe.py (D40), so `d40` must read -13.67.

Attempts: NHITS as a fourth member / instead of the LightGBM; larger and smaller foundation-model sets;
median13, CatBoost and the zero gate as extra members; and full weight grids over the main members.
Writes results_kilo_blend-probe.csv. Nothing is uploaded: a person decides that (TEAM-RULES.md section 6).
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "foundation"))
sys.path.insert(0, str(ROOT / "src" / "meta"))
import ensemble_recipe as er  # noqa: E402
import make_submission as ms  # noqa: E402
import validate as v  # noqa: E402

OUT = ROOT / "iterations" / "06-neural" / "results_kilo_blend-probe.csv"
NF_FC = ROOT / "iterations" / "06-neural" / "forecasts"
NF_FINAL = ROOT / "iterations" / "06-neural" / "forecasts_final"
O2_FC = ROOT / "iterations" / "02-robust-eval" / "forecasts"
FM_SET = ["moirai-2.0-small", "tirex", "chronos-t5-small"]          # D40
FM_SET_BIG = FM_SET + ["chronos-2", "moirai-1.1-large"]              # + two more all-4-cut models
FM_ALL = ["chronos-2", "chronos-bolt-base", "chronos-bolt-small", "chronos-t5-base", "chronos-t5-large",
          "chronos-t5-small", "moirai-1.0-large", "moirai-1.1-base", "moirai-1.1-large", "moirai-2.0-small",
          "moirai-moe-base", "moirai-moe-small", "timesfm-2.5", "tirex"]
H = er.H


def matrix_from_parquet(path, M):
    fc = pd.read_parquet(path)
    P = fc.pivot_table(index=v.KEYS, columns="week", values="forecast", aggfunc="sum")
    return P.reindex(pd.MultiIndex.from_frame(M.keys)).fillna(0.0).to_numpy(float)


def members(ctx, cache):
    """Every member as [n x 13] unit forecasts per cut, already post-processed where the recipe says so."""
    out = {}

    def add(name, per_cut):
        out[name] = per_cut

    add("FL", {k: ctx[k][5] for k in range(4)})
    add("FD", {k: np.load(O2_FC / f"d33_cut{k}_seed0.npy") for k in range(4)})
    add("FC", {k: np.load(O2_FC / f"cb_F_cut{k}_seed0.npy") for k in range(4)})
    add("FN", {k: matrix_from_parquet(NF_FC / f"NHITS__test__cut{k}__padded.parquet", ctx[k][0]) for k in range(4)})
    add("FMed", {k: np.repeat(np.median(ctx[k][0].U[:, -13:], axis=1, keepdims=True), H, axis=1).astype(float)
                 for k in range(4)})
    gate = {k: np.where(np.load(O2_FC / f"loss_P_cut{k}_seed0.npy") < 0.5, 0.0, ctx[k][5]) for k in range(4)}
    add("FLgate", gate)
    for label, models in [("E3", FM_SET), ("E5", FM_SET_BIG), ("Eall", FM_ALL)]:
        raw = {k: [er.raw_practice(m, k, ctx, cache) for m in models] for k in range(4)}
        add(label, {k: zero_rule(np.mean(raw[k], axis=0), ctx, k) for k in range(4)})
        add(label + "N", {k: zero_rule(np.mean(raw[k] + [out["FN"][k]], axis=0), ctx, k) for k in range(4)})
    return out


def zero_rule(F, ctx, k):
    F = np.array(F, dtype=float, copy=True)
    F[ctx[k][4]] = 0.0
    return F


def score(k, F, ctx):
    return er.score_cut(k, F, ctx)


def scores(Fd, ctx):
    return np.array([score(k, Fd[k], ctx) for k in range(4)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="write submission CSVs for the best honest attempts")
    a = ap.parse_args()
    ctx, cache = er.cut_contexts(), {}
    MB = members(ctx, cache)
    rows = []

    def run(name, kind, weights, note=""):
        F = {k: sum(w * MB[m][k] for m, w in weights.items() if w) for k in range(4)}
        sc = scores(F, ctx)
        rows.append({"attempt": name, "kind": kind, "weights": "+".join(f"{w:g}{m}" for m, w in weights.items() if w),
                     "weights_json": json.dumps(weights), "cut0": sc[0], "cut1": sc[1], "cut2": sc[2],
                     "cut3": sc[3], "mean4": sc.mean(), "sel_1_3": sc[1:].mean(), "note": note})
        return sc

    ref = run("d40", "reference", {"FL": 0.3, "E3": 0.7})
    d40 = ref.copy()

    # --- fixed attempts that could plausibly beat D40 -------------------------------------------
    for w in (0.2, 0.3, 0.4):
        run(f"E3N w={w}", "fixed", {"FL": w, "E3N": 1 - w}, "NHITS as a fourth member of the FM ensemble")
    run("nhits replaces lgbm", "fixed", {"FN": 0.3, "E3": 0.7}, "0.3 * NHITS + 0.7 * FM3")
    run("triple 0.15/0.15/0.7", "fixed", {"FL": 0.15, "FN": 0.15, "E3": 0.7})
    for w in (0.2, 0.3, 0.4):
        run(f"E5 w={w}", "fixed", {"FL": w, "E5": 1 - w}, "FM3 + chronos-2 + moirai-1.1-large")
        run(f"E5N w={w}", "fixed", {"FL": w, "E5N": 1 - w})
        run(f"Eall w={w}", "fixed", {"FL": w, "Eall": 1 - w}, "all 14 all-4-cut foundation models")
        run(f"EallN w={w}", "fixed", {"FL": w, "EallN": 1 - w})
    for wl, wm in [(0.25, 0.05), (0.2, 0.1), (0.1, 0.2)]:
        run(f"FL+FMed+E3 {wl:g}/{wm:g}/{0.7:.1f}", "fixed", {"FL": wl, "FMed": wm, "E3": 0.7})
    run("FL+FC+E3 0.2/0.1/0.7", "fixed", {"FL": 0.2, "FC": 0.1, "E3": 0.7}, "CatBoost as a fourth member")
    run("gate 0.3 FL + 0.7 E3", "fixed", {"FLgate": 0.3, "E3": 0.7}, "zero gate on the LightGBM (D20-generalised)")
    run("FM only (E3)", "fixed", {"E3": 1.0}, "no LightGBM")

    # --- weight grids; the honest number is the LOCO mean ---------------------------------------
    def grid(label, keys, step):
        best_per_holdout, loco_scores, combos = {}, [], []
        for parts in itertools.product(np.arange(0, 1 + 1e-9, step), repeat=len(keys) - 1):
            if sum(parts) > 1 + 1e-9:
                continue
            w = dict(zip(keys, list(parts) + [round(1 - sum(parts), 10)]))
            F = {k: sum(w[m] * MB[m][k] for m in keys if w[m] > 0) for k in range(4)}
            combos.append((w, scores(F, ctx)))
        combos.sort(key=lambda t: t[1][1:].mean(), reverse=True)
        wsel, scsel = combos[0]
        rows.append({"attempt": f"grid {label} (chosen on cuts 1-3)", "kind": f"grid step {step:g}",
                     "weights": "+".join(f"{v:g}{m}" for m, v in wsel.items() if v), "weights_json": json.dumps(wsel),
                     "cut0": scsel[0], "cut1": scsel[1], "cut2": scsel[2], "cut3": scsel[3],
                     "mean4": scsel.mean(), "sel_1_3": scsel[1:].mean(),
                     "note": f"best of {len(combos)} combos on cuts 1-3"})
        for hold in range(4):
            others = [j for j in range(4) if j != hold]
            w, sc = max(combos, key=lambda t: t[1][others].mean())
            loco_scores.append(sc[hold])
        rows.append({"attempt": f"grid {label} (LOCO)", "kind": f"grid step {step:g}", "weights": "",
                     "weights_json": "{}", "cut0": loco_scores[0], "cut1": loco_scores[1], "cut2": loco_scores[2],
                     "cut3": loco_scores[3], "mean4": np.mean(loco_scores), "sel_1_3": np.mean(loco_scores[1:]),
                     "note": "held-out cut per LOCO choice; honest estimate of the search"})
        oracle = max(combos, key=lambda t: t[1].mean())
        rows.append({"attempt": f"grid {label} (oracle, NOT honest)", "kind": f"grid step {step:g}",
                     "weights": "+".join(f"{v:g}{m}" for m, v in oracle[0].items() if v),
                     "weights_json": json.dumps(oracle[0]),
                     "cut0": oracle[1][0], "cut1": oracle[1][1], "cut2": oracle[1][2], "cut3": oracle[1][3],
                     "mean4": oracle[1].mean(), "sel_1_3": oracle[1][1:].mean(),
                     "note": "best combo on all 4 cuts; upper bound only"})
        print(f"grid {label}: {len(combos)} combos, chosen {wsel}, LOCO mean {np.mean(loco_scores):.2f}")

    grid("FL+FN+E3", ["FL", "FN", "E3"], 0.1)
    grid("FL+FN+E3+FMed", ["FL", "FN", "E3", "FMed"], 0.1)
    grid("FL+FC+FN+E3", ["FL", "FC", "FN", "E3"], 0.1)
    grid("FL+FN+E5", ["FL", "FN", "E5"], 0.1)
    grid("FL+FLgate+E3", ["FL", "FLgate", "E3"], 0.2)

    res = pd.DataFrame(rows)
    res["vs_d40_mean"] = res["mean4"] - d40.mean()
    res["cuts_won_vs_d40"] = [(np.array(r[["cut0", "cut1", "cut2", "cut3"]]) > d40).sum() for _, r in res.iterrows()]
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 260)
    print(f"\nreference d40: " + " ".join(f"{x:7.2f}" for x in d40) + f"   mean {d40.mean():.2f}\n")
    cols = ["attempt", "weights", "cut0", "cut1", "cut2", "cut3", "mean4", "vs_d40_mean", "cuts_won_vs_d40"]
    print("all attempts, ranked by the 4-cut mean:")
    print(res[cols].sort_values("mean4", ascending=False).round(2).to_string(index=False))
    honest = res[~res["attempt"].str.contains("oracle")]
    best = honest.sort_values("mean4", ascending=False).iloc[0]
    print(f"\nbest honest attempt: {best['attempt']}  weights={best['weights']}  "
          f"mean {best['mean4']:.2f} ({best['vs_d40_mean']:+.2f} vs D40, {best['cuts_won_vs_d40']}/4 cuts)")
    ok = best["vs_d40_mean"] >= 0.5 and best["cuts_won_vs_d40"] >= 3
    print("verdict:", "CLEARS THE BAR" if ok else "does not clear the +0.5 / 3-of-4 bar; keep D40")
    if a.build:
        build(honest)


def build(honest):
    """Write submission CSVs for the best honest attempts from final-week forecasts (no upload).

    Members whose final-week forecasts do not exist (no `--final` run yet) are skipped with a message."""
    ids = ms.history_ids()
    dead = er.dead_mask(ids)
    L = pd.read_csv(ms.SUB_DIR / "lgbm_own_v1.csv").set_index("article_id").loc[ids].to_numpy(float)
    MC = {}

    def zero_rule_final(V):
        V = np.array(V, dtype=float, copy=True)
        V[dead] = 0.0
        return V

    def fm_final(m):
        p = er.FM_FINAL / f"{m}__test__final__padded.parquet"
        if not p.exists():
            raise FileNotFoundError(f"no final forecast for {m!r} (run src/foundation/run_model.py {m} test --final)")
        P = pd.read_parquet(p).pivot_table(index="article_id", columns="week", values="forecast", aggfunc="sum").reindex(ids)
        assert P.notna().all().all(), f"{m}: final forecast incomplete"
        return P.to_numpy(float)

    def member(name):
        if name in MC:
            return MC[name]
        if name == "FL":
            V = L
        elif name == "FN":
            p = NF_FINAL / "NHITS__test__final__padded.parquet"
            if not p.exists():
                raise FileNotFoundError("no final NHITS forecast (run src/neural/run_nf.py --final)")
            P = pd.read_parquet(p).pivot_table(index="article_id", columns="week", values="forecast", aggfunc="sum").reindex(ids)
            assert P.notna().all().all(), "NHITS: final forecast incomplete"
            V = P.to_numpy(float)
        elif name == "FMed":
            V = er.final_matrix("dummy-median13", ids)
        elif name in ("E3", "E5", "Eall"):
            models = {"E3": FM_SET, "E5": FM_SET_BIG, "Eall": FM_ALL}[name]
            V = zero_rule_final(np.mean([fm_final(m) for m in models], axis=0))
        elif name.endswith("N") and name[:-1] in ("E3", "E5", "Eall"):
            V = zero_rule_final(np.mean([member(name[:-1]), member("FN")], axis=0))
        else:
            raise FileNotFoundError(f"no final forecast for member {name!r}")
        MC[name] = V
        return V

    seen, built = set(), 0
    for _, r in honest.sort_values("mean4", ascending=False).iterrows():
        weights = {m: w for m, w in json.loads(r["weights_json"]).items() if w > 0}
        if not weights:
            continue
        key = tuple(sorted((m, round(w, 6)) for m, w in weights.items()))
        if key in seen:
            continue
        try:
            F = np.clip(np.round(sum(w * member(m) for m, w in weights.items())), 0, None)
        except FileNotFoundError as exc:
            print(f"skip {r['attempt']} ({r['weights']}): {exc}")
            continue
        seen.add(key)
        built += 1
        if built > 3:
            break
        safe = r["attempt"].replace(" ", "_").replace("=", "").replace("/", "-").replace("(", "").replace(")", "")
        out = ms.SUB_DIR / f"blend_probe_{built}_{safe}.csv"
        ms.write_submission(F, ids, out)
        ms.validate_submission(out)
        ref = pd.read_csv(ms.SUB_DIR / "ensemble_top3_blend_v1.csv").set_index("article_id").loc[ids].to_numpy(float)
        print(f"wrote {out.name}: {r['attempt']} weights={r['weights']} | local 4-cut {r['mean4']:.2f} "
              f"({r['vs_d40_mean']:+.2f} vs D40) | vs current submission: mean |diff| {np.abs(F - ref).mean():.2f} units, "
              f"{100 * (F != ref).mean():.1f}% cells changed")


if __name__ == "__main__":
    main()
