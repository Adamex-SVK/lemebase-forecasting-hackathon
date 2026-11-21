"""Leave-one-cut-out selection between neural configs on the standard 4 cuts (no neuralforecast needed).

    .venv/bin/python src/neural/select_loco.py --models NHITS <stem1> <stem2> ... [--handle kilo]

Each stem is a full output stem in iterations/06-neural/forecasts/ (e.g. NHITS, NHITS_s3000_lr3e-4_in104).
For every stem and cut the neural matrix N is loaded exactly like eval_nf_d40.nn_matrix (ShopMatrix key
order). The D40 foundation ensemble uses the zero rule (fm3); N and LightGBM Fl stay raw. Three fixed-weight
targets mirror eval_nf_d40: alone = N, replace = 0.3*N + 0.7*fm3, split = 0.15*Fl + 0.15*N + 0.7*fm3.
LOCO: for each held-out cut K pick the stem with the best mean over the other three cuts; the honest score
is that stem's score on K. Writes iterations/06-neural/results_{handle}_loco.csv.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "meta"))
sys.path.insert(0, str(ROOT / "src" / "foundation"))
import ensemble_recipe as er  # noqa: E402
import validate as v  # noqa: E402

NF = ROOT / "iterations" / "06-neural"
TARGETS = ("alone", "replace", "split")
COLUMNS = ["mode", "target", "stem", "cut", "score", "held_out_cut", "chosen_stem",
           "chosen_other3_mean", "honest_score"]


def forecast_path(stem, k):
    return NF / "forecasts" / f"{stem}__test__cut{k}__padded.parquet"


def missing_cuts(stem):
    return [k for k in range(er.N_CUTS) if not forecast_path(stem, k).exists()]


def nn_matrix(stem, k, M):
    fc = pd.read_parquet(forecast_path(stem, k))
    return fc.pivot_table(index=v.KEYS, columns="week", values="forecast", aggfunc="sum").reindex(
        pd.MultiIndex.from_frame(M.keys)).fillna(0.0).to_numpy(float)


def blend(N, fm3, Fl, target):
    if target == "alone":
        return N
    if target == "replace":
        return 0.3 * N + 0.7 * fm3
    return 0.15 * Fl + 0.15 * N + 0.7 * fm3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="full output stems, e.g. NHITS NHITS_s3000")
    ap.add_argument("--handle", default="kilo")
    a = ap.parse_args()

    stems = []
    for stem in a.models:
        miss = missing_cuts(stem)
        if miss:
            warnings.warn(f"skipping {stem!r}: missing cut(s) {miss} in {NF / 'forecasts'}")
            continue
        stems.append(stem)
    if not stems:
        raise SystemExit("no stem has all four practice-cut forecasts")

    ctx, cache = er.cut_contexts(), {}
    fm3 = {}
    for k in range(er.N_CUTS):
        Ffm = [er.raw_practice(m, k, ctx, cache) for m in er.REFERENCE_MODELS]
        fm3[k] = np.mean(Ffm, axis=0)
        fm3[k][ctx[k][4]] = 0.0

    scores = {t: np.full((len(stems), er.N_CUTS), np.nan) for t in TARGETS}
    for i, stem in enumerate(stems):
        for k in range(er.N_CUTS):
            N = nn_matrix(stem, k, ctx[k][0])
            Fl = ctx[k][5]
            for target in TARGETS:
                scores[target][i, k] = er.score_cut(k, blend(N, fm3[k], Fl, target), ctx)

    rows = []
    for target in TARGETS:
        S = scores[target]
        print(f"\n=== {target} ===")
        print("in-sample, for information")
        print(f"{'stem':40s} " + " ".join(f"{f'cut{k}':>8s}" for k in range(er.N_CUTS)) + f" {'mean':>8s}")
        for i, stem in enumerate(stems):
            print(f"{stem:40s} " + " ".join(f"{x:8.2f}" for x in S[i]) + f" {S[i].mean():8.2f}")
            for k in range(er.N_CUTS):
                rows.append({"mode": "in_sample", "target": target, "stem": stem, "cut": k,
                             "score": round(float(S[i, k]), 4), "held_out_cut": "", "chosen_stem": "",
                             "chosen_other3_mean": "", "honest_score": ""})

        print("LOCO (leave-one-cut-out)")
        print(f"{'held_out_cut':>12s} {'chosen_stem':40s} {'other3_mean':>11s} {'honest_score':>12s}")
        honest = []
        for K in range(er.N_CUTS):
            others = [k for k in range(er.N_CUTS) if k != K]
            means = S[:, others].mean(axis=1)
            i = int(np.argmax(means))
            honest.append(S[i, K])
            print(f"{K:12d} {stems[i]:40s} {means[i]:11.2f} {S[i, K]:12.2f}")
            rows.append({"mode": "loco", "target": target, "stem": "", "cut": "",
                         "score": "", "held_out_cut": K, "chosen_stem": stems[i],
                         "chosen_other3_mean": round(float(means[i]), 4),
                         "honest_score": round(float(S[i, K]), 4)})
        print(f"{'honest LOCO mean':>65s} {np.mean(honest):12.2f}")

    out = NF / f"results_{a.handle}_loco.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
