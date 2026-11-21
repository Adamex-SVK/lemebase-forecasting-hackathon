"""T6 (iteration 08): score the quantile / sample-count variants against D40 on the 4 cuts.

The forecast parquets must already be in iterations/03-foundation-model/forecasts/ (new names only).
The solo scores come from score_all.py run over the same forecasts (results CSV written separately).

    .venv/bin/python src/improve/t6_quantiles.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "meta"))
import ensemble_recipe as er  # noqa: E402

M, T, C = "moirai-2.0-small", "tirex", "chronos-t5-small"
VARIANTS = {
    "q40_all": [f"{M}-q40", f"{T}-q40", f"{C}-q40"],
    "q45_all": [f"{M}-q45", T, C],          # only Moirai has a q45 spec; equals the one-at-a-time test
    "moirai_q45_only": [f"{M}-q45", T, C],
    "moirai_q40_only": [f"{M}-q40", T, C],
    "tirex_q40_only": [M, f"{T}-q40", C],
    "chronos_q40_only": [M, T, f"{C}-q40"],
    "s200_replaces_moirai": ["moirai-1.1-base-s200", T, C],
    "s200_as_4th_member": [M, T, C, "moirai-1.1-base-s200"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="iterations/08-improvements/t6/results_adamex-svk_t6.csv")
    ap.add_argument("--solo-csv", default=None,
                    help="results_zero_shot.csv written by score_all.py on the GPU (for the solo rows)")
    a = ap.parse_args()

    ctx, cache = er.cut_contexts(), {}
    ref = er.evaluate_scores(er.REFERENCE_MODELS, er.REFERENCE_W, ctx, cache)
    rows = [dict(variant="D40_reference", cut0=ref[0], cut1=ref[1], cut2=ref[2], cut3=ref[3],
                 mean=ref.mean(), gain=0.0, cuts_won=0, verdict="reference")]
    for name, models in VARIANTS.items():
        s = er.evaluate_scores(models, er.REFERENCE_W, ctx, cache)
        d = s - ref
        won = int((d > 0).sum())
        ok = d.mean() >= er.ADOPT_MEAN and won >= er.ADOPT_CUTS
        rows.append(dict(variant=name, cut0=s[0], cut1=s[1], cut2=s[2], cut3=s[3], mean=s.mean(),
                         gain=d.mean(), cuts_won=won, verdict="ADOPT" if ok else "reject",
                         models=",".join(models)))

    if a.solo_csv:
        solo = pd.read_csv(a.solo_csv)
        solo = solo[solo["limit"] == "full"]
        for model, g in solo.groupby("model"):
            p = g.pivot_table(index="post", columns="cut_index", values="score")
            if "round+zero8" not in p.index:
                continue
            r = p.loc["round+zero8"]
            rows.append(dict(variant=f"solo:{model}", cut0=r.get(0, np.nan), cut1=r.get(1, np.nan),
                             cut2=r.get(2, np.nan), cut3=r.get(3, np.nan), mean=r.mean(),
                             gain=np.nan, cuts_won=np.nan, verdict="solo (round+zero8)"))

    res = pd.DataFrame(rows)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    print(res.round(2).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
