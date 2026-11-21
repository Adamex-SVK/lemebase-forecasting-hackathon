"""Review immutable saved forecasts without merging or executing donor code.

Run: .venv/bin/python iterations/07-ensemble-review/evaluate.py
Requires git fetch origin to have downloaded the recorded source commit.
All cuts were already examined in prior work: results are exploratory, not a fresh test.
"""
import argparse
import hashlib
import io
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import validate as v

SOURCE = "71b82f33f0fe5647a13d3786cd17d11b4c4e5493"
MODELS = ["moirai-2.0-small", "tirex", "chronos-t5-small"]


def load_blob(commit, path, hashes):
    data = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
    hashes[path] = hashlib.sha256(data).hexdigest()
    return data


def matrix(fc, keys, weeks):
    expected = pd.DataFrame({"store_id": np.repeat(keys.store_id.to_numpy(), len(weeks)),
                             "article_id": np.repeat(keys.article_id.to_numpy(), len(weeks)),
                             "week": np.tile(weeks, len(keys))})
    fc = fc.copy()
    fc["week"] = pd.to_datetime(fc.week)
    m = expected.merge(fc[v.KEYS + ["week", "forecast"]], on=v.KEYS + ["week"],
                       how="outer", validate="one_to_one", indicator=True)
    if not (m["_merge"] == "both").all() or not np.isfinite(m.forecast).all():
        raise ValueError("Missing, extra, or non-finite forecast rows")
    # merge outer may sort keys; explicitly recover expected order.
    m = expected.merge(m.drop(columns="_merge"), on=v.KEYS + ["week"], validate="one_to_one")
    a = m.forecast.to_numpy(float).reshape(len(keys), len(weeks))
    if (a < 0).any():
        raise ValueError("Negative component forecasts need an explicit processing decision")
    return a


def compose(parts, lgbm, dead, weight=.3, rounded=True, zero_rule=True):
    mean = np.mean(parts, axis=0)
    if zero_rule:
        mean[dead] = 0
    out = weight*lgbm + (1-weight)*mean
    return np.clip(np.round(out) if rounded else out, 0, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", args.source], cwd=ROOT, text=True).strip()
    hashes, rows, grids, gains, correlations = {}, [], [], [], []
    full = v.load_weekly("test")
    for k in range(4):
        d = full[full.week <= full.week.max() - pd.Timedelta(weeks=13*k)]
        train, hold, cut = v.split(d)
        keys = train[v.KEYS].drop_duplicates().sort_values(v.KEYS).reset_index(drop=True)
        weeks = pd.date_range(cut, periods=13, freq="7D")
        hist = train.pivot(index=v.KEYS, columns="week", values="units").reindex(pd.MultiIndex.from_frame(keys))
        dead = hist.iloc[:, -8:].sum(axis=1).to_numpy() == 0
        actual = matrix(hold.rename(columns={"units": "forecast"}), keys, weeks)
        scales = v.history_scale(train).reindex(pd.MultiIndex.from_frame(keys)).to_numpy()
        ok = np.isfinite(scales) & (scales > 0)
        # NPY row order: verify against keyed, locally saved reference before using it.
        path = f"iterations/02-robust-eval/forecasts/base_cut{k}_seed0.npy"
        lgbm = np.load(io.BytesIO(load_blob(commit, path, hashes)), allow_pickle=False)
        keyed_ref = matrix(pd.read_parquet(ROOT / f"logs/chronos_four/LightGBM_{k}.parquet"), keys, weeks)
        if lgbm.shape != actual.shape or not np.allclose(lgbm, keyed_ref, rtol=0, atol=1e-8):
            raise ValueError("Donor NPY order/values do not match keyed LightGBM reference")
        forecasts = {}
        for model in MODELS:
            path = f"iterations/03-foundation-model/forecasts/{model}__test__cut{k}__padded.parquet"
            forecasts[model] = matrix(pd.read_parquet(io.BytesIO(load_blob(commit, path, hashes))), keys, weeks)

        def score(F, period=slice(None)):
            err = np.abs(actual[ok, period]-F[ok, period]).mean(axis=1)/scales[ok]
            return float(100*(1-err.mean()))

        current = compose(list(forecasts.values()), lgbm, dead)
        base_error = np.abs(actual[ok]-current[ok]).mean(axis=1)/scales[ok]
        variants = {"current_30pct_lgbm": current, "lgbm_raw": lgbm, "lgbm_rounded": np.round(lgbm),
                    "top3_no_lgbm": compose(list(forecasts.values()), lgbm, dead, weight=0),
                    "current_no_rounding": compose(list(forecasts.values()), lgbm, dead, rounded=False),
                    "current_no_zero_rule": compose(list(forecasts.values()), lgbm, dead, zero_rule=False)}
        for model in MODELS:
            variants[f"alone_{model}"] = compose([forecasts[model]], lgbm, dead, weight=0)
            variants[f"without_{model}"] = compose([f for m, f in forecasts.items() if m != model], lgbm, dead)
        for name, F in variants.items():
            err = np.abs(actual[ok]-F[ok]).mean(axis=1)/scales[ok]
            gain = 100*(base_error-err)
            rng = np.random.default_rng(0)
            bs = [rng.choice(gain, len(gain), replace=True).mean() for _ in range(500)]
            rows.append(dict(cut=k, holdout_start=str(cut.date()), variant=name,
                             score=score(F), early_score=score(F, slice(0,7)), late_score=score(F, slice(7,13)),
                             gain_vs_current=gain.mean(), n_series=len(keys), n_scored=int(ok.sum()),
                             excluded=int((~ok).sum()),
                             scored_units_share=float(actual[ok].sum()/actual.sum()),
                             paired_gain_p025=np.percentile(bs,2.5), paired_gain_p975=np.percentile(bs,97.5)))
        # Compare aggregate NumPy calculation against the authoritative evaluator.
        fc = hold[v.KEYS + ["week"]].copy()
        ordered = pd.DataFrame({"store_id": np.repeat(keys.store_id,13), "article_id": np.repeat(keys.article_id,13),
                                "week": np.tile(weeks,len(keys)), "forecast": current.reshape(-1)})
        fc = fc.merge(ordered, on=v.KEYS+["week"], validate="one_to_one")
        assert np.isclose(score(current), v.score(train, hold, fc)["score"])
        # Same foundation set and processing, varying only the LightGBM share.
        for w in np.round(np.arange(0,1.01,.1),1):
            F = compose(list(forecasts.values()), lgbm, dead, weight=float(w))
            grids.append(dict(cut=k, lgbm_weight=w, score=score(F), early_score=score(F,slice(0,7)),
                              late_score=score(F,slice(7,13))))
        # Group diagnostics use history only. Groups overlap; their error shares are not additive.
        for name, mask in {"recent13_active": hist.iloc[:,-13:].sum(axis=1).to_numpy()>0,
                           "recent13_inactive": hist.iloc[:,-13:].sum(axis=1).to_numpy()==0}.items():
            mask = mask & ok
            for label, F in [("current",current),("lgbm_rounded",np.round(lgbm))]:
                err = np.abs(actual[mask]-F[mask]).mean(axis=1)/scales[mask]
                gains.append(dict(cut=k, group=name, model=label, n_series=int(mask.sum()),
                                  score=float(100*(1-err.mean()))))
        # Signed scaled errors, rather than raw forecast correlations dominated by volume.
        components = {**forecasts, "lgbm":lgbm}
        for a,b in itertools.combinations(components,2):
            ea=((components[a][ok]-actual[ok])/scales[ok,None]).reshape(-1)
            eb=((components[b][ok]-actual[ok])/scales[ok,None]).reshape(-1)
            correlations.append(dict(cut=k, model_a=a, model_b=b, signed_scaled_error_correlation=np.corrcoef(ea,eb)[0,1]))

    results=pd.DataFrame(rows)
    results.to_csv(out/"ablation_by_cut.csv",index=False)
    summary=results.groupby("variant").agg(mean_score=("score","mean"), mean_gain_vs_current=("gain_vs_current","mean"),
        early_score=("early_score","mean"),late_score=("late_score","mean"),cuts_better_than_current=("gain_vs_current",lambda x:int((x>1e-8).sum())))
    summary.to_csv(out/"ablation_summary.csv")
    grid=pd.DataFrame(grids)
    grid.to_csv(out/"weights_by_cut.csv",index=False)
    weights=grid.groupby("lgbm_weight").score.mean().rename("all_cut_mean").to_frame()
    weights["earlier_cuts_mean"]=grid[grid.cut>0].groupby("lgbm_weight").score.mean()
    weights["newest_cut_score"]=grid[grid.cut==0].set_index("lgbm_weight").score
    weights.to_csv(out/"weights_summary.csv")
    pd.DataFrame(gains).to_csv(out/"activity_groups.csv",index=False)
    pd.DataFrame(correlations).to_csv(out/"error_correlations.csv",index=False)
    # Rebuild the current final file in memory, then compare with the recorded submission.
    ids=full.article_id.drop_duplicates().sort_values().to_numpy()
    finalkeys=pd.DataFrame({"store_id":"T","article_id":ids})
    finalweeks=pd.date_range(full.week.max()+pd.Timedelta(weeks=1),periods=13,freq="7D")
    parts=[]
    for model in MODELS:
        path=f"iterations/03-foundation-model/forecasts_final/{model}__test__final__padded.parquet"
        parts.append(matrix(pd.read_parquet(io.BytesIO(load_blob(commit,path,hashes))),finalkeys,finalweeks))
    def submission(path):
        f=pd.read_csv(io.BytesIO(load_blob(commit,path,hashes)))
        assert f.article_id.is_unique and set(f.article_id)==set(ids)
        return f.set_index("article_id").loc[ids,[f"w{i}" for i in range(1,14)]].to_numpy(float)
    L=submission("submissions/lgbm_own_v1.csv")
    finalhist=full.pivot(index="article_id",columns="week",values="units").reindex(ids)
    rebuilt=compose(parts,L,finalhist.iloc[:,-8:].sum(axis=1).to_numpy()==0)
    recorded=submission("submissions/ensemble_top3_blend_v1.csv")
    discrepancy=float(np.abs(rebuilt-recorded).max())
    assert discrepancy==0, "Recorded submission differs from reconstructed recipe"
    metadata=dict(source_commit=commit, source_branch="it03/adamex-svk/foundation-models",
                  source_sha256=hashes, input_sha256=hashlib.sha256((v.IM/"weekly_test.parquet").read_bytes()).hexdigest(),
                  recipe="mean raw FM -> zero if no sales in last 8 weeks -> 70/30 FM/LGBM -> np.round -> clip",
                  reported_public_score=-4.8, score_source="User confirmation and source submissions/UPLOADS.md",
                  final_submission_max_abs_difference=discrepancy,
                  selected_weight_on_earlier_cuts=float(weights.earlier_cuts_mean.idxmax()),
                  warning="All four cuts previously used in development. Ablations, weight searches, and intervals are exploratory; no fresh confirmation.")
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2))
    print(summary.sort_values("mean_score",ascending=False).round(4).to_string())
    print("\nWeights:\n",weights.round(4).to_string())
    print("Final submission recipe reproduced exactly:",discrepancy)


if __name__=="__main__":
    main()
