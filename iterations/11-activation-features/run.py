"""One prespecified feature ablation; no final submission generation.
Run: .venv/bin/python iterations/11-activation-features/run.py
Outputs must be new; optional --output PATH for independent reproduction.
"""
import argparse
import gc
import hashlib
import io
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'iterations/07-ensemble-review'),
                str(ROOT/'iterations/10-benchmark-diagnosis')]
import features as f
import train_gbm as g
import validate as v
import evaluate as review
import diagnose as diagnosis
import activation as a

OFFSETS = [0, 13, 26, 39, 16, 32]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(win, val=None, rounds=None, candidate=False):
    names = a.NAMES if candidate else f.FEATURES
    ds = lgb.Dataset(win['X'], win['y'], weight=win['w'], feature_name=names)
    if val is not None:
        dv = lgb.Dataset(val['X'], val['y'], weight=val['w'], reference=ds)
        model = lgb.train(g.PARAMS, ds, num_boost_round=3000, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        return model, model.best_iteration
    return lgb.train(g.PARAMS, ds, num_boost_round=rounds), rounds


def train(M, candidate, folder):
    start = time.time()
    origins = f.make_origins(M.W)
    val = origins[-7:]
    tr = origins[origins <= val[0]-13]
    assert len(tr) and tr.max()+13 <= val.min()
    def windows(origins):
        w = f.build_windows(M, origins)
        return a.augment(w, M) if candidate else w
    wt, wv = windows(tr), windows(val)
    m, best = fit(wt, wv, candidate=candidate)
    del wt, wv, m
    gc.collect()
    rounds = int(best*1.1)+1
    wa = windows(origins)
    model, _ = fit(wa, rounds=rounds, candidate=candidate)
    del wa
    X, scale, valid = a.inference(M) if candidate else f.inference_features(M)
    pred = np.maximum(model.predict(X).reshape(M.n, 13)*scale[:, None], 0)
    pred[~valid] = 0
    assert np.isfinite(pred).all()
    model.save_model(str(folder/'model.txt'))
    np.save(folder/'forecast.npy', pred, allow_pickle=False)
    pd.DataFrame({'feature': a.NAMES if candidate else f.FEATURES,
                  'gain': model.feature_importance('gain')}).to_csv(folder/'importance.csv', index=False)
    info = dict(best_iteration=int(best), final_rounds=rounds, seconds=time.time()-start,
                training_origins=tr.tolist(), validation_origins=val.tolist(), final_origins=origins.tolist())
    (folder/'training.json').write_text(json.dumps(info, indent=2))
    del model, X
    gc.collect()
    return pred, info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=HERE/'results')
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    runs = out/'runs'
    runs.mkdir()
    config = json.loads((ROOT/'iterations/10-benchmark-diagnosis/validation.json').read_text())
    full = v.load_weekly('test')
    locked = ROOT/'submissions/ensemble_top3_blend_v1.csv'
    inputpath = v.IM/'weekly_test.parquet'
    initial = {'submission': sha(locked), 'data': sha(inputpath)}
    rows, groups, importance, hashes = [], [], [], {}
    for offset in OFFSETS:
        print(f'Offset {offset}: starting paired fits', flush=True)
        d = full[full.week <= full.week.max()-pd.Timedelta(weeks=offset)]
        history, hold, cut = v.split(d)
        M = f.ShopMatrix(d[d.week < cut])
        weeks = pd.date_range(cut, periods=13, freq='7D')
        actual = review.matrix(hold.rename(columns={'units':'forecast'}), M.keys, weeks)
        scale = v.history_scale(history).reindex(pd.MultiIndex.from_frame(M.keys)).to_numpy()
        ok = np.isfinite(scale) & (scale > 0)
        original = offset in [0,13,26,39]
        source = review.SOURCE if original else diagnosis.ADDITIONAL_SOURCE
        def blob(path):
            temp = {}
            data = review.load_blob(source, path, temp)
            hashes.update({f'{source}:{p}': h for p,h in temp.items()})
            return data
        parts = []
        for name in review.MODELS:
            folder = 'forecasts' if original else 'forecasts_cuts'
            suffix = f'cut{offset//13}' if original else f'w{offset}'
            path = f'iterations/03-foundation-model/{folder}/{name}__test__{suffix}__padded.parquet'
            parts.append(review.matrix(pd.read_parquet(io.BytesIO(blob(path))), M.keys, weeks))
        path = (f'iterations/02-robust-eval/forecasts/base_cut{offset//13}_seed0.npy' if original
                else f'iterations/05-meta-ensemble/base_forecasts/lgbm_w{offset}.npy')
        incumbent = np.load(io.BytesIO(blob(path)), allow_pickle=False)
        assert incumbent.shape == actual.shape and np.isfinite(incumbent).all() and (incumbent >= 0).all()
        if original:
            keyed = review.matrix(pd.read_parquet(ROOT/f'logs/chronos_four/LightGBM_{offset//13}.parquet'), M.keys, weeks)
            np.testing.assert_allclose(incumbent, keyed, atol=1e-8, rtol=0)
        labels = diagnosis.history_groups(M.U.astype(float), M.OH.astype(float), parts+[incumbent], scale)
        forecasts = {'incumbent': incumbent}
        infos = {}
        for variant in ['control', 'activation']:
            folder = runs/f'w{offset}_{variant}'
            folder.mkdir()
            forecasts[variant], infos[variant] = train(M, variant == 'activation', folder)
            print(f"Offset {offset} {variant}: {infos[variant]['final_rounds']} trees, {infos[variant]['seconds']:.0f}s", flush=True)
            imp = pd.read_csv(folder/'importance.csv')
            imp['offset'], imp['variant'] = offset, variant
            importance.append(imp)
        baseline = review.compose(parts, incumbent, M.U[:,-8:].sum(axis=1)==0)
        for variant, L in forecasts.items():
            F = review.compose(parts, L, M.U[:,-8:].sum(axis=1)==0)
            err = np.abs(F[ok]-actual[ok])/scale[ok,None]
            fc = g.to_long(M, F, weeks)
            official = v.score(history, hold, fc)
            calculated = 100*(1-err.mean())
            assert np.isclose(official['score'], calculated, atol=1e-8)
            row = dict(offset=offset, primary=original, forward_check=offset in [0,16,32],
                       variant=variant, holdout_start=str(cut.date()), holdout_end=str(weeks[-1].date()),
                       score=calculated, mase=err.mean(), late_score=100*(1-err[:,7:].mean()),
                       early_score=100*(1-err[:,:7].mean()), n_series=M.n, n_scored=int(ok.sum()),
                       excluded=int((~ok).sum()), scored_units_share=float(actual[ok].sum()/actual.sum()),
                       lgbm_max_abs_drift_vs_saved=float(np.abs(L-incumbent).max()),
                       **infos.get(variant, {}))
            # Lists live in per-fit metadata, not numerical comparison CSV.
            rows.append({k:x for k,x in row.items() if not isinstance(x, list)})
            for dimension in ['observed_activation_age','activity13','last_stock','stock_missing13']:
                for label in sorted(set(labels[dimension][ok])):
                    mask = labels[dimension][ok] == label
                    groups.append(dict(offset=offset, primary=original, variant=variant, dimension=dimension,
                                       group=label, n_scored=int(mask.sum()),
                                       score=float(100*(1-err[mask].mean())),
                                       total_mase_contribution=float(err[mask].sum()/len(err)/13)))
        results = pd.DataFrame(rows)
        results.to_csv(out/'scores.csv', index=False)
        pd.DataFrame(groups).to_csv(out/'groups.csv', index=False)
        pd.concat(importance).to_csv(out/'feature_importance.csv', index=False)
        print(results[results.offset==offset][['variant','score','late_score']].round(5).to_string(index=False), flush=True)
    summary = results[results.primary].groupby('variant')[['score','late_score']].mean()
    summary.to_csv(out/'primary_summary.csv')
    results[results.forward_check].groupby('variant')[['score','late_score']].mean().to_csv(out/'forward_summary.csv')
    primary = results[results.primary].pivot(index='offset', columns='variant', values='score')
    gain = float(summary.loc['activation','score']-summary.loc['incumbent','score'])
    wins = int((primary.activation > primary.incumbent).sum())
    late_gain = float(summary.loc['activation','late_score']-summary.loc['incumbent','late_score'])
    review_pass = gain >= .5 and wins >= 3 and late_gain >= 0
    assert initial == {'submission':sha(locked), 'data':sha(inputpath)}
    metadata = dict(input_sha256=initial, source_sha256=hashes, validation_contract=config,
                    params=g.PARAMS, python=platform.python_version(), lightgbm=lgb.__version__,
                    numpy=np.__version__, pandas=pd.__version__, offsets=OFFSETS,
                    primary_gain=gain, primary_wins=wins, late_gain=late_gain,
                    promotion_review_pass=review_pass, independent_confirmation=False,
                    cross_shop_evaluation=False, submission_changed=False,
                    code_sha256={p.name:sha(p) for p in [HERE/'run.py', HERE/'activation.py', ROOT/'src/features.py', ROOT/'src/train_gbm.py', ROOT/'src/validate.py']})
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(summary.round(5).to_string(), flush=True)
    print(f'Gain={gain:.5f}, wins={wins}/4, late gain={late_gain:.5f}, review pass={review_pass}', flush=True)


if __name__ == '__main__':
    main()
