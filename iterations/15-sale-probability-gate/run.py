"""Train a binary sale gate on each history prefix and score its paired effect."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import lightgbm
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'iterations/07-ensemble-review'),
                str(ROOT / 'iterations/10-benchmark-diagnosis')]
import diagnose
import evaluate as review
import features
import train_gbm
import validate


def binary_windows(matrix, origins):
    windows = features.build_windows(matrix, origins)
    windows['y'] = (windows['y'] > 0).astype(np.float32)
    return windows


def sale_probabilities(matrix):
    origins = features.make_origins(matrix.W)
    validation = origins[-7:]
    training = origins[origins <= validation[0] - 13]
    assert training[-1] + 13 <= validation[0] and origins[-1] + 13 <= matrix.W
    config = dict(train_gbm.PARAMS, objective='binary', seed=0)
    model, best = train_gbm.fit(binary_windows(matrix, training), binary_windows(matrix, validation), params=config)
    del model
    rounds = int(best * 1.1) + 1
    model, _ = train_gbm.fit(binary_windows(matrix, origins), rounds=rounds, params=config)
    X, _, valid = features.inference_features(matrix)
    probability = model.predict(X).reshape(matrix.n, 13)
    probability[~valid] = 0
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    return probability, rounds


def read_components(offset, keys, weeks, primary, hashes):
    source = review.SOURCE if primary else diagnose.ADDITIONAL_SOURCE
    folder = 'forecasts' if primary else 'forecasts_cuts'
    suffix = f'cut{offset//13}' if primary else f'w{offset}'
    foundation = []
    for name in review.MODELS:
        path = f'iterations/03-foundation-model/{folder}/{name}__test__{suffix}__padded.parquet'
        forecast = pd.read_parquet(io.BytesIO(review.load_blob(source, path, hashes)))
        foundation.append(review.matrix(forecast, keys, weeks))
    path = (f'iterations/02-robust-eval/forecasts/base_cut{offset//13}_seed0.npy' if primary else
            f'iterations/05-meta-ensemble/base_forecasts/lgbm_w{offset}.npy')
    regression = np.load(io.BytesIO(review.load_blob(source, path, hashes)), allow_pickle=False)
    assert regression.shape == (len(keys), 13) and np.isfinite(regression).all() and (regression >= 0).all()
    return foundation, regression


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    full = validate.load_weekly('test')
    contract = json.loads((ROOT/'iterations/10-benchmark-diagnosis/validation.json').read_text())
    primary = contract['primary_development_offsets']
    extra = [w for w in contract['purged_forward_check_offsets'] if w not in primary]
    hashes = {}
    donor = pd.read_csv(io.BytesIO(review.load_blob(diagnose.ADDITIONAL_SOURCE,
        'iterations/02-robust-eval/results_adamex_loss.csv', hashes)))
    donor = donor[donor.seed == 0].set_index(['cut','variant']).score
    benchmark = pd.read_csv(ROOT/'iterations/10-benchmark-diagnosis/results/scores.csv').set_index('offset')
    rows, group_rows = [], []
    for offset in primary + extra:
        prefix = full[full.week <= full.week.max() - pd.Timedelta(days=7*offset)]
        history, hold, cut = validate.split(prefix)
        matrix = features.ShopMatrix(prefix[prefix.week < cut])
        weeks = pd.date_range(cut, periods=13, freq='7D')
        foundation, regression = read_components(offset, matrix.keys, weeks, offset in primary, hashes)
        prob, rounds = sale_probabilities(matrix)
        gated = np.where(prob < .5, 0., regression)
        dead = matrix.U[:, -8:].sum(axis=1) == 0
        actual = review.matrix(hold.rename(columns={'units':'forecast'}), matrix.keys, weeks)
        scale = validate.history_scale(history).reindex(pd.MultiIndex.from_frame(matrix.keys)).to_numpy()
        ok = np.isfinite(scale) & (scale > 0)
        groups = diagnose.history_groups(matrix.U.astype(float),matrix.OH.astype(float),foundation+[regression],scale)
        control = review.compose(foundation, regression, dead)
        candidate = review.compose(foundation, gated, dead)
        assert (candidate <= control).all()
        for name, forecast in [('tree',regression),('tree_gated',gated),('champion',control),('candidate',candidate)]:
            result = validate.score(history,hold,train_gbm.to_long(matrix,forecast,weeks))
            errors = np.abs(actual[ok]-forecast[ok])/scale[ok,None]
            np.testing.assert_allclose(100*(1-errors.mean()),result['score'],rtol=0,atol=1e-8)
            if offset in primary and name.startswith('tree'):
                np.testing.assert_allclose(result['score'],donor.loc[offset//13,'l1' if name=='tree' else 'l1+gate0.5'],rtol=0,atol=1e-7)
            if name=='champion':
                np.testing.assert_allclose(result['score'],benchmark.loc[offset,'score'],atol=1e-8,rtol=0)
            rows.append(dict(offset=offset,first_week=str(cut.date()),variant=name,score=result['score'],
                mase=result['mase'],early=100*(1-errors[:,:7].mean()),late=100*(1-errors[:,7:].mean()),
                n_scored=int(ok.sum()),excluded=int((~ok).sum()),unit_coverage=result['holdout_units_share_scored'],
                classifier_rounds=rounds,changed_ensemble_cells=int((candidate!=control).sum())))
            if name in ('champion','candidate'):
                for dimension in ('observed_activation_age','activity13','last_stock','stock_missing13'):
                    labels=groups[dimension][ok]
                    for label in np.unique(labels):
                        mask=labels==label
                        group_rows.append(dict(offset=offset,variant=name,dimension=dimension,group=label,
                            n=int(mask.sum()),mase=errors[mask].mean()))
        pd.DataFrame(rows).to_csv(out/'scores.csv',index=False)
        pd.DataFrame(group_rows).to_csv(out/'groups.csv',index=False)
        print(f"Offset {offset}: ensemble gain {rows[-1]['score']-rows[-2]['score']:+.5f}",flush=True)
    scores=pd.DataFrame(rows)
    score=scores.pivot(index='offset',columns='variant',values='score')
    late=scores.pivot(index='offset',columns='variant',values='late')
    score['gain']=score.candidate-score.champion
    score['late_gain']=late.candidate-late.champion
    score.to_csv(out/'paired.csv')
    gain=score.loc[primary,'gain']
    summary=dict(primary_gain=gain.mean(),wins=int((gain>0).sum()),late_gain=score.loc[primary,'late_gain'].mean(),
        passed=bool(gain.mean()>=.5 and (gain>0).sum()>=3 and score.loc[primary,'late_gain'].mean()>=0 and (score.loc[extra,'gain']>=0).all()),
        seed=0,threshold=.5,lightgbm=lightgbm.__version__,source_commits=[review.SOURCE,diagnose.ADDITIONAL_SOURCE],
        blob_hashes=hashes,data_sha256=hashlib.sha256((validate.IM/'weekly_test.parquet').read_bytes()).hexdigest())
    (out/'metadata.json').write_text(json.dumps(summary,indent=2))
    print(score.round(4).to_string())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results')
    main(parser.parse_args().output)
