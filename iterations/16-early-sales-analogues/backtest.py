"""Run the prespecified nearest-neighbour test on six historical holdouts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from analogues import peer_forecast

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'iterations/15-sale-probability-gate'))
from run import read_components, review, features, validate, train_gbm, diagnose

CONTRACT = ROOT / 'iterations/10-benchmark-diagnosis/validation.json'


def evaluate_cut(full, offset, primary, locked, hashes):
    prefix = full[full.week <= full.week.max() - pd.Timedelta(days=7 * offset)]
    history, holdout, cut = validate.split(prefix)
    matrix = features.ShopMatrix(prefix[prefix.week < cut])
    weeks = pd.date_range(cut, periods=13, freq='7D')
    parts, tree = read_components(offset, matrix.keys, weeks, primary, hashes)
    control = review.compose(parts, tree, matrix.U[:, -8:].sum(axis=1) == 0)
    candidate, matches = peer_forecast(matrix.U, control)
    recent = np.zeros(matrix.n, dtype=bool)
    matched = np.zeros(matrix.n, dtype=bool)
    for row in matches:
        recent[row['target']] = True
        matched[row['target']] = row['used'] > 0
    np.testing.assert_array_equal(candidate[~matched], control[~matched])
    scale = validate.history_scale(history).reindex(pd.MultiIndex.from_frame(matrix.keys)).to_numpy()
    ok = np.isfinite(scale) & (scale > 0)
    actual = review.matrix(holdout.rename(columns={'units': 'forecast'}), matrix.keys, weeks)
    labels = diagnose.history_groups(matrix.U.astype(float), matrix.OH.astype(float), parts+[tree], scale)
    scores, groups = [], []
    for name, forecast in [('champion', control), ('analogue', candidate)]:
        official = validate.score(history, holdout, train_gbm.to_long(matrix, forecast, weeks))
        errors = np.abs(actual[ok] - forecast[ok]) / scale[ok,None]
        np.testing.assert_allclose(official['score'], 100*(1-errors.mean()), atol=1e-8, rtol=0)
        if name == 'champion':
            np.testing.assert_allclose(official['score'], locked, atol=1e-8, rtol=0)
        scores.append(dict(offset=offset, primary=primary, start=str(cut.date()), end=str(weeks[-1].date()),
            variant=name, score=official['score'], mase=official['mase'], early=100*(1-errors[:,:7].mean()),
            late=100*(1-errors[:,7:].mean()), n_scored=int(ok.sum()), excluded=int((~ok).sum()),
            unit_coverage=official['holdout_units_share_scored'], recent=int(recent.sum()),
            matched=int(matched.sum()), changed=int(np.any(candidate!=control,axis=1).sum()),
            recent_mase=float(errors[recent[ok]].mean())))
        for dimension in ['observed_activation_age','activity13','last_stock','stock_missing13']:
            scored_labels = labels[dimension][ok]
            for label in np.unique(scored_labels):
                mask = scored_labels == label
                groups.append(dict(offset=offset, variant=name, dimension=dimension, group=label,
                    n=int(mask.sum()), mase=errors[mask].mean()))
    for row in matches:
        row['article_id'] = matrix.keys.article_id.iloc[row.pop('target')]
        row['offset'] = offset
    return scores, groups, matches


def main(out):
    cfg = json.loads(CONTRACT.read_text())
    primary = cfg['primary_development_offsets']
    extra = [w for w in cfg['purged_forward_check_offsets'] if w not in primary]
    locked = pd.read_csv(CONTRACT.parent/'results/scores.csv').set_index('offset').score
    full = validate.load_weekly('test')
    scores, groups, matches, hashes = [], [], [], {}
    for offset in primary + extra:
        rows, partitions, peers = evaluate_cut(full, offset, offset in primary, locked.loc[offset], hashes)
        scores.extend(rows)
        groups.extend(partitions)
        matches.extend(peers)
    scores = pd.DataFrame(scores)
    paired = scores[scores.variant=='champion'].set_index('offset')[['score','late','recent_mase']].join(
        scores[scores.variant=='analogue'].set_index('offset')[['score','late','recent_mase','recent','matched','changed']],
        lsuffix='_champion', rsuffix='_analogue')
    paired['gain'] = paired.score_analogue - paired.score_champion
    paired['late_gain'] = paired.late_analogue - paired.late_champion
    gain = paired.loc[primary,'gain']
    late_gain = paired.loc[primary,'late_gain'].mean()
    threshold = cfg['promotion_review']
    passed = (gain.mean() >= threshold['minimum_primary_mean_gain_score_points'] and
              (gain > 0).sum() >= threshold['minimum_primary_periods_improved'] and
              late_gain >= 0 and (paired.loc[extra,'gain'] >= 0).all())
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = dict(primary_offsets=primary, additional_offsets=extra, primary_gain=float(gain.mean()),
        primary_wins=int((gain > 0).sum()), late_gain=float(late_gain), review_pass=bool(passed),
        numpy=np.__version__, pandas=pd.__version__, python=sys.version.split()[0], source_blobs=hashes,
        source_commits=[review.SOURCE, diagnose.ADDITIONAL_SOURCE],
        hashes={str(p.relative_to(ROOT)): digest(p) for p in
                [HERE/'analogues.py', HERE/'backtest.py', CONTRACT, validate.IM/'weekly_test.parquet',
                 ROOT/'submissions/ensemble_top3_blend_v1.csv']})
    out.mkdir(parents=True, exist_ok=False)
    scores.to_csv(out/'scores.csv', index=False)
    paired.to_csv(out/'paired.csv')
    pd.DataFrame(groups).to_csv(out/'groups.csv', index=False)
    pd.DataFrame(matches).to_csv(out/'matches.csv', index=False)
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(paired.round(4).to_string())
    print(f'Primary gain {gain.mean():+.4f}, late {late_gain:+.4f}, passed {passed}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'results')
    main(parser.parse_args().output)
