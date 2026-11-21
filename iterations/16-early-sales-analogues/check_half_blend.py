"""One exploratory 50/50 blend of the published integer peer and baseline forecasts."""
import numpy as np
import pandas as pd

import backtest as b


def compare():
    full = b.validate.load_weekly('test')
    frozen = pd.read_csv(b.HERE/'results/paired.csv').set_index('offset')
    rows = []
    for offset, original in frozen.iterrows():
        prefix = full[full.week <= full.week.max() - pd.Timedelta(days=7*offset)]
        history, holdout, cut = b.validate.split(prefix)
        matrix = b.features.ShopMatrix(prefix[prefix.week < cut])
        weeks = pd.date_range(cut, periods=13, freq='7D')
        parts, tree = b.read_components(offset, matrix.keys, weeks, offset in [0,13,26,39], {})
        champion = b.review.compose(parts, tree, matrix.U[:, -8:].sum(axis=1) == 0)
        peers, _ = b.peer_forecast(matrix.U, champion)
        mixed = np.rint((peers + champion)/2)
        actual = b.review.matrix(holdout.rename(columns={'units':'forecast'}),matrix.keys,weeks)
        scale = b.validate.history_scale(history).reindex(pd.MultiIndex.from_frame(matrix.keys)).to_numpy()
        ok = scale > 0
        def errors(prediction):
            return np.abs(prediction[ok]-actual[ok])/scale[ok,None]
        original_error = errors(champion)
        peer_gain = 100*(original_error-errors(peers)).mean(axis=1)/ok.sum()
        np.testing.assert_allclose(peer_gain.sum(), original.gain, atol=1e-9)
        official = b.validate.score(history,holdout,b.train_gbm.to_long(matrix,mixed,weeks))['score']
        rows.append(dict(offset=offset,start=str(cut.date()),full_peer_gain=peer_gain.sum(),
                         half_gain=official-original.score_champion,
                         half_late_gain=100*(original_error[:,7:]-errors(mixed)[:,7:]).mean(),
                         best_single_product_full_peer_gain=peer_gain.max(),
                         full_peer_gain_without_best_contribution=peer_gain.sum()-peer_gain.max()))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    results = compare()
    results.to_csv(b.HERE/'results/half_blend.csv', index=False)
    print(results.round(4).to_string(index=False))
