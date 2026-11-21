"""Fixed zero-mass plus conditional lognormal marginal forecast distribution."""
import gc
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import ndtri
import features as f
import train_gbm as g

PARAMS = {'sale': dict(g.PARAMS, objective='binary'),
          'size': dict(g.PARAMS, objective='regression_l2')}


def median(p, mu, sigma):
    p, mu = np.broadcast_arrays(np.asarray(p, float), np.asarray(mu, float))
    if not (np.isfinite(p).all() and np.isfinite(mu).all() and np.isfinite(sigma)
            and sigma >= 0 and ((p >= 0) & (p <= 1)).all()):
        raise ValueError('Invalid hurdle distribution parameters')
    result = np.zeros_like(p)
    positive = p > .5
    q = 1 - .5 / p[positive]
    result[positive] = np.exp(mu[positive] + sigma * ndtri(q))
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite hurdle median')
    return result


def transform(w, kind):
    keep = w['y'] > 0 if kind == 'size' else np.ones(len(w['y']), bool)
    target = np.log(w['y'][keep]) if kind == 'size' else (w['y'][keep] > 0).astype(float)
    if len(target) == 0:
        raise ValueError('No eligible training observations')
    return dict(X=w['X'][keep], y=target, w=w['w'][keep])


def train(M, folder):
    start = time.time()
    origins = f.make_origins(M.W)
    val = origins[-7:]
    tr = origins[origins <= val[0] - 13]
    assert len(tr) and tr.max() + 13 <= val.min() and origins.max() + 13 <= M.W
    wt, wv = f.build_windows(M, tr), f.build_windows(M, val)
    bests, sigma = {}, None
    for kind in ['sale', 'size']:
        a, b = transform(wt, kind), transform(wv, kind)
        model, best = g.fit(a, b, params=PARAMS[kind])
        bests[kind] = int(best)
        if kind == 'size':
            residual = b['y'] - model.predict(b['X'])
            sigma = float(np.sqrt(np.average(residual ** 2, weights=b['w'])))
        del a, b, model
        gc.collect()
    del wt, wv
    gc.collect()
    wa = f.build_windows(M, origins)
    X, scale, valid = f.inference_features(M)
    values, importance = {}, []
    for kind in ['sale', 'size']:
        data = transform(wa, kind)
        rounds = int(bests[kind] * 1.1) + 1
        model, _ = g.fit(data, rounds=rounds, params=PARAMS[kind])
        model.save_model(str(folder / f'{kind}_model.txt'))
        values[kind] = model.predict(X).reshape(M.n, 13)
        importance.append(pd.DataFrame({'feature':f.FEATURES,
                          'gain':model.feature_importance('gain'), 'component':kind}))
        del model, data
        gc.collect()
    pred = median(values['sale'], values['size'], sigma) * scale[:, None]
    pred[~valid] = 0
    assert np.isfinite(pred).all() and (pred >= 0).all()
    np.save(folder/'forecast.npy', pred, allow_pickle=False)
    np.savez_compressed(folder/'distribution.npz', probability=values['sale'],
                        mu=values['size'], sigma=sigma, scale=scale, valid=valid)
    M.keys.to_csv(folder/'keys.csv',index=False)
    pd.concat(importance).to_csv(folder/'importance.csv',index=False)
    info = dict(best_iterations=bests, final_rounds={k:int(v*1.1)+1 for k,v in bests.items()},
                sigma=sigma, seconds=time.time()-start, training_origins=tr.tolist(),
                validation_origins=val.tolist(), final_origins=origins.tolist())
    (folder/'training.json').write_text(json.dumps(info,indent=2))
    return pred, info
