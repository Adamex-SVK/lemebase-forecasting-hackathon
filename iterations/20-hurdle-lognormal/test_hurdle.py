"""Verify mixture median, inverse CDF, simulation, input checks and temporal features."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
from scipy.special import ndtr
import hurdle
import features as f
import validate as v

p=np.array([0.,.2,.5,.5001,.75,1.])
mu=np.zeros(6); sigma=.7
med=hurdle.median(p,mu,sigma)
np.testing.assert_array_equal(med[:3],0)
np.testing.assert_allclose((1-p[3:])+p[3:]*ndtr(np.log(med[3:])/sigma),.5,atol=1e-12)
np.testing.assert_allclose(med[-1],1.)
np.testing.assert_allclose(hurdle.median([.4,.8],[2.,2.],0),[0,np.exp(2)])
rng=np.random.default_rng(0)
samples=np.where(rng.random(300000)<.75,rng.lognormal(0,.7,300000),0)
assert abs(np.median(samples)-med[-2]) < .01
for probability in [-.1,1.1,np.nan]:
    try: hurdle.median(probability,0.,.7)
    except ValueError: pass
    else: raise AssertionError('Invalid probability accepted')
# Changes at or after origin must not change any origin feature or its scale.
d=v.load_weekly('test');M=f.ShopMatrix(d);o=60
before=f.features_at(M,o)
M.U[:,o:]+=1000;M.OH[:,o:]=1000;M.OOS[:,o:]=True
after=f.features_at(M,o)
for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
print('Passed: mixture median/CDF, sigma-zero edge, Monte Carlo check, input validation, feature timing.')
