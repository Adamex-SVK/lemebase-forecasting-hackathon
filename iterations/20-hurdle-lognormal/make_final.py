"""User-requested local full-history challenger CSV; never uploads."""
import io
import json
import hashlib
import numpy as np
import pandas as pd
from run import ROOT, HERE, f, v, review, hurdle, sha
from make_submission import validate_submission

folder=HERE/'final'
folder.mkdir(exist_ok=False)
locked=ROOT/'submissions/ensemble_top3_blend_v1.csv'
locked_hash=sha(locked)
M=f.ShopMatrix(v.load_weekly('test'))
weeks=pd.date_range(pd.Timestamp(M.weeks[-1])+pd.Timedelta(days=7),periods=13,freq='7D')
assert weeks[0]==pd.Timestamp('2023-04-24') and weeks[-1]==pd.Timestamp('2023-07-17')
cols=[f'w{i}' for i in range(1,14)]
hashes={};parts=[]
for name in review.MODELS:
    path=f'iterations/03-foundation-model/forecasts_final/{name}__test__final__padded.parquet'
    parts.append(review.matrix(pd.read_parquet(io.BytesIO(review.load_blob(review.SOURCE,path,hashes))),M.keys,weeks))
raw=review.load_blob(review.SOURCE,'submissions/lgbm_own_v1.csv',hashes)
old=pd.read_csv(io.BytesIO(raw)).set_index('article_id',verify_integrity=True).loc[M.keys.article_id,cols].to_numpy()
champion=pd.read_csv(locked).set_index('article_id',verify_integrity=True).loc[M.keys.article_id,cols].to_numpy()
dead=M.U[:,-8:].sum(axis=1)==0
np.testing.assert_array_equal(review.compose(parts,old,dead),champion)
pred,info=hurdle.train(M,folder)
result=review.compose(parts,pred,dead)
assert np.isfinite(result).all() and (result>=0).all()
frame=pd.DataFrame(result.astype(np.int64),columns=cols)
frame.insert(0,'article_id',M.keys.article_id.to_numpy())
target=ROOT/'submissions/ensemble_top3_hurdle_lognormal30_v1.csv'
frame.to_csv(target,index=False,mode='x')
validate_submission(target)
assert sha(locked)==locked_hash
metadata=dict(training=info,params=hurdle.PARAMS,source_sha256=hashes,data_sha256=sha(v.IM/'weekly_test.parquet'),
    output=str(target.relative_to(ROOT)),output_sha256=sha(target),champion_sha256=locked_hash,
    changed_cells=int((result!=champion).sum()),n_articles=len(frame),future_weeks=[str(w.date()) for w in weeks],
    warning='Challenger failed historical promotion gate; generated at user request; no upload.',
    script_sha256=sha(HERE/'make_final.py'),hurdle_sha256=sha(HERE/'hurdle.py'))
(folder/'metadata.json').write_text(json.dumps(metadata,indent=2))
print(target)
print('Validated',len(frame),'articles; changed cells',metadata['changed_cells'])
