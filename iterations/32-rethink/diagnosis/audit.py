"""Summarize current-model errors against pre-origin article unit granularity."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT/'iterations/31-demand-distribution'))
import score as s

rows = []
lifecycle = []
full = s.v.load_weekly("test")
for offset in [0,13,26,39]:
    frame = full[full.week <= full.week.max() - pd.Timedelta(weeks=offset)]
    train, hold, cut = s.v.split(frame)
    grid = hold.set_index(s.KEY).sort_index()
    U = train.pivot(index=s.v.KEYS,columns="week",values="units").sort_index().to_numpy()
    Y = grid.units.to_numpy().reshape(-1,13)
    scale = np.abs(np.diff(U)).mean(axis=1)
    ok = scale > 0
    suffix = f'cut{offset//13}'
    parts = [s.matrix(s.blob(s.FROZEN, f'iterations/03-foundation-model/forecasts/{m}__test__{suffix}__padded.parquet'), grid.index) for m in s.FM]
    parts.append(s.matrix(s.SOURCE / f'chronos-2-ft__test__{suffix}__padded.parquet', grid.index))
    tree = s.matrix(s.OUT / f'test_{offset}.parquet',grid.index)
    F = s.combine(parts,tree,U[:,-8:].sum(axis=1)==0)
    loss = np.abs(Y[ok]-F[ok])/scale[ok,None]
    known = pd.read_csv(s.OUT/'target_metrics.csv').query("offset == @offset and model=='distribution'").mase.item()
    np.testing.assert_allclose(loss.mean(), known)
    U,Y,F,scale=U[ok],Y[ok],F[ok],scale[ok]
    recent=U[:,-52:]
    g=np.gcd.reduce(np.rint(recent).astype('int64'),axis=1)
    eligible=((recent==np.rint(recent)).all(axis=1) & ((recent>0).sum(axis=1)>=5) & (g>1))
    count=(U>0).sum(axis=1)
    age=U.shape[1]-(U>0).argmax(axis=1)
    # Cataloguing the proportion of actual and forecast cells on the inferred support.
    for name,mask in [('all',np.ones(len(U),bool)),('pack',eligible),('recent',age<=13),
                      ('pack_recent', eligible & (age<=13)),('nonpack_recent',~eligible & (age<=13))]:
        E=loss[mask]
        z=Y[mask];p=F[mask];gu=np.maximum(1,g[mask,None]); off=p%gu!=0
        wt=1/scale[mask,None]
        rows.append(dict(offset=offset,cut=cut.date(),group=name,articles=int(mask.sum()),share=mask.mean(),
                         loss_share=E.sum()/loss.sum(),mase=E.mean(),pred_off_lattice=off.mean(),
                         actual_off_lattice=(z%gu!=0).mean(),mean_pack=np.median(gu),
                         off_positive_accuracy=(z[off]>p[off]).mean() if off.any() else np.nan,
                         error_above=((z-p).clip(min=0)*wt).mean(),error_below=((p-z).clip(min=0)*wt).mean()))
    last_stock = train[train.week == train.week.max()].set_index(s.v.KEYS).sort_index().on_hand.to_numpy()[ok]
    for label, mask in [('recent',age<=13),('recent_age1_4',age<=4),('recent_age5_13',(age>4)&(age<=13)),
                        ('recent_pos1',(age<=13)&(count==1)),('recent_pos2_3',(age<=13)&(count>=2)&(count<=3)),
                        ('recent_pos4plus',(age<=13)&(count>=4)),('recent_lastpos',(age<=13)&(U[:,-1]>0)),
                        ('recent_stockpos',(age<=13)&(last_stock>0)),
                        ('recent_stockzero',(age<=13)&(last_stock==0))]:
        y,p=Y[mask],F[mask]
        w=np.broadcast_to(1/scale[mask,None],y.shape)
        # Derivative of MAE aggregates different scales; a tied cell resists both directions.
        lifecycle.append(dict(offset=offset,cut=cut.date(),group=label,n=int(mask.sum()),
                          error_share=loss[mask].sum()/loss.sum(),zero_pred=(p==0).mean(),
                          positive_actual=(y>0).mean(),weighted_actual_above=np.average(y>p,weights=w),
                          weighted_actual_below=np.average(y<p,weights=w),
                          above_when_pos=np.average(y[p>0]>p[p>0],weights=w[p>0]) if (p>0).any() else np.nan,
                          above_when_zero=np.average(y[p==0]>0,weights=w[p==0]) if (p==0).any() else np.nan))
    # Compare directional outcomes when a forecast falls strictly between known support values.
    gu=np.maximum(1,g[:,None]);fr=(F%gu)/gu
    for name,mask in [('lower_half', (fr>0)&(fr<.5)),('midpoint',fr==.5),('upper_half',fr>.5)]:
        mask=mask&eligible[:,None]
        wt=np.broadcast_to(1/scale[:,None],Y.shape)[mask]
        if mask.any():
            rows.append(dict(offset=offset,cut=cut.date(),group=name,articles=int(mask.any(axis=1).sum()),
                             share=mask.mean(),loss_share=loss[mask].sum()/loss.sum(),mase=loss[mask].mean(),
                             scaled_up_frequency=np.average(Y[mask]>F[mask],weights=wt),
                             up_frequency=np.mean(Y[mask]>F[mask])))
pd.DataFrame(rows).to_csv(OUT/'support-audit.csv',index=False)
print(pd.DataFrame(rows).groupby('group').mean(numeric_only=True).round(4).to_string())

pd.DataFrame(lifecycle).to_csv(OUT/"lifecycle-audit.csv",index=False)
