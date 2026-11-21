import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/adomi/Documents/hackathlon/src')
import pandas as pd, numpy as np, validate as v, baselines as b
def leading_zero_grid(df):
    series=df[v.KEYS].drop_duplicates(); weeks=pd.DataFrame({'week':sorted(df.week.unique())})
    g=series.merge(weeks,how='cross').merge(df[v.KEYS+['week','units']],on=v.KEYS+['week'],how='left'); g['units']=g.units.fillna(0.0); return g
brief={('B','autoarima'):20.4,('B','topdown_mstl'):20.4,('C','autoarima'):41.8,('C','topdown_mstl'):37.7,('D','autoarima'):57.2,('D','topdown_mstl'):44.9}
def main():
  for shop,models in [('C',['zero','topdown_mstl','autoarima']),('D',['zero','topdown_mstl'])]:
      df=leading_zero_grid(v.load_weekly(shop)); train,hold,cut=v.split(df); weeks=pd.date_range(cut,periods=13,freq='7D')
      for level in ['series','article']:
          tr=v.to_article_level(train) if level=='article' else train
          P=b.wide(tr[v.KEYS+['week','units']]); mk=b.baselines(P,weeks)
          for name in models:
              fc=b.complete(mk[name](), v.to_article_level(hold) if level=='article' else hold)
              r=v.score(train,hold,fc,level=level,scale_mode='full')
              print(f"{shop} {level:7s} {name:13s} leading-zero grid: score {r['score']:6.1f} scored {r['n_scored']:6d}/{r['n_series']:6d} units-covered {r['holdout_units_share_scored']:.2f}  | brief {brief.get((shop,name),'-')}",flush=True)

if __name__=='__main__':
    main()
