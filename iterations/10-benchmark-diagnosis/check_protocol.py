"""Materialize and validate the phase-1 label-window purge contract, without scoring reserved windows."""
import json
from pathlib import Path

import pandas as pd

HERE=Path(__file__).resolve().parent


def eligible_offsets(available,test_offset,horizon=13):
    return sorted(w for w in available if w>=test_offset+horizon)


def main():
    cfg=json.loads((HERE/"validation.json").read_text())
    inventory=pd.read_csv(HERE/"results/forecast_inventory.csv")
    available=inventory.loc[inventory.complete,"offset"].to_list()
    rows=[]
    for test in cfg["purged_forward_check_offsets"]:
        testrow=inventory.set_index("offset").loc[test]
        for train in eligible_offsets(available,test,cfg["horizon_weeks"]):
            trainrow=inventory.set_index("offset").loc[train]
            assert pd.Timestamp(trainrow.holdout_end)<pd.Timestamp(testrow.holdout_start)
            rows.append(dict(test_offset=test,train_offset=train,train_label_end=trainrow.holdout_end,
                             test_label_start=testrow.holdout_start))
    targets=inventory[inventory.offset.isin(cfg["purged_forward_check_offsets"])].sort_values("holdout_start")
    dates=[set(pd.date_range(r.holdout_start,r.holdout_end,freq="7D")) for r in targets.itertuples()]
    assert all(not dates[i]&dates[j] for i in range(len(dates)) for j in range(i))
    assert not cfg["untouched_forward_confirmation_available"]
    pd.DataFrame(rows).to_csv(HERE/"results/purged_forward_folds.csv",index=False)
    print("Verified disjoint forward-check targets and",len(rows),"eligible purged train/test label-window pairs")


if __name__=="__main__":
    main()
