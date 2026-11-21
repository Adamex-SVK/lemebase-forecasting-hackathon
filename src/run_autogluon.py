"""Bounded own-shop AutoGluon challenger; no submissions or implicit model downloads.

Example: .venv-autogluon/bin/python src/run_autogluon.py --shops test --cuts 0
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor

from autogluon_adapter import ChallengeMASE, from_predictions, to_timeseries, validation_windows
import validate as v

DEFAULT_OUT = v.ROOT / "iterations/06-autogluon/runs"


def model_config(names, cpus):
    # Explicit shortlist: no broad preset, no external model weights.
    models = {
        "Naive": {"n_jobs": cpus},
        "AutoETS": {"seasonal_period": 1, "n_jobs": cpus},
        "DirectTabular": {"lags": [1, 2, 4, 8, 13], "date_features": [],
                          "max_num_samples": 100_000, "max_num_items": None,
                          "model_hyperparameters": {"num_boost_round": 200, "num_threads": cpus}},
        "DeepAR": {"max_epochs": 5, "num_batches_per_epoch": 20,
                   "batch_size": 32, "context_length": 26,
                   "trainer_kwargs": {"accelerator": "cpu", "devices": 1}},
    }
    return {name: {**models[name], "ag_args_fit": {"num_cpus": cpus, "num_gpus": 0}} for name in names}


def median_forecast(train, hold):
    last = train.week.max() - pd.Timedelta(weeks=12)
    med = train[train.week >= last].groupby(v.KEYS).units.median().rename("forecast")
    return hold[v.KEYS + ["week"]].merge(med.reset_index(), on=v.KEYS, validate="many_to_one")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shops", nargs="+", choices=["A", "B", "C", "D", "test"], default=["test"])
    parser.add_argument("--cuts", nargs="+", type=int, choices=range(4), default=[0, 1, 2, 3])
    parser.add_argument("--models", nargs="+", choices=["Naive", "AutoETS", "DirectTabular", "DeepAR"],
                        default=["Naive", "AutoETS", "DirectTabular"])
    parser.add_argument("--time-limit", type=int, default=300, help="Soft AutoGluon fit budget per shop/cut, seconds")
    parser.add_argument("--val-windows", type=int, default=3)
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-series", type=int, help="Seeded subset for plumbing checks; not a full benchmark")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.time_limit, args.val_windows, args.cpus) < 1 or (args.limit_series is not None and args.limit_series < 1):
        parser.error("Budgets, windows, CPUs, and optional series limit must be positive")
    import torch
    torch.set_num_threads(args.cpus)
    out = args.output or DEFAULT_OUT / time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    out.mkdir(parents=True, exist_ok=False)
    hp = model_config(args.models, args.cpus)
    cfg = vars(args).copy()
    cfg["output"] = str(out)
    cfg.update(model_hyperparameters=hp, python=platform.python_version(),
               autogluon=importlib.metadata.version("autogluon.timeseries"),
               git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=v.ROOT, text=True).strip(),
               metric="D26; adjacent-week scale; constant histories excluded; nonnegative median point forecasts",
               scope="own-shop only, units only; not pooled or unseen-shop transfer")
    cfg["source_sha256"] = {str(p.relative_to(v.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__), Path(__file__).with_name("autogluon_adapter.py"),
                                       Path(__file__).with_name("validate.py")]}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    freeze = subprocess.check_output([__import__("sys").executable, "-m", "pip", "freeze"], text=True)
    (out / "environment.txt").write_text(freeze)
    rows, statuses, hashes = [], [], {}
    for shop in args.shops:
        source = v.IM / f"weekly_{shop}.parquet"
        hashes[str(source.relative_to(v.ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        full = v.load_weekly(shop)
        if args.limit_series:
            keys = full[v.KEYS].drop_duplicates().sort_values(v.KEYS)
            keys = keys.sample(n=min(args.limit_series, len(keys)), random_state=args.seed)
            full = full.merge(keys, on=v.KEYS, validate="many_to_one")
        for cut_index in args.cuts:
            job = out / f"{shop}_cut{cut_index}"
            job.mkdir()
            d = full[full.week <= full.week.max() - pd.Timedelta(weeks=13*cut_index)]
            n_train = d.week.nunique() - v.HORIZON
            windows = validation_windows(n_train, args.val_windows)
            status = dict(shop=shop, cut=cut_index, train_weeks=n_train, internal_val_windows=windows)
            if not windows:
                status.update(status="skipped", reason="Too little history for an internal AG validation window")
                statuses.append(status)
                (out / "status.json").write_text(json.dumps(statuses, indent=2))
                continue
            train, hold, cut = v.split(d)
            ts, mapping = to_timeseries(train, shop)
            mapping.to_csv(job / "series_mapping.csv", index=False)
            rows.append(dict(shop=shop, cut=cut_index, model="median13", training_data="own",
                             subset=args.limit_series is not None, **v.score(train, hold, median_forecast(train, hold))))
            reference = v.ROOT / f"logs/chronos_four/LightGBM_{cut_index}.parquet"
            if shop == "test" and reference.exists():
                saved = pd.read_parquet(reference)
                saved = hold[v.KEYS + ["week"]].merge(saved, on=v.KEYS + ["week"],
                                                       how="left", validate="one_to_one")
                rows.append(dict(shop=shop, cut=cut_index, model="reference_lgbm_saved", training_data="own",
                                 subset=args.limit_series is not None, **v.score(train, hold, saved)))
                cfg.setdefault("reference_sha256", {})[str(reference.relative_to(v.ROOT))] = hashlib.sha256(reference.read_bytes()).hexdigest()
            start = time.time()
            try:
                predictor = TimeSeriesPredictor(path=str(job / "models"), prediction_length=13,
                    target="target", freq="W-MON", eval_metric=ChallengeMASE(), eval_metric_seasonal_period=1,
                    quantile_levels=[0.5], verbosity=2)
                predictor.fit(ts, hyperparameters=hp, time_limit=args.time_limit,
                              num_val_windows=windows, val_step_size=13, random_seed=args.seed,
                              enable_ensemble=len(hp) > 1, refit_full=False)
                # Internal-only ranking; do not choose models/weights on the outer holdout.
                lb = predictor.leaderboard()
                lb.to_csv(job / "internal_leaderboard.csv", index=False)
                best = predictor.model_best
                names = predictor.model_names()
                successful = []
                for name in names:
                    try:
                        predictions = predictor.predict(ts, model=name, random_seed=args.seed)
                        fc = from_predictions(predictions, mapping, hold)
                        fc.to_parquet(job / f"forecast_{name}.parquet", index=False)
                        r = v.score(train, hold, fc, detail=True)
                        r.pop("detail").reset_index().to_csv(job / f"errors_{name}.csv", index=False)
                        rows.append(dict(shop=shop, cut=cut_index, model=name, training_data="own",
                                         selected_internally=name == best, subset=args.limit_series is not None,
                                         seconds=time.time()-start, **r))
                        successful.append(name)
                        print(f"{shop} cut {cut_index} {name}: score {r['score']:.3f}; selected={name == best}", flush=True)
                    except Exception as error:
                        statuses.append(dict(shop=shop, cut=cut_index, model=name,
                                             status="prediction_failed", reason=repr(error)))
                status.update(status="complete" if len(successful) == len(names) else "partial",
                              selected_model=best, trained_models=names, predicted_models=successful)
                if best not in successful:
                    raise RuntimeError("Internally selected model did not produce valid forecasts")
            except Exception as error:
                status.update(status="failed", reason=repr(error))
            statuses.append(status)
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
            (out / "status.json").write_text(json.dumps(statuses, indent=2))
    cfg["input_sha256"] = hashes
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    if rows:
        pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    (out / "status.json").write_text(json.dumps(statuses, indent=2))
    if any(s["status"] in ("failed", "partial", "prediction_failed") for s in statuses):
        raise SystemExit(f"Some runs failed; see {out / 'status.json'}")
    if not any(s["status"] == "complete" for s in statuses):
        raise SystemExit("No model run completed; all requested cuts were skipped")
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
