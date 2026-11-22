"""Fit target development/final models and own-shop/pooled A–D models."""
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import distribution as dist
import features as f
import train_gbm as gb
import validate as v

OUT = dist.HERE / "results"
TARGET = [0, 13, 26, 39, 16, 32, "final"]


def history(shop, fold):
    df = v.load_weekly(shop)
    cut = (df.week.max() + pd.Timedelta(weeks=1) if fold == "final"
           else v.cut_week(df) - pd.Timedelta(weeks=fold))
    # Include known assortment keys for zero-history forecasts, as in D26.
    train = v.pad_to_shop_start(df[df.week < cut], df[v.KEYS].drop_duplicates())
    matrix = f.ShopMatrix(train)
    ledger = train.on_hand.mask(train.on_hand_missing | train.on_hand_was_negative)
    matrix.ledger = train.assign(ledger=ledger).pivot(index=v.KEYS, columns="week", values="ledger").reindex(
        index=pd.MultiIndex.from_frame(matrix.keys), columns=matrix.weeks).to_numpy(dtype=float)
    return matrix, cut


def save(model, matrix, levels, cut, name):
    forecast = dist.medians(model, matrix, levels)
    frame = gb.to_long(matrix, forecast, pd.date_range(cut, periods=13, freq="7D"))
    frame.to_parquet(OUT / f"{name}.parquet", index=False)
    print(name, cut.date(), "saved", flush=True)


def run_target():
    for fold in TARGET:
        matrix, cut = history("test", fold)
        sample = dist.examples(matrix)
        model = dist.fit([sample])
        print("target", fold, "training rows", len(sample[1]), flush=True)
        if fold == "final":
            model.save_model(str(OUT / "target_full_model.txt"))
        save(model, matrix, dist.support([matrix]), cut, f"test_{fold}")
        del sample, model, matrix
        gc.collect()


def run_shops():
    inputs, samples = {}, []
    for shop in "ABCD":
        matrix, cut = history(shop, 0)
        sample = dist.examples(matrix)
        model = dist.fit([sample])
        print(shop, "training rows", len(sample[1]), flush=True)
        save(model, matrix, dist.support([matrix]), cut, f"{shop}_own")
        inputs[shop] = matrix, cut
        samples.append(sample)
    model = dist.fit(samples)
    levels = dist.support([m for m, _ in inputs.values()])
    for shop, (matrix, cut) in inputs.items():
        save(model, matrix, levels, cut, f"{shop}_pooled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["target", "shops"])
    if parser.parse_args().stage == "target":
        run_target()
    else:
        run_shops()
