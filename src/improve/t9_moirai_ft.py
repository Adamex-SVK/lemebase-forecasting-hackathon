"""T9: fine-tune Moirai 1.1-small on the test shop and forecast the 4 standard cuts + final.

Runs only on the GPU machine (uni2ts 2.0.0 in ~/hack/.venv-moirai; torch 2.4.1). Self-contained so
`src/foundation/models.py` is not touched.

Leakage rule: for cut K the training series are the test shop's padded weekly history that ends
*before* cut K's holdout (run_model.contexts_for does the split); `--final` uses all history.
Hyper-parameters were fixed before any score was seen (see iterations/08-improvements/t9/notes_*).

    python src/improve/t9_moirai_ft.py run    --cut 0 --workdir ~/hack-t9/runs/cut0
    python src/improve/t9_moirai_ft.py run    --final --workdir ~/hack-t9/runs/final
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "foundation"))
import run_model as rm  # noqa: E402
import validate as v  # noqa: E402

HF_ID = "Salesforce/moirai-1.1-R-small"
MODEL_NAME = "moirai-1.1-small-ft"
CONTEXT = 65          # weeks of context per training window (cut3 has 91 padded weeks)
PATCH = 16            # fixed patch size (one of the module's patch sizes)
DIST = 1              # window stride
STEPS = 400           # optimiser steps
BATCH = 32
LR = 1e-4
WARMUP = 40
WD = 0.01
NUM_SAMPLES = 50      # forecast quantile samples (same as the zero-shot adapter)
SEED = 0


def build_dataset_entry(A, start, path):
    import datasets
    from datasets import Features, Sequence, Value

    features = Features(dict(item_id=Value("string"), start=Value("timestamp[s]"),
                             freq=Value("string"), target=Sequence(Value("float32"))))
    target = [row.astype("float32") for row in A]

    def gen():
        for i, t in enumerate(target):
            yield {"item_id": f"s{i}", "start": start, "freq": "7D", "target": t}

    ds = datasets.Dataset.from_generator(gen, features=features)
    ds.save_to_disk(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cut", type=int, choices=[0, 1, 2, 3])
    g.add_argument("--final", action="store_true")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--no-fit", action="store_true", help="skip fine-tuning (zero-shot control run)")
    ap.add_argument("--name", default=None, help="model name for the output files")
    a = ap.parse_args()

    import lightning as L
    import torch
    from torch.utils.data import DataLoader
    from uni2ts.data.dataset import FinetuneDataset
    from uni2ts.data.indexer import HuggingFaceDatasetIndexer
    from uni2ts.data.loader import PadCollate
    from uni2ts.model.moirai import MoiraiFinetune, MoiraiModule
    import datasets

    L.seed_everything(SEED, workers=True)
    work = Path(a.workdir).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    index, ctx, active, hold, cut, train = rm.contexts_for("test", a.cut or 0, "padded", final=a.final)
    A = np.stack(ctx)
    start = pd.Timestamp(train["week"].min())
    T = A.shape[1]
    alive = np.array([row.sum() > 0 for row in A])
    A = A[alive]
    windows = T - CONTEXT - v.HORIZON + 1
    assert windows >= 1, (T, CONTEXT)
    print(f"train series {len(A)}/{len(ctx)}; T={T}; windows={windows}; end={start + pd.Timedelta(weeks=T-1)}", flush=True)

    entry = work / "hf_dataset"
    build_dataset_entry(A, start, entry)
    idx = HuggingFaceDatasetIndexer(datasets.load_from_disk(str(entry)))

    model = MoiraiFinetune(
        min_patches=2, min_mask_ratio=0.15, max_mask_ratio=0.5, max_dim=1,
        num_training_steps=a.steps, num_warmup_steps=min(WARMUP, a.steps),
        module=MoiraiModule.from_pretrained(HF_ID),
        context_length=CONTEXT, prediction_length=v.HORIZON, patch_size=PATCH,
        finetune_pattern="full", lr=LR, weight_decay=WD,
    )
    transform = model.train_transform_map["default"](distance=DIST, prediction_length=v.HORIZON,
                                                    context_length=CONTEXT, patch_size=PATCH)
    ds = FinetuneDataset(windows=windows, indexer=idx, transform=transform)
    probe = ds[0]
    max_length = len(probe["target"])
    collate = PadCollate(max_length=max_length, seq_fields=model.seq_fields,
                         pad_func_map=model.pad_func_map, target_field="target")
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=4,
                    persistent_workers=False)
    print(f"dataset len {len(ds)}, patch seq len {max_length}", flush=True)

    trainer = L.Trainer(accelerator="gpu", devices=1, max_steps=a.steps, max_epochs=-1,
                        logger=False, enable_checkpointing=False, enable_progress_bar=False,
                        num_sanity_val_steps=0, log_every_n_steps=50, use_distributed_sampler=False,
                        enable_model_summary=False)
    if not a.no_fit:
        model.train()
        trainer.fit(model, train_dataloaders=dl)
    t1 = time.time()
    module = MoiraiModule.from_pretrained(HF_ID)
    if not a.no_fit:
        torch.save(model.module.state_dict(), work / "module.pt")
        module.load_state_dict(torch.load(work / "module.pt", map_location="cuda"))
    module.eval().cuda()
    from uni2ts.model.moirai import MoiraiForecast
    Lc = int(max(len(c) for c in ctx))
    fc = MoiraiForecast(module=module, prediction_length=v.HORIZON, context_length=Lc, patch_size="auto",
                        num_samples=NUM_SAMPLES, target_dim=1, feat_dynamic_real_dim=0,
                        past_feat_dynamic_real_dim=0)
    predictor = fc.create_predictor(batch_size=64)
    from gluonts.dataset.pandas import PandasDataset
    end_week = cut - pd.Timedelta(weeks=1)
    series = {f"s{k}": pd.Series(c.astype("float32"),
                                 index=pd.date_range(end=end_week, periods=len(c), freq="W-MON"))
              for k, c in enumerate(ctx)}
    gds = PandasDataset(series, freq="W-MON")
    out = np.zeros((len(index), v.HORIZON))
    for k, f in enumerate(predictor.predict(gds)):
        try:
            out[k] = np.asarray(f.quantile(0.5))
        except Exception:
            out[k] = np.asarray(f.median)
    out = np.clip(np.nan_to_num(out, nan=0.0), 0.0, None)
    t2 = time.time()

    weeks = pd.date_range(cut, periods=v.HORIZON, freq="7D")
    df = pd.DataFrame({"store_id": np.repeat(index.get_level_values(0).to_numpy(), v.HORIZON),
                       "article_id": np.repeat(index.get_level_values(1).to_numpy(), v.HORIZON),
                       "week": np.tile(weeks, len(index)), "forecast": out.reshape(-1)})
    tag = "final" if a.final else f"cut{a.cut}"
    out_dir = ROOT / "iterations" / "03-foundation-model" / ("forecasts_final" if a.final else "forecasts")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{a.name or MODEL_NAME}__test__{tag}__padded"
    df.to_parquet(out_dir / f"{stem}.parquet", index=False)
    meta = dict(model=a.name or MODEL_NAME, base=HF_ID, tag=tag, context_padded_weeks=T, train_series=int(len(A)),
                windows=windows, steps=a.steps, batch=BATCH, lr=LR, warmup=WARMUP, wd=WD,
                context_length=CONTEXT, patch_size=PATCH, distance=DIST, num_samples=NUM_SAMPLES,
                train_last_week=str(train["week"].max().date()),
                holdout_first_week=str(hold["week"].min().date()) if hold is not None else None,
                fit_seconds=round(t1 - t0, 1), forecast_seconds=round(t2 - t1, 1),
                device=torch.cuda.get_device_name(0))
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
