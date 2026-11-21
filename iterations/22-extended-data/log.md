# Iteration 22 experiment log

## 2026-11-21 — source review

Set A source is VN2, Zenodo 23265416, including its nine cumulative weekly sales snapshots. Every earlier snapshot is an exact prefix of Week 8. Set B uses the published Mancino synthetic transaction headers and lines from data/v1_2025-11-01. Set C uses Zenodo 19784493 sales_txn.parquet. Set D uses Mendeley 27x8mjm8k4 v1 sales and stock tables. Source manifests and downloads used their public GitHub, Zenodo and Mendeley APIs. Compared all source file hashes to publisher digests, including Git blob hashes for B. All aggregated native sales keys and units matched. D's 284,755 stock interval/key/quantity records also matched. No additional 13-week holdout available. Additional original source fields are descriptive or operational: A challenge costs; B customer/store/product descriptors and derived tables; C uncoded category labels; D product descriptions and unmasked category/supplier labels. B is synthetic. Raw intraday timestamps supply within-week sale timing.

Command: .venv/bin/python iterations/22-extended-data/audit.py --source-dir /tmp. Results and input hashes in results/source_comparison.csv and results/source_hashes.json. Default sandbox DNS failed; approved public downloads succeeded. Initial stock-date parser rejected the published 9999-12-31 sentinel. Fixed verification to compare original ISO date strings exactly. Stage reviewed before model fitting.

## 2026-11-21 — two pharmacy features

Commands: .venv/bin/python iterations/22-extended-data/daily.py <historical_archive>; .venv/bin/python iterations/22-extended-data/lightgbm_trial.py.
Used 143 historical weeks only. Exact weekly sales prefixes determine each feature independently by candidate consensus. Missing recency represents no positive sale or disagreement. Six physically truncated source checks at origins 13, 65, 91, 117, 130 and 143 reproduced the cached values exactly. Timestamp counts and recencies use only dates before the Monday origin. All 131 eligible origin matrices were built. Seed 0 and original LightGBM parameters, tree counts selected only with purged earlier target windows. Six pinned ensemble scores reproduced. No A–D or UCI examples in this feature trial. Full scores, scored coverage, coverage of the new features and subgroup errors are in results/lightgbm/.

Primary mean gain +0.02572, 2/4 wins. Both supplementary gains were negative. Component fails review. Keep original LightGBM for the Chronos trial.

## 2026-11-21 — GPU training

Direct SSH initially disconnected; the supplied host subsequently authenticated successfully. Used an isolated lb22-maya directory. All five prepared dataset SHA256 values on the GPU matched local inputs. NVIDIA RTX 3090, Python 3.11.10, torch 2.5.1+cu121, transformers 4.57.6 and chronos 2.3.2. Model revision a971ba21945c4f1796b17a91fe69214b5f4ad472 from local Hugging Face cache. Offline mode. Smoke test reproduced two independently seeded predictions exactly, performed a pooled gradient update and returned finite 16x13 predictions. Peak allocated GPU memory in smoke test 1.163 GiB.

Remote commands in the isolated directory: HF_HUB_OFFLINE=1 ~/hack/.venv-chronos/bin/python -u chronos_trial.py --mode smoke --data ~/hack/data/interim --out results; same command with --mode historical. Stored complete fit weights on training host for adapted arms, step-selection and cut metadata and keyed forecast matrices for every arm. Fixed inference batch of 48, pretrained default sample count, nonnegative median. Selection is based on project scores on fully enumerated outer holdouts.

Precision correction for the CPU summary: primary mean gain +0.02571058, early -0.00066829, late +0.05648593.

Source acquisition for reproducibility (public read-only URLs). The expected hashes are saved in results/source_hashes.json; the four manifests were saved under /tmp/lb22-{A,B,C,D}.json.

    curl -fLsS https://zenodo.org/api/records/23265416 -o /tmp/lb22-A.json
    curl -fLsS 'https://zenodo.org/api/records/23265416/files/SupChains%2FVN2-v1.0.1.zip/content' -o /tmp/lb22-A.zip
    curl -fLsS https://api.github.com/repos/jrmst102/mancino/contents/data/v1_2025-11-01 -o /tmp/lb22-B.json
    curl -fLsS https://raw.githubusercontent.com/jrmst102/mancino/main/data/v1_2025-11-01/transactions.csv -o /tmp/lb22-B-headers.csv
    curl -fLsS https://raw.githubusercontent.com/jrmst102/mancino/main/data/v1_2025-11-01/transaction_line_items.csv -o /tmp/lb22-B-lines.csv
    curl -fLsS https://zenodo.org/api/records/19784493 -o /tmp/lb22-C.json
    curl -fLsS https://zenodo.org/api/records/19784493/files/sales_txn.parquet/content -o /tmp/lb22-C.parquet
    curl -fLsS 'https://data.mendeley.com/public-api/datasets/27x8mjm8k4/files?folder_id=root&version=1' -o /tmp/lb22-D.json
    curl -fLsS https://data.mendeley.com/public-files/datasets/27x8mjm8k4/files/0f2e12a6-6110-4bda-b343-63a76234fa02/file_downloaded -o /tmp/lb22-D-sales.csv
    curl -fLsS https://data.mendeley.com/public-files/datasets/27x8mjm8k4/files/6cdf316b-453e-482d-a9df-142262cfc6f6/file_downloaded -o /tmp/lb22-D-stock.csv

Local read-back verified equal, conflicting and all-zero transaction histories and reproduced all six saved LightGBM component forecasts exactly after model reload. Local library versions and five prepared input hashes are in results/local_environment.json. GPU has no rsync executable. Transferred only completed keyed forecasts and metadata using tar over the authenticated tunnel, validating archive member names and prior result content.

Across the four primary periods, the two timing features gained 0.299 group-score points among products first observed selling within 13 weeks and 0.095 among products with sales in one to three of the prior 13 weeks. These are within-group means; the overall gain was only 0.026. Independently replayed the zero baseline for all four A–D holdouts using the new scorer; all scores and scored counts matched the prior data audit to 1e-8.

## 2026-11-21 — completed GPU and promotion review

All 30 arms completed (three variants on six pharmacy and four A–D holdouts). Twenty fine-tuned transformer checkpoints remain on the supplied GPU under ~/lb22-maya/results/<shop>-<offset>-<arm>/weights/. The other ten runs are freshly regenerated pretrained controls. All 30 keyed forecast files and inner-selection records are available locally in results/gpu. The executed GPU runner SHA256 is 8da0983c2b7087c645804bddef6d6eb3a8a518d2f3e0124ea47db52d0eaeee23. GPU memory returned to its original free level after training.

    .venv/bin/python iterations/22-extended-data/import_results.py /tmp/lb22-results.tgz
    .venv/bin/python iterations/22-extended-data/score_trial.py
    .venv/bin/python iterations/22-extended-data/finalize.py

Using the unchanged four-component ensemble with one component varied, the original primary mean was -13.665947. Pharmacy-only Chronos fine-tuning reached -13.568964 (+0.096983, wins 3/4, early +0.052605, late +0.148758), and fell by 0.177182 and 0.573564 on the two supplementary periods. Pooled Chronos reached -13.699679 (-0.033732, wins 1/4), and fell on both supplementary periods. The freshly regenerated pretrained control scored -13.708315. The original LightGBM with transaction timing reached -13.640236 (+0.025711). All tested challengers failed the +0.5 primary mean requirement. The combined change was ineligible.

On the individual shops' latest 13-week holdouts, own/pooled Chronos scores were A 16.705755/17.789864, B 19.803643/19.220837, C 58.719762/58.617226 and D 69.110995/69.091705. Equal-shop means 41.085039/41.179908. Only A favored pooling. D excluded 10,460 zero-history series; the 39,983 scored series cover 53.22% of holdout units. These are within-shop forecasts with own-shop history. The per-shop reports have all exclusions, unit coverage and separate early/late metrics. Subgroup scores for the pharmacy are in results/groups.csv.

Selected the full-history original ensemble. Rebuilt the full 2,904 x 13 matrix from the pinned final components and confirmed exact equality to submissions/ensemble_top3_blend_v1.csv. The file passed the submission validator. Stored its hash in results/frozen_forecast.json. No leaderboard upload.

## Full-history CSVs for all four comparison rows

The user requested upload-format forecasts for the original and each of the three tested changes. Reuse the original full-history matrix and the fixed ensemble recipe. Generate the three challenger files independently; keep the historical promotion decision. Refit the daily-feature tree locally, and refit each Chronos arm on the GPU through 2023-04-17 using the same purged inner checkpoint selection and seed as in the trials. Verify the output values against the intended blend after reading back each CSV.

Exports complete. Ran chronos_trial.py --mode final --arm own and --arm pooled on the isolated GPU with the original prepared input tables. Both selected 500 updates and refit through 2023-04-17. Imported their keyed forecasts and ran .venv/bin/python iterations/22-extended-data/export_challenger.py separately for daily, own and pooled. The three files differ from the original in 2,996, 5,675 and 5,512 cells respectively. The project validator and exact blend read-back passed for every CSV. Packaged all four in submissions/iteration22_four_forecasts.zip and verified the archive members byte-for-byte. File hashes and fit metadata are in results/challenger_exports/.
