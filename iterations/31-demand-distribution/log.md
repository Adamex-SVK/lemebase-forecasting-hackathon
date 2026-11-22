2026-11-22. The original PR 20 upload record and the SHA-256 hash of the Downloads file agree (424efba0fc50e3ce49869f7f4129d0e1752a608088b2dd0931c569ce1f0aa936). The original four individual sibling-trained Chronos-2 historical predictions are absent from the Git tree. Key-only SSH access failed. The authenticated project SSH connection located the four files in hack/iterations/03-foundation-model/forecasts_oos-ft. Use the committed final Chronos-2 predictions and preserve the original component versions.

2026-11-22. Recovered the four original component files through the authenticated project SSH connection. For supplementary dates, the GPU ran the existing x1_chronos2_ft.py script with --cut-weeks 0 16 32 --tag dist31 under HF_HUB_OFFLINE=1 and the project's .venv-chronos in an isolated directory. Read-only copies of its foundation code and a link to the prepared data were used. The seed-0, 300-step sibling-only fit used learning rate 1e-5 and the original full-fine-tuning fallback. All 37,752 forecast values at the overlapping control origin matched the original saved component exactly by key. Thus both supplementary comparisons also contain all five components. All four primary controls reproduce PR 20's recorded scores to the published precision. Source component hashes are in the provenance JSON files.

Local commands from repository root, in order within each stage:

    .venv/bin/python iterations/31-demand-distribution/train.py target
    .venv/bin/python iterations/31-demand-distribution/train.py shops
    .venv/bin/python iterations/31-demand-distribution/score.py shops
    .venv/bin/python iterations/31-demand-distribution/score.py target
    .venv/bin/python iterations/31-demand-distribution/score.py final
    .venv/bin/python tests/test_validate.py

Python 3.12.4, NumPy 2.5.5, pandas 2.3.3 and LightGBM 4.7.0. Seed 0. The final target model fit 866,688 sampled threshold labels. Loading the stored model reproduced all 37,752 tree outputs. Checked both sides of the median crossing on 68 sampled valid histories and monotonicity across sampled threshold curves. Both controls and challengers have identical product eligibility on each paired period. The saved source, cut dates, coverage, groups and metrics are under results/.

Across the four primary target periods, mean MASE decreased from 1.136499 to 1.133205 (+0.3294 score points). Weeks 1–7 improved on all four periods (+0.5667 points on average). Weeks 8–13 improved on two (+0.0525 on average), with the largest regression in the July 2022 period (−1.0997). The supplementary full-period changes were −0.0418 at offset 16 and +0.2292 at offset 32. Three primary wins, average gain below the predeclared +0.5 threshold, and mixed supplementary evidence. Recommend retaining the incumbent as the preferred model. The results do not estimate a new public or private leaderboard score.

Across A–D the equal-shop mean improved by 0.1161 score points in the pooled comparison and fell by 0.0332 in the own-only comparison. Pooling includes each evaluated shop during training. The reference pooled arm retains iteration 29's own-shop tree and pooled Chronos-2. The pooled challenger fits both the tree and Chronos-2 on A–D, so the paired pooled contrast includes the changed tree training population. Individual shop results, cuts and scored-unit coverage are in shops_metrics.csv. Set D excluded 10,460 all-zero histories, scored 39,983 and covered 53.22% of units.

Final export reproduced every original rounded incumbent cell before substitution, and validated 2,904 unique articles against raw history across 13 weeks. The challenger changed 4,694 of 37,752 weekly cells across 702 articles. SHA-256 e9a262c3f90fdd17204a438a515d20a1841a649ae4c0ec4ebf29429ee6097da5. The candidate is a local export for review. No upload was made.

2026-11-22 — Maya selected ensemble_pr20_weekly_distribution_v1.csv for the next submission based on the observed historical gain. This supersedes the earlier preference for the incumbent in iteration 31. The +0.5-point review threshold was waived for this selection. The prior ensemble holds the recorded −4.5 public score and the selected file awaits leaderboard evaluation.
