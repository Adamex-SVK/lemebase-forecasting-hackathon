# Next steps (prepared 2026-11-21, night)

## RESUME HERE (updated 2026-11-21, afternoon)
**Read `HANDOVER.md` first**: it has the current state, the best recipe (live -4.8), the jobs in flight and the ordered next steps. Short version: (1) collect the TabPFN and overlapping-cut forecasts from the GPU machine (`iterations/03-foundation-model/gpu-machine.md`); (2) write and run the FFORMA-style meta-learner (`iterations/05-meta-ensemble/plan.md`, `src/meta/fforma.py` not written yet); (3) decide the final recipe and let a person upload a few files; (4) finish `EXTRA-CREDIT.md`; (5) clean up the SSH key.

Status: iteration 00 complete; iteration 01 steps 1.1-1.2 done; step 1.3 (all shops x arms) is paused, not finished (see RESUME HERE above). Decisions to settle are in `decisions-to-make.md` (D30-D32 open). Nothing below is run until you approve it.

## A. Now: finishing iteration 01
| Step | What | Notes |
|---|---|---|
| 1.3 | 20 models: own, pooled (equalised), pooled (unweighted), others only, per shop | Running. The first pooled arm is slower than the own arm (35 s): at about 5+ minutes per arm the whole run could take 2+ hours. Speed option D31 below. |
| 1.4 | Analysis: paired per-article comparisons (own vs pooled vs others only vs median13), bootstrap intervals, the extra-credit table, `changes.md` | Light on the CPU. |
| 1.5 | Review with you: what worked, what to change | Decision gate. |

## B. Iteration 02: make it more robust (laptop, tomorrow morning)
Goal: pick models on evidence that is not one lucky 13-week window, and look for the remaining gain in the active articles (mean MASE 1.34 vs 0.35 for dead ones).
1. **Rolling-origin evaluation on the test shop (D32).** Score at 4 cuts (the last 13, 26, 39 and 52 weeks) instead of one. The own arm takes about 35 s, so this is cheap, and it tells us whether a 2-point gain is real.
2. **F6: the moving yearly peak.** Find out why the test shop's peak moves about two weeks a year (2020-11-16, 2021-11-01, 2022-10-17) while A's does not. Light script; then decide on calendar features.
3. **Ensembles and target tricks:** LightGBM blended with the median of the last 13 weeks, several seeds, per-horizon variants, a quantile-0.5 head.
4. **Zero-handling:** only if step 1.3 shows the model is weak on near-dead articles (the zero rule was worth +0.03 so far).

## C. Iteration 03: foundation model (GPU, tomorrow)
Needs from the CEO / team: what machine (CUDA GPU and memory), how we access it (SSH, notebook, shared drive), Python version, internet access to download model weights, and how long we have it.
- **3.0 Environment:** new venv on that machine from `requirements.txt` plus torch and the model package. Copy `src/`, `data/interim/` (small) and the raw parquet (about 23 MB). Re-run `tests/` there first.
- **3.1 Zero-shot** on the five local cuts with `src/validate.py`; compare with LightGBM and the median.
- **3.2 Fine-tune** on pooled windows (own and other shops, D27) and compare own vs pooled.
- **3.3 Blend** with the LightGBM.
- Candidate models (unverified, from my knowledge, to be checked online before we choose): Chronos-Bolt / Chronos-2, TimesFM, Moirai, TiRex. Retail series with 60% zeros are hard for them; expect them to be useful mainly as an ensemble member.

## D. Iteration 04: the rich-column techniques (D18, order D19)
1. **Feature dropout:** needs the rich columns joined per shop (D7: check the grain first): price and promo flags (B `promotions`, C `sales_type`, D `sales_type`), stock status.
2. **Cleaning:** mark promo and markdown weeks so they do not look like random spikes.
3. **Auxiliary tasks:** predict e.g. promo-week or category from the four columns.
4. **Teacher-student:** a teacher with all columns, a student with the four.
Each is scored on the same cuts as the extra-credit comparison.

## E. Iteration 05: final
- Train the chosen model(s) on all five shops' full histories (D27) and forecast the real 13 weeks.
- Write the CSV with `src/make_submission.py` (format validator included); you upload it.
- Extra-credit write-up: own vs pooled for every shop with an explanation, and the team summary.

## F. Already prepared tonight
- `src/make_submission.py`: writes and validates the submission file (header `article_id,w1..w13`, 2,904 articles, w1 = 2023-04-24, non-negative, ids match the history).
- `submissions/dry_run_median13.csv`: the median-of-last-13-weeks forecast for the real test shop. It passes the format checks. It is optional for the sanity upload (step 0.8): it shows the live score of the best simple baseline and confirms the format. Registering and uploading are yours (D25).

## G. New open decisions (also in `decisions-to-make.md`)
- **D31 Speed of step 1.3.** If the pooled arms take too long: (a) let it run; (b) restart with window origins every 4 weeks for the other shops (adjacent 2-week origins are highly redundant), cutting rows by half; (c) also cap Set D to about 1M rows. Rec.: check the time of the first pooled arm, then (b) if it is over about 8 minutes.
- **D32 Rolling-origin evaluation** on the test shop (4 cuts) for model selection. Rec.: yes, from iteration 02.

## H. What to tell the team (so far)
Our local metric reproduces the organizers' baselines; a robust median forecast is surprisingly hard to beat; the first global LightGBM is about 2 points better than the median on the test-shop cut, not yet statistically clear; the answer to "does borrowing help?" comes from step 1.3.
