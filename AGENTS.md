# Maya's forecasting workflow

## Technical and structural decisions

Whenever a technical or structural decision is needed, try to consult the
`senior-data-scientist` advisor skill before recommending or implementing a choice.
Apply its relevant guidance to the project's forecasting task, official metric,
temporal validation, and cross-shop generalization requirements. Use judgment
where a decision falls outside the skill's scope; do not apply unrelated generic
defaults in place of project-specific requirements.

The skill is installed locally at
`/Users/mayasjain/.codex/skills/senior-data-scientist/SKILL.md`.
On another machine, locate the skill in that user's skills directory. If it is
unavailable, state that limitation rather than claiming to have consulted it.

## Project workflow

- Keep raw data in `hackathon-retail-sets/` read-only. Preserve successful models,
  submissions, and historical experiment records.
- The user decides data interpretation, validation splits, features, objectives,
  model choices, and experiment scope. Present evidence, options, and a
  recommendation for unresolved decisions before implementation. End each agreed
  pipeline stage with a review checkpoint before starting the next stage.
- Record decisions in `decisions-to-make.md`. Use iteration `plan.md` for the
  hypothesis and scope, append-only `log.md` for evidence and failures, and CSVs
  for numerical comparisons. Record commands, seeds, versions, and cut dates.
- Compare models on identical 13-week temporal holdouts using the project MASE
  evaluator. Report results per shop and the equal-weight A-D average, with the
  target shop separately. Treat public leaderboard results as supplementary.
- Report own-only versus pooled training and Set D's cold-start exclusions and
  scored coverage explicitly. Do not claim unseen-shop generalization from
  target-shop-only validation.
- Keep the implementation model-agnostic; LightGBM is the reference, not a
  requirement for future models. Follow `pipeline-architecture.md` for pipeline
  context and iteration records for historical results.
- Do not register, upload, or submit forecasts without explicit user approval.
- Delegate only when authorized. If agents are used, give each a saved brief,
  isolate output paths, verify their claims, and record results in the iteration log.
