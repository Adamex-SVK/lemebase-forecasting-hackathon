# Process rules

> Team setting: where this file says "the user", read "the collaborator who owns the decision". Any collaborator can be a decision owner; see `TEAM-RULES.md`.

How we work. The user is in the middle of every decision; Claude and agents do the legwork and report evidence.

## 1. Who decides what
| Always the user's call (decision gate) | Claude does without asking |
|---|---|
| Anything that changes what the data means: returns, zero-filling, partial weeks, series start/stop | Mechanical work already agreed (type conversion, running agreed code) |
| Splits and validation protocol | Reading raw data and profiling it (read-only) |
| Features, loss function, model choice, experiment matrix | Writing logs and updating the decision list |
| Anything outward-facing: registering a team, uploading a submission | Re-running an iteration step unchanged |

How a decision is presented: the **evidence**, **2-3 options**, and Claude's **recommendation**. Then Claude stops and waits. A decision gets an ID (D#) in `decisions-to-make.md`; when settled it moves to "Decided" with the date and the choice, and is referenced in the iteration log.

## 2. Checkpoints
Every pipeline step ends with a **checkpoint**: Claude stops, summarizes what was done and found (numbers, not adjectives), lists any open decisions, and waits for the user's go. No step starts before the previous checkpoint is cleared.

## 3. Agents
- Agents gather evidence and run agreed code. **They never decide.** Anything that looks like a decision comes back as an option with evidence.
- Each agent gets a written brief saved in `iterations/NN-name/briefs/` (goal, inputs, outputs, what it must NOT do, how to report).
- Raw data in `hackathon-retail-sets/` is **read-only**. Agents write only to `data/interim/`, `src/`, and the current iteration folder.
- Every agent result is summarized into the iteration `log.md` by Claude, with the file paths of its outputs. An agent's claim is checked (a number re-computed or a diff reviewed) before it is logged as fact.
- Agents run in parallel only when their outputs do not overlap.

## 4. Iterations
One iteration = one full pass through the pipeline (see `pipeline-architecture.md`) with an explicit hypothesis. Folder layout:

```text
iterations/
  NN-short-name/
    plan.md        hypothesis, steps, which decisions are involved, what "done" means
    log.md         every step in order: what was done, what we saw, each decision (ID, options, who chose, why)
    changes.md     ONLY what differs from the previous iteration, and the score change it caused
    results.csv    scores per shop: own-only vs pooled, per model, local validation
    briefs/        the written agent briefs
```
- `log.md` is append-only and dated. `changes.md` is short: a table of *change -> reason -> effect on score*.
- A new iteration starts only when the previous one's results are reviewed with the user.
- Each iteration changes as little as possible, ideally one thing group at a time, so an effect can be attributed.

## 5. Results and honesty
- **Primary score = local validation over all 13 weeks**, per shop and averaged with equal weight over the four sets, plus the local cut of the test shop. The live leaderboard (w1-w7 only) is a sanity check.
- Every iteration reports **own-only vs pooled** per shop (extra credit), so it never needs to be rerun later.
- Set D is always reported **with and without** the series that have no history before the cut (D8).
- Failures and negative results are logged like successes. A score is only reported with the command / file that produced it.
- Seeds, library versions and cut dates are recorded in `log.md`.

## 6. Where things live
- Decisions: `decisions-to-make.md` (global). Pipeline: `pipeline-architecture.md`. This file: rules.
- Code: `src/`. Derived data: `data/interim/` and `data/processed/`. Raw data: `hackathon-retail-sets/` (never modified).
