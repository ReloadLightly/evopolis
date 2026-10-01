# Task 06 — Institutional validity: do learned human models predict which allocation rule works?

**Status: active. Execute end to end in one run:** implement, fit, freeze and push the forecasts, open Experiment 2, score, write the report, update the README, commit and push. No pilot and no pauses for approval. If a genuine correctness or access blocker stops you, save resumable state and report the exact blocker; never describe an unrun phase as done.

**Autonomy:** the user is not watching this run and will not answer questions. Do not stop to ask for confirmation, do not end your turn with progress reports, and do not wait for input between phases; keep working until the final push is done. If a command fails, diagnose it, fix it and retry. Long jobs (fitting, rollouts) should run as resumable scripts that you keep polling until they finish. Stop early only for a genuine blocker you cannot fix yourself, and then say exactly what is needed.

First, save this brief verbatim as `docs/tasks/06-institutional-validity.md`. Add one line at the top of `docs/tasks/06-frozen-human-transfer.md`: "Superseded on <date> by Task 06 — Institutional validity; not executed." Leave that file otherwise unchanged. Read `AGENTS.md`, `README.md`, `docs/protocol.md`, `docs/data-inventory.md`, `docs/task05-review.md` and the Task 03–05 model, training and generation code before implementing. The pinned source CSV is expected at `data/raw/sustainable_behavior.csv`; if it is missing, `bash scripts/reproduce.sh` restores it with checksum verification.

## Why this replaces the old Task 06

An external audit of commit `fecb2bf` found three problems. Re-verify every number below from the repository's own artifacts before relying on it. If something does not reproduce, report the discrepancy; do not adjust the design to make it reproduce.

**1. The obvious benchmark was missing.** The release contains DeepMind's own simulator output (recorded BC1 games). Mean surplus per player-round:

| | Equal | Mixed | Proportional |
| :--- | ---: | ---: | ---: |
| Humans, Exp 1 (40 groups each) | 2.20 | 4.53 | 6.05 |
| Recorded BC1 games (512 each) | 3.88 | 4.79 | 6.41 |
| Task 03 GRU (192 generated games) | 2.01 | 2.18 | 3.56 |
| Task 03 feedforward | 2.02 | 2.20 | 4.12 |

The Task 03 constant model reaches 2.46 under Mixed, so the GRU is worse than an input-blind model under that rule. In the Task 04/05 origin-0 forecasts of the eight Mixed test groups, every model family predicts a pool of 0.2–12 after round 20, with 2–34% of branches still alive. The humans have a pool of about 72, and 75% of groups are alive.

**2. The game sits at a tipping point, and likelihood barely registers it.** With full allocation, the pool survives only if the average return fraction exceeds 1/1.4 ≈ 0.714. Teacher-forced on the 24 Equal/Mixed/Proportional test groups, GRU seed 17 predicts mean return fractions of 0.435/0.600/0.710, against 0.404/0.629/0.745 observed. It under-reacts to the allocation rule.

Multiplying its legal PMF by `exp(0.6·c/floor(e))` raises those fractions by about 0.03–0.05. Test NLL worsens by only 0.019 nats, yet simulated Mixed survival rises from about 6% to about 38% (64 games per cell).

The Task 04 continued GRUs gain about 0.09 nats on these groups, while their Mixed fraction falls to 0.58–0.59. That is a plausible mechanical account of "better NLL, worse collective forecast". Under Proportional, the GRU predicts a zero return 9.1% of the time on nonforced test choices, against 5.2% observed. Under that rule one zero means permanent exclusion.

**3. The old Task 06 spent the only untouched human cohort on the wrong contrast.** That contrast was H1−H0, already a tight null, and it came with about 7 hours of continuation and about 2 million forecast branches.

Training-budget context (not a target): Task 03 used 1,440 optimizer steps on 96 groups. BC1 used 700,000 steps (batch 256) on 537 games that included random allocations, which this release does not contain.

## Question

> Does a behavioral model fitted to human play predict which allocation rule works, including a rule it never saw?

This is the second half of the README's research question. Judge simulators by group-level outcomes under each rule and by their predicted differences between rules. Individual NLL is secondary.

## Out of scope

No evolutionary search, ShinkaEvolve, LLM calls, viewer or UI changes, new institution design, or Experiment 3/4 outcomes. Do not modify Task 01–05 artifacts. You may list an LLM-agent baseline as future work in the report.

## Evidence boundaries

- **Fitting and checkpoint selection** use Exp 1 training/validation groups only (the existing 96/32 split). Exp 1 test groups are already opened and serve as diagnostics only.
- **Recorded BC1/BC2 games** are synthetic and may be parsed now. Verify from the paper's Methods which human data each was trained on, and quote the evidence.
  - BC1 (Train Set 1, 537 games) predates the Interpolating rule, so its recorded Interpolating games are an out-of-sample forecast for Experiment 2.
  - BC2 (Train Set 3, 990 games) likely includes earlier human experiments. Report it only descriptively, with that label.
- **Experiment 2** (40 Proportional, 40 Interpolating and 40 M1 groups) is the transfer cohort. Do not parse its behavioral rows until the Phase 3 freeze commit has been pushed. Counts and labels from `results/task01/inventory.json` may be used before then.
- **Experiment 3 stays closed.** It is reserved for any later evolutionary study. Experiment 4 is out of scope.
- **Instructions differ between cohorts.** Exp 2 participants received rule instructions; Exp 1 participants did not, and the models have no instruction input. Level errors therefore mix model error with cohort and instruction shift. The within-Exp-2 Interpolating−Proportional difference cancels shared shifts to first order, which is why it is the primary estimand.
- **M1 and M2 are not executable.** Use recorded M1 offers only for teacher-forced likelihood. Never reconstruct a policy from recorded future offers.
- **Calibration is permitted.** Calibrating a simulator to match observed human outcome distributions on training groups is fidelity, not cooperation-seeking. Never target higher surplus.

## Phase 0 — Freeze the configuration

Before importing torch, profile available RAM, swap, disk and CPU. Use one compute thread, sequential jobs and resumable phases, and keep peak RSS ideally below 600 MiB.

Before any fitting, write `configs/task06.json` containing every seed, grid, game count, metric, primary comparison and decision rule from this brief:

- training seeds 17/29/43;
- rollout namespace 20261102;
- calibration common-random-number namespace 20261103;
- bootstrap seed 20261101, with 2,000 replicates.

If a phase's measured throughput projects beyond 8 hours, stop and report rather than shrinking the design silently.

## Phase 1 — Institution benchmark on Experiment 1

**Simulator.** Build a vectorized free-rollout simulator: games × residents tensors, one PMF per resident, inverse-CDF sampling from indexed uniforms. Rollouts start at round 0 with no human prefix and must use:

- the existing `allocate()` and `WorldState` accounting;
- the exact-zero / 40-round scheduler and the legal integer support;
- frozen weights, separate resident states and simultaneous conditionally independent draws;
- one persistent effect per resident, drawn at the start, for H families.

Validate it against a slow per-game reference loop using the same uniforms (actions and pools must be identical). Also check that Task 03 GRU summaries agree with the archived Task 03 distributions within Monte Carlo error.

**Rules.** Equal, Mixed, Proportional and Interpolating. Before using Interpolating, replay all `Interpolating Baseline BC 1` rows: recompute offers from the recorded pool and previous contributions, and report the residuals.

**Simulators to evaluate:**

- **Frozen references:** Task 03 constant/linear/feedforward/GRU, Task 04 continued feedforward/GRU, and Task 05 P0/P1/H0/H1, all three seeds each.
- **Recorded BC1:** Equal, Mixed, Proportional, Interpolating, plus M1 descriptively.
- **Recorded BC2:** Interpolating, plus M2 descriptively.
- **Phase 2 candidates,** once fitted.

Run 512 games per checkpoint and rule, pooling seeds within a family (1,536 games per rule). Store per-game summaries; keep full trajectories only for the first four games of each cell.

**Outcomes per game** (same definitions as Tasks 01/03):

- mean surplus per player-round over 40 rounds, with zero padding after exact-zero termination (human records keep their residual floor);
- Gini of the four player means;
- survival, defined as pool after round 40 > 1;
- pool after round 20;
- first round with pool < 1.

**Exp 1 benchmark.** Compare every simulator with Exp 1 humans, twice: all 40 groups per rule (labelled as including opened test groups) and train+val groups only. Report per rule:

- mean surplus, survival, Gini and pool after round 20;
- the two-sample energy distance on (surplus/10, pool20/200, pool40/200), using the off-diagonal U-statistic estimator;
- whether the human ordering across Equal/Mixed/Proportional is reproduced.

## Phase 2 — Can simple ingredients close the gap?

### (a) Institution signal: FA families

Add a rule-responsiveness feature built only from public pre-decision information. For round t ≥ 1, let the offer shares be `a_i = e_{t,i} / Σ e_t` and the previous-contribution shares be `b_i = c_{t−1,i} / Σ c_{t−1}`. Then

`s_t = Σ (b_i − 1/4)(a_i − 1/4) / Σ (b_i − 1/4)²`.

For mixture rules with full allocation, `a_i = w/4 + (1−w)·b_i` exactly, so `s_t = 1 − w`: 0 for Equal, 0.5 for Mixed, 1 for Proportional, and pool-dependent for Interpolating. For a fixed mixture weight below the cap, the one-step private return of a unit contribution is `0.35 + 1.05·s_t`.

Handling edge cases:

- If `Σ c_{t−1} = 0`, `Σ e_t = 0` or `Σ (b_i − 1/4)² < 1e−9`, carry forward the last defined value.
- Before any value is defined, use 0.5 with an "undefined" indicator of 1.
- Clip to [−1, 2] and report how often clipping occurs.

Test three things: exact recovery of 0/0.5/1 on constructed inputs; Interpolating slopes against the BC1 rows; and that no current or future action can enter the feature.

Fit three families × seeds 17/29/43, 480 epochs each from fresh initialization:

- **FA-GRU:** the Task 03 GRU with 11 inputs.
- **FA-P0:** Task 05 P0 controls plus `s_t` and the indicator, in float64.
- **FA-H0:** Task 05 H0 controls plus `s_t` and the indicator, in float64. Use the validated 41-node quadrature, and re-check 41 versus 81 nodes against the Task 05 tolerances.

Use the Task 03/05 optimizer settings and effective batches. Select each fit at the earliest minimum of validation NLL, and report minima that fall at the budget boundary. **FA-best** is the family with the lowest mean validation NLL.

### (b) Closed-loop calibration: CL variants

Calibrate six lines, each seed separately: continued GRU, P0, H0, FA-GRU, FA-P0 and FA-H0. The calibrated distribution on the legal support is

`P_cal(c) ∝ P(c) · exp((τ0 + τ1·s_t) · c/e)`.

Choose `(τ0, τ1)` to minimize the summed energy distance between simulated and human training-group outcome vectors (surplus/10, pool20/200, pool40/200) for Equal, Mixed and Proportional:

- training groups only, 256 games per rule;
- common random numbers across grid points;
- grid τ0 ∈ {−1.2, …, 1.2} in steps of 0.3 and τ1 ∈ {−1.0, …, 2.5} in steps of 0.5, then a 5×5 refinement at half step around the best point.

Report the validation-NLL cost of each calibrated tilt.

### (c) Diagnostics on validation groups

Run these for the references and for the FA and CL lines.

- **Attenuation:** teacher-forced mean predicted versus observed return fraction per rule, computed on nonforced choices and weighted by group, then by rule. Attenuation index = `(pred_P − pred_E) / (obs_P − obs_E)`.
- **Tipping-point sensitivity:** multiply each legal PMF by `exp(τ·c/e)` for τ ∈ {−0.3, 0, 0.3, 0.6, 0.9}, using 256 games per point. Plot the change in validation NLL against simulated survival and surplus per rule, and mark 1/1.4.

### D1: headroom decision (record before Phase 3)

Development groups are Exp 1 train+val (32 per rule) for uncalibrated models and validation only (8 per rule) for CL models. Score BC1 on the same groups each time.

Institution error is `IE = mean over Equal/Mixed/Proportional of |simulated − human mean surplus|`; also report the survival analogue.

- **No headroom:** at least one FA or CL candidate matches the strict human ordering across Equal/Mixed/Proportional for both mean surplus and survival, and has `IE ≤ IE(BC1)` on the same groups.
- **Headroom:** otherwise.

Report IE with group-bootstrap intervals.

## Phase 3 — Freeze and pre-register

Before touching any Experiment 2 row:

1. Generate the full rollout sets for every simulator under all four rules. These are cohort-independent, so they are the Experiment 2 forecasts.
2. Write `results/task06/manifest.json` with checkpoint hashes, calibrated `(τ0, τ1)` values, seed tables, forecast-summary hashes, the D1 outcome, the primary comparisons and the analysis-code hash.
3. Commit and push as "Freeze Task 06 forecasts before opening Experiment 2". This public commit is the pre-registration.

## Phase 4 — Open Experiment 2 and score

Parse Experiment 2 rows into a separate typed cohort with full keys. Verify:

- group disjointness from Exp 1;
- round indexing and legal actions;
- allocation replay residuals for Proportional and Interpolating.

**Primary estimands:**

- **E1 (effect):** predicted minus observed Interpolating−Proportional difference in mean surplus within Experiment 2; report |error|.
- **E2 (level):** |predicted − observed| Interpolating mean surplus in Experiment 2.

**Primary simulators:** BC1 (the incumbent), FA-best and CL(FA-best). Report the Task 03 GRU and the constant model as named secondary references; all other simulators are secondary.

**Comparison with BC1:** compute paired |error| differences against BC1 over the same human-group bootstrap resamples, stratified by rule. Call a simulator "better" or "worse" than BC1 only if the 95% interval excludes zero; otherwise call it "not distinguishable". Report Monte Carlo standard errors separately.

**Secondary:**

- survival versions of E1 and E2;
- Proportional level error;
- energy distances per rule and pool-after-20 errors;
- per-seed results;
- teacher-forced NLL on all 120 Experiment 2 groups for every checkpoint, with M1 reported separately;
- the descriptive cohort shift between Exp 1 and Exp 2 humans, under Proportional and under M1.

**D2:** classify each primary simulator relative to BC1 on E1 and E2.

## Phase 5 — Report, README and publication

Write `docs/institutional-validity.md` in at most 2,500 words, for a scientist reader. Cover, in order:

1. the question and why the tipping point matters;
2. the simulators;
3. the Exp 1 benchmark;
4. attenuation and sensitivity;
5. FA and CL results;
6. D1;
7. Experiment 2 transfer and D2;
8. limitations, stated once;
9. the recommendation;
10. a short "paper readiness" paragraph: the claim, the baselines and what is still missing.

Keep machine-readable provenance in `results/task06/`, not in prose.

The recommendation follows mechanically:

- **D1 no headroom, D2 not worse than BC1:** the contribution is the simple ingredient. Recommend no evolutionary search, and sketch a paper outline.
- **D1 headroom:** write `docs/tasks/07-evolved-behavioral-programs.md` (specified, not run). It covers LLM-guided program search over behavioral-model programs, scored on closed-loop fidelity on Exp 1, with a random-search control at matched budget and a frozen test on Experiment 3. Do not run it.
- **D1 no headroom, D2 worse than BC1:** this is a transfer failure. Recommend investigating instruction and cohort shift, not evolution.

**Figures** (PNG + SVG, existing theme and plotting helpers):

- the Exp 1 institution benchmark;
- attenuation;
- tipping-point sensitivity;
- Experiment 2 predicted versus observed effects with intervals.

**README:** rewrite it in arXiv-paper style (title, abstract, introduction, data and methods, results, limitations, references), with the figures embedded inline, in about 1,800 words of prose or less plus tables and figures, centred on this question. Include:

- the current status (no evolutionary search yet);
- the BC1 benchmark table;
- the tipping-point diagnosis;
- the Task 06 results;
- the roadmap, gated by D1 and D2.

Keep the cover and one viewer screenshot, and link the earlier reports instead of repeating them. Write plainly and state limits once.

**GitHub description:** update `docs/github-description.txt` to "Can learned models of human behavior predict which institution works? Closed-loop validity tests on human common-pool data; evolutionary search planned." Run `scripts/configure-github.sh` if `gh` is authenticated.

**Verification** (focused, not exhaustive):

- the feature tests above;
- vectorized-versus-reference rollout identity;
- tilt normalization;
- the energy-distance estimator against a brute-force reference;
- bootstrap pairing;
- one fresh-process regeneration per family;
- protected Task 01–05 hashes unchanged;
- the full existing test suite.

Record runtime and peak RSS per phase in one JSON file.

Confirm that `origin` is `ReloadLightly/evopolis`, then commit and push without force, preserving unrelated user changes. Expect two pushes: the Phase 3 freeze and the final result.
