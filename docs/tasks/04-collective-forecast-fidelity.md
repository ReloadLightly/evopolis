# Task 04 — When do learned inhabitants forecast collective behavior?

## Outcome and research question

Run a substantive follow-up to the first trained agents: **does more optimization improve forecasts of collective outcomes, and how far can a learned community forecast after observing real human history?** Complete the continued training, conditional forecasts, scoring, figures and a working forecast view. An implementation without the experiment is incomplete.

Task 03 established useful prediction of individual choices, but its generated Mixed and Proportional communities depleted much more often than the held-out human groups. GRU's small one-step likelihood advantage over the feedforward model was uncertain, and most selected neural checkpoints were near the 120-epoch limit. Check insufficient optimization before attributing collective errors to memory, heterogeneity or dependence between people.

This is a prospectively specified **diagnostic follow-up informed by already opened Task 03 results**, not a fresh confirmatory test or an established paper contribution. Freeze the following choices before running new outcome comparisons. Keep Experiments 2–3 closed; no transfer outcomes, new architecture, institutional search or evolutionary search are needed in this task.

## Read and preserve

Read `AGENTS.md`, `docs/behavioral-agents.md`, `docs/protocol.md`, `docs/data-inventory.md`, the Task 03 brief and `configs/task03.json`. Inspect the existing preparation, model, training, generation, evaluation and viewer modules and their provenance checks.

Preserve all original Task 03 weights, configurations, split membership, numerical results and hashed training/generation source files. Add a thin continuation/forecast module and a separate Task 04 configuration and results directory. Reuse pure model, loss and optimizer-step helpers where appropriate; do not relax the old experiment's contract or overwrite its checkpoints to make continuation convenient. Preserve user changes and verify `origin` is `ReloadLightly/evopolis` before publication.

## Fixed fourfold optimization control

Continue exactly six fits: **feedforward and recurrent families × seeds 17, 29, 43**. Restore each original **epoch-120 last checkpoint**, including model parameters, optimizer state and RNG/shuffle state. Train epochs **121–480**, adding 360 epochs per fit and **2,160 epochs overall**. Do not restart from the best-validation epoch or reset Adam.

Keep the original 96 training groups, 32 validation groups, nine inputs, legal support, emission distribution, architecture, group-balanced nonforced NLL, learning rate, weight decay, gradient clipping and effective batch unchanged. Use the existing CPU-only locked PyTorch, one thread, zero data workers, and two-group microbatches accumulated into eight-group optimizer batches.

Select the lowest validation-NLL checkpoint among **all epochs 1–480**, including the original best checkpoint; break exact ties by earliest epoch. Retain an original checkpoint if no continuation improves it. Continue through epoch 480 regardless. No early stopping, learning-rate search, favorable-seed selection, architecture change or selection by generated prosperity is permitted.

The original four families × three seeds supply **12 frozen references**; continuation adds **six selected checkpoints**, for **18 checkpoint conditions**. Compare each continued neural learner with its own 120-epoch procedure. A longer budget need not improve prediction, and epoch 480 alone does not prove convergence.

Before importing Torch, measure current WSL RAM/swap and disk. Benchmark at most two training-only continuation epochs using disposable copies, restore the original states afterward, and count this work separately. Measure forecast throughput on validation groups only to choose a safe inference chunk size, without selecting a model or changing scientific settings. Record actual main-run CPU/wall time and peak memory; do not turn an estimated duration into a silent reduction of seeds or epochs.

Run one process at a time, resume atomically saved continuation checkpoints, and report progress by fit/epoch and forecast batch. Preserve checkpoint identity, original parent hash, optimizer/RNG state, selected epoch, source/split/configuration/code hashes and learning curves. Verify a resumed continuation against uninterrupted updates on a small training-only check. A material correctness repair or budget deviation must be recorded before interpreting new outcome scores.

## Evidence and forecast origins

Use the existing Task 03 **test groups from human Experiment 1** under the three executable rules: Equal, Mixed and Proportional, eight groups each, **24 groups total**. These groups were held out from fitting but their Task 03 results are already public. Do not describe them as an untouched test of a newly proposed mechanism. Keep interacting groups as the replication unit; participant independence across group IDs remains unverifiable.

The recorded M1 allocator is unavailable. It can remain in the unchanged Task 03 observation-prediction summaries, but do not generate a live M1 future or replay its future offers as an executable institution. Interpolating has no Experiment 1 human counterpart and is outside this empirical forecast comparison.

Define `k` as the number of completed observed rounds, and use origins **k = 0, 5, 10, 20**. The forecast origin is **after the allocation for source round k is observed, but before its four contributions**. Thus the first forecast uses the recorded current pool and current offers, plus contributions from round `k-1` (zero initialization when k=0).

Warm each GRU by processing only recorded observations for rounds `0 ... k-1`; its next forward call processes the current source-round-k observation once. Give each resident its own state. Feedforward/linear/constant models receive the same current observation and no invented memory. Neither current contributions nor any future outcome enters a forecast input.

Start each branch with this observed boundary. Sample all four current contributions independently conditional on their observations and histories; resolve them simultaneously. Apply the existing resource equation to the observed first allocation, preserving any observed unallocated resource. From the next round onward, recompute allocations with the group's existing Equal/Mixed/Proportional rule and generated previous contributions, and use generated pool/history throughout. There is no later teacher forcing or clamping to observed future offers or pools.

Do not use a sequence of real future offers as a free-running institutional policy. That would supply future information and can become infeasible under the generated pool. The primary forecasts remain physically consistent under the declared environment.

## Numerical boundary audit and sensitivity

Audit the recorded origin's pool, allocations and legal targets without modifying source values. A planning audit found all **96 origins** (24 groups × four origins) feasible: allocations leave approximately **0.0008049–0.0012054** units unallocated. Recheck that assertion before using them. If an origin is infeasible, diagnose it explicitly rather than clipping offers, inflating the pool or silently omitting the group.

Canonical allocation can differ slightly from recorded allocation. Across these groups' 960 recorded baseline rounds, the planning audit found **166 of 3,840 player-round legal maxima** differ: 32 Equal, 32 Mixed and 102 Proportional, all with canonical `floor(offer)` one unit higher. Initial recorded offers just below 50 permit a maximum return of 49; a canonical allocation of exactly 50 permits 50. These are numerical/support differences, not learned behavior.

The main forecast honors the recorded current allocation. Predeclare one sensitivity at the primary origin/horizon: for both original and continued neural models, instead recompute **only the first allocation** canonically from the observed pool and previous contributions; subsequent allocations are already canonical in both arms. Warm history identically, but feed the chosen current allocation into the model. Reuse the paired rollout seed bank and report changes in legal support and proper scores. This tests a defined boundary convention, not exact recovery of the unpublished simulator.

Preserve integer support `0 ... floor(offer)` without an epsilon. Retain the reconstructed capacity, multiplier and exact-zero termination. Do not insert the source's unexplained 0.01 floor. Report recorded/simulated transition residuals and distinguish their possible contribution from behavioral-model errors; small arithmetic residuals alone do not establish the cause of the collective gap.

## Primary endpoint and fixed secondary horizons

The **primary comparison** is the continued GRU procedure minus the original GRU procedure, forecasting **h = 10 rounds after k = 5 observed rounds**. Score the joint endpoint vector:

```text
Z = (pool_after_h / 200,
     retained_resources_over_the_h_forecast_rounds / (200 * h))
```

Both coordinates have fixed physical bounds `[0,1]`. The surplus coordinate uses all four residents and only the forecast window; it excludes the observed prefix. The pool denominator is 200, not `200*h`. These are predetermined physical scales, not normalizers fitted to test outcomes.

The primary population contains groups with **at least one recorded current allocation >= 1 at origin k=5**. Eligibility uses only information available at the forecast origin. The planning audit gives **21 eligible groups: five Equal, eight Mixed and eight Proportional**; verify and report these counts. Do not drop groups based on whether they later survive or cooperate. Pair identical eligible groups across every model and budget.

For each checkpoint and group, score its Monte Carlo distribution with the Euclidean energy score. For M draws `z_m` and observed endpoint `y`, use the off-diagonal estimator:

```text
ES = mean_m ||z_m - y||_2
     - sum_(m != l) ||z_m - z_l||_2 / (2 * M * (M - 1))
```

Use the off-diagonal denominator rather than replacing it with `2*M*M`. This removes the finite-ensemble bias when scoring the underlying forecast distribution. A finite-sample estimate may be slightly negative; do not clip it. Lower is better.

Compute the score **separately for each fitted seed**, then average the three seed scores within each human group. Average groups within mechanism and the three mechanism means equally. Do not first mix all model seeds' draws into an unannounced ensemble. The primary effect is the paired continued-minus-original GRU score; report the analogous feedforward effect as secondary.

Use **2,000 mechanism-stratified paired group bootstrap replicates** with seed **20261014** for the primary effect. Resample the same eligible groups for all procedures; keep their origins, horizons, optimization seeds and forecast draws attached. Monte Carlo branches and optimization seeds do not increase the number of human replications.

Secondary horizons are **h = 1, 5, 10, 20** at each of the four origins, all within the recorded 40 rounds. Use one maximal 20-round branch to read every requested horizon. Report all-group results as well as origin-eligible results. At k=10 the expected eligible counts are 2/7/7; at k=20 they are 0/6/7 for Equal/Mixed/Proportional. Report unavailable strata explicitly: do not invent an Equal score or label an average over remaining mechanisms as the same three-mechanism estimand.

Report marginal pool/surplus CRPS and 80% predictive-interval coverage at the declared horizons, plus pool/participation forecast bands over the full window. Use a separately labeled mean of per-round marginal pool CRPS for trajectory accuracy. **Joint endpoint energy is not a score of the full temporally dependent path**; endpoint success alone does not validate path dependence or counterfactual institutions. Keep the declared primary effect distinct from exploratory horizon/condition findings.

## Finite forecast budget and Monte Carlo stability

For every original/continued checkpoint, human group and origin, generate **64 branches**: `18 × 24 × 4 × 64 = 110,592` principal branches, each at most 20 rounds. All requested horizons share those branches. Batch inference without mixing resident states or random streams; avoid saving every intermediate neural tensor.

At the primary eligible k=5 origins, add an **independent second bank of 64 branches** for every checkpoint: `18 × 21 × 64 = 24,192` additional branches of ten rounds. The declared primary score uses the **first bank's 64 draws per checkpoint/group**. Report the second bank's effect estimate separately as a Monte Carlo stability check. Do not pool banks into a replacement primary score, add samples, or choose a bank because it favors a model. If bank disagreement is material, report that numerical limitation rather than declaring a winner.

The canonical-first-allocation sensitivity uses **64 branches** for the twelve original/continued neural checkpoints at the 21 primary eligible origins: `12 × 21 × 64 = 16,128` additional ten-round branches. Compare it to the main first bank using the same seeds. Total planned branches: **150,912**. There are still only 24 human baseline groups, and 21 in the primary population.

Freeze a seed table under namespace **20261015**, keyed by human group, origin, family, training seed, bank and branch index. Original and continued budgets within the same family share streams for Monte Carlo comparison; different residents receive separate draws. This computational coupling does not establish individual human counterfactuals. Exact copies of an unchanged checkpoint under identical settings should reproduce the same forecast bank.

If a simulated pool reaches exact zero, preserve its absorbing state and zero-pad future pool/offers/returns/surplus through the requested horizon; cumulative window surplus remains at its final value. Keep actual termination time separate. Do not average rewards only over executed rounds. Preserve source residuals in observed outcomes rather than replacing observed futures with equation-based estimates.

## One-step collective calibration

First re-evaluate **individual contribution prediction** for every continued checkpoint on all original **32 Experiment 1 test groups**, including recorded M1, using the unchanged Task 03 nonforced NLL and group/mechanism aggregation. Report paired original-versus-continued differences, alongside the original scores. Also report the same individual metric on the matched 24 baseline groups, with the three mechanisms equally weighted, so a change driven by M1 is not mistaken for improvement on the collective-forecast cohort. This is diagnostic reuse of already opened evidence; no new test set is claimed. Keep individual NLL distinct from the aggregate-return NLL below.

Before attributing long-run divergence to recurrence or missing social structure, examine one-step group response under **recorded current state and recorded history**. For each checkpoint, obtain the four integer-return PMFs and convolve them to the exact predicted distribution of the group's total return. No sampled fake human population is required for this calculation.

On the 24 human baseline groups, report aggregate-return NLL and Brier score for **returns sufficient to replace the current allocation**, defined by `1.4 * total_return >= sum(current_offers)`. Use the same criterion on observed contributions. Restrict headline calibration to rounds with at least one nonforced current action, average rounds within group, and weight groups/mechanisms equally. Fix probability bins at increments of 0.1 before evaluation.

Compare original and continued learners, using exact probability sums for the event. This checks whether reasonable individual predictions yield calibrated one-step collective renewal probabilities. The convolution assumes conditional independence; a discrepancy can reflect marginal misspecification, heterogeneity, residual dependence or numerical conventions. It does not uniquely identify any one cause. Do not add a new copula, latent-type model or tuned dependence correction in this task.

## Evidence, figures and a forecast view

Save the frozen Task 04 configuration, continuation provenance/checkpoints, complete learning curves, seed table, per-group/per-seed/per-origin/horizon scores, eligibility counts, endpoint draws and calibration summaries. Store enough selected full branches to reproduce the illustrated cases; all branches must be regenerable from the seed table and checkpoints. Keep large caches and redundant per-step emissions out of git. Keep original Task 03 artifacts byte-identical.

Produce figures answering three questions: how additional training changes validation and one-step prediction; how forecast error changes with horizon and observed prefix; and how predicted collective trajectories differ from observed futures. Show the primary budget comparison and group interval prominently. Separate human sample uncertainty, training-seed variation, forecast distributions and Monte Carlo stability; do not label one as another.

Add a compact **forecast-from-observed-history** view to the existing town. Select a human group, model/budget and origin; show its actual prefix followed by a selected generated branch, alongside the actual continuation and an ensemble uncertainty band. Mark the precise observed/forecast boundary. Use a deterministic default and allow access to every group/condition, including failed forecasts. A median branch is an illustration, not a substitute for its distribution.

Keep common time coordinates and corresponding y-axis scales, all source/checkpoint/seed labels, resident inspection and real numerical values. Distinguish observed future, model forecast, and post-termination padding. If a forecast continues after the visible 40-round episode limit, that is a bug; no new horizon protocol is being introduced.

Verify the actual browser interaction and one displayed forecast against saved Python evidence. Capture a real screenshot. Do not redesign the application, introduce a hosted service, add an LLM narrative layer or spend the task on infrastructure.

## Complete, interpret and publish

Verify consequential boundaries: continuation restores epoch-120 optimizer/RNG state; original files remain intact; no current/future actions enter prefix warming; the origin observation is processed once; all residents act simultaneously; legal support and budgets hold; horizon indexing and zero padding agree with hand-checked examples; energy-score weighting and paired bootstrap use groups; and fresh-process loading regenerates a saved forecast.

Run all six 360-epoch continuations, all prescribed forecasts and scores, the numerical sensitivity, and one-step calibration. Report actual resource use. If a blocker prevents completion, preserve checkpoints and name the missing experiment rather than reporting the framework as the result.

Write the measured conclusion without forcing a favorable finding. More optimization may improve one-step prediction and collective forecasts, improve only one of them, leave both unchanged, or worsen out-of-sample performance. The original versus continued controls distinguish a budget effect within this model family; they do not establish that all remaining error requires a particular social theory.

Keep Experiments 2–3 unopened and name them as potential future transfer evidence, subject to a separately declared protocol and instruction differences. A later contribution could test whether explicitly modeled persistent heterogeneity, social dependence or learned memory yields reliable collective forecasts and institutional rankings under genuine shifts. This task supplies the measured failure boundary and stronger baselines needed to choose that question; it does not yet claim such a contribution.

Update the README and research report with the actual result, limitations, figures, trained/forecast launch commands and screenshot. Mark Stage 04 complete only with completed experiments and working forecast inspection. Keep operational details out of the main research narrative. Do not begin institutional optimization, NAS or evolutionary search during this task.

The user authorizes this continuation experiment, forecast study, viewer addition, documentation and publication. Commit and push completed work to the verified `ReloadLightly/evopolis` origin without force, following ordinary access controls. Preserve existing user work and report a precise access/resource blocker if publication or completion fails.
