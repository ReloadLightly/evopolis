# Task 03 — Train the first inhabitants

## Outcome

Implement and **run to completion** the first behavioral-learning experiment: train probabilistic agents on released human decisions, measure their predictions on held-out groups, save reloadable weights and learning curves, and watch newly generated communities in the existing 16-bit viewer.

This task ends with trained models and measured results, not training scaffolding or a pilot. It does not require the recurrent model to win. A simpler model winning is an interpretable finding. Preserve the distinction between predicting people and producing prosperous simulations.

## Read and preserve

Read `AGENTS.md`, `README.md`, `docs/data-inventory.md`, `docs/protocol.md`, `docs/VISUAL_STYLE.md`, `docs/viewer.md`, and the completed Task 01–02 briefs. Inspect `evopolis/world.py`, the source/checksum helpers, and the existing viewer/data interfaces before adding code.

Build cumulatively in this repository. Preserve existing user edits, recorded trajectories, source hashes, Task 01 results, and the working viewer. Confirm `origin` is `ReloadLightly/evopolis` before publication. Use the existing visual theme while keeping numerical scales, chart geometry, labels and uncertainty exact.

## Freeze the experiment before fitting

Use **human Experiment 1 only**: 160 groups, four mechanisms, four players, and 40 recorded rounds per group. Read `Exp 1` provenance explicitly; never admit recorded BC1/BC2 outcomes as human training targets. The unavailable original BC1 training cohort cannot be reconstructed from this release. These are new EvoPolis fits.

A fresh planning audit of the pinned source verified **6,400 group-rounds and 25,600 player decisions**, with no noninteger targets or violations of `0 <= contribution <= floor(offer)`. Only **12,814 decisions are nonforced**: Equal 1,832; Proportional 2,893; M1 4,198; Mixed 3,891. The remaining 12,786 decisions are forced zero. Recheck these counts when preparing the training arrays; they explain why the primary denominator must exclude forced actions.

Create a deterministic, mechanism-stratified group split with split seed **20261001**:

| Split | Groups per mechanism | Total groups |
| :--- | ---: | ---: |
| Training | 24 | 96 |
| Validation | 8 | 32 |
| Task 03 test | 8 | 32 |

Sort group keys before seeded assignment, record the random-generator implementation, and save the exact membership manifest. Keep all four players and all 40 rounds together. Use full episode keys and validate `launch_id` isolation between splits. Never split rows or player positions independently. Audit expected counts and source provenance before training.

Keep Experiments 2–3 outside all Task 03 fitting, selection, and outcome inspection; they are candidates for a later declared transfer experiment. Their instructions and mechanism distributions differ from Experiment 1. Defer Experiment 4's repeated games and different continuation protocol.

The Task 03 test is held out from fitting and selection, not historically unseen by the project: Task 01 already described all Experiment 1 groups. Record when test evaluation is opened, the frozen configuration, checkpoint hashes and code revision. Finish all declared fits and validation selections before opening it. Do not revise the model or hyperparameters in response to test results. Necessary correctness repairs must be documented with any affected rerun and prior opening.

After publication, these test outcomes are available to inform subsequent work; they cannot then be called pristine confirmation of a later evolutionary experiment. Opening a future transfer cohort likewise consumes its status as an untouched endpoint. Cross-group repeat participation remains unverifiable because participant identifiers are absent; do not claim participant-independent evaluation.

## Data and observation boundary

Stream the pinned CSV once into compact human-only arrays. Do not load all 113 columns or the full human/synthetic release into a large dataframe. Retain observed values and complete histories, including depleted rounds.

Construct the nine inputs through the existing `participant_observation()` convention: current offers to all four players, previous contributions from all four players, and current pool, with the focal player first and the existing cyclic rotation. Divide these inputs by 200. Reconstruct previous contributions within the same episode; initialize them to zero at the first round. All models receive the same nine inputs.

Use a shared behavioral model across resident positions, with separate recurrent states. Reset hidden states between episodes, training batches and independent evaluations. Neither training order nor one group's hidden state may carry information into another group.

Never input current-round contributions, current rewards, next pool, unlagged cumulative rewards, post-game responses, completion flags, group IDs, mechanism labels, source paths, future game length, or a remaining-round countdown. Recurrent memory may infer elapsed experience; the undisclosed horizon is not an observation. Current choices are simultaneous.

For every focal offer `e`, define the legal integer support **exactly** as `0, ..., floor(e)` using the stored source value. Audit integer-valued targets against that support before fitting. Do not round offers upward, add an epsilon to the floor, clip targets, drop discrepant rows, or otherwise repair support violations silently. Diagnose an unexpected violation and document the consequence before claiming a valid fit.

## One common probabilistic emission

Use a zero/maximum-inflated beta-binomial distribution for every fitted family. Let `n = floor(e)` and let the network emit three mixture logits plus two shape parameters:

```text
(w_zero, w_max, w_beta) = softmax(mixture_logits)
alpha = softplus(alpha_raw) + 0.05
beta  = softplus(beta_raw)  + 0.05
P(c | x, history) = w_zero * I[c = 0]
                  + w_max * I[c = n]
                  + w_beta * BetaBinomial(c; n, alpha, beta)
```

The beta-binomial component includes both endpoints. For `n=0`, define `P(0)=1` directly and zero loss; retain that round in the state history. For `n=1`, correctly combine overlapping endpoint and beta-binomial probability masses. Evaluate log probabilities with stable log-gamma/log-sum-exp operations, including appropriate precision where needed. Assert finite losses and gradients; do not conceal numerical failures through arbitrary probability clipping.

The distribution has full legal support and permits exact integer sampling. Verify normalization and sampling on boundary and typical offers. Its shape may miss interior multimodality or heaping in human choices; expose that limitation in diagnostics rather than beginning an unplanned architecture search.

## Models and fixed training budget

Fit the following four families, each under training seeds **17, 29, 43**: **12 fits total**.

| Family | Specification | Purpose |
| :--- | :--- | :--- |
| Constant | Five fitted emission parameters, independent of inputs except legal support | Population distribution baseline |
| Linear | Nine inputs mapped linearly to five emission parameters | Simple conditional baseline |
| Feedforward | `9 → 64 → 52 → 5`, ReLU hidden activations | Nonlinear control without recurrent state |
| Recurrent | One-layer GRU, input 9, hidden 32, linear five-output head | Learned behavioral memory |

The stated MLP and standard biased GRU/head have approximately 4,285 and 4,293 parameters respectively; report actual counts. The feedforward model is not completely memoryless: its observations already include previous contributions. Do not add dropout, mechanism embeddings, latent personalities, feature search, or hyperparameter sweeps in this experiment.

Also score an **unfitted uniform distribution over legal integers** as a reference. Deterministic scripts are not substitutes for probabilistic baselines.

**Before importing PyTorch**, measure current available WSL RAM, swap, disk and CPU. Task 02 observed only 831 MiB available out of 3.7 GiB, with swap full; that is a prior measurement, not an assumption about current capacity. Then benchmark two training epochs on training groups only. Record time and peak process memory, estimate the full experiment, discard benchmark state and restart declared fits from their seeds. Do not tune architecture, epochs or learning rates from this benchmark.

Use at most two CPU compute threads, one when memory/runtime measurements favor it, and `num_workers=0`. Preserve an **effective batch of eight complete groups**: if necessary, use one- or two-group microbatches with gradient accumulation, weighting each group's loss by `1/8` before one optimizer step. Keep all four player sequences together within a group. Declare the microbatch/thread choice before the main run. Do not silently cut seeds, epochs or model families after an OOM; preserve checkpoints and record any material budget deviation before opening test results.

Resolve a compatible **CPU-only PyTorch build**, pin its exact version and CPU wheel source in the dependency lock, and use the repository's supported Python range. Do not install CUDA dependencies or require a GPU. Reuse the existing dependency workflow; no new service, experiment platform or container stack is needed. Keep only one training process active and avoid a browser process during memory-intensive work when practical.

Freeze these settings for the main experiment:

- **120 epochs per fit**, with complete 40-round sequences and no early stopping.
- Adam with learning rate `1e-3`, weight decay `1e-4`, and gradient-norm clipping at `1.0`.
- A seeded shuffle of whole training groups per epoch; all four player sequences stay together, with an effective optimizer batch of eight groups.
- Validation after every epoch; retain the checkpoint with the lowest declared validation score, breaking exact ties by earliest epoch.
- No learning-rate schedule, test-driven retry, or selection of a lucky training seed.

Run all 12 fits and the prescribed evaluations. A runtime estimate is not permission to replace the experiment with a smoke test. If a genuine access or resource blocker prevents completion, preserve runnable work and checkpoints and report the exact blocker and remaining work; do not claim Task 03 complete.

## Training objective and predictive evaluation

The primary metric is **nonforced contribution negative log-likelihood**, in nats per choice. Include a target only when `floor(offer) >= 1`. First average its losses over nonforced player-round choices within each group; then average group scores within each mechanism; finally average the four mechanism means equally.

Use this same group-balanced objective for training and checkpoint selection. In an eight-group minibatch, compute each group's mean nonforced loss before averaging groups. The balanced training split and whole-group shuffle give the corresponding equal-condition epoch objective. Do not substitute a global per-row average that weights long-active groups more heavily. Report the exact scored choice counts and forced-zero fractions by split and mechanism.

Forced-zero observations still update recurrent histories and remain in trajectory analyses. They contribute no primary loss and no gradient through a forced action. Never claim predictive success from an all-row score dominated by mechanically required zero returns. A group with no scoreable choices is an audit issue, not an implicit zero-scored success.

In **teacher-forced prediction**, use recorded current offers/pool and recorded past actions; do not use any current action before producing its distribution. Recurrent histories likewise contain only observations available up to that decision. Evaluate M1 records this way without pretending its allocator is executable.

Report:

- Primary NLL by model, mechanism and seed, with an equally weighted mean across the three seeds.
- Paired NLL differences for GRU versus feedforward (the primary model comparison), plus comparisons with linear, constant and uniform references.
- Mean absolute error `abs(E[c] - c)` in contribution units, and relative MAE `abs(E[c] - c) / offer`, on nonforced choices, using the same group/mechanism aggregation. The relative denominator is the observed offer, not its floor.
- Predicted-versus-observed zero-return and maximum-feasible-return probabilities on nonforced choices, with the same group/mechanism aggregation and calibration/offer bins frozen before opening the test. Forced-choice diagnostics may appear separately, never pooled into headline calibration.
- Training and validation curves, and frozen-initialization versus fitted validation scores showing what parameter training changed.

For uncertainty, average a group's scores across the three fitted seeds, then use **2,000 paired, mechanism-stratified group bootstrap replicates** with bootstrap seed **20261003**. Resample the same groups for every model in a replicate. Three seeds are optimization repeats, not three independent copies of the human groups. Report seed variation separately; avoid treating the 32 test groups, their four players, rounds and seeds as independent observations.

Do not choose the reported model seed using test performance. Keep all seed checkpoints. An ensemble is optional only as a separately labeled additional predictor with its combination rule fixed before test opening; it must not replace the declared seed-averaged comparison.

## Save weights and support practical continuation

Provide simple documented commands to prepare the split/data, run or resume training, evaluate frozen checkpoints, generate trajectories, and launch the existing viewer. A small module/configuration is sufficient; keep modules focused on this experiment.

Save best-validation weights and a last-epoch resume checkpoint per fit. Resume state includes model, optimizer, epoch, best validation score and RNG/shuffle state; validate source, split and configuration hashes before continuation. Write checkpoints atomically. Save compact checkpoints regularly during substantial runs and print useful epoch/fit progress, loss, elapsed time and remaining fit count.

Persist feature order, normalization, emission definition, architecture, seed, selected epoch, source/split/configuration hashes, software versions and code revision beside checkpoints. Loading on CPU in a fresh process must reproduce saved predictions. Preserve training logs and machine-readable group metrics under a compact Task 03 results directory. Keep raw data and bulky temporary caches out of git; the tiny trained weights and their metadata are part of the deliverable.

## Generate actual learned communities

For **each of four fitted families, three training seeds and four executable allocation baselines**, generate **64 independent games**: **3,072 games total**. Use all best-validation checkpoints, without retaining only successful games. Predeclare a deterministic rollout-seed table, distinct from training/split/bootstrap seeds, and save it. Reusing seed indices across conditions does not make simulated residents identifiable human counterfactuals.

Use Equal, Proportional, Mixed and Interpolating through the existing Python `allocate()` and `WorldState`; do not implement a second environment in JavaScript. Each game starts with the specified 200-unit pool and four reset resident states. All four residents share fitted weights but have separate histories and sampled actions. Produce all four distributions from the same pre-decision state, sample legal integer returns using independent draws conditional on their observations and histories, then advance the world once. Never reuse one random variate across residents. Conditional independence is a modeling assumption, not evidence that human simultaneous choices are independent. Recurrent state changes during play; parameters remain frozen.

Keep the exact-zero/40-round scheduler and the documented accounting. Do not insert the source's unexplained 0.01 floor. **Do not replay recorded M1 offers into a generated world** or label a baseline as M1. M1 remains recorded playback and teacher-forced response evidence only.

Use a common 40-round outcome denominator. For generated games, mean surplus is **total retained resources / (4 × 40)**, including games ending at exact zero before round 40. Zero-pad pool, allocations, returns and surplus after exact-zero termination for these summaries; hold cumulative surplus at its final value. Report actual termination time separately. Never reuse the sandbox's average over executed/active rounds as a figure-comparable surplus statistic.

Compute Gini from the four players' full-horizon mean surplus, consistent with the existing definition; mark an all-zero denominator undefined and report its count. Report depletion (`pool < 1`), exact-zero termination, active allocations (`offer >= 1`), and final sustainment (`pool > 1`) distinctly. Do not collapse these definitions into one survival measure.

Plot generated surplus, inequality, pool trajectories and participation distributions, preserving variation across training seeds and games. Compare Equal/Mixed/Proportional with the relevant Task 03 held-out human groups. Interpolating has no human Experiment 1 counterpart: label it exploratory transfer. Generated games quantify model behavior, not new human observations or independent human sample size. Account for the known recorded/simulated numerical-floor difference when interpreting depletion tails. Full trajectory validation and institutional generalization remain Stage 04 research.

## Put trained inhabitants in the town

Add a selectable **“New trained EvoPolis agents”** source to the working viewer. Keep recorded humans, recorded upstream BC models, and scripted policies separately selectable and clearly labeled. Make every prescribed generated game accessible through checkpoint/mechanism/seed selection; never offer only handpicked good episodes.

Show family, training seed, selected epoch/checkpoint identity and rollout seed. Use the existing resident inspection, exact numerical values, playback and history panels. Select a deterministic representative default, such as the median-surplus generated GRU/Equal episode with stable tie-breaking, and state its selection rule. The default is a display choice, not selection of a superior checkpoint.

For comparison panes, add **common round coordinates and shared y-axis limits for corresponding quantities** so trajectories can be compared honestly. Do not rescale each pane to make unequal outcomes look alike. Preserve existing recorded playback; show padded generated rounds as post-termination padding, with the actual ending marked. A matched clock does not make different human groups paired counterfactuals.

Check play/pause, backward seeking, resident inspection, source/checkpoint changes, comparisons, and seed reproducibility in the actual local browser. Verify one generated episode against Python output and loaded checkpoint predictions, not just a mocked route. Capture a real screenshot of trained inhabitants; use no invented motives, dialogues, learning curves or metrics.

## Verification, reporting and publication

Focus verification on consequential risks: exact integer support and `n=0/1`, finite normalized probabilities and gradients, observation timing and rotation, group separation, hidden-state reset, simultaneous decisions, fresh-process checkpoint reload, resume equivalence, and fixed-horizon generated summaries. Confirm future-outcome fields cannot affect a current prediction. Preserve existing numerical tests and empirical artifact hashes; avoid infrastructure-only test expansion.

Produce a concise report with data/split counts, compute used, actual learning curves, seed-averaged predictive results and group uncertainty, calibration failures, generated outcomes, and limitations. Keep human predictive fidelity separate from simulated prosperity. If GRU does not outperform simpler learners, report it without reframing simulation surplus as learning success.

Update the README with what was actually trained, the measured result, accurate status, figures, screenshot and runnable commands. Keep operational details in the task report. Do not claim upstream BC1 reproduction, human-level fidelity, online weight learning, evolutionary improvement, new RL mechanism training, or RSI. A fitting success alone does not establish reliable counterfactual social prediction.

Completion requires all 12 fits, declared prediction evaluation, reloadable checkpoints, learning curves, 3,072 generated games with correct summaries, and verified trained-agent playback. Mark Stage 03 complete only with that evidence. Identify the specific remaining scientific question for Stage 04; do not begin evolutionary search in this task.

Commit and push the completed work to the verified `ReloadLightly/evopolis` origin without force. The user authorizes implementation, the declared training/evaluation, viewer integration, documentation and publication. Continue through reversible implementation choices with ordinary permission controls intact. Report a concrete blocker if access prevents completion rather than silently leaving a scaffolding-only task.
