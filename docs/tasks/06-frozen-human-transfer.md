Superseded on 2026-10-01 by Task 06 — Institutional validity; not executed.
# Task 06 — Learned human responses under changed conditions

**Status: specified; not executed.** Execute the complete experiment when assigned: continue fitting, validate numerical precision, freeze the procedures, evaluate human Experiment 2, inspect the findings, update the scientific presentation, commit and push. Implementation alone is incomplete. This task follows the [Task 05 scientific review](../task05-review.md).

## Question and rationale

> Do explicit peer responses and persistent individual differences help predict new human communities after the learning procedure is fixed?

Task 05 found no clear matched benefit on the already examined Experiment 1 test groups. Its twelve validation minima lie at epochs 479/480, and 64-branch forecast banks leave material numerical variation. This task increases the declared learning budget and forecast precision before testing transfer. It does not add another mechanism in response to an unfavorable result.

The research sequence is one end-to-end assignment. No separate pilot, manual approval between phases or extra confirmation is required. Routine resource measurements guide implementation and resumption, not changes to hypotheses or sample selection. Preserve all completed evidence and user changes. If a genuine correctness or access blocker prevents completion, save the exact checkpoint and explain the blocker; do not substitute a successful-run claim.

Read `AGENTS.md`, this brief, `docs/task05-review.md`, the Task 04/05 reports and configurations, `docs/protocol.md`, `docs/data-inventory.md`, the source receipts, and relevant model/training/forecast code. Inspect the original paper's allocation and instruction methods. Preserve the FFV-inspired viewer and earlier modes.

## Evidence and analysis boundaries

- Training and checkpoint selection use only the existing Experiment 1 training/validation split: 96/32 groups. The 32 Experiment 1 test groups remain diagnostic evidence that has already been examined.
- Experiment 2 is the new transfer population: 40 Proportional, 40 Interpolating and 40 M1 groups. These counts and published aggregate findings are known; this is a reserved EvoPolis model-evaluation cohort, not a claim that the source publication was unknown.
- Read Experiment 2 behavioral trajectories for this task only after the transfer protocol and all selected procedures are frozen. Before then, metadata-only routing and previously documented source/schema information are sufficient. Keep Experiment 3 outcomes out of fitting, selection and new evaluation; keep Experiment 4 outside this task.
- Do not retrain, recalibrate, choose a seed, alter features or choose a model family using Experiment 2 outcomes. Updating a resident's posterior or recurrent state using its legally observed prefix is permitted and is distinct from changing fitted population weights.
- Retain all four matched families and all three starts. H1−H0 remains primary. H0's favorable Task 05 point estimate is not a new selection rule.
- No institutional optimization or evolutionary search is performed in this task. Their evaluation must remain separate from human predictive fidelity.

## Phase A: fixed continuation of the learned procedures

Freeze `configs/task06.json` before new fitting. Record source identities, all protocol choices, numerical tolerances, checkpoint rules, features, seeds, scoring and forecast budgets.

Continue **P0, P1, H0 and H1 × seeds 17, 29, 43** from each Task 05 `last.pt` at epoch 480 to **epoch 1,920**, adding 1,440 epochs per fit. Restore weights, optimizer, RNG and shuffle state. Keep the same model, likelihood, learning rate, regularization, clipping and effective batch size. Do not reset optimizer state, change architecture or substitute fresh starts. Use separate Task 06 modules/artifacts and output directories; never overwrite Task 05 checkpoints, curves, sources or results protected by its manifests.

The selection candidates are the **frozen Task 05 best checkpoint plus every newly completed epoch 481–1,920**. Choose the earliest candidate with minimum Experiment 1 validation group-balanced nonforced NLL, independently for each family/start. The old candidate can win. Do not claim to have reevaluated unavailable intermediate checkpoints from epochs 1–480. Save sufficient intermediate Task 06 state to reproduce selection if numerical correction requires rescoring or rerunning.

All fits finish the assigned budget. A minimum at the boundary remains evidence of incomplete convergence, not a reason to extend indefinitely or stop before transfer. This compares explicitly named fixed-budget procedures; it is not a definitive test of optimally fitted model families. Report validation curves, selected epochs, changes over the last 160 epochs, and beta/eta/sigma for every start.

Use float64 and the established persistent likelihood. Begin with the validated 41-node quadrature, and compare selected persistent models with 81 nodes on all Experiment 1 training/validation groups. Retain the Task 05 tolerances: maximum group/resident NLL change per nonforced choice and maximum expected-fraction change below 0.001, with posterior normalization and tail checks. Inspect concentration and scales. If needed, increase resolution or use accurate adaptive integration, document the correction and redo affected continuations/selection before opening transfer outcomes. Rescore the original selected checkpoint at the accepted resolution as its fixed anchor; preserve the original published scores.

Fit sequentially, with one CPU thread, no data workers and effective batches of eight complete groups through memory-appropriate microbatches. Profile free RAM, swap and disk before Torch. Save atomic best/last checkpoints and resumable phase progress. Existing Task 05 fit times imply approximately **7.2 additional hours for the continuation alone** at comparable throughput; forecasts and integration add time. This is an estimate, not a time guarantee or a reason to reduce the assigned experiment silently.

## Phase B: freeze the transfer procedure

Before parsing Experiment 2 behavioral outcomes, write an immutable opening manifest containing:

- Selected hashes for all twelve continued-family checkpoints, including any retained old checkpoint.
- Six fixed neural references: the three continued feedforward and three continued GRU checkpoints from Task 04. Do not retune or retrain them in this task. Their different information set prevents an architecture-only interpretation.
- Source/config/code hashes, all cohort definitions, selection rules, numerical settings, scores, forecast banks, coupling and analysis seeds.
- A record of any previously exposed Experiment 2 evidence beyond public design/aggregate information. Do not relabel exposed evidence as unseen.

Verify Interpolating against the published rule and the existing implementation: equal-mixture weight `w=(R/200)**22`, with previous contribution shares in the proportional component and equal first allocation. Test endpoint weights and resource accounting using constructed inputs or development information. Preserve the existing documented zero-return fallback and numerical conventions. Do not reconstruct an M1 policy from recorded future allocations.

After freezing, parse Experiment 2 rows only into a new typed cohort with complete `(mechanism label, launch_id, episode_id)` keys. Verify disjoint group identifiers relative to Experiment 1, complete round indexing, legal actions and published allocation replay residuals. The release cannot establish participant disjointness across different groups; retain that qualification. Do not alter the physical equation to fit new residuals. A genuine source/implementation inconsistency requires a documented correction and an explicit record that outcomes have been opened, not silent protocol replacement.

## Phase C: transfer predictions

The primary forecast origin is **k=5**, horizon **h=10**, source rounds **5–14**. The eligible transfer population consists of Experiment 2 Proportional and Interpolating groups with at least one offer >=1 at the origin. Determine eligibility from the observed boundary, never future survival. Report counts separately; the number is unknown until opening. Also score the full 80-group population as a secondary analysis so that the eligibility restriction remains visible.

Two primary outcomes, H1 minus H0:

1. Individual nonforced prequential NLL in source rounds 5–14 of those exact groups. Warm from rounds 0–4. Predict each round before updating from recorded actions. Average choices within group, optimization starts within group, groups within condition, then weight Proportional and Interpolating equally. Report any unavailable individual score explicitly rather than silently changing cohorts.
2. Off-diagonal energy score for the **two-dimensional** vector `[pool after round 14 / 200, sum of all four residents' retained resources in rounds 5–14 / 2000]`. Generate the future freely from the observed origin. Score each fitted start's distribution separately and then average within group; do not pool starts into a new mixture forecast.

Use 2,000 condition-stratified paired whole-group bootstrap replicates with PCG64 seed **20261025** and identical resampled keys for every procedure/outcome. Report both condition-specific estimates and the prespecified equally weighted result. Human groups are empirical units; fitted starts, rounds and branches are not. Intervals condition on the fitted models and finite forecast banks.

The primary bank supplies the declared comparison. Claim joint predictive improvement only if **both primary 95% intervals lie below zero**. Positive-response interpretation additionally requires beta > 0 in all three selected H1 fits; report every H1 and P1 coefficient. H0 fixes beta at zero by construction. Report mixed signs, uncertain intervals and failures plainly. The independent bank must also be shown; disagreement prevents an unqualified robustness claim. Failure of the criterion is not evidence of equivalence or rejection of conditional-cooperation theory.

Secondary analyses are P1−P0, H1−P1 and H0−P0; full-game individual NLL on all 120 Experiment 2 groups, with M1 reported separately; matched-window scores of the six fixed neural references; pool/surplus CRPS, nominal 80% coverage and temporally ordered pool-marginal CRPS; and recorded-state one-step renewal calibration. For full-game NLL, average nonforced choices within group, starts within group, groups within condition, then the three conditions equally; report each condition separately and any group with no nonforced choices. An optional two-executable-condition summary is a separately labeled estimand. Label multiple secondary comparisons exploratory. Neural comparisons concern whole fitted procedures with different input representations.

M1 permits recorded-state choice evaluation only. Proportional and Interpolating are the only live allocation rules used for transfer. Do not use future observed M1 offers as if they were an autonomous policy. Models receive only their existing permitted inputs; the changed mechanism instructions are known to humans but are absent from these predictor interfaces.

For persistent models, infer each resident's posterior from its observed prefix and draw one effect per resident/branch, held fixed throughout that future. Keep separate resident histories and resolve simultaneous choices before updates. Preserve integer support, recorded origin offers and unallocated residue, the capacity/multiplier, exact-zero absorption and full-horizon flow padding. Update future allocations from generated behavior. All population parameters stay frozen.

Retain the numerical accuracy contract on transfer inference. Before examining cross-model outcome comparisons, check each persistent checkpoint over all transfer histories using the ladder 41, 81, 161, 321 nodes, starting no lower than its accepted development resolution. Compare adjacent levels and accept the lower resolution only when maximum group/resident NLL change per nonforced choice, expected-fraction change and upper-tail probability change are each below 0.001, with normalized finite posteriors. Use one accepted resolution per checkpoint across all transfer groups and both banks. If the ladder fails, use documented accurate adaptive one-dimensional integration and verify it at stricter numerical tolerance; do not refit or reselect population parameters. Refinement depends on integration discrepancy, not favorable model ranking. Record the initially frozen rule, the accepted resolution and any dated numerical amendment; rerun all affected predictions and both banks consistently. If accuracy cannot be established, label the affected estimates numerically unresolved rather than treating them as empirical evidence.

## Forecast precision and budget

Use **512 branches per checkpoint/group in each of two independent banks**. Every principal forecast covers the same ten-round window. There is no need to regenerate every Task 04/05 origin and horizon for this question.

| Comparison | Checkpoints | Groups | Banks × branches | Maximum branches |
| :--- | ---: | ---: | ---: | ---: |
| Experiment 1 budget diagnostic: original and continued four-family fits | 24 | Existing 21 primary groups | 2 × 512 | 516,096 |
| Experiment 2 transfer: continued four-family fits plus neural references | 18 | All 80 executable-condition groups | 2 × 512 | 1,474,560 |
| **Total** | | | | **1,990,656** |

Scoring all 80 transfer groups also supplies the eligible subset without another simulation. Exact point-mass depleted forecasts may be stored compactly, with their logical branch counts explicit. Complete this fixed budget; do not increase it until an interval becomes significant. Reuse checkpoint-identical distributions only if identities and random streams agree, and report computational reuse.

Use namespace **20261024** with counter/index-based uniforms keyed by experiment/full group key, optimization start, bank, branch, resident, forecast step and draw purpose. Within a comparison, couple families and budgets using the same action uniforms and fixed inverse-CDF sampling; omit family/budget from the coupling key. Latent and action uniforms must have distinct purposes. Neural references use their legal PMFs with the same action uniforms. Preserve each model's marginal distribution and independent branches within a model. Different banks are independent. Test these properties directly; shared initial RNG seeds alone are insufficient and variance reduction is not guaranteed.

Estimate computational uncertainty separately from human-group uncertainty. For the principal paired energy contrasts, use a prespecified eight-block delete-block jackknife over disjoint 64-branch blocks, recomputing scores and the group-aggregated contrast after each deletion. Use the same deleted block across coupled models, retain off-diagonal score denominators, and report the resulting approximate Monte Carlo standard error conditional on the observed groups and fitted models. Show the independent-bank contrast alongside it. Do not present this numerical standard error as uncertainty about human populations or combine the banks post hoc to improve the headline.

The Experiment 1 budget comparisons are explicitly exploratory. They can show whether the additional learning changes the prior conclusion but do not become new independent evidence. Report H1−H0 at both budgets and within-family continued-minus-original effects. Evaluate transfer only after the models were frozen. Do not subtract differently weighted cohort scores and call the difference an institutional treatment effect; Proportional is the available same-rule cross-cohort comparison, and those groups are unpaired.

## Results, interpretation and presentation

Answer these questions in the final report:

1. Did the larger learning budget change individual fit and collective forecast behavior? Are curves still improving?
2. Does the explicit peer term help on new groups, separately under Proportional and Interpolating, after accounting for persistent variation?
3. Does persistence help, and how do the fixed neural references perform?
4. Which conclusions survive the independent forecast bank, and how much uncertainty comes from simulation versus the limited human cohort?
5. Where do predictions fail: resource recovery/collapse, contribution distributions or uncertainty calibration? Use outcome-independent case selection for illustrations and report support/scale changes in observed prefixes.

Instructions, cohorts and allocation conditions change together. Transfer success or failure does not isolate an institutional causal effect, identify motives, establish a learned instruction representation or explain Task 04 uniquely. An effect under Interpolating is a forecast under a specified known allocation rule, not discovery of that rule by the agents. Post-hoc analyses must be labeled and cannot alter the frozen procedure.

Read and cite the primary prior work linked in `docs/task05-review.md`. The broad individual-versus-collective discrepancy, human grounding and multi-level evaluation are not novelty claims. The contribution under investigation is the empirical behavior and transfer of explicitly compared learned response structures under resource feedback. Evolutionary model revision remains the next research direction if the evidence and evaluation design support it; no positive result is required to report this task honestly.

Create `docs/human-transfer.md`, machine-readable scores/parameters/contrasts, full checkpoint/protocol identities, phase progress, runtime/RSS measurements, and all selected/resumable fits under `results/task06/`. Save every scored branch's endpoint and pool/surplus path in compressed shards small enough for GitHub (target <40 MiB each); retain full action/latent histories for the fixed illustrative cases and deterministic regeneration for any archived branch. Stream score computation and archive verification instead of loading the whole run into RAM.

Add measured scientific figures in the existing theme: training-budget curves; individual/collective contrasts with human intervals and separately marked Monte Carlo uncertainty; transfer performance by condition; and observed-versus-predicted resource paths. Keep readable labels and exact chart geometry. Select illustrative groups by the first eligible full key within each condition, not by favorable outcomes. Integrate recorded Experiment 2 and saved transfer forecasts into the existing viewer, preserving previous modes; label cohort, procedure, prefix and generated future clearly.

Verify consequential new behavior: correct continuation/resume, protected prior artifacts, cohort routing, no future leakage, source-faithful Interpolating allocation, latent persistence, valid coupling, correct score aggregation and independent reconstruction of raw targets and primary contrasts. Reuse existing tests and limit new tests to these risks. Verify any changed viewer paths in the browser and inspect actual rendered figures. Keep README additions focused on question, methods, findings, limits and figures; put operational details in the task report.

Finish with the actual scientific conclusion, all completion/blocker facts and the next research implication. Confirm `origin` is `ReloadLightly/evopolis`, commit and push without force-pushing, preserving unrelated user changes. A published README must never imply that this experiment ran before its actual artifacts exist.
