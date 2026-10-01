# Scientific review after Task 05

Reviewed 2026-10-01 at `022eff6d23d93b65df2ac0afc8b9cb9509364a34`. Task 05 is complete. [Task 06](tasks/06-frozen-human-transfer.md) is specified, not executed.

## What the completed experiment establishes

The four matched P0/P1/H0/H1 families were fitted with three optimization starts each, all completing 480 epochs. All 89,856 forecast branches were generated. The explicit peer-history response did not establish incremental predictive improvement after accounting for persistent resident variation.

| H1 minus H0; lower is better | Paired estimate | Reported 95% human-group interval |
| :--- | ---: | :--- |
| Individual prequential NLL; primary window | +0.003898 | [−0.00103, +0.00921] |
| Joint collective endpoint energy; principal bank | +0.001736 | [−0.00428, +0.00849] |
| Joint collective endpoint energy; independent bank | +0.00025 | [−0.01020, +0.01218] |

The primary population is the same 21 human groups used in Task 04: five Equal, eight Mixed and eight Proportional groups. Choices in source rounds 5–14 are scored after warming from rounds 0–4. The collective score measures the two-dimensional endpoint comprising pool/200 and total retained resources over the ten-round window/2000. Three fitted starts and repeated simulation draws do not increase the number of human groups.

All fitted peer coefficients are positive, but a positive coefficient does not demonstrate useful incremental prediction or identify conditional-cooperation preferences. The three secondary factorial contrasts also have primary intervals spanning zero. These results concern the fitted procedures and available evidence; they are not equivalence tests proving the mechanisms irrelevant.

The [completed report](conditional-responses.md) contains the full results. Earlier Task 04 findings remain intact. Neither task identifies why the GRU's individual and collective scores diverged.

## What this review checked

Read the Task 05 specification, configuration, fitted-parameter table, report and completion receipts. Inspected the conditional model, training, generation and evaluation implementations for whole-sequence latent integration, past-only prediction, simultaneous updates, persistent branch effects and group-level aggregation. No blocking error was found in that inspected path.

An independent standard-library calculation from `results/task05/primary_paired_groups.csv` reproduced H1−H0 point estimates of 0.003898214066491507 NLL and 0.0017363767773741854 energy, using seed averages within groups and equal mechanism weighting. The forecast-verification, independent-score-audit, integration and preservation receipt hashes match the completion manifest. This review did not refit models, regenerate forecasts or repeat the full numerical audit. The intervals above are the published Task 05 intervals, not newly computed intervals in this review.

The run's existing independent audit reports agreement with raw Experiment 1 targets and scores, and preservation of 158 earlier scientific files. Its integration correction was substantive: 21 nodes failed development-only checks, while the final six persistent fits passed 41-versus-81-node comparisons. The correction was made before test scoring.

## Two limits that change the next task

**The training budget did not establish convergence.** All twelve selected checkpoints occur at epoch 479 or 480. Validation NLL improved by approximately 0.0022–0.0054 over the final 40 epochs. The latent scales are neither capped nor shown to be numerically invalid, but the fitted procedures may still change with additional optimization. This justifies a declared continuation budget for every family, without claiming that any finite budget guarantees convergence.

**Small collective differences are sensitive to simulation noise.** For example, P1−P0 energy changes from −0.00336 in the principal bank to +0.00862 in the independent bank. Both use only 64 branches per checkpoint/group. The next comparison needs greater fixed sampling precision and a separate estimate of Monte Carlo error. More branches will not resolve limited human sample size.

The descriptive peer-trace projection does not support near-complete linear redundancy: substantial residual variation remains. Broader nonlinear redundancy, endogenous allocations and omitted information remain possible. Broad five-round latent posteriors describe limited information about individuals; they do not by themselves establish failed population-parameter estimation.

## The next scientific question

Task 06 asks whether the explicitly specified, Experiment-1-trained response models predict new human groups under changed conditions. It combines a fixed continuation budget, improved forecast precision and an Experiment 2 transfer evaluation in one complete experiment. It retains H1−H0 as the primary contrast rather than promoting a favorable exploratory ranking to primary status. Experiment 3 remains reserved for a later model-revision/evolution study; this preserves an evaluation resource without claiming that its outcomes will remain appropriate for every future hypothesis.

Experiment 2 provides 40 Proportional, 40 Interpolating and 40 M1 groups. Proportional and Interpolating support live resource forecasts after source verification; the unreleased M1 policy supports recorded-choice evaluation only. Instructions and participants change, and Interpolating also changes the allocation rule. These are joint shifts, not isolated causal treatment effects. The current predictors do not take mechanism instructions as input.

## Contribution and literature boundary

The [original Koster/Pîslar study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11929920/) already grounds models in human decisions and evaluates collective outcomes. [Local Predictability and Collective Fidelity in LLM-Agent Societies](https://arxiv.org/abs/2609.35813) directly studies local-versus-collective forecast discrepancies. [Collective cooperation without individual fidelity in LLM agents](https://arxiv.org/abs/2606.30454) compares LLM cooperation with human network-game behavior, including individual heterogeneity and conditional responses.

EvoPolis should therefore not claim the first discovery that individual accuracy and collective fidelity can disagree, or the first human-grounded multi-level evaluation. The potential contribution is a well-specified empirical comparison of behavioral structure, resource feedback and transfer. Task 06 can strengthen or weaken that account. It does not promise publication, identify motives, or demonstrate evolutionary improvement. The broader evolutionary program remains a substantive subsequent experiment with fixed-model and budget-matched search controls.
