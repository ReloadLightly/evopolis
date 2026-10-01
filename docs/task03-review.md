# Task 03 review: learned choices, unreliable collective futures

Reviewed implementation: [`e2c3747`](https://github.com/ReloadLightly/evopolis/commit/e2c3747), 1 October 2026. The next experiment is [Task 04 — Collective forecast fidelity](tasks/04-collective-forecast-fidelity.md).

## What the experiment established

EvoPolis now has genuinely trained behavioral agents. All twelve fits completed 120 epochs, produced changed parameters, and saved reloadable checkpoints. They learn distributions over legal integer contributions from human Experiment 1, with whole interacting groups separated between training, validation and evaluation. All 3,072 generated communities are available in the existing viewer. Recurrent state changes during a rollout; weights do not learn online.

The primary human test results were independently reconstructed from raw Experiment 1 records and the published weights, using a separate SciPy beta-binomial likelihood calculation. Maximum disagreement with saved per-group NLL was **1.07e-13**. Split membership, checkpoint hashes, parameter counts, earliest validation minima and the paired group bootstrap were also checked. No material error was found in the reported primary comparison.

| Predictor | Nonforced test NLL | Parameters |
| :--- | ---: | ---: |
| Uniform reference | 2.9590 | 0 |
| Constant distribution | 2.9224 | 5 |
| Linear conditional distribution | 2.8617 | 50 |
| Feedforward | 2.7238 | 4,285 |
| Recurrent GRU | 2.6944 | 4,293 |

Lower NLL is better. GRU minus feedforward is **-0.029398 nats**, with paired 95% interval **[-0.063671, +0.006312]**. This is an inconclusive incremental memory result, not proof that memory is useless. Both models already receive previous-round contributions. These intervals reflect resampling groups in this split, conditional on the fitted seeds; they do not capture every alternative split or retraining outcome.

The more consequential finding is the gap between individual prediction and generated collective behavior:

| Allocation | Human surplus | GRU surplus | Human final pool > 1 | GRU final pool > 1 |
| :--- | ---: | ---: | ---: | ---: |
| Equal | 1.788 | 2.013 | 0% | 3.65% |
| Mixed | 5.594 | 2.178 | 50% | 4.69% |
| Proportional | 6.756 | 3.563 | 87.5% | 18.75% |

Surplus is retained resources per player per round, using the full 40-round denominator. Each human cell contains eight test groups; each GRU cell contains 192 generated games across three fitted seeds. Generated games do not enlarge the human sample. These are descriptive, unpaired comparisons, not new treatment effects. Feedforward Proportional surplus is 4.117, closer in mean to the human value than GRU's 3.563, but that alone does not establish a better predictive distribution.

An independent archive audit checked all **45,081 executed and 77,799 padded rounds**, recomputed the resource equation, legal actions, retained totals, full-horizon surplus and Gini, and reproduced the reported generated means. Equation disagreement was zero. This review did not rerun browser interactions; the original implementation's browser evidence remains separately recorded in `results/task03/browser-verification.json`.

## Explanations still to distinguish

The current evidence does not identify the cause of premature simulated collapse. Candidate explanations include limited optimization, a restrictive response distribution, insufficient representation of persistent individual differences, residual dependence between residents, and errors that compound when generated actions change future observations. These can coexist.

Numerical conventions also deserve a bounded sensitivity check. Recomputing baseline allocations from the observed state in the 24 test groups yields very small offer differences, but **166 of 3,840 player-rounds** change their legal integer maximum: Equal 32, Mixed 32, Proportional 102. In these cases the canonical implementation's maximum is higher. A source offer just below 50 and a reconstructed offer exactly 50 have different integer supports and different locations for the maximum-return atom. This does not invalidate the source-faithful predictive scores, and does not by itself explain the much larger collective discrepancy. It does prevent attributing every rollout difference to psychology without checking the boundary.

Several selected neural checkpoints occur near the 120-epoch limit. Task 03 therefore establishes performance at that budget, not optimization convergence. Task 04 will continue the six neural fits to a fixed total of 480 epochs, preserving data, architecture, objective and optimizer. Its other component will forecast futures from observed human prefixes, so failure can be measured by forecast horizon and starting state instead of only by final averages.

## A credible paper direction

Task 03 is a useful empirical foundation and a concrete failure case. By itself, training a compact GRU and observing imperfect rollouts is not a demonstrated frontier contribution. A stronger paper question is:

> Which features of individual persistence and social dependence must a learned behavioral model preserve to forecast cooperation under different allocation rules?

A defensible contribution would identify a failure mechanism or boundary, compare targeted alternatives against simple models and additional-compute controls, and test the resulting claim on separately reserved human evidence. Improving simulated prosperity alone is insufficient: a simulator must predict distributions of human outcomes, including collapse and inequality. A carefully established limitation can be a contribution even if no proposed model solves it.

There is substantial prior work to build on. [Ross, Gordon and Bagnell (2011)](https://proceedings.mlr.press/v15/ross11a.html) explain the general sequential-distribution problem in imitation learning. [SLALOM (Lee and Seering, 2026)](https://arxiv.org/abs/2604.11466), a workshop paper, argues for assessing social trajectories rather than only endpoints. [Shao et al. (2026)](https://arxiv.org/abs/2609.24012) study diagnosis and adequacy of generative social simulation. These make it inappropriate to claim that the general need for trajectory validation is new. The opportunity is a specific empirical result about learned cooperation, its mechanisms, and transfer. This is a scoped literature check, not an exhaustive novelty assessment.

Task 04 is a diagnostic follow-up using already opened Experiment 1 evidence. Experiments 2–3 stay outside it, preserving them for a later declared modeling/transfer evaluation. Their instruction and institution changes require explicit treatment. Later evolutionary search can target interpretable history or dependence mechanisms, evaluated for human predictive fidelity against fixed and random-search controls. Evolution is a method for answering that question, not evidence of novelty on its own.

## What the compute comparison supports

The final Task 03 training invocation took **439.53 seconds** on one CPU thread, peaking at **306,864 KiB** process RSS. Generation took **31.76 seconds**. These exclude preparation, the documented discarded repair run, review and browser work.

The foundational [Koster/Pislar et al. paper](https://arxiv.org/pdf/2404.15059), implementation details and Methods Table 1, reports a single NVIDIA Tesla P100 and completion within 24 hours. BC1 used 700,000 updates and a 64-unit GRU; the later BC2 was substantially larger. That was more extensive training, but the relevant study did not require a giant GPU cluster. Hardware, update counts and wall time cannot be converted directly into a justified compute ratio. The original initial training cohort also comprised 537 games under a broader mechanism distribution; our 96 training groups are different evidence, not a substitute reconstruction of that unreleased cohort.

[Simile's research statement](https://www.simile.com/blog/simulation-next-frontier) emphasizes fidelity, efficient population simulation and calibrated outcomes, but does not disclose the compute budget needed for a like-for-like comparison. There is no verified basis here for a precise 1,000-times claim. EvoPolis can test a focused scientific hypothesis without matching company-wide resources; the next constraints are the modeling question and credible human evaluation, while additional optimization remains inexpensive enough to test directly.
