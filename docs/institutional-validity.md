# Institutional validity: do learned human models predict which allocation rule works?

## 1. Question and tipping point

> Does a behavioral model fitted to human play predict which allocation rule works, including a rule it never saw?

Four residents receive allocations from a 200-unit common pool, return integer amounts, and retain the rest. Returns grow by 1.4, capped at 200. With full allocation and below the cap, the next pool is 1.4Σc. Nonshrinkage requires the allocation-weighted fraction Σc/Σe ≥ 1/1.4 = 5/7; this is not an instantaneous death threshold or the arithmetic mean of individual return fractions. Small choice-distribution errors near this feedback threshold can produce large collective errors while changing likelihood little.

## 2. Simulators

Frozen references comprise Task 03 constant, linear, feedforward and GRU models; continued Task 04 feedforward and GRU models; and Task 05 P0/P1/H0/H1 controls. FA-GRU, FA-P0 and FA-H0 add the public offer-versus-previous-contribution slope and its undefined indicator. CL variants tilt legal PMFs by exp((τ0+τ1s)c/e). All learned lines use seeds 17/29/43; H families retain one resident effect throughout each game. We distinguish fitted parameters, deterministic history updates and posterior inference; no online parameter learning or evolutionary search occurs.

Recorded BC1 is the incumbent. The paper identifies its fitting evidence as “Based on Train Set 1” (537 games). BC1 predates Interpolating; BC2's 990-game collection covers “all aforementioned data”, described as “a total of 990 games”, so BC2 is descriptive. These are recorded upstream outcomes, not recreated networks. [Methods](https://www.nature.com/articles/s41467-025-58043-7)

Each learned checkpoint forecasts 512 games under Equal, Mixed, Proportional and Interpolating, with no human prefix: 1,536 games per family/rule. Means use all 40 rounds, with zero padding after exact-zero termination; human residual floors remain recorded. Surplus, survival (pool40>1), Gini, pool20 and depletion time are group outcomes. Distributional fidelity uses energy distance on (surplus/10,pool20/200,pool40/200), excluding within-sample diagonals.

## 3. Experiment 1 benchmark

The table includes all 40 human groups per rule, including already opened test groups. Separate 32-group training+validation comparisons, survival, inequality, pool20 and energy distances are in the [benchmark artifact](../results/task06/benchmark.json).

| Mean surplus/player/round | Equal | Mixed | Proportional |
| :--- | :--- | :--- | :--- |
| Humans (40 groups/rule) | 2.202 | 4.533 | 6.047 |
| BC1 (512 recorded games/rule) | 3.879 | 4.787 | 6.406 |
| Task 03 constant (1,536 games/rule) | 2.457 | 2.415 | 2.472 |
| Task 03 GRU (1,536 games/rule) | 2.102 | 2.131 | 3.663 |
| FA-H0 (1,536 games/rule) | 1.857 | 2.461 | 5.511 |
| CL(FA-H0) (1,536 games/rule) | 2.355 | 3.227 | 6.051 |

![Experiment 1 institution benchmark.](assets/task06-institution-benchmark.png)

## 4. Attenuation and sensitivity

Diagnostics average nonforced choices within groups, then groups within rule and fitted seeds. Observed validation return fractions under Equal/Mixed/Proportional are 0.494/0.595/0.515. Attenuation divides the predicted Proportional−Equal fraction difference by the observed difference.

The external audit's six fractions reproduce at quoted precision: predicted E/M/P 0.435/0.600/0.710, observed 0.404/0.629/0.745. They use nonforced Σc/Σe within each group, then equal groups: allocation-weighted fractions, distinct from Task 06's unchanged choice means. Continued Mixed fractions are 0.589/0.586/0.580 (seeds 17/29/43). The historical exp(0.6c/floor(e)) tilt raises E/M/P fractions by 0.050/0.041/0.027, costing 0.0186 NLL nats; Task 06 instead tilts by actual c/e. Continuation changes NLL by -0.0835 (three-seed mean versus original seed 17). On nonforced Proportional test choices, GRU seed 17 predicts 9.1% zero returns against 5.2% observed; one zero causes permanent exclusion under this rule. Our new Mixed audit draws survive 1/64→16/64 (MC SE 0.016/0.055). These differ from the brief's approximately 6%→38%; the external 64-game seed was unavailable, preventing exact bank replication. [Audit artifacts](../results/task06/external_audit_predictions.json) retain all rechecks and estimand definitions.

| Simulator | Predicted return fraction E/M/P | Attenuation index |
| :--- | :--- | :--- |
| Task 03 constant | 0.528/0.540/0.535 | 0.315 |
| Task 03 GRU | 0.494/0.606/0.533 | 1.814 |
| Continued GRU | 0.489/0.585/0.532 | 1.980 |
| FA-H0 | 0.468/0.581/0.561 | 4.300 |
| CL(FA-H0) | 0.505/0.596/0.571 | 3.029 |

The human validation Proportional−Equal choice-mean contrast is only 0.0216; Task 03 GRU has index 1.814, FA-H0 has index 4.300, CL(FA-H0) has index 3.029. Above-one ratios indicate amplification of this small validation contrast. The opened-test audit uses a different group subset and allocation-weighted fractions.

![Validation attenuation.](assets/task06-attenuation.png)

The fixed sensitivity sweep applies τ∈{−0.3,0,0.3,0.6,0.9} with 256 games per checkpoint/rule/point. This slice reports Mixed at an additional τ=0.6; CL already includes its selected tilt. The figures show validation NLL cost against collective outcomes and mark the resource-weighted 5/7 reference separately from group-balanced fractions.

| Mixed; additional τ=0→0.6 | Validation ΔNLL | Survival | Surplus |
| :--- | :--- | :--- | :--- |
| Task 03 GRU | 0.0195 | 0.040 → 0.320 | 2.177 → 3.768 |
| FA-H0 | -0.0060 | 0.059 → 0.177 | 2.399 → 3.184 |
| CL(FA-H0) | -0.0013 | 0.165 → 0.405 | 3.130 → 4.448 |

![Tipping-point sensitivity.](assets/task06-tipping-sensitivity.png)

## 5. FA and CL results

All nine fresh FA fits complete 480 epochs on the existing 96/32 train/validation split. Earliest validation minima select checkpoints; **FA-best is FA-H0**. Budget-boundary minima: FA-P0_43, FA-H0_43. Boundary selection does not establish convergence. Persistent fits pass the declared 41-versus-81-node checks. Feature clipping affects 13/5120 development rounds: Equal 0/1280, Mixed 0/1280, Proportional 0/1280, M1 13/1280.

| FA family | Mean validation NLL | Selected epochs (17/29/43) |
| :--- | :--- | :--- |
| FA-GRU | 2.5915 | 459/380/461 |
| FA-P0 | 2.5806 | 478/479/480 |
| FA-H0 | 2.5637 | 473/477/480 |

Calibration searches the complete declared coarse grid and 5×5 refinement per seed, using only training-group outcome distributions and common random numbers. It targets human fidelity, not higher surplus. Validation cost is calibrated minus base NLL over all 32 groups, including recorded M1 offers; seed-level tilts and costs remain in the artifacts.

| Calibrated line | Validation NLL cost (nats) |
| :--- | :--- |
| CL(FA-GRU) | 0.0135 |
| CL(FA-H0) | 0.0005 |
| CL(FA-P0) | -0.0021 |
| CL(H0) | 0.0028 |
| CL(P0) | 0.0002 |
| CL(Continued GRU) | 0.0129 |

## 6. D1: development headroom

**D1: no headroom.** IE averages absolute surplus errors equally over the three rules. Uncalibrated candidates use 32 development groups/rule; CL uses eight validation groups/rule. BC1 is rescored on exactly the same groups. Qualification requires matching both strict human surplus and survival orderings and IE≤BC1. Intervals resample complete human groups, holding forecasts fixed.

| Candidate | Groups/rule | IE [95% interval] | Matched BC1 IE [95% interval] | Qualifies |
| :--- | :--- | :--- | :--- | :--- |
| FA-GRU | 32 | 1.548 [1.016, 2.094] | 0.876 [0.582, 1.412] | no |
| FA-P0 | 32 | 0.743 [0.373, 1.326] | 0.876 [0.582, 1.412] | yes |
| FA-H0 | 32 | 0.871 [0.525, 1.422] | 0.876 [0.582, 1.412] | yes |
| CL(Continued GRU) | 8 | 1.141 [0.554, 2.428] | 1.231 [0.498, 2.416] | no |
| CL(P0) | 8 | 1.470 [0.682, 2.704] | 1.231 [0.498, 2.416] | no |
| CL(H0) | 8 | 1.327 [0.578, 2.613] | 1.231 [0.498, 2.416] | no |
| CL(FA-GRU) | 8 | 1.289 [0.568, 2.524] | 1.231 [0.498, 2.416] | no |
| CL(FA-P0) | 8 | 1.515 [0.649, 2.719] | 1.231 [0.498, 2.416] | no |
| CL(FA-H0) | 8 | 1.415 [0.629, 2.642] | 1.231 [0.498, 2.416] | no |

Survival-IE analogues with 95% group-bootstrap intervals are FA-H0 0.116 [0.054, 0.200], matched BC1 0.212 [0.139, 0.285]; CL(FA-H0) 0.114 [0.059, 0.316], matched BC1 0.191 [0.096, 0.358]. All candidates' survival errors and ordering checks are in the [D1 artifact](../results/task06/headroom_decision.json).

Validation survival ties Mixed and Proportional at 0.375; validation surplus orders Equal < Proportional < Mixed. Consequently all CL candidates are mechanically ineligible under the strict-order criterion, regardless of their IE. This is a constraint of the declared decision rule, not evidence of inaccurate calibrated behavior.

## 7. Experiment 2 transfer and D2

The [public forecast freeze](https://github.com/ReloadLightly/evopolis/commit/bf994ea7acc41592710f519802a75f2abe7e6b33) preceded any Experiment 2 behavioral parsing. The cohort has 40 Proportional, 40 Interpolating and 40 recorded M1 groups. Observed surplus is 6.563 under Proportional and 8.456 under Interpolating: I−P=1.894, human-group 95% interval [0.545, 3.295].

E1 is absolute error in I−P; E2 is Interpolating level error. The primary simulators are BC1, FA-H0 and CL(FA-H0). Task 03 GRU and constant are named secondary references. Paired differences subtract BC1's absolute error on the identical 2,000 bootstrap resamples, stratified by rule. Negative intervals excluding zero indicate better prediction; positive intervals excluding zero indicate worse prediction.

| Simulator | Predicted I−P | E1 \|error\| | E1 Δ\|error\| vs BC1: 95% CI; D2 | E2 \|error\| | E2 Δ\|error\| vs BC1: 95% CI; D2 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BC1 | 0.966 | 0.928 | [0.000, 0.000]; not distinguishable | 1.085 | [0.000, 0.000]; not distinguishable |
| FA-H0 | 0.130 | 1.764 | [-0.005, 0.836]; not distinguishable | 2.815 | [1.730, 1.730]; worse |
| CL(FA-H0) | 0.386 | 1.507 | [-0.262, 0.579]; not distinguishable | 2.019 | [0.934, 0.934]; worse |
| Task 03 GRU | 0.086 | 1.807 | [0.038, 0.879]; worse | 4.706 | [3.622, 3.622]; worse |
| Task 03 constant | 0.069 | 1.825 | [0.056, 0.897]; worse | 5.916 | [4.831, 4.831]; worse |

**D2:** FA-H0: E1 not distinguishable, E2 worse; CL(FA-H0): E1 not distinguishable, E2 worse. Zero-width E2 percentile intervals reflect cancellation of the shared human mean when both fixed forecasts underpredict it; simulation Monte Carlo uncertainty remains separate. Monte Carlo SEs, reported separately as effect/Interpolating-level SE, are BC1 0.192/0.144; FA-H0 0.111/0.083; CL(FA-H0) 0.112/0.086. These use within-checkpoint variation with fixed seed weights.

![Experiment 2 effects, levels and paired comparisons.](assets/task06-transfer-effects.png)

Secondary survival contrasts, Proportional level error, energy and pool20 errors, and per-seed results remain in the [transfer artifact](../results/task06/transfer.json).

All 57 learned checkpoints/variants receive teacher-forced evaluation on all 120 groups; M1 uses recorded current offers only. NLL averages nonforced choices within groups, then groups and fitted seeds equally; the all-120 column averages three equally sized rule strata. The table gives primary learned lines and named references; full per-seed NLL and outcome results are machine-readable.

| Learned simulator | Proportional NLL | Interpolating NLL | M1 NLL | All 120 groups |
| :--- | :--- | :--- | :--- | :--- |
| FA-H0 | 2.9145 | 2.9247 | 2.8898 | 2.9097 |
| CL(FA-H0) | 2.9115 | 2.9208 | 2.8855 | 2.9059 |
| Task 03 GRU | 2.9902 | 2.9950 | 3.0018 | 2.9957 |
| Task 03 constant | 3.2050 | 3.1737 | 3.1714 | 3.1834 |

Feature clipping on transfer histories affects 13/4800 rounds (Proportional 0/1600, Interpolating 0/1600, M1 13/1600); this public feature is model-independent.

Descriptive Exp2−Exp1 surplus shifts are 0.516 for Proportional and 0.017 for M1. These comparisons do not reuse people as paired observations.

## 8. Limitations

Experiment 2 changes participants and provides rule instructions absent in Experiment 1; the models have no instruction input. The within-cohort effect removes shared shifts only to first order. Published continuous Interpolating allocation has maximum offer replay residuals 7.2899 for BC1 and 7.39318 for human Experiment 2. The human Proportional residual is 0.00105384. Human transfer also contains an allocation-convention discrepancy. Scores evaluate the prespecified continuous implementation, without claiming exact institutional reproduction. BC1/BC2 terminal pools are equation-inferred because final next-pool fields are missing. BC1 used a different 537-game training collection, continuous actions and outcome-selected checkpoints; its Interpolating rule was also optimized against that simulator. The paper's component training counts conflict with its stated total; both are retained in the source audit. Participant identities cannot establish independence across launch groups. Exp 1 test outcomes were already opened, so later diagnostics are not pristine confirmation. Intervals spanning zero mean not distinguishable, not equivalence. These limits constrain causal and general claims beyond this game.

## 9. Recommendation

The declared rule identifies a transfer failure: the development criterion was met, but a primary transfer comparison is worse than BC1. Investigate instruction and cohort shift, and resolve the allocation-replay convention before future replication. Do not run evolutionary search.

## 10. Paper readiness

The defensible claim concerns measured institutional forecast validity and the gap between individual likelihood and collective fidelity. Baselines include recorded BC1, an input-blind constant, neural and conditional controls, and the simple FA/CL ingredients. A paper needs broader independent human evidence and instruction-matched replication; executable upstream models would improve comparability. An LLM-agent baseline is future work. This study does not establish motives, first discovery of the local-versus-collective discrepancy, or evolutionary improvement.

The [source audit](../results/task06/institutional_source_audit.json), [frozen manifest](../results/task06/manifest.json), [transfer results](../results/task06/transfer.json) and [phase resources](../results/task06/runtime.json) carry provenance. Earlier reports cover [behavioral fitting](behavioral-agents.md), [collective forecasts](collective-forecast-fidelity.md) and [conditional responses](conditional-responses.md).
