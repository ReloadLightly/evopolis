# Scientific review after Task 04

Reviewed 2026-10-01 at commit `053ac5016bc23b608000c522e4d41ec5d7d4f3e2`. This is a review of completed evidence, not a new experiment. [Task 05](tasks/05-conditional-responses.md) is specified but has not run.

## Where the project stands

EvoPolis has an empirical reproduction, newly fitted probabilistic behavioral agents, an executable four-person resource game, and forecasts evaluated against actual human futures. Task 04 provides a substantive diagnostic result: **the longer GRU training procedure predicts individual choices better but predicts the declared collective endpoint worse**. This is useful evidence about this modeling procedure and dataset. It does not yet establish a psychological explanation, a general law about training, or a novel research contribution sufficient on its own for a paper.

The scientific object is a simulation of a specific multiplayer trust/reinvestment game with centrally allocated resources. Calling it a *social simulation* is defensible because interacting behavioral models generate outcomes under explicit rules and are compared with human groups. Calling it a validated model of society, human motives, institutional reform, or self-governance would exceed the evidence. No established social-science theory has yet been directly tested by EvoPolis. Evolutionary search and recursive self-improvement have not been implemented or demonstrated.

## What Task 04 found

Six existing neural fits continued from epoch 120 to 480, restoring their optimizer and random states. Architecture, data split, objective and selection rule stayed fixed; selection could retain the original best checkpoint. Each comparison therefore concerns two training-and-selection procedures, not necessarily the raw epoch-120 and epoch-480 weights.

| Outcome; lower is better | Original budget | Continued budget | Paired change and 95% group interval |
| :--- | ---: | ---: | :--- |
| Feedforward individual NLL; 32 groups | 2.7238 | 2.6890 | −0.0347 [−0.0528, −0.0170] |
| GRU individual NLL; 32 groups | 2.6944 | 2.6189 | −0.0755 [−0.1071, −0.0400] |
| **GRU primary collective energy; 21 groups** | **0.09660** | **0.12538** | **+0.02878 [+0.00747, +0.05352]** |
| Feedforward collective energy; secondary | 0.16271 | 0.15708 | −0.00563 [−0.01385, +0.00172] |

The collective endpoint is the normalized pool and total private surplus ten rounds after five observed rounds. The primary cohort has five Equal, eight Mixed and eight Proportional groups with a feasible contribution at the origin. Each fitted seed is scored separately, then scores are averaged within human groups and equally across mechanisms. The 150,912 branches are numerical forecast draws; they are not additional human observations.

The independent second forecast bank gives GRU deterioration of **+0.02944 [ +0.00663, +0.05570 ]**. All three GRU optimization seeds deteriorate in the primary collective score. Feedforward's small collective change is uncertain in the primary bank; a favorable secondary bank must not replace it.

Two additional comparisons were calculated during this review and are **post-hoc diagnostics**, not new primary results. On exactly the 21 primary groups, whole-game individual NLL improves by −0.08445 [−0.11002, −0.05697] for GRU and −0.04484 [−0.07162, −0.01974] for feedforward. This rules out differing cohort membership as the whole explanation for the individual/collective contrast; it does not match their time windows or prediction targets. At the continued budget, GRU minus feedforward individual NLL across all 32 groups is −0.07010 [−0.10584, −0.03288]. The README's earlier uncertain GRU advantage applies specifically to the 120-epoch budget. This architecture comparison alone does not isolate the contribution of memory.

See [the completed report](collective-forecast-fidelity.md) and [saved results](../results/task04/) for the original estimands, all secondary outcomes and numerical provenance. The [new statistical-review receipt](../results/reviews/task04-statistics-review.json) preserves this review's independent recomputations and explicitly labeled exploratory comparisons.

## Independent checks in this review

- Checked the continuation, prefix processing, resident state separation, sampling, resource transitions and proper-score code. No blocking implementation or future-information leak was found. Eight consequential existing tests passed; this was not a rerun of the entire experiment.
- Verified 102 protected scientific files against their recorded hashes. Existing fits, reference outputs and environment sources were preserved.
- Reconstructed energy scores and 80% interval coverage from 756 primary checkpoint/group banks in the main and second archives: 48,384 saved branches. Recomputed the stratified paired bootstrap effects above.
- Independently streamed the pinned raw CSV, selecting Experiment 1 test groups only. Directly summed all four recorded private surpluses over source rounds 5–14 and read the recorded next pool at round 14. All 1,008 primary target records across main, second and boundary-sensitivity banks agree exactly. The [source-check receipt](../results/reviews/task04-source-review.json) records the source hash and zero maximum disagreement. Reserved experiment outcomes were not parsed by this check.
- Checked that the two forecast-bank seed sets do not overlap. Shared random-stream seeds across model procedures are a computational coupling; differing draw consumption means they do not guarantee variance reduction after trajectories diverge.

## What the result does not explain

Individual log likelihood and multiround joint energy score evaluate different predictive targets. Their disagreement is not a mathematical contradiction. Possible explanations include response misspecification, inadequate history, persistent differences between people, unmodeled dependence, and changes in simulated states. Task 04 does not identify which explanation is responsible.

One-step GRU total-return likelihood improves, while its renewal Brier score changes by +0.00871 with an interval spanning zero. This is not decisive evidence that contemporaneous correlation causes the long-run error. The primary deterioration is concentrated in Mixed and Proportional groups; some forecasts from the initial round improve with training. Neither a universal training failure nor universal GRU inferiority follows.

The first-allocation numerical sensitivity changes the primary GRU effect by about 0.00019, much less than the observed 0.02878 deterioration. It weakens that particular boundary explanation. It does not reconstruct the unreleased upstream simulator or eliminate later integer-boundary and residual-floor effects. The endpoint's pool coordinate contributes more than surplus in these data; retain marginal scores alongside the joint score.

There are only 21 groups in the primary comparison. Group intervals condition on three fitted models per procedure and do not capture every source of training uncertainty. Participant overlap across group identifiers cannot be checked from the release. Experiment 1 has already been opened and described, so follow-up analyses are exploratory or diagnostic even when future analysis choices are fixed in advance. Experiments 2–3 remain reserved for EvoPolis model evaluation; their published design and aggregate findings are already public.

## A defensible theoretical next step

[Fischbacher, Gächter and Fehr (2001)](https://doi.org/10.1016/S0165-1765(01)00394-9) study conditional cooperation using elicited contribution schedules. [Fischbacher and Gächter (2010)](https://doi.org/10.1257/aer.100.1.541) connect elicited preferences, incentivized beliefs and repeated public-goods behavior. Their explanation emphasizes imperfect conditional cooperation; persistent heterogeneity is not interchangeable with that explanation.

The EvoPolis dataset does not contain those elicited schedules or beliefs. Its allocation rules also change investment incentives. Below the resource cap, with full allocation and other contributions fixed, one extra unit returned increases one's next offer by 0.35 under Equal, 0.875 under Mixed and 1.4 under Proportional. A contribution therefore need not express altruism or reciprocity. This game is not the standard equal-endowment public-goods experiment.

Task 05 will test a restricted, explicitly adapted prediction: **does a fitted response to peers' past feasible contributions improve predictions, and does modeling persistent individual differences alter that result?** A matched four-model comparison separates the peer-response term from persistent variation. Peer-response coefficients can be negative, so the proposed positive response can fail. All models condition on opportunity and own history; forced non-contributions are never treated as voluntary free riding.

This is a theory-informed predictive mechanism comparison, not identification of preferences or a replication of the original preference/belief experiments. Current offers themselves encode earlier contributions, especially under Proportional allocation. Even a baseline without the extra peer-history term therefore contains indirect social information. Model curves describe conditional associations and simulated mechanisms, not human causal treatment effects.

## Paper potential and compute

The repository is a credible foundation for a small empirical modeling paper, but the current result is not yet a demonstrated frontier advance. The strongest route is a carefully delimited contribution: an explicit behavioral mechanism, a reproducible explanation or failure to explain the individual/collective prediction gap, and a frozen procedure evaluated on a reserved cohort. Negative results can contribute when they discriminate between meaningful accounts and survive numerical and empirical checks. A larger branch count or an attractive viewer does not establish novelty.

Compute is not the immediate bottleneck. Task 04's six continuations took about 824 seconds and forecast generation about 411 seconds in the recorded environment, with peak resident memory around 301–307 MiB. These measurements cannot support a numerical comparison with Simile's or Google's total compute. Large organizations may train much larger systems, but more compute does not add independent human groups, elicit missing beliefs, or identify a social mechanism. The next investment should be theoretical clarity, matched comparisons and reserved-cohort evaluation. Evolutionary search remains a subsequent experiment, with fixed and random-search controls and its own evaluation budget.
