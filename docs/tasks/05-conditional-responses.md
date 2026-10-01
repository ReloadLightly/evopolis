# Task 05 — Conditional responses and persistent individual differences

**Status: specified; not executed.** This task follows the [Task 04 scientific review](../task04-review.md). Execute it end to end when assigned: implement, fit, forecast, evaluate, inspect, document and publish the actual experiment. An implementation without fitted models and measured results is incomplete.

## Scientific question and scope

Does an explicit response to peers' previous cooperation improve predictions of human choices and collective resource dynamics after controlling for opportunity and own history? Does persistent individual variation change that comparison?

The theoretical starting points are [Fischbacher, Gächter and Fehr (2001)](https://doi.org/10.1016/S0165-1765(01)00394-9) and [Fischbacher and Gächter (2010)](https://doi.org/10.1257/aer.100.1.541). Read their actual methods and distinguish preferences, beliefs, conditional responses and heterogeneity. The 2010 explanation centers on imperfect conditional cooperation, not merely a mixture of different people. Our data contain neither elicited response schedules nor incentivized beliefs, and the resource game has unequal, endogenous opportunities and investment incentives. This task tests a restricted predictive implication using observed history, **not a direct replication or causal identification of conditional-cooperation preferences**.

Use only the existing human Experiment 1 split: 96 training, 32 validation and 32 test groups. This evidence has already been opened. Predeclaring this task makes its analysis disciplined, not independently confirmatory. Leave Experiments 2–3 reserved for a later frozen transfer study and Experiment 4 outside scope. Do not add institutional optimization, evolutionary search, LLM agents or a new dataset to this task.

Read `AGENTS.md`, this brief, `docs/task04-review.md`, `docs/protocol.md`, `docs/data-inventory.md`, the completed Task 03/04 reports, and their configurations, model, training, forecast and viewer implementations. Preserve all earlier scientific sources protected by their manifests, checkpoints, splits and results; add separate modules and artifacts. Preserve the Final Fantasy V-inspired town, resident inspection, comparisons and existing source modes. Verify the Git origin before publication.

## Four matched model families

| Family | Common opportunity/own-history response | Extra peer-history response | Persistent resident effect |
| :--- | :---: | :---: | :---: |
| P0 | Yes | No: beta = 0 | No: sigma = 0 |
| P1 | Yes | Fitted signed beta | No: sigma = 0 |
| H0 | Yes | No: beta = 0 | Fitted sigma |
| H1 | Yes | Fitted signed beta | Fitted sigma |

These are fitted stochastic models, not hand-assigned altruist/free-rider types. H1 minus H0 is the primary contrast. P1 minus P0 tests the peer term without persistent variation; H1 minus P1 and H0 minus P0 are secondary persistence contrasts. Report all four families and every optimization seed.

### Opportunity and history

For a resident at round `t`, let observed offer be `e`, legal maximum `n=floor(e)`, and return `c` an integer in `0..n`. Define the cooperation fraction as `q=c/e` only when `e>=1`. Do not substitute `c/n`: allocating 1.8 units and returning 1 is a different opportunity from allocating 1 and returning 1. If `e<1`, the only legal action is zero; it supplies no observation about willingness to contribute.

Maintain an own-history trace `a` and peer-history trace `b`, both initialized to 0.5. After all four round-t actions resolve, update own trace toward that resident's valid fraction, and peer trace toward the mean valid fraction of the other three residents. Exclude peers with `e<1`. If no eligible observation exists, carry the trace unchanged. Use `trace_next = (1-eta)*trace + eta*observed_fraction`, with a fitted `eta=sigmoid(raw_eta)` common to own and peer traces within a model. Keep validity and exposure indicators explicit. These traces summarize recorded or simulated actions; **b is not an observed or elicited belief**.

The common feature vector `z` consists exactly of:

- Intercept, own current offer / 200, the three other current offers / 200 sorted by magnitude, and current pool / 200.
- Own trace `a`, own previous valid fraction (zero if the immediately preceding own offer was ineligible), and that previous round's own-valid indicator.
- Previous-round eligible-peer count / 3, and separate indicators for whether any valid own and peer observations have occurred in the prefix.

At round zero, previous-round indicators/counts are zero and both traces are 0.5. Sorting peer offers enforces invariance to arbitrary ordering of the other residents. The peer trace `b` enters only the extra response term below, never the common head. Do not supply institution labels, participant/group identity, current contributions, future observations or an announced ending time. Residents did not know the 40-round horizon.

Current allocations can already convey other residents' earlier behavior. P0 and H0 therefore mean **no additional explicit peer-history term**, not no social information, selfish preferences, or no conditional cooperation of any kind. Endogenous opportunity and own/peer correlation limit interpretation even after these controls.

### Emission and persistent variation

Use a shared affine head from `z` to the five raw parameters of Task 03's zero/maximum-inflated beta-binomial distribution. Reuse its pure legal-support emission helpers without changing the protected source. Call its normalized mass `p_theta(c|z,e)`.

Tilt that mass and renormalize over the exact legal support:

```text
v = beta * (b - 0.5) + u
P(c | z, b, e, u) = p_theta(c | z, e) * exp(v*c/e)
                    / sum_{r=0..floor(e)} p_theta(r | z, e) * exp(v*r/e)
```

Compute in log space. When `n=0`, return the exact point mass at zero without dividing by zero. In P0/P1, `u=0`. In H0/H1, each resident has a persistent latent effect `u ~ Normal(0,sigma^2)`, independent across residents conditional on the fitted population model. Fit sigma with a positive parameterization; do not impose an arbitrary narrow upper cap or introduce named behavioral classes. Population mean zero anchors the common intercept. A latent effect persists throughout the resident's game.

Beta is signed and unconstrained, not forced positive. Holding `z`, `u` and opportunity fixed, the model implies:

```text
d E[C/e] / db = beta * Var(C/e)
```

Thus positive beta implements a positive conditional response, zero beta removes it, and negative beta contradicts its proposed direction within this family. This derivative describes the specified model under fixed conditioning, not a causal effect identified in human behavior. After marginalizing a posterior, hold its weights fixed when drawing these curves. Setting beta and sigma to zero must recover P0's distribution exactly.

All four families have the same common feature and emission structure. A persistent effect adds temporal dependence; it does not add a shared contemporaneous group shock or solve every possible source of interpersonal dependence.

## Fit actual models

Freeze `configs/task05.json` and a protocol/configuration hash before training and new outcome comparisons. Record feature order, history timing, initialization, likelihood, numerical integration, selection rule, seeds, metrics, cohorts and forecasts. A correctness repair must have a dated explanation and rerun the affected computation; never silently tune to test results.

Fit **four families × seeds 17, 29, 43 = twelve fits**, each for **480 epochs**. Use Adam with learning rate 0.001, weight decay 0.0001, gradient clipping 1.0 and an effective batch of eight complete groups. Use one CPU thread, no data workers, and one- or two-group microbatches according to available memory. Initialize raw eta at zero and beta at zero; initialize sigma at 0.5 in the persistent families. Initialize the common affine head with the same seeded scheme across matched families. Record it exactly. Three seeds are optimization starts, not three populations of human evidence.

For persistent families, maximize the **marginal likelihood of each resident's complete sequence**:

```text
L_i = integral Normal(u; 0, sigma^2)
      * product_t P(c_i,t | observed history strictly before choice t, u) du
```

Evaluate products as sums of logs. Sum resident log likelihoods within each interacting group, divide by that group's count of nonforced choices, then average group losses. Forced choices have probability one and contribute zero. Exclude only groups with no nonforced choices from this likelihood average and report any such count; do not remove them from applicable collective forecasts. Training groups have equal counts by mechanism. If a conditional subset lacks a mechanism, label its aggregation explicitly rather than silently redefining the estimand.

Do not independently integrate a fresh u at every round: that would erase the intended persistence. Validation/test likelihood is the equivalent **prequential** likelihood: start from the population prior, predict a choice using its preceding history, then update that resident's posterior using the choice. Future targets must never influence an earlier prediction. Update the four histories only after resolving the round's simultaneous decisions. No weight gradients or population refits are allowed on validation/test evidence.

Use Gauss–Hermite quadrature initially with 21 nodes for the one-dimensional normal integral; nodes are numerical integration points, not 21 human types. On selected training/validation models, compare 21, 41 and, if necessary, 81 or more nodes. Before test scoring, require changes below 0.001 nats per nonforced choice and 0.001 in predicted contribution fraction, examining group and resident predictions rather than only a cancelling grand mean. If insufficient, increase resolution or use adaptive one-dimensional integration and rerun affected fits/selection as a documented numerical correction. Check posterior normalization, representative prefix-posterior summaries and predictive tail probabilities as well. Large sigma or concentrated posteriors can defeat even 81 fixed nodes; node count alone is not evidence of accuracy. Do not select an integration resolution because it yields favorable test scores.

Select the earliest minimum validation group-balanced nonforced NLL from epochs 1–480 for each family/seed. Complete the assigned budget; retain all twelve selected models. Do not select by simulated cooperation, collective test score, a favorable seed, or an institutional ranking. Report whether validation curves are still improving and whether fitted scales approach numerical boundaries; the epoch budget does not establish convergence.

Save atomic last/best checkpoints with model, optimizer, random/shuffle state, full configuration, source/split/code hashes, selected epoch and measured learning curves. A paused job must resume without restarting completed fits. Global parameters stay frozen at evaluation; posterior inference and trace updates are distinct from online weight learning.

## Human evaluation and primary decision

Use the Task 04 primary cohort unchanged: **21 human groups, k=5, h=10, with at least one recorded offer >=1 at the origin**. Expected strata are five Equal, eight Mixed and eight Proportional. Verify complete group keys and eligibility from the observed boundary; do not select for later survival.

The primary comparison H1 minus H0 has two explicitly declared outcomes:

1. Individual prequential NLL on nonforced choices in source rounds **5–14** of these same groups. Warm each resident's posterior and histories using rounds 0–4 only. During scoring, predict each subsequent round before updating from its recorded actions. Average choices within group, optimization seeds within group, then groups within mechanism and mechanisms equally. Report any zero-choice group/window explicitly; if present, it cannot be called an identical individual-score cohort.
2. The existing joint endpoint energy score on pool after source round 14 / 200 and all four residents' retained resources in source rounds 5–14 / 2000. Generate futures freely from the round-5 observed boundary; do not teacher-force future allocations or contributions.

Use 2,000 mechanism-stratified paired group bootstrap replicates with PCG64 seed **20261022**. The same resampled group keys attach every family, seed and outcome. Score each fitted seed's forecast separately before averaging within group. Human groups, not rounds, fitted seeds, quadrature nodes or forecast branches, are the empirical replication units. Intervals are conditional on these fits; do not advertise them as covering all parameter uncertainty.

Describe **joint predictive improvement** only if both H1-minus-H0 primary 95% intervals lie below zero. This is an intersection criterion for that joint claim, not permission to choose whichever endpoint succeeds. A conditional-cooperation interpretation additionally requires a positive fitted peer response; report beta for every seed rather than treating seed variation as a human-population significance test. Improvement with a negative response supports a different predictive association. A null result rejects neither the original elicited-preference findings nor every possible conditional-cooperation model.

Report the three secondary factorial contrasts, full-game individual NLL on the original 32 test groups, and Task 04's collective secondary horizons and marginal scores. Keep the continued and original Task 04 neural procedures as frozen reference predictors. New models reconstruct fractions from additional legal history; old feedforward models receive the original nine inputs. Their comparison is informative but **not an information-matched test of architecture**. The four new families provide the matched comparisons. Reuse verified existing archives where their metric/cohort definitions agree; recompute comparable individual scores from frozen weights when necessary.

Retain exact proper-score definitions from Task 04: off-diagonal energy/CRPS estimator, no clipping, physically fixed normalizations, separate pool/surplus marginals, and 80% predictive intervals. Report the full temporally ordered path separately; a joint endpoint score is not a joint path score.

## Conditional forecasts and sampling budget

At each origin, reconstruct a resident's latent posterior only from that resident's observed prefix choices, using legally observable histories in its likelihood. Do not fit a resident effect using its complete held-out trajectory. Start the first generated action at the recorded current pool and allocations; retain the observed unallocated residue. Subsequent allocations use the existing executable Equal/Mixed/Proportional rule and generated history.

For H0/H1, draw **one latent u per resident per branch from the prefix posterior and keep it fixed through that branch**. Sample legal contributions conditional on that u. Never redraw a personality each round or infer it using the actual human future. For P0/P1 use u=0. All four actions use the same preceding-round state and resolve simultaneously before history updates. Separate resident posteriors, traces and random draws. A single branch's displayed predictions must say whether they condition on its sampled resident effect; ensemble forecasts marginalize the latent uncertainty.

| Forecast bank | Calculation | New branches |
| :--- | :--- | ---: |
| Principal; up to 20 rounds, horizons 1/5/10/20 | 12 checkpoints × 24 baseline groups × origins 0/5/10/20 × 64 | 73,728 |
| Independent stability bank; primary k=5, h=10 only | 12 checkpoints × 21 eligible groups × 64 | 16,128 |
| **Total** | Twelve newly fitted checkpoints | **89,856** |

The primary result uses only the principal bank. Report the second-bank contrast separately; do not pool banks or replace an unfavorable result. Use seed namespace **20261023**, keyed by family, training seed, full human group key, origin, bank, branch, resident and draw purpose. Separate latent draws from action draws. If using common random numbers within comparisons, use fixed inverse-CDF uniforms with explicit indexing and report the coupling; do not assume shared initial seeds guarantee lasting alignment or variance reduction.

Preserve exact legal support and the existing resource equation, capacity, multiplier and exact-zero absorbing state. At depletion, zero-pad flow variables through the full horizon and hold cumulative surplus constant. Keep recorded human residuals unmodified. Do not implement live M1: its allocation policy is unavailable. M1 can appear only in the recorded-choice likelihood evaluation. Interpolating has no Experiment 1 human counterpart and is outside this forecast comparison.

## Mechanism diagnostics and interpretation

- Plot opportunity-conditioned response curves against peer trace b, with fixed own history, current resources and posterior weights. Mark the empirical support of the covariate combinations. Curves outside that support illustrate the model, not observed behavior.
- Report training/validation variation in b remaining after a fixed linear projection on the common controls, plus coefficient stability across optimization seeds. This descriptive redundancy check is not a causal adjustment or another fitted candidate. Little independent peer-history variation limits what a null incremental comparison can distinguish.
- Plot expected contribution fraction minus b only where matching is feasible: `b <= floor(e)/e`, with `e>=1`. Separate opportunity constraints from behavioral under-matching. Call this **imperfect history matching**, not an elicited preference or biased belief. Beta below one is not a valid under-matching criterion because beta is a log-tilt coefficient.
- Report fitted beta, eta and sigma for each seed and the shrinkage/uncertainty of illustrative prefix posteriors. Do not assign psychological types or demographic meanings to latent effects. Endogenous resources and confounded histories can produce predictive resident variation without identifying stable preferences.
- Compute one-step total-return likelihood and renewal calibration under recorded states from the posterior-predictive resident PMFs. Exact convolution is valid under the model's conditional resident-independence assumption; it does not empirically establish that assumption.
- Compare the two-by-two families on matched individual and collective outcomes, with paired differences and intervals. Show observed-versus-forecast pool and surplus trajectories, including failures, not only attractive surviving communities.

None of these diagnostics licenses a causal claim about human motives. Explain whether the tested mechanism improves both predictive targets, one, or neither; whether its response direction agrees with the hypothesis; and what remains unidentified. Do not cite Ostrom, inequity aversion, trust or reciprocity as tested theories without implementing their distinct empirical implications and checking that this dataset can identify them.

## Execution, artifacts and completion

Measure available RAM, swap and disk before importing Torch. Use the locked CPU environment and sequential numerical jobs. Timing the first actual training epochs and validation forecasts may guide microbatch size; no separate pilot or manual approval gate is required. Keep scientific settings fixed, save progress, and report measured elapsed time, peak RSS, completed fits and forecast batches. If a real correctness, numerical or access blocker prevents completion, preserve the work and state it; do not replace unfinished fits with illustrative results.

Produce the following as one completed task:

- A concise theory-to-measurement table distinguishing source theory, this operationalization, predicted direction, observable evidence and identification limits.
- Frozen configuration, twelve selected and resumable fits, actual learning curves, parameter tables, numerical-integration checks and immutable provenance.
- Complete forecast archives or a lossless reproducible representation with all seeds, source identities and model hashes; verify a selected branch by fresh regeneration. Save group-level outcomes, bootstrap summaries and both bank results.
- A substantive report in `docs/conditional-responses.md`, measured figures in the existing style, and an updated README led by the scientific result, including null or negative findings. Use sensible precision and distinguish human sample size from computational sampling.
- Extend the existing forecast viewer to select the new mechanism families and inspect their saved distributions and histories. Preserve existing modes and Final Fantasy V-inspired graphics. Do not build a second dashboard or a new infrastructure platform.
- Focused verification of consequential behavior: legal normalized PMFs at n=0/1 and ordinary offers; beta/sigma-zero reductions; signed-response derivative; quadrature convergence; past-only posterior inference under future-target poisoning; simultaneous history updates; latent persistence; group weighting; resumable fitting; and actual browser inspection of new model selection plus existing replay/forecast flows. Reuse existing checks where sufficient.

Before publication, verify the protected earlier artifacts, inspect the diff, and update task status from measured completion evidence. Commit and push to `ReloadLightly/evopolis` using ordinary access controls and no force push. Do not stop after writing code, after one model, or after a scaffold. No successful-result criterion is required for publication: a correctly completed negative experiment is a result.

The following task should freeze a selected procedure and evaluate transfer on a reserved cohort, with its own protocol before outcome access. Different instructions and allocation conditions make that a joint distribution shift, not automatically a clean causal institution intervention. Evolutionary improvement then remains a separate comparison against fixed and random-search baselines, with a protected evaluation strategy.
