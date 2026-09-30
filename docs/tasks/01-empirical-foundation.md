# Task 01 — Reconstruct the commons and reproduce its empirical starting point

## Objective

Implement the published common-pool experiment and reproduce a substantive empirical result from its released trajectories. Finish with an executable analysis, a numerical comparison, and a figure that explains the social mechanism. This task establishes the environment and human evidence for EvoPolis; it does not train the later behavioral models or start evolutionary search.

Read `AGENTS.md` and `README.md` first. Work in this repository's WSL checkout. Preserve any existing user changes and verify that `origin` is `ReloadLightly/evopolis` before publishing.

## Primary sources

- Koster, Pîslar et al. (2025): https://doi.org/10.1038/s41467-025-58043-7
- Open full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC11929920/
- Official repository: https://github.com/google-deepmind/sustainable_behavior
- Reference revision: `4f1a99a9d150f9fa6bad1a0f70f11c0673d46763`
- Notebook: `notebooks/sustainable_behavior.ipynb`
- CSV endpoint used by that notebook: https://storage.googleapis.com/sustainable_behavior/sustainable_behavior.csv

Read the notebook and the relevant Methods sections directly. Use those sources to resolve implementation details; do not fill gaps with plausible inventions and call them faithful reproduction. Keep a concise record of any ambiguity that affects the result.

## Important release constraints

The repository contains data access and an analysis notebook, not the original training pipeline, pretrained weights, or an executable environment. The CSV is approximately 204 MiB and includes many synthetic behavioral-clone rollouts. Separate human and synthetic records explicitly. Do not train future human-behavior models on an accidentally mixed dataset.

Initial inspection found 167,782 data rows: 143,360 labeled as behavioral-clone rollouts and 24,422 as human trajectories. Recompute this inventory from the downloaded file, rather than trusting these values as an assertion. No labels for the initial 537-game training cohort were found. If future inspection changes this interpretation, document the evidence. New models fitted to the released human evaluation trajectories must be described as new models, not exact reconstructions of the original BC1 model.

## Work to complete

### 1. Make the analysis runnable on the actual machine

Inspect Python, available RAM, disk, and the repository. Choose a small, pinned Python environment appropriate to those constraints. Use a streaming download and chunked/selected-column parsing; avoid loading multiple copies of the full CSV into memory. Save the source revision, download URL, SHA-256, row counts, and schema in a small provenance manifest. Keep the raw download in ignored `data/raw/` and keep credentials out of files and logs.

Build only the package and command structure needed for this experiment. Prefer one documented reproduction command and a small set of inspectable Python modules. Do not create a dashboard, generic experiment framework, neural-training stack, or large CI matrix in this task.

If the GitHub CLI is already installed and authenticated, run `bash scripts/configure-github.sh` to apply the prepared About description and research topics. If it is unavailable, continue the scientific work and report that metadata step separately. No access-control changes are needed.

### 2. Inventory the empirical evidence

Identify each experimental condition, human versus synthetic origin, game/group identifiers, repeated games, available participant identifiers, round indexing, missing values, and units. Recover nested vectors carefully if values are serialized in CSV fields. Work out which observations a participant actually saw; logged hidden simulator state is not automatically an admissible feature.

Document what can support future training and which identities permit leakage-resistant splits. If repeat participants cannot be linked reliably, state that limitation and restrict later evaluation accordingly. Do not infer demographics: the paper states that none were collected.

### 3. Reconstruct the numerical world

Implement the published accounting with these reference values:

| Quantity | Initial protocol |
| :--- | :--- |
| Players | 4 |
| Initial pool and capacity | 200 |
| Returned-resource multiplier | 1.4 |
| Experiments 1–3 | 40 rounds; horizon unknown to participants |
| Allocation | Nonnegative, with total allocation at most the current pool |
| Contribution | Between zero and that player's allocation |
| Retained surplus | Allocation minus contribution |
| Next pool | Current pool minus allocations plus multiplied contributions, capped at 200 |

Check initial timing, numerical precision, rounding, zero-pool termination, and observation structure against the sources. Avoid giving agents the undisclosed horizon as an observation. Experiment 4 has a different continuation design and repeated games; do not silently apply the 40-round protocol to it.

Implement equal, proportional, and mixed allocation baselines. Initially allocations are equal. Thereafter the published mixture combines equal allocation with allocation proportional to the previous contributions: mixture weight 1 for equal, 0 for proportional, and 0.5 for mixed. The published interpolating baseline uses weight `(R / 200) ** 22`; include it when its conventions are resolved. Treat the no-previous-contributions denominator explicitly. If the sources do not specify the edge convention, identify a defensible implementation choice and label it as such.

Validate consequential invariants and replay recorded allocations/contributions against the reconstructed pool transitions. Report mismatches and tolerances with the affected cases. Never clip invalid data silently to manufacture a successful replay.

### 4. Reproduce Figure 2A's substantive comparison

Start with **mean player surplus versus Gini**, separating the recorded BC1 and human Experiment 1 panels. The human panel contains four conditions with 40 groups each, according to the released analysis. Recompute group counts and follow the notebook's definitions, aggregation, and exclusions before comparing results.

Produce a readable two-panel figure and a numerical table. Preserve the distinction between recorded outcomes of the upstream RL mechanism and mechanisms implemented or trained by EvoPolis. For statistical uncertainty, respect the group structure rather than treating every player-round as independent. Handle zero-surplus cases according to the source analysis, with any deviation made explicit.

Use labels understandable to a reader unfamiliar with internal run identifiers. Show units, conditions, group counts, and the data source. Use consistent colors that distinguish the allocation mechanisms and remain legible on a light figure background. Save editable/source plotting code plus SVG and PNG figures.

If an exact figure statistic cannot be reproduced, report the actual discrepancy and the most likely cause supported by inspection. Do not replace real records with generated data or claim agreement by visual resemblance alone.

### 5. Explain and publish the result

Update the main README with the empirical comparison, a clear explanation of the allocation dilemma, the real figure, and the current implementation state. Keep it focused on research; move detailed schema/provenance notes into a concise supporting document. Retain the existing visual identity and research roadmap, adjusting statuses only when justified.

Verify the documented reproduction command from the repository environment. Check numerical invariants, figure labels, relative links, and whether the plotted points match the reported table. Record actual runtime and peak memory where feasible.

Commit the completed task and push to the verified `origin` without force. The user has authorized completion and publication of this task. Continue through ordinary reversible implementation decisions; do not disable permission controls or create unrelated services. If authentication or network access prevents publication, keep the completed local work and report the precise blocked action.

## Completion evidence

- A working reproduction command with documented dependencies.
- A source/data manifest and a human-versus-synthetic inventory.
- Implemented resource dynamics and the elementary allocation baselines.
- A numerical replay comparison with explicit residuals and tolerances.
- The Figure 2A comparison, numerical table, and source plotting code.
- A research-focused README update identifying what was reproduced, what was newly implemented, and what remains uncertain.

Finish by summarizing the scientific result and identifying the next concrete task. Do not launch behavioral training or evolutionary search in this run. Those experiments follow once this foundation is visible and reviewable.
