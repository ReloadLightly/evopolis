# Released evidence and observation boundaries

The reproduction streams the [official released CSV](https://storage.googleapis.com/sustainable_behavior/sustainable_behavior.csv). Its provenance label is `mech_name_by_player`: suffixes `BC 1` and `BC 2` identify recorded synthetic behavioral-clone rollouts; `Exp 1`–`Exp 4` identify human experimental records. Unknown suffixes cause an error rather than being admitted as human evidence. The [machine-readable inventory](../results/task01/inventory.json) contains every column name, missing-value counts separately by origin, vector shape counts, condition sizes, identifier cardinalities, and game-length distributions. The provenance manifest records source hashes and the notebook revision.

The measured release contains **167,782 rows and 113 columns**: **143,360 synthetic rows** in **3,584 recorded games**, and **24,422 human rows** in **680 game episodes from 520 groups**. Each row records one round for all four players. These are not 167,782 independent observations. The four participant positions in each human group amount to 2,080 positions; the release does not establish that they are 2,080 distinct people across groups. The paper's 4,952-participant total describes a larger collection than this released evaluation subset.

| Origin / experiment | Condition | Games | Rows | Recorded rounds per game |
| :--- | :--- | ---: | ---: | ---: |
| Synthetic BC1 | Equal | 512 | 20,480 | 40 |
| Synthetic BC1 | Mixed | 512 | 20,480 | 40 |
| Synthetic BC1 | Proportional | 512 | 20,480 | 40 |
| Synthetic BC1 | Interpolating | 512 | 20,480 | 40 |
| Synthetic BC1 | RL M1 | 512 | 20,480 | 40 |
| Synthetic BC2 | Interpolating | 512 | 20,480 | 40 |
| Synthetic BC2 | RL M2 | 512 | 20,480 | 40 |
| Human Exp. 1 | Equal | 40 | 1,600 | 40 |
| Human Exp. 1 | Mixed | 40 | 1,600 | 40 |
| Human Exp. 1 | Proportional | 40 | 1,600 | 40 |
| Human Exp. 1 | RL M1 | 40 | 1,600 | 40 |
| Human Exp. 2 | Proportional | 40 | 1,600 | 40 |
| Human Exp. 2 | Interpolating | 40 | 1,600 | 40 |
| Human Exp. 2 | RL M1 | 40 | 1,600 | 40 |
| Human Exp. 3 | Interpolating | 80 | 3,200 | 40 |
| Human Exp. 3 | RL M2 | 80 | 3,200 | 40 |
| Human Exp. 4 | Proportional, game 1 | 40 | 1,211 | 25–49 |
| Human Exp. 4 | Proportional, game 2 | 40 | 1,091 | 25–35 |
| Human Exp. 4 | Proportional, game 3 | 40 | 1,130 | 25–41 |
| Human Exp. 4 | RL M2, game 1 | 40 | 1,124 | 25–46 |
| Human Exp. 4 | RL M2, game 2 | 40 | 1,135 | 25–47 |
| Human Exp. 4 | RL M2, game 3 | 40 | 1,131 | 25–39 |

## Identifiers and repeat games

- `launch_id` identifies a human interacting group: 520 distinct values, with no value shared between experimental cohorts. Experiments 1–3 each have one episode per group. All 80 Experiment 4 groups have `episode_id` 0, 1, and 2, corresponding to the three successive games played by the same group. These games must stay together in future splits and uncertainty estimates.
- `episode_id` is 0 for human Experiments 1–3. For synthetic records its values 0–511 recur across mechanisms. Synthetic `launch_id` also repeats between the BC1 and BC2 interpolating conditions. A safe game key is `(mech_name_by_player, launch_id, episode_id)`; neither identifier alone is globally unique.
- `launch_id_gamewise` has 240 nonmissing distinct values, all in Experiment 4. `launch_id_str` and `episode_id_str` are alternate representations for that cohort. They add no cross-group participant identity.
- `block_id` is 1.0 in every human row and absent for synthetic records. `set` is `game4` in Experiment 4 and absent elsewhere. Neither is a train/test designation. The unnamed index columns are export artifacts.
- `round_id` starts at 0 in every game and advances without gaps; there are no duplicate game-round keys. Forty-round games run 0–39. The Experiment 4 maximum is 48, giving 49 recorded rounds. Stored end-of-game rows remain in the source aggregation, including depleted-pool rounds.

The four player suffixes (`_0`–`_3`) are positions within the group, not released account or participant identifiers. The paper describes the same participants remaining together in Experiment 4, and the launch/episode structure supports that linkage. Repeated participation under a different `launch_id` cannot be checked. Future modeling must at minimum split complete interacting groups, keep all three Experiment 4 games together, and explicitly qualify cross-group participant independence. A verified participant-independent evaluation requires additional linkage or evidence from the source owners; identifiers must not be reconstructed from behavioral similarity.

## Schema, missingness, and units

Core action, reward, offer, pool, condition, and game/round fields are present for every row. `player_action_i` is the contribution; `offer_i` is the allocation; `player_reward_i` is retained surplus. Pool and allocations are in the game's flower units; one flower corresponds to one coin of surplus. Relative actions, Gini coefficients, and baseline mixing weights are dimensionless. Figure 2A's reported mean surplus is coins **per player per round**, following the notebook's averaging over all stored rounds. Monetary bonuses require the paper's payment rule and are not the plotted outcome.

The CSV mixes whitespace-separated arrays such as `[1. 2. 3. 4.]`, comma-separated lists, and nested wrappers such as `[[1], [2], [3], [4]]`. The inventory explicitly parses these numeric fields without evaluating Python expressions. Every nonempty vector has four finite entries. Flattening checks the number of player entries; it does not determine their ordering. The player-observation vectors use focal-player order, while mechanism vectors and scalar player columns use group order.

The complete missingness table is in the inventory JSON. Missing means an empty/whitespace-only cell or a scalar `nan`, `null`, or `none`, case insensitive; nonfinite scalar values and vector entries are counted separately. Material patterns are:

- Human-only fields, including completion flags, recorded next-pool state, environment parameters, and questionnaire responses, are absent from synthetic records. A missing synthetic next-pool field is not a missing whole round; the following round's observed pool can still support replay.
- The four relative-contribution columns have 38,131 missing human player-round cells, all at offers below 1. These rows are not evidence of missing underlying human actions; the scalar action columns are complete. Derived reciprocity ranks and proportionality also contain undefined cases. Do not impute their missing values as behavioral zeroes.
- All human rows are marked `completed=True`, `first_x_games=True`, `four_usable_players=True`, and `four_players_present_or_no_pool=True`. There are 196 rows with `four_players_present=False`, all marked `game_has_no_pool=True`. Filtering them would remove depleted-pool rounds and alter the estimand. This release has already been selected for the published analysis; the original dropout and overshoot records cannot be recovered from these flags.
- Questionnaire fields are present in the 11,200 Experiment 2–3 rows and missing elsewhere. Per-round duplication of a response does not create independent questionnaire observations.

No participant demographics appear in the schema. The [paper's Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC11929920/) explicitly state that none were collected; origin, presence flags, contribution patterns, and group IDs must not be treated as demographics.

## Evidence admissible for later behavioral models

The paper's behavioral-clone input comprises current offers to all four players, previous-round contributions from all four players, and current pool size: nine numbers normalized by 200, with the focal player first. The human interface also showed the preceding round's outcomes and cumulative surplus after decisions were complete. Those historical quantities may be reconstructed from the player's past information. Logged current-round contributions, current rewards, the next pool, post-game ratings, completion flags, mechanism file paths, and future game length are not pre-decision behavioral inputs. In particular, current cumulative-reward fields include the current outcome; using them without a lag leaks the target. Experiment 4 cumulative counters can carry across game boundaries, while the resource pool resets.

The horizon was undisclosed in Experiments 1–3. A participant can remember elapsed rounds, but must not receive the scheduled termination round or a remaining-round countdown. Experiment 4 has a disclosed minimum of 25 rounds followed by stochastic continuation and known game restarts. Experiments 2–4 also supplied mechanism instructions, so pooling conditions must account for that difference in information.

No released condition identifies the initial **537-game human training cohort**, random-allocation training games, pilot mechanism M0, or the memoryless M1′ evaluation condition described in Methods. The CSV therefore cannot establish an exact reconstruction of the upstream BC1 training process. Human-only fitting on its observed evaluation cohorts would produce **new behavioral models**. Recorded BC1/BC2 outcomes may be analyzed as upstream simulation results but must not be mixed into the human fitting target. Before any fitting or model search, specify group-level development and final-evaluation partitions; keep final human outcomes out of selection and report the unresolved participant-linkage limitation.
