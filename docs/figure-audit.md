# Figure 2A reproduction audit

The reference is Koster, Pîslar et al. (2025), [*Deep reinforcement learning can promote sustainable human behaviour in a common-pool resource problem*](https://doi.org/10.1038/s41467-025-58043-7), Figure 2A, its Results and Methods, and the [official analysis notebook](https://github.com/google-deepmind/sustainable_behavior/blob/4f1a99a9d150f9fa6bad1a0f70f11c0673d46763/notebooks/sustainable_behavior.ipynb) at revision `4f1a99a9d150f9fa6bad1a0f70f11c0673d46763`. The notebook has no saved execution outputs. Numerical agreement must therefore be checked against its executable definitions and the paper's reported statistics, rather than against an upstream summary table.

## What a point measures

Notebook cell 4, `get_surplus_and_ginis`, selects an exact `mech_name_by_player` label and groups rows by `launch_id`. Within a game it computes each player's mean `player_reward_i` over recorded rounds. The point's vertical coordinate is the mean of those four player means: **resource units retained per player per round**, not total group earnings. Its horizontal coordinate is the uncorrected population Gini of the four player means:

$$
G = \frac{\sum_{i=1}^{4}\sum_{j=1}^{4}|\bar{s}_i-\bar{s}_j|}{2\cdot4^2\cdot\operatorname{mean}_i(\bar{s}_i)}.
$$

Because each player has the same number of rounds in these games, using each player's total instead gives the same Gini. It is **not** the mean of round-specific Ginis, or the inequality across all players in a condition. With four players this definition has a maximum of 0.75, not 1. The large condition point is the arithmetic mean of the game coordinates; games are weighted equally.

Cell 4's `get_exp_info(1)` specifies the Equal, Mixed, Proportional and RL Agent (M1) conditions with the `BC 1` and `Exp 1` suffixes. Cell 6, `plot_leaderboard`, produces the two panels. It makes no further exclusions and performs no clipping or zero replacement. All recorded rounds, including depleted-pool rounds with small residual surplus, contribute to the means. The RL and BC outcomes are **recorded upstream outcomes**; reproducing them does not train or implement either network.

## Cohorts and exclusions

The paper's Methods, “Participants” and “Human datasets,” say that dropout/incomplete games were excluded and only the first 40 eligible games per Experiment 1 condition were analyzed after data collection overshot its target. The full Experiment 1 collection included five conditions and 213 games; the fifth was the M1′ memory ablation. Figure 2A uses four conditions, hence 160 groups / 640 participants, and does not include that ablation. A reproduction of the figure should use the released figure labels, rather than reinterpret `first_x_games` or add filters that the notebook does not apply.

The original notebook divides by the mean surplus without a special case. A game with zero surplus for every player would consequently have an undefined Gini (`NaN`), and its condition's `np.mean` Gini would also be `NaN`. Assigning such a game Gini zero would be a change to the analysis. The independent check found **no zero-surplus games and no missing player rewards** in either selected panel, so this ambiguity does not change Figure 2A.

## Statistical comparison

The notebook applies `scipy.stats.ranksums` to game-level values, with average ranks for ties but no tie correction and no continuity correction. The default test is two-sided. It does not perform a multiple-comparison correction for Figure 2A. The sign follows the order of arguments; the paper generally reports the positive magnitude with a directional inequality.

The original figure contains individual game points and condition means, without error bars. The notebook's separate `get_mean_and_conf` helper computes `1.96 * std(ddof=0) / sqrt(n)`; its confidence width is discarded by `plot_leaderboard`. Any error bars added by EvoPolis must identify their estimator and keep the game as the replication unit. Repeated player-rounds are not independent replicates.

An independent standard-library calculation is available as:

```bash
python3 scripts/check_upstream.py data/raw/sustainable_behavior.csv
```

It streams the raw CSV, verifies exactly 40 observations and rounds 0–39 per selected game, and computes the notebook's means, Ginis and both cohorts' rank-sum comparisons without importing EvoPolis analysis code. This provides a check of aggregation and cohort selection, not a second source of experimental evidence. To check the main pipeline's saved output:

```bash
python3 scripts/check_upstream.py --compare-dir results/task01 > results/task01/independent_check.json
```

This comparison verified all 2,208 game coordinates, eight condition summaries and intervals, and 24 rank-sum tests. Across 4,504 numerical comparisons, the maximum absolute discrepancy was **1.78 × 10⁻¹⁵**, below the **10⁻¹²** tolerance. The [machine-readable check](../results/task01/independent_check.json) preserves the results.

## Independently recomputed results

The checker read 167,782 source rows. Each BC1 condition contains 512 games and 20,480 round records; each human Experiment 1 condition contains 40 groups and 1,600 round records. All selected games have exactly 40 rows and cover rounds 0–39. The eight coordinates below follow the notebook without additional exclusions.

| Condition | BC1 mean surplus | BC1 mean Gini | Human mean surplus | Human mean Gini |
| :--- | ---: | ---: | ---: | ---: |
| Equal | 3.879248260303 | 0.166636233450 | 2.201761387372 | 0.150449740224 |
| Mixed | 4.786777225636 | 0.096669437468 | 4.532721120187 | 0.086762365026 |
| Proportional | 6.405637533493 | 0.426183406215 | 6.046868735640 | 0.359595159806 |
| Recorded RL M1 | 8.478325654220 | 0.286078098605 | 8.718465267470 | 0.252998591493 |

All ten rank-sum statistics reported in the paper's Figure 2A discussion agree at the reported precision. The table reports magnitudes; the inequalities state the direction.

| Human comparison | Metric | Published \|z\| | Recomputed \|z\| | Two-sided p |
| :--- | :--- | ---: | ---: | ---: |
| Equal < Mixed | Surplus | 4.58 | 4.580312 | 0.000004643 |
| Equal < Proportional | Surplus | 5.28 | 5.282755 | 0.0000001273 |
| Proportional > Mixed | Gini | 5.89 | 5.888973 | 0.000000003886 |
| Proportional > Equal | Gini | 4.76 | 4.763140 | 0.000001906 |
| RL M1 > Proportional | Surplus | 3.25 | 3.252407 | 0.001144 |
| RL M1 > Mixed | Surplus | 4.38 | 4.378240 | 0.00001196 |
| RL M1 > Equal | Surplus | 6.31 | 6.312363 | 0.0000000002748 |
| RL M1 < Proportional | Gini | 2.78 | 2.780904 | 0.005421 |
| RL M1 > Mixed | Gini | 6.29 | 6.293118 | 0.0000000003112 |
| RL M1 > Equal | Gini | 4.19 | 4.185789 | 0.00002842 |

One prose claim does not agree literally with the released analysis: the paper describes M1 surplus as approximately “150% greater” than the proportional baseline. The observed means give **1.4418149 times as much**, or **44.18% greater**, surplus. The evidence supports an approximation to 150% *of* baseline, not 150% greater than baseline. The reproduced direction and rank-sum statistic agree; the README reports the measured 44.2% difference.
