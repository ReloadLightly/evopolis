# Task 02 — Make the commons observable

## Outcome

Build a working local **16-bit community viewer** in which the user can watch four residents receive resources, return contributions, and retain surplus. Connect the scene to the existing empirical evidence and numerical environment. The finished application must make it possible to inspect how a community sustains its commons, loses resources, or excludes a participant over time.

This is Stage 02 of the agreed research program. Task 01 has already reproduced the Figure 2A comparison and documented the numerical limits of the reconstructed environment. Build on that work. The deliverable is an interactive application with actual data, a small transparent simulation mode, and a real screenshot—not another design document or a static mockup.

## Read before implementing

- `AGENTS.md` and `README.md`.
- `docs/VISUAL_STYLE.md` and `docs/assets/evopolis-theme.css`.
- `docs/protocol.md`, `docs/data-inventory.md`, and `docs/figure-audit.md`.
- `evopolis/world.py`, `evopolis/replay.py`, `evopolis/analysis.py`, and the source/checksum helpers.
- `results/task01/figure2a_groups.csv`, `figure2a_summary.csv`, `manifest.json`, and `replay.json`.

Preserve the current empirical results and existing user changes. Confirm the checkout's `origin` is `ReloadLightly/evopolis` before pushing.

## The experience to build

The main view is an original top-down or three-quarter pixel town: four distinguishable resident sprites, their homes, a shared resource store or commons, and clear paths for animated resource transfers. Use the established midnight/indigo palette, parchment text, square menu frames, and mint/cyan/amber/coral accents. Keep the scene compact enough to sit beside the numerical panels on a laptop screen.

Residents are **A–D, positions within a recorded group**. Their sprites are visual identifiers. Do not invent participant demographics, names, motives, dialogue, or private beliefs. Selecting a resident opens a dialogue-style information panel containing the actual allocations, contributions, retained surplus, and history available for that record.

One round has visible phases: **allocate → return / retain → replenish**. These phases explain the accounting; the four contributions were simultaneous, so sequential visual animation must not imply that later players observed earlier current-round decisions. Sprite movement is presentation, not an added spatial mechanism.

Implement these controls and displays:

| Feature | Required behavior |
| :--- | :--- |
| Playback | Play, pause, previous/next round, restart, speed control, and a round scrubber. Seeking restores the exact state without accumulating arithmetic drift. |
| Community state | Pool before the round, four allocations, four returns, retained resources, and observed pool after the round where available. |
| Resident inspection | Current values and the resident's history; distinguish round surplus from cumulative surplus. |
| Timeline | Pool and cumulative retained surplus, with a visible cursor synchronized to the selected round. |
| Participation | Number of allocations at least one unit, labeled with that definition. Show offers below one distinctly. |
| Numerical inspection | Exact source values accessible in a table or tooltip; rounded display values never drive state updates. |
| Provenance | A persistent readable label identifying recorded human data, recorded upstream model outcomes, or a new scripted simulation. |
| Comparison | Two communities with synchronized playback, independently selected episodes or matched scripted settings. |

The round count is a researcher playback control, not an observation supplied to a behavioral agent. Use resource/flower units and retained surplus; do not label these values actual cash payments or convert them to invented percentages. One offer below one unit is an observation about current opportunity, not proof of permanent exclusion. If displaying running Gini, label its current-game cumulative basis and distinguish it from the completed-game Figure 2A statistic; an all-zero denominator is undefined, and the four-player definition has maximum 0.75.

## Mode 1 — Recorded communities

The first version covers the **Figure 2A cohorts**: human Experiment 1 and recorded BC1, each under Equal, Mixed, Proportional, and recorded RL M1. That is 160 human episodes and 2,048 recorded model episodes, each with 40 rounds. Keep all eligible episodes available through selectors. Experiment 4's repeated variable-length games and other cohorts can be added later; do not silently treat them as this protocol.

Use the original source records for round-level replay. The saved Figure 2A CSV contains game-level summaries and cannot reconstruct individual rounds. Reuse the pinned data checksum. Stream the cached 204 MiB source once into a compact indexed cache under ignored `data/cache/`; avoid parsing the full source on every interaction or loading it wholesale into the browser. If the source is missing, obtain it through the existing verified download path. Ordinary playback must work offline after preparation.

The safe episode key is `(mech_name_by_player, launch_id, episode_id)`. Validate the selected cohort's expected episode and round counts and map group-order values without confusing them with focal-player-rotated observation vectors. Use recorded `offer_i`, `player_action_i`, and `player_reward_i` as the displayed allocation, contribution, and surplus.

**Preserve recorded accounting:** `mechanism_observation.pool` is the pool **before** that round's allocation. Use the recorded next-pool field when available; otherwise use the following round's pre-allocation pool within the same episode. The last recorded BC1 round has no observed following pool: show that value as unavailable. A separately labeled equation-based estimate is optional, but it must not become a recorded observation.

The released data retain a pool near 0.01 in many depleted rounds and include small accounting residuals. Display the observed trajectory as observed. Do not rerun it through `WorldState` to replace recorded values, cut off all rounds after pool falls below one, or clamp away the residuals. A compact optional inspection row may show **recorded next pool / equation estimate / residual**, drawing on the existing audit.

Default to the human Experiment 1 Equal condition and choose the episode nearest that condition's median surplus, with ties broken by stable episode key. State the selection rule. Give the user a way to choose episodes by their surplus/Gini outcomes, ideally through a small interactive scatter view using the existing game summaries. Make highlighted episode selection reproducible; do not present handpicked successful runs as representative evidence.

In recorded comparison mode, label the panes as **different observed groups**. A group under one mechanism is not the same people under another mechanism, and these paired pictures do not establish an individual counterfactual. Keep cohort and mechanism visible in both panes.

## Mode 2 — Scripted sandbox

Provide a lightweight interactive use of the already implemented Equal, Proportional, Mixed, and Interpolating allocation rules. Reuse `allocate` and `step`/`WorldState` from `evopolis/world.py`; do not maintain a divergent JavaScript copy of the environment equation.

Start with a transparent resident rule: each resident has a user-adjustable return fraction `q_i` between 0 and 1, and returns `floor(q_i * offer_i)` resource units. Use the human integer contribution grid. Document this as a **fixed-fraction scripted policy**, with no fitting or learning. Do not describe the behavior as a human prediction. Show the four fractions as editable settings, and use the same settings in both comparison panes unless the user explicitly changes them.

Parameter or allocation-rule edits start a new run. They must not edit a recorded episode in place or splice a synthetic future onto a human record without explanation. Label sandbox comparisons as outcomes under the stated scripted assumptions. There is no new RL M1 allocator: the upstream RL mechanism remains available only through recorded playback.

Keep this sandbox small. It is a way to inspect the implemented mechanism, including how integer return choices and zero contributions affect the pool, before training behavioral agents in Stage 03. Show exact-zero termination and the 40-round limit as implemented. Do not introduce the source's undocumented 0.01 floor into the reconstructed simulation.

## Implementation and usability

Use the smallest maintainable stack that fits the current Python repository and WSL machine. A small Python local server with static HTML/CSS/JavaScript and Canvas/SVG is sufficient; a frontend framework is optional only if it materially simplifies the implementation. No model API calls, GPU, external database service, or hosted deployment are needed.

Provide one documented command from the repository root, for example:

```bash
uv run python -m evopolis.viewer --port 8765
```

Implement that command or document an equally simple actual command. Bind locally to `127.0.0.1`, print the URL, serve only the intended app resources, and support clean shutdown. If the requested port is occupied, report an actionable alternative. Show preparation progress when creating the replay cache and reuse it on subsequent launches, with source/schema provenance sufficient to invalidate a stale cache.

Use original small pixel sprites and tile assets, or repo-native SVG/Canvas art that follows the style guide. Scale pixel art cleanly. Keep scientific text, tables, and charts readable; do not apply pixelation filters to data. The shared CSS is a starting point, not a mandate to force every chart into bitmap lettering. Include keyboard-operable controls and a reduced-motion option that preserves all state transitions without requiring animation.

## Verify the actual experience

Open the real local application and exercise the complete flow: select a human game, play and pause, scrub backwards, inspect a resident, switch to recorded BC1, compare two episodes, then run and reset the scripted sandbox under at least two allocation rules. Check the browser console and visible state, not just whether the server starts.

Use targeted checks for risks that can change interpretation: full episode-key collisions, group-order versus focal-player ordering, before/after-round timing, contribution simultaneity, cumulative totals after seeking, depleted recorded rounds, missing final BC1 next-pool, and sandbox parameter reset. Compare replay values for selected episodes with source rows and compare aggregate episode surplus/Gini with the saved summaries. Use the existing tolerances explicitly; do not silently repair observations. Preserve the prior numerical artifacts and source hashes.

Take a **real screenshot** of the working pixel-art interface for the README; a short replay GIF or video is optional if easy to produce faithfully. Do not use generated concept art as a screenshot. Check the initial screen at a practical laptop viewport and ensure panels, controls, units, and provenance labels fit.

## Publish and hand back

Update the README with the actual launch command, screenshot, implemented capabilities, and a short explanation of what a reader can inspect. Add a concise viewer guide if details would crowd the research narrative. Keep the existing measured findings intact. Mark Stage 02 complete only when the usable replay and sandbox are verified.

Commit and push the finished work to the verified `origin` without force. The user has authorized completion and publication of this task. Continue through ordinary reversible implementation choices; keep existing permission controls in place. If access prevents an action, preserve the completed work and report the precise blocker.

Finish with the launch command, what works, one concrete observation from a recorded community with its episode identity, and any material limitations. Recommend Stage 03—the first trained behavioral agents—as the next research step. Do not launch training or evolutionary search in this task.

## Completion record

Task 02 is complete. The clean checkout was safely fast-forwarded from `6f30e81` to `cc9e0fa` before implementation; `origin` was verified as `https://github.com/ReloadLightly/evopolis.git`. No local user edits were present. The implementation uses the existing Python environment, a local standard-library HTTP server, compressed SQLite episode storage, and static HTML/CSS/JavaScript with original Canvas pixel art. No frontend runtime dependency or service was added.

Verified launch:

```bash
bash scripts/viewer.sh --port 8765
```

Open `http://127.0.0.1:8765`; stop with Ctrl+C. The server binds only to loopback and serves a fixed application-resource allowlist. The app implements the two recorded cohorts, all eight cohort/condition cells, every eligible episode, three explanatory phases, resident inspection, exact values, cumulative history, timeline cursor, keyboard controls, reduced motion, synchronized comparisons, reproducible view links and the fixed-fraction sandbox. The [guide](../viewer.md) records source semantics and controls.

Initial WSL profiling found 831 MiB available out of 3.7 GiB total RAM and fully occupied 1 GiB swap. A streaming cold preparation of the 204 MiB CSV took **46.19 s / 23.6 MiB peak process RSS** and produced a **23,556,096-byte** indexed cache. A verified cached preparation took **0.39 s / 30,592 KiB peak process RSS**, including interpreter/wrapper startup. These exclude browser memory. All **2,208 episodes / 88,320 rounds** are indexed; every completed-game surplus and Gini agrees with the saved Task 01 summary within **2.22e-16** (tolerance `1e-9`). Repeated playback loads selected episodes only. Source/schema/summary provenance controls cache reuse, and offline cache use was tested without the raw file.

The running app was opened with the `vercel:agent-browser` workflow in Chromium at **1366×900**. The full browser path—selectors → local API → pinned records → values, histories and scene—was exercised. Actual play/pause advanced to round 2 and held there; backward seeking restored cumulative totals. Checks covered resident C, depleted human round 12, final BC1 N/A, 512 available BC1 episodes per mechanism, independent synchronized selections, keyboard scatter selection, sandbox rule edits and resets, decimal fraction 0.58, exact-zero termination, shorter-run holding and reduced motion. The comparison layout also fit without horizontal overflow, and the 390-pixel narrow layout had no page overflow. Browser console and error logs were empty. [Machine-readable browser evidence](../../results/task02/browser-verification.json) and [provenance/verification](../../results/task02/verification.json) preserve the results.

The full Python suite passed **21 tests**. Added tests target episode-key collisions, group-order scalar mapping, exact source strings, pool timing, cumulative seeking, preserved depleted rows, missing final BC1 pools, source checksum rejection, offline reuse, exact median ties and scripted numerical boundaries. A diagnostic sweep of 4,000 seeded scripts across the four mechanisms also confirmed feasible allocations, nonnegative pools, integer returns and horizon bounds after the allocation fix. Node syntax checks passed for the two application modules.

Two consequential simulation boundary fixes were required. Independently computed baseline shares could exceed the pool by an ulp and yield a tiny negative next pool; balancing the largest computed share now preserves feasibility. Decimal `q=0.58` times offer 50 must floor to 29, so scripted contributions evaluate the decimal fraction exactly against the represented allocation before flooring. Neither fix alters recorded values or introduces the source's 0.01 pool floor. All **13 prior empirical artifacts**, including Task 01 tables, audit outputs and figure files, retain their original SHA-256 values.

The [README screenshot](../assets/community-viewer.png) is the real running application at round 1 for `('Equal Baseline Exp 1', '67430636', '0')`, the default median-surplus selection. At displayed round 12 this community has four offers of `0.699999988079071`, four zero returns, and recorded next pool `0.009999999776482582`. All 40 rows remain accessible. This is an identified descriptive example, not a new empirical treatment estimate.

Material limits remain explicit: only Figure 2A cohorts are covered; BC1/RL networks are replayed, not executed; the final BC1 next pool is unavailable; observed floor/rounding residuals differ from the reconstructed numerical equation; fixed scripts are not human predictions; separate groups are not individual counterfactuals; and participant linkage across groups remains unavailable. Task 03 should fit and evaluate the first behavioral models with declared group boundaries. No training or evolutionary search was started.
