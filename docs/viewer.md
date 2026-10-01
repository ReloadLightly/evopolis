# Community viewer

Run from the Linux checkout with `uv` installed:

```bash
bash scripts/viewer.sh --port 8765
```

Open `http://127.0.0.1:8765`. Ctrl+C closes the local server. An occupied port produces an alternative-port suggestion; for example use `--port 8766`. The wrapper uses the locked Python environment and a repository-local package cache. With the environment already active, `python -m evopolis.viewer --port 8765` runs the same application. There is no Node build or hosted service.

## Recorded evidence

The archive contains Figure 2A's **160 human Experiment 1 games and 2,048 recorded BC1 games**, under Equal, Mixed, Proportional and recorded RL M1 allocation. All have four residents and 40 recorded rounds. Other experiments, including repeated variable-length Experiment 4 games, are outside this viewer.

Choose an evidence cohort, allocation condition and episode. The menu includes each episode's completed-game mean surplus and Gini; the optional scatter is clickable and keyboard selectable. The initial episode is nearest the human Equal condition's median mean surplus. On changing condition the same rule applies, with exact median ties broken by `(condition, launch_id, episode_id)`. All records remain available; no successful-run filter is applied.

Play/pause, previous/next, restart, speed and the scrubber share a clock across both comparison panes. Space toggles playback and arrow keys seek when focus is outside a form control. Resident buttons and source/episode selectors support keyboard operation. Reduced motion defaults to the system preference and can be toggled; it retains each discrete phase with static cues. A copied view link preserves the mode, full episode IDs or sandbox settings, selected residents, and round. It points to this local server; another reader needs their own running checkout.

Residents **A–D are positions 0–3 within a group**, not released participant identities. Clicking a sprite or its button opens that position's record. Each round visually progresses through allocation, simultaneous return/retention, and replenishment. Tokens symbolize positive transfers rather than counting units. Spatial layout creates no transport costs, communication or extra observations.

Pool before is `mechanism_observation.pool`. Human next pool comes from the recorded next-state field; BC1 uses the next round's pool within the same complete episode key. The **last BC1 after-pool is unavailable**, displayed as N/A. The optional equation estimate and its residual are separately labeled and never replace observed values. All 40 rows remain visible after the pool falls below one, including the source's approximately 0.01 residual floor.

Round offers, returns and surplus use the group-order `offer_i`, `player_action_i` and `player_reward_i` fields. Focal-player observation vectors are not substituted. Exact source strings are accessible in the inspection table; rounded labels and scene tokens never drive state updates. **Cumulative retained** means the sum of recorded round rewards through the cursor. It is recomputed from history with `math.fsum`, not copied from the source's separately rounded cumulative counter, which is also exposed in inspection. Seeking selects precomputed records rather than incrementing totals.

Participation is explicitly **the number of offers at least one resource unit**. A smaller offer marks this round's opportunity, not permanent exclusion. The inspector distinguishes human public information, BC1's nine input features plus recurrent memory, and researcher context. Current simultaneous choices, the research playback horizon, and future outcomes are not behavioral inputs.

The timeline's left axis is pool before the round; its right axis is cumulative group surplus. It shows the whole episode, including future outcomes relative to the cursor, as a research view. Current cumulative Gini uses four players' cumulative surplus, has maximum 0.75, and is undefined when all totals are zero. Completed-game Gini is shown separately. Comparison panes depict **different observed groups or recorded model episodes**, not individual counterfactuals.

## Transparent sandbox

Switching to Scripted sandbox starts a new simulation. Each resident uses a fixed editable `q` between zero and one and returns `floor(q × offer)`. The decimal `q` is multiplied exactly by the represented floating allocation before flooring: `q=0.58` with an offer of 50 returns 29. This avoids a binary-product artifact giving 28. It does not round or repair recorded human decisions.

Equal, Proportional, Mixed and Interpolating call `allocate` and `WorldState` from [world.py](../evopolis/world.py). There is no second JavaScript environment, and no newly implemented RL allocator. A small floating-point conservation correction balances a baseline's largest offer against the remaining shares so their sum cannot overshoot the pool and create a negative next state. No 0.01 floor is introduced.

Both comparison panes receive the same fractions when entering sandbox comparison; subsequent pane-specific edits are explicit. Every parameter or rule edit starts at pool 200 and round 1. Reset settings restores that pane's default fractions `[0.75, 0.75, 0.75, 0.25]` and Equal rule. Restart replays the current settings. At exact-zero termination the shorter pane holds its final state, labeled as ended, while the longer pane continues to at most 40 rounds. Fixed fractions are demonstrations, not fitted behavior, human predictions, online adaptation or learning.

## Preparation and verification

On first launch the pinned CSV is verified and streamed once into `data/cache/viewer/figure2a.sqlite3`. The SQLite index contains compressed episodes, compact metadata and source/schema provenance; the browser receives an index and only selected round-level episodes. It never downloads the 204 MiB CSV. Progress is printed during preparation. A compatible cache can be used offline even if raw data are unavailable; changed source fingerprints trigger checksum verification, and source/schema/summary changes invalidate stale caches.

Preparation alone:

```bash
bash scripts/viewer.sh --prepare-only
```

The measured cache is **23,556,096 bytes**. First preparation took **46.19 seconds / 23.6 MiB peak RSS**; a cached command took **0.39 seconds / 30,592 KiB peak RSS**, including process startup. These are separate process measurements, excluding Chromium. There is no long-running compute job to checkpoint.

Validation covered all **2,208 episode surplus/Gini summaries**, agreeing with Task 01 within **2.22e-16**, under a `1e-9` comparison tolerance. Replay diagnostics retain the prior `1e-4` tolerance. The 21-test Python suite includes source-string fidelity, episode keys, ordering, timing, missing BC1 values, cumulative seeking, cache reuse, altered-source rejection, integer contributions and sandbox boundaries. Existing Task 01 artifact hashes remain unchanged.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The application was exercised in Chromium at **1366×900**, including actual play/pause, backward scrubbing, resident selection, BC1, independent synchronized selection, keyboard scatter selection, scripted rule/fraction changes, reset, unequal termination lengths and reduced motion. [The interaction check](../scripts/check-viewer-browser.js) can be run against a freshly opened viewer using `agent-browser eval --stdin`; it does not import application internals. [Browser evidence](../results/task02/browser-verification.json) and [verification/provenance](../results/task02/verification.json) retain the measured results. Browser errors and console logs were empty. The [README screenshot](assets/community-viewer.png) was captured from the actual default community at round 1; it is not concept art or a rendered fixture.

Behavioral training remains Task 03. The release cannot link participants across separate group IDs; later modeling must retain group boundaries and qualify participant independence. The viewer makes these recorded trajectories inspectable without making them an untouched evaluation set for future model search.
