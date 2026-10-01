"""Frozen learned inhabitants in the existing common-pool environment.

Each game owns a distinct PCG64 stream and four separate recurrent histories.
Only this Python module advances the world; the viewer replays published rounds.
"""

import argparse
from collections import defaultdict
from contextlib import closing
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import resource
import sqlite3
import time
import zlib

import numpy as np

from .behavior_data import ROOT, RESULTS_DIR, content_hash, read_config, write_json
from .sources import sha256
from .world import CAPACITY, PLAYERS, ROUNDS, WorldState, allocate, participant_observation


DISPLAY = {"equal": "Equal", "proportional": "Proportional", "mixed": "Mixed", "interpolating": "Interpolating"}
ARCHIVE = RESULTS_DIR / "generated.sqlite3"


def packed(value) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def gini(values) -> float | None:
    total = math.fsum(values)
    return None if total == 0 else math.fsum(abs(a - b) for a in values for b in values) / (8 * total)


def load_model(family, seed):
    import torch
    from .behavior_models import BehaviorModel

    path = RESULTS_DIR / "weights" / f"{family}_{seed}" / "best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["family"] != family or checkpoint["seed"] != seed:
        raise ValueError("Checkpoint identity differs from requested family/seed")
    frozen = json.loads((RESULTS_DIR / "frozen_training.json").read_text())["identity"]
    if checkpoint["metadata"] != frozen:
        raise ValueError("Checkpoint source/split/configuration differs from the frozen fit")
    model = BehaviorModel(family)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint, sha256(path)


def sample_integer(rng, n, mixture, alpha, beta):
    """Exact mixture sampling; independent calls use independent resident draws."""
    if n == 0:
        return 0
    component = int(rng.choice(3, p=mixture))
    if component == 0:
        return 0
    if component == 1:
        return n
    return int(rng.binomial(n, rng.beta(alpha, beta)))


def generate_cell(model, checkpoint, checkpoint_hash, recipes):
    """Batch independent worlds for inference, never for decisions or RNG state.

    Every live resident distribution is computed before any world advances.
    Pool/offer arithmetic and integer support remain Python float64/exact floor;
    only the normalized network observations are cast to float32.
    """
    import torch
    from .behavior_models import emission_parts, emission_stats

    if not recipes or len({(r["family"], r["training_seed"], r["mechanism"]) for r in recipes}) != 1:
        raise ValueError("An inference cell must contain one checkpoint and allocation rule")
    states = [WorldState() for _ in recipes]
    streams = [np.random.Generator(np.random.PCG64(r["rollout_seed"])) for r in recipes]
    histories = [[[] for _ in range(PLAYERS)] for _ in recipes]
    rounds = [[] for _ in recipes]
    actual_rounds = [0 for _ in recipes]
    hidden = torch.zeros(1, len(recipes) * PLAYERS, 32) if checkpoint["family"] == "recurrent" else None
    for time_index in range(ROUNDS):
        live = [i for i, state in enumerate(states) if not state.done]
        offers = {i: allocate(states[i].pool, states[i].previous_contributions, mechanism=recipes[i]["mechanism"]) for i in live}
        predictions = {}
        if live:
            inputs = [participant_observation(player, states[i].pool, offers[i], states[i].previous_contributions)
                      for i in live for player in range(PLAYERS)]
            x = torch.tensor(np.asarray(inputs) / CAPACITY, dtype=torch.float32).unsqueeze(1)
            resident_indices = [i * PLAYERS + player for i in live for player in range(PLAYERS)]
            with torch.no_grad():
                raw, next_hidden = model(x, None if hidden is None else hidden[:, resident_indices])
                if hidden is not None:
                    hidden[:, resident_indices] = next_hidden
                raw = raw[:, 0]
                n = torch.tensor([math.floor(value) for i in live for value in offers[i]], dtype=torch.int64)
                log_weights, alpha, beta = emission_parts(raw)
                stats = emission_stats(raw, n)
            # Constant expands a fitted Parameter as a view; explicitly detach
            # even inside no_grad before exposing frozen predictions to NumPy.
            raw_np, weights_np = raw.detach().numpy(), log_weights.exp().detach().numpy()
            alpha_np, beta_np = alpha.detach().numpy(), beta.detach().numpy()
            stats_np = {name: value.detach().numpy() for name, value in stats.items()}
            if not all(np.isfinite(value).all() for value in (raw_np, weights_np, alpha_np, beta_np, *stats_np.values())):
                raise FloatingPointError("Nonfinite generated distribution")
            for position, i in enumerate(live):
                selected = slice(position * PLAYERS, (position + 1) * PLAYERS)
                predictions[i] = (raw_np[selected], weights_np[selected], alpha_np[selected], beta_np[selected],
                                  {name: value[selected] for name, value in stats_np.items()})
        for i, recipe in enumerate(recipes):
            state = states[i]
            padded = state.done
            if padded:
                current_offers, contributions, surplus = [0.0] * PLAYERS, [0] * PLAYERS, [0.0] * PLAYERS
                next_pool = 0.0
                resident_predictions = [{"expected_contribution": 0.0, "p_zero": 1.0, "p_max": 1.0, "legal_max": 0, "emission_parameters": None, "evaluated": False} for _ in range(PLAYERS)]
            else:
                current_offers = offers[i]
                raw, weights, alpha, beta, stats = predictions[i]
                legal = [math.floor(value) for value in current_offers]
                contributions = [sample_integer(streams[i], legal[p], weights[p], alpha[p], beta[p]) for p in range(PLAYERS)]
                states[i], result = state.advance(current_offers, contributions, integer_contributions=True)
                next_pool, surplus = result.next_pool, result.surplus
                actual_rounds[i] += 1
                resident_predictions = [{"expected_contribution": float(stats["mean"][p]), "p_zero": float(stats["p_zero"][p]),
                                         "p_max": float(stats["p_max"][p]), "legal_max": legal[p],
                                         "emission_parameters": raw[p].tolist(), "mixture_weights": weights[p].tolist(),
                                         "alpha": float(alpha[p]), "beta": float(beta[p]), "evaluated": True} for p in range(PLAYERS)]
            for player, value in enumerate(surplus):
                histories[i][player].append(value)
            cumulative = [math.fsum(values) for values in histories[i]]
            rounds[i].append({
                "round_id": time_index, "pool_before": 0.0 if padded else state.pool,
                "offers": list(current_offers), "contributions": contributions, "surplus": list(surplus),
                "cumulative_surplus": cumulative, "group_cumulative_surplus": math.fsum(cumulative),
                "pool_after": next_pool, "equation_after": next_pool, "equation_input_valid": True, "pool_residual": 0.0,
                "after_source": "post-termination zero padding" if padded else "new simulation through WorldState",
                "active_count": sum(value >= 1 for value in current_offers), "padded": padded, "predictions": resident_predictions,
            })
    episodes = []
    for i, recipe in enumerate(recipes):
        means = [math.fsum(values) / ROUNDS for values in histories[i]]
        state = states[i]
        identity = f"trained-{recipe['family']}-{recipe['training_seed']}-{recipe['mechanism']}-{recipe['rollout_index']:03d}"
        first_depletion = next((r["round_id"] + 1 for r in rounds[i] if r["pool_after"] < 1), None)
        episodes.append({
            "id": identity, "cohort": "trained", "family": recipe["family"], "training_seed": recipe["training_seed"],
            "selected_epoch": checkpoint["epoch"], "checkpoint_sha256": checkpoint_hash,
            "rollout_seed": str(recipe["rollout_seed"]), "rollout_index": recipe["rollout_index"],
            "condition": f"New trained EvoPolis agents / {recipe['family']} / {DISPLAY[recipe['mechanism']]}",
            "mechanism": DISPLAY[recipe["mechanism"]], "baseline": recipe["mechanism"],
            "launch_id": f"{recipe['family']}_{recipe['training_seed']}", "episode_id": str(recipe["rollout_index"]),
            "surplus": math.fsum(means) / PLAYERS, "gini": gini(means), "player_mean_surplus": means,
            "round_count": ROUNDS, "actual_rounds": actual_rounds[i], "rounds": rounds[i],
            "termination": f"exact zero after round {actual_rounds[i]}; padded to 40" if state.pool == 0 else "40-round horizon",
            "exact_zero_termination": state.pool == 0, "depleted": first_depletion is not None,
            "first_depletion_after_round": first_depletion, "final_pool": state.pool, "final_sustainment": state.pool > 1,
            "active_allocation_fraction": sum(r["active_count"] for r in rounds[i]) / (PLAYERS * ROUNDS),
            "provenance_label": "New trained EvoPolis agents · frozen learned parameters",
            "source_label": "New trained EvoPolis agents", "exploratory_transfer": recipe["mechanism"] == "interpolating",
        })
    return episodes


def generate_game(model, checkpoint, checkpoint_hash, *, rollout_seed, mechanism, rollout_index=0):
    """Generate one independent game for checks, with newly reset memory/RNG.

    The published archive uses batches of 64 independent games for inference;
    batch-size floating-point differences may affect raw outputs at machine
    precision. Archive verification therefore reruns its complete original cell.
    """
    recipe = {"family": checkpoint["family"], "training_seed": checkpoint["seed"], "mechanism": mechanism,
              "rollout_seed": int(rollout_seed), "rollout_index": rollout_index}
    return generate_cell(model, checkpoint, checkpoint_hash, [recipe])[0]


def generation_identity(config, seeds):
    return {"config_sha256": sha256(ROOT / "configs/task03.json"), "split_sha256": sha256(RESULTS_DIR / "split.json"),
            "rollout_seed_table_hash": seeds["table_hash"], "generator_sha256": sha256(Path(__file__)),
            "world_sha256": sha256(ROOT / "evopolis/world.py"), "behavior_models_sha256": sha256(ROOT / "evopolis/behavior_models.py"),
            "checkpoints": {f"{family}_{seed}": sha256(RESULTS_DIR / "weights" / f"{family}_{seed}" / "best.pt")
                            for family in config["families"] for seed in config["training_seeds"]}}


def validate_generation_contract(identity):
    """Refuse trajectory generation from changed post-selection weights/rules."""
    contract = json.loads((RESULTS_DIR / "test_opening.json").read_text())["frozen_contract"]
    trained = contract["training_identity"]
    if any(identity[key] != trained[key] for key in ("config_sha256", "split_sha256")):
        raise ValueError("Generation configuration/split differs from the test-opening contract")
    for name, digest in trained["code_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Frozen training code changed before generation: {name}")
    for specification in contract["checkpoints"]:
        key = f"{specification['family']}_{specification['seed']}"
        metadata = json.loads((RESULTS_DIR / "weights" / key / "metadata.json").read_text())
        if identity["checkpoints"][key] != specification["best_sha256"] or metadata["selected_epoch"] != specification["selected_epoch"]:
            raise ValueError("Generation checkpoint differs from frozen validation selection")
    for name, digest in contract.get("generation_code_sha256", {}).items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Generation code differs from test opening: {name}")


def generate():
    import torch
    from .behavior_train import runtime_setup

    runtime_setup()
    config = read_config()
    if not (RESULTS_DIR / "test_opening.json").exists():
        raise ValueError("Complete all fits, freeze selection and open the declared evaluation before generating/reporting comparisons")
    seeds = json.loads((RESULTS_DIR / "rollout_seeds.json").read_text())
    if len(seeds["games"]) != 3072 or content_hash(seeds["games"]) != seeds["table_hash"]:
        raise ValueError("Frozen rollout-seed table is incomplete or changed")
    identity = generation_identity(config, seeds)
    validate_generation_contract(identity)
    partial = ARCHIVE.with_suffix(".sqlite3.partial")
    target = ARCHIVE if ARCHIVE.exists() else partial
    database = sqlite3.connect(target)
    started = time.monotonic()
    try:
        database.execute("CREATE TABLE IF NOT EXISTS episodes (id TEXT PRIMARY KEY, metadata TEXT NOT NULL, body BLOB NOT NULL)")
        database.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        existing = database.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if existing and json.loads(existing[0]) != identity:
            raise ValueError("Generation resume refused: frozen seeds, configuration, code, or checkpoint hashes changed")
        database.execute("INSERT OR IGNORE INTO metadata VALUES ('identity',?)", (packed(identity),))
        database.commit()
        cells = defaultdict(list)
        for recipe in seeds["games"]:
            cells[(recipe["family"], recipe["training_seed"], recipe["mechanism"])].append(recipe)
        for count, ((family, training_seed, mechanism), recipes) in enumerate(cells.items(), start=1):
            prefix = f"trained-{family}-{training_seed}-{mechanism}-"
            existing_count = database.execute("SELECT COUNT(*) FROM episodes WHERE id LIKE ?", (prefix + "%",)).fetchone()[0]
            if existing_count == 64:
                print(f"Generated cell {count}/48 already verified in resumable archive", flush=True)
                continue
            if existing_count:
                raise ValueError("Incomplete generation cell should not occur after atomic transaction")
            model, checkpoint, checkpoint_hash = load_model(family, training_seed)
            episodes = generate_cell(model, checkpoint, checkpoint_hash, recipes)
            with database:
                for episode in episodes:
                    metadata = {key: value for key, value in episode.items() if key != "rounds"}
                    database.execute("INSERT INTO episodes VALUES (?,?,?)", (episode["id"], packed(metadata), zlib.compress(packed({"rounds": episode["rounds"]}).encode(), 6)))
            print(f"Generated {count}/48: {family}/{training_seed}/{mechanism}, all 64 games; {time.monotonic()-started:.1f}s", flush=True)
            del model, episodes
        metadata = [json.loads(row[0]) for row in database.execute("SELECT metadata FROM episodes")]
        if len(metadata) != 3072:
            raise ValueError("Expected every one of the 3,072 generated games")
        candidates = [e for e in metadata if e["family"] == "recurrent" and e["training_seed"] == 17 and e["mechanism"] == "Equal"]
        ordered = sorted(Fraction(e["surplus"]) for e in candidates)
        median = (ordered[31] + ordered[32]) / 2
        default = min(candidates, key=lambda e: (abs(Fraction(e["surplus"]) - median), e["rollout_index"]))
        provenance = {"episode_count": 3072, "round_count": 3072 * ROUNDS, "source_label": "New trained EvoPolis agents",
                      "default_selection": config["representative_display"], "surplus_denominator": PLAYERS * ROUNDS,
                      "weights_frozen": True, "online_parameter_adaptation": False, "memory": "separate resident hidden states; reset between independent games",
                      "environment": "evopolis.world.allocate and WorldState; exact-zero/40-round scheduler; no undocumented 0.01 floor",
                      "sampling": "independent NumPy PCG64 stream per game; independent resident mixture/Beta/Binomial draws",
                      "source_sha256": json.loads((RESULTS_DIR / "split.json").read_text())["source_sha256"],
                      "identity": identity, "torch_version": str(torch.__version__), "numpy_version": np.__version__}
        with database:
            database.execute("INSERT OR REPLACE INTO metadata VALUES ('default_id',?)", (packed(default["id"]),))
            database.execute("INSERT OR REPLACE INTO metadata VALUES ('provenance',?)", (packed(provenance),))
        database.execute("VACUUM")
    finally:
        database.close()
    if target == partial:
        partial.replace(ARCHIVE)
    summarize()
    write_json(RESULTS_DIR / "generation_complete.json", {"games": 3072, "padded_rounds": 3072 * ROUNDS, "invocation_seconds": time.monotonic() - started,
                                                         "archive_sha256": sha256(ARCHIVE), "archive_bytes": ARCHIVE.stat().st_size,
                                                         "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "identity": identity})


def archive_episodes():
    with closing(sqlite3.connect(ARCHIVE.resolve().as_uri() + "?mode=ro", uri=True)) as database:
        for metadata, body in database.execute("SELECT metadata,body FROM episodes ORDER BY id"):
            yield {**json.loads(metadata), **json.loads(zlib.decompress(body))}


def summarize():
    rows, cells = [], defaultdict(list)
    for episode in archive_episodes():
        row = {key: episode[key] for key in ("id", "family", "training_seed", "selected_epoch", "checkpoint_sha256", "mechanism", "rollout_index", "rollout_seed", "surplus", "gini", "actual_rounds", "exact_zero_termination", "depleted", "first_depletion_after_round", "final_pool", "final_sustainment", "active_allocation_fraction", "exploratory_transfer")}
        rows.append(row)
        cells[(row["family"], row["training_seed"], row["mechanism"])].append(row)
    with (RESULTS_DIR / "generated_groups.csv").open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for (family, seed, mechanism), values in cells.items():
        summary = {"family": family, "training_seed": seed, "mechanism": mechanism, "games": len(values), "undefined_gini": sum(v["gini"] is None for v in values)}
        for metric in ("surplus", "gini", "actual_rounds", "final_pool", "active_allocation_fraction"):
            numbers = [v[metric] for v in values if v[metric] is not None]
            summary[metric] = {"mean": float(np.mean(numbers)) if numbers else None,
                               "median": float(np.median(numbers)) if numbers else None,
                               "q10": float(np.quantile(numbers, 0.1)) if numbers else None,
                               "q90": float(np.quantile(numbers, 0.9)) if numbers else None}
        for metric in ("exact_zero_termination", "depleted", "final_sustainment"):
            summary[metric + "_fraction"] = sum(v[metric] for v in values) / len(values)
        summaries.append(summary)
    write_json(RESULTS_DIR / "generated_outcomes.json", {"games": len(rows), "cells": summaries, "unit": "new independent simulated game, not human participants", "surplus_denominator": 160,
              "depletion": "any post-round pool <1", "exact_zero": "simulation ends at exactly zero", "active": "allocation>=1; all160player-rounds denominator", "sustainment": "pool after40rounds>1", "quantiles": "descriptive game distribution, not human confidence intervals"})


def verify():
    """Fresh-process checkpoint reload, exact default-cell replay and all accounting."""
    from .behavior_train import runtime_setup
    from .trained_viewer_data import generated_catalog, generated_episode

    runtime_setup()
    catalog = generated_catalog()
    default = generated_episode(catalog["default_id"])
    model, checkpoint, digest = load_model(default["family"], default["training_seed"])
    table = json.loads((RESULTS_DIR / "rollout_seeds.json").read_text())
    recipes = [row for row in table["games"] if row["family"] == default["family"] and row["training_seed"] == default["training_seed"] and DISPLAY[row["mechanism"]] == default["mechanism"]]
    for replay in generate_cell(model, checkpoint, digest, recipes):
        stored = generated_episode(replay["id"])
        if replay != stored:
            raise AssertionError(f"Checkpoint and rollout seed did not reproduce {replay['id']}")
    forced, padding, live_rounds = 0, 0, 0
    for episode in archive_episodes():
        state, cumulative = WorldState(), [[] for _ in range(4)]
        for row in episode["rounds"]:
            if row["padded"]:
                padding += 1
                assert state.done and state.pool == 0
                assert all(value == 0 for value in row["offers"] + row["contributions"] + row["surplus"])
            else:
                live_rounds += 1
                offers = allocate(state.pool, state.previous_contributions, mechanism=episode["baseline"])
                assert list(offers) == row["offers"]
                assert state.pool == row["pool_before"]
                for player, offer in enumerate(offers):
                    n, choice = math.floor(offer), row["contributions"][player]
                    assert isinstance(choice, int) and 0 <= choice <= n
                    assert row["predictions"][player]["legal_max"] == n
                    forced += n == 0
                state, result = state.advance(offers, row["contributions"], integer_contributions=True)
                assert result.next_pool == row["pool_after"]
                assert list(result.surplus) == row["surplus"]
            for player, value in enumerate(row["surplus"]):
                cumulative[player].append(value)
            assert [math.fsum(values) for values in cumulative] == row["cumulative_surplus"]
        means = [math.fsum(values) / 40 for values in cumulative]
        assert math.fsum(means) / 4 == episode["surplus"]
        assert gini(means) == episode["gini"]
        assert state.round_index == episode["actual_rounds"]
        assert state.pool == episode["final_pool"]
    write_json(RESULTS_DIR / "generation_verification.json", {"verified_games": 3072, "verified_live_rounds": live_rounds, "verified_padding_rounds": padding, "forced_live_choices": forced,
               "fresh_process_reloaded_checkpoint": digest, "exact_reproduced_games": 64, "default_id": default["id"], "all_accounting_exact": True,
               "same_simultaneous_observation_state": True, "separate_resident_rng_draws": True, "full_horizon_denominator": 160})
    print("Verified every generated accounting path and exact checkpoint/seed replay of the 64-game default cell.", flush=True)


def plot():
    """Descriptive game distributions, seed variation and held-out human context."""
    if not (RESULTS_DIR / "test_opening.json").exists():
        raise ValueError("Held-out human trajectory comparison requires the declared test opening")
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from .behavior_data import load
    from .plotting import BACKGROUND, PANEL, PARCHMENT, MUTED, COLORS

    arrays, manifest = load()
    config = read_config()
    families = config["families"]
    cells = defaultdict(list)
    for episode in archive_episodes():
        cells[(episode["family"], episode["mechanism"])].append({
            "seed": episode["training_seed"], "surplus": episode["surplus"], "gini": episode["gini"], "participation": episode["active_allocation_fraction"],
            "pool": [r["pool_before"] for r in episode["rounds"]], "active": [r["active_count"] for r in episode["rounds"]]})
    human = defaultdict(list)
    for group in manifest["groups"]:
        if group["split"] != "test" or group["mechanism"] == "M1":
            continue
        i = group["index"]
        means = [math.fsum(values) / 40 for values in arrays["surplus"][i]]
        human[group["mechanism"]].append({"surplus": math.fsum(means) / 4, "gini": gini(means),
               "participation": float(np.mean(arrays["offers"][i] >= 1)), "pool": arrays["pool"][i], "active": np.sum(arrays["offers"][i] >= 1, axis=0)})
    mechanisms = ["Equal", "Proportional", "Mixed", "Interpolating"]
    labels = ["Constant", "Linear", "MLP", "GRU"]
    rc = {"font.family": "DejaVu Sans Mono", "font.size": 9, "text.color": PARCHMENT, "axes.labelcolor": PARCHMENT,
          "xtick.color": MUTED, "ytick.color": MUTED, "svg.fonttype": "none", "svg.hashsalt": "evopolis-task03-generated"}
    directory = ROOT / "docs/assets"
    with plt.rc_context(rc):
        fig, axes = plt.subplots(3, 4, figsize=(17, 11), sharey="row")
        fig.patch.set_facecolor(BACKGROUND)
        for column, mechanism in enumerate(mechanisms):
            for row, (metric, ylabel) in enumerate((("surplus", "Surplus / player / round"), ("gini", "Four-resident Gini"), ("participation", "Fraction of offers >=1"))):
                ax = axes[row, column]
                _style_axis(ax, PANEL, MUTED)
                for position, (family, color) in enumerate(zip(families, COLORS), start=1):
                    values = cells[(family, mechanism)]
                    for seed_index, seed in enumerate(config["training_seeds"]):
                        numbers = [v[metric] for v in values if v["seed"] == seed and v[metric] is not None]
                        xs = position + (seed_index - 1) * 0.2 + (np.arange(len(numbers)) % 8 - 3.5) * 0.014
                        ax.scatter(xs, numbers, s=5, alpha=0.25, color=color, linewidths=0)
                        ax.scatter(position + (seed_index - 1) * 0.2, np.mean(numbers), color=color, marker=("o", "s", "^")[seed_index], edgecolors=PARCHMENT, linewidths=0.6, s=33, zorder=4)
                if human[mechanism]:
                    numbers = [v[metric] for v in human[mechanism] if v[metric] is not None]
                    ax.scatter((np.arange(len(numbers)) - 3.5) * 0.035, numbers, color=PARCHMENT, marker="x", s=27)
                    ax.scatter(0, np.mean(numbers), color=PARCHMENT, marker="D", s=38)
                ax.set_xticks(range(5), ["Human", *labels], rotation=30, ha="right")
                ax.set_xlim(-0.45, 4.5)
                if column == 0:
                    ax.set_ylabel(ylabel)
                if row == 0:
                    ax.set_title(mechanism + ("\nexploratory transfer" if mechanism == "Interpolating" else "\n8 held-out human groups"), loc="left", color=PARCHMENT, fontsize=11)
                if row == 1:
                    ax.set_ylim(-0.015, 0.765)
                if row == 2:
                    ax.set_ylim(-0.02, 1.02)
        axes[0, 0].set_ylim(bottom=0)
        fig.suptitle("EVOPOLIS / TRAINED WORLDS 03\nPredicted communities vary across fitted seeds and allocation rules", x=0.055, y=0.98, ha="left", fontsize=16, fontweight="bold")
        fig.text(0.055, 0.052, "Small marks: all 64 generated games per seed; outlined circle/square/triangle: means for seeds 17/29/43. Human: held-out groups.", color=MUTED, fontsize=8.5)
        fig.text(0.055, 0.028, "Surplus and participation include all 40 rounds, with zero padding after exact-zero termination. Generated games are model outcomes, not new human evidence.", color=MUTED, fontsize=8.5)
        fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.12, wspace=0.14, hspace=0.36)
        _save_figure(fig, directory / "trained-outcomes", BACKGROUND)
        plt.close(fig)
        fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey="row")
        fig.patch.set_facecolor(BACKGROUND)
        for column, mechanism in enumerate(mechanisms):
            for row, (metric, ylabel) in enumerate((("pool", "Pool before allocation"), ("active", "Residents with offer >=1"))):
                ax = axes[row, column]
                _style_axis(ax, PANEL, MUTED)
                for family, color in zip(families, COLORS):
                    values = cells[(family, mechanism)]
                    all_values = np.asarray([v[metric] for v in values])
                    lower, upper = np.quantile(all_values, [0.1, 0.9], axis=0)
                    ax.fill_between(range(1, 41), lower, upper, color=color, alpha=0.075, linewidth=0)
                    for seed_index, seed in enumerate(config["training_seeds"]):
                        average = np.mean([v[metric] for v in values if v["seed"] == seed], axis=0)
                        ax.plot(range(1, 41), average, color=color, lw=0.8, alpha=0.65, linestyle=("-", "--", ":")[seed_index])
                    ax.plot(range(1, 41), np.mean(all_values, axis=0), color=color, lw=2)
                if human[mechanism]:
                    observed = np.asarray([v[metric] for v in human[mechanism]])
                    for values in observed:
                        ax.plot(range(1, 41), values, color=PARCHMENT, alpha=0.12, lw=0.5)
                    ax.plot(range(1, 41), observed.mean(axis=0), color=PARCHMENT, lw=2, linestyle="--")
                if column == 0:
                    ax.set_ylabel(ylabel)
                if row == 0:
                    ax.set_title(mechanism + (" / exploratory" if mechanism == "Interpolating" else ""), loc="left", color=PARCHMENT)
                    ax.set_ylim(0, 205)
                else:
                    ax.set_ylim(0, 4.1)
                    ax.set_xlabel("Common round coordinate")
                ax.set_xlim(1, 40)
                ax.set_xticks([1, 10, 20, 30, 40])
        handles = [Line2D([0], [0], color=color, lw=2, label=label) for color, label in zip(COLORS, labels)] + [Line2D([0], [0], color=PARCHMENT, lw=2, ls="--", label="Held-out human mean")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.085), frameon=False, ncol=5)
        fig.suptitle("EVOPOLIS / TRAINED WORLDS 03\nPool and participation on shared numerical scales", x=0.055, y=0.98, ha="left", fontsize=16, fontweight="bold")
        fig.text(0.055, 0.060, "Thin colored lines: each seed's 64-game mean; thick: equal seed means. Pale thin lines: individual held-out human groups.", color=MUTED, fontsize=8.5)
        fig.text(0.055, 0.037, "Shading: central 10–90% of 192 generated games per family/rule; descriptive game variation, not confidence intervals.", color=MUTED, fontsize=8.5)
        fig.text(0.055, 0.014, "Recorded residual pool near 0.01 is preserved; generated worlds have no pool floor. Interpolating has no human Exp. 1 counterpart.", color=MUTED, fontsize=8.5)
        fig.subplots_adjust(left=0.065, right=0.985, top=0.84, bottom=0.21, wspace=0.14, hspace=0.24)
        _save_figure(fig, directory / "trained-trajectories", BACKGROUND)
        plt.close(fig)


def _style_axis(ax, panel, muted):
    ax.set_facecolor(panel)
    ax.grid(color=muted, alpha=0.16, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(muted)


def _save_figure(fig, path, background):
    fig.savefig(path.with_suffix(".png"), dpi=160, facecolor=background)
    fig.savefig(path.with_suffix(".svg"), metadata={"Date": None}, facecolor=background)
    svg = path.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "verify", "plot"))
    args = parser.parse_args()
    {"generate": generate, "verify": verify, "plot": plot}[args.command]()


if __name__ == "__main__":
    main()
