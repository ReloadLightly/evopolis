"""Stream the released CSV and describe its empirical units and missingness.

This is an original inventory of the release, not upstream training code. IDs
are retained only during counting; the returned manifest does not enumerate
participant/group identifiers.
"""

from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import re


_IDENTIFIERS = (
    "launch_id", "episode_id", "block_id", "set", "launch_id_gamewise",
    "launch_id_str", "episode_id_str",
)
_MISSING = {"", "nan", "null", "none"}
_VECTOR_FIELDS = {
    "mechanism_action", "mechanism_observation.prev_reciprocations",
    "mechanism_observation.prev_offers", "next_environment_state.reciprocations",
    "prev_environment_state.reciprocations", "players_reward",
    "players_cumulative_reward",
} | {
    f"player_observation_{i}.{field}"
    for i in range(4) for field in ("offers", "prev_reciprocations")
}


def origin(label: str) -> str:
    """Fail closed rather than classifying an unfamiliar cohort as human."""
    if re.search(r" BC [12]$", label):
        return "synthetic"
    if re.search(r" Exp [1-4]$", label):
        return "human"
    raise ValueError(f"Unrecognized provenance label: {label!r}")


def _vector(value: str) -> list[float]:
    """Read the release's bracketed, comma- or whitespace-separated vectors.

    Some columns wrap each scalar or the whole vector in a further bracket.
    Flattening is used only to count four entries, never to infer player order.
    No Python expression evaluation is used.
    """
    if not (value.startswith("[") and value.endswith("]")):
        raise ValueError(f"Expected bracketed vector: {value[:80]!r}")
    return [float(token) for token in value.translate(str.maketrans("[],", "   ")).split()]


def _id_summary(values: set[str]) -> dict:
    result = {"distinct_nonmissing": len(values)}
    if len(values) <= 12:
        result["values"] = sorted(values)
    return result


def inventory(path: Path) -> dict:
    """Return JSON-serializable counts with memory bounded by game metadata."""
    counts = Counter()
    conditions = {}
    missing = {kind: Counter() for kind in ("human", "synthetic")}
    nonfinite = {kind: Counter() for kind in ("human", "synthetic")}
    vectors = defaultdict(lambda: {"lengths": Counter(), "nonfinite_rows": 0, "formats": set()})
    identifiers = {kind: defaultdict(set) for kind in ("human", "synthetic")}
    human_groups = defaultdict(set)
    human_launch_experiments = defaultdict(set)
    flag_counts = defaultdict(Counter)
    missing_relative_context = {kind: Counter() for kind in ("human", "synthetic")}
    absent_human_context = Counter()
    games = {}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        schema = reader.fieldnames
        if schema is None:
            raise ValueError("Empty data file")
        if len(set(schema)) != len(schema):
            raise ValueError("CSV contains duplicate column names")
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row at line {line}")
            label = row["mech_name_by_player"]
            kind = origin(label)
            counts[kind] += 1
            condition = conditions.setdefault(label, {
                "origin": kind, "rows": 0, "ids": defaultdict(set),
            })
            condition["rows"] += 1
            for name, value in row.items():
                normalized = value.strip().lower()
                if normalized in _MISSING:
                    missing[kind][name] += 1
                elif normalized in {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
                    nonfinite[kind][name] += 1
                elif name in _VECTOR_FIELDS:
                    entries = _vector(value.strip())
                    vector = vectors[name]
                    vector["lengths"][len(entries)] += 1
                    vector["nonfinite_rows"] += int(any(not math.isfinite(x) for x in entries))
                    vector["formats"].add("nested" if "[[" in value else "flat")
                    vector["formats"].add("comma separated" if "," in value else "whitespace separated")
            for name in _IDENTIFIERS:
                value = row[name]
                if value.strip().lower() not in _MISSING:
                    condition["ids"][name].add(value)
                    identifiers[kind][name].add(value)
            for player in range(4):
                if row[f"player_action_{player}_relative"].strip().lower() in _MISSING:
                    context = "offer_below_one" if float(row[f"offer_{player}"]) < 1 else "offer_at_least_one"
                    missing_relative_context[kind][context] += 1
            round_id = int(row["round_id"])
            if round_id < 0:
                raise ValueError(f"Negative round at line {line}")
            game_key = (label, row["launch_id"], row["episode_id"])
            game = games.setdefault(game_key, {"rows": 0, "round_bits": 0, "duplicates": 0})
            bit = 1 << round_id
            game["duplicates"] += int(bool(game["round_bits"] & bit))
            game["round_bits"] |= bit
            game["rows"] += 1
            if kind == "human":
                experiment = label.rsplit(" Exp ", 1)[1]
                human_groups[(experiment, row["launch_id"])].add(row["episode_id"])
                human_launch_experiments[row["launch_id"]].add(experiment)
                for name in ("completed", "first_x_games", "four_usable_players", "four_players_present_or_no_pool", "four_players_present"):
                    flag_counts[name][row[name]] += 1
                if row["four_players_present"] == "False":
                    absent_human_context[f"game_has_no_pool={row['game_has_no_pool']}"] += 1

    output_conditions = []
    for label, condition in sorted(conditions.items()):
        selected = [game for key, game in games.items() if key[0] == label]
        lengths = Counter(game["rows"] for game in selected)
        output_conditions.append({
            "label": label,
            "origin": condition["origin"],
            "rows": condition["rows"],
            "games": len(selected),
            "game_length_min": min(lengths),
            "game_length_max": max(lengths),
            "game_length_counts": dict(sorted(lengths.items())),
            "round_index_min": min((game["round_bits"] & -game["round_bits"]).bit_length() - 1 for game in selected),
            "round_index_max": max(game["round_bits"].bit_length() - 1 for game in selected),
            "duplicate_game_rounds": sum(game["duplicates"] for game in selected),
            "noncontiguous_or_nonzero_start_games": sum(game["round_bits"] != (1 << game["rows"]) - 1 for game in selected),
            "identifiers": {name: _id_summary(condition["ids"][name]) for name in _IDENTIFIERS},
        })

    schema_summary = []
    for name in schema:
        field = {
            "name": name,
            "missing_by_origin": {kind: missing[kind][name] for kind in missing},
            "nonfinite_scalar_by_origin": {kind: nonfinite[kind][name] for kind in nonfinite},
        }
        if name in vectors:
            field["vector"] = {
                "flattened_length_counts": dict(sorted(vectors[name]["lengths"].items())),
                "rows_with_nonfinite_entries": vectors[name]["nonfinite_rows"],
                "formats": sorted(vectors[name]["formats"]),
            }
        schema_summary.append(field)

    return {
        "rows": sum(counts.values()),
        "rows_by_origin": dict(sorted(counts.items())),
        "games_by_origin": {
            kind: sum(condition["games"] for condition in output_conditions if condition["origin"] == kind)
            for kind in ("human", "synthetic")
        },
        "column_count": len(schema),
        "missing_definition": "Empty/whitespace-only CSV cell or scalar nan/null/none (case insensitive); vector entries counted separately.",
        "conditions": output_conditions,
        "schema": schema_summary,
        "identifiers_by_origin": {
            kind: {name: _id_summary(values[name]) for name in _IDENTIFIERS}
            for kind, values in identifiers.items()
        },
        "human_flags": {name: dict(sorted(values.items())) for name, values in sorted(flag_counts.items())},
        "missing_relative_contribution_context": {kind: dict(sorted(values.items())) for kind, values in missing_relative_context.items()},
        "human_rows_without_four_players_present": dict(sorted(absent_human_context.items())),
        "human_group_summary": {
            "experiment_launch_groups": len(human_groups),
            "games_per_group_counts": dict(sorted(Counter(len(episodes) for episodes in human_groups.values()).items())),
            "groups_per_experiment": dict(sorted(Counter(key[0] for key in human_groups).items())),
            "launch_ids_seen_in_multiple_experiments": sum(len(value) > 1 for value in human_launch_experiments.values()),
            "experiment_4_group_count": sum(key[0] == "4" for key in human_groups),
            "experiment_4_groups_with_episodes_0_1_2": sum(key[0] == "4" and value == {"0", "1", "2"} for key, value in human_groups.items()),
        },
        "identity_limitations": [
            "Player indices 0–3 are positions within a group, not globally linkable participant identifiers.",
            "Keep all players and rounds in a launch group together; keep all three Experiment 4 episodes together.",
            "No released participant identifier establishes whether people recur across different launch groups or experiments; cross-group participant independence cannot be certified.",
            "Synthetic launch_id and episode_id values repeat across conditions; use the full condition label with launch_id and episode_id for a game key.",
            "The observed labels do not include the initial 537-game human training cohort; fitting released evaluation data would create new behavioral models.",
            "The paper reports no collected demographics; player presence flags are not demographic data.",
        ],
    }
