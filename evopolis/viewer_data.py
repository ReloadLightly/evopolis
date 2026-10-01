"""Indexed, source-preserving Figure 2A replay and a fixed-fraction sandbox.

The cache contains recorded outcomes, not a reconstruction through WorldState.
Surplus/Gini definitions follow the attributed Task 01 analysis; see NOTICE.
Only the explicitly scripted sandbox advances the numerical environment.
"""

from collections import Counter
from contextlib import closing
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import tempfile
import zlib

from .replay import REPLAY_TOLERANCE, TIGHT_TOLERANCE
from .sources import REVISION, SOURCES, acquire, sha256
from .world import CAPACITY, MULTIPLIER, ROUNDS, WorldState, allocate, step


SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results/task01/figure2a_groups.csv"
SOURCE_NAME = "sustainable_behavior.csv"
MECHANISMS = {
    "Equal Baseline": "Equal",
    "Mixed Baseline": "Mixed",
    "Proportional Baseline": "Proportional",
    "RL Agent (M1)": "Recorded RL M1",
}
SELECTED = {
    f"{mechanism} {suffix}": (cohort, display, mechanism, suffix)
    for suffix, cohort in (("Exp 1", "human"), ("BC 1", "bc1"))
    for mechanism, display in MECHANISMS.items()
}
RAW_FIELDS = (
    "mech_name_by_player", "launch_id", "episode_id", "round_id",
    "mechanism_observation.pool", "next_environment_state.pool",
    *(f"offer_{i}" for i in range(4)),
    *(f"player_action_{i}" for i in range(4)),
    *(f"player_reward_{i}" for i in range(4)),
    "players_cumulative_reward",
)


def _json(value) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def episode_id(condition: str, launch_id: str, game_id: str) -> str:
    """Address an episode by its complete key, preserving identifier strings."""
    return hashlib.sha256(_json([condition, launch_id, game_id]).encode()).hexdigest()


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()), "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
    }


def _open(path: Path) -> sqlite3.Connection:
    # Read-only connections cannot accidentally create a missing replay cache.
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def _stored_provenance(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with closing(_open(path)) as database:
            row = database.execute("SELECT value FROM metadata WHERE key='provenance'").fetchone()
            return json.loads(row[0]) if row else None
    except (sqlite3.DatabaseError, ValueError):
        return None


def _number(raw: dict, name: str) -> float:
    result = float(raw[name])
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite replay value: {name}")
    return result


def _gini(values: list[float]) -> float | None:
    total = math.fsum(values)
    if total == 0:
        return None  # JSON null expresses the undefined zero denominator.
    return math.fsum(abs(a - b) for a in values for b in values) / (8 * total)


def _recorded_episode(rows: list[dict]) -> dict:
    """Convert one full game, independent of CSV order and observation vectors."""
    rows = sorted(rows, key=lambda row: int(row["round_id"]))
    if [int(row["round_id"]) for row in rows] != list(range(ROUNDS)):
        raise ValueError("Figure 2A replay requires exactly rounds 0–39")
    first = rows[0]
    condition = first["mech_name_by_player"]
    cohort, mechanism, _, _ = SELECTED[condition]
    full_key = (condition, first["launch_id"], first["episode_id"])
    if any((row["mech_name_by_player"], row["launch_id"], row["episode_id"]) != full_key for row in rows):
        raise ValueError("Rows from distinct episode keys cannot be combined")
    history = [[], [], [], []]
    rounds = []
    for index, raw in enumerate(rows):
        pool = _number(raw, "mechanism_observation.pool")
        offers = [_number(raw, f"offer_{i}") for i in range(4)]
        contributions = [_number(raw, f"player_action_{i}") for i in range(4)]
        surplus = [_number(raw, f"player_reward_{i}") for i in range(4)]
        for player, value in enumerate(surplus):
            history[player].append(value)
        cumulative = [math.fsum(values) for values in history]
        if raw["next_environment_state.pool"].strip():
            after = _number(raw, "next_environment_state.pool")
            after_source = "recorded next-pool field"
            after_raw = raw["next_environment_state.pool"]
        elif index + 1 < len(rows):
            after = _number(rows[index + 1], "mechanism_observation.pool")
            after_source = "following recorded round"
            after_raw = rows[index + 1]["mechanism_observation.pool"]
        else:
            after = None
            after_source = "unavailable (final recorded BC1 round)"
            after_raw = None
        # Diagnostic only: as in the Task 01 audit, invalid source inputs remain
        # visible and the observed next pool is never replaced with this number.
        try:
            equation = step(pool, offers, contributions, tolerance=REPLAY_TOLERANCE).next_pool
            equation_input_valid = True
        except ValueError:
            equation = min(CAPACITY, pool - math.fsum(offers) + MULTIPLIER * math.fsum(contributions))
            equation_input_valid = False
        rounds.append({
            "round_id": index, "pool_before": pool, "offers": offers,
            "contributions": contributions, "surplus": surplus,
            "cumulative_surplus": cumulative,
            "group_cumulative_surplus": math.fsum(cumulative),
            "pool_after": after, "after_source": after_source,
            "pool_after_raw": after_raw, "equation_after": equation,
            "equation_input_valid": equation_input_valid,
            "pool_residual": None if after is None else after - equation,
            "active_count": sum(offer >= 1 for offer in offers),
            "raw": raw,
        })
    means = [math.fsum(values) / ROUNDS for values in history]
    return {
        "id": episode_id(*full_key), "condition": condition, "cohort": cohort,
        "mechanism": mechanism, "launch_id": full_key[1], "episode_id": full_key[2],
        "surplus": math.fsum(means) / 4, "gini": _gini(means),
        "round_count": len(rounds), "rounds": rounds,
        "termination": "40 recorded rounds, including depleted rounds",
        "provenance_label": "Recorded human data · Experiment 1" if cohort == "human" else "Recorded upstream model outcomes · BC1",
    }


def _validate_summary(episode: dict, summaries: dict) -> float:
    _, _, mechanism, suffix = SELECTED[episode["condition"]]
    key = (suffix, mechanism, episode["launch_id"])
    if key not in summaries:
        raise ValueError(f"Episode is missing from saved Figure 2A summary: {key}")
    reference = summaries[key]
    differences = [abs(episode[metric] - float(reference[metric])) for metric in ("surplus", "gini")]
    if max(differences) > TIGHT_TOLERANCE:
        raise ValueError(f"Replay disagrees with saved Figure 2A summary beyond 1e-9: {key}")
    return max(differences)


def prepare_cache(
    raw_dir: Path = Path("data/raw"),
    cache_dir: Path = Path("data/cache/viewer"),
) -> Path:
    """Verify the pinned release, stream once, and atomically index all games.

    A previously verified cache works without the raw file or a network. A
    changed source fingerprint triggers checksum verification; a changed hash
    fails explicitly. Schema, pinned hash, and saved summary changes invalidate
    the cache. Unchanged source files are not read again on ordinary launches.
    """
    raw_dir, cache_dir = Path(raw_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "figure2a.sqlite3"
    raw_path = raw_dir / SOURCE_NAME
    expected = json.loads(Path(__file__).with_name("source_checksums.json").read_text())
    expected_hash = expected[SOURCE_NAME]
    summary_hash = sha256(SUMMARY_PATH)
    existing = _stored_provenance(target)
    compatible = existing and all((
        existing.get("schema_version") == SCHEMA_VERSION,
        existing.get("source_sha256") == expected_hash,
        existing.get("summary_sha256") == summary_hash,
        existing.get("episode_count") == 2208,
    ))
    if compatible:
        if not raw_path.exists() or existing.get("source_fingerprint") == _fingerprint(raw_path):
            print("Replay cache ready: 2,208 verified episodes (offline playback available).", flush=True)
            return target
    print("Preparing recorded replay: verifying the pinned source checksum…", flush=True)
    if not raw_path.exists():
        acquire(raw_dir, expected)
    actual_hash = sha256(raw_path)
    if actual_hash != expected_hash:
        raise ValueError("Source checksum changed: sustainable_behavior.csv; inspect before updating the manifest")
    fingerprint = _fingerprint(raw_path)
    if compatible:
        existing["source_fingerprint"] = fingerprint
        with closing(sqlite3.connect(target)) as database:
            with database:
                database.execute("UPDATE metadata SET value=? WHERE key='provenance'", (_json(existing),))
        print("Replay source reverified; reusing its indexed cache.", flush=True)
        return target
    with SUMMARY_PATH.open(newline="") as source:
        summary_rows = list(csv.DictReader(source))
    summaries = {(row["cohort"], row["mechanism"], row["launch_id"]): row for row in summary_rows}
    if len(summaries) != 2208 or len(summary_rows) != 2208:
        raise ValueError("Saved Figure 2A group summary must contain 2,208 unique games")
    with tempfile.NamedTemporaryFile(prefix="preparing-", suffix=".sqlite3", dir=cache_dir, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    database = sqlite3.connect(temporary_path)
    try:
        database.execute("CREATE TABLE staging (id TEXT, round INTEGER, row TEXT, PRIMARY KEY(id, round))")
        database.execute("CREATE TABLE episodes (id TEXT PRIMARY KEY, condition TEXT, launch_id TEXT, episode_id TEXT, metadata TEXT, body BLOB)")
        database.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        selected_rows = 0
        print("Streaming the source once into an indexed replay cache…", flush=True)
        with raw_path.open(newline="") as source:
            for source_index, row in enumerate(csv.DictReader(source), start=1):
                condition = row["mech_name_by_player"]
                if condition in SELECTED:
                    raw = {name: row.get(name, "") for name in RAW_FIELDS}
                    identifier = episode_id(condition, row["launch_id"], row["episode_id"])
                    try:
                        database.execute("INSERT INTO staging VALUES (?, ?, ?)", (identifier, int(row["round_id"]), _json(raw)))
                    except sqlite3.IntegrityError as error:
                        raise ValueError(f"Duplicate full episode/round key at CSV data row {source_index}") from error
                    selected_rows += 1
                if source_index % 40000 == 0:
                    print(f"  Read {source_index:,} source rows; retained {selected_rows:,} Figure 2A rounds.", flush=True)
        if _fingerprint(raw_path) != fingerprint:
            raise ValueError("Source changed during replay preparation; retry with the pinned source")
        if selected_rows != 2208 * ROUNDS:
            raise ValueError(f"Expected 88,320 Figure 2A rows, found {selected_rows:,}")
        counts = Counter()
        maximum_difference = 0.0
        episode_ids = [row[0] for row in database.execute("SELECT DISTINCT id FROM staging")]
        summary_keys = set()
        for identifier in episode_ids:
            records = [json.loads(row[0]) for row in database.execute("SELECT row FROM staging WHERE id=? ORDER BY round", (identifier,))]
            episode = _recorded_episode(records)
            maximum_difference = max(maximum_difference, _validate_summary(episode, summaries))
            _, _, raw_mechanism, raw_cohort = SELECTED[episode["condition"]]
            summary_keys.add((raw_cohort, raw_mechanism, episode["launch_id"]))
            counts[episode["condition"]] += 1
            metadata = {key: value for key, value in episode.items() if key != "rounds"}
            database.execute("INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?)", (
                identifier, episode["condition"], episode["launch_id"], episode["episode_id"],
                _json(metadata), zlib.compress(_json(episode).encode(), level=6),
            ))
        for condition, (cohort, _, _, _) in SELECTED.items():
            if counts[condition] != (40 if cohort == "human" else 512):
                raise ValueError(f"Unexpected episode count for {condition}: {counts[condition]}")
        if summary_keys != set(summaries):
            raise ValueError("Replay episodes do not cover all saved Figure 2A group summaries")
        provenance = {
            "schema_version": SCHEMA_VERSION, "source_file": SOURCE_NAME,
            "source_url": SOURCES[SOURCE_NAME], "source_sha256": actual_hash,
            "source_fingerprint": fingerprint, "summary_sha256": summary_hash,
            "upstream_revision": REVISION, "paper_doi": "10.1038/s41467-025-58043-7",
            "data_license": "CC-BY-4.0", "episode_count": len(episode_ids),
            "round_count": selected_rows, "counts_by_condition": dict(counts),
            "summary_max_abs_difference": maximum_difference,
            "summary_tolerance": TIGHT_TOLERANCE,
            "equation_diagnostic_tolerance": REPLAY_TOLERANCE,
            "default_selection": "Human Experiment 1 Equal; nearest condition median surplus; ties by full (condition, launch_id, episode_id) key.",
            "cumulative_basis": "Sum of recorded player_reward_i from round 0 through the selected round; not rounded display values.",
        }
        database.execute("INSERT INTO metadata VALUES ('provenance', ?)", (_json(provenance),))
        database.execute("DROP TABLE staging")
        database.commit()
        database.execute("VACUUM")
        database.close()
        temporary_path.replace(target)
        print(f"Replay ready: {len(episode_ids):,} episodes / {selected_rows:,} rounds; all saved game summaries agree within 1e-9.", flush=True)
        return target
    finally:
        database.close()
        temporary_path.unlink(missing_ok=True)


def catalog(cache_path: Path) -> dict:
    """Return small game metadata; round histories load only when selected."""
    with closing(_open(Path(cache_path))) as database:
        episodes = [json.loads(row[0]) for row in database.execute("SELECT metadata FROM episodes ORDER BY condition, launch_id, episode_id")]
        provenance = json.loads(database.execute("SELECT value FROM metadata WHERE key='provenance'").fetchone()[0])
    equal = [episode for episode in episodes if episode["cohort"] == "human" and episode["mechanism"] == "Equal"]
    # Exact rationals of stored floats keep the two central games equidistant
    # for an even cohort; binary midpoint rounding must not break a key tie.
    median = statistics.median(Fraction(episode["surplus"]) for episode in equal)
    default = min(equal, key=lambda episode: (
        abs(Fraction(episode["surplus"]) - median), episode["condition"], episode["launch_id"], episode["episode_id"],
    ))
    return {"episodes": episodes, "default_id": default["id"], "provenance": provenance}


def get_episode(cache_path: Path, identifier: str) -> dict:
    with closing(_open(Path(cache_path))) as database:
        row = database.execute("SELECT body FROM episodes WHERE id=?", (identifier,)).fetchone()
    if row is None:
        raise KeyError(f"Unknown replay episode: {identifier}")
    return json.loads(zlib.decompress(row[0]))


def sandbox(mechanism: str, fractions: list[float]) -> dict:
    """Start a new fixed-fraction run through the existing Python environment."""
    if not isinstance(mechanism, str) or mechanism.lower() not in {"equal", "mixed", "proportional", "interpolating"}:
        raise ValueError("Sandbox mechanism must be Equal, Mixed, Proportional, or Interpolating")
    if not isinstance(fractions, (list, tuple)) or len(fractions) != 4:
        raise ValueError("Provide exactly four return fractions between 0 and 1")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in fractions):
        raise ValueError("Return fractions must be finite numbers between 0 and 1")
    fractions = [float(value) for value in fractions]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in fractions):
        raise ValueError("Return fractions must be finite numbers between 0 and 1")
    # Treat the entered decimal fraction exactly, preserving the environment's
    # float offer. A binary product such as 0.58 * 50 would otherwise floor to
    # 28 instead of the intended 29 units.
    decimal_fractions = [Fraction(str(value)) for value in fractions]
    mechanism = mechanism.lower()
    state = WorldState()
    rounds = []
    history = [[], [], [], []]
    while not state.done:
        offers = allocate(state.pool, state.previous_contributions, mechanism=mechanism)
        contributions = [
            math.floor(fraction * Fraction.from_float(offer))
            for fraction, offer in zip(decimal_fractions, offers)
        ]
        next_state, result = state.advance(offers, contributions, integer_contributions=True)
        for player, value in enumerate(result.surplus):
            history[player].append(value)
        cumulative = [math.fsum(values) for values in history]
        rounds.append({
            "round_id": state.round_index, "pool_before": state.pool,
            "offers": list(offers), "contributions": contributions,
            "surplus": list(result.surplus), "cumulative_surplus": cumulative,
            "group_cumulative_surplus": math.fsum(cumulative),
            "pool_after": next_state.pool,
            "after_source": "published equation (new scripted simulation)",
            "pool_after_raw": None, "equation_after": next_state.pool,
            "equation_input_valid": True, "pool_residual": 0.0,
            "active_count": sum(offer >= 1 for offer in offers), "raw": {},
        })
        state = next_state
    means = [math.fsum(values) / len(rounds) for values in history]
    return {
        "id": "scripted-" + hashlib.sha256(_json([mechanism, fractions]).encode()).hexdigest(),
        "condition": "Fixed-fraction scripted policy", "cohort": "scripted",
        "mechanism": mechanism.capitalize(), "launch_id": None, "episode_id": None,
        "surplus": math.fsum(means) / 4, "gini": _gini(means),
        "round_count": len(rounds), "rounds": rounds, "fractions": fractions,
        "termination": "exact-zero pool" if state.pool == 0 else "40-round limit",
        "provenance_label": "New scripted simulation · fixed fractions · no learning",
        "policy": "Each resident returns floor(q_i × offer_i), with q_i interpreted as the entered decimal and the numerical allocation preserved. Integer units; no fitting or learning.",
        "surplus_basis": "Mean per resident per completed simulated round; exact-zero runs are not padded to 40 rounds.",
    }
