"""Read published Task 04 forecasts without importing a learning framework.

Only recorded-boundary principal-bank forecasts are the town's default study.
Saved illustrative branches do not replace the distribution of all 64 draws.
"""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import zlib

ROOT = Path(__file__).resolve().parents[1]
FORECAST_PATH = ROOT / "results/task04/forecasts.sqlite3"


def _open(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def _principal(metadata):
    return (metadata.get("bank") == "main"
            and metadata.get("boundary", metadata.get("convention")) == "recorded")


def _identity(metadata):
    result = dict(metadata)
    key = result.get("key", [])
    if len(key) == 3:
        for field, value in zip(("condition", "launch_id", "episode_id"), key):
            result.setdefault(field, str(value))
    result.setdefault("training_seed", result.get("seed"))
    result.setdefault("budget_epochs", 480 if result.get("budget") == "continued" else 120)
    return result


def forecast_catalog(path=FORECAST_PATH):
    if not Path(path).exists():
        return {"cells": [], "default_id": None}
    with closing(_open(path)) as database:
        cells = [_identity(json.loads(row[0])) for row in database.execute(
            "SELECT metadata FROM cells ORDER BY id")]
    cells = [cell for cell in cells if _principal(cell)]
    preferred = [cell for cell in cells if cell["mechanism"] == "Mixed"
                 and cell["family"] == "recurrent" and cell["training_seed"] == 17
                 and cell["budget"] == "continued" and cell["origin"] == 5]
    candidates = preferred or cells
    default = min(candidates, key=lambda cell: (cell.get("key", []), cell["id"]))["id"] if candidates else None
    return {"cells": cells, "default_id": default,
            "default_rule": "First sorted Mixed human group; continued GRU, training seed 17, origin k=5, branch zero. The optional branch nearest the median final-window surplus is illustrative, not selected for accuracy."}


def forecast_episode(identity, branch="first", path=FORECAST_PATH):
    if branch not in ("median", "first"):
        raise ValueError("Choose the median illustration or branch zero")
    if not Path(path).exists():
        raise KeyError("Forecast archive is not available")
    with closing(_open(path)) as database:
        row = database.execute("SELECT metadata,body FROM cells WHERE id=?", (identity,)).fetchone()
    if row is None:
        raise KeyError("Unknown forecast condition")
    metadata, body = _identity(json.loads(row[0])), json.loads(zlib.decompress(row[1]))
    if not _principal(metadata):
        raise ValueError("The town shows recorded-boundary principal-bank forecasts")
    origin, horizon = int(metadata["origin"]), int(metadata["horizon"])
    if origin not in (0, 5, 10, 20) or horizon != 20 or origin + horizon > 40:
        raise ValueError("Forecast exceeds the declared recorded episode window")
    observed = body["observed_rounds"]
    selected = body["selected_branch"]
    if branch == "first":
        alternatives = body.get("alternatives", [])
        if isinstance(alternatives, dict):
            alternatives = list(alternatives.values())
        if body.get("branch_zero"):
            alternatives.append(body["branch_zero"])
        selected = next((item for item in alternatives if item["branch_index"] == 0),
                        selected if selected["branch_index"] == 0 else None)
        if selected is None:
            raise ValueError("Branch zero was not archived")
    generated = selected["rounds"]
    if len(observed) != 40 or len(generated) != horizon:
        raise ValueError("Incomplete published forecast")
    rows = [{**record, "observed": True, "padded": False} for record in observed[:origin]]
    rows += [{**record, "round_id": origin + index, "observed": False}
             for index, record in enumerate(generated)]
    if any(record["round_id"] != index for index, record in enumerate(rows)):
        raise ValueError("Nonconsecutive forecast rounds")
    actual_rounds = origin + sum(not record.get("padded", False) for record in generated)
    ending = "exact-zero termination" if actual_rounds < origin + horizon else "declared forecast window complete"
    base = {**metadata, "gini": None, "actual_rounds": actual_rounds,
            "termination": ending, "provenance": body.get("provenance", {}),
            "rollout_seed": str(selected["rollout_seed"]),
            "branch_index": selected["branch_index"], "branch_selection": branch,
            "forecast_end": origin + horizon, "bands": body["bands"]}
    forecast = {**base, "cohort": "forecast", "rounds": rows}
    human = {**base, "cohort": "forecast-observed", "id": f"{identity}-observed",
             "rounds": [{**record, "observed": True, "padded": False}
                        for record in observed[:origin + horizon]],
             "termination": "observed comparison window complete", "actual_rounds": origin + horizon}
    return {"forecast": forecast, "observed": human,
            "branch_rule": body.get("branch_selection", "Nearest median final-window surplus, ties by branch index"),
            "provenance": body.get("provenance", {})}
