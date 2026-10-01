"""Read frozen Task 03 trajectories without importing a training framework.

The generated archive is a published result, not a mutable viewer cache.  The
local service reads selected episodes and never samples or updates weights.
"""

from contextlib import closing
from fractions import Fraction
import json
from pathlib import Path
import sqlite3
import zlib

ROOT = Path(__file__).resolve().parents[1]
GENERATED_PATH = ROOT / "results/task03/generated.sqlite3"


def _open(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def generated_catalog(path=GENERATED_PATH):
    if not Path(path).exists():
        return {"episodes": [], "default_id": None, "provenance": {}}
    with closing(_open(path)) as database:
        episodes = [json.loads(row[0]) for row in database.execute("SELECT metadata FROM episodes ORDER BY id")]
        metadata = {key: json.loads(value) for key, value in database.execute("SELECT key,value FROM metadata")}
    default = metadata.get("default_id")
    if default is None:
        candidates = [e for e in episodes if e["family"] == "recurrent" and e["training_seed"] == 17 and e["mechanism"] == "Equal"]
        if candidates:
            ordered = sorted(Fraction(e["surplus"]) for e in candidates)
            median = (ordered[(len(ordered)-1)//2] + ordered[len(ordered)//2]) / 2
            key = lambda e: (abs(Fraction(e["surplus"]) - median), e["rollout_index"], e["id"])
            default = min(candidates, key=key)["id"]
    return {"episodes": episodes, "default_id": default, "provenance": metadata.get("provenance", {})}


def generated_episode(identity, path=GENERATED_PATH):
    if not Path(path).exists():
        raise KeyError("Trained community archive is not available")
    with closing(_open(path)) as database:
        row = database.execute("SELECT metadata,body FROM episodes WHERE id=?", (identity,)).fetchone()
    if row is None:
        raise KeyError("Unknown trained community")
    return {**json.loads(row[0]), **json.loads(zlib.decompress(row[1]))}
