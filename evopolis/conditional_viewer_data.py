"""Read saved Task 05 forecasts and resident diagnostics without fitting models."""

from contextlib import closing
import json
from pathlib import Path
import zlib

from .forecast_viewer_data import _open, forecast_catalog, forecast_episode

CONDITIONAL_PATH = Path(__file__).resolve().parents[1] / "results/task05/forecasts.sqlite3"
FAMILIES = ("P0", "P1", "H0", "H1")


def conditional_catalog(path=CONDITIONAL_PATH):
    catalog = forecast_catalog(path)
    for cell in catalog["cells"]:
        if cell["family"] not in FAMILIES:
            raise ValueError("Unexpected conditional-response model family")
        cell.update(study="task05", budget="conditional", budget_epochs=480)
    catalog["default_rule"] = (
        "Task 05 uses the same recorded human boundaries as Task 04. "
        "Branch zero is the deterministic illustration; all 64 branches form the bands."
    )
    return catalog


def conditional_episode(identity, branch="first", path=CONDITIONAL_PATH):
    result = forecast_episode(identity, branch, path)
    with closing(_open(path)) as database:
        row = database.execute("SELECT body FROM cells WHERE id=?", (identity,)).fetchone()
    body = json.loads(zlib.decompress(row[0]))
    selected = body["selected_branch"] if branch == "median" else body.get("branch_zero")
    if selected is None:
        alternatives = body.get("alternatives", [])
        if isinstance(alternatives, dict):
            alternatives = alternatives.values()
        selected = next((item for item in alternatives if item["branch_index"] == 0),
                        body["selected_branch"])
    for episode in (result["forecast"], result["observed"]):
        if episode["family"] not in FAMILIES:
            raise ValueError("Unexpected conditional-response model family")
        episode.update(study="task05", budget="conditional", budget_epochs=480)
        episode["parameters"] = selected.get("parameters", body.get("parameters", {}))
        episode["prefix_posteriors"] = selected.get("prefix_posteriors", body.get("prefix_posteriors", []))
    result["forecast"]["latent_effects"] = selected.get("latent_effects", [])
    return result
