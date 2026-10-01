"""Frozen Task 03 human prediction evaluation and editable research figures.

The unit of uncertainty is an interacting human group. Optimization seeds are
averaged within that group before paired, mechanism-stratified resampling.
No simulation outcome enters checkpoint or model selection.
"""

from collections import defaultdict
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np


FAMILIES = ("constant", "linear", "feedforward", "recurrent")
SEEDS = (17, 29, 43)
PROBABILITY_BINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
OFFER_BINS = (1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 201.0)
BOOTSTRAP_SEED = 20261003
BOOTSTRAP_REPLICATES = 2000
METRICS = ("nll", "mae", "relative_mae", "p_zero", "observed_zero", "p_max", "observed_max")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No measured rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def score_groups(raw, n, targets, offers, groups, family, seed, split):
    """Return group metrics and nonforced endpoint calibration records.

    Arguments use complete [group, player, round] sequences; ``raw`` has an
    additional five-output axis. Offers remain source float64, independent of
    normalized float32 neural inputs. Passing raw=None scores the legal uniform
    reference. No forced choice is pooled into the primary diagnostics.
    """
    import torch
    from .behavior_models import emission_log_prob, emission_stats

    legal = np.asarray(n)
    actual = np.asarray(targets)
    allocations = np.asarray(offers, dtype=np.float64)
    if raw is None:
        losses = np.log(legal + 1)
        expected = legal / 2.0
        p_zero = p_max = 1.0 / (legal + 1)
    else:
        nt = torch.as_tensor(legal)
        ct = torch.as_tensor(actual)
        with torch.no_grad():
            losses = -emission_log_prob(raw, nt, ct).detach().cpu().numpy()
            stats = emission_stats(raw, nt)
        expected, p_zero, p_max = (stats[key].detach().cpu().numpy() for key in ("mean", "p_zero", "p_max"))
    if not all(np.isfinite(value).all() for value in (losses, expected, p_zero, p_max)):
        raise ValueError("Nonfinite frozen predictions")
    rows, calibration = [], []
    mechanism_counts = defaultdict(int)
    for group in groups:
        mechanism_counts[group["mechanism"]] += 1
    for index, group in enumerate(groups):
        mask = legal[index] >= 1
        count = int(mask.sum())
        if not count:
            raise ValueError(f"Group with no nonforced choices: {group['key']}")
        c, e, maximum = actual[index][mask], allocations[index][mask], legal[index][mask]
        prediction = expected[index][mask]
        zero, full = p_zero[index][mask], p_max[index][mask]
        observed_zero, observed_max = (c == 0).astype(float), (c == maximum).astype(float)
        row = {
            "family": family, "seed": seed, "split": split,
            "group_index": group["index"], "group_key": json.dumps(group["key"], separators=(",", ":")),
            "mechanism": group["mechanism"], "nonforced_choices": count,
            "forced_choices": int(mask.size - count),
            "nll": float(losses[index][mask].mean()),
            "mae": float(np.abs(prediction - c).mean()),
            "relative_mae": float((np.abs(prediction - c) / e).mean()),
            "p_zero": float(zero.mean()), "observed_zero": float(observed_zero.mean()),
            "p_max": float(full.mean()), "observed_max": float(observed_max.mean()),
        }
        rows.append(row)
        weight = 1.0 / (len(mechanism_counts) * mechanism_counts[group["mechanism"]] * count)
        for endpoint, predicted, observed in (("zero", zero, observed_zero), ("maximum", full, observed_max)):
            for dimension, values, edges in (("probability", predicted, PROBABILITY_BINS), ("offer", e, OFFER_BINS)):
                for bin_index, (lower, upper) in enumerate(zip(edges, edges[1:])):
                    selected = (values >= lower) & ((values <= upper) if bin_index == len(edges) - 2 else (values < upper))
                    size = int(selected.sum())
                    if size:
                        calibration.append({
                            "family": family, "seed": seed, "split": split,
                            "group_index": group["index"], "mechanism": group["mechanism"],
                            "endpoint": endpoint, "dimension": dimension, "bin": bin_index,
                            "lower": lower, "upper": upper, "choices": size,
                            "weight": size * weight,
                            "predicted_weighted_sum": float(predicted[selected].sum() * weight),
                            "observed_weighted_sum": float(observed[selected].sum() * weight),
                        })
    return rows, calibration


def calibration_summary(records: list[dict]) -> list[dict]:
    """Seed-averaged bins, retaining the declared human group weights.

    A bin is a weighted conditional diagnostic, so its weight need not be the
    same as another bin's weight. Every family has total probability-bin weight
    one for each endpoint. All fitted seeds contribute equally.
    """
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["family"], row["split"], row["endpoint"], row["dimension"], row["bin"])].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        family, split, endpoint, dimension, bin_index = key
        seeds = {row["seed"] for row in records if row["family"] == family and row["split"] == split}
        weight = math.fsum(row["weight"] for row in rows)
        output.append({
            "family": family, "split": split, "endpoint": endpoint,
            "dimension": dimension, "bin": bin_index,
            "lower": rows[0]["lower"], "upper": rows[0]["upper"],
            "weighted_mass": weight / len(seeds),
            "predicted": math.fsum(row["predicted_weighted_sum"] for row in rows) / weight,
            "observed": math.fsum(row["observed_weighted_sum"] for row in rows) / weight,
            "groups_represented": len({row["group_index"] for row in rows}),
            "choice_seed_observations": sum(row["choices"] for row in rows),
            "optimization_seeds": len(seeds),
        })
    return output


def summarize_predictions(rows: list[dict], *, bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED) -> dict:
    """Paired uncertainty after averaging optimization seeds within each group."""
    mechanisms = sorted({row["mechanism"] for row in rows})
    families = [family for family in (*FAMILIES, "uniform") if any(row["family"] == family for row in rows)]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["mechanism"], row["group_index"])].append(row)
    group_ids = {mechanism: sorted({row["group_index"] for row in rows if row["mechanism"] == mechanism}) for mechanism in mechanisms}
    if len({len(value) for value in group_ids.values()}) != 1:
        raise ValueError("Task 03 requires equally sized mechanism strata")
    values = np.empty((len(families), len(mechanisms), len(group_ids[mechanisms[0]]), len(METRICS)), dtype=np.float64)
    per_seed = []
    for f, family in enumerate(families):
        expected_seeds = {"reference"} if family == "uniform" else set(SEEDS)
        for m, mechanism in enumerate(mechanisms):
            for g, group_id in enumerate(group_ids[mechanism]):
                group_rows = grouped[(family, mechanism, group_id)]
                if {row["seed"] for row in group_rows} != expected_seeds or len(group_rows) != len(expected_seeds):
                    raise ValueError(f"Missing/duplicate optimization seeds: {family} {group_id}")
                values[f, m, g] = [np.mean([row[metric] for row in group_rows]) for metric in METRICS]
        for seed in sorted(expected_seeds, key=str):
            selected = [row for row in rows if row["family"] == family and row["seed"] == seed]
            for mechanism in (*mechanisms, "all"):
                target = [row for row in selected if mechanism == "all" or row["mechanism"] == mechanism]
                per_seed.append({"family": family, "seed": seed, "mechanism": mechanism,
                                 **{metric: float(np.mean([row[metric] for row in target])) for metric in METRICS}})
    # PCG64 and the same group indices in a replicate for every fitted family.
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    draws = rng.integers(values.shape[2], size=(bootstrap_replicates, len(mechanisms), values.shape[2]))
    bootstrap = np.empty((bootstrap_replicates, len(families), len(METRICS)))
    for replicate, indices in enumerate(draws):
        bootstrap[replicate] = np.stack([values[:, m, indices[m], :].mean(axis=1) for m in range(len(mechanisms))]).mean(axis=0)
    summary = []
    for f, family in enumerate(families):
        for mechanism in (*mechanisms, "all"):
            mean = values[f].mean(axis=(0, 1)) if mechanism == "all" else values[f, mechanisms.index(mechanism)].mean(axis=0)
            result = {"family": family, "mechanism": mechanism,
                      **{metric: float(mean[k]) for k, metric in enumerate(METRICS)}}
            if mechanism == "all":
                result["ci95"] = {metric: np.quantile(bootstrap[:, f, k], [0.025, 0.975]).tolist() for k, metric in enumerate(METRICS)}
                fitted_seed_scores = [row["nll"] for row in per_seed if row["family"] == family and row["mechanism"] == "all"]
                result["seed_nll_minmax"] = [min(fitted_seed_scores), max(fitted_seed_scores)]
            summary.append(result)
    comparisons = []
    if "recurrent" in families:
        recurrent = families.index("recurrent")
        for comparator in ("feedforward", "linear", "constant", "uniform"):
            if comparator not in families:
                continue
            other = families.index(comparator)
            difference = values[recurrent, :, :, 0] - values[other, :, :, 0]
            differences = bootstrap[:, recurrent, 0] - bootstrap[:, other, 0]
            comparisons.append({"comparison": f"recurrent - {comparator}",
                                "nll_difference": float(difference.mean()),
                                "ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
                                "by_mechanism": {mechanism: float(difference[m].mean()) for m, mechanism in enumerate(mechanisms)}})
    return {"aggregation": "Nonforced choices within group, groups within mechanism, equal mechanisms; seeds averaged within group before uncertainty",
            "bootstrap": {"replicates": bootstrap_replicates, "seed": bootstrap_seed,
                          "generator": "numpy.random.Generator(PCG64)", "paired": True,
                          "mechanism_stratified": True, "group_ids": group_ids},
            "summary": summary, "per_seed": per_seed, "paired_comparisons": comparisons}


def plot_prediction_results(summary: dict, calibration: list[dict], logs: dict, directory: Path) -> None:
    """Actual measured curves/uncertainty; smooth plot geometry is preserved."""
    from .plotting import BACKGROUND, PANEL, PARCHMENT, MUTED, COLORS
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "DejaVu Sans Mono", "font.size": 9,
             "text.color": PARCHMENT, "axes.labelcolor": PARCHMENT,
             "xtick.color": MUTED, "ytick.color": MUTED,
             "svg.fonttype": "none", "svg.hashsalt": "evopolis-task03"}

    def decorate(axes):
        for ax in np.asarray(axes).reshape(-1):
            ax.set_facecolor(PANEL)
            ax.grid(color=MUTED, alpha=0.16, linewidth=0.6)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_color(MUTED)

    def save(fig, name):
        fig.savefig(directory / f"{name}.png", dpi=170, facecolor=BACKGROUND)
        fig.savefig(directory / f"{name}.svg", metadata={"Date": None}, facecolor=BACKGROUND)
        svg = directory / f"{name}.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        plt.close(fig)

    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        fig.patch.set_facecolor(BACKGROUND)
        decorate(axes)
        for family, ax, color in zip(FAMILIES, axes.flat, COLORS):
            for seed in SEEDS:
                entries = logs[(family, seed)]
                epochs = [row["epoch"] for row in entries]
                ax.plot(epochs, [row["train_nll"] for row in entries], color=color, alpha=0.45, linewidth=1, linestyle="--")
                ax.plot(epochs, [row["validation_nll"] for row in entries], color=color, alpha=0.85, linewidth=1)
            ax.set_title(family.upper(), color=PARCHMENT, loc="left", fontweight="bold")
            ax.set_xlabel("Training epoch")
            ax.set_ylabel("Nonforced NLL (nats / choice)")
        fig.suptitle("EVOPOLIS / PARAMETER TRAINING", fontsize=16, x=0.08, ha="left", fontweight="bold")
        fig.text(0.08, 0.04, "Three optimization seeds. Dashed: online epoch training loss. Solid: end-epoch validation. Lower is better.\nComplete groups; equal mechanism weights. Checkpoints selected on validation only.", color=MUTED, fontsize=9)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.17, hspace=0.35, wspace=0.2)
        save(fig, "task03-learning-curves")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
        fig.patch.set_facecolor(BACKGROUND)
        decorate(axes)
        overall = [row for row in summary["summary"] if row["mechanism"] == "all"]
        families = [row["family"] for row in overall]
        ax = axes[0]
        for index, row in enumerate(overall):
            lower, upper = row["ci95"]["nll"]
            ax.errorbar(row["nll"], index, xerr=[[row["nll"] - lower], [upper - row["nll"]]],
                        fmt="s", color=COLORS[index] if index < 4 else MUTED, capsize=4)
        ax.set_yticks(range(len(families)), ["GRU" if family == "recurrent" else family.title() for family in families])
        ax.invert_yaxis()
        ax.set_xlabel("Nonforced NLL (nats / choice)")
        ax.set_title("HELD-OUT HUMAN PREDICTION", loc="left", color=PARCHMENT, fontsize=10)
        for ax, endpoint in zip(axes[1:], ("zero", "maximum")):
            ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
            for family, color in zip(FAMILIES, COLORS):
                selected = [row for row in calibration if row["family"] == family and row["endpoint"] == endpoint and row["dimension"] == "probability"]
                ax.plot([row["predicted"] for row in selected], [row["observed"] for row in selected], "s-", color=color, markersize=4, label="GRU" if family == "recurrent" else family.title())
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Observed fraction")
            ax.set_title(f"{endpoint.upper()} RETURN CALIBRATION", loc="left", color=PARCHMENT, fontsize=10)
        axes[-1].legend(frameon=False, fontsize=8, labelcolor=PARCHMENT)
        fig.suptitle("EVOPOLIS / TEST GROUPS", fontsize=16, x=0.1, ha="left", fontweight="bold")
        fig.text(0.1, 0.055, "32 human groups; 8 per mechanism. Means across 3 fitted seeds, then 2,000 paired group bootstrap replicates.\nIntervals: marginal 95% bootstrap. Calibration excludes forced returns; fixed probability bins with group weights.", color=MUTED, fontsize=8.5)
        fig.subplots_adjust(left=0.1, right=0.98, top=0.83, bottom=0.24, wspace=0.46)
        save(fig, "task03-prediction")


def open_frozen_test(results: Path) -> tuple[list[dict], dict]:
    """Verify the complete declared fits, then persist the first opening receipt.

    This function does not read test outcomes. Re-evaluation must preserve the
    original opening timestamp and the same frozen checkpoint identities.
    """
    import torch
    from .behavior_data import CONFIG_PATH, ROOT, read_config, write_json
    from .sources import sha256

    configuration = read_config()
    if tuple(configuration["probability_bin_edges"]) != PROBABILITY_BINS or tuple(configuration["offer_bin_edges"]) != OFFER_BINS:
        raise ValueError("Evaluation bins differ from the pre-training declaration")
    if configuration["bootstrap_replicates"] != BOOTSTRAP_REPLICATES or configuration["bootstrap_seed"] != BOOTSTRAP_SEED:
        raise ValueError("Bootstrap differs from the pre-training declaration")
    frozen = json.loads((results / "frozen_training.json").read_text())
    identity = frozen["identity"]
    if identity["config_sha256"] != sha256(CONFIG_PATH) or identity["split_sha256"] != sha256(results / "split.json"):
        raise ValueError("Frozen split/configuration changed before evaluation")
    for name, digest in identity["code_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Training/data code changed after freezing: {name}")
    complete = json.loads((results / "training_complete.json").read_text())
    if complete["fits"] != 12 or complete["epochs_per_fit"] != 120:
        raise ValueError("All twelve declared fits must finish before test opening")
    checkpoints, logs = [], {}
    for family in FAMILIES:
        for seed in SEEDS:
            directory = results / "weights" / f"{family}_{seed}"
            entries = json.loads((directory / "training.json").read_text())
            if [row["epoch"] for row in entries] != list(range(1, 121)):
                raise ValueError(f"Incomplete training history: {family}/{seed}")
            selected = min(entries, key=lambda row: (row["validation_nll"], row["epoch"]))
            best = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
            last = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
            metadata = json.loads((directory / "metadata.json").read_text())
            if best["family"] != family or best["seed"] != seed or last["family"] != family or last["seed"] != seed or best["metadata"] != identity or last["metadata"] != identity:
                raise ValueError(f"Checkpoint identity mismatch: {family}/{seed}")
            if last["epoch"] != 120 or best["epoch"] != selected["epoch"] or best["validation_score"] != selected["validation_nll"]:
                raise ValueError(f"Checkpoint selection is not earliest best validation: {family}/{seed}")
            for name in ("best", "last"):
                if metadata[f"{name}_sha256"] != sha256(directory / f"{name}.pt"):
                    raise ValueError(f"Checkpoint hash mismatch: {family}/{seed}/{name}")
            checkpoints.append({"family": family, "seed": seed, "selected_epoch": best["epoch"],
                                "best_sha256": metadata["best_sha256"], "last_sha256": metadata["last_sha256"],
                                "validation_nll": best["validation_score"]})
            logs[(family, seed)] = entries
    contract = {"training_identity": identity, "checkpoints": checkpoints,
                "evaluation_code_sha256": sha256(Path(__file__)),
                "generation_code_sha256": {name: sha256(ROOT / name) for name in
                                           ("evopolis/behavior_generate.py", "evopolis/world.py")},
                "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "probability_bin_edges": list(PROBABILITY_BINS), "offer_bin_edges": list(OFFER_BINS)}
    receipt_path = results / "test_opening.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["frozen_contract"] != contract:
            raise ValueError("Evaluation contract/checkpoints differ from first test opening; document a correctness repair before rerunning")
    else:
        receipt = {"opened_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "status": "Held out from Task 03 fitting and selection; historically described in Task 01. Outcomes consume this endpoint for later model search.",
                   "frozen_contract": contract}
        write_json(receipt_path, receipt)
    return checkpoints, logs


def evaluate(results: Path, figures: Path) -> dict:
    import torch
    from .behavior_data import load, write_json
    from .behavior_models import BehaviorModel

    torch.set_num_threads(1)
    checkpoints, logs = open_frozen_test(results)
    # Outcomes are touched only after the immutable opening receipt is present.
    arrays, manifest = load()
    all_rows, all_calibration, learning = [], [], []
    for split in ("validation", "test"):
        groups = [group for group in manifest["groups"] if group["split"] == split]
        indices = [group["index"] for group in groups]
        x = torch.tensor(arrays["x"][indices], dtype=torch.float32)
        n, y, offers = (arrays[name][indices] for name in ("n", "y", "offers"))
        for specification in checkpoints:
            family, seed = specification["family"], specification["seed"]
            directory = results / "weights" / f"{family}_{seed}"
            checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
            model = BehaviorModel(family).eval()
            model.load_state_dict(checkpoint["model_state"])
            predictions = []
            with torch.no_grad():
                for offset in range(0, len(groups), 2):
                    batch = x[offset:offset + 2]
                    raw, _ = model(batch.reshape(-1, 40, 9))
                    predictions.append(raw.reshape(len(batch), 4, 40, 5))
            raw = torch.cat(predictions)
            rows, calibration = score_groups(raw, n, y, offers, groups, family, seed, split)
            if split == "validation":
                score = float(np.mean([row["nll"] for row in rows]))
                if not math.isclose(score, specification["validation_nll"], rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(f"CPU reload did not reproduce validation prediction: {family}/{seed}")
                metadata = json.loads((directory / "metadata.json").read_text())
                learning.append({"family": family, "seed": seed, "parameter_count": metadata["parameter_count"],
                                 "selected_epoch": specification["selected_epoch"],
                                 "initial_validation_nll": metadata["initial_validation_nll"],
                                 "fitted_validation_nll": score,
                                 "nll_improvement": metadata["initial_validation_nll"] - score})
            all_rows.extend(rows)
            all_calibration.extend(calibration)
            print(f"Evaluated {split}: {family}/{seed}, {len(groups)} whole human groups", flush=True)
        rows, calibration = score_groups(None, n, y, offers, groups, "uniform", "reference", split)
        all_rows.extend(rows)
        all_calibration.extend(calibration)
    test_rows = [row for row in all_rows if row["split"] == "test"]
    summary = summarize_predictions(test_rows)
    calibration = calibration_summary(all_calibration)
    write_csv(results / "prediction_groups.csv", all_rows)
    write_csv(results / "calibration.csv", calibration)
    write_csv(results / "learning.csv", learning)
    write_json(results / "prediction_summary.json", summary)
    test_calibration = [row for row in calibration if row["split"] == "test"]
    plot_prediction_results(summary, test_calibration, logs, figures)
    return summary


def checkpoint_prediction_digests(results: Path) -> list[dict]:
    """Repeat in separate CPU processes to check exact reload reproducibility.

    Uses only the deterministic first validation group, never test outcomes.
    The digest covers all five emission outputs over all four resident histories.
    """
    import torch
    from .behavior_data import load
    from .behavior_models import BehaviorModel

    torch.set_num_threads(1)
    arrays, manifest = load()
    group = next(group for group in manifest["groups"] if group["split"] == "validation")
    x = torch.tensor(arrays["x"][group["index"]], dtype=torch.float32)
    output = []
    for family in FAMILIES:
        for seed in SEEDS:
            checkpoint = torch.load(results / "weights" / f"{family}_{seed}" / "best.pt", map_location="cpu", weights_only=True)
            model = BehaviorModel(family).eval()
            model.load_state_dict(checkpoint["model_state"])
            with torch.no_grad():
                prediction, _ = model(x)
            digest = hashlib.sha256(prediction.detach().numpy().tobytes()).hexdigest()
            output.append({"family": family, "seed": seed, "epoch": checkpoint["epoch"],
                           "validation_group": group["key"], "prediction_sha256": digest})
    return output


def main():
    from .behavior_data import RESULTS_DIR, ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures", type=Path, default=ROOT / "docs/assets")
    parser.add_argument("--prediction-digests", action="store_true", help="Hash CPU predictions on one validation group without opening test outcomes")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if args.prediction_digests:
        print(json.dumps(checkpoint_prediction_digests(args.results), indent=2))
        return
    result = evaluate(args.results, args.figures)
    for row in result["summary"]:
        if row["mechanism"] == "all":
            print(f"{row['family']}: test NLL {row['nll']:.6f} 95% CI {row['ci95']['nll']}")


if __name__ == "__main__":
    main()
