"""Measured Task 04 graphics, kept separate from the frozen scoring code."""

import csv
import json
import numpy as np

from .behavior_data import ROOT, load, write_json
from .forecast_evaluate import (RESULTS, FIGURES, SEEDS, ORIGINS, HORIZONS, MECHANISMS, _overall)

def plot_results(results=RESULTS, figures=FIGURES):
    """Measured continuous axes in the established console palette, editable SVG."""
    from .plotting import BACKGROUND, PANEL, PARCHMENT, MUTED, COLORS
    from .forecast_generate import iter_cells
    import matplotlib.pyplot as plt

    one = json.loads((results / "one_step_summary.json").read_text())
    forecast = json.loads((results / "forecast_summary.json").read_text())
    figures.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "DejaVu Sans Mono", "font.size": 9, "text.color": PARCHMENT,
             "axes.labelcolor": PARCHMENT, "xtick.color": MUTED, "ytick.color": MUTED,
             "svg.fonttype": "none", "svg.hashsalt": "evopolis-task04"}

    def decorate(axes):
        for ax in np.asarray(axes).reshape(-1):
            ax.set_facecolor(PANEL)
            ax.grid(color=MUTED, alpha=0.16, linewidth=0.6)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_color(MUTED)

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", dpi=170, facecolor=BACKGROUND)
        fig.savefig(figures / f"{name}.svg", metadata={"Date": None}, facecolor=BACKGROUND)
        svg = figures / f"{name}.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        plt.close(fig)

    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.7))
        fig.patch.set_facecolor(BACKGROUND)
        decorate(axes)
        for ax, family, color in zip(axes[:2], ("feedforward", "recurrent"), COLORS[2:]):
            for seed in SEEDS:
                original = json.loads((ROOT / "results/task03/weights" / f"{family}_{seed}" / "training.json").read_text())
                continued = json.loads((results / "weights" / f"{family}_{seed}" / "training.json").read_text())
                entries = {row["epoch"]: row for row in [*original, *continued]}
                entries = [entries[key] for key in sorted(entries)]
                ax.plot([e["epoch"] for e in entries], [e["validation_nll"] for e in entries], color=color, alpha=0.55, linewidth=1)
            ax.axvline(120, color=MUTED, linestyle="--", linewidth=1)
            ax.set_title(f"{family.upper()} / VALIDATION", loc="left", color=PARCHMENT, fontsize=10)
            ax.set_xlabel("Total fitted epochs")
            ax.set_ylabel("Nonforced NLL (nats / choice)")
        ax = axes[2]
        labels = []
        for position, (family, population) in enumerate(( (family, population) for family in ("feedforward", "recurrent") for population in ("individual_32", "individual_24") )):
            comparison = next(row for row in one[population]["comparisons"] if row["family"] == family)
            delta = comparison["differences"]["nll"]
            low, high = comparison["ci95"]["nll"]
            ax.errorbar(delta, position, xerr=[[delta - low], [high - delta]], fmt="s", capsize=4, color=COLORS[2] if family == "feedforward" else COLORS[3])
            labels.append(f"{'FF' if family == 'feedforward' else 'GRU'} / {population[-2:]} groups")
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_yticks(range(4), labels)
        ax.invert_yaxis()
        ax.set_xlabel("Continued − original NLL")
        ax.set_title("HUMAN PREDICTION", loc="left", color=PARCHMENT, fontsize=10)
        fig.suptitle("EVOPOLIS / FOURFOLD OPTIMIZATION CONTROL", x=0.065, ha="left", fontweight="bold", fontsize=15)
        fig.text(0.065, 0.075, "Six fits complete epoch 480; selection uses validation only. Curves: three fitted seeds.\nBars: paired 95% human-group bootstrap intervals; opened Experiment 1 evidence. Lower is better.", color=MUTED)
        fig.subplots_adjust(left=0.065, right=0.98, top=0.84, bottom=0.24, wspace=0.5)
        save(fig, "task04-optimization")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.7))
        fig.patch.set_facecolor(BACKGROUND)
        decorate(axes)
        ax = axes[0]
        for i, (label, summary, family) in enumerate((
            ("GRU / primary bank", forecast["primary"], "recurrent"),
            ("GRU / second bank", forecast["second_bank"], "recurrent"),
            ("FF / primary bank", forecast["primary"], "feedforward"),
            ("FF / second bank", forecast["second_bank"], "feedforward"))):
            row = next(r for r in summary["comparisons"] if r["family"] == family)
            delta = row["differences"]["energy"]
            low, high = row["ci95"]["energy"]
            color = COLORS[3] if family == "recurrent" else COLORS[2]
            ax.errorbar(delta, i, xerr=[[delta - low], [high - delta]], fmt="s", capsize=4, color=color)
            original = next(r for r in summary["summary"] if r["procedure"] == f"{family}_original" and r["mechanism"] == "all")
            continued = next(r for r in summary["summary"] if r["procedure"] == f"{family}_continued" and r["mechanism"] == "all")
            for jitter, seed in zip((-.12, 0, .12), SEEDS):
                old = next(r["energy"] for r in original["per_seed"] if r["seed"] == seed)
                new = next(r["energy"] for r in continued["per_seed"] if r["seed"] == seed)
                ax.scatter(new - old, i + jitter, marker="|", color=color, alpha=.6, s=65)
        ax.set_yticks(range(4), ["GRU / first bank", "GRU / second bank", "FF / first bank", "FF / second bank"])
        ax.invert_yaxis()
        ax.axvline(0, color=MUTED)
        ax.set_title("PRIMARY / k=5, h=10", color=PARCHMENT, loc="left", fontsize=10)
        ax.set_xlabel("Continued − original energy")
        for ax, family, color in zip(axes[1:], ("feedforward", "recurrent"), COLORS[2:]):
            for origin, linestyle in zip(ORIGINS, ("-", "--", "-.", ":")):
                values = []
                for horizon in HORIZONS:
                    summary = next(s for s in forecast["secondary"] if s["origin"] == origin and s["horizon"] == horizon and s["population"] == "all_groups")
                    values.append(_overall(summary, f"{family}_continued", "energy"))
                ax.plot(HORIZONS, values, marker="s", markersize=3, color=color, linestyle=linestyle, label=f"k={origin}")
            ax.set_title(f"{family.upper()} / CONTINUED", loc="left", color=PARCHMENT, fontsize=10)
            ax.set_xlabel("Forecast horizon h (rounds)")
            ax.set_ylabel("Joint endpoint energy score")
            ax.set_xticks(HORIZONS)
            ax.legend(frameon=False, labelcolor=PARCHMENT, fontsize=8)
        fig.suptitle("EVOPOLIS / COLLECTIVE FORECAST FIDELITY", x=0.075, ha="left", fontweight="bold", fontsize=15)
        fig.text(0.075, 0.075, "Primary: 21 eligible groups (5 Equal, 8 Mixed, 8 Proportional); bars = 95% group intervals; ticks = fitted seeds.\nRight: all 24 groups, equally weighted mechanisms; 64 draws per seed/group. Endpoint energy is not a path score.", color=MUTED)
        fig.subplots_adjust(left=0.14, right=0.98, top=0.84, bottom=0.24, wspace=0.45)
        save(fig, "task04-forecast-scores")

        bins = list(csv.DictReader((results / "collective_calibration_bins.csv").open()))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
        fig.patch.set_facecolor(BACKGROUND)
        decorate(axes)
        for ax, family, color in zip(axes, ("feedforward", "recurrent"), COLORS[2:]):
            ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
            for budget, linestyle in (("original", "--"), ("continued", "-")):
                entries = [r for r in bins if r["procedure"] == f"{family}_{budget}" and r["mechanism"] == "all" and r["predicted"]]
                ax.plot([float(r["predicted"]) for r in entries], [float(r["observed"]) for r in entries], marker="s", color=color, linestyle=linestyle, label=budget)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted allocation-renewal probability", ylabel="Observed renewal fraction")
            ax.set_title(family.upper(), color=PARCHMENT, loc="left")
            ax.legend(frameon=False, labelcolor=PARCHMENT)
        fig.suptitle("EVOPOLIS / ONE-STEP COLLECTIVE RENEWAL", x=0.08, ha="left", fontweight="bold", fontsize=15)
        fig.text(0.08, 0.075, "Exact convolution of four legal-return PMFs; renewal = 1.4 × total returns ≥ current allocations.\nRecorded states/history; nonforced rounds; group/mechanism weights. Bins fixed at increments of 0.1.", color=MUTED)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.24, wspace=0.3)
        save(fig, "task04-renewal-calibration")

        arrays, manifest = load()
        chosen = {m: min(g["index"] for g in manifest["groups"] if g["split"] == "test" and g["mechanism"] == m and (arrays["offers"][g["index"], :, 5] >= 1).any()) for m in MECHANISMS}
        cases = {}
        for meta, body, paths in iter_cells(include_paths=True):
            if meta["family"] == "recurrent" and int(meta.get("seed", meta.get("training_seed"))) == 17 and meta["origin"] == 5 and meta["bank"] == "main" and meta["convention"] == "recorded" and meta["group_index"] == chosen[meta["mechanism"]]:
                cases[(meta["mechanism"], meta["budget"])] = (meta, paths)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey="row")
        fig.patch.set_facecolor(BACKGROUND)
        decorate(axes)
        case_details = []
        for column, m in enumerate(MECHANISMS):
            group_index = chosen[m]
            group = next(g for g in manifest["groups"] if g["index"] == group_index)
            time_axis = np.arange(6, 26)
            for budget, color, line in (("original", COLORS[0], "--"), ("continued", COLORS[3], "-")):
                meta, paths = cases[(m, budget)]
                for ax, variable in ((axes[0, column], "pool_after"), (axes[1, column], "participation")):
                    lower, middle, upper = np.quantile(paths[variable], (0.1, 0.5, 0.9), axis=0)
                    ax.fill_between(time_axis, lower, upper, color=color, alpha=0.13)
                    ax.plot(time_axis, middle, color=color, linestyle=line, label=f"{budget} median / 80%")
                case_details.append({"mechanism": m, "group_index": group_index, "group_key": group["key"], "cell_id": meta["id"], "budget": budget, "seed": 17})
            axes[0, column].plot(np.arange(1, 26), arrays["next_pool"][group_index, :25], color=PARCHMENT, label="recorded human")
            axes[1, column].plot(np.arange(1, 26), (arrays["offers"][group_index, :, :25] >= 1).sum(axis=0), color=PARCHMENT, label="recorded human")
            for ax in axes[:, column]:
                ax.axvline(5.5, color=COLORS[1], linewidth=1.5)
                ax.set_xlim(1, 25)
                ax.set_xlabel("Source decision round (1-based)")
            axes[0, column].set_title(f"{m.upper()} / {group['launch_id']}", loc="left", color=PARCHMENT, fontsize=10)
            axes[0, column].set_ylim(0, 205)
            axes[1, column].set_ylim(-0.1, 4.1)
        axes[0, 0].set_ylabel("Pool after decision (units)")
        axes[1, 0].set_ylabel("Residents offered ≥ 1")
        axes[0, 0].legend(frameon=False, labelcolor=PARCHMENT, fontsize=7)
        fig.suptitle("EVOPOLIS / OBSERVED PREFIX → GENERATED FUTURES", x=0.07, ha="left", fontweight="bold", fontsize=15)
        fig.text(0.07, 0.055, "GRU seed 17, k=5; first eligible group by fixed source index in each mechanism. Weights frozen.\nBands: 10th–90th percentiles of 64 forecast branches, not human-sample confidence intervals. Boundary precedes round 6 choices.", color=MUTED)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.15, hspace=0.3, wspace=0.18)
        save(fig, "task04-forecast-trajectories")
        write_json(results / "illustrated_cases.json", {"selection": "Smallest source group index among k=5 eligible test groups in each baseline mechanism; fixed GRU seed 17, both budgets", "cases": case_details})
