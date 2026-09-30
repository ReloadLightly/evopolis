"""Editable source for the Figure 2A reproduction; no newly simulated records."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path("data/cache/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .analysis import COHORTS, MECHANISMS

COLORS = ("#2476B7", "#9A5E9D", "#CC5639", "#198568")
MARKERS = ("o", "s", "^", "D")
LABELS = ("Equal", "Mixed", "Proportional", "Recorded RL mechanism (M1)")


def plot(groups: list[dict], summary: list[dict], directory: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "evopolis-task01",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6), sharex=True, sharey=True)
    fig.patch.set_facecolor("#ffffff")
    titles = ("A  Recorded behavioral-clone outcomes (BC1)", "B  Human Experiment 1")
    for ax, cohort, title in zip(axes, COHORTS, titles):
        ax.set_facecolor("#FAFBFC")
        ax.grid(color="#E2E6EA", linewidth=0.7, zorder=0)
        for mechanism, color, marker in zip(MECHANISMS, COLORS, MARKERS):
            points = [r for r in groups if r["cohort"] == cohort and r["mechanism"] == mechanism]
            mean = next(r for r in summary if r["cohort"] == cohort and r["mechanism"] == mechanism)
            ax.scatter([r["gini"] for r in points], [r["surplus"] for r in points],
                       color=color, marker=marker, s=15 if cohort == "BC 1" else 32,
                       alpha=0.19 if cohort == "BC 1" else 0.4, linewidths=0, zorder=2)
            ax.errorbar(mean["gini_mean"], mean["surplus_mean"],
                        xerr=mean["gini_ci95_halfwidth"], yerr=mean["surplus_ci95_halfwidth"],
                        color=color, elinewidth=2, capsize=3, zorder=4)
            ax.scatter(mean["gini_mean"], mean["surplus_mean"], color=color,
                       marker=marker, s=120, edgecolors="white", linewidths=1.4, zorder=5)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=28)
        n = 512 if cohort == "BC 1" else 40
        ax.text(0, 1.025, f"{n:,} games per mechanism · four players per game", transform=ax.transAxes, color="#586675", fontsize=9)
        ax.set_xlabel("Inequality of player surplus (Gini)", labelpad=9)
        ax.set_xlim(-0.02, 0.76)
        ax.set_ylim(-0.3, 16)
        ax.set_xticks([0, 0.15, 0.3, 0.45, 0.6, 0.75])
    axes[0].set_ylabel("Mean player surplus (game units / player / round)", labelpad=10)
    fig.suptitle("Sustaining the commons while sharing its returns", x=0.08, y=0.98,
                 ha="left", fontsize=17, fontweight="bold", color="#192E3F")
    legend = [Line2D([0], [0], color=c, marker=m, linestyle="", markersize=8, label=l)
              for c, m, l in zip(COLORS, MARKERS, LABELS)]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.105), ncol=4, frameon=False)
    fig.text(0.08, 0.075, "Small marks: individual games. Large marks: means. Bars: marginal 95% intervals across games (normal approximation).", fontsize=8.5, color="#586675")
    fig.text(0.08, 0.045, "Source: Koster, Pîslar et al. (2025), Figure 2A; official DeepMind trajectories, revision 4f1a99a. CC BY 4.0.", fontsize=8.5, color="#586675")
    fig.text(0.08, 0.018, "EvoPolis reproduction: all points are released outcomes; no behavioral model or RL mechanism was trained here.", fontsize=8.5, color="#586675")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.25, wspace=0.13)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / "figure2a.png", dpi=180, facecolor="white")
    fig.savefig(directory / "figure2a.svg", metadata={"Date": None}, facecolor="white")
    svg = directory / "figure2a.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
