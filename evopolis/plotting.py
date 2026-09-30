"""Editable source for the Figure 2A reproduction; no newly simulated records."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path("data/cache/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .analysis import COHORTS, MECHANISMS

# The EvoPolis 16-bit palette is decorative; data remain on continuous axes.
BACKGROUND = "#172038"
PANEL = "#253a5e"
PARCHMENT = "#fff1d2"
MUTED = "#b6c6df"
COLORS = ("#ff8b8b", "#ffcf6e", "#6dcff6", "#73e0b3")
MARKERS = ("o", "s", "^", "D")
LABELS = ("Equal", "Mixed", "Proportional", "Recorded RL mechanism (M1)")


def plot(groups: list[dict], summary: list[dict], directory: Path) -> None:
    with plt.rc_context({
        "font.family": "DejaVu Sans Mono", "font.size": 10,
        "text.color": PARCHMENT, "axes.labelcolor": PARCHMENT,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "svg.fonttype": "none", "svg.hashsalt": "evopolis-task01",
    }):
        _draw(groups, summary, directory)


def _draw(groups: list[dict], summary: list[dict], directory: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.6), sharex=True, sharey=True)
    fig.patch.set_facecolor(BACKGROUND)
    # Square color tiles echo a console palette without altering the evidence.
    for index, color in enumerate(COLORS):
        fig.add_artist(Rectangle((0.08 + index * 0.011, 0.943), 0.007, 0.011,
                                 transform=fig.transFigure, color=color, linewidth=0))
    fig.text(0.136, 0.946, "EVOPOLIS / EMPIRICAL RECORD 01", fontsize=10,
             color=MUTED, va="center", fontweight="bold")
    fig.text(0.98, 0.946, "FIGURE 2A", fontsize=10, color=MUTED,
             va="center", ha="right")
    fig.suptitle("Sustaining the commons while sharing its returns",
                 x=0.08, y=0.915, ha="left", fontsize=18,
                 fontweight="bold", color=PARCHMENT)
    fig.text(0.08, 0.858, "Recorded outcomes: behavioral clones and human participants",
             fontsize=11, color=MUTED)
    titles = ("A  BEHAVIORAL-CLONE OUTCOMES (BC1)", "B  HUMAN EXPERIMENT 1")
    for ax, cohort, title in zip(axes, COHORTS, titles):
        ax.set_facecolor(PANEL)
        ax.set_axisbelow(True)
        ax.grid(color=MUTED, linewidth=0.6, alpha=0.18, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(MUTED)
            spine.set_linewidth(1)
        ax.tick_params(axis="both", labelsize=9, length=4, pad=6)
        for mechanism, color, marker in zip(MECHANISMS, COLORS, MARKERS):
            points = [r for r in groups if r["cohort"] == cohort and r["mechanism"] == mechanism]
            mean = next(r for r in summary if r["cohort"] == cohort and r["mechanism"] == mechanism)
            ax.scatter([r["gini"] for r in points], [r["surplus"] for r in points],
                       color=color, marker=marker, s=15 if cohort == "BC 1" else 32,
                       alpha=0.27 if cohort == "BC 1" else 0.52, linewidths=0, zorder=2)
            ax.errorbar(mean["gini_mean"], mean["surplus_mean"],
                        xerr=mean["gini_ci95_halfwidth"], yerr=mean["surplus_ci95_halfwidth"],
                        color=color, elinewidth=2, capsize=4, capthick=1.5, zorder=4)
            ax.scatter(mean["gini_mean"], mean["surplus_mean"], color=color,
                       marker=marker, s=130, edgecolors=PARCHMENT, linewidths=1.4, zorder=5)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold",
                     color=PARCHMENT, pad=33)
        n = 512 if cohort == "BC 1" else 40
        ax.text(0, 1.028, f"{n:,} games per mechanism · four players per game",
                transform=ax.transAxes, color=MUTED, fontsize=9)
        ax.set_xlabel("Inequality of player surplus (Gini)", labelpad=12)
        ax.set_xlim(-0.02, 0.76)
        ax.set_ylim(-0.3, 16)
        ax.set_xticks([0, 0.15, 0.3, 0.45, 0.6, 0.75])
    axes[0].set_ylabel("Mean player surplus (game units / player / round)", labelpad=14)
    legend = [Line2D([0], [0], color=c, marker=m, linestyle="", markersize=8,
                     markeredgecolor=PARCHMENT, markeredgewidth=0.8, label=l)
              for c, m, l in zip(COLORS, MARKERS, LABELS)]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.53, 0.177),
               ncol=4, frameon=False, columnspacing=2, fontsize=10)
    fig.add_artist(Line2D([0.08, 0.98], [0.159, 0.159], transform=fig.transFigure,
                          color=MUTED, alpha=0.4, linewidth=1))
    fig.text(0.08, 0.13, "Small marks: individual games. Large marks: means.",
             fontsize=8.5, color=PARCHMENT)
    fig.text(0.08, 0.106, "Bars: marginal 95% intervals across games (normal approximation).",
             fontsize=8.5, color=MUTED)
    fig.text(0.08, 0.072, "Source: Koster, Pîslar et al. (2025), Figure 2A; official DeepMind trajectories, revision 4f1a99a. CC BY 4.0.",
             fontsize=8.2, color=MUTED)
    fig.text(0.08, 0.04, "EvoPolis reproduction: all points are released outcomes; no behavioral model or RL mechanism was trained here.",
             fontsize=8.2, color=MUTED)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.74, bottom=0.30, wspace=0.13)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / "figure2a.png", dpi=180, facecolor=BACKGROUND)
    fig.savefig(directory / "figure2a.svg", metadata={"Date": None}, facecolor=BACKGROUND)
    svg = directory / "figure2a.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
