"""Task 06 evidence graphics, rendered only from completed analysis artifacts.

The shared EvoPolis palette, monospaced labels and square silver-edged panels
preserve continuous chart geometry. Model forecasts, recorded BC1 outcomes,
and recorded human outcomes remain explicitly distinguished.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .behavior_data import ROOT, write_json
from .plotting import BACKGROUND, PANEL, PARCHMENT, MUTED, COLORS, MARKERS
from .sources import sha256

RESULTS = ROOT / 'results/task06'
FIGURES = ROOT / 'docs/assets'
RULES = ('Equal', 'Mixed', 'Proportional')
SEEDS = (17, 29, 43)
STYLE = {'font.family': 'DejaVu Sans Mono', 'font.size': 9,
         'text.color': PARCHMENT, 'axes.labelcolor': PARCHMENT,
         'xtick.color': MUTED, 'ytick.color': MUTED,
         'svg.fonttype': 'none', 'svg.hashsalt': 'evopolis-task06'}


def label(family):
    return {'Human': 'Recorded humans', 'BC1': 'Recorded BC1',
            'T03-constant': 'Task 03 constant', 'T03-GRU': 'Task 03 GRU',
            'T04-GRU': 'Continued GRU'}.get(family, family.replace('CL-FA-', 'CL(FA-') + ')'
                                           if family.startswith('CL-FA-') else family)


def _decorate(axes):
    for ax in np.asarray(axes).ravel():
        ax.set_facecolor(PANEL)
        ax.set_axisbelow(True)
        ax.grid(color=MUTED, alpha=.16, linewidth=.6)
        for spine in ax.spines.values():
            spine.set_color('#a8b7d1')
            spine.set_linewidth(.85)
        ax.tick_params(labelsize=8, length=4)


def _save(fig, directory, name):
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    png, svg = directory / f'{name}.png', directory / f'{name}.svg'
    fig.savefig(png, dpi=180, facecolor=BACKGROUND)
    fig.savefig(svg, facecolor=BACKGROUND, metadata={'Date': None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
    return [png, svg]


def _interval(ax, estimate, bounds, position, *, color, marker='o', size=7, width=1.5):
    """Draw percentile intervals without assuming they contain the estimate."""
    lower, upper = bounds
    ax.hlines(position, lower, upper, color=color, linewidth=width)
    ax.vlines([lower, upper], position-.055, position+.055, color=color, linewidth=width)
    ax.plot(estimate, position, marker=marker, color=color, markersize=size,
            markeredgecolor=PARCHMENT, markeredgewidth=.5, linestyle='')


def benchmark_figure(values, fa_best, directory=FIGURES):
    """``values[family][rule]`` is a completed outcome-summary dictionary."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    selected = ('Human', 'BC1', 'T03-constant', 'T03-GRU', fa_best, 'CL-'+fa_best)
    for family in selected:
        if set(RULES) - set(values[family]):
            raise ValueError(f'Incomplete benchmark rules for {family}')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    fig.patch.set_facecolor(BACKGROUND)
    _decorate(axes)
    settings = (('surplus', 'Mean surplus / player / round'),
                ('survival', 'Pool after round 40 > 1 (fraction)'),
                ('gini', 'Gini of four player mean surpluses'),
                ('pool20', 'Pool after round 20 (units)'))
    x = np.arange(len(selected))
    for ax, (metric, ylabel) in zip(axes.ravel(), settings):
        for j, (rule, color, marker) in enumerate(zip(RULES, COLORS, MARKERS)):
            y = [values[family][rule][metric] for family in selected]
            if any(value is None or not np.isfinite(value) for value in y):
                raise ValueError(f'Undefined displayed benchmark: {metric}/{rule}')
            ax.scatter(x+(j-1)*.20, y, color=color, marker=marker, s=48,
                       edgecolor=PARCHMENT, linewidth=.65, zorder=4)
        ax.axvline(1.5, color=MUTED, alpha=.6, linestyle=':', linewidth=1)
        ax.set_xticks(x, [label(f).replace('Recorded ', '') for f in selected], rotation=20, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_xlim(-.6, len(selected)-.4)
        ax.set_title(ylabel.upper().split(' (')[0], loc='left', fontsize=10, color=PARCHMENT, pad=12)
        if metric == 'survival':
            ax.set_ylim(-.04, 1.04)
        else:
            ax.set_ylim(bottom=0)
    legend = [Line2D([0], [0], marker=m, color=c, linestyle='', markersize=7, label=r)
              for r, c, m in zip(RULES, COLORS, MARKERS)]
    fig.legend(handles=legend, loc='upper right', bbox_to_anchor=(.97, .925),
               ncol=3, frameon=False, labelcolor=PARCHMENT)
    fig.suptitle('EVOPOLIS / EXPERIMENT 1 INSTITUTION BENCHMARK', x=.075, y=.98,
                 ha='left', fontsize=15, fontweight='bold')
    fig.text(.075, .065, 'Human means include all 40 groups per rule, including opened test groups. BC1: 512 recorded games per rule.\n'
             'New simulations: 1,536 games per family/rule, three frozen fits pooled. FA-best selected by validation NLL; CL calibrates the same family.',
             fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=.075, right=.975, top=.835, bottom=.17, hspace=.53, wspace=.26)
    return _save(fig, directory, 'task06-institution-benchmark')


def attenuation_figure(diagnostics, fa_best, directory=FIGURES):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    selected = ('T03-constant', 'T03-GRU', 'T04-GRU', fa_best, 'CL-'+fa_best)
    per_rule = {(r['family'], r['mechanism']): r for r in diagnostics['per_rule']}
    attenuation = {r['family']: r for r in diagnostics['attenuation']}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.7), gridspec_kw={'width_ratios': [1.55, 1]})
    fig.patch.set_facecolor(BACKGROUND)
    _decorate(axes)
    x = np.arange(len(selected))
    for j, (rule, color, marker) in enumerate(zip(RULES, COLORS, MARKERS)):
        data = [per_rule[family, rule] for family in selected]
        observed = data[0]['observed_fraction']
        if any(abs(r['observed_fraction'] - observed) > 1e-12 for r in data):
            raise ValueError('Attenuation methods do not share the same human targets')
        axes[0].scatter(x+(j-1)*.18, [r['predicted_fraction'] for r in data], color=color,
                        marker=marker, s=56, edgecolor=PARCHMENT, linewidth=.6, zorder=4)
        axes[0].axhline(observed, color=color, linewidth=1.2, alpha=.8, linestyle=':')
    axes[0].axhline(1/1.4, color=PARCHMENT, linewidth=.9, linestyle='--')
    axes[0].text(.98, 1/1.4+.018, '1/1.4', transform=axes[0].get_yaxis_transform(),
                 ha='right', color=PARCHMENT, fontsize=8)
    axes[0].set_xticks(x, [label(f) for f in selected], rotation=22, ha='right')
    axes[0].set_ylabel('Nonforced predicted return fraction E[c]/e')
    axes[0].set_ylim(0, 1)
    axes[0].set_xlim(-.6, len(selected)-.4)
    axes[0].set_title('RULE RESPONSIVENESS / VALIDATION', loc='left', color=PARCHMENT, fontsize=11)
    for position, family in enumerate(selected):
        value = attenuation[family]['attenuation_index']
        if value is None:
            raise ValueError('Undefined attenuation index cannot be plotted')
        axes[1].plot(value, position, 's', color=COLORS[3] if family.startswith('CL-') else COLORS[2],
                     markeredgecolor=PARCHMENT, markersize=7)
    axes[1].axvline(1, color=PARCHMENT, linestyle='--', linewidth=1)
    axes[1].set_yticks(range(len(selected)), [label(f) for f in selected])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Predicted P−E / observed P−E')
    axes[1].set_title('ATTENUATION INDEX', loc='left', color=PARCHMENT, fontsize=11)
    handles = [Line2D([0], [0], color=c, marker=m, linestyle='', label=r)
               for r, c, m in zip(RULES, COLORS, MARKERS)]
    handles.append(Line2D([0], [0], color=MUTED, linestyle=':', label='Human rule mean'))
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.97, .92), ncol=4,
               frameon=False, labelcolor=PARCHMENT, fontsize=9)
    fig.suptitle('EVOPOLIS / DO PREDICTIONS REACT TO THE ALLOCATION RULE?', x=.07, y=.99,
                 ha='left', fontsize=15, fontweight='bold')
    fig.text(.07, .06, 'Eight validation groups per rule; nonforced choices averaged within groups, then equally across groups and three fits.\n'
             'Dashed white line: resource-weighted nonshrink reference (5/7). Displayed fractions weight nonforced choices within groups, not allocated resources.',
             fontsize=8.2, color=MUTED)
    fig.subplots_adjust(left=.07, right=.98, top=.78, bottom=.30, wspace=.63)
    return _save(fig, directory, 'task06-attenuation')


def sensitivity_figure(cells, fa_best, taus, directory=FIGURES):
    """``cells[family, seed, tau][rule]`` holds complete sensitivity outputs."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    selected = ('T03-GRU', fa_best, 'CL-'+fa_best)
    fig, axes = plt.subplots(3, 3, figsize=(15, 11.5), sharex=False, sharey='col')
    fig.patch.set_facecolor(BACKGROUND)
    _decorate(axes)
    measures = (('simulation', 'survival', 'Simulated survival'),
                ('simulation', 'surplus', 'Simulated mean surplus'),
                ('validation', 'predicted_fraction', 'Predicted return fraction'))
    zero = list(taus).index(0)
    surplus_values = []
    for row, family in enumerate(selected):
        for rule, color, marker in zip(RULES, COLORS, MARKERS):
            series = []
            for tau in taus:
                selected_cells = [cells[family, seed, float(tau)][rule] for seed in SEEDS]
                series.append({kind: {metric: float(np.mean([c[kind][metric] for c in selected_cells]))
                                      for metric in ('nll', 'predicted_fraction') if metric in selected_cells[0][kind]}
                               if kind == 'validation' else
                               {metric: float(np.mean([c[kind][metric] for c in selected_cells])) for metric in ('survival', 'surplus')}
                               for kind in ('simulation', 'validation')})
            costs = [v['validation']['nll']-series[zero]['validation']['nll'] for v in series]
            for col, (kind, metric, title) in enumerate(measures):
                y = [v[kind][metric] for v in series]
                if metric == 'surplus':
                    surplus_values.extend(y)
                ax = axes[row, col]
                ax.plot(costs, y, color=color, marker=marker, markersize=4, linewidth=1.25, label=rule)
                ax.scatter(costs[zero], y[zero], facecolor=PANEL, edgecolor=PARCHMENT, marker='o',
                           s=82, linewidth=1.3, zorder=5)
                ax.scatter(costs[zero], y[zero], color=color, marker=marker, s=20, zorder=6)
                ax.axvline(0, color=MUTED, alpha=.6, linewidth=.7)
                ax.set_xlabel('Change in validation NLL (nats)')
                if row == 0:
                    ax.set_title(title.upper(), loc='left', fontsize=10, color=PARCHMENT, pad=12)
                if col == 0:
                    ax.set_ylabel(label(family)+'\n'+title)
                else:
                    ax.set_ylabel(title)
        axes[row, 0].set_ylim(-.04, 1.04)
        axes[row, 2].set_ylim(0, 1)
        axes[row, 2].axhline(1/1.4, color=PARCHMENT, linestyle='--', linewidth=1)
        axes[row, 2].text(.97, 1/1.4+.02, '1/1.4', transform=axes[row, 2].get_yaxis_transform(),
                          ha='right', color=PARCHMENT, fontsize=8)
    # Set the shared surplus range after every family's points are present.
    # Setting a bound within the row loop disables later shared-axis autoscaling.
    axes[0, 1].set_ylim(0, max(surplus_values)*1.06)
    handles = [Line2D([0], [0], color=c, marker=m, label=r)
               for r, c, m in zip(RULES, COLORS, MARKERS)]
    handles.append(Line2D([0], [0], color=PARCHMENT, marker='o', markerfacecolor=PANEL,
                          linestyle='', label='Base distribution (τ=0)'))
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.98, .947), ncol=4,
               frameon=False, labelcolor=PARCHMENT, fontsize=9)
    fig.suptitle('EVOPOLIS / SMALL CHOICE SHIFTS, DIFFERENT COLLECTIVE FUTURES', x=.075, y=.99,
                 ha='left', fontsize=15, fontweight='bold')
    fig.text(.075, .04, 'Legal PMFs multiplied by exp(τ·c/e), τ = −0.3, 0, 0.3, 0.6, 0.9; connected in τ order, without smoothing.\n'
             'Each point pools 3 × 256 simulated games per rule. NLL and predicted fractions use eight validation groups per rule.\n'
             'Changes use each family’s τ=0 PMF; CL includes its fitted tilt. Fraction panels mark the resource-weighted nonshrink reference; plotted fractions are group-balanced.',
             color=MUTED, fontsize=8.5)
    fig.subplots_adjust(left=.075, right=.98, top=.865, bottom=.145, wspace=.29, hspace=.36)
    return _save(fig, directory, 'task06-tipping-sensitivity')


def transfer_figure(comparisons, fa_best, directory=FIGURES):
    """``comparisons[family]['E1'/'E2']`` follows paired_error_comparison."""
    import matplotlib.pyplot as plt
    selected = ('BC1', fa_best, 'CL-'+fa_best, 'T03-GRU', 'T03-constant')
    colors = (COLORS[3], COLORS[2], COLORS[1], COLORS[0], MUTED)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.3), sharey='row')
    fig.patch.set_facecolor(BACKGROUND)
    _decorate(axes)
    for col, (estimand, title) in enumerate((('E1', 'INTERPOLATING − PROPORTIONAL'),
                                            ('E2', 'INTERPOLATING LEVEL'))):
        first = comparisons['BC1'][estimand]
        observed = first['observed']
        human_ci = first['observed_ci95']
        ax = axes[0, col]
        ax.axvspan(*human_ci, color=PARCHMENT, alpha=.065, zorder=0)
        ax.axvline(observed, color=PARCHMENT, alpha=.6, linestyle=':', linewidth=1)
        _interval(ax, observed, human_ci, 0, color=PARCHMENT, marker='D', size=8, width=2)
        for pos, (family, color) in enumerate(zip(selected, colors), start=1):
            row = comparisons[family][estimand]
            if row['observed'] != observed or row['observed_ci95'] != human_ci:
                raise ValueError('Transfer figure comparisons use different human bootstrap evidence')
            prediction, mc = row['predicted'], row['prediction_mc_se']
            _interval(ax, prediction, [prediction-1.96*mc, prediction+1.96*mc], pos,
                      color=color, marker='s' if pos <= 3 else 'o', size=7 if pos <= 3 else 5, width=1)
            bottom = axes[1, col]
            _interval(bottom, row['paired_absolute_error_difference'], row['paired_difference_ci95'], pos-1,
                      color=color, marker='s' if pos <= 3 else 'o', size=7 if pos <= 3 else 5)
        ax.set_title(title, loc='left', color=PARCHMENT, fontsize=11, pad=13)
        ax.set_yticks(range(6), ['Recorded humans', *[label(f) for f in selected]])
        ax.set_ylim(5.5, -.5)
        ax.set_xlabel('Mean surplus / player / round')
        axes[1, col].axvline(0, color=PARCHMENT, alpha=.7, linewidth=1)
        axes[1, col].set_yticks(range(5), [label(f) for f in selected])
        axes[1, col].set_ylim(4.5, -.5)
        axes[1, col].set_xlabel('Absolute-error difference versus BC1\n← lower error                    higher error →')
        axes[1, col].set_title(estimand+' / PAIRED COMPARISON WITH BC1', loc='left', color=PARCHMENT,
                               fontsize=10, pad=12)
    fig.suptitle('EVOPOLIS / EXPERIMENT 2 TRANSFER', x=.16, y=.98, ha='left',
                 fontsize=16, fontweight='bold')
    fig.text(.16, .056, 'Top: recorded human 95% group-bootstrap interval (shading); prediction bars = ±1.96 Monte Carlo SE, shown separately.\n'
             'Bottom: paired 95% intervals from the same 2,000 human-group bootstrap resamples, stratified by rule; forecasts held fixed.\n'
             'BC1, FA-best and its CL variant are primary. Task 03 GRU and constant are named secondary references.',
             color=MUTED, fontsize=8.3)
    fig.subplots_adjust(left=.16, right=.98, top=.88, bottom=.20, hspace=.47, wspace=.23)
    return _save(fig, directory, 'task06-transfer-effects')


def plot_results(results=RESULTS, figures=FIGURES):
    """Load and validate all plotted JSON evidence before creating any figure."""
    import matplotlib.pyplot as plt
    inputs = {}

    def read(path):
        value = json.loads(path.read_text())
        inputs[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = sha256(path)
        return value

    configuration = read(ROOT / 'configs/task06.json')
    training = read(results / 'training_complete.json')
    benchmark = read(results / 'benchmark.json')
    diagnostics = read(results / 'diagnostics_validation.json')
    transfer = read(results / 'transfer.json')
    best = training['fa_best']
    if transfer['fa_best'] != best:
        raise ValueError('Transfer primary family differs from validation-selected FA-best')
    if transfer['primary_simulators'] != ['BC1', best, 'CL-'+best]:
        raise ValueError('Transfer primary comparisons differ from the declared design')
    # Exp1 benchmark is the explicitly labelled all-groups comparison, including
    # previously opened test groups. The train+validation analysis stays in JSON.
    values = {'Human': {}}
    for row in benchmark['populations']['all']['rows']:
        family, rule = row['family'], row['rule']
        if rule not in RULES:
            continue
        values.setdefault(family, {})[rule] = {
            metric: row['predicted'][metric]['mean'] for metric in ('surplus', 'survival', 'gini', 'pool20')}
        human = {metric: row['observed'][metric]['mean']
                 for metric in ('surplus', 'survival', 'gini', 'pool20')}
        if rule in values['Human'] and values['Human'][rule] != human:
            raise ValueError('Benchmark rows use different human cohorts')
        values['Human'][rule] = human
        if row['observed']['groups'] != 40:
            raise ValueError('Figure requires all40 Exp1 groups per rule')
    expected_checkpoints = 3 * (len(configuration['reference_families']) +
                                 len(configuration['fa_families']) + len(configuration['calibration_families']))
    if diagnostics['checkpoints'] != expected_checkpoints or diagnostics['human_groups'] != 32:
        raise ValueError('Validation diagnostics are incomplete')
    cells = {}
    taus = configuration['sensitivity']['taus']
    for family in ('T03-GRU', best, 'CL-'+best):
        for seed in SEEDS:
            for tau in taus:
                path = results / 'sensitivity' / f'{family}_{seed}_{tau:+.1f}.json'
                saved = read(path)
                record = saved['contract']['record']
                if (record['family'], record['seed'], saved['contract']['tau']) != (family, seed, tau):
                    raise ValueError(f'Sensitivity identity differs from filename: {path}')
                if any(saved['cells'][rule]['simulation']['games'] != 256 for rule in RULES):
                    raise ValueError(f'Sensitivity game count is incomplete: {path}')
                cells[family, seed, float(tau)] = saved['cells']
    comparisons = {family: row['surplus'] for family, row in transfer['scores'].items()}
    with plt.rc_context(STYLE):
        generated = [*benchmark_figure(values, best, figures),
                     *attenuation_figure(diagnostics, best, figures),
                     *sensitivity_figure(cells, best, taus, figures),
                     *transfer_figure(comparisons, best, figures)]
    receipt = {'inputs_sha256': inputs,
               'figures_sha256': {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha256(p)
                                 for p in generated},
               'fa_best': best,
               'displayed_primary': ['BC1', best, 'CL-'+best],
               'displayed_named_secondary': ['T03-GRU', 'T03-constant'],
               'diagnostic_addition': 'Continued GRU shown in attenuation; complete simulator comparisons remain in analysis JSON',
               'theme': 'docs/VISUAL_STYLE.md; evopolis.plotting palette and matplotlib styling',
               'uncertainty': 'Exp2 human95% bootstrap and simulation1.96MCSE intervals distinguished; benchmark and attenuation show means',
               'geometry': 'Continuous measured axes; no smoothing or generated image assets'}
    write_json(results / 'figures.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--figures', type=Path, default=FIGURES)
    args = parser.parse_args()
    plot_results(args.results, args.figures)


if __name__ == '__main__':
    main()
