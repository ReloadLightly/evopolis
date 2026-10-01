"""Task 05 descriptive mechanism checks and measured, editable research figures.

Diagnostics never refit population parameters or choose a forecast procedure.
Projection coefficients are fitted to training controls only; validation measures
their transported residual variation. Curves hold the prefix posterior fixed.
"""
import argparse
from contextlib import closing
import csv
import json
from pathlib import Path
import resource
import sqlite3
import time

import numpy as np

from .behavior_data import ROOT, load, write_json
from .sources import sha256

RESULTS = ROOT / 'results/task05'
FIGURES = ROOT / 'docs/assets'
FAMILIES = ('P0', 'P1', 'H0', 'H1')
SEEDS = (17, 29, 43)
MECHANISMS = ('Equal', 'Mixed', 'Proportional')


def observed_design(arrays, groups, eta):
    """Independent NumPy construction of prechoice z and b, legal choices only."""
    rows, traces, weights, identities = [], [], [], []
    for group in groups:
        index = group['index']
        offers, actions = arrays['offers'][index].T, arrays['y'][index].T
        count = int((offers >= 1).sum())
        if not count:
            continue
        own = np.full(4, .5)
        peer = np.full(4, .5)
        previous_fraction = np.zeros(4)
        previous_valid = np.zeros(4)
        previous_count = np.zeros(4)
        own_exposed = np.zeros(4)
        peer_exposed = np.zeros(4)
        for t, (e, c) in enumerate(zip(offers, actions)):
            others = np.array([np.sort(np.delete(e, i)) for i in range(4)])
            z = np.column_stack((np.ones(4), e/200, others/200,
                                 np.full(4, arrays['pool'][index, t]/200), own,
                                 previous_fraction, previous_valid, previous_count/3,
                                 own_exposed, peer_exposed))
            valid = e >= 1
            for i in np.flatnonzero(valid):
                rows.append(z[i].copy())
                traces.append(float(peer[i]))
                weights.append(1/count)
                identities.append((index, t, int(i)))
            fractions = np.divide(c, e, out=np.zeros(4), where=valid)
            peer_count = valid.sum() - valid.astype(int)
            peer_mean = (fractions.sum() - fractions) / np.maximum(peer_count, 1)
            own = np.where(valid, (1-eta)*own+eta*fractions, own)
            peer = np.where(peer_count > 0, (1-eta)*peer+eta*peer_mean, peer)
            previous_fraction, previous_valid, previous_count = fractions, valid, peer_count
            own_exposed = np.maximum(own_exposed, valid)
            peer_exposed = np.maximum(peer_exposed, peer_count > 0)
    return np.asarray(rows), np.asarray(traces), np.asarray(weights), identities


def projection_summary(z, b, weights, coefficient):
    weights = weights / weights.sum()
    residual = b-z@coefficient
    variance = float(np.sum(weights*(b-np.sum(weights*b))**2))
    residual_variance = float(np.sum(weights*(residual-np.sum(weights*residual))**2))
    mse = float(np.sum(weights*residual**2))
    return {'choices': len(b), 'peer_trace_sd': variance**.5,
            'residual_sd': residual_variance**.5, 'residual_rmse': mse**.5,
            'residual_mean': float(np.sum(weights*residual)),
            'r_squared_relative_to_split_mean': 1-mse/variance if variance else None,
            'residual_quantiles_05_50_95_unweighted': np.quantile(residual, [.05,.5,.95]).tolist()}


def _write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_diagnostics(results=RESULTS):
    import torch
    from .behavior_train import runtime_setup
    from .conditional_models import ConditionalModel, FEATURES, features, infer_prefix, tilted_log_mass
    runtime_setup()
    start, cpu = time.monotonic(), time.process_time()
    arrays, manifest = load()
    groups = {split: [g for g in manifest['groups'] if g['split'] == split]
              for split in ('train', 'validation')}
    # Source-order case choice is independent of fitted response or test outcomes.
    cases = []
    for mechanism in MECHANISMS:
        group = min((g for g in groups['train'] if g['mechanism'] == mechanism
                     and (arrays['offers'][g['index'], :, 5] >= 1).any()), key=lambda g:g['index'])
        resident = int(np.flatnonzero(arrays['offers'][group['index'], :, 5] >= 1)[0])
        cases.append({'mechanism': mechanism, 'group_index': group['index'],
                      'group_key': group['key'], 'origin': 5, 'resident': resident})
    parameters, projections, curves, posterior_rows = [], [], [], []
    checkpoint_hashes = {}
    for family in FAMILIES:
        for seed in SEEDS:
            directory = results / 'weights' / f'{family}_{seed}'
            saved = torch.load(directory/'best.pt', map_location='cpu', weights_only=True)
            metadata = json.loads((directory/'metadata.json').read_text())
            model = ConditionalModel(family)
            model.load_state_dict(saved['model_state'])
            model.eval()
            nodes = int(metadata.get('integration_nodes', metadata.get('nodes', 21)))
            with torch.no_grad():
                eta, beta, sigma = float(model.eta), float(model.beta), float(model.sigma)
            logs = json.loads((directory/'training.json').read_text())
            parameter = {'family': family, 'seed': seed,
                         'selected_epoch': int(saved['epoch']), 'beta': beta,
                         'eta': eta, 'sigma': sigma, 'integration_nodes': nodes,
                         'validation_nll': float(saved.get('validation_score', metadata['validation_score'])),
                         'final_validation_nll': logs[-1]['validation_nll'],
                         'final_40_epoch_validation_change': logs[-1]['validation_nll']-logs[-41]['validation_nll'],
                         'checkpoint_sha256': sha256(directory/'best.pt')}
            parameters.append(parameter)
            checkpoint_hashes[f'{family}_{seed}'] = parameter['checkpoint_sha256']
            train = observed_design(arrays, groups['train'], eta)
            validation = observed_design(arrays, groups['validation'], eta)
            z, b, weight, ids = train
            coefficient, _, rank, singular = np.linalg.lstsq(z*np.sqrt(weight[:,None]), b*np.sqrt(weight), rcond=None)
            projections.append({'family': family, 'seed': seed, 'features': list(FEATURES),
                                'coefficient': coefficient.tolist(), 'rank': int(rank),
                                'singular_values': singular.tolist(),
                                'train': projection_summary(*train[:3], coefficient),
                                'validation': projection_summary(*validation[:3], coefficient)})
            if seed != 17:
                continue
            for case in cases:
                index, origin, resident = case['group_index'], case['origin'], case['resident']
                with torch.no_grad():
                    prefix = infer_prefix(model, arrays, index, origin, nodes)
                    offer = torch.tensor(arrays['offers'][index, :, origin], dtype=torch.float64)
                    pool = torch.tensor(arrays['pool'][index, origin], dtype=torch.float64)
                    zz = features(pool, offer, prefix['history'])[resident]
                    raw = model.head(zz)
                    grid = prefix['nodes']
                    posterior = prefix['weights'][resident]
                    actual_b = float(prefix['history']['peer_trace'][resident])
                    e = float(offer[resident])
                    b_grid = np.linspace(0,1,101)
                    per_node = tilted_log_mass(raw[None,None,:].expand(101,len(grid),5),
                                               offer[resident].expand(101,len(grid)),
                                               model.beta*(torch.tensor(b_grid)[:,None]-.5)+grid)
                    conditional = per_node.exp()
                    mass = (conditional*posterior[None,:,None]).sum(1)
                    fraction = torch.arange(201, dtype=torch.float64)/e
                    means = (mass*fraction).sum(-1)
                    node_means = (conditional*fraction).sum(-1)
                    node_variances = (conditional*fraction.square()).sum(-1)-node_means.square()
                    # Mixture derivative averages conditional variance; posterior is held fixed.
                    derivatives = model.beta*(node_variances*posterior).sum(-1)
                zz = zz.numpy()
                diagnostic_index = ids.index((index, origin, resident))
                np.testing.assert_allclose(z[diagnostic_index], zz, rtol=0, atol=1e-12)
                np.testing.assert_allclose(b[diagnostic_index], actual_b, rtol=0, atol=1e-12)
                # Binary/count exposure controls must match; continuous covariates
                # use group-weighted training SD. The case itself is in support.
                categorical = [8,9,10,11]
                continuous = [1,2,3,4,5,6,7]
                categorical_match = (np.abs(z[:,categorical]-zz[categorical]) < 1e-12).all(1)
                wn = weight/weight.sum()
                center = (wn[:,None]*z).sum(0)
                scale = np.sqrt((wn[:,None]*(z-center)**2).sum(0))
                scale = np.maximum(scale, 1e-12)
                distance = np.sqrt(np.mean(((z[:,continuous]-zz[continuous])/scale[continuous])**2, axis=1))
                near = categorical_match & (distance <= .5)
                support = np.quantile(b[near], [.05,.95]).tolist() if near.any() else None
                ceiling = np.floor(e)/e
                curve = {**case, 'family': family, 'seed': seed,
                         'offer':e, 'feasible_matching_maximum':ceiling,
                         'fixed_z':zz.tolist(), 'observed_b':actual_b,
                         'b_grid':b_grid.tolist(), 'expected_fraction':means.numpy().tolist(),
                         'derivative_fixed_posterior':derivatives.numpy().tolist(),
                         'imperfect_history_matching': [float(y-x) if x <= ceiling else None
                                                        for x,y in zip(b_grid,means.numpy())],
                         'nearby_training_choices':int(near.sum()),
                         'nearby_b_central90':support,
                         'nearby_b_values':b[near].tolist()}
                curves.append(curve)
                if family.startswith('H'):
                    values, probability = grid.numpy(), posterior.numpy()
                    mean = float(np.dot(values, probability))
                    sd = float(np.sqrt(np.dot((values-mean)**2, probability)))
                    cumulative = np.cumsum(probability)
                    quantiles = [float(values[min(np.searchsorted(cumulative,q),len(values)-1)]) for q in (.05,.5,.95)]
                    posterior_rows.append({**case, 'family':family,'seed':seed,
                                           'prior_mean':0.,'prior_sd':sigma,
                                           'posterior_mean':mean,'posterior_sd':sd,
                                           'posterior_sd_over_prior_sd':sd/sigma,
                                           'posterior_q05_q50_q95':quantiles,
                                           'posterior_normalization':float(probability.sum()),
                                           'nodes':values.tolist(),'weights':probability.tolist()})
    results.mkdir(parents=True, exist_ok=True)
    _write_csv(results/'parameters.csv', parameters)
    write_json(results/'mechanism_diagnostics.json', {
        'checkpoint_hashes':checkpoint_hashes,
        'source_sha256':manifest['source_sha256'],
        'split_sha256':sha256(ROOT/'results/task03/split.json'),
        'config_sha256':sha256(ROOT/'configs/task05.json'),
        'diagnostics_code_sha256':sha256(Path(__file__)),
        'projection_method':'Weighted minimum-norm OLS b~the exact common z. Fit on nonforced training choices with each group total weight one; apply unchanged coefficients to validation. Fitted eta fixes each model history. This is descriptive redundancy, not causal adjustment or a candidate behavioral model.',
        'projections':projections,
        'curve_selection':'Seed 17; first source-index eligible training group within each baseline rule at origin 5; first eligible resident. Fixed observed z and prefix posterior weights for the entire b curve. No future choices enter posterior.',
        'support_method':'Exact agreement of own-valid, peer-count and two exposure controls; RMS distance <=0.5 across seven continuous controls standardized by group-weighted training SD. Rug marks neighboring training b; band is its 5th–95th percentile. Proximity is descriptive, not proof of joint support.',
        'design_consistency':'Independent NumPy projection design agrees with model prechoice features and peer trace at every illustrated family/case to absolute 1e-12.',
        'curves':curves,'illustrative_prefix_posteriors':posterior_rows,
        'matching_definition':'Expected contribution fraction minus b only when e>=1 and b<=floor(e)/e. This is imperfect history matching, not elicited preference or biased belief. Beta is a log tilt, not a matching slope.',
        'runtime':{'wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,
                   'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}})


def _plot_setup():
    from .plotting import BACKGROUND, PANEL, PARCHMENT, MUTED, COLORS
    import matplotlib.pyplot as plt
    style={'font.family':'DejaVu Sans Mono','font.size':9,'text.color':PARCHMENT,
           'axes.labelcolor':PARCHMENT,'xtick.color':MUTED,'ytick.color':MUTED,
           'svg.fonttype':'none','svg.hashsalt':'evopolis-task05'}
    return plt, style, (BACKGROUND,PANEL,PARCHMENT,MUTED,COLORS)


def _decorate(axes, palette):
    for ax in np.asarray(axes).reshape(-1):
        ax.set_facecolor(palette[1])
        ax.grid(color=palette[3],alpha=.16,linewidth=.6)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(palette[3])


def _save(fig, name, figures, plt, background):
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures/f'{name}.png',dpi=170,facecolor=background)
    fig.savefig(figures/f'{name}.svg',metadata={'Date':None},facecolor=background)
    path=figures/f'{name}.svg'
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def plot_mechanisms(results=RESULTS,figures=FIGURES):
    plt,style,palette=_plot_setup()
    background,panel,parchment,muted,colors=palette
    diagnostics=json.loads((results/'mechanism_diagnostics.json').read_text())
    with plt.rc_context(style):
        fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
        fig.patch.set_facecolor(background)
        _decorate(axes,palette)
        for ax,family in zip(axes.flat,FAMILIES):
            for seed,color in zip(SEEDS,colors):
                entries=json.loads((results/'weights'/f'{family}_{seed}'/'training.json').read_text())
                ax.plot([r['epoch'] for r in entries],[r['validation_nll'] for r in entries],color=color,alpha=.9,linewidth=1.1,label=f'seed {seed}')
                ax.plot([r['epoch'] for r in entries],[r['train_nll'] for r in entries],color=color,alpha=.35,linewidth=.8,linestyle='--')
            ax.set_title(f'{family} / THREE OPTIMIZATION STARTS',loc='left',color=parchment,fontsize=10)
            ax.set_xlabel('Completed epochs')
            ax.set_ylabel('Group-balanced nonforced NLL')
        axes[0,0].legend(frameon=False,labelcolor=parchment,fontsize=8)
        fig.suptitle('EVOPOLIS / MATCHED RESPONSE MODELS',x=.08,ha='left',fontsize=15,fontweight='bold')
        fig.text(.08,.045,'Solid: validation; dashed: training. Every fit completes 480 epochs; earliest validation minimum selects weights.\nCurves describe optimization, not new human observations or demonstrated convergence.',color=muted)
        fig.subplots_adjust(left=.08,right=.98,top=.89,bottom=.16,hspace=.4,wspace=.3)
        _save(fig,'task05-learning',figures,plt,background)

        fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey='row')
        fig.patch.set_facecolor(background)
        _decorate(axes,palette)
        for col,mechanism in enumerate(MECHANISMS):
            for family,color in zip(FAMILIES,colors):
                row=next(r for r in diagnostics['curves'] if r['mechanism']==mechanism and r['family']==family)
                grid=np.asarray(row['b_grid'])
                mean=np.asarray(row['expected_fraction'])
                within=np.zeros(len(grid),dtype=bool)
                if row['nearby_b_central90']:
                    lo,hi=row['nearby_b_central90']
                    within=(grid>=lo)&(grid<=hi)
                for ax,values in ((axes[0,col],mean),(axes[1,col],np.asarray([np.nan if v is None else v for v in row['imperfect_history_matching']]))):
                    ax.plot(grid,values,color=color,linewidth=1,linestyle=':',alpha=.75)
                    ax.plot(grid,np.where(within,values,np.nan),color=color,linewidth=2,label=family)
                # Rug is drawn in axes coordinates, independent of plotted y values.
                axes[0,col].plot(row['nearby_b_values'],np.full(row['nearby_training_choices'],.025+.035*FAMILIES.index(family)),linestyle='',marker='|',markersize=3,color=color,alpha=.3,transform=axes[0,col].get_xaxis_transform())
            axes[0,col].plot([0,1],[0,1],color=muted,linewidth=.7,linestyle='--',label='history matching')
            axes[1,col].axhline(0,color=muted,linewidth=.7)
            axes[0,col].set_title(f'{mechanism.upper()} / OFFER {row["offer"]:.2f}',loc='left',color=parchment,fontsize=10)
            for ax in axes[:,col]:
                ax.set_xlim(0,1)
                ax.set_xlabel('Peer-history trace b')
            axes[0,col].set_ylim(0,1.04)
        axes[0,0].set_ylabel('Expected contribution / own offer')
        axes[1,0].set_ylabel('Expected fraction − b\n(feasible matching only)')
        axes[0,0].legend(frameon=False,labelcolor=parchment,fontsize=8)
        fig.suptitle('EVOPOLIS / CONDITIONAL RESPONSE AND HISTORY MATCHING',x=.07,ha='left',fontsize=15,fontweight='bold')
        fig.text(.07,.04,'Training cases, seed 17, observed five-round prefix. Own history/resources and posterior weights held fixed.\nSolid: central 90% of nearby training b; rugs: nearby observations; dotted: illustrative extrapolation. No causal preference claim.',color=muted)
        fig.subplots_adjust(left=.07,right=.98,top=.88,bottom=.16,hspace=.32,wspace=.2)
        _save(fig,'task05-response-curves',figures,plt,background)


def plot_outcomes(results=RESULTS, figures=FIGURES):
    """Plot saved group-score contrasts and fixed cases; no new model selection."""
    from .conditional_generate import get_cell
    plt,style,palette=_plot_setup()
    background,panel,parchment,muted,colors=palette
    summary=json.loads((results/'forecast_summary.json').read_text())
    contrasts=('H1 minus H0','P1 minus P0','H1 minus P1','H0 minus P0')

    def comparison(block, label):
        return next(r for r in block['comparisons'] if r.get('comparison',r.get('contrast'))==label)

    with plt.rc_context(style):
        fig,axes=plt.subplots(1,3,figsize=(15,5.7))
        fig.patch.set_facecolor(background)
        _decorate(axes,palette)
        for panel_index,(ax,metric,title) in enumerate(zip(axes[:2],('nll','energy'),('INDIVIDUAL / SAME ROUNDS','COLLECTIVE / JOINT ENDPOINT'))):
            for i,(label,color) in enumerate(zip(contrasts,colors)):
                row=comparison(summary['primary'],label)
                mean=row['differences'][metric]
                low,high=row['ci95'][metric]
                ax.errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt='s',capsize=4,color=color)
            ax.set_yticks(range(4),[x.replace(' minus ',' − ') for x in contrasts])
            ax.invert_yaxis()
            ax.axvline(0,color=muted,linewidth=1)
            ax.set_xlabel('Difference in NLL' if metric=='nll' else 'Difference in energy score')
            ax.set_title(title,loc='left',fontsize=10,color=parchment)
        ax=axes[2]
        for i,(bank,label,color) in enumerate((('primary','Principal bank',colors[0]),('second_bank','Independent bank',colors[3]))):
            row=comparison(summary[bank],'H1 minus H0')
            mean=row['differences']['energy']
            low,high=row['ci95']['energy']
            ax.errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt='s',capsize=4,color=color)
        ax.axvline(0,color=muted,linewidth=1)
        ax.set_yticks([0,1],['Principal','Independent'])
        ax.invert_yaxis()
        ax.set_xlabel('H1 − H0 energy score')
        ax.set_title('MONTE CARLO STABILITY',loc='left',fontsize=10,color=parchment)
        fig.suptitle('EVOPOLIS / CONDITIONAL RESPONSE: TWO PREDICTIVE TARGETS',x=.07,ha='left',fontsize=15,fontweight='bold')
        fig.text(.07,.05,'21 groups: 5 Equal, 8 Mixed, 8 Proportional; source rounds 5–14 after a five-round observed prefix.\nBars: paired 95% human-group intervals, conditional on three fitted seeds. Both primary H1−H0 intervals must be below zero.',color=muted)
        fig.subplots_adjust(left=.1,right=.98,top=.84,bottom=.24,wspace=.56)
        _save(fig,'task05-predictive-comparisons',figures,plt,background)

        arrays,manifest=load()
        selected={m:min((g for g in manifest['groups'] if g['split']=='test' and g['mechanism']==m
                         and (arrays['offers'][g['index'],:,5]>=1).any()),key=lambda g:g['index']) for m in MECHANISMS}
        cases={}
        details=[]
        archive=results/'forecasts.sqlite3'
        with closing(sqlite3.connect(archive.resolve().as_uri()+'?mode=ro',uri=True)) as database:
            for row in database.execute('SELECT metadata FROM cells ORDER BY id'):
                meta=json.loads(row[0])
                if (meta['bank']=='main' and meta['origin']==5 and int(meta.get('seed',meta.get('training_seed')))==17
                        and meta['group_index']==selected[meta['mechanism']]['index']):
                    _,_,paths=get_cell(meta['id'],include_paths=True,path=archive)
                    cases[(meta['mechanism'],meta['family'])]=(meta,paths)
        fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey='row')
        fig.patch.set_facecolor(background)
        _decorate(axes,palette)
        for col,mechanism in enumerate(MECHANISMS):
            group=selected[mechanism]
            index=group['index']
            for family,color in zip(FAMILIES,colors):
                meta,paths=cases[(mechanism,family)]
                x=np.arange(6,6+paths['pool_after'].shape[1])
                for ax,variable in ((axes[0,col],'pool_after'),(axes[1,col],'window_surplus')):
                    low,median,high=np.quantile(paths[variable],[.1,.5,.9],axis=0)
                    ax.fill_between(x,low,high,color=color,alpha=.065)
                    ax.plot(x,median,color=color,linestyle='-' if family.startswith('H') else '--',linewidth=1.5,label=family)
                details.append({'family':family,'seed':17,'mechanism':mechanism,'group_key':group['key'],
                                'group_index':index,'cell_id':meta['id'],'origin':5})
            axes[0,col].plot(np.arange(1,26),arrays['next_pool'][index,:25],color=parchment,linewidth=2,label='Recorded human')
            observed_surplus=np.cumsum(arrays['surplus'][index,:,5:25].sum(0))
            axes[1,col].plot(np.arange(5,26),np.r_[0,observed_surplus],color=parchment,linewidth=2,label='Recorded human')
            for ax in axes[:,col]:
                ax.axvline(5.5,color=colors[1],linewidth=1)
                ax.set_xlim(1,25)
                ax.set_xlabel('Source decision round (1-based)')
            axes[0,col].set_title(f'{mechanism.upper()} / {group["launch_id"]}',loc='left',fontsize=10,color=parchment)
            axes[0,col].set_ylim(0,205)
        axes[0,0].set_ylabel('Pool after decision (units)')
        axes[1,0].set_ylabel('Cumulative group surplus\nfrom forecast origin (units)')
        axes[0,0].legend(frameon=False,labelcolor=parchment,fontsize=8,ncol=2)
        fig.suptitle('EVOPOLIS / HUMAN FUTURES AND GENERATED RESOURCE PATHS',x=.07,ha='left',fontsize=15,fontweight='bold')
        fig.text(.07,.04,'First eligible test group by source index in each mechanism, all four families, seed 17; cases do not select for survival or fit.\nLines: branch medians; shaded bands: pointwise 10th–90th percentiles of 64 branches. Paths shown separately from endpoint scores.',color=muted)
        fig.subplots_adjust(left=.07,right=.98,top=.89,bottom=.16,hspace=.3,wspace=.2)
        _save(fig,'task05-forecast-trajectories',figures,plt,background)
        write_json(results/'illustrated_cases.json',{'selection':'Smallest source-index eligible test group per baseline mechanism at k=5; seed17; all four families; no outcome criterion.','cases':details})

        bin_path=results/'collective_calibration_bins.csv'
        if bin_path.exists():
            bins=list(csv.DictReader(bin_path.open()))
            fig,ax=plt.subplots(figsize=(8,6))
            fig.patch.set_facecolor(background)
            _decorate([ax],palette)
            ax.plot([0,1],[0,1],linestyle='--',color=muted,linewidth=1)
            for family,color in zip(FAMILIES,colors):
                rows=[r for r in bins if r.get('procedure',r.get('family'))==family and r['mechanism']=='all' and r['predicted']]
                ax.plot([float(r['predicted']) for r in rows],[float(r['observed']) for r in rows],color=color,marker='s',label=family)
            ax.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted renewal probability',ylabel='Observed renewal fraction')
            ax.legend(frameon=False,labelcolor=parchment)
            fig.suptitle('EVOPOLIS / RECORDED-STATE RENEWAL',x=.13,ha='left',fontsize=14,fontweight='bold')
            fig.text(.13,.04,'Posterior-predictive resident PMFs; exact convolution.\nRenewal: 1.4 × total returns ≥ allocated resources.\nFixed 0.1 bins; groups/mechanisms weighted equally.',color=muted,fontsize=9)
            fig.subplots_adjust(left=.13,right=.96,top=.87,bottom=.23)
            _save(fig,'task05-renewal-calibration',figures,plt,background)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('diagnostics','plot'))
    args=parser.parse_args()
    if args.command=='diagnostics':
        # Reuse the existing shared numerical lock; never overlap training.
        import fcntl
        with (ROOT/'data/cache/task03/training.lock').open('w') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            generate_diagnostics()
    else:
        plot_mechanisms()
        plot_outcomes()


if __name__=='__main__':
    main()
