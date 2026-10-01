"""Training/validation-only numerical gate before Task 05 outcome access."""
import datetime
import argparse
import json
import resource
import time
import numpy as np
import torch
from .behavior_data import ROOT,load,write_json
from .behavior_train import runtime_setup
from .conditional_models import ConditionalModel,prequential
from .sources import sha256

RESULTS=ROOT/'results/task05'


def run(available_only=False):
    runtime_setup()
    arrays,manifest=load()
    groups=[g for g in manifest['groups'] if g['split'] in ('train','validation')]
    example_ids={min(g['index'] for g in groups if g['split']==s and g['mechanism']==m)
                 for s in ('train','validation') for m in ('Equal','Mixed','Proportional','M1')}
    started=time.monotonic();rows=[];examples=[];checks=[]
    for family in ('H0','H1'):
        for seed in (17,29,43):
            path=RESULTS/'weights'/f'{family}_{seed}'/'best.pt'
            if available_only and not (path.parent/'metadata.json').exists():
                continue
            saved=torch.load(path,weights_only=True,map_location='cpu')
            model=ConditionalModel(family);model.load_state_dict(saved['model_state']);model.eval()
            worst={'group_nll':0.,'resident_nll':0.,'prediction_fraction':0.,'tail_probability':0.,'posterior_normalization':0.}
            concentrated=None
            for group in groups:
                i=group['index'];mask=arrays['offers'][i].T>=1
                lo=prequential(model,arrays,i,21);hi=prequential(model,arrays,i,41)
                nlls=[]
                for result in (lo,hi):
                    nlls.append(-result['log_prob'].sum(0)/np.maximum(mask.sum(0),1))
                group_delta=abs(lo['log_prob'].sum()-hi['log_prob'].sum())/max(mask.sum(),1)
                resident_delta=float(np.max(np.abs(nlls[0]-nlls[1])))
                offers=arrays['offers'][i].T
                fraction=np.arange(201)[None,None,:]/np.maximum(offers[...,None],1)
                mean_delta=np.abs(((lo['pmfs']-hi['pmfs'])*fraction).sum(-1))
                tails=np.abs(((lo['pmfs']-hi['pmfs'])*(fraction>=.8)).sum(-1))
                norm=max(float(np.max(np.abs(r['posterior_weights'].sum(-1)-1))) for r in (lo,hi))
                w=hi['posterior_weights'];u=hi['latent_nodes']
                posterior_mean=(w*u).sum(-1)
                posterior_sd=np.sqrt((w*(u-posterior_mean[...,None])**2).sum(-1))
                position=np.unravel_index(np.argmin(posterior_sd),posterior_sd.shape)
                sd=float(posterior_sd[position])
                if concentrated is None or sd<concentrated['posterior_sd']:
                    weights=w[position]
                    concentrated={'key':group['key'],'split':group['split'],'prefix':int(position[0]),
                                  'resident':int(position[1]),'posterior_sd':sd,
                                  'posterior_mean':float(posterior_mean[position]),
                                  'prior_sigma':float(model.sigma.detach()),'nodes':41,
                                  'weights':weights.tolist(),'latent_nodes':u.tolist()}
                row={'family':family,'seed':seed,'group_index':i,'key':group['key'],'split':group['split'],
                     'group_nll':float(group_delta),'resident_nll':resident_delta,
                     'prediction_fraction':float(mean_delta[mask].max()) if mask.any() else 0.,
                     'tail_probability':float(tails[mask].max()) if mask.any() else 0.,'posterior_normalization':norm}
                rows.append(row)
                for key in worst:worst[key]=max(worst[key],row[key])
                if i in example_ids:
                    for prefix in (0,5,10,20,40):
                        for label,r in ((21,lo),(41,hi)):
                            w=r['posterior_weights'][prefix];u=r['latent_nodes']
                            mean=(w*u).sum(-1);sd=np.sqrt((w*(u-mean[:,None])**2).sum(-1))
                            examples.append({'family':family,'seed':seed,'key':group['key'],'split':group['split'],'prefix':prefix,
                                             'nodes':label,'mean':mean.tolist(),'sd':sd.tolist(),
                                             'effective_nodes':(1/(w*w).sum(-1)).tolist()})
            passed=all(worst[k]<.001 for k in ('group_nll','resident_nll','prediction_fraction','tail_probability')) and worst['posterior_normalization']<1e-12
            checks.append({'family':family,'seed':seed,'checkpoint_sha256':sha256(path),
                           'beta':float(model.beta.detach()),'eta':float(model.eta.detach()),'sigma':float(model.sigma.detach()),
                           'most_concentrated_train_validation_posterior':concentrated,
                           'nodes_compared':[21,41],'maxima':worst,'passed':passed})
            print(f'Integration {family}/{seed}: {worst}; passed={passed}',flush=True)
    report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'all128 train/validation groups; no test outcomes',
            'threshold_nats_and_fraction':.001,'tail_check':'P(C/e >= 0.8), also required to differ by <0.001',
            'checks':checks,'group_checks':rows,'prefix_examples':examples,
            'passed':bool(checks) and all(c['passed'] for c in checks),'all_six_persistent_fits_checked':len(checks)==6,'selected_nodes':21,
            'wall_seconds':time.monotonic()-started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(RESULTS/('integration_early_checks.json' if available_only else 'integration_checks.json'),report)
    if not report['passed']:raise RuntimeError('Quadrature failed; document correction and rerun affected fits before evaluation')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--available-only',action='store_true',help='Development-only early check of completed selected persistent fits; cannot open test evaluation')
    run(parser.parse_args().available_only)
