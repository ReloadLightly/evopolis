"""Frozen twelve-fit Task 05 budget with atomic, exact continuation."""
import argparse
import copy
import datetime
import fcntl
import json
import resource
import time
import numpy as np
import torch
from .behavior_data import load, ROOT
from .behavior_train import atomic_checkpoint, atomic_json, runtime_setup
from .conditional_models import ConditionalModel, FAMILIES, prepare_group, group_loss
from .sources import sha256

RESULTS=ROOT/'results/task05'
CONFIG=ROOT/'configs/task05.json'


def preserved():
    files=json.loads((RESULTS/'preservation.json').read_text())['files']
    changed=[p for p,d in files.items() if sha256(ROOT/p)!=d]
    if changed: raise ValueError(f'Protected sources changed: {changed}')
    return len(files)


def identity():
    config=json.loads(CONFIG.read_text())
    value={'configuration':config,'config_sha256':sha256(CONFIG),
           'protocol_sha256':sha256(ROOT/'docs/tasks/05-conditional-responses.md'),
           'source_sha256':json.loads((ROOT/'results/task03/split.json').read_text())['source_sha256'],
           'split_sha256':sha256(ROOT/'results/task03/split.json'),
           'code_sha256':{p:sha256(ROOT/p) for p in ('evopolis/conditional_models.py','evopolis/conditional_train.py','evopolis/behavior_models.py')},
           'software':{'torch':str(torch.__version__),'numpy':np.__version__}}
    path=RESULTS/'frozen_training.json'
    if path.exists():
        frozen=json.loads(path.read_text())['identity']
        # Task completion status is documentation, not a new protocol.
        value['protocol_sha256']=frozen['protocol_sha256']
        if value!=frozen: raise ValueError('Frozen training identity changed')
    else:
        atomic_json(path,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'identity':value})
    return value


def new_fit(family,seed):
    torch.manual_seed(seed)
    model=ConditionalModel(family)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
    return model,optimizer,np.random.Generator(np.random.PCG64(seed))


@torch.no_grad()
def score(model,data,indices,nodes=21):
    values=[float(group_loss(model,data[i],nodes)) for i in indices if data[i]['count']]
    return float(np.mean(values))


def train_epoch(model,optimizer,data,indices,shuffle,nodes=21):
    order=shuffle.permutation(indices)
    values=[]
    for begin in range(0,len(order),8):
        optimizer.zero_grad(set_to_none=True)
        ids=[i for i in order[begin:begin+8] if data[i]['count']]
        for i in ids:
            loss=group_loss(model,data[i],nodes)
            values.append(float(loss.detach()))
            (loss/len(ids)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
    return float(np.mean(values))


def snapshot(model,optimizer,shuffle):
    return copy.deepcopy({'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                          'torch_rng_state':torch.get_rng_state(),'shuffle_state':shuffle.bit_generator.state})


def restore(family,seed,saved):
    model,optimizer,shuffle=new_fit(family,seed)
    model.load_state_dict(saved['model_state'])
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer_state']))
    torch.set_rng_state(saved['torch_rng_state'])
    shuffle.bit_generator.state=saved['shuffle_state']
    return model,optimizer,shuffle


def fit(family,seed,data,train_ids,val_ids,contract,nodes=21):
    folder=RESULTS/'weights'/f'{family}_{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    model,optimizer,shuffle=new_fit(family,seed)
    logs=[];first=1;best=float('inf');elapsed=0.;cpu_elapsed=0.;selected_epoch=0
    initial=score(model,data,val_ids,nodes)
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',weights_only=True,map_location='cpu')
        if saved['metadata']!=contract or saved['integration_nodes']!=nodes: raise ValueError('Resume identity changed')
        model,optimizer,shuffle=restore(family,seed,saved)
        logs=saved['logs'];first=saved['epoch']+1;best=saved['best_validation_score']
        selected_epoch=saved['selected_epoch'];elapsed=saved['elapsed_seconds'];cpu_elapsed=saved['cpu_seconds']
        initial=saved['initial_validation_nll']
    started=time.monotonic();cpu=time.process_time()
    for epoch in range(first,481):
        tick=time.monotonic()
        train=train_epoch(model,optimizer,data,train_ids,shuffle,nodes)
        validation=score(model,data,val_ids,nodes)
        logs.append({'epoch':epoch,'train_nll':train,'validation_nll':validation,'seconds':time.monotonic()-tick,
                     'beta':float(model.beta.detach()),'eta':float(model.eta.detach()),'sigma':float(model.sigma.detach())})
        improved=validation<best
        if improved: best=validation;selected_epoch=epoch
        base=dict(snapshot(model,optimizer,shuffle),family=family,seed=seed,epoch=epoch,metadata=contract,
                  integration_nodes=nodes,validation_score=validation,best_validation_score=best,
                  selected_epoch=selected_epoch,logs=logs,initial_validation_nll=initial,
                  elapsed_seconds=elapsed+time.monotonic()-started,cpu_seconds=cpu_elapsed+time.process_time()-cpu)
        if improved:atomic_checkpoint(folder/'best.pt',base)
        atomic_checkpoint(folder/'last.pt',base)
        if epoch==1 or epoch%10==0:
            atomic_json(folder/'training.json',logs)
            print(f'{family}/{seed} epoch {epoch}/480 train {train:.6f} val {validation:.6f} best {best:.6f} '
                  f'eta {float(model.eta.detach()):.3f} beta {float(model.beta.detach()):.3f} sigma {float(model.sigma.detach()):.3f}; '
                  f'epoch {time.monotonic()-tick:.2f}s',flush=True)
    atomic_json(folder/'training.json',logs)
    saved=torch.load(folder/'best.pt',weights_only=True,map_location='cpu')
    model.load_state_dict(saved['model_state'])
    reloaded=score(model,data,val_ids,nodes)
    assert abs(reloaded-best)<1e-12
    assert saved['epoch']==min(logs,key=lambda v:(v['validation_nll'],v['epoch']))['epoch']
    atomic_json(folder/'metadata.json',{'family':family,'seed':seed,'selected_epoch':saved['epoch'],
               'validation_score':best,'reloaded_validation_score':reloaded,'initial_validation_nll':initial,
               'integration_nodes':nodes,'beta':float(model.beta.detach()),'eta':float(model.eta.detach()),
               'sigma':float(model.sigma.detach()),'best_sha256':sha256(folder/'best.pt'),
               'last_sha256':sha256(folder/'last.pt'),'elapsed_seconds':elapsed+time.monotonic()-started,
               'cpu_seconds':cpu_elapsed+time.process_time()-cpu,'epochs':480,
               'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=FAMILIES)
    parser.add_argument('--seed',type=int,choices=(17,29,43))
    args=parser.parse_args()
    if not (RESULTS/'resource_before_torch.json').exists():raise ValueError('Profile before importing Torch')
    runtime_setup();preserved();contract=identity()
    arrays,manifest=load()
    ids={s:[g['index'] for g in manifest['groups'] if g['split']==s] for s in ('train','validation')}
    data={i:prepare_group(arrays,i) for i in ids['train']+ids['validation']}
    atomic_json(RESULTS/'likelihood_cohort.json',{s:{'groups':len(v),'zero_choice_groups':sum(not data[i]['count'] for i in v)} for s,v in ids.items()})
    start=time.monotonic()
    with (ROOT/'data/cache/task03/training.lock').open('w') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for family in ([args.family] if args.family else FAMILIES):
            for seed in ([args.seed] if args.seed else (17,29,43)):
                fit(family,seed,data,ids['train'],ids['validation'],contract)
    complete=list((RESULTS/'weights').glob('*/metadata.json'))
    if len(complete)==12:
        atomic_json(RESULTS/'training_complete.json',{'fits':12,'epochs_per_fit':480,
                    'elapsed_seconds':sum(json.loads(p.read_text())['elapsed_seconds'] for p in complete),
                    'cpu_seconds':sum(json.loads(p.read_text())['cpu_seconds'] for p in complete),
                    'peak_rss_kib':max(json.loads(p.read_text())['peak_rss_kib'] for p in complete),
                    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'invocation_seconds':time.monotonic()-start})


if __name__=='__main__':main()
