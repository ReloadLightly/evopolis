"""Task 05 matched conditional-history models; no changes to Task 03 emission.

Population weights are fitted; traces are deterministic memories, and resident
posterior updates are Bayesian inference under frozen population parameters.
"""
from functools import lru_cache
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .behavior_models import emission_log_prob

FAMILIES = ('P0', 'P1', 'H0', 'H1')
FEATURES = ('intercept', 'offer_self/200', 'peer_offer_low/200',
            'peer_offer_middle/200', 'peer_offer_high/200', 'pool/200',
            'own_trace', 'previous_valid_fraction', 'previous_own_valid',
            'previous_eligible_peers/3', 'own_exposed', 'peer_exposed')
HISTORY = ('own_trace', 'peer_trace', 'previous_valid_fraction',
           'previous_own_valid', 'previous_eligible_peers', 'own_exposed', 'peer_exposed')


def initial_history(shape=()):
    return {k: torch.full((*shape, 4), .5 if k.endswith('trace') else 0., dtype=torch.float64)
            for k in HISTORY}


def features(pool, offers, history):
    peers = torch.stack([offers[..., [j for j in range(4) if j != i]].sort(-1).values
                         for i in range(4)], -2) / 200
    return torch.cat((torch.ones_like(offers)[..., None], offers[..., None]/200, peers,
                      (pool[..., None].expand_as(offers)/200)[..., None],
                      torch.stack([history[k] / (3 if k == 'previous_eligible_peers' else 1)
                                   for k in HISTORY if k != 'peer_trace'], -1)), -1)


def update_history(history, offers, actions, eta):
    valid = offers >= 1
    fraction = torch.where(valid, actions / offers.clamp_min(1), 0.)
    count = valid.sum(-1, keepdim=True) - valid.to(torch.int64)
    peer = (fraction.sum(-1, keepdim=True) - fraction) / count.clamp_min(1)
    return {'own_trace': torch.where(valid, (1-eta)*history['own_trace']+eta*fraction, history['own_trace']),
            'peer_trace': torch.where(count > 0, (1-eta)*history['peer_trace']+eta*peer, history['peer_trace']),
            'previous_valid_fraction': fraction, 'previous_own_valid': valid.double(),
            'previous_eligible_peers': count.double(),
            'own_exposed': torch.maximum(history['own_exposed'], valid.double()),
            'peer_exposed': torch.maximum(history['peer_exposed'], (count > 0).double())}


@lru_cache(None)
def quadrature(nodes):
    x, w = np.polynomial.hermite.hermgauss(nodes)
    return torch.tensor(x*math.sqrt(2), dtype=torch.float64), torch.tensor(np.log(w/math.sqrt(math.pi)), dtype=torch.float64)


def base_log_mass(raw, offers, max_support=200):
    n = offers.floor()
    actions = torch.arange(max_support+1, dtype=torch.float64)
    ns = n[..., None].expand(*n.shape, max_support+1)
    rs = raw[..., None, :].expand(*raw.shape[:-1], max_support+1, 5)
    logp = emission_log_prob(rs, ns, torch.minimum(actions, ns))
    return torch.where(actions <= ns, logp, -torch.inf)


def tilted_log_mass(raw, offers, v, max_support=200):
    base = base_log_mass(raw, offers, max_support)
    fraction = torch.arange(max_support+1, dtype=torch.float64) / offers.clamp_min(1)[..., None]
    logits = base + v[..., None]*fraction
    return logits - torch.logsumexp(logits, -1, keepdim=True)


class ConditionalModel(nn.Module):
    def __init__(self, family):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(family)
        self.family = family
        # Explicit intercept is column zero, so no duplicate affine bias.
        # Seeded torch uniform U(-1/sqrt(12),1/sqrt(12)), all families identical.
        self.head = nn.Linear(12, 5, bias=False, dtype=torch.float64)
        self.raw_eta = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.beta_parameter = nn.Parameter(torch.tensor(0., dtype=torch.float64), requires_grad=family.endswith('1'))
        self.raw_sigma = nn.Parameter(torch.tensor(math.log(math.expm1(.5)), dtype=torch.float64), requires_grad=family.startswith('H'))

    @property
    def eta(self): return self.raw_eta.sigmoid()
    @property
    def beta(self): return self.beta_parameter if self.family.endswith('1') else self.beta_parameter*0
    @property
    def sigma(self): return F.softplus(self.raw_sigma) if self.family.startswith('H') else self.raw_sigma*0

    def latent_grid(self, nodes=21):
        if self.family.startswith('P'):
            return torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
        x, w = quadrature(nodes)
        return x*self.sigma, w

    def probs(self, pool, offers, history, u=0.):
        raw = self.head(features(pool, offers, history))
        return tilted_log_mass(raw, offers, self.beta*(history['peer_trace']-.5)+u).exp()


def prepare_group(arrays, index):
    """Fixed observed inputs plus past-only trace kernels, without a test fit."""
    offers = torch.tensor(arrays['offers'][index].T, dtype=torch.float64)
    actions = torch.tensor(arrays['y'][index].T, dtype=torch.float64)
    pool = torch.tensor(arrays['pool'][index], dtype=torch.float64)
    valid = offers >= 1
    fractions = torch.where(valid, actions / offers.clamp_min(1), 0.)
    peer_count = valid.sum(-1, keepdim=True)-valid.long()
    peer_fraction = (fractions.sum(-1, keepdim=True)-fractions)/peer_count.clamp_min(1)
    history = initial_history()
    zs=[]
    for t in range(len(pool)):
        zs.append(features(pool[t], offers[t], history))
        history = update_history(history, offers[t], actions[t], .5)
    z = torch.stack(zs)
    kernels=[]
    # count before each choice; each valid earlier observation gets eta*(1-eta)^lag.
    for values, mask in ((fractions,valid),(peer_fraction,peer_count>0)):
        before = mask.long().cumsum(0)-mask.long()
        lag = before[:,None,:] - before[None,:,:] - 1
        past = torch.arange(len(pool))[None,:,None] < torch.arange(len(pool))[:,None,None]
        include = past & mask[None,:,:]
        kernels.append((before, lag.clamp_min(0), values[None,:,:]*include))
    return {'offers':offers, 'actions':actions, 'pool':pool,'z':z,'kernels':kernels,'valid':valid,
            'count':int(valid.sum()),'max_support':int(offers.max().floor())}


def group_components(model, data, nodes=21):
    eta=model.eta
    traces=[.5*(1-eta)**before + eta*((1-eta)**lag*values).sum(1)
            for before,lag,values in data['kernels']]
    z=data['z'].clone()
    z[...,6]=traces[0]
    raw=model.head(z)
    mask=data['valid']
    selected_raw=raw[mask]
    offers=data['offers'][mask]
    actions=data['actions'][mask]
    u, logw=model.latent_grid(nodes)
    v=model.beta*(traces[1][mask]-.5)[:,None]+u
    if model.family=='P0':
        target=emission_log_prob(selected_raw, offers.floor(), actions)[:,None]
    else:
        base=base_log_mass(selected_raw,offers,data['max_support'])
        q=(torch.arange(data['max_support']+1,dtype=torch.float64)/offers[:,None]).clamp_max(1)
        # Log-sum-exp via a bounded exponential shift, preventing overflow while
        # avoiding redundant log-gamma calculations at each quadrature node.
        shift=v.clamp_min(0)
        weights=torch.exp(v[...,None]*q[:,None,:]-shift[...,None])
        logz=(weights*base.exp()[:,None,:]).sum(-1).log()+shift
        target=emission_log_prob(selected_raw,offers.floor(),actions)[:,None]+v*(actions/offers)[:,None]-logz
    full=raw.new_zeros((*mask.shape,len(u)))
    full[mask]=target
    return full, logw


def group_loss(model, data, nodes=21):
    if not data['count']:
        return None
    logs, logw=group_components(model,data,nodes)
    return -torch.logsumexp(logs.sum(0)+logw, -1).sum()/data['count']


@torch.no_grad()
def prequential(model, arrays, index, nodes=21, stop=40):
    history=initial_history()
    u, prior=model.latent_grid(nodes)
    logw=prior.expand(4,-1).clone()
    pmfs=[]; log_prob=[]; histories=[]; posteriors=[logw.exp().numpy().copy()]
    for t in range(stop):
        offers=torch.tensor(arrays['offers'][index,:,t],dtype=torch.float64)
        actions=torch.tensor(arrays['y'][index,:,t],dtype=torch.int64)
        pool=torch.tensor(arrays['pool'][index,t],dtype=torch.float64)
        raw=model.head(features(pool,offers,history))
        per_node=tilted_log_mass(raw[:,None,:].expand(4,len(u),5),offers[:,None].expand(4,len(u)),
                                 model.beta*(history['peer_trace'][:,None]-.5)+u)
        marginal=torch.logsumexp(per_node+logw[...,None],1)
        lp=marginal[torch.arange(4),actions]
        chosen=per_node[torch.arange(4),:,actions]
        logw=logw+chosen
        logw=logw-torch.logsumexp(logw,-1,keepdim=True)
        pmfs.append(marginal.exp().numpy());log_prob.append(lp.numpy())
        histories.append({k:v.numpy().copy() for k,v in history.items()})
        posteriors.append(logw.exp().numpy().copy())
        history=update_history(history,offers,actions,model.eta)
    return {'pmfs':np.asarray(pmfs),'log_prob':np.asarray(log_prob), 'histories':histories,
            'posterior_weights':np.asarray(posteriors),'latent_nodes':u.numpy(), 'history':history,
            'log_weights':logw}


@torch.no_grad()
def infer_prefix(model, arrays, index, origin, nodes=21):
    result=prequential(model,arrays,index,nodes,stop=origin)
    return {'history':result['history'],'nodes':torch.tensor(result['latent_nodes']),
            'weights':result['log_weights'].exp(), 'log_weights':result['log_weights']}
