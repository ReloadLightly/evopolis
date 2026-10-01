import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from evopolis.conditional_models import (ConditionalModel,initial_history,features,update_history,
    tilted_log_mass,prepare_group,group_loss,prequential, infer_prefix)
from evopolis.conditional_train import new_fit,restore,snapshot,train_epoch
from evopolis.forecast_train import assert_exact

torch.set_num_threads(1)


def example():
    offers=np.array([[1.8,4.,0.,10.],[1.,3.,.8,9.],[4.,0.,8.,9.],[5.,5.,5.,5.]])
    actions=np.array([[1,2,0,7],[0,1,0,8],[2,0,3,7],[1,3,2,4]])
    return {'offers':offers.T[None], 'y':actions.T[None], 'pool':offers.sum(-1)[None]}


class ConditionalChecks(unittest.TestCase):
    def test_legal_mass_reduction_and_signed_derivative(self):
        torch.manual_seed(17)
        m=ConditionalModel('P0');h=initial_history()
        e=torch.tensor([0.,.8,1.8,50.],dtype=torch.float64);pool=e.sum()
        base=m.probs(pool,e,h)
        self.assertEqual(float(base[0,0]),1.)
        self.assertEqual(float(base[1,0]),1.)
        self.assertTrue(torch.allclose(base.sum(-1),torch.ones(4,dtype=torch.float64),atol=1e-12))
        for p in range(4):self.assertEqual(float(base[p,int(e[p].floor())+1:].sum()),0.)
        for family in ('P1','H0','H1'):
            other=ConditionalModel(family);other.load_state_dict(m.state_dict())
            self.assertTrue(torch.equal(base,other.probs(pool,e,h,u=0.)))
        raw=m.head(features(pool,e,h))[3]
        q=torch.arange(201,dtype=torch.float64)/e[3]
        for beta in (-2.,2.):
            b=torch.tensor(.4,dtype=torch.float64,requires_grad=True)
            probs=tilted_log_mass(raw,e[3],beta*(b-.5)).exp()
            mean=(probs*q).sum();deriv=torch.autograd.grad(mean,b)[0]
            variance=(probs*(q-mean)**2).sum()
            self.assertAlmostEqual(float(deriv),float(beta*variance),places=12)

    def test_simultaneous_history_and_peer_order(self):
        h=initial_history();e=torch.tensor([1.8,0.8,2.,0.]);c=torch.tensor([1.,0.,2.,0.])
        after=update_history(h,e,c,.5)
        self.assertAlmostEqual(float(after['own_trace'][0]),.25+.5/1.8,places=7)
        self.assertAlmostEqual(float(after['peer_trace'][0]),.75)
        self.assertAlmostEqual(float(after['peer_trace'][2]),.25+.5/1.8,places=7)
        self.assertEqual(float(after['own_trace'][1]),.5)
        zero=update_history(after,torch.zeros(4),torch.zeros(4),.5)
        self.assertTrue(torch.equal(zero['peer_trace'],after['peer_trace']))
        perm=[0,3,1,2]
        self.assertTrue(torch.equal(features(e.sum(),e,h)[0],features(e.sum(),e[perm],{k:v[perm] for k,v in h.items()})[0]))

    def test_marginal_sequence_equals_prequential_and_past_only(self):
        a=example();d=prepare_group(a,0);m=ConditionalModel('H1')
        with torch.no_grad():m.beta_parameter.fill_(-.7)
        result=prequential(m,a,0,41,stop=4)
        self.assertAlmostEqual(float(group_loss(m,d,41)),float(-result['log_prob'].sum()/d['count']),places=12)
        self.assertLess(np.max(np.abs(result['posterior_weights'].sum(-1)-1)),1e-12)
        poisoned=copy.deepcopy(a);poisoned['y'][:,:,2:]=0;poisoned['offers'][:,:,2:]=100
        prefix=infer_prefix(m,a,0,2,41);other=infer_prefix(m,poisoned,0,2,41)
        assert_exact(prefix,other)
        history=initial_history()
        for t in range(4):
            z=features(torch.tensor(a['pool'][0,t]),torch.tensor(a['offers'][0,:,t]),history)
            self.assertTrue(torch.allclose(z[:,:6],d['z'][t,:,:6]))
            history=update_history(history,torch.tensor(a['offers'][0,:,t]),torch.tensor(a['y'][0,:,t]),m.eta)

    def test_whole_group_weighting_and_exact_resume(self):
        a=example();d=prepare_group(a,0);data={i:d for i in range(8)}
        model,opt,shuffle=new_fit('H1',17)
        train_epoch(model,opt,data,list(data),shuffle)
        saved=snapshot(model,opt,shuffle)
        loss=train_epoch(model,opt,data,list(data),shuffle)
        target=snapshot(model,opt,shuffle)
        from evopolis.behavior_train import atomic_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'last.pt'
            atomic_checkpoint(path,saved)
            reloaded=torch.load(path,map_location='cpu',weights_only=True)
        model,opt,shuffle=restore('H1',17,reloaded)
        self.assertEqual(loss,train_epoch(model,opt,data,list(data),shuffle))
        assert_exact(target,snapshot(model,opt,shuffle))
        # Unequal eligible counts and distinct choices: averaging choices across
        # groups would give the short group too little weight.
        shorter=copy.deepcopy(a);shorter['offers'][:,:,1:]=0;shorter['y'][:,:,1:]=0
        short=prepare_group(shorter,0)
        from evopolis.conditional_train import score
        left,right=float(group_loss(model,d)),float(group_loss(model,short))
        equal=(left+right)/2
        pooled=(left*d['count']+right*short['count'])/(d['count']+short['count'])
        self.assertGreater(abs(equal-pooled),.001)
        self.assertAlmostEqual(score(model,{0:d,1:short},[0,1]),equal,places=12)

    def test_zero_sigma_marginal_and_history_kernel(self):
        a=example();d=prepare_group(a,0)
        plain=ConditionalModel('P0');persistent=ConditionalModel('H1')
        persistent.load_state_dict(plain.state_dict())
        with torch.no_grad():persistent.raw_sigma.fill_(-torch.inf)
        self.assertAlmostEqual(float(group_loss(plain,d)),float(group_loss(persistent,d)),places=12)
        x=prequential(plain,a,0,stop=4);y=prequential(persistent,a,0,stop=4)
        self.assertLess(float(np.max(np.abs(x['pmfs']-y['pmfs']))),1e-12)
        # Non-default eta and missing observations exercise the closed-form
        # training traces against separately stepped, inference-time histories.
        with torch.no_grad():persistent.raw_eta.fill_(-1.3)
        result=prequential(persistent,a,0,stop=4)
        eta=persistent.eta
        trace=[.5*(1-eta)**count+eta*((1-eta)**lag*values).sum(1)
               for count,lag,values in d['kernels']]
        for t,h in enumerate(result['histories']):
            self.assertTrue(np.allclose(trace[0][t].detach().numpy(),h['own_trace'],atol=1e-14))
            self.assertTrue(np.allclose(trace[1][t].detach().numpy(),h['peer_trace'],atol=1e-14))


if __name__=='__main__':unittest.main()
