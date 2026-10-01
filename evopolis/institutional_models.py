"""Task 06 public institution features and fresh FA response families.

Task 03's emission and Task 05's history and whole-sequence likelihood are
reused unchanged. Institution signals use only current public offers and the
previous round's public actions; no rule label or current target is an input.
"""

import math

import numpy as np
import torch
from torch import nn

from .behavior_models import BehaviorModel
from .conditional_models import (
    ConditionalModel, base_log_mass, features, group_components, group_loss,
    initial_history, prepare_group, tilted_log_mass, update_history,
)

FAMILIES = ("FA-GRU", "FA-P0", "FA-H0")


def signal_step(offers, previous_actions, last, undefined):
    """Return (s, undefined, clipped) over any leading game dimensions.

    ``undefined`` means no valid signal has ever been observed. An undefined
    current regression carries both the last signal and this state forward.
    The final axis contains the four residents. Inputs are not modified.
    """
    if offers.shape[-1] != 4 or previous_actions.shape != offers.shape:
        raise ValueError("Institution signal requires matching four-resident arrays")
    offers = offers.to(dtype=torch.float64)
    previous_actions = previous_actions.to(dtype=torch.float64)
    last = torch.as_tensor(last, dtype=torch.float64, device=offers.device)
    undefined = torch.as_tensor(undefined, dtype=torch.bool, device=offers.device)
    total_e = offers.sum(-1)
    total_c = previous_actions.sum(-1)
    a = offers / torch.where(total_e > 0, total_e, 1.)[..., None]
    b = previous_actions / torch.where(total_c > 0, total_c, 1.)[..., None]
    denominator = ((b - .25) ** 2).sum(-1)
    defined = (total_c > 0) & (total_e > 0) & (denominator >= 1e-9)
    candidate = ((b - .25) * (a - .25)).sum(-1) / torch.where(defined, denominator, 1.)
    clipped = defined & ((candidate < -1.) | (candidate > 2.))
    value = torch.where(defined, candidate.clamp(-1., 2.), last)
    return value, undefined & ~defined, clipped


def signal_sequence(offers, actions):
    """Past-only signals for a [round, resident] observed sequence.

    NumPy inputs return NumPy arrays; tensor inputs return tensors. Round zero
    has signal .5 and undefined=True, independent of every recorded action.
    """
    numpy_input = not isinstance(offers, torch.Tensor)
    offers = torch.as_tensor(offers, dtype=torch.float64)
    actions = torch.as_tensor(actions, dtype=torch.float64)
    if offers.ndim != 2 or offers.shape[-1] != 4 or actions.shape != offers.shape:
        raise ValueError("Expected round-by-four offers and actions")
    last = offers.new_tensor(.5)
    undefined = torch.tensor(True, device=offers.device)
    previous = torch.zeros(4, dtype=torch.float64, device=offers.device)
    values, flags, clipping = [], [], []
    for time in range(len(offers)):
        last, undefined, clipped = signal_step(offers[time], previous, last, undefined)
        values.append(last)
        flags.append(undefined)
        clipping.append(clipped)
        previous = actions[time]
    result = (torch.stack(values), torch.stack(flags), torch.stack(clipping))
    return tuple(v.numpy() for v in result) if numpy_input else result


def augment_arrays(arrays):
    """Append unscaled s and undefined to the original normalized nine inputs."""
    signals = [signal_sequence(e.T, c.T) for e, c in zip(arrays["offers"], arrays["y"])]
    s = np.stack([row[0] for row in signals])
    undefined = np.stack([row[1] for row in signals])
    clipping = np.stack([row[2] for row in signals])
    extra = np.stack((s, undefined), -1)[:, None].repeat(4, axis=1)
    return dict(arrays, x=np.concatenate((arrays["x"], extra), -1),
                institution_signal=s, institution_undefined=undefined,
                institution_clipping=clipping)


def features_fa(pool, offers, history, signal, undefined):
    base = features(pool, offers, history)
    signal = torch.as_tensor(signal, dtype=base.dtype, device=base.device)
    undefined = torch.as_tensor(undefined, dtype=base.dtype, device=base.device)
    extra = torch.stack((signal, undefined), -1)
    extra = extra[..., None, :].expand(*base.shape[:-1], 2)
    return torch.cat((base, extra), -1)


class FAGRU(BehaviorModel):
    task06_fa = True
    fa_family = "FA-GRU"

    def __init__(self):
        nn.Module.__init__(self)
        self.family = "recurrent"
        self.memory = nn.GRU(11, 32, batch_first=True)
        self.head = nn.Linear(32, 5)

    def forward(self, x, hidden=None):
        if x.ndim != 3 or x.shape[-1] != 11:
            raise ValueError("Expected resident sequences, time, eleven inputs")
        sequence, hidden = self.memory(x, hidden)
        return self.head(sequence), hidden


class FAConditional(ConditionalModel):
    task06_fa = True

    def __init__(self, base_family):
        if base_family not in ("P0", "H0"):
            raise ValueError("FA conditional families are P0 and H0")
        nn.Module.__init__(self)
        self.family = base_family
        self.fa_family = "FA-" + base_family
        # Fresh 14-column initialization, explicit intercept, no duplicate bias.
        self.head = nn.Linear(14, 5, bias=False, dtype=torch.float64)
        self.raw_eta = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.beta_parameter = nn.Parameter(torch.tensor(0., dtype=torch.float64), requires_grad=False)
        self.raw_sigma = nn.Parameter(torch.tensor(math.log(math.expm1(.5)), dtype=torch.float64),
                                      requires_grad=base_family == "H0")

    def probs(self, pool, offers, history, u=0., *, signal, undefined):
        raw = self.head(features_fa(pool, offers, history, signal, undefined))
        effect = torch.as_tensor(u, dtype=raw.dtype, device=raw.device)
        return tilted_log_mass(raw, offers, effect).exp()


def make_model(family):
    if family == "FA-GRU":
        return FAGRU()
    if family in ("FA-P0", "FA-H0"):
        return FAConditional(family[3:])
    raise ValueError(f"Unknown Task 06 FA family: {family}")


def prepare_fa_group(arrays, index):
    data = prepare_group(arrays, index)
    signal, undefined, clipping = signal_sequence(data["offers"], data["actions"])
    extra = torch.stack((signal, undefined.double()), -1)[:, None, :].expand(-1, 4, -1)
    data["z"] = torch.cat((data["z"], extra), -1)
    data.update(institution_signal=signal, institution_undefined=undefined,
                institution_clipping=clipping)
    return data


@torch.no_grad()
def prequential_fa(model, arrays, index, nodes=41, stop=40, tilt=(0., 0.)):
    """Correct marginal prediction and posterior updates under tilted H fits.

    The calibration tilt is applied to each conditional-on-resident-effect
    emission before integrating, so its history-dependent posterior matches
    the same generative model used in closed-loop simulation.
    """
    if stop < 0 or stop > arrays["offers"].shape[-1]:
        raise ValueError("Invalid prediction horizon")
    history = initial_history()
    u, prior = model.latent_grid(nodes)
    logw = prior.expand(4, -1).clone()
    signal, undefined, clipping = signal_sequence(arrays["offers"][index].T, arrays["y"][index].T)
    pmfs, log_prob, histories = [], [], []
    posteriors = [logw.exp().numpy().copy()]
    for time in range(stop):
        offers = torch.tensor(arrays["offers"][index, :, time], dtype=torch.float64)
        actions = torch.tensor(arrays["y"][index, :, time], dtype=torch.int64)
        pool = torch.tensor(arrays["pool"][index, time], dtype=torch.float64)
        raw = model.head(features_fa(pool, offers, history, signal[time], undefined[time]))
        base = base_log_mass(raw, offers)
        q = torch.arange(201, dtype=torch.float64) / offers.clamp_min(1)[:, None]
        v = u[None, :] + float(tilt[0]) + float(tilt[1]) * signal[time]
        per_node = base[:, None, :] + v[..., None] * q[:, None, :]
        per_node -= torch.logsumexp(per_node, -1, keepdim=True)
        marginal = torch.logsumexp(per_node + logw[..., None], 1)
        lp = marginal[torch.arange(4), actions]
        chosen = per_node[torch.arange(4), :, actions]
        logw += chosen
        logw -= torch.logsumexp(logw, -1, keepdim=True)
        pmfs.append(marginal.exp().numpy())
        log_prob.append(lp.numpy())
        histories.append({k: v.numpy().copy() for k, v in history.items()})
        posteriors.append(logw.exp().numpy().copy())
        history = update_history(history, offers, actions, model.eta)
    return {"pmfs": np.asarray(pmfs), "log_prob": np.asarray(log_prob),
            "histories": histories, "posterior_weights": np.asarray(posteriors),
            "latent_nodes": u.numpy(), "history": history, "log_weights": logw,
            "institution_signal": signal[:stop], "institution_undefined": undefined[:stop],
            "institution_clipping": clipping[:stop]}
