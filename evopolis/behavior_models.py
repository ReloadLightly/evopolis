"""Task 03's four fixed response families and common integer emission.

These are new EvoPolis fits, not upstream BC1. Observations follow the published
nine-input convention; the inflated beta-binomial is the Task 03 extension.
"""

import torch
from torch import nn
from torch.nn import functional as F


FAMILIES = ("constant", "linear", "feedforward", "recurrent")


class BehaviorModel(nn.Module):
    def __init__(self, family):
        super().__init__()
        self.family = family
        if family == "constant":
            self.raw = nn.Parameter(torch.randn(5) * 0.05)
        elif family == "linear":
            self.network = nn.Linear(9, 5)
        elif family == "feedforward":
            self.network = nn.Sequential(nn.Linear(9, 64), nn.ReLU(),
                                         nn.Linear(64, 52), nn.ReLU(), nn.Linear(52, 5))
        elif family == "recurrent":
            self.memory = nn.GRU(9, 32, batch_first=True)
            self.head = nn.Linear(32, 5)
        else:
            raise ValueError(f"Unknown family: {family}")

    def forward(self, x, hidden=None):
        if x.ndim != 3 or x.shape[-1] != 9:
            raise ValueError("Expected [resident sequences, time, nine inputs]")
        if self.family == "constant":
            return self.raw.expand(*x.shape[:-1], 5), None
        if self.family == "recurrent":
            sequence, hidden = self.memory(x, hidden)
            return self.head(sequence), hidden
        return self.network(x), None


def emission_parts(raw):
    # Double precision is important in cancellation of log-gamma terms at n=200.
    raw = raw.double()
    return F.log_softmax(raw[..., :3], dim=-1), F.softplus(raw[..., 3]) + 0.05, F.softplus(raw[..., 4]) + 0.05


def emission_log_prob(raw, n, c):
    """Exact legal mass, including overlapping endpoints for n=0 and n=1."""
    if torch.any(n < 0) or torch.any(c < 0) or torch.any(c > n):
        raise ValueError("Action outside legal integer support")
    if torch.any(n != n.floor()) or torch.any(c != c.floor()):
        raise ValueError("Support and targets must be integer-valued")
    w, a, b = emission_parts(raw)
    n, c = n.double(), c.double()
    bb = (torch.lgamma(n + 1) - torch.lgamma(c + 1) - torch.lgamma(n - c + 1)
          + torch.lgamma(c + a) + torch.lgamma(n - c + b) - torch.lgamma(n + a + b)
          + torch.lgamma(a + b) - torch.lgamma(a) - torch.lgamma(b))
    terms = torch.stack((torch.where(c == 0, w[..., 0], -torch.inf),
                         torch.where(c == n, w[..., 1], -torch.inf), w[..., 2] + bb), dim=-1)
    result = torch.where(n == 0, raw[..., 0].double() * 0, torch.logsumexp(terms, dim=-1))
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite contribution log probability")
    return result


def emission_stats(raw, n):
    w, a, b = emission_parts(raw)
    n = n.double()
    mean = n * (w[..., 1].exp() + w[..., 2].exp() * a / (a + b))
    return {"mean": mean, "p_zero": emission_log_prob(raw, n, torch.zeros_like(n)).exp(),
            "p_max": emission_log_prob(raw, n, n).exp()}


def emission_probs(raw, n):
    """Padded probabilities over 0..200; illegal outcomes have exactly zero mass."""
    if torch.any(n > 200):
        raise ValueError("Pool capacity limits legal support to 200")
    actions = torch.arange(201, device=raw.device)
    expanded_n = n[..., None].expand(*n.shape, 201)
    expanded_raw = raw[..., None, :].expand(*raw.shape[:-1], 201, 5)
    legal = actions <= expanded_n
    safe_actions = torch.minimum(actions, expanded_n)
    probabilities = emission_log_prob(expanded_raw, expanded_n, safe_actions).exp()
    return torch.where(legal, probabilities, 0.0)


def group_nll(raw, n, c):
    """Mean nonforced NLL within each complete four-player group."""
    if raw.ndim != 4 or raw.shape[1] != 4:
        raise ValueError("Expected [groups, four players, time, five outputs]")
    mask = n >= 1
    counts = mask.sum(dim=(1, 2))
    if torch.any(counts == 0):
        raise ValueError("Group has no scoreable decisions")
    return -(emission_log_prob(raw, n, c) * mask).sum(dim=(1, 2)) / counts
