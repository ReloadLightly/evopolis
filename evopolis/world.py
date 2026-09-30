"""Common-pool accounting and baseline policies from Koster, Pîslar et al. (2025).

This is a new implementation of Eq. 1 and Methods, not the unreleased simulator.
See docs/protocol.md for sources, human/clone precision differences and explicit
conventions for details the publication does not resolve.
"""

from dataclasses import dataclass
import math
from typing import Sequence


PLAYERS = 4
CAPACITY = 200.0
MULTIPLIER = 1.4
ROUNDS = 40


def _vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != PLAYERS or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain four finite numbers")
    return result


def _pool(pool: float) -> float:
    value = float(pool)
    if not math.isfinite(value) or not 0 <= value <= CAPACITY:
        raise ValueError("pool must be finite and between 0 and 200")
    return value


@dataclass(frozen=True)
class RoundResult:
    """One simultaneous allocation/return round; all amounts are resource units."""

    next_pool: float
    surplus: tuple[float, ...]


def step(
    pool: float,
    offers: Sequence[float],
    contributions: Sequence[float],
    *,
    integer_contributions: bool = False,
    tolerance: float = 1e-9,
) -> RoundResult:
    """Validate a round and apply the published accounting without rounding.

    Human-interface contributions use integer units; clone rollouts are real
    valued. Set ``integer_contributions=True`` to enforce the human action grid.
    ``tolerance`` only controls validation: inputs and outputs are never clipped
    to conceal residuals. Replay should report any tolerated data violations.
    A zero pool with zero offers/contributions is an absorbing padded round.
    """
    pool = _pool(pool)
    offers = _vector(offers, "offers")
    contributions = _vector(contributions, "contributions")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if any(offer < -tolerance for offer in offers):
        raise ValueError("offers must be nonnegative")
    if math.fsum(offers) > pool + tolerance:
        raise ValueError("total offers exceed the pool")
    for offer, contribution in zip(offers, contributions):
        if contribution < -tolerance or contribution > offer + tolerance:
            raise ValueError("contribution must be between zero and its offer")
        if integer_contributions and abs(contribution - round(contribution)) > tolerance:
            raise ValueError("human contributions must use integer units")
    next_pool = min(
        CAPACITY, pool - math.fsum(offers) + MULTIPLIER * math.fsum(contributions)
    )
    surplus = tuple(offer - contribution for offer, contribution in zip(offers, contributions))
    return RoundResult(next_pool=next_pool, surplus=surplus)


def weighted_allocation(
    pool: float,
    previous_contributions: Sequence[float] | None = None,
    *,
    weight: float,
) -> tuple[float, ...]:
    """Eq. 1: weight times equal share plus (1 - weight) times relative return.

    ``None`` means the first round, when allocation is equal. If all previous
    contributions are zero, an equal fallback avoids the undefined denominator.
    That fallback at a positive pool is an EvoPolis convention, not a claim about
    unpublished source code. At a zero pool every allocation is zero.
    """
    pool = _pool(pool)
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("weight must be between 0 and 1")
    equal = (pool / PLAYERS,) * PLAYERS
    if previous_contributions is None:
        return equal
    previous = _vector(previous_contributions, "previous_contributions")
    if any(value < 0 for value in previous):
        raise ValueError("previous contributions must be nonnegative")
    total = math.fsum(previous)
    if total == 0 or weight == 1:
        return equal
    return tuple(
        weight * pool / PLAYERS + (1 - weight) * pool * contribution / total
        for contribution in previous
    )


def allocate(
    pool: float,
    previous_contributions: Sequence[float] | None = None,
    *,
    mechanism: str = "equal",
) -> tuple[float, ...]:
    """Allocate with an equal, proportional, mixed, or interpolating baseline."""
    pool = _pool(pool)
    weights = {
        "equal": 1.0,
        "proportional": 0.0,
        "mixed": 0.5,
        "interpolating": (pool / CAPACITY) ** 22,
    }
    if mechanism not in weights:
        raise ValueError(f"unknown baseline: {mechanism}")
    return weighted_allocation(pool, previous_contributions, weight=weights[mechanism])


def participant_observation(
    player: int,
    pool: float,
    offers: Sequence[float],
    previous_contributions: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Nine raw-unit BC inputs, in self-first cyclic order, with no time horizon.

    The paper divides these inputs by 200 before feeding a network. This is the
    reported BC feature subset, not the whole human interface: humans also saw
    round-end public surplus totals and could remember the visible history.
    """
    if not isinstance(player, int) or not 0 <= player < PLAYERS:
        raise ValueError("player must be an integer from 0 to 3")
    pool = _pool(pool)
    offers = _vector(offers, "offers")
    previous = _vector(
        (0.0,) * PLAYERS if previous_contributions is None else previous_contributions,
        "previous_contributions",
    )
    return offers[player:] + offers[:player] + previous[player:] + previous[:player] + (pool,)


@dataclass(frozen=True)
class WorldState:
    """Experiment 1–3 state. ``round_index`` is bookkeeping, not an observation."""

    pool: float = CAPACITY
    round_index: int = 0
    previous_contributions: tuple[float, ...] | None = None

    @property
    def done(self) -> bool:
        return self.pool == 0 or self.round_index >= ROUNDS

    def advance(
        self,
        offers: Sequence[float],
        contributions: Sequence[float],
        *,
        integer_contributions: bool = False,
    ) -> tuple["WorldState", RoundResult]:
        """Resolve a round, stopping at zero pool or after 40 completed rounds.

        Experiment 4 needs a separate continuation scheduler and is not run by
        this class. Use ``step`` alone to replay logged post-depletion padding.
        """
        if self.done:
            raise ValueError("game has ended")
        result = step(self.pool, offers, contributions, integer_contributions=integer_contributions)
        state = WorldState(result.next_pool, self.round_index + 1, tuple(contributions))
        return state, result
