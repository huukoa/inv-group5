"""Discrete lead-time distributions and integer (s,Q) cost optimization."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from scipy.stats import binom, poisson
from .models import CostParameters
from .solver import CostBreakdown, economic_order_quantity


class DiscreteLeadTimeDemand(Protocol):
    @property
    def mean(self) -> float: ...
    def sf(self, threshold: int) -> float: ...
    def expected_shortage(self, threshold: int) -> float: ...


@dataclass(frozen=True, kw_only=True)
class PoissonLeadTime:
    lam: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.lam) or not 0 < self.lam <= 1e8:
            raise ValueError("Poisson lead-time mean λ must be in (0, 1e8].")

    @property
    def mean(self) -> float:
        return self.lam

    def sf(self, threshold: int) -> float:
        return float(poisson.sf(threshold, self.lam))

    def expected_shortage(self, threshold: int) -> float:
        if threshold < 0:
            return self.lam - threshold
        return max(0.0, self.lam * self.sf(threshold - 1) - threshold * self.sf(threshold))


@dataclass(frozen=True, kw_only=True)
class BinomialLeadTime:
    trials: int
    probability: float

    def __post_init__(self) -> None:
        if isinstance(self.trials, bool) or not isinstance(self.trials, int) or not 1 <= self.trials <= 10**8:
            raise ValueError("Binomial trial count must be an integer in [1, 1e8].")
        if not math.isfinite(self.probability) or not 0 < self.probability < 1:
            raise ValueError("Binomial success probability must be in (0,1).")

    @property
    def mean(self) -> float:
        return self.trials * self.probability

    def sf(self, threshold: int) -> float:
        return float(binom.sf(threshold, self.trials, self.probability))

    def expected_shortage(self, threshold: int) -> float:
        if threshold < 0:
            return self.mean - threshold
        if threshold >= self.trials:
            return 0.0
        tail = self.mean * float(binom.sf(threshold - 1, self.trials - 1, self.probability))
        return max(0.0, tail - threshold * self.sf(threshold))


@dataclass(frozen=True)
class DiscreteEvaluation:
    step: int
    reorder_point: int
    order_quantity: int
    shortage: float
    tail_probability: float
    cost: float
    best_so_far: float


@dataclass(frozen=True, kw_only=True)
class DiscreteSolution:
    eoq: float
    order_quantity: int
    reorder_point: int
    evaluations: tuple[DiscreteEvaluation, ...]
    converged: bool
    costs: CostBreakdown
    lead_time_mean: float
    shortage_per_cycle: float
    stockout_probability: float
    fill_rate_approx: float
    safety_stock: float
    search_upper_bound: int
    evaluated_candidates: int
    certification: str


class DiscreteSearchLimit(RuntimeError):
    """Could not certify the integer optimum within the evaluation budget."""


def solve_discrete(costs: CostParameters, dist: DiscreteLeadTimeDemand,
                   *, max_evaluations: int = 100_000) -> DiscreteSolution:
    """Globally minimize the lecture's approximate cost on s>=0, Q>=1 integers.

    For each s only floor/ceil of optimal real Q need be considered.
    Valid lower bounds prune integer reorder-point intervals.
    """
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive")
    D, A, h, p = costs.demand_rate, costs.order_cost, costs.holding_cost, costs.shortage_cost
    theta = dist.mean
    if not math.isfinite(theta) or theta < 0:
        raise ValueError("Lead-time demand mean must be nonnegative and finite")

    @lru_cache(maxsize=300_000)
    def shortage(s: int) -> float:
        n = dist.expected_shortage(s)
        if not math.isfinite(n) or n < 0:
            raise ArithmeticError("Invalid expected-shortage evaluation")
        return n

    records: list[DiscreteEvaluation] = []
    best_y = math.inf
    best_q = 0
    best_s = 0

    def evaluate(s: int) -> None:
        nonlocal best_y, best_q, best_s
        if len(records) >= max_evaluations:
            raise DiscreteSearchLimit("Search budget exceeded before optimum was certified")
        n = shortage(s)
        q_real = math.sqrt(2 * D * (A + p * n) / h)
        candidates = {max(1, math.floor(q_real)), max(1, math.ceil(q_real))}
        q, objective = min(
            ((q, D * (A + p * n) / q + h * (q / 2 + s - theta)) for q in candidates),
            key=lambda pair: (pair[1], pair[0]),
        )
        if not math.isfinite(objective):
            raise ArithmeticError("Nonfinite objective")
        if objective < best_y:
            best_y, best_q, best_s = objective, q, s
        records.append(DiscreteEvaluation(len(records), s, q, n, dist.sf(s), objective, best_y))

    evaluate(max(0, int(round(theta))))
    evaluate(0)
    base_min = math.sqrt(2 * D * A * h)
    boundary = theta + (best_y - base_min) / h
    if not math.isfinite(boundary) or boundary > 10**15:
        raise DiscreteSearchLimit("Certified search interval is numerically too large")
    upper = max(0, math.ceil(boundary) + 2)

    def lower_bound(lo: int, hi: int) -> float:
        return math.sqrt(2 * D * h * (A + p * shortage(hi))) + h * (lo - theta)

    queue: list[tuple[float, int, int]] = []

    def push(lo: int, hi: int) -> None:
        if lo <= hi:
            lb = lower_bound(lo, hi)
            if lb < best_y:
                heapq.heappush(queue, (lb, lo, hi))

    push(0, upper)
    while queue:
        lb, lo, hi = heapq.heappop(queue)
        if lb >= best_y:
            continue
        mid = (lo + hi) // 2
        evaluate(mid)
        push(lo, mid - 1)
        push(mid + 1, hi)

    n = shortage(best_s)
    return DiscreteSolution(
        eoq=economic_order_quantity(costs), order_quantity=best_q,
        reorder_point=best_s, evaluations=tuple(records), converged=True,
        costs=CostBreakdown(
            ordering=A * D / best_q, holding=h * (best_q / 2 + best_s - theta),
            shortage=p * D * n / best_q, purchasing=costs.unit_cost * D,
        ),
        lead_time_mean=theta, shortage_per_cycle=n,
        stockout_probability=dist.sf(best_s), fill_rate_approx=1 - n / best_q,
        safety_stock=best_s - theta, search_upper_bound=upper,
        evaluated_candidates=len(records),
        certification="Global integer minimum of the lecture's approximate cost on s>=0, Q>=1 (numerical precision).",
    )
