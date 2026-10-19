"""Teaching-oriented EOQ-initialized fixed-point iteration for the (s, Q) model."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import CostParameters, LeadTimeDemand, NormalLeadTime


class StationaryPointUnavailable(ValueError):
    """The interior first-order equation has no admissible inverse-CDF value."""


@dataclass(frozen=True, kw_only=True)
class CostBreakdown:
    ordering: float
    holding: float
    shortage: float
    purchasing: float

    @property
    def variable_total(self) -> float:
        return self.ordering + self.holding + self.shortage

    @property
    def total_with_purchasing(self) -> float:
        return self.variable_total + self.purchasing


@dataclass(frozen=True, kw_only=True)
class Iteration:
    index: int
    q_in: float
    tail_probability: float
    s: float
    normal_k: float | None
    expected_shortage: float
    q_out: float
    relative_q_change: float


@dataclass(frozen=True, kw_only=True)
class Solution:
    eoq: float
    order_quantity: float
    reorder_point: float
    records: tuple[Iteration, ...]
    converged: bool
    costs: CostBreakdown
    lead_time_mean: float
    shortage_per_cycle: float
    stockout_probability: float
    fill_rate_approx: float
    safety_stock: float
    stationarity_q: float
    stationarity_s: float


def economic_order_quantity(costs: CostParameters) -> float:
    return math.sqrt(2 * costs.demand_rate * costs.order_cost / costs.holding_cost)


def cost_breakdown(
    q: float, s: float, costs: CostParameters, dist: LeadTimeDemand
) -> CostBreakdown:
    if not math.isfinite(q) or q <= 0:
        raise ValueError("Order quantity Q must be finite and > 0")
    if not math.isfinite(s):
        raise ValueError("Reorder point s must be finite")
    return CostBreakdown(
        ordering=costs.order_cost * costs.demand_rate / q,
        holding=costs.holding_cost * (q / 2 + s - dist.mean),
        shortage=costs.shortage_cost * costs.demand_rate / q * dist.expected_shortage(s),
        purchasing=costs.unit_cost * costs.demand_rate,
    )


def cost_gradient(
    q: float, s: float, costs: CostParameters, dist: LeadTimeDemand
) -> tuple[float, float]:
    """First-order derivatives of the lecture's approximation to Y(Q,s)."""
    if q <= 0:
        raise ValueError("Q must be > 0")
    dq = costs.holding_cost / 2 - costs.demand_rate * (
        costs.order_cost + costs.shortage_cost * dist.expected_shortage(s)
    ) / q**2
    ds = costs.holding_cost - costs.shortage_cost * costs.demand_rate / q * dist.sf(s)
    return dq, ds


def solve_iteratively(
    costs: CostParameters,
    dist: LeadTimeDemand,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
    min_iterations: int = 1,
) -> Solution:
    """EOQ -> inverse survival function -> n(s) -> updated Q -> repeat.

    The model seeks an interior stationary point. If h*Q/(p*D)>=1,
    the lecture's inverse-CDF equation has no finite solution. This
    implementation raises instead of silently clipping or extrapolating.

    min_iterations forces a minimum number of recorded updates before stopping
    for convergence. The default (1) preserves the original solver behavior.
    """
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and > 0")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if min_iterations < 1 or min_iterations > max_iterations:
        raise ValueError("min_iterations must be in [1, max_iterations]")
    q = economic_order_quantity(costs)
    eoq = q
    records: list[Iteration] = []
    converged = False
    last_s: float | None = None

    for index in range(max_iterations):
        tail = q * costs.holding_cost / (costs.shortage_cost * costs.demand_rate)
        if not 0 < tail < 1:
            raise StationaryPointUnavailable(
                f"At iteration {index}, hQ/(pD)={tail:.6g}; it must lie in (0, 1). "
                "The interior equation in the lecture has no finite solution "
                "with these values. Revisit costs or impose and separately "
                "optimize explicit reorder-point constraints."
            )
        s = dist.isf(tail)
        shortage = dist.expected_shortage(s)
        q_next = math.sqrt(
            2 * costs.demand_rate * (costs.order_cost + costs.shortage_cost * shortage)
            / costs.holding_cost
        )
        if not all(math.isfinite(value) for value in (s, shortage, q_next)):
            raise ArithmeticError("Encountered a non-finite intermediate result")
        k = dist.normal_k(s) if isinstance(dist, NormalLeadTime) else None
        relative_change = abs(q_next - q) / max(1.0, abs(q))
        records.append(
            Iteration(
                index=index,
                q_in=q,
                tail_probability=tail,
                s=s,
                normal_k=k,
                expected_shortage=shortage,
                q_out=q_next,
                relative_q_change=relative_change,
            )
        )
        reorder_stable = last_s is not None and (
            abs(s - last_s) / max(1.0, abs(s)) <= tolerance
        )
        if len(records) >= min_iterations and relative_change <= tolerance and reorder_stable:
            converged = True
            q = q_next
            break
        q = q_next
        last_s = s

    # Match the displayed Q with its own CDF-equation s for final metrics.
    final_tail = q * costs.holding_cost / (costs.shortage_cost * costs.demand_rate)
    if not 0 < final_tail < 1:
        raise StationaryPointUnavailable("Terminal Q makes the interior CDF equation invalid")
    final_s = dist.isf(final_tail)
    n = dist.expected_shortage(final_s)
    return Solution(
        eoq=eoq,
        order_quantity=q,
        reorder_point=final_s,
        records=tuple(records),
        converged=converged,
        costs=cost_breakdown(q, final_s, costs, dist),
        lead_time_mean=dist.mean,
        shortage_per_cycle=n,
        stockout_probability=dist.sf(final_s),
        fill_rate_approx=1 - n / q,
        safety_stock=final_s - dist.mean,
        stationarity_q=cost_gradient(q, final_s, costs, dist)[0],
        stationarity_s=cost_gradient(q, final_s, costs, dist)[1],
    )
