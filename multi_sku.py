"""Portfolio (s,Q) planning outside the frozen inventory_sq calculation core.

Independent mode delegates each SKU to app._context (the existing solver).
Joint mode minimizes the *same* sum of per-SKU variable costs subject to
optional common planning-inventory budget and space constraints. Joint solutions
are numerical candidates, NOT certified global optima.

Planning stock proxy: I_i = Q_i/2 + max(0, s_i - E[X_i]).
This is NOT an exact physical on-hand inventory or peak warehouse requirement.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from excel_import import ImportedSKU, make_context_arguments
from inventory_sq.discrete import DiscreteSearchLimit
from inventory_sq.solver import CostBreakdown, StationaryPointUnavailable, cost_breakdown


MAX_JOINT_SKUS = 20


@dataclass(frozen=True)
class PortfolioItem:
    sku: str
    distribution: str
    time_unit: str
    costs: Any
    dist: Any
    independent_q: float
    independent_s: float
    independent_cost: float
    is_discrete: bool
    space_per_unit: float = 1.0

    @property
    def theta(self) -> float:
        return float(self.dist.mean)


@dataclass(frozen=True)
class SKUOutcome:
    sku: str
    distribution: str
    time_unit: str
    item: PortfolioItem | None
    error: str | None
    converged: bool


@dataclass(frozen=True)
class JointLine:
    sku: str
    distribution: str
    q: float
    s: float
    planning_stock: float
    budget_used: float
    capacity_used: float
    costs: CostBreakdown
    stockout_probability: float
    fill_rate: float
    independent_q: float
    independent_s: float
    independent_variable_cost: float


@dataclass(frozen=True)
class JointResult:
    lines: tuple[JointLine, ...]
    budget_limit: float | None
    capacity_limit: float | None
    budget_used: float
    capacity_used: float
    variable_cost: float
    independent_variable_cost: float
    status: str
    explanation: str


def independent_batch(
    entries: Sequence[ImportedSKU], parse_math: Callable[[str], Any],
    solve_context: Callable[..., dict[str, Any]],
    *, space_factors: dict[str, float] | None = None,
) -> list[SKUOutcome]:
    """Run the *existing* solver once per valid row; isolate SKU failures."""
    factors = space_factors or {}
    outcomes: list[SKUOutcome] = []
    for row in entries:
        if not row.valid:
            outcomes.append(SKUOutcome(row.sku, row.distribution, row.time_unit,
                                       None, "; ".join(row.errors), False))
            continue
        try:
            context = solve_context(**make_context_arguments(row, parse_math))
            result = context["result"]
            if not result.converged:
                outcomes.append(SKUOutcome(row.sku, row.distribution, row.time_unit,
                                           None, "Solver did not converge or certify a result.", False))
                continue
            item = PortfolioItem(
                sku=row.sku, distribution=row.distribution, time_unit=row.time_unit,
                costs=context["costs"], dist=context["dist"],
                independent_q=float(result.order_quantity),
                independent_s=float(result.reorder_point),
                independent_cost=float(result.costs.variable_total),
                is_discrete=row.distribution in ("Poisson", "Binomial"),
                space_per_unit=float(factors.get(row.sku, 1.0)),
            )
            if not math.isfinite(item.space_per_unit) or item.space_per_unit <= 0:
                raise ValueError("Space_per_unit must be finite and > 0.")
            outcomes.append(SKUOutcome(row.sku, row.distribution, row.time_unit,
                                       item, None, True))
        except (ValueError, ArithmeticError, OverflowError, DiscreteSearchLimit,
                StationaryPointUnavailable) as exc:
            outcomes.append(SKUOutcome(row.sku, row.distribution, row.time_unit,
                                       None, str(exc), False))
    return outcomes


def planning_stock(q: float, s: float, theta: float) -> float:
    return q / 2 + max(0.0, s - theta)


def _resources(items: Sequence[PortfolioItem], x: np.ndarray) -> tuple[float, float]:
    budget = capacity = 0.0
    for j, item in enumerate(items):
        stock = planning_stock(float(x[2*j]), float(x[2*j+1]), item.theta)
        budget += item.costs.unit_cost * stock
        capacity += item.space_per_unit * stock
    return budget, capacity


def _discrete_relaxed_shortage(dist: Any, s: float) -> float:
    """Linear interpolation of the *unchanged* integer expected-shortage values.

    At integer s, this is exactly dist.expected_shortage(s). It is used only
    in the continuous relaxation of the separate portfolio optimizer.
    """
    lo = max(0, int(math.floor(s)))
    frac = s - lo
    n0 = dist.expected_shortage(lo)
    if frac <= 1e-12:
        return n0
    return (1 - frac) * n0 + frac * dist.expected_shortage(lo + 1)


def _variable_cost(item: PortfolioItem, q: float, s: float, *, relaxed: bool) -> float:
    c = item.costs
    shortage = (_discrete_relaxed_shortage(item.dist, s) if item.is_discrete and relaxed
                else item.dist.expected_shortage(int(round(s)) if item.is_discrete else s))
    return c.demand_rate * (c.order_cost + c.shortage_cost * shortage) / q + (
        c.holding_cost * (q / 2 + s - item.theta))


def _costs_exact(item: PortfolioItem, q: float, s: float) -> CostBreakdown:
    if item.is_discrete:
        return cost_breakdown(int(round(q)), int(round(s)), item.costs, item.dist)
    return cost_breakdown(q, s, item.costs, item.dist)


def _feasible(items: Sequence[PortfolioItem], x: np.ndarray,
              budget: float | None, capacity: float | None) -> bool:
    if any(not math.isfinite(float(v)) for v in x):
        return False
    for j, item in enumerate(items):
        if x[2*j] < (1 if item.is_discrete else 1e-6) - 1e-9 or x[2*j+1] < -1e-9:
            return False
        if item.is_discrete and (abs(x[2*j] - round(x[2*j])) > 1e-7 or
                                 abs(x[2*j+1] - round(x[2*j+1])) > 1e-7):
            return False
    b, c = _resources(items, x)
    return ((budget is None or b <= budget + 1e-7 * max(1, budget)) and
            (capacity is None or c <= capacity + 1e-7 * max(1, capacity)))


def solve_joint(
    items: Sequence[PortfolioItem], *, budget: float | None = None,
    capacity: float | None = None, maxiter: int = 450,
) -> JointResult:
    """Find a feasible, locally improved mixed-integer portfolio candidate.

    SLSQP solves a continuous relaxation. Discrete Q and s are rounded DOWN
    (to preserve resource feasibility) and locally improved with exact integer
    cost evaluations. Continuous coordinates are reoptimized when applicable.
    Never claim a globally certified optimum.
    """
    items = tuple(items)
    if not items:
        raise ValueError("Select at least one successfully solved SKU.")
    if len(items) > MAX_JOINT_SKUS:
        raise ValueError(f"Joint mode supports at most {MAX_JOINT_SKUS} SKUs per run.")
    if len({item.time_unit for item in items}) != 1:
        raise ValueError("All SKUs in Joint mode must use the same Time_unit.")
    if budget is None and capacity is None:
        raise ValueError("Enable at least one shared constraint for Joint mode.")
    for name, value in (("budget", budget), ("capacity", capacity)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"Shared {name} must be finite and > 0.")

    n = len(items)
    independent = np.array([v for it in items for v in
                            (it.independent_q, it.independent_s)], dtype=float)
    q_min = np.array([1.0 if it.is_discrete else 1e-6 for it in items])
    min_budget = sum(it.costs.unit_cost * q_min[j] / 2 for j, it in enumerate(items))
    min_capacity = sum(it.space_per_unit * q_min[j] / 2 for j, it in enumerate(items))
    if budget is not None and budget < min_budget - 1e-9:
        raise ValueError("Budget is below the minimum possible planning stock under Q bounds.")
    if capacity is not None and capacity < min_capacity - 1e-9:
        raise ValueError("Capacity is below the minimum possible planning stock under Q bounds.")

    baseline_total = sum(it.independent_cost for it in items)
    if _feasible(items, independent, budget, capacity):
        x_best = independent
        status = "independent_feasible"
        explanation = ("Independent solutions satisfy all enabled constraints. "
                       "No adjustment is needed; both modes return the same policy.")
    else:
        # Scaled variables keep the optimizer numerically stable across SKUs.
        scales = np.array([v for it in items for v in
                           (max(1, it.independent_q),
                            max(1, abs(it.independent_s), it.theta))], dtype=float)
        bounds = [(q_min[j] / scales[2*j], None) if k % 2 == 0
                  else (0.0, None) for k, j in ((k, k // 2) for k in range(2*n))]
        obj_scale = max(1.0, abs(baseline_total))

        def raw_x(z: np.ndarray) -> np.ndarray:
            return z * scales

        def objective(z: np.ndarray) -> float:
            x = raw_x(z)
            try:
                val = sum(_variable_cost(it, float(x[2*j]), float(x[2*j+1]), relaxed=True)
                          for j, it in enumerate(items))
                return val / obj_scale if math.isfinite(val) else 1e100
            except (ArithmeticError, OverflowError, ValueError):
                return 1e100

        constraints = []
        if budget is not None:
            constraints.append({"type": "ineq", "fun": lambda z:
                (budget - _resources(items, raw_x(z))[0]) / max(1, budget)})
        if capacity is not None:
            constraints.append({"type": "ineq", "fun": lambda z:
                (capacity - _resources(items, raw_x(z))[1]) / max(1, capacity)})

        # Feasible seed: remove positive safety stock, reduce cycle stock.
        seed = independent.copy()
        for j, it in enumerate(items):
            seed[2*j+1] = max(0, min(it.independent_s, it.theta))
        seed_budget, seed_capacity = _resources(items, seed)
        factor = 1.0
        if budget is not None and seed_budget > 0:
            factor = min(factor, 0.8 * budget / seed_budget)
        if capacity is not None and seed_capacity > 0:
            factor = min(factor, 0.8 * capacity / seed_capacity)
        for j in range(n):
            seed[2*j] = max(q_min[j], seed[2*j] * factor)
        # If integer Q minima exhaust a resource, shrink continuous Q further.
        for _ in range(12):
            if _resources(items, seed)[0] <= (budget if budget is not None else math.inf) and \
               _resources(items, seed)[1] <= (capacity if capacity is not None else math.inf):
                break
            for j in range(n):
                seed[2*j] = max(q_min[j], seed[2*j] * 0.5)

        candidates: list[np.ndarray] = []
        if _feasible(items, seed, budget, capacity) and not any(it.is_discrete for it in items):
            candidates.append(seed.copy())

        for start in (seed, np.maximum(independent, np.array(
                [v for j in range(n) for v in (q_min[j], 0.0)]))):
            result = minimize(objective, start / scales, method="SLSQP", bounds=bounds,
                              constraints=constraints, options={"maxiter": maxiter,
                              "ftol": 1e-10, "disp": False})
            relaxed = raw_x(result.x)
            if not np.all(np.isfinite(relaxed)):
                continue
            # Discrete coordinates: floor is monotone in resource usage.
            for j, it in enumerate(items):
                if it.is_discrete:
                    relaxed[2*j] = max(1, math.floor(relaxed[2*j] + 1e-9))
                    relaxed[2*j+1] = max(0, math.floor(relaxed[2*j+1] + 1e-9))
            if _feasible(items, relaxed, budget, capacity):
                candidates.append(relaxed)

        # Also consider a guaranteed small-inventory integer starting policy.
        fallback = seed.copy()
        for j, it in enumerate(items):
            if it.is_discrete:
                fallback[2*j] = max(1, math.floor(fallback[2*j]))
                fallback[2*j+1] = max(0, math.floor(fallback[2*j+1]))
        if _feasible(items, fallback, budget, capacity):
            candidates.append(fallback)
        if not candidates:
            raise ValueError("No feasible numerical policy found. Check tight constraints and inputs.")

        # Refine continuous variables with discrete variables fixed.
        cont_indices = [k for j, it in enumerate(items) if not it.is_discrete
                        for k in (2*j, 2*j+1)]

        def refine_continuous(x: np.ndarray) -> np.ndarray:
            if not cont_indices:
                return x
            base = x.copy()
            sub_scales = scales[cont_indices]
            sub_bounds = [(q_min[k//2] / scales[k], None) if k % 2 == 0
                          else (0.0, None) for k in cont_indices]

            def combine(z: np.ndarray) -> np.ndarray:
                xx = base.copy()
                xx[cont_indices] = z * sub_scales
                return xx

            sub_constraints = []
            if budget is not None:
                sub_constraints.append({"type": "ineq", "fun": lambda z:
                    (budget - _resources(items, combine(z))[0]) / max(1, budget)})
            if capacity is not None:
                sub_constraints.append({"type": "ineq", "fun": lambda z:
                    (capacity - _resources(items, combine(z))[1]) / max(1, capacity)})
            res = minimize(lambda z: objective(combine(z)/scales),
                           base[cont_indices] / sub_scales, method="SLSQP",
                           bounds=sub_bounds, constraints=sub_constraints,
                           options={"maxiter": maxiter, "ftol": 1e-10})
            xx = combine(res.x)
            if (_feasible(items, xx, budget, capacity) and
                objective(xx/scales) < objective(base/scales) - 1e-11):
                return xx
            return base

        candidates = [refine_continuous(x) for x in candidates]
        x_best = min(candidates, key=lambda x: objective(x / scales))

        # Integer coordinate neighborhood: exact cost and exact resource checks.
        discrete_indices = [j for j, it in enumerate(items) if it.is_discrete]
        for _ in range(12):
            improved = False
            best_cost = sum(_variable_cost(it, float(x_best[2*j]), float(x_best[2*j+1]),
                                           relaxed=False) for j, it in enumerate(items))
            for j in discrete_indices:
                for k in (2*j, 2*j+1):
                    val = int(round(x_best[k]))
                    for delta in (-2, -1, 1, 2):
                        candidate = x_best.copy()
                        candidate[k] = val + delta
                        if not _feasible(items, candidate, budget, capacity):
                            continue
                        cost = sum(_variable_cost(it, float(candidate[2*t]),
                                                  float(candidate[2*t+1]), relaxed=False)
                                   for t, it in enumerate(items))
                        if cost < best_cost - 1e-8:
                            x_best = candidate
                            best_cost = cost
                            improved = True
            if not improved:
                break
        x_best = refine_continuous(x_best)
        status = "numerical_feasible"
        explanation = ("Feasible numerical candidate from constrained continuous relaxation, "
                       "integer rounding and local refinement. Global optimality is NOT certified.")

    if not _feasible(items, x_best, budget, capacity):
        raise ArithmeticError("Final policy failed constraint or integrality checks.")
    b_used, c_used = _resources(items, x_best)
    lines: list[JointLine] = []
    for j, item in enumerate(items):
        q = int(round(x_best[2*j])) if item.is_discrete else float(x_best[2*j])
        s = int(round(x_best[2*j+1])) if item.is_discrete else float(x_best[2*j+1])
        costs = _costs_exact(item, q, s)
        stock = planning_stock(q, s, item.theta)
        shortage = item.dist.expected_shortage(s)
        lines.append(JointLine(
            sku=item.sku, distribution=item.distribution, q=q, s=s,
            planning_stock=stock, budget_used=item.costs.unit_cost * stock,
            capacity_used=item.space_per_unit * stock, costs=costs,
            stockout_probability=item.dist.sf(s), fill_rate=1 - shortage / q,
            independent_q=item.independent_q, independent_s=item.independent_s,
            independent_variable_cost=item.independent_cost,
        ))
    return JointResult(
        lines=tuple(lines), budget_limit=budget, capacity_limit=capacity,
        budget_used=b_used, capacity_used=c_used,
        variable_cost=sum(line.costs.variable_total for line in lines),
        independent_variable_cost=baseline_total,
        status=status, explanation=explanation,
    )
