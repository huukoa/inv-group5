"""Sensitivity explorer UI for Inventory Lab. Reuses original numerical solvers."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import pandas as pd
import plotly.graph_objects as go

from inventory_sq import NormalLeadTime, solve_iteratively
from inventory_sq.discrete import PoissonLeadTime, solve_discrete


@dataclass(frozen=True)
class SensitivityResult:
    parameter: str
    pct_change: float
    changed_value: float
    order_quantity: float
    reorder_point: float
    safety_stock: float
    control_cost: float
    stockout_probability: float
    fill_rate: float
    converged: bool


PARAMETERS = {
    "A": "A — Ordering cost per order",
    "h": "h — Holding cost per unit and period",
    "p": "p — Shortage penalty per unit",
    "D": "D — Mean demand rate",
    "L": "L — Constant replenishment lead time",
    "sigma": "σ_D — Demand standard deviation (accounting period)",
}

EXPLANATIONS = {
    "A": "A higher ordering cost encourages fewer, larger orders. At an interior cost-minimizing solution Q* increases and s* decreases locally.",
    "h": "More expensive holding changes both cycle stock and safety stock. s* decreases locally; the direction of Q* depends on model parameters.",
    "p": "A higher shortage penalty encourages protection against shortages. s* increases locally; Q* does not necessarily increase.",
    "D": "For Normal demand, change the average demand rate while preserving absolute demand variability per accounting period. For Poisson, λ = D × L changes automatically.",
    "L": "A longer constant lead time changes the demand accumulated while waiting: for Normal, the mean rises with L and variance rises with L; for Poisson, λ = D × L.",
    "sigma": "For Normal demand, changing demand standard deviation changes the spread, not the expected demand. The effect on cost-optimal Q* and s* is determined by re-solving the model.",
}


def supported_parameters(data: dict) -> list[str]:
    distribution = data["distribution"]
    options = ["A", "h", "p"]
    if distribution in ("Normal", "Poisson"):
        options.extend(["D", "L"])
    if distribution == "Normal":
        options.append("sigma")
    return options


def parameter_value(data: dict, parameter: str) -> float:
    if parameter == "A":
        return float(data["costs"].order_cost)
    if parameter == "h":
        return float(data["costs"].holding_cost)
    if parameter == "p":
        return float(data["costs"].shortage_cost)
    if parameter == "D":
        return float(data["costs"].demand_rate)
    if parameter == "L":
        return float(data["lead"].value)
    if parameter == "sigma":
        # Normal demand is converted to accounting-period units by _context.
        return math.sqrt(float(data["sd2_exact"]))
    raise ValueError(f"Unknown sensitivity parameter: {parameter}")


def calculate_sensitivity(data: dict, parameter: str, pct_change: float) -> SensitivityResult:
    """One-factor-at-a-time re-solving, without editing baseline or solver logic.

    General continuous distributions specify total lead-time distributions directly,
    so D/L variations are offered only when a coherent stochastic-process scaling
    is available (Normal and Poisson). Binomial D is derived, not an independent input.
    """
    if parameter not in supported_parameters(data):
        raise ValueError(f"{parameter} not supported for {data['distribution']} sensitivity")
    factor = 1 + pct_change / 100
    if factor <= 0 or not math.isfinite(factor):
        raise ValueError("All sensitivity multipliers must be strictly positive")
    baseline_costs = data["costs"]
    scenario_costs = baseline_costs
    scenario_dist = data["dist"]
    L = float(data["lead"].value)

    if parameter == "A":
        scenario_costs = replace(baseline_costs, order_cost=baseline_costs.order_cost * factor)
    elif parameter == "h":
        scenario_costs = replace(baseline_costs, holding_cost=baseline_costs.holding_cost * factor)
    elif parameter == "p":
        scenario_costs = replace(baseline_costs, shortage_cost=baseline_costs.shortage_cost * factor)
    elif parameter == "D":
        scenario_costs = replace(baseline_costs, demand_rate=baseline_costs.demand_rate * factor)
    elif parameter == "L":
        L *= factor

    if data["distribution"] == "Normal":
        annual_sd = math.sqrt(float(data["sd2_exact"]))
        if parameter == "sigma":
            annual_sd *= factor
        scenario_dist = NormalLeadTime.from_period_demand(
            mean_per_period=scenario_costs.demand_rate,
            sd_per_period=annual_sd,
            lead_time_periods=L,
        )
    elif data["distribution"] == "Poisson":
        scenario_dist = PoissonLeadTime(lam=scenario_costs.demand_rate * L)

    if data["distribution"] in ("Poisson", "Binomial"):
        # Original discrete integer optimizer; no continuous derivative approximation.
        scenario_result = solve_discrete(scenario_costs, scenario_dist)
    else:
        scenario_result = solve_iteratively(
            scenario_costs, scenario_dist, tolerance=float(data["tolerance"]),
            max_iterations=max(100, int(data["max_iterations"])), min_iterations=11,
        )

    if not scenario_result.converged:
        raise ArithmeticError("Scenario did not converge; no optimal sensitivity value is reported")
    return SensitivityResult(
        parameter=parameter, pct_change=float(pct_change),
        changed_value=parameter_value(data, parameter) * factor,
        order_quantity=float(scenario_result.order_quantity),
        reorder_point=float(scenario_result.reorder_point),
        safety_stock=float(scenario_result.safety_stock),
        control_cost=float(scenario_result.costs.variable_total),
        stockout_probability=float(scenario_result.stockout_probability),
        fill_rate=float(scenario_result.fill_rate_approx),
        converged=bool(scenario_result.converged),
    )


def _change(new: float, old: float) -> str:
    difference = new - old
    return f"{difference:+,.3f}"


def render_sensitivity(data: dict) -> None:
    """Fifth Streamlit tab, common to continuous and discrete variants."""
    import streamlit as st
    st.markdown("### Sensitivity Lab — What changes Q* and s*?")
    st.markdown(
        "Change **one input at a time**, using the current sidebar solution as the "
        "baseline. The selected scenario is re-solved with the same inventory model. "
        "Your sidebar values and original solution are never overwritten."
    )
    options = supported_parameters(data)
    labels = [PARAMETERS[name] for name in options]
    selected_label = st.selectbox("Parameter to vary", options=labels, index=0,
                                  key=f"sensitivity_parameter_{data['distribution']}")
    parameter = next(name for name in options if PARAMETERS[name] == selected_label)
    pct_change = st.slider("Change relative to baseline (%)", min_value=-50,
                           max_value=100, value=0, step=5,
                           key=f"sensitivity_percent_{data['distribution']}")
    old_value = parameter_value(data, parameter)
    st.caption(f"Baseline {parameter}: {old_value:,.6g} · Scenario {parameter}: {old_value*(1+pct_change/100):,.6g} ({pct_change:+d}%).")
    st.markdown(EXPLANATIONS[parameter])
    if data["distribution"] not in ("Normal", "Poisson"):
        st.caption(
            "D and L are not independent here: this distribution was specified "
            "for total lead-time demand. Only A, h, and p can be varied without "
            "assuming an additional demand-over-time scaling rule."
        )
    if parameter == "h" and data["mode"].startswith("Rate"):
        st.caption("The slider changes the effective h = i×v directly; the baseline i and v remain unchanged.")
    if parameter == "sigma":
        st.caption("σ_D is the accounting-period demand SD; the baseline input may use months or another base period.")

    base = data["result"]
    try:
        scenario = calculate_sensitivity(data, parameter, pct_change)
    except (ValueError, ArithmeticError, OverflowError, RuntimeError) as exc:
        st.warning(f"This scenario has no valid reported solution: {exc}")
        return

    st.markdown("#### Baseline vs. changed scenario")
    left, right = st.columns(2, gap="small")
    with left:
        st.metric("Scenario Q*", f"{scenario.order_quantity:,.3f}",
                  delta=_change(scenario.order_quantity, float(base.order_quantity)), delta_color="off")
        st.caption(f"Baseline Q* = {float(base.order_quantity):,.3f}")
    with right:
        st.metric("Scenario s*", f"{scenario.reorder_point:,.3f}",
                  delta=_change(scenario.reorder_point, float(base.reorder_point)), delta_color="off")
        st.caption(f"Baseline s* = {float(base.reorder_point):,.3f}")
    first, second = st.columns(2, gap="small")
    with first:
        st.metric("Scenario safety stock", f"{scenario.safety_stock:,.3f}",
                  delta=_change(scenario.safety_stock, float(base.safety_stock)), delta_color="off")
    with second:
        st.metric("Scenario control cost", f"{scenario.control_cost:,.3f}",
                  delta=_change(scenario.control_cost, float(base.costs.variable_total)), delta_color="off")

    st.caption("Differences use neutral colors: an increase or decrease is not automatically a business improvement.")
    st.markdown("#### Performance comparison")
    comparison = pd.DataFrame([
        {"Metric": "Order quantity Q*", "Baseline": float(base.order_quantity), "Scenario": scenario.order_quantity},
        {"Metric": "Reorder point s*", "Baseline": float(base.reorder_point), "Scenario": scenario.reorder_point},
        {"Metric": "Safety stock", "Baseline": float(base.safety_stock), "Scenario": scenario.safety_stock},
        {"Metric": "Control cost", "Baseline": float(base.costs.variable_total), "Scenario": scenario.control_cost},
        {"Metric": "Stockout probability", "Baseline": float(base.stockout_probability), "Scenario": scenario.stockout_probability},
        {"Metric": "Approximate fill rate", "Baseline": float(base.fill_rate_approx), "Scenario": scenario.fill_rate},
    ])
    comparison["Difference"] = comparison["Scenario"] - comparison["Baseline"]
    st.dataframe(comparison.style.format({"Baseline": "{:.5f}", "Scenario": "{:.5f}",
                                          "Difference": "{:+.5f}"}),
                 hide_index=True, use_container_width=True)

    st.markdown("#### Response curves — change one variable, re-solve")
    percent_grid = sorted({-50, -40, -25, -10, 0, 10, 25, 40, 50, 75, 100, int(pct_change)})
    rows = []
    for pct in percent_grid:
        try:
            sample = calculate_sensitivity(data, parameter, pct)
        except (ValueError, ArithmeticError, OverflowError, RuntimeError):
            continue  # No claim outside the valid/converged scenario domain.
        rows.append({"Change (%)": pct, "Q*": sample.order_quantity,
                     "s*": sample.reorder_point, "Control cost": sample.control_cost})
    if len(rows) < 2:
        st.warning("Too few valid scenarios to draw response curves.")
        return
    df = pd.DataFrame(rows)
    if len(rows) != len(percent_grid):
        st.caption("Some parameter values have no valid converged solution and are omitted from the curves.")
    for metric in ("Q*", "s*"):
        fig = go.Figure()
        fig.add_scatter(x=df["Change (%)"], y=df[metric], mode="lines+markers", name=metric)
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.add_scatter(x=[int(pct_change)],
                        y=[scenario.order_quantity if metric == "Q*" else scenario.reorder_point],
                        name="Selected scenario", mode="markers",
                        marker={"size": 12, "symbol": "diamond"})
        fig.update_layout(title=f"{metric} versus change in {parameter}", height=270,
                          xaxis_title=f"Change in {parameter} (%)", yaxis_title=metric,
                          margin={"l": 10, "r": 10, "t": 42, "b": 10})
        st.plotly_chart(fig, use_container_width=True)
    st.caption("These are one-factor-at-a-time comparisons of re-solved policies, not causal estimates from observed business data."
               " Results are specific to the selected cost model, distribution, and admissible parameter range.")
