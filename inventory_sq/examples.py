"""Source-faithful worked examples from handwritten a.pdf, pages 1--3."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import CostParameters, NormalLeadTime


@dataclass(frozen=True, kw_only=True)
class WorkedExample:
    title: str
    note: str
    costs: CostParameters
    lead_time_distribution: NormalLeadTime
    period_demand_mean: float
    period_demand_sd: float
    lead_time_periods: float
    accounting_period: str


def example_1() -> WorkedExample:
    """Monthly N(28,8^2) -> annual mean=336, variance=768; L=14/52 years."""
    demand_rate = 28.0 * 12
    annual_variance = 8.0**2 * 12
    return WorkedExample(
        title="Example 1 — a.pdf, pp. 1–3",
        note=(
            "Nhu cầu tháng N(28, 8²); đổi sang năm: D=336, Var(D)=768; "
            "lead time L=14/52 năm; A=15, v=6, i=0.30, h=1.8, p=7."
        ),
        costs=CostParameters(
            demand_rate=demand_rate,
            order_cost=15.0,
            holding_cost=6.0 * 0.3,
            shortage_cost=7.0,
            unit_cost=6.0,
        ),
        lead_time_distribution=NormalLeadTime.from_period_demand(
            mean_per_period=demand_rate,
            sd_per_period=math.sqrt(annual_variance),
            lead_time_periods=14 / 52,
        ),
        period_demand_mean=demand_rate,
        period_demand_sd=math.sqrt(annual_variance),
        lead_time_periods=14 / 52,
        accounting_period="năm",
    )


def example_2() -> WorkedExample:
    """Demand per cost period N(100,3^2); L=2 periods; A=200."""
    return WorkedExample(
        title="Example 2 — a.pdf, p. 3",
        note=(
            "D ~ N(100, 3²) mỗi kỳ; L=2 kỳ; A=200, v=40, i=0.10, "
            "h=4 và p=30. Kỳ cơ sở phải trùng kỳ tính holding cost."
        ),
        costs=CostParameters(
            demand_rate=100.0,
            order_cost=200.0,
            holding_cost=40 * 0.1,
            shortage_cost=30.0,
            unit_cost=40.0,
        ),
        lead_time_distribution=NormalLeadTime.from_period_demand(
            mean_per_period=100.0, sd_per_period=3.0, lead_time_periods=2.0
        ),
        period_demand_mean=100.0,
        period_demand_sd=3.0,
        lead_time_periods=2.0,
        accounting_period="kỳ (chưa nêu trong đề)",
    )
