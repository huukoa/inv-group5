"""Auditable, human-readable substitutions for the lecture's (s, Q) solver.

These functions have no UI dependencies. Each explanation is generated from the
actual solver inputs/results, not separately entered textbook example constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

from .models import CostParameters, ExponentialLeadTime, LeadTimeDemand, NormalLeadTime, UniformLeadTime
from .solver import Iteration, Solution


@dataclass(frozen=True, kw_only=True)
class Calculation:
    """One compact numeric result with optionally expandable LaTeX working."""

    title: str
    value: str
    steps: tuple[str, ...]
    note: str = ""


def _f(value: float) -> str:
    """Use up to 8 decimal places in working without hiding small probabilities."""
    if abs(value) < 1e-8 and value != 0:
        return f"{value:.4e}"
    return f"{value:.8f}".rstrip("0").rstrip(".") if value else "0"


def lead_time_calculations(
    costs: CostParameters,
    dist: LeadTimeDemand,
    *,
    lead_time: float,
    period_sd: float | None = None,
) -> tuple[Calculation, ...]:
    """Explain the lead-time demand transformation and EOQ initialization."""
    items: list[Calculation] = []
    if isinstance(dist, NormalLeadTime):
        if period_sd is None:
            raise ValueError("Normal conversion needs period_sd")
        items.extend(
            (
                Calculation(
                    title="Kỳ vọng cầu trong lead time — θ",
                    value=_f(dist.mean),
                    steps=(
                        r"\theta=DL",
                        rf"\theta={_f(costs.demand_rate)}\times{_f(lead_time)}={_f(dist.mean)}",
                    ),
                    note="D và L phải dùng cùng đơn vị thời gian.",
                ),
                Calculation(
                    title="Độ lệch chuẩn cầu trong lead time — σ_X",
                    value=_f(dist.sigma),
                    steps=(
                        r"\operatorname{Var}(X)=L\sigma_D^2;\quad \sigma_X=\sigma_D\sqrt{L}",
                        rf"\sigma_X={_f(period_sd)}\sqrt{{{_f(lead_time)}}}={_f(dist.sigma)}",
                        rf"\operatorname{{Var}}(X)={_f(lead_time)}\times{_f(period_sd)}^2={_f(dist.sigma**2)}",
                    ),
                    note="Ô σ_D là độ lệch chuẩn, không phải phương sai σ_D².",
                ),
            )
        )
    elif isinstance(dist, UniformLeadTime):
        items.append(
            Calculation(
                title="Kỳ vọng cầu trong lead time — θ",
                value=_f(dist.mean),
                steps=(
                    r"X\sim U(a,b);\quad\theta=\frac{a+b}{2}",
                    rf"\theta=\frac{{{_f(dist.low)}+{_f(dist.high)}}}{{2}}={_f(dist.mean)}",
                ),
                note="a, b đã là tham số của tổng cầu trong lead time.",
            )
        )
    elif isinstance(dist, ExponentialLeadTime):
        items.append(
            Calculation(
                title="Kỳ vọng cầu trong lead time — θ",
                value=_f(dist.mean),
                steps=(r"X\sim\operatorname{Exp}(\mathrm{mean}=m);\quad \theta=m", rf"\theta={_f(dist.scale)}"),
            )
        )
    eoq = math.sqrt(2 * costs.demand_rate * costs.order_cost / costs.holding_cost)
    items.append(
        Calculation(
            title="EOQ khởi tạo — Q₀",
            value=_f(eoq),
            steps=(
                r"Q_0=\sqrt{\frac{2DA}{h}}",
                rf"Q_0=\sqrt{{\frac{{2\times{_f(costs.demand_rate)}\times{_f(costs.order_cost)}}}{{{_f(costs.holding_cost)}}}}}={_f(eoq)}",
            ),
        )
    )
    return tuple(items)


def _distribution_working(dist: LeadTimeDemand, s: float, tail: float) -> tuple[str, ...]:
    if isinstance(dist, NormalLeadTime):
        k = dist.normal_k(s)
        phi = float(norm.pdf(k))
        return (
            r"k=\Phi^{-1}(1-\alpha);\quad s=\theta+\sigma_X k",
            rf"k=\Phi^{{-1}}(1-{_f(tail)})={_f(k)}",
            rf"s={_f(dist.mu)}+{_f(dist.sigma)}\times({_f(k)})={_f(s)}",
            r"n(s)=\sigma_X[\phi(k)-k(1-\Phi(k))]",
            rf"\phi(k)={_f(phi)};\quad 1-\Phi(k)={_f(tail)}",
            rf"n(s)={_f(dist.sigma)}[{_f(phi)}-({_f(k)})\times{_f(tail)}]={_f(dist.expected_shortage(s))}",
        )
    if isinstance(dist, UniformLeadTime):
        return (
            r"s=b-(b-a)\alpha;\quad n(s)=\frac{(b-s)^2}{2(b-a)}\quad(a\leq s\leq b)",
            rf"s={_f(dist.high)}-({_f(dist.high)}-{_f(dist.low)})\times{_f(tail)}={_f(s)}",
            rf"n(s)=\frac{{({_f(dist.high)}-{_f(s)})^2}}{{2({_f(dist.high)}-{_f(dist.low)})}}={_f(dist.expected_shortage(s))}",
        )
    if isinstance(dist, ExponentialLeadTime):
        return (
            r"s=-m\ln\alpha;\quad n(s)=m e^{-s/m}\quad(s\geq0)",
            rf"s=-{_f(dist.scale)}\ln({_f(tail)})={_f(s)}",
            rf"n(s)={_f(dist.scale)}e^{{-{_f(s)}/{_f(dist.scale)}}}={_f(dist.expected_shortage(s))}",
        )
    raise TypeError(f"Unsupported demand model: {type(dist).__name__}")


def iteration_calculations(
    row: Iteration,
    costs: CostParameters,
    dist: LeadTimeDemand,
) -> tuple[Calculation, ...]:
    """Show the four individual transformations in exactly one solver iteration."""
    alpha = row.tail_probability
    working = _distribution_working(dist, row.s, alpha)
    return (
        Calculation(
            title=f"Bước {row.index} · Xác suất thiếu hàng αᵢ",
            value=_f(alpha),
            steps=(
                r"\alpha_i=P(X>s_i)=\frac{Q_i h}{pD}",
                rf"\alpha_{{{row.index}}}=\frac{{{_f(row.q_in)}\times{_f(costs.holding_cost)}}}{{{_f(costs.shortage_cost)}\times{_f(costs.demand_rate)}}}={_f(alpha)}",
            ),
        ),
        Calculation(
            title=f"Bước {row.index} · Reorder point sᵢ",
            value=_f(row.s),
            steps=(r"s_i=G^{-1}(1-\alpha_i)",) + working[:3],
        ),
        Calculation(
            title=f"Bước {row.index} · Expected shortage n(sᵢ)",
            value=_f(row.expected_shortage),
            steps=(r"n(s_i)=E[(X-s_i)^+]",) + (working[:1] + working[3:] if isinstance(dist, NormalLeadTime) else working[:1] + working[2:]),
        ),
        Calculation(
            title=f"Bước {row.index} · Lượng đặt mới Qᵢ₊₁",
            value=_f(row.q_out),
            steps=(
                r"Q_{i+1}=\sqrt{\frac{2D[A+p\,n(s_i)]}{h}}",
                rf"Q_{{{row.index+1}}}=\sqrt{{\frac{{2\times{_f(costs.demand_rate)}[{_f(costs.order_cost)}+{_f(costs.shortage_cost)}\times{_f(row.expected_shortage)}]}}{{{_f(costs.holding_cost)}}}}}={_f(row.q_out)}",
                rf"\delta_Q=\frac{{|Q_{{i+1}}-Q_i|}}{{\max(1,|Q_i|)}}={_f(row.relative_q_change)}",
            ),
            note="Tiêu chuẩn dừng còn yêu cầu s ổn định giữa hai vòng lặp.",
        ),
    )


def final_calculations(
    costs: CostParameters,
    dist: LeadTimeDemand,
    result: Solution,
) -> tuple[Calculation, ...]:
    """Explain final policy and costs with direct substitutions."""
    q = result.order_quantity
    s = result.reorder_point
    n = result.shortage_per_cycle
    tail = result.stockout_probability
    distribution_lines = _distribution_working(dist, s, tail)
    # Each part is independently viewable under an expander in the UI.
    steps_s = (
        r"1-G(s^*)=\frac{Q^*h}{pD}=\alpha^*",
        rf"\alpha^*=\frac{{{_f(q)}\times{_f(costs.holding_cost)}}}{{{_f(costs.shortage_cost)}\times{_f(costs.demand_rate)}}}={_f(tail)}",
    ) + distribution_lines[:3]
    steps_n = (r"n(s^*)=E[(X-s^*)^+]",) + (
        distribution_lines[:1] + distribution_lines[3:]
        if isinstance(dist, NormalLeadTime)
        else distribution_lines[:1] + distribution_lines[2:]
    )
    return (
        Calculation(
            title="Reorder point tối ưu — s*",
            value=_f(s),
            steps=steps_s,
            note="s* được tính lại từ Q* sau vòng lặp cuối, bảo đảm hai kết quả cùng một nghiệm.",
        ),
        Calculation(
            title="Expected shortage tại nghiệm — n(s*)",
            value=_f(n),
            steps=steps_n,
        ),
        Calculation(
            title="Lượng đặt tối ưu — Q*",
            value=_f(q),
            steps=(
                r"Q^*=\sqrt{\frac{2D[A+p\,n(s^*)]}{h}}",
                rf"Q^*\approx\sqrt{{\frac{{2\times{_f(costs.demand_rate)}[{_f(costs.order_cost)}+{_f(costs.shortage_cost)}\times{_f(n)}]}}{{{_f(costs.holding_cost)}}}}}={_f(q)}",
            ),
            note="Đây là nghiệm số đạt tolerance, không phải giá trị đã làm tròn nguyên.",
        ),
        Calculation(
            title="Safety stock — SS",
            value=_f(result.safety_stock),
            steps=(
                r"SS=s^*-\theta",
                rf"SS={_f(s)}-{_f(dist.mean)}={_f(result.safety_stock)}",
            ),
        ),
        Calculation(
            title="Chi phí đặt hàng / kỳ",
            value=_f(result.costs.ordering),
            steps=(
                r"C_{order}=\frac{AD}{Q^*}",
                rf"C_{{order}}=\frac{{{_f(costs.order_cost)}\times{_f(costs.demand_rate)}}}{{{_f(q)}}}={_f(result.costs.ordering)}",
            ),
        ),
        Calculation(
            title="Chi phí tồn trữ / kỳ (approximation)",
            value=_f(result.costs.holding),
            steps=(
                r"C_{hold}=h\left(\frac{Q^*}{2}+s^*-\theta\right)",
                rf"C_{{hold}}={_f(costs.holding_cost)}\left(\frac{{{_f(q)}}}{{2}}+{_f(s)}-{_f(dist.mean)}\right)={_f(result.costs.holding)}",
            ),
        ),
        Calculation(
            title="Chi phí thiếu hàng / kỳ",
            value=_f(result.costs.shortage),
            steps=(
                r"C_{short}=\frac{pD\,n(s^*)}{Q^*}",
                rf"C_{{short}}=\frac{{{_f(costs.shortage_cost)}\times{_f(costs.demand_rate)}\times{_f(n)}}}{{{_f(q)}}}={_f(result.costs.shortage)}",
            ),
        ),
        Calculation(
            title="Tổng chi phí kiểm soát / kỳ (không tính mua hàng)",
            value=_f(result.costs.variable_total),
            steps=(
                r"Y(Q^*,s^*)=C_{order}+C_{hold}+C_{short}",
                rf"Y={_f(result.costs.ordering)}+{_f(result.costs.holding)}+{_f(result.costs.shortage)}={_f(result.costs.variable_total)}",
            ),
            note="Không cộng Dv vì giá mua là hằng số khi tối ưu (s,Q).",
        ),
        Calculation(
            title="Tổng chi phí / kỳ (gồm mua hàng)",
            value=_f(result.costs.total_with_purchasing),
            steps=(
                r"Y_{total}=Y(Q^*,s^*)+Dv",
                rf"Y_{{total}}={_f(result.costs.variable_total)}+{_f(costs.demand_rate)}\times{_f(costs.unit_cost)}={_f(result.costs.total_with_purchasing)}",
            ),
        ),
        Calculation(
            title="Stockout probability & service level (Type I)",
            value=f"{_f(tail)} / {_f(1 - tail)}",
            steps=(
                r"P(X>s^*)=1-G(s^*)=\alpha^*;\quad S_I=G(s^*)",
                rf"\alpha^*={_f(tail)};\quad S_I=1-{_f(tail)}={_f(1-tail)}",
            ),
        ),
        Calculation(
            title="Fill rate (Type II, xấp xỉ)",
            value=_f(result.fill_rate_approx),
            steps=(
                r"S_{II}\approx1-\frac{n(s^*)}{Q^*}",
                rf"S_{{II}}\approx1-\frac{{{_f(n)}}}{{{_f(q)}}}={_f(result.fill_rate_approx)}",
            ),
        ),
    )
