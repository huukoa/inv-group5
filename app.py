"""Compact, English-language (s, Q) inventory teaching app.

The solver (inventory_sq.solver) remains unchanged. SymPy preserves exact input
arithmetic and auditable substitutions; inverse CDF and special-function loss
expressions are evaluated numerically by SciPy (never claimed exactly rational).
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from scipy.stats import norm

from inventory_sq import (
    CostParameters,
    ExponentialLeadTime,
    GammaLeadTime,
    LognormalLeadTime,
    TriangularLeadTime,
    WeibullLeadTime,
    NormalLeadTime,
    StationaryPointUnavailable,
    UniformLeadTime,
    solve_iteratively,
)
from inventory_sq.discrete import (
    BinomialLeadTime, PoissonLeadTime, DiscreteSearchLimit, solve_discrete,
)

from theory_book import render_textbook
from sensitivity_lab import render_sensitivity

ACCENT = "#50D9FF"

# Compute and display iterations 0–10 (11 updates), then continue if needed.
# Rounding affects presentation only; the solver retains full precision.
MIN_COMPUTED_ITERATIONS = 11
WORKED_ITERATIONS_SHOWN = 11


@dataclass(frozen=True)
class MathInput:
    raw: str
    exact: sp.Expr
    value: float
    typed_latex: str

    @property
    def latex(self) -> str:
        return sp.latex(self.exact)


def format_number(x: float, digits: int = 11) -> str:
    """Significant figures are display-only; never round inside the solver."""
    return f"{x:.{digits}g}"


def _normalize_math(raw: str) -> str:
    s = raw.strip().replace("×", "*").replace("÷", "/")
    s = s.replace("^", "**").replace("²", "**2")
    s = s.replace("√(", "sqrt(")
    s = re.sub(r"√\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", r"sqrt(\1)", s)
    return s


def parse_math(raw: str) -> MathInput:
    """Whitelist mathematical AST nodes; preserve exact rational expressions."""
    source = _normalize_math(raw)
    if not source or len(source) > 96:
        raise ValueError("Enter an expression of 1–96 characters.")
    try:
        root = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid mathematical expression.") from exc

    def visit(node: ast.AST, depth: int = 0) -> tuple[sp.Expr, str]:
        if depth > 16:
            raise ValueError("Expression is too complex.")
        if isinstance(node, ast.Expression):
            return visit(node.body, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if len(str(node.value)) > 36:
                raise ValueError("Numeric literal is too long.")
            value = sp.Rational(str(node.value))
            return value, sp.latex(value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            a, a_tex = visit(node.operand, depth + 1)
            if isinstance(node.op, ast.USub):
                return -a, f"-({a_tex})"
            return a, f"+({a_tex})"
        if isinstance(node, ast.BinOp):
            a, a_tex = visit(node.left, depth + 1)
            b, b_tex = visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                return a + b, f"({a_tex}+{b_tex})"
            if isinstance(node.op, ast.Sub):
                return a - b, f"({a_tex}-{b_tex})"
            if isinstance(node.op, ast.Mult):
                return a * b, f"{a_tex}\\times {b_tex}"
            if isinstance(node.op, ast.Div):
                if b == 0:
                    raise ValueError("Division by zero.")
                return a / b, rf"\frac{{{a_tex}}}{{{b_tex}}}"
            if isinstance(node.op, ast.Pow):
                if not b.is_Rational or abs(float(b)) > 12:
                    raise ValueError("The exponent must be rational and in [-12, 12].")
                if a == 0 and b < 0:
                    raise ValueError("Division by zero.")
                value = a**b
                return value, f"({a_tex})^{{{b_tex}}}"
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sqrt"
            and len(node.args) == 1
            and not node.keywords
        ):
            a, a_tex = visit(node.args[0], depth + 1)
            if a.is_negative is True:
                raise ValueError("Square root of a negative number.")
            return sp.sqrt(a), rf"\sqrt{{{a_tex}}}"
        raise ValueError("Only numbers, +, -, *, /, powers and sqrt are allowed.")

    try:
        exact, typed_latex = visit(root)
        if exact.is_real is not True:
            raise ValueError("Result must be real.")
        # Budget expression complexity and evaluation size.
        if int(sp.count_ops(exact)) > 50:
            raise ValueError("Expression is too complex.")
        value = float(exact.evalf(17))
        if not math.isfinite(value) or abs(value) > 1e14:
            raise ValueError("Result must be finite and below 1e14 in magnitude.")
        return MathInput(raw=raw, exact=exact, value=value, typed_latex=typed_latex)
    except (OverflowError, ZeroDivisionError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("Cannot evaluate this mathematical expression.") from exc


def input_expression(
    label: str,
    default: str,
    *,
    key: str,
    minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = False,
) -> MathInput | None:
    # No field-level help icons: the input syntax is documented once above the form.
    raw = st.text_input(label, value=default, key=key)
    try:
        number = parse_math(raw)
        if minimum is not None and number.value < minimum:
            raise ValueError(f"Must be ≥ {format_number(minimum)}.")
        if maximum is not None and number.value > maximum:
            raise ValueError(f"Must be ≤ {format_number(maximum)}.")
        if integer and number.exact.is_integer is not True:
            raise ValueError("Must be a whole number.")
        return number
    except ValueError as exc:
        st.session_state["_sq_errors"].append(f"{label}: {exc}")
        return None


def _field(value: str) -> MathInput:
    """Internal source-faithful default; validated like user input."""
    return parse_math(value)


def _presets(key: str) -> dict[str, str]:
    if key == "ex1":
        return dict(mu="28", sigma="8", periods="12", lead="14/52", D="336", A="15", v="6", i="0.3", h="1.8", p="7")
    if key == "ex2":
        return dict(mu="100", sigma="3", periods="1", lead="2", D="100", A="200", v="40", i="0.1", h="4", p="30")
    return dict(mu="100", sigma="3", periods="1", lead="2", D="100", A="200", v="40", i="0.1", h="4", p="30")


def _context(
    *,
    case: str,
    distribution: str,
    mode: str,
    mu: MathInput | None,
    sigma: MathInput | None,
    periods: MathInput | None,
    demand: MathInput | None,
    lead: MathInput | None,
    a: MathInput | None,
    b: MathInput | None,
    scale: MathInput | None,
    A: MathInput | None,
    v: MathInput | None,
    i: MathInput | None,
    h_direct: MathInput | None,
    p: MathInput | None,
    tolerance: float,
    max_iterations: MathInput | None,
    shape: MathInput | None = None,
    tri_mode: MathInput | None = None,
    lt_mean: MathInput | None = None,
    lt_sd: MathInput | None = None,
    trials: MathInput | None = None,
    success_probability: MathInput | None = None,
) -> dict[str, Any]:
    if distribution == "Normal":
        assert mu is not None and sigma is not None and periods is not None
        d_sym = mu.exact * periods.exact
        sd2_sym = sigma.exact**2 * periods.exact
        mu_d, sd_d = float(d_sym), math.sqrt(float(sd2_sym))
    elif distribution == "Binomial":
        assert trials is not None and success_probability is not None and lead is not None
        # n independent Bernoulli opportunities during the full lead time.
        # D is *derived* from n*pi/L so that E[X]=D*L by construction.
        d_sym = trials.exact * success_probability.exact / lead.exact
        sd2_sym = None
        mu_d, sd_d = float(d_sym), None
    else:
        assert demand is not None
        d_sym = demand.exact
        sd2_sym = None
        mu_d, sd_d = float(d_sym), None

    assert lead is not None and A is not None and v is not None and p is not None
    if mode == "Rate i × unit cost v":
        assert i is not None
        h_sym = i.exact * v.exact
    else:
        assert h_direct is not None
        h_sym = h_direct.exact
    costs = CostParameters(
        demand_rate=mu_d, order_cost=A.value,
        holding_cost=float(h_sym), shortage_cost=p.value, unit_cost=v.value,
    )
    if distribution == "Normal":
        assert sd_d is not None
        dist = NormalLeadTime.from_period_demand(
            mean_per_period=mu_d,
            sd_per_period=sd_d,
            lead_time_periods=lead.value,
        )
        theta_sym = d_sym * lead.exact
        variance_sym = sd2_sym * lead.exact
    elif distribution == "Uniform":
        assert a is not None and b is not None
        dist = UniformLeadTime(low=a.value, high=b.value)
        theta_sym = (a.exact + b.exact) / 2
        variance_sym = (b.exact - a.exact)**2 / 12
    elif distribution == "Exponential":
        assert scale is not None
        dist = ExponentialLeadTime(scale=scale.value)
        theta_sym = scale.exact
        variance_sym = scale.exact**2
    elif distribution == "Gamma":
        assert shape is not None and scale is not None
        dist = GammaLeadTime(shape=shape.value, scale=scale.value)
        theta_sym = shape.exact * scale.exact
        variance_sym = shape.exact * scale.exact**2
    elif distribution == "Lognormal":
        assert lt_mean is not None and lt_sd is not None
        dist = LognormalLeadTime(mean_demand=lt_mean.value, sd_demand=lt_sd.value)
        theta_sym = lt_mean.exact
        variance_sym = lt_sd.exact**2
    elif distribution == "Weibull":
        assert shape is not None and scale is not None
        dist = WeibullLeadTime(shape=shape.value, scale=scale.value)
        theta_sym = scale.exact * sp.gamma(1 + 1 / shape.exact)
        variance_sym = scale.exact**2 * (
            sp.gamma(1 + 2 / shape.exact) - sp.gamma(1 + 1 / shape.exact)**2
        )
    elif distribution == "Triangular":
        assert a is not None and b is not None and tri_mode is not None
        dist = TriangularLeadTime(low=a.value, mode=tri_mode.value, high=b.value)
        theta_sym = (a.exact + tri_mode.exact + b.exact) / 3
        variance_sym = (
            a.exact**2 + tri_mode.exact**2 + b.exact**2
            - a.exact * tri_mode.exact - a.exact * b.exact - tri_mode.exact * b.exact
        ) / 18
    elif distribution == "Poisson":
        theta_sym = d_sym * lead.exact
        variance_sym = theta_sym
        dist = PoissonLeadTime(lam=float(theta_sym))
    elif distribution == "Binomial":
        assert trials is not None and success_probability is not None
        theta_sym = trials.exact * success_probability.exact
        variance_sym = theta_sym * (1 - success_probability.exact)
        dist = BinomialLeadTime(
            trials=int(trials.value), probability=success_probability.value,
        )
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    assert max_iterations is not None
    if distribution in ("Poisson", "Binomial"):
        solution = solve_discrete(
            costs, dist, max_evaluations=int(max_iterations.value),
        )
    else:
        solution = solve_iteratively(
            costs, dist,
            tolerance=tolerance,
            max_iterations=int(max_iterations.value),
            min_iterations=MIN_COMPUTED_ITERATIONS,
        )
    return dict(
        case=case, distribution=distribution, mode=mode,
        mu=mu, sigma=sigma, periods=periods,
        lead=lead, a=a, b=b, scale=scale,
        shape=shape, tri_mode=tri_mode, lt_mean=lt_mean, lt_sd=lt_sd,
        trials=trials, success_probability=success_probability,
        A=A, v=v, i=i, h_direct=h_direct, p=p,
        D_exact=d_sym, h_exact=h_sym,
        sd2_exact=sd2_sym, theta_exact=theta_sym, variance_exact=variance_sym,
        costs=costs, dist=dist, result=solution,
        tolerance=tolerance, max_iterations=int(max_iterations.value),
    )


def _default_context(case: str) -> dict[str, Any]:
    """Show solved presets on page load, without user having to click Calculate."""
    key = "ex1" if case.startswith("Example 1") else "ex2"
    t = _presets(key)
    return _context(
        case=case, distribution="Normal", mode="Rate i × unit cost v",
        mu=_field(t["mu"]), sigma=_field(t["sigma"]), periods=_field(t["periods"]),
        demand=None, lead=_field(t["lead"]), a=None, b=None, scale=None,
        A=_field(t["A"]), v=_field(t["v"]), i=_field(t["i"]),
        h_direct=None, p=_field(t["p"]), tolerance=1e-10,
        max_iterations=_field("100"),
    )


def sidebar() -> tuple[dict[str, Any] | None, str | None]:
    """Compact, aligned inputs. Keep formula notation and field semantics intact."""
    st.session_state["_sq_errors"] = []
    with st.sidebar:
        st.markdown(
            '<div class="sq-help">'
            '<strong>Math input</strong> &nbsp;14/52 · √18 · 3² · (1+2)^2<br>'
            '<span>Use consistent time units for D, h and L.</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        case = st.selectbox(
            "Dataset", ["Example 1 — a.pdf", "Example 2 — a.pdf", "Custom"],
            key="sq_case_choice",
        )
        key = "ex1" if case.startswith("Example 1") else (
            "ex2" if case.startswith("Example 2") else "custom"
        )
        values = _presets(key)
        # Parameter defaults are consistent with D × L in each selected dataset.
        demand_lead_default = f"({values['D']})*({values['lead']})"
        distribution = st.selectbox(
            "Lead-time demand distribution",
            ["Normal", "Uniform", "Exponential", "Gamma", "Lognormal", "Weibull", "Triangular", "Poisson", "Binomial"],
            key=f"sq_distribution_{key}",
        )
        mode = st.selectbox(
            "Holding cost", ["Rate i × unit cost v", "Direct holding cost h"],
            key=f"sq_holding_mode_{key}",
        )
        with st.form(key=f"sq_form_{key}_{distribution}_{mode}"):
            st.markdown('<div class="sq-section-title">Demand &amp; lead time</div>',
                        unsafe_allow_html=True)
            mu = sigma = periods = demand = a = b = scale = None
            shape = tri_mode = lt_mean = lt_sd = None
            trials = success_probability = None
            if distribution == "Normal":
                col1, col2 = st.columns(2, gap="small")
                with col1:
                    mu = input_expression("μ — Mean", values["mu"], key=f"sq_mu_{key}", minimum=1e-12)
                with col2:
                    sigma = input_expression("σ — SD", values["sigma"], key=f"sq_sigma_{key}", minimum=1e-12)
                col3, col4 = st.columns(2, gap="small")
                with col3:
                    periods = input_expression(
                        "m — Periods", values["periods"],
                        key=f"sq_periods_{key}", minimum=1e-12,
                    )
                with col4:
                    lead = input_expression(
                        "L — Lead time", values["lead"], key=f"sq_lead_{key}", minimum=1e-12,
                    )
            else:
                col1, col2 = st.columns(2, gap="small")
                if distribution == "Binomial":
                    with col1:
                        trials = input_expression(
                            "n — Trials in L", "400", key=f"sq_trials_{key}",
                            minimum=1, maximum=1e8, integer=True,
                        )
                    with col2:
                        lead = input_expression(
                            "L — Lead time", values["lead"], key=f"sq_lead_other_{key}",
                            minimum=1e-12,
                        )
                    success_probability = input_expression(
                        "π — Demand probability", "1/2", key=f"sq_prob_{key}",
                        minimum=1e-12, maximum=1 - 1e-12,
                    )
                    st.caption("Binomial: D = nπ / L. One unit per successful opportunity; independent trials.")
                else:
                    with col1:
                        demand = input_expression(
                            "D — Demand", values["D"], key=f"sq_demand_{key}",
                            minimum=1e-12,
                        )
                    with col2:
                        lead = input_expression(
                            "L — Lead time", values["lead"], key=f"sq_lead_other_{key}",
                            minimum=1e-12,
                        )
                if distribution == "Poisson":
                    st.caption("Poisson: λ = D × L (expected units during lead time).")
                if distribution in ("Uniform", "Triangular"):
                    first, second = st.columns(2, gap="small")
                    upper_default = f"2*({demand_lead_default})"
                    with first:
                        a = input_expression("a — Lower", "0", key=f"sq_a_{key}", minimum=0)
                    with second:
                        b = input_expression("b — Upper", upper_default,
                                             key=f"sq_b_{key}", minimum=0)
                    if distribution == "Triangular":
                        tri_mode = input_expression("c — Mode", demand_lead_default,
                                                    key=f"sq_tri_mode_{key}", minimum=0)
                elif distribution in ("Exponential", "Gamma", "Weibull"):
                    if distribution == "Exponential":
                        scale = input_expression("β — Mean (scale)", demand_lead_default,
                                                 key=f"sq_scale_{key}", minimum=1e-12)
                    else:
                        first, second = st.columns(2, gap="small")
                        with first:
                            shape = input_expression("k — Shape", "4" if distribution == "Gamma" else "2",
                                                     key=f"sq_shape_{key}", minimum=1e-12)
                        with second:
                            scale = input_expression("β — Scale",
                                                     f"({demand_lead_default})/4" if distribution == "Gamma"
                                                     else format_number(float(parse_math(demand_lead_default).exact) *
                                                                        2 / math.sqrt(math.pi), digits=16),
                                                     key=f"sq_scale_{key}", minimum=1e-12)
                elif distribution == "Lognormal":
                    first, second = st.columns(2, gap="small")
                    with first:
                        lt_mean = input_expression("θ — LT mean", demand_lead_default,
                                                   key=f"sq_lt_mean_{key}", minimum=1e-12)
                    with second:
                        lt_sd = input_expression("σX — LT SD", f"({demand_lead_default})/4",
                                                 key=f"sq_lt_sd_{key}", minimum=1e-12)

            st.markdown('<div class="sq-section-title">Cost parameters</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2, gap="small")
            with c1:
                A = input_expression("A — Order cost", values["A"], key=f"sq_A_{key}", minimum=1e-12)
            with c2:
                v = input_expression("v — Unit cost", values["v"], key=f"sq_v_{key}", minimum=0)
            i = h_direct = None
            c3, c4 = st.columns(2, gap="small")
            with c3:
                if mode.startswith("Rate"):
                    i = input_expression("i — Hold rate", values["i"], key=f"sq_i_{key}", minimum=1e-12)
                else:
                    h_direct = input_expression(
                        "h — Hold cost", values["h"], key=f"sq_h_{key}", minimum=1e-12,
                    )
            with c4:
                p = input_expression("p — Shortage", values["p"], key=f"sq_p_{key}", minimum=1e-12)

            with st.expander("Solver settings", expanded=False):
                if distribution in ("Poisson", "Binomial"):
                    st.caption("Certified integer cost search; stops only when optimality is established.")
                    tolerance = 1e-10  # Not used by the integer branch-and-bound solver.
                    max_iterations = input_expression(
                        "Maximum candidate checks", "100000", key=f"sq_discrete_budget_{key}",
                        minimum=10, maximum=500000, integer=True,
                    )
                else:
                    tolerance = st.selectbox(
                        "Relative tolerance", [1e-4, 1e-6, 1e-8, 1e-10, 1e-12], index=3,
                    )
                    max_iterations = input_expression(
                        "Maximum iterations", "100", key=f"sq_iter_{key}",
                        minimum=MIN_COMPUTED_ITERATIONS, maximum=1000, integer=True,
                    )
            submitted = st.form_submit_button(
                "Calculate", type="primary", use_container_width=True,
            )

    existing = st.session_state.get("sq_current_result")
    # Automatic results only for the two supplied Normal examples; custom inputs
    # and altered parameter values always require an explicit Calculate click.
    preset_key = (case, distribution, mode)
    if not submitted and (existing is None or existing.get("_preset_key") != preset_key):
        if key in ("ex1", "ex2") and distribution == "Normal" and mode.startswith("Rate"):
            try:
                existing = _default_context(case)
                existing["_preset_key"] = preset_key
                st.session_state["sq_current_result"] = existing
            except (ValueError, ArithmeticError) as exc:
                st.session_state["sq_current_result"] = None
                return None, f"Could not load example: {exc}"
        else:
            st.session_state["sq_current_result"] = None
            existing = None

    if submitted:
        st.session_state["sq_current_result"] = None
        if st.session_state["_sq_errors"]:
            return None, "\n".join(st.session_state["_sq_errors"])
        try:
            run = _context(
                case=case, distribution=distribution, mode=mode,
                mu=mu, sigma=sigma, periods=periods, demand=demand,
                lead=lead, a=a, b=b, scale=scale,
                shape=shape, tri_mode=tri_mode, lt_mean=lt_mean, lt_sd=lt_sd,
                trials=trials, success_probability=success_probability,
                A=A, v=v, i=i, h_direct=h_direct, p=p,
                tolerance=tolerance, max_iterations=max_iterations,
            )
        except (ValueError, ArithmeticError, StationaryPointUnavailable, DiscreteSearchLimit) as exc:
            return None, str(exc)
        run["_preset_key"] = preset_key
        st.session_state["sq_current_result"] = run
        return run, None

    return existing, None


def _theme() -> None:
    """Neutral layout that respects Streamlit's responsive sizing.

    Avoid fixed field/metric heights, negative margins or forcibly hidden controls.
    These tend to create overlapping text across Streamlit releases.
    """
    st.markdown(
        """
        <style>
        :root { --sq-accent: #50D9FF; --sq-hairline: rgba(128, 145, 158, .28); }

        /* Main content: sensible reading width, not a stretched full-screen canvas. */
        .block-container {
            max-width: 1240px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .block-container h1 {
            font-size: clamp(1.4rem, 2vw, 1.9rem);
            line-height: 1.3;
            margin-bottom: .1rem;
        }

        /* Sidebar controls remain native to Streamlit; shrink only empty padding. */
        section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
            min-height: 2.6rem;
            height: 2.6rem;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding-top: .25rem;
            padding-bottom: 1.2rem;
        }
        section[data-testid="stSidebar"] [data-testid="stForm"] {
            border: 0;
            padding: 0;
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: .55rem;
        }
        section[data-testid="stSidebar"] div[data-testid="stTextInput"] label p {
            font-size: .82rem;
            white-space: normal;
            overflow-wrap: anywhere;
            line-height: 1.25;
        }
        section[data-testid="stSidebar"] div[data-testid="stTextInput"] input {
            width: 100%;
            min-width: 0;
            box-sizing: border-box;
        }
        section[data-testid="stSidebar"] [data-testid="column"] {
            min-width: 0;
        }
        section[data-testid="stSidebar"] div[data-testid="stTextInput"] input:focus {
            border-color: var(--sq-accent);
            box-shadow: 0 0 0 1px var(--sq-accent);
        }

        /* Quiet instructions and section dividers: no dark callouts. */
        .sq-help {
            padding: .55rem .7rem;
            border: 1px solid var(--sq-hairline);
            border-left: 3px solid var(--sq-accent);
            border-radius: 6px;
            color: inherit;
            background: transparent;
            font-size: .79rem;
            line-height: 1.55;
            overflow-wrap: anywhere;
        }
        .sq-help span { opacity: .75; }
        .sq-section-title {
            font-weight: 650;
            font-size: .87rem;
            margin: .45rem 0 .15rem;
            padding-top: .4rem;
            border-top: 1px solid var(--sq-hairline);
        }
        .sq-main-note {
            color: inherit;
            opacity: .7;
            font-size: .86rem;
            margin: 0 0 .75rem;
        }

        /* Metrics get a readable tile each, with labels allowed to wrap. */
        div[data-testid="stMetric"] {
            border: 1px solid var(--sq-hairline);
            border-radius: 9px;
            padding: .85rem 1rem;
            box-sizing: border-box;
            min-width: 0;
            height: 100%;
        }
        div[data-testid="stMetricLabel"] {
            white-space: normal;
            overflow-wrap: anywhere;
            line-height: 1.35;
        }
        div[data-testid="stMetricValue"] {
            font-variant-numeric: tabular-nums;
            overflow-wrap: anywhere;
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--sq-hairline);
            border-radius: 8px;
        }
        /* Allow very long mathematical derivations to scroll, never collide. */
        .katex-display { overflow-x: auto; overflow-y: hidden; padding: .25rem 0; }
        @media (max-width: 750px) {
            .block-container { padding-top: .9rem; }
            div[data-testid="stMetric"] { padding: .7rem .8rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _expander(title: str, result: float | str, expressions: list[str], *, note: str = "") -> None:
    displayed = result if isinstance(result, str) else format_number(result)
    with st.expander(f"{title} = {displayed}", expanded=False):
        for latex in expressions:
            st.latex(latex)
        if note:
            st.caption(note)


def _latex(sym: sp.Expr) -> str:
    return sp.latex(sym)


def _exact_setup(data: dict[str, Any]) -> None:
    costs, dist, result = data["costs"], data["dist"], data["result"]
    D, h, theta, var = data["D_exact"], data["h_exact"], data["theta_exact"], data["variance_exact"]
    A, v, p, L = data["A"], data["v"], data["p"], data["lead"]
    lines: list[str] = []
    if data["distribution"] == "Normal":
        mu, sigma, mult = data["mu"], data["sigma"], data["periods"]
        lines += [
            rf"D=\mu_{{base}}\,m={mu.latex}\times{mult.latex}={_latex(D)}",
            rf"\sigma_D^2=\sigma_{{base}}^2 m=({sigma.latex})^2\times{mult.latex}={_latex(data['sd2_exact'])}",
            rf"L={L.typed_latex}={L.latex}" if L.typed_latex != L.latex else rf"L={L.latex}",
            rf"\theta=DL={_latex(D)}\times {L.latex}={_latex(theta)}\approx {format_number(dist.mean)}",
            rf"\sigma_X^2=\sigma_D^2L={_latex(data['sd2_exact'])}\times {L.latex}={_latex(var)}",
            rf"\sigma_X=\sqrt{{{_latex(var)}}}={_latex(sp.sqrt(var))}\approx {format_number(dist.sigma)}",
        ]
    elif data["distribution"] == "Uniform":
        a, b = data["a"], data["b"]
        lines += [
            rf"\theta=\frac{{a+b}}{{2}}=\frac{{{a.latex}+{b.latex}}}{{2}}={_latex(theta)}",
            rf"\sigma_X^2=\frac{{(b-a)^2}}{{12}}={_latex(var)}",
        ]
    elif data["distribution"] == "Exponential":
        lines += [rf"\theta=\beta={_latex(theta)}", rf"\operatorname{{Var}}(X)=\beta^2={_latex(var)}"]
    elif data["distribution"] == "Gamma":
        shape, scale = data["shape"], data["scale"]
        lines += [
            rf"\theta=k\beta={shape.latex}\times{scale.latex}={_latex(theta)}",
            rf"\operatorname{{Var}}(X)=k\beta^2={shape.latex}({scale.latex})^2={_latex(var)}",
        ]
    elif data["distribution"] == "Lognormal":
        mean, sd = data["lt_mean"], data["lt_sd"]
        lines += [
            rf"\theta={mean.latex}", rf"\operatorname{{Var}}(X)=({sd.latex})^2={_latex(var)}",
            r"\sigma_{\ln X}^2=\ln\left(1+\frac{\sigma_X^2}{\theta^2}\right)",
            r"\mu_{\ln X}=\ln(\theta)-\frac{\sigma_{\ln X}^2}{2}",
        ]
    elif data["distribution"] == "Weibull":
        shape, scale = data["shape"], data["scale"]
        lines += [
            rf"\theta=\beta\Gamma(1+1/k)={scale.latex}\Gamma(1+1/{shape.latex})={_latex(theta)}",
            rf"\operatorname{{Var}}(X)={_latex(var)}",
        ]
    elif data["distribution"] == "Triangular":
        a, b, mode_val = data["a"], data["b"], data["tri_mode"]
        lines += [
            rf"\theta=\frac{{a+c+b}}{{3}}=\frac{{{a.latex}+{mode_val.latex}+{b.latex}}}{{3}}={_latex(theta)}",
            rf"\operatorname{{Var}}(X)={_latex(var)}",
        ]
    if data["mode"].startswith("Rate"):
        i = data["i"]
        lines.append(rf"h=iv={i.latex}\times{v.latex}={_latex(h)}\approx{format_number(costs.holding_cost)}")
    else:
        lines.append(rf"h={h.latex}")

    _expander("Exact inputs and lead-time conversion", "show", lines, note=(
        "Rational algebra and radicals are exact. Normal inverse-CDF and normal loss-function "
        "values require numerical evaluation; ≈ denotes approximation."
    ))
    eoq_exact = sp.sqrt(2 * D * A.exact / h)
    _expander("Initial EOQ  Q₀", result.eoq, [
        r"Q_0=\sqrt{\frac{2DA}{h}}",
        rf"Q_0=\sqrt{{\frac{{2\times{_latex(D)}\times{A.latex}}}{{{_latex(h)}}}}}={_latex(eoq_exact)}\approx {format_number(result.eoq)}",
    ])


def _additional_dist_latex(data: dict[str, Any], alpha: float, s: float, n: float) -> list[str]:
    """Numerical substitutions for additional continuous lead-time distributions."""
    name = data["distribution"]
    a_t = format_number(alpha, digits=13)
    s_t = format_number(s, digits=13)
    n_t = format_number(n, digits=13)
    if name == "Gamma":
        k, beta = data["shape"].latex, data["scale"].latex
        return [
            r"s=G^{-1}(1-\alpha)=\operatorname{GammaISF}(\alpha;k,\beta)",
            rf"s\approx\operatorname{{GammaISF}}({a_t};{k},{beta})\approx{s_t}",
            r"n(s)=k\beta\operatorname{Q}(k+1,s/\beta)-s\operatorname{Q}(k,s/\beta)",
            rf"n({s_t})\approx({k})({beta})\operatorname{{Q}}({k}+1,{s_t}/{beta})"
            rf"-{s_t}\operatorname{{Q}}({k},{s_t}/{beta})\approx{n_t}",
        ]
    if name == "Lognormal":
        theta, sd = data["lt_mean"].latex, data["lt_sd"].latex
        dist = data["dist"]
        eta, tau = format_number(dist.mu_log, 13), format_number(dist.sigma_log, 13)
        return [
            r"\tau=\sqrt{\ln(1+\sigma_X^2/\theta^2)},\quad\eta=\ln\theta-\tau^2/2",
            rf"\tau\approx{tau},\quad\eta\approx{eta}\quad(\theta={theta},\;\sigma_X={sd})",
            r"s=\exp[\eta+\tau\Phi^{-1}(1-\alpha)]",
            rf"s\approx\exp[{eta}+{tau}\Phi^{{-1}}(1-{a_t})]\approx{s_t}",
            r"n(s)=\theta\Phi[(\eta+\tau^2-\ln s)/\tau]-s\Phi[(\eta-\ln s)/\tau]",
            rf"n({s_t})\approx{theta}\Phi[({eta}+{tau}^2-\ln {s_t})/{tau}]"
            rf"-{s_t}\Phi[({eta}-\ln {s_t})/{tau}]\approx{n_t}",
        ]
    if name == "Weibull":
        k, beta = data["shape"].latex, data["scale"].latex
        return [
            r"s=\beta[-\ln(\alpha)]^{1/k}",
            rf"s\approx{beta}[-\ln({a_t})]^{{1/{k}}}\approx{s_t}",
            r"n(s)=\beta\Gamma(1+1/k)\operatorname{Q}(1+1/k,(s/\beta)^k)"
            r"-s e^{-(s/\beta)^k}",
            rf"n({s_t})\approx{beta}\Gamma(1+1/{k})\operatorname{{Q}}"
            rf"(1+1/{k},({s_t}/{beta})^{{{k}}})"
            rf"-{s_t}e^{{-({s_t}/{beta})^{{{k}}}}}\approx{n_t}",
        ]
    if name == "Triangular":
        a, c, b = data["a"].latex, data["tri_mode"].latex, data["b"].latex
        if s < data["tri_mode"].value:
            return [
                r"s=a+\sqrt{(1-\alpha)(b-a)(c-a)}",
                rf"s\approx{a}+\sqrt{{(1-{a_t})({b}-{a})({c}-{a})}}\approx{s_t}",
                r"n(s)=\theta-s+\frac{(s-a)^3}{3(b-a)(c-a)}",
                rf"n({s_t})\approx{_latex(data['theta_exact'])}-{s_t}"
                rf"+\frac{{({s_t}-{a})^3}}{{3({b}-{a})({c}-{a})}}\approx{n_t}",
            ]
        return [
            r"s=b-\sqrt{\alpha(b-a)(b-c)}",
            rf"s\approx{b}-\sqrt{{{a_t}({b}-{a})({b}-{c})}}\approx{s_t}",
            r"n(s)=\frac{(b-s)^3}{3(b-a)(b-c)}",
            rf"n({s_t})\approx\frac{{({b}-{s_t})^3}}{{3({b}-{a})({b}-{c})}}\approx{n_t}",
        ]
    raise ValueError(f"No supplemental formula for distribution {name}")


def _worked_iteration(data: dict[str, Any], row: Any) -> None:
    """Display only an auditable early iteration; never round in the solver.

    Exact algebra appears as SymPy fractions/radicals; nonlinear quantiles,
    loss-function values and iterates after Q0 are numerical approximations.
    """
    idx = row.index
    dist = data["dist"]
    D, h, A, p = data["D_exact"], data["h_exact"], data["A"], data["p"]
    theta, sigma = data["theta_exact"], sp.sqrt(data["variance_exact"])
    q_text = format_number(row.q_in, digits=13)
    alpha_text = format_number(row.tail_probability, digits=13)
    s_text = format_number(row.s, digits=13)
    shortage_text = format_number(row.expected_shortage, digits=13)
    q_next_text = format_number(row.q_out, digits=13)

    with st.expander(
        f"Iteration {idx}: Q{idx} = {row.q_in:.3f}  →  "
        f"s{idx} = {row.s:.3f}  →  Q{idx + 1} = {row.q_out:.3f}",
        expanded=False,
    ):
        # Q0 is exact EOQ. Later Q values depend on inverse-CDF evaluations.
        if idx == 0:
            q0_exact = sp.sqrt(2 * D * A.exact / h)
            st.latex(
                rf"Q_0=\sqrt{{\frac{{2DA}}{{h}}}}={_latex(q0_exact)}"
                rf"\approx {q_text}"
            )
            alpha_exact = sp.simplify(q0_exact * h / (p.exact * D))
            st.latex(
                rf"\alpha_0=\frac{{Q_0h}}{{pD}}="
                rf"\frac{{{_latex(q0_exact)}\times {_latex(h)}}}"
                rf"{{{p.latex}\times {_latex(D)}}}"
                rf"={_latex(alpha_exact)}\approx {alpha_text}"
            )
        else:
            st.latex(rf"Q_{{{idx}}}\approx {q_text}")
            st.latex(
                rf"\alpha_{{{idx}}}=\frac{{Q_{{{idx}}}h}}{{pD}}"
                rf"\approx\frac{{{q_text}\times {_latex(h)}}}"
                rf"{{{p.latex}\times {_latex(D)}}}\approx {alpha_text}"
            )

        if isinstance(dist, NormalLeadTime):
            k_text = format_number(row.normal_k, digits=13)
            pdf_text = format_number(float(norm.pdf(row.normal_k)), digits=13)
            st.latex(
                rf"k_{{{idx}}}=\Phi^{{-1}}(1-\alpha_{{{idx}}})"
                rf"\approx \Phi^{{-1}}(1-{alpha_text})\approx {k_text}"
            )
            st.latex(
                rf"s_{{{idx}}}=\theta+\sigma_X k_{{{idx}}}"
                rf"\approx {_latex(theta)}+{_latex(sigma)}\times({k_text})"
                rf"\approx {s_text}"
            )
            st.latex(
                rf"n(s_{{{idx}}})=\sigma_X[\phi(k_{{{idx}}})-"
                rf"k_{{{idx}}}(1-\Phi(k_{{{idx}}}))]"
            )
            st.latex(
                rf"n(s_{{{idx}}})\approx {_latex(sigma)}"
                rf"\left[{pdf_text}-({k_text})({alpha_text})\right]"
                rf"\approx {shortage_text}"
            )
        elif isinstance(dist, UniformLeadTime):
            lo, hi = data["a"].latex, data["b"].latex
            st.latex(
                rf"s_{{{idx}}}=b-(b-a)\alpha_{{{idx}}}"
                rf"\approx {hi}-({hi}-{lo})({alpha_text})\approx {s_text}"
            )
            st.latex(
                rf"n(s_{{{idx}}})=\frac{{(b-s_{{{idx}}})^2}}{{2(b-a)}}"
                rf"\approx \frac{{({hi}-{s_text})^2}}{{2({hi}-{lo})}}"
                rf"\approx {shortage_text}"
            )
        elif isinstance(dist, ExponentialLeadTime):
            m = data["scale"].latex
            st.latex(
                rf"s_{{{idx}}}=-\theta\ln(\alpha_{{{idx}}})"
                rf"\approx -{m}\ln({alpha_text})\approx {s_text}"
            )
            st.latex(
                rf"n(s_{{{idx}}})=\theta\exp(-s_{{{idx}}}/\theta)"
                rf"\approx {m}\exp(-{s_text}/{m})\approx {shortage_text}"
            )
        else:
            for formula in _additional_dist_latex(
                data, row.tail_probability, row.s, row.expected_shortage
            ):
                st.latex(formula)
            st.caption("Special functions are evaluated numerically at full precision.")

        st.latex(
            rf"Q_{{{idx+1}}}=\sqrt{{\frac{{2D[A+p\,n(s_{{{idx}}})]}}{{h}}}}"
            rf"\approx\sqrt{{\frac{{2\times {_latex(D)}"
            rf"\left[{A.latex}+{p.latex}\times {shortage_text}\right]}}"
            rf"{{{_latex(h)}}}}}\approx {q_next_text}"
        )
        st.latex(
            rf"\frac{{|Q_{{{idx+1}}}-Q_{{{idx}}}|}}{{\max(1,|Q_{{{idx}}}|)}}"
            rf"\approx {format_number(row.relative_q_change, digits=9)}"
        )


def _final_workings(data: dict[str, Any]) -> None:
    costs, dist, result = data["costs"], data["dist"], data["result"]
    D, h, theta = data["D_exact"], data["h_exact"], data["theta_exact"]
    p, A, v = data["p"], data["A"], data["v"]
    q, s, n = result.order_quantity, result.reorder_point, result.shortage_per_cycle
    alpha = result.stockout_probability
    _expander("Final reorder point  s*", s, [
        r"1-G(s^*)=\frac{Q^*h}{pD}=\alpha^*",
        rf"\alpha^*\approx\frac{{{format_number(q)}\times {_latex(h)}}}{{{p.latex}\times {_latex(D)}}}"
        rf"\approx{format_number(alpha)}",
        (
            rf"s^*=\theta+\sigma_X\Phi^{{-1}}(1-\alpha^*)\approx {_latex(theta)}+"
            rf"{_latex(sp.sqrt(data['variance_exact']))}\Phi^{{-1}}(1-{format_number(alpha)})"
            if isinstance(dist, NormalLeadTime)
            else r"s^*=G^{-1}(1-\alpha^*)"
        ),
        rf"s^*\approx{format_number(s)}",
    ])
    shortage_steps = [r"n(s^*)=E[(X-s^*)^+]"]
    if isinstance(dist, NormalLeadTime):
        k_final = (s - dist.mean) / dist.sigma
        phi = float(norm.pdf(k_final))
        shortage_steps.extend([
            r"n(s^*)=\sigma_X[\phi(k^*)-k^*(1-\Phi(k^*))]",
            rf"k^*=\frac{{s^*-\theta}}{{\sigma_X}}\approx{format_number(k_final)}",
            rf"n(s^*)\approx{_latex(sp.sqrt(data['variance_exact']))}\left["
            rf"{format_number(phi)}-({format_number(k_final)})\times{format_number(alpha)}\right]",
        ])
    elif isinstance(dist, UniformLeadTime):
        a, b = data["a"].latex, data["b"].latex
        shortage_steps.extend([
            r"n(s^*)=\frac{(b-s^*)^2}{2(b-a)}\quad(a\le s^*\le b)",
            rf"n(s^*)\approx\frac{{({b}-{format_number(s)})^2}}{{2({b}-{a})}}",
        ])
    elif isinstance(dist, ExponentialLeadTime):
        scale = data["scale"].latex
        shortage_steps.extend([
            r"n(s^*)=\beta\exp(-s^*/\beta)",
            rf"n(s^*)\approx{scale}\exp(-{format_number(s)}/{scale})",
        ])
    else:
        shortage_steps.extend(_additional_dist_latex(data, alpha, s, n))
    shortage_steps.append(rf"n(s^*)\approx{format_number(n)}")
    _expander("Expected shortage  n(s*)", n, shortage_steps)
    _expander("Final order quantity  Q*", q, [
        r"Q^*=\sqrt{\frac{2D[A+p\,n(s^*)]}{h}}",
        rf"Q^*\approx\sqrt{{\frac{{2\times {_latex(D)}\left[{A.latex}+{p.latex}\times {format_number(n)}\right]}}{{{_latex(h)}}}}}"
        rf"\approx{format_number(q)}",
    ])
    _expander("Safety stock  SS", result.safety_stock, [
        r"SS=s^*-\theta",
        rf"SS\approx{format_number(s)}-{_latex(theta)}\approx{format_number(result.safety_stock)}",
    ])


def _cost_workings(data: dict[str, Any]) -> None:
    costs, result = data["costs"], data["result"]
    D, h, theta = data["D_exact"], data["h_exact"], data["theta_exact"]
    A, p, v = data["A"], data["p"], data["v"]
    q, s, n = result.order_quantity, result.reorder_point, result.shortage_per_cycle
    _expander("Ordering cost / period", result.costs.ordering, [
        r"C_{order}=\frac{AD}{Q^*}",
        rf"C_{{order}}\approx\frac{{{A.latex}\times {_latex(D)}}}{{{format_number(q)}}}\approx{format_number(result.costs.ordering)}",
    ])
    _expander("Holding cost / period", result.costs.holding, [
        r"C_{hold}=h\left(\frac{Q^*}{2}+s^*-\theta\right)",
        rf"C_{{hold}}\approx {_latex(h)}\left(\frac{{{format_number(q)}}}{{2}}+{format_number(s)}-{_latex(theta)}\right)"
        rf"\approx{format_number(result.costs.holding)}",
    ])
    _expander("Shortage cost / period", result.costs.shortage, [
        r"C_{short}=\frac{pD\,n(s^*)}{Q^*}",
        rf"C_{{short}}\approx\frac{{{p.latex}\times {_latex(D)}\times {format_number(n)}}}{{{format_number(q)}}}"
        rf"\approx{format_number(result.costs.shortage)}",
    ])
    _expander("Inventory control cost / period", result.costs.variable_total, [
        r"Y=C_{order}+C_{hold}+C_{short}",
        rf"Y\approx{format_number(result.costs.ordering)}+{format_number(result.costs.holding)}+"
        rf"{format_number(result.costs.shortage)}\approx{format_number(result.costs.variable_total)}",
    ])
    _expander("Total cost including purchases", result.costs.total_with_purchasing, [
        r"Y_{all}=Y+Dv",
        rf"Y_{{all}}\approx{format_number(result.costs.variable_total)}+{_latex(D)}\times{v.latex}"
        rf"\approx{format_number(result.costs.total_with_purchasing)}",
    ])



def _cost_service_guide() -> None:
    """Optional glossary for readers unfamiliar with the cost/service metrics."""
    with st.expander("Understanding costs & service levels", expanded=False):
        st.markdown("#### Cost components")
        st.markdown(
            "**Ordering cost** — Costs incurred each time a replenishment order "
            "is placed. Larger Q generally means fewer orders per accounting period."
        )
        st.latex(r"C_{\mathrm{order}}=\frac{AD}{Q}")
        st.markdown(
            "**Holding cost** — The cost of carrying inventory, including "
            "storage, insurance, and tied-up capital. This app uses the "
            "lecture's approximate average inventory."
        )
        st.latex(r"C_{\mathrm{hold}}=h\left(\frac Q2+s-\theta\right)")
        st.markdown(
            "**Shortage cost** — Expected penalty for demand that cannot be "
            "fulfilled immediately during the replenishment cycle. "
            "The model assumes backorders."
        )
        st.latex(r"C_{\mathrm{short}}=\frac{pD}{Q}n(s)")
        st.markdown(
            "**Inventory control cost** — Ordering + holding + shortage costs, "
            "excluding purchase expenditures; this is the objective optimized by the solver."
        )
        st.latex(r"Y_{\mathrm{control}}=C_{\mathrm{order}}+C_{\mathrm{hold}}+C_{\mathrm{short}}")
        st.markdown(
            "**Total including purchases** — Adds the cost of buying D units "
            "at a fixed unit price v; Dv does not affect the optimal Q and s "
            "when the price is constant."
        )
        st.latex(r"Y_{\mathrm{all}}=Y_{\mathrm{control}}+Dv")
        st.markdown("#### Service indicators")
        st.markdown(
            "**Stockout probability** — Probability that demand accumulated "
            "over the fixed lead time exceeds the reorder point."
        )
        st.latex(r"P(X>s^*)=1-G(s^*)")
        st.markdown(
            "**Type I service level** — Probability of no lead-time stockout. "
            "This measures a shortage *event*, not the number of units served."
        )
        st.latex(r"S_I=G(s^*)")
        st.markdown(
            "**Type II service level (approximate fill rate)** — Fraction of "
            "demand units immediately fulfilled. For example, 0.98 means "
            "approximately 98% of demand units are served without waiting."
        )
        st.latex(r"S_{II}\approx1-\frac{n(s^*)}{Q^*}")
        st.caption("Stockout probability and fill rate are not interchangeable.")
        st.markdown("#### Numerical optimality checks")
        st.markdown(
            "**Stationarity residuals** — The two partial derivatives of "
            "the lecture's approximate cost function. Values near zero "
            "indicate an approximate interior stationary point, not proof of a global optimum."
        )
        st.latex(r"\frac{\partial Y}{\partial Q}\approx0,\qquad "
                 r"\frac{\partial Y}{\partial s}\approx0")
        st.caption(
            "Assumptions: constant lead time, continuous review, independent demand "
            "increments, backorders, and consistent cost/time units."
        )


def _discrete_results(data: dict[str, Any]) -> None:
    """Integer-search display, separate from the lecture's smooth iteration."""
    costs, dist, result = data["costs"], data["dist"], data["result"]
    D, h, A, penalty = data["D_exact"], data["h_exact"], data["A"], data["p"]
    theta = data["theta_exact"]
    st.caption(
        "Discrete integer optimization · best integer policy for the lecture's approximate "
        f"objective · {result.evaluated_candidates} candidates checked"
    )
    a1, a2 = st.columns(2, gap="small")
    a1.metric("Optimal integer Q*", f"{result.order_quantity:,d}")
    a2.metric("Integer reorder point s*", f"{result.reorder_point:,d}")
    b1, b2 = st.columns(2, gap="small")
    b1.metric("Safety stock", f"{result.safety_stock:,.3f}")
    b2.metric("Cost / period (excluding purchases)", f"{result.costs.variable_total:,.3f}")

    tabs = st.tabs(["Results & formulas", "Discrete search", "Costs & service", "Theory", "Sensitivity"])
    with tabs[0]:
        st.markdown("**Exact input transformations**")
        with st.expander("Demand and lead-time distribution", expanded=False):
            if isinstance(dist, PoissonLeadTime):
                st.latex(rf"D={sp.latex(D)},\quad L={data['lead'].latex}")
                st.latex(
                    rf"\lambda=DL={sp.latex(D)}\times {data['lead'].latex}"
                    rf"={sp.latex(theta)}"
                )
                st.latex(rf"X\sim\mathrm{{Poisson}}(\lambda={sp.latex(theta)})")
                st.latex(rf"E[X]=\operatorname{{Var}}(X)=\lambda={sp.latex(theta)}")
            else:
                n_in, pi_in = data["trials"], data["success_probability"]
                st.latex(rf"X\sim\mathrm{{Binomial}}(n={n_in.latex},\pi={pi_in.latex})")
                st.latex(rf"\theta=n\pi={sp.latex(theta)}")
                st.latex(
                    rf"D=\frac{{n\pi}}{{L}}="
                    rf"\frac{{{n_in.latex}\times {pi_in.latex}}}"
                    rf"{{{data['lead'].latex}}}={sp.latex(D)}"
                )
                st.latex(
                    rf"\operatorname{{Var}}(X)=n\pi(1-\pi)="
                    rf"{sp.latex(data['variance_exact'])}"
                )
            st.latex(rf"h={sp.latex(h)},\quad A={A.latex},\quad p={penalty.latex}")
        with st.expander("Expected shortage — discrete loss", expanded=False):
            st.latex(r"n(s)=\sum_{x=s+1}^{\infty}(x-s)P(X=x)")
            if isinstance(dist, PoissonLeadTime):
                st.latex(r"n(s)=\lambda P(X\ge s)-sP(X>s)")
            else:
                st.latex(
                    r"n(s)=n\pi P(Y\ge s)-sP(X>s),\quad"
                    r"Y\sim\mathrm{Binomial}(n-1,\pi)"
                )
            st.latex(
                rf"n({result.reorder_point})\approx "
                rf"{format_number(result.shortage_per_cycle)}"
            )
        with st.expander("Choosing the optimal integer order quantity", expanded=False):
            st.latex(r"Y(Q,s)=\frac{D[A+p\,n(s)]}{Q}+h\left(\frac Q2+s-\theta\right)")
            st.latex(r"Q_{\mathrm{real}}(s)=\sqrt{\frac{2D[A+p\,n(s)]}{h}}")
            st.latex(
                r"Q_{\mathrm{integer}}(s)\in"
                r"\{\max(1,\lfloor Q_{\mathrm{real}}\rfloor),"
                r"\max(1,\lceil Q_{\mathrm{real}}\rceil)\}"
            )
            q_real = math.sqrt(
                2 * costs.demand_rate *
                (costs.order_cost + costs.shortage_cost * result.shortage_per_cycle)
                / costs.holding_cost
            )
            st.latex(
                rf"Q_{{\mathrm{{real}}}}({result.reorder_point})\approx "
                rf"{format_number(q_real)}\quad\Longrightarrow\quad"
                rf"Q^*={result.order_quantity}"
            )
            st.caption("Both adjacent integer Q candidates are evaluated for every s.")
        with st.expander("Global-search certificate", expanded=False):
            st.latex(r"Y(Q,s)\ge\sqrt{2Dh[A+p\,n(s)]}+h(s-\theta)")
            st.latex(
                r"s>\theta+\frac{Y_{\mathrm{best}}-\sqrt{2DhA}}{h}"
                r"\Longrightarrow\text{cannot improve the best cost}"
            )
            st.caption(
                f"Certified range: nonnegative integer s up to {result.search_upper_bound:,}. "
                "Valid interval cost bounds eliminate ranges without examining every value. "
                "The certificate is numerical and specific to the lecture's approximation."
            )
    with tabs[1]:
        st.markdown(
            f"**First 11 integer candidates** · {result.evaluated_candidates} candidates "
            "evaluated. These are branch-and-bound search checks, "
            "not the fixed-point iterations of continuous demand."
        )
        rows = pd.DataFrame([asdict(e) for e in result.evaluations])
        display = rows.head(11).rename(columns={
            "step": "Check", "reorder_point": "s", "order_quantity": "Q",
            "shortage": "n(s)", "tail_probability": "P(X>s)",
            "cost": "Y(Q,s)", "best_so_far": "Best cost",
        })
        st.dataframe(
            display.style.format(precision=7),
            use_container_width=True, hide_index=True,
        )
        for e in result.evaluations[:11]:
            with st.expander(
                f"Check {e.step} — s={e.reorder_point}, "
                f"Q={e.order_quantity}, Y={e.cost:.3f}", expanded=False,
            ):
                st.latex(
                    rf"n({e.reorder_point})\approx{format_number(e.shortage)}"
                )
                q_real = math.sqrt(
                    2 * costs.demand_rate *
                    (costs.order_cost + costs.shortage_cost * e.shortage)
                    / costs.holding_cost
                )
                st.latex(
                    r"Q_{\mathrm{real}}(s)=\sqrt{\frac{2D[A+p\,n(s)]}{h}}"
                    rf"\approx{format_number(q_real)}"
                )
                st.latex(
                    rf"Y({e.order_quantity},{e.reorder_point})\approx"
                    rf"\frac{{{sp.latex(D)}[{A.latex}+{penalty.latex}\times"
                    rf"{format_number(e.shortage)}]}}{{{e.order_quantity}}}"
                    rf"+{sp.latex(h)}\left(\frac{{{e.order_quantity}}}{{2}}"
                    rf"+{e.reorder_point}-{sp.latex(theta)}\right)"
                    rf"\approx{format_number(e.cost)}"
                )
        fig = go.Figure()
        fig.add_scatter(
            x=rows["step"], y=rows["best_so_far"],
            mode="lines+markers", name="Best cost",
        )
        fig.update_layout(
            height=260, title="Best objective cost during integer search",
            margin=dict(l=10, r=10, t=38, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            "Export full search (CSV)",
            rows.to_csv(index=False).encode("utf-8-sig"),
            "sq_discrete_search.csv", "text/csv", use_container_width=True,
        )
        export = {
            "model": "integer (s,Q), discrete lead-time demand",
            "distribution": data["distribution"],
            "parameters": asdict(costs), "solution": asdict(result),
        }
        st.download_button(
            "Export discrete result (JSON)", json.dumps(export, indent=2).encode(),
            "sq_discrete_result.json", "application/json", use_container_width=True,
        )
    with tabs[2]:
        c = result.costs
        t1, t2 = st.columns(2, gap="small")
        t1.metric("Ordering", f"{c.ordering:,.3f}")
        t2.metric("Holding (approx.)", f"{c.holding:,.3f}")
        t3, t4 = st.columns(2, gap="small")
        t3.metric("Shortage", f"{c.shortage:,.3f}")
        t4.metric("Total incl. purchases", f"{c.total_with_purchasing:,.3f}")
        with st.expander("Understanding costs & service — discrete model", expanded=False):
            st.markdown(
                "**Ordering cost** is incurred per order; **holding cost** "
                "uses the lecture's approximate average inventory; "
                "**shortage cost** penalizes expected backorders. "
                "Stockout probability and fill rate measure different aspects of service."
            )
            st.latex(r"C_{\mathrm{order}}=AD/Q")
            st.latex(r"C_{\mathrm{hold}}=h(Q/2+s-\theta)")
            st.latex(r"C_{\mathrm{short}}=pD\,n(s)/Q")
            st.latex(r"\mathrm{Fill\ rate}\approx1-n(s)/Q")
        st.latex(
            rf"P(X>s^*)\approx{format_number(result.stockout_probability)}"
        )
        st.latex(
            rf"\mathrm{{Fill\ rate}}\approx1-"
            rf"\frac{{{format_number(result.shortage_per_cycle)}}}"
            rf"{{{result.order_quantity}}}"
            rf"\approx{format_number(result.fill_rate_approx)}"
        )
        if c.holding < 0:
            st.warning(
                "The approximate holding term is negative for this policy. "
                "This is not a physically negative warehousing expense."
            )
    with tabs[3]:
        st.markdown("### Integer-demand supplement")
        st.markdown("For discrete demand, s* and Q* are integers. The continuous derivatives in Chapter 1 are not used for the discrete optimizer.")
        st.latex(r"n(s)=\sum_{x=s+1}^{\infty}(x-s)P(X=x)")
        st.latex(r"n(s+1)-n(s)=-P(X>s)")
        if isinstance(dist, PoissonLeadTime):
            st.latex(r"X\sim\operatorname{Poisson}(\lambda),\qquad \lambda=DL")
            st.latex(r"n(s)=\lambda P(X\ge s)-sP(X>s)")
        else:
            st.latex(r"X\sim\operatorname{Binomial}(n,\pi),\qquad E[X]=n\pi")
            st.latex(r"n(s)=n\pi P(Y\ge s)-sP(X>s),\quad Y\sim\operatorname{Binomial}(n-1,\pi)")
        st.markdown("---")
        render_textbook()
    with tabs[4]:
        render_sensitivity(data)


def _results(data: dict[str, Any]) -> None:
    if data["distribution"] in ("Poisson", "Binomial"):
        _discrete_results(data)
        return
    costs, dist, result = data["costs"], data["dist"], data["result"]
    if not result.converged:
        st.warning("Not converged. Displayed values are provisional, not an established optimum.")
    else:
        st.caption(f"Converged in {len(result.records)} iterations · tolerance {data['tolerance']:.0e} · full precision retained")

    if data["distribution"] != "Normal":
        implied = costs.demand_rate * data["lead"].value
        if abs(implied - dist.mean) > 1e-9 * max(1.0, abs(implied), abs(dist.mean)):
            st.warning(
                f"Input consistency: D × L = {format_number(implied)}, but E[X] = "
                f"{format_number(dist.mean)}. This is not a single consistent demand-rate "
                "specification. Review the model assumptions before interpreting safety stock."
            )
    top_left, top_right = st.columns(2, gap="small")
    top_left.metric("Optimal Q*" if result.converged else "Provisional Q", f"{result.order_quantity:,.3f}")
    top_right.metric("Reorder point s*" if result.converged else "Provisional s", f"{result.reorder_point:,.3f}")
    bottom_left, bottom_right = st.columns(2, gap="small")
    bottom_left.metric("Safety stock", f"{result.safety_stock:,.3f}")
    bottom_right.metric("Cost / period (excluding purchases)", f"{result.costs.variable_total:,.3f}")
    tabs = st.tabs(["Results & formulas", "Iterations", "Costs & service", "Theory", "Sensitivity"])
    with tabs[0]:
        st.markdown("**Exact transformations** (open to view the original fractions and simplified values)")
        _exact_setup(data)
        st.markdown("**Final solution** (numerically evaluated where required)")
        _final_workings(data)
        st.caption("Exact input arithmetic is preserved. CDF quantiles and loss evaluations use full float64 precision; displayed values are approximations when special functions are required.")
    with tabs[1]:
        full_table = pd.DataFrame([asdict(r) for r in result.records]).rename(columns={
            "index": "Iteration", "q_in": "Q_i", "tail_probability": "α_i",
            "s": "s_i", "normal_k": "k_i", "expected_shortage": "n(s_i)",
            "q_out": "Q_(i+1)", "relative_q_change": "Relative ΔQ",
        })
        table = full_table.head(WORKED_ITERATIONS_SHOWN)
        last_shown = min(WORKED_ITERATIONS_SHOWN - 1, len(result.records) - 1)
        last_computed = result.records[-1].index
        st.markdown(
            f"**Worked iterations 0–{last_shown}** · "
            f"Computed internally through iteration {last_computed} "
            "(at least iteration 10); convergence is assessed after the minimum updates. "
            "Steps beyond iteration 10 are hidden."
        )
        st.dataframe(table.style.format(precision=9, na_rep="—"), hide_index=True, use_container_width=True)
        for record in result.records[:WORKED_ITERATIONS_SHOWN]:
            _worked_iteration(data, record)
        qfig = go.Figure()
        qfig.add_scatter(x=table["Iteration"], y=table["Q_i"], name="Q", mode="lines+markers")
        qfig.add_hline(y=result.order_quantity, line_dash="dash")
        qfig.update_layout(height=240, title="Order quantity updates (iterations 0–10)", margin=dict(l=10, r=10, t=38, b=10))
        st.plotly_chart(qfig, use_container_width=True)
        sfig = go.Figure()
        sfig.add_scatter(x=table["Iteration"], y=table["s_i"], name="s", mode="lines+markers")
        sfig.add_hline(y=result.reorder_point, line_dash="dash")
        sfig.update_layout(height=240, title="Reorder points (iterations 0–10)", margin=dict(l=10, r=10, t=38, b=10))
        st.plotly_chart(sfig, use_container_width=True)
        a, b = st.columns(2)
        with a:
            st.download_button("Export iterations (CSV)", full_table.to_csv(index=False).encode("utf-8-sig"), "sq_iterations.csv", "text/csv", use_container_width=True)
        with b:
            export = {"parameters": asdict(costs), "solution": asdict(result),
                      "exact_input_expressions": {
                          key: (getattr(data.get(key), "raw", None)) for key in
                          ("mu", "sigma", "periods", "lead", "A", "v", "i", "h_direct", "p", "demand", "a", "b", "scale", "shape", "tri_mode", "lt_mean", "lt_sd")
                      }}
            st.download_button("Export result (JSON)", json.dumps(export, indent=2).encode(), "sq_results.json", "application/json", use_container_width=True)
    with tabs[2]:
        c = result.costs
        first, second = st.columns(2, gap="small")
        first.metric("Ordering", f"{c.ordering:.3f}")
        second.metric("Holding", f"{c.holding:.3f}")
        third, fourth = st.columns(2, gap="small")
        third.metric("Shortage", f"{c.shortage:.3f}")
        fourth.metric("Total incl. purchases", f"{c.total_with_purchasing:.3f}")
        _cost_service_guide()
        _cost_workings(data)
        st.markdown("**Approximate service indicators**")
        st.latex(rf"P(X>s^*)\approx{format_number(result.stockout_probability)}")
        st.latex(rf"\text{{Fill rate}}\approx1-\frac{{n(s^*)}}{{Q^*}}\approx {format_number(result.fill_rate_approx)}")
        st.caption(
            f"Stationarity residuals: ∂Y/∂Q = {result.stationarity_q:.3e}, "
            f"∂Y/∂s = {result.stationarity_s:.3e}. These should be close to zero."
        )
        if c.holding < 0:
            st.warning("The lecture's approximate holding term is negative here; interpret this approximation carefully.")
    with tabs[3]:
        render_textbook()
    with tabs[4]:
        render_sensitivity(data)


def main() -> None:
    st.set_page_config(page_title="Inventory Lab | (s, Q)", page_icon="📦", layout="wide")
    _theme()
    st.title("📦 Inventory Lab — (s, Q) Iterative Solver")
    st.markdown(
        '<p class="sq-main-note">'
        'Stochastic demand · Constant lead time · Continuous review</p>',
        unsafe_allow_html=True,
    )
    data, error = sidebar()
    if error:
        st.error(error)
        return
    if data is None:
        st.caption("Enter the parameters in the sidebar, then click Calculate.")
        return
    _results(data)


if __name__ == "__main__":
    main()
