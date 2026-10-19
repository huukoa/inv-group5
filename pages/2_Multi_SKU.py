"""Multi-SKU analysis: independent core solvers and joint resource-constrained planning."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st

import app
from excel_import import SHEET_NAME, read_sku_sheet, validate_skus
from multi_sku import (
    MAX_JOINT_SKUS, independent_batch, planning_stock, solve_joint,
)

st.set_page_config(page_title="Multi-SKU | Inventory Lab", page_icon="📦", layout="wide")
app._theme()
st.title("📦 Multi-SKU — (s, Q) Inventory Planning")
st.caption("Independent optimization · Joint budget and warehouse-capacity constraints · Core solver unchanged")
st.markdown(
    "**Independent:** run the existing solver for each SKU separately.  "
    "**Joint:** minimize the sum of the same SKU cost functions under shared resource limits. "
    "Joint solutions are numerical candidates, **not certified global optima**."
)

with st.expander("Model definitions and units", expanded=False):
    st.latex(r"I_i=\frac{Q_i}{2}+\max\{0,\,s_i-E[X_i]\}")
    st.latex(r"\sum_i v_i I_i\leq B,\qquad \sum_i w_i I_i\leq C")
    st.markdown(
        "Here **I** is a *planning inventory proxy*: cycle stock plus positive safety stock. "
        "It is not exact on-hand inventory or peak space usage. "
        "**v** is the existing unit purchase cost. **w** is the optional Excel "
        "`Space_per_unit` column (default **1**); **B** is money and **C** is "
        "warehouse-space units. All selected SKUs must use the same accounting "
        "time unit for costs to be comparable. Joint mode enforces Q > 0 and s ≥ 0; "
        "discrete distributions require integer Q ≥ 1 and integer s ≥ 0. "
        "Purchasing cost D×v is constant and excluded from the objective, "
        "but the result table reports the same variable costs as the existing solver."
    )

root = Path(__file__).resolve().parents[1]
template = root / "assets" / "SKU_Multi_SKU_Template.xlsx"
if template.exists():
    st.download_button("Download Excel template", template.read_bytes(),
                       file_name="SKU_Multi_SKU_Template.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
uploaded = st.file_uploader("Upload parameter-based Excel (.xlsx)", type=["xlsx"],
                            help=f"The workbook needs a '{SHEET_NAME}' sheet; one row per SKU.")
if uploaded is None:
    st.info("Upload the Excel file to begin. The original single-SKU Excel Import page remains available.")
    st.stop()

file_bytes = uploaded.getvalue()
try:
    frame = read_sku_sheet(file_bytes)
    entries = validate_skus(frame, app.parse_math)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# Optional SKU-specific space factors, without changing the v1 Excel parser.
space_factors: dict[str, float] = {}
space_errors: dict[str, str] = {}
for item in entries:
    raw = frame.loc[item.excel_row - 2, "Space_per_unit"] if "Space_per_unit" in frame else None
    if pd.isna(raw) or str(raw).strip() == "":
        space_factors[item.sku] = 1.0
        continue
    try:
        w = app.parse_math(str(raw)).value
        if not 0 < w < float("inf"):
            raise ValueError("must be finite and > 0")
        space_factors[item.sku] = w
    except (ValueError, ArithmeticError, OverflowError) as exc:
        space_errors[item.sku] = f"Space_per_unit: {exc}"

st.subheader("1 · Validate and select SKUs")
preview = pd.DataFrame([{
    "Excel row": row.excel_row, "SKU": row.sku,
    "Distribution": row.distribution, "Time unit": row.time_unit,
    "Space per unit": space_factors.get(row.sku, "—"),
    "Status": "Invalid" if row.errors or row.sku in space_errors else
              ("Warning" if row.warnings else "Valid"),
    "Issues": " | ".join((*row.errors, *row.warnings,
                           *((space_errors[row.sku],) if row.sku in space_errors else ()))),
} for row in entries])
st.dataframe(preview, hide_index=True, use_container_width=True)
valid_entries = [row for row in entries if row.valid and row.sku not in space_errors]
if not valid_entries:
    st.error("No valid SKU is available. Fix the Excel input errors first.")
    st.stop()

labels = {f"{row.sku} · {row.distribution} · row {row.excel_row}": row
          for row in valid_entries}
chosen_labels = st.multiselect("SKUs to analyze", list(labels), default=list(labels))
if not chosen_labels:
    st.info("Select one or more SKUs.")
    st.stop()
chosen = [labels[label] for label in chosen_labels]
selected_factors = {row.sku: space_factors[row.sku] for row in chosen}
fingerprint = (sha256(file_bytes).hexdigest(), tuple((row.excel_row, row.sku,
              selected_factors[row.sku]) for row in chosen))

if st.button("Run independent batch", type="primary"):
    with st.spinner("Running the unchanged per-SKU solvers..."):
        outcomes = independent_batch(chosen, app.parse_math, app._context,
                                     space_factors=selected_factors)
    st.session_state["_multi_sku_fingerprint"] = fingerprint
    st.session_state["_multi_sku_outcomes"] = outcomes
    st.session_state.pop("_multi_sku_joint", None)

outcomes = (st.session_state.get("_multi_sku_outcomes")
            if st.session_state.get("_multi_sku_fingerprint") == fingerprint else None)
if outcomes is None:
    st.info("Click **Run independent batch** to calculate the selected SKUs. Then configure Joint mode.")
    st.stop()

st.subheader("2 · Independent optimization results")
items = [outcome.item for outcome in outcomes if outcome.item is not None]
rows = []
for outcome in outcomes:
    it = outcome.item
    if it is None:
        rows.append({"SKU": outcome.sku, "Distribution": outcome.distribution,
                     "Status": "Failed", "Issue": outcome.error})
        continue
    stock = planning_stock(it.independent_q, it.independent_s, it.theta)
    rows.append({
        "SKU": it.sku, "Distribution": it.distribution, "Status": "Converged",
        "Q*": it.independent_q, "s*": it.independent_s,
        "Safety stock": it.independent_s - it.theta,
        "Planning stock": stock,
        "Inventory value": it.costs.unit_cost * stock,
        "Warehouse space": it.space_per_unit * stock,
        "Variable cost": it.independent_cost,
        "Issue": "",
    })
independent_df = pd.DataFrame(rows)
st.dataframe(independent_df, hide_index=True, use_container_width=True)
st.download_button("Download independent results (CSV)",
                   independent_df.to_csv(index=False).encode("utf-8-sig"),
                   file_name="multi_sku_independent.csv", mime="text/csv")

if len(items) != len(outcomes):
    st.warning(f"{len(outcomes)-len(items)} SKU(s) failed the existing solver. "
               "They are excluded from Joint mode; inspect the Issue column.")
if not items:
    st.stop()
if len({it.time_unit for it in items}) != 1:
    st.warning("Selected SKUs use different Time_unit values. Results are shown "
               "separately; total costs cannot be meaningfully added and Joint mode is disabled.")
    st.stop()

baseline_cost = sum(it.independent_cost for it in items)
baseline_budget = sum(it.costs.unit_cost * planning_stock(it.independent_q,
                        it.independent_s, it.theta) for it in items)
baseline_capacity = sum(it.space_per_unit * planning_stock(it.independent_q,
                          it.independent_s, it.theta) for it in items)
a, b, c = st.columns(3)
a.metric("Total variable cost", f"{baseline_cost:,.2f}")
b.metric("Inventory value proxy", f"{baseline_budget:,.2f}")
c.metric("Warehouse space proxy", f"{baseline_capacity:,.2f}")

st.subheader("3 · Joint optimization")
if len(items) > MAX_JOINT_SKUS:
    st.warning(f"Joint mode accepts at most {MAX_JOINT_SKUS} successfully solved SKUs "
               "per run. Select fewer SKUs; Independent mode has no such restriction.")
    st.stop()

with st.form("joint_constraints"):
    st.markdown("**Shared constraints** — enable one or both. "
                "The default limit is 90% of the independent portfolio usage.")
    col1, col2 = st.columns(2)
    with col1:
        use_budget = st.checkbox("Limit inventory budget", value=True)
        budget_limit = st.number_input("Budget B (money)", min_value=0.000001,
                                       value=max(1.0, baseline_budget * 0.9),
                                       format="%.6f", disabled=not use_budget)
    with col2:
        use_capacity = st.checkbox("Limit warehouse capacity", value=True)
        capacity_limit = st.number_input("Capacity C (space units)", min_value=0.000001,
                                         value=max(1.0, baseline_capacity * 0.9),
                                         format="%.6f", disabled=not use_capacity)
    run_joint = st.form_submit_button("Run joint optimization", type="primary")

if run_joint:
    if not use_budget and not use_capacity:
        st.error("Enable at least one shared constraint.")
    else:
        try:
            with st.spinner("Finding a feasible constrained portfolio policy..."):
                result = solve_joint(items,
                                     budget=float(budget_limit) if use_budget else None,
                                     capacity=float(capacity_limit) if use_capacity else None)
            st.session_state["_multi_sku_joint"] = (fingerprint, result)
        except (ValueError, ArithmeticError, OverflowError) as exc:
            st.session_state.pop("_multi_sku_joint", None)
            st.error(f"Joint optimization could not return a feasible policy: {exc}")

stored = st.session_state.get("_multi_sku_joint")
if stored is None or stored[0] != fingerprint:
    st.stop()
result = stored[1]
st.success("All reported joint policies satisfy the enabled resource constraints.")
st.info(result.explanation)

joint_rows = []
for line in result.lines:
    joint_rows.append({
        "SKU": line.sku, "Distribution": line.distribution,
        "Q (joint)": line.q, "s (joint)": line.s,
        "Q (independent)": line.independent_q, "s (independent)": line.independent_s,
        "Safety stock": line.s - next(it.theta for it in items if it.sku == line.sku),
        "Planning stock": line.planning_stock,
        "Budget used": line.budget_used, "Capacity used": line.capacity_used,
        "Ordering cost": line.costs.ordering, "Holding cost": line.costs.holding,
        "Shortage cost": line.costs.shortage,
        "Variable cost": line.costs.variable_total,
        "Cost increase": line.costs.variable_total - line.independent_variable_cost,
        "Cycle service level": 1 - line.stockout_probability,
        "Fill rate (approx.)": line.fill_rate,
    })
joint_df = pd.DataFrame(joint_rows)
st.dataframe(joint_df, hide_index=True, use_container_width=True)
st.download_button("Download joint results (CSV)",
                   joint_df.to_csv(index=False).encode("utf-8-sig"),
                   file_name="multi_sku_joint.csv", mime="text/csv")

st.markdown("**Portfolio comparison**")
col1, col2, col3 = st.columns(3)
col1.metric("Joint variable cost", f"{result.variable_cost:,.2f}",
            delta=f"{result.variable_cost-result.independent_variable_cost:+,.2f} vs independent",
            delta_color="inverse")
col2.metric("Inventory budget used", f"{result.budget_used:,.2f}",
            help=(f"Limit: {result.budget_limit:,.2f}" if result.budget_limit is not None
                  else "Budget constraint disabled"))
col3.metric("Warehouse space used", f"{result.capacity_used:,.2f}",
            help=(f"Limit: {result.capacity_limit:,.2f}" if result.capacity_limit is not None
                  else "Capacity constraint disabled"))
if result.budget_limit is not None:
    st.progress(min(1.0, result.budget_used / result.budget_limit),
                text=f"Budget utilization: {result.budget_used/result.budget_limit:.1%}")
if result.capacity_limit is not None:
    st.progress(min(1.0, result.capacity_used / result.capacity_limit),
                text=f"Capacity utilization: {result.capacity_used/result.capacity_limit:.1%}")
st.caption("Joint uses a numerical mixed-integer heuristic. Feasibility is checked, "
           "but global optimality is not certified. The single-SKU solvers are unchanged.")
