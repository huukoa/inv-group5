"""Parameter-based Excel Import page. The existing app and solvers are unchanged."""
from __future__ import annotations

from pathlib import Path
from hashlib import sha256

import pandas as pd
import streamlit as st

import app
from excel_import import (
    SHEET_NAME, make_context_arguments, read_sku_sheet, validate_skus,
)
from inventory_sq import StationaryPointUnavailable
from inventory_sq.discrete import DiscreteSearchLimit

st.set_page_config(page_title="Excel Import | Inventory Lab", page_icon="📊", layout="wide")
app._theme()
st.title("📊 Excel Import — (s, Q)")
st.caption("Parameter-based Excel input · One row per SKU · Existing solver, unchanged")

st.markdown(
    "Download the template, replace the `DEMO-` rows with your own SKUs, "
    "and upload the completed `.xlsx` file. "
    "Demand rates, lead time and holding costs must use the **same time unit**."
)
template = Path(__file__).resolve().parents[1] / "assets" / "SKU_Import_Template.xlsx"
if template.exists():
    st.download_button(
        "Download Excel template", data=template.read_bytes(),
        file_name="SKU_Import_Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

uploaded = st.file_uploader("Upload an Excel parameter file", type=["xlsx"])
if uploaded is None:
    st.info(f"The workbook must contain a sheet named `{SHEET_NAME}`. No solver is run until you select Calculate.")
    st.stop()

file_bytes = uploaded.getvalue()
file_fingerprint = sha256(file_bytes).hexdigest()
try:
    frame = read_sku_sheet(file_bytes)
    entries = validate_skus(frame, app.parse_math)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

valid = [item for item in entries if item.valid]
invalid = [item for item in entries if not item.valid]
st.markdown(f"**{len(entries)} SKU rows** · {len(valid)} valid · {len(invalid)} invalid")

preview = []
for item in entries:
    preview.append({
        "Excel row": item.excel_row,
        "SKU": item.sku,
        "Distribution": item.distribution,
        "Time unit": item.time_unit,
        "Status": "Invalid" if item.errors else ("Warning" if item.warnings else "Valid"),
        "Issues": " | ".join((*item.errors, *item.warnings)),
    })
st.dataframe(pd.DataFrame(preview), hide_index=True, use_container_width=True)
if invalid:
    st.warning("Invalid SKU rows are excluded from the selection below. Fix the listed errors in Excel and upload again.")
if not valid:
    st.stop()

options = {f"{item.sku} · row {item.excel_row} · {item.distribution}": item for item in valid}
chosen_label = st.selectbox("Select SKU to calculate", list(options))
chosen = options[chosen_label]
selection_fingerprint = (file_fingerprint, chosen.excel_row, chosen.sku)
if chosen.warnings:
    for warning in chosen.warnings:
        st.warning(warning)

if st.button("Calculate selected SKU", type="primary"):
    try:
        arguments = make_context_arguments(chosen, app.parse_math)
        result = app._context(**arguments)
        result["_excel_sku"] = chosen.sku
        result["_excel_unit"] = chosen.time_unit
        st.session_state["_excel_import_result"] = result
        st.session_state["_excel_import_selection"] = selection_fingerprint
    except (ValueError, ArithmeticError, StationaryPointUnavailable, DiscreteSearchLimit) as exc:
        st.session_state.pop("_excel_import_result", None)
        st.session_state.pop("_excel_import_selection", None)
        st.error(f"Solver could not complete this SKU: {exc}")

result = st.session_state.get("_excel_import_result")
if result is not None:
    if st.session_state.get("_excel_import_selection") != selection_fingerprint:
        st.info("Select Calculate to compute this SKU.")
    else:
        st.success(f"Results for {chosen.sku} ({chosen.time_unit} as accounting time unit)")
        if not result["result"].converged:
            st.warning("The solver did not report convergence. Do not treat this result as a certified optimum.")
        app._results(result)
