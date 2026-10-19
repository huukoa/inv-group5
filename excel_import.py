"""Excel parameter-import adapter. No inventory optimization logic lives here.

The caller supplies the application's existing parse_math function so spreadsheet
values use exactly the same mathematical expression rules as manual inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from typing import Any, Callable

import pandas as pd

SHEET_NAME = "SKU_Inputs"
DISTRIBUTIONS = (
    "Normal", "Uniform", "Exponential", "Gamma", "Lognormal",
    "Weibull", "Triangular", "Poisson", "Binomial",
)
HOLDING_MODES = {"rate": "Rate i × unit cost v", "direct": "Direct holding cost h"}
TIME_UNITS = ("year", "month", "week", "day")
FIELDS = (
    "SKU", "Distribution", "Holding_mode", "Time_unit",
    "mu", "sigma", "m", "D", "L", "a", "b", "c", "scale", "shape",
    "lt_mean", "lt_sd", "n", "pi", "A", "v", "i", "h", "p",
    "tolerance", "max_iterations",
)
REQUIRED_COLUMNS = ("SKU", "Distribution", "Holding_mode", "Time_unit", "L", "A", "v", "p")
DEFAULT_TOLERANCE = 1e-10
DEFAULT_MAX_ITERATIONS = {"continuous": 100, "discrete": 100000}
ALLOWED_TOLERANCES = (1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_SKUS = 5000


@dataclass(frozen=True)
class ImportedSKU:
    excel_row: int
    sku: str
    distribution: str
    holding_mode: str
    time_unit: str
    inputs: dict[str, Any]
    tolerance: float
    max_iterations: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def read_sku_sheet(content: bytes) -> pd.DataFrame:
    """Read the dedicated data sheet. The Guide sheet is never interpreted as SKUs."""
    if not content:
        raise ValueError("The uploaded Excel file is empty.")
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("Excel file exceeds the 10 MB upload limit.")
    try:
        df = pd.read_excel(BytesIO(content), sheet_name=SHEET_NAME, engine="openpyxl", dtype=object)
    except ValueError as exc:
        raise ValueError(f"Expected an Excel worksheet named '{SHEET_NAME}'. {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Could not read this .xlsx file: {exc}") from exc
    df.columns = [str(c).strip() for c in df.columns]
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("Duplicate column headers detected (including headers differing only by spaces).")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("Missing required Excel columns: " + ", ".join(missing))
    df = df.dropna(how="all")
    if len(df) > MAX_SKUS:
        raise ValueError(f"Too many SKU rows: {len(df)}; maximum {MAX_SKUS}.")
    if df.empty:
        raise ValueError("No SKU rows found in the SKU_Inputs worksheet.")
    return df


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def validate_skus(df: pd.DataFrame, parse_math: Callable[[str], Any]) -> list[ImportedSKU]:
    """Validate each row; invalid rows remain visible and cannot be calculated.

    `parse_math` is app.parse_math; returned MathInput objects go straight to
    app._context, without a second interpretation of the spreadsheet values.
    """
    rows: list[ImportedSKU] = []
    seen: dict[str, int] = {}
    for index, raw in df.iterrows():
        excel_row = int(index) + 2
        sku = _as_text(raw.get("SKU"))
        distribution = _as_text(raw.get("Distribution"))
        holding_mode = _as_text(raw.get("Holding_mode")).lower()
        time_unit = _as_text(raw.get("Time_unit")).lower()
        errors: list[str] = []
        warnings: list[str] = []
        parsed: dict[str, Any] = {}

        if not sku:
            errors.append("SKU is required.")
        elif sku in seen:
            errors.append(f"Duplicate SKU '{sku}' (first appears on Excel row {seen[sku]}).")
        else:
            seen[sku] = excel_row
        if distribution not in DISTRIBUTIONS:
            errors.append(f"Distribution must be one of: {', '.join(DISTRIBUTIONS)}.")
        if holding_mode not in HOLDING_MODES:
            errors.append("Holding_mode must be 'rate' or 'direct'.")
        if time_unit not in TIME_UNITS:
            errors.append("Time_unit must be year, month, week, or day.")

        def number(name: str, *, required: bool = True, minimum: float | None = None,
                   strictly_positive: bool = False, maximum: float | None = None,
                   integer: bool = False) -> Any | None:
            raw_text = _as_text(raw.get(name))
            if not raw_text:
                if required:
                    errors.append(f"{name} is required.")
                return None
            try:
                item = parse_math(raw_text)
                if strictly_positive and item.value <= 0:
                    raise ValueError("must be > 0")
                if minimum is not None and item.value < minimum:
                    raise ValueError(f"must be >= {minimum}")
                if maximum is not None and item.value > maximum:
                    raise ValueError(f"must be <= {maximum}")
                if integer and item.exact.is_integer is not True:
                    raise ValueError("must be an integer")
                parsed[name] = item
                return item
            except (ValueError, OverflowError, TypeError) as exc:
                errors.append(f"{name}: {exc}")
                return None

        number("L", strictly_positive=True)
        number("A", strictly_positive=True)
        number("v", minimum=0)
        number("p", strictly_positive=True)
        if holding_mode == "rate":
            i = number("i", strictly_positive=True)
            if i is not None and "v" in parsed and i.value * parsed["v"].value <= 0:
                errors.append("Rate holding cost requires i × v > 0.")
        elif holding_mode == "direct":
            number("h", strictly_positive=True)

        if distribution == "Normal":
            number("mu", strictly_positive=True)
            number("sigma", strictly_positive=True)
            number("m", strictly_positive=True)
        elif distribution == "Binomial":
            number("n", minimum=1, maximum=1e8, integer=True)
            number("pi", strictly_positive=True, maximum=1 - 1e-12)
        elif distribution in DISTRIBUTIONS:
            number("D", strictly_positive=True)
            if distribution in ("Uniform", "Triangular"):
                a = number("a", minimum=0)
                b = number("b", minimum=0)
                if a is not None and b is not None and a.value >= b.value:
                    errors.append("Require a < b.")
                if distribution == "Triangular":
                    c = number("c", minimum=0)
                    if c is not None and a is not None and b is not None and not a.value <= c.value <= b.value:
                        errors.append("Require a <= c <= b.")
            elif distribution in ("Exponential", "Gamma", "Weibull"):
                number("scale", strictly_positive=True)
                if distribution != "Exponential":
                    number("shape", strictly_positive=True)
            elif distribution == "Lognormal":
                number("lt_mean", strictly_positive=True)
                number("lt_sd", strictly_positive=True)

        if distribution == "Poisson" and "D" in parsed and "L" in parsed:
            if parsed["D"].value * parsed["L"].value > 1e8:
                errors.append("Poisson λ = D × L must be <= 1e8.")

        discrete = distribution in ("Poisson", "Binomial")
        tolerance = DEFAULT_TOLERANCE
        raw_tol = _as_text(raw.get("tolerance"))
        if raw_tol and not discrete:
            t = number("tolerance", strictly_positive=True)
            if t is not None:
                if not any(math.isclose(t.value, allowed, rel_tol=1e-12) for allowed in ALLOWED_TOLERANCES):
                    errors.append("tolerance must be one of 1e-4, 1e-6, 1e-8, 1e-10, 1e-12.")
                else:
                    tolerance = t.value
        max_iterations = DEFAULT_MAX_ITERATIONS["discrete" if discrete else "continuous"]
        if _as_text(raw.get("max_iterations")):
            count = number("max_iterations", minimum=10 if discrete else 11,
                           maximum=500000 if discrete else 1000, integer=True)
            if count is not None:
                max_iterations = int(count.value)

        if not errors and distribution not in ("Normal", "Binomial", "Poisson"):
            d_l = parsed["D"].value * parsed["L"].value
            try:
                if distribution == "Uniform":
                    mean = (parsed["a"].value + parsed["b"].value) / 2
                elif distribution == "Triangular":
                    mean = (parsed["a"].value + parsed["b"].value + parsed["c"].value) / 3
                elif distribution == "Exponential":
                    mean = parsed["scale"].value
                elif distribution == "Gamma":
                    mean = parsed["shape"].value * parsed["scale"].value
                elif distribution == "Lognormal":
                    mean = parsed["lt_mean"].value
                else:
                    mean = parsed["scale"].value * math.gamma(1 + 1 / parsed["shape"].value)
                if not math.isfinite(mean):
                    errors.append("Lead-time distribution mean is not finite.")
                elif abs(mean - d_l) > 1e-6 * max(1.0, abs(d_l), abs(mean)):
                    warnings.append(
                        f"Lead-time mean E[X]={mean:.8g} differs from D×L={d_l:.8g}. "
                        "The existing solver does not automatically reconcile them."
                    )
            except (OverflowError, ValueError, ZeroDivisionError) as exc:
                errors.append(f"Could not evaluate lead-time mean: {exc}")

        rows.append(ImportedSKU(
            excel_row=excel_row, sku=sku, distribution=distribution,
            holding_mode=holding_mode, time_unit=time_unit,
            inputs=parsed, tolerance=tolerance, max_iterations=max_iterations,
            errors=tuple(errors), warnings=tuple(warnings),
        ))
    return rows


def make_context_arguments(item: ImportedSKU, parse_math: Callable[[str], Any]) -> dict[str, Any]:
    """Translate one validated SKU into app._context keyword arguments."""
    if not item.valid:
        raise ValueError("Cannot calculate an invalid SKU row.")
    v = item.inputs
    get = v.get
    return dict(
        case=f"Excel · {item.sku}", distribution=item.distribution,
        mode=HOLDING_MODES[item.holding_mode],
        mu=get("mu"), sigma=get("sigma"), periods=get("m"),
        demand=get("D"), lead=get("L"), a=get("a"), b=get("b"),
        scale=get("scale"), shape=get("shape"), tri_mode=get("c"),
        lt_mean=get("lt_mean"), lt_sd=get("lt_sd"),
        trials=get("n"), success_probability=get("pi"),
        A=get("A"), v=get("v"), i=get("i"), h_direct=get("h"),
        p=get("p"), tolerance=item.tolerance,
        max_iterations=parse_math(str(item.max_iterations)),
    )
