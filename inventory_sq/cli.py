"""Zero-UI command-line runner for reproducing the handwritten examples."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .examples import example_1, example_2
from .solver import solve_iteratively


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative stochastic (s,Q) solver")
    parser.add_argument("--case", choices=["ex1", "ex2"], default="ex1")
    parser.add_argument("--csv", type=Path, help="Export complete iteration table to CSV")
    parser.add_argument("--json", type=Path, help="Export summary and iterations to JSON")
    parser.add_argument("--tol", type=float, default=1e-10)
    args = parser.parse_args()

    case = example_1() if args.case == "ex1" else example_2()
    solution = solve_iteratively(case.costs, case.lead_time_distribution, tolerance=args.tol)
    print(case.title)
    print(case.note)
    print(f"Lead-time mean={solution.lead_time_mean:.8f}")
    print(f"EOQ={solution.eoq:.8f} | Converged={solution.converged}")
    print("iteration       Q_in            s           n(s)         Q_out")
    for row in solution.records:
        print(
            f"{row.index:>9} {row.q_in:>12.6f} {row.s:>12.6f} "
            f"{row.expected_shortage:>12.6f} {row.q_out:>12.6f}"
        )
    print(f"OPTIMUM: Q={solution.order_quantity:.8f}, s={solution.reorder_point:.8f}")
    print(f"Expected annual/per-period controllable cost={solution.costs.variable_total:.6f}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(asdict(solution.records[0])))
            writer.writeheader()
            writer.writerows(asdict(record) for record in solution.records)
        print(f"Exported CSV: {args.csv}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as stream:
            json.dump(asdict(solution), stream, indent=2, ensure_ascii=False)
        print(f"Exported JSON: {args.json}")


if __name__ == "__main__":
    main()
