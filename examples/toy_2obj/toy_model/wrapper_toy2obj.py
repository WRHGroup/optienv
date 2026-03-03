# examples/toy_3obj/toy_model/wrapper_toy.py
# A tiny 3-objective test "simulator" using the adapter contract.
# Objectives:
#   - f1_sum_squares (minimize):  sum(x_i^2)
#   - f2_sum_one_minus_squares (minimize):  sum((1 - x_i)^2)
#   - f3_sum (maximize):  sum(x_i)
#
# The adapter will read variable_values.csv (Name, Value) and expects us to
# write objective_values.csv (Name, Value). It also handles maximization by
# sign-flipping internally, so we write raw (non-negated) values here.

from __future__ import annotations
import csv
import os

def _read_variable_values(model_folder: str) -> dict[str, float]:
    path = os.path.join(model_folder, "variable_values.csv")
    values: dict[str, float] = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            values[str(row["Name"])] = float(row["Value"])
    return values

def _write_objectives(model_folder: str, obj: dict[str, float]) -> None:
    path = os.path.join(model_folder, "objective_values.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Name", "Value"])
        for k, v in obj.items():
            w.writerow([k, v])

def search_and_apply_variables(model_folder: str) -> None:
    """
    Required entry-point for the adapter.
    1) Read decision vector from variable_values.csv.
    2) Compute three objective values.
    3) Write objective_values.csv (Name, Value).
    """
    x = _read_variable_values(model_folder)   # dict: {"x1": val, "x2": val, ...}
    xs = list(x.values())

    # f1: sum of squares (minimize)
    f1 = sum(v*v for v in xs)

    # f2: sum of (1 - x)^2 (minimize)
    f2 = sum((1.0 - v) * (1.0 - v) for v in xs)

    # f3: sum of x (maximize)
    f3 = sum(xs)

    _write_objectives(model_folder, {
        "f1_sum_squares": f1,
        "f2_sum_one_minus_squares": f2,
        "f3_sum": f3
    })

# Optional hook for completeness
def simulate(model_folder: str) -> None:
    search_and_apply_variables(model_folder)