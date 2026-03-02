# examples/toy_6obj/toy_model/wrapper_toy6.py
from __future__ import annotations
import csv
import os

def _read_variable_values(model_folder: str) -> dict[str, float]:
    path = os.path.join(model_folder, "variable_values.csv")
    vals: dict[str, float] = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            vals[str(row["Name"])] = float(row["Value"])
    return vals

def _write_objectives(model_folder: str, obj: dict[str, float]) -> None:
    path = os.path.join(model_folder, "objective_values.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Name", "Value"])
        for k, v in obj.items():
            w.writerow([k, v])

def search_and_apply_variables(model_folder: str) -> None:
    x_dict = _read_variable_values(model_folder)
    xs = list(x_dict.values())

    def sq(center: float) -> float:
        return sum((v - center) * (v - center) for v in xs)

    f1 = sq(0.0)     # minimize
    f2 = sq(1.0)     # minimize
    f3 = sq(0.25)    # minimize
    f4 = sq(0.50)    # minimize
    f5 = sq(0.75)    # minimize
    f6 = sum(xs)     # maximize (declared as maximize in CSV; raw positive here)

    _write_objectives(model_folder, {
        "f1_min_sq_0.00": f1,
        "f2_min_sq_1.00": f2,
        "f3_min_sq_0.25": f3,
        "f4_min_sq_0.50": f4,
        "f5_min_sq_0.75": f5,
        "f6_sum_max": f6
    })