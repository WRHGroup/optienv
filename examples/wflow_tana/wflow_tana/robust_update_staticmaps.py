#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Staticmaps Updater for Wflow.jl SBM (Optiverse-MOEA-NSGAII)

Features:
- Updates only target_vars: ["thetaS","InfiltCapSoil","SoilThickness","KsatVer","KsatHorFrac"]
- Never modifies wflow_ldd during parameter updates.
- Hardens wflow_ldd at the end: NaNs -> nodata, clamp codes to 0..9, cast to int32.
- Preserves dataset structure and metadata.
- Validates/clips to decision variables ranges; enforces thetaS > thetaR (without changing thetaR).
- Atomic write with compression; optional backup and JSON summary.
"""

import argparse, json, os, sys, shutil, tempfile
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import xarray as xr

# -------------------------
# Logging Helpers
# -------------------------
def log(msg: str) -> None:
    print(f"[StaticmapsUpdater] {msg}", flush=True)

def warn(msg: str) -> None:
    print(f"[StaticmapsUpdater][WARN] {msg}", flush=True)

def fail(msg: str, code: int = 2) -> None:
    print(f"[StaticmapsUpdater][ERROR] {msg}", flush=True)
    sys.exit(code)

# -------------------------
# .tbl Reader
# -------------------------
def read_tbl(path: str, pname: str) -> pd.DataFrame:
    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            rows.append(parts)
    if not rows:
        raise RuntimeError(f"No valid rows in {path}")
    first = [r[0] for r in rows]
    last = [r[-1] for r in rows]
    df = pd.DataFrame({"landuse_id": first, pname: last})
    df["landuse_id"] = pd.to_numeric(df["landuse_id"], errors="coerce")
    if df["landuse_id"].isna().any():
        raise RuntimeError(f"Invalid landuse_id values in {path}")
    df["landuse_id"] = df["landuse_id"].astype(int)
    df[pname] = pd.to_numeric(df[pname], errors="coerce")
    if df[pname].isna().any():
        raise RuntimeError(f"Invalid numeric values for {pname} in {path}")
    df[pname] = df[pname].astype(float)
    return df

# -------------------------
# Unit Handling
# -------------------------
def normalize_unit(u: Optional[str]) -> Optional[str]:
    if u is None:
        return None
    u = u.lower().replace(" ", "")
    u = u.replace("per", "/").replace("s-1", "/s").replace("day-1", "/day")
    return u

def scale_value_to_ds_units(var: str, val: float, ds_unit: Optional[str], src_unit: Optional[str]) -> float:
    ds_u = normalize_unit(ds_unit)
    src_u = normalize_unit(src_unit)

    if var in ("thetaS", "N_River"):
        return float(val)

    if var in ("SoilThickness", "RootingDepth"):
        if src_u is None: src_u = "mm"
        if ds_u in (None, "mm"):
            return float(val) if src_u == "mm" else float(val * 1000.0)
        if ds_u == "m":
            return float(val / 1000.0) if src_u == "mm" else float(val)

    if var == "KsatVer":
        if src_u is None: src_u = "mm/day"
        if ds_u in (None, "mm/day"):
            return float(val) if src_u == "mm/day" else float(val * 1000.0)
        if ds_u == "m/day":
            return float(val / 1000.0) if src_u == "mm/day" else float(val)
        if ds_u in ("m/s", "m/second"):
            if src_u == "mm/day": return float((val / 1000.0) / 86400.0)
            if src_u == "m/day":  return float(val / 86400.0)
            if src_u == "mm/s":   return float(val / 1000.0)
            if src_u == "m/s":    return float(val)
        if ds_u in ("mm/s",):
            if src_u == "mm/day": return float(val / 86400.0)
            if src_u == "m/day":  return float((val * 1000.0) / 86400.0)
            if src_u == "m/s":    return float(val * 1000.0)
            if src_u == "mm/s":   return float(val)

    return float(val)

# -------------------------
# Validation
# -------------------------
def clip_positive(arr: np.ndarray, minval: float = 1e-6) -> None:
    arr[np.isnan(arr) | (arr <= 0)] = minval

def enforce_bounds(arr: np.ndarray, lo: float, hi: float) -> int:
    before = arr.copy()
    np.clip(arr, lo, hi, out=arr)
    return int(np.sum(arr != before))

# -------------------------
# Harden wflow_ldd at the end
# -------------------------
def harden_ldd_da(ldd_da: xr.DataArray, nodata: int = -9999) -> xr.DataArray:
    arr = ldd_da.values
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.where(np.isnan(arr), nodata, np.rint(arr))
    arr = np.clip(arr, 0, 9)
    arr_i32 = arr.astype(np.int32)
    hardened = xr.DataArray(
        arr_i32, dims=ldd_da.dims, coords=ldd_da.coords,
        attrs={"long_name": "Local Drain Direction", "units": "-"}
    )
    return hardened

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Robust Staticmaps Updater for Wflow.jl SBM (MOEA-NSGAII)")
    ap.add_argument("--staticmap", required=True, help="Path to staticmaps.nc")
    ap.add_argument("--param_dir", required=True, help="Directory with .tbl parameter files")
    ap.add_argument("--landuse_var", default="wflow_landuse", help="Landuse variable name in staticmaps.nc")
    ap.add_argument("--mapping_file", default=None, help="Optional JSON mapping")
    ap.add_argument("--source_units", default=None, help="Optional JSON {var: unit}")
    ap.add_argument("--backup", action="store_true", help="Create staticmaps.nc.bak before writing")
    ap.add_argument("--summary_json", default=None, help="Write JSON summary to path")
    ap.add_argument("--strict", action="store_true", help="Abort on invalid values instead of clipping")
    ap.add_argument("--ldd_nodata", type=int, default=-9999, help="Nodata value for wflow_ldd")
    args = ap.parse_args()

    staticmaps_path = os.path.abspath(args.staticmap)
    param_dir = os.path.abspath(args.param_dir)

    if not os.path.isfile(staticmaps_path):
        fail(f"Staticmap not found: {staticmaps_path}")
    if not os.path.isdir(param_dir):
        fail(f"Param dir not found: {param_dir}")

    # Mapping 14. 20. 30. 40. 60. 110. 120. 130. 140. 150. 200. 210.
    mapping = {1:14,2:20,3:30,4:40,5:60,6:110,7:120,8:130, 9:140,10:150,11:200,12:210,13:None}
    if args.mapping_file:
        with open(args.mapping_file,"r",encoding="utf-8") as f:
            mm = json.load(f)
        mapping = {int(k):(None if v is None else int(v)) for k,v in mm.items()}

    src_units = {"SoilThickness":"mm","InfiltCapSoil":"mm/day","KsatVer":"mm/day","KsatHorFrac":"mm/day","RootingDepth":"mm","thetaS":None}
    if args.source_units:
        with open(args.source_units,"r",encoding="utf-8") as f:
            uu = json.load(f)
        for k,v in uu.items():
            src_units[k] = v

    # Only these variables are updated now
    target_vars = ["thetaS","InfiltCapSoil","SoilThickness","KsatVer","KsatHorFrac","RootingDepth","N_River"]

    # Load dataset
    ds = xr.load_dataset(staticmaps_path)
    if args.landuse_var not in ds:
        ds.close(); fail(f"Landuse variable '{args.landuse_var}' not found")
    landuse_da = ds[args.landuse_var]
    landuse = landuse_da.values
    if np.issubdtype(landuse.dtype, np.floating):
        landuse = np.where(np.isnan(landuse), -9999, np.rint(landuse)).astype(np.int32)
    else:
        landuse = landuse.astype(np.int32)

    # Read .tbl files
    tbl_paths = {var: os.path.join(param_dir, f"{var}.tbl") for var in target_vars}
    dfs: Dict[str, pd.DataFrame] = {}
    for var, path in tbl_paths.items():
        if os.path.exists(path):
            dfs[var] = read_tbl(path, var)
            log(f"Loaded {var}.tbl ({len(dfs[var])} rows)")
        else:
            warn(f"{var}.tbl not found; skipping")
    if not dfs:
        ds.close(); fail("No valid .tbl parameter files parsed")

    # Track which variables were actually updated by .tbl files
    updated_vars = set(dfs.keys())

    arrays: Dict[str, np.ndarray] = {}
    for var in dfs.keys():
        if var in ds:
            base = ds[var].values
            if base.shape != landuse.shape:
                ds.close(); fail(f"Shape mismatch for {var}")
            arrays[var] = base.astype(np.float32).copy()
        else:
            # If the var is not present in ds, initialize with NaN and fill only mapped classes
            arrays[var] = np.full_like(landuse, np.nan, dtype=np.float32)

    summary: Dict[str, Dict[str, int]] = {}

    for var, df in dfs.items():
        applied_classes = 0; applied_cells = 0
        ds_unit = ds[var].attrs.get("units") if var in ds else None
        src_unit = src_units.get(var)
        for _, row in df.iterrows():
            oid  = int(row["landuse_id"])
            val_src = float(row[var])
            wflow_id = mapping.get(oid, None)
            if wflow_id is None:
                continue
            mask = (landuse == int(wflow_id))
            if mask.any():
                val_scaled = scale_value_to_ds_units(var, val_src, ds_unit, src_unit)
                arrays[var][mask] = val_scaled
                applied_classes += 1; applied_cells += int(mask.sum())
        summary[var] = {"applied_classes": applied_classes, "applied_cells": applied_cells}
        log(f"{var}: applied to {applied_classes} classes; {applied_cells} cells")

    # Update only target_vars
    for var, arr in arrays.items():
        attrs = ds[var].attrs if var in ds else {"long_name": var, "units": src_units.get(var) or "-"}
        ds[var] = xr.DataArray(arr, dims=landuse_da.dims, coords=landuse_da.coords, attrs=attrs)

    # Validation / bounds (keep your current bounds)
    ost_bounds = {
        "thetaS": (0.01, 0.70),
        "N_River": (0.04, 0.5),
        "SoilThickness": (2000, 9000),
        "KsatVer": (500, 10000),
        "RootingDepth": (100, 5000),
        "KsatHorFrac": (500,1000),
        "InfiltCapSoil":(100,400),
    }

    clipped_total = 0
    for var, (lo_src, hi_src) in ost_bounds.items():
        if var in ds:
            lo = scale_value_to_ds_units(var, lo_src, ds[var].attrs.get("units"), src_units.get(var))
            hi = scale_value_to_ds_units(var, hi_src, ds[var].attrs.get("units"), src_units.get(var))
            arr = ds[var].values
            clipped_total += enforce_bounds(arr, lo, hi)

    for var in ["SoilThickness","RootingDepth","KsatVer","KsatHorFrac"]:
        if var in ds:
            arr = ds[var].values
            clip_positive(arr, 1e-6)
            ds[var].values[:] = arr

    # Enforce theta relation without touching thetaR:
    # Only when thetaS was updated and thetaR was NOT updated in this run.
    if ("thetaS" in ds and "thetaR" in ds
            and "thetaS" in updated_vars and "thetaR" not in updated_vars):
        ts = ds["thetaS"].values
        tr = ds["thetaR"].values
        margin = 0.02
        bad = ts <= tr
        if bad.any():
            ts[bad] = np.maximum(ts[bad], tr[bad] + margin)
            ds["thetaS"].values[:] = ts
            clipped_total += int(bad.sum())
            warn(f"thetaS raised at {int(bad.sum())} cells to satisfy thetaS>thetaR+{margin}")

    # Harden wflow_ldd at the end (unchanged)
    if "wflow_ldd" in ds:
        ds["wflow_ldd"] = harden_ldd_da(ds["wflow_ldd"], nodata=args.ldd_nodata)
        log("Final hardening: wflow_ldd -> int32")

    # Encoding and write
    encoding: Dict[str, Dict] = {}
    for var in ds.data_vars:
        enc = {"zlib": True, "complevel": 4}
        if var == "wflow_ldd":
            enc["dtype"] = "int32"; enc["_FillValue"] = np.int32(args.ldd_nodata)
        elif np.issubdtype(ds[var].dtype, np.floating):
            enc["dtype"] = "float32"
        encoding[var] = enc

    if args.backup:
        shutil.copy2(staticmaps_path, staticmaps_path + ".bak")

    # Strict mode: abort if any clipping/adjustment occurred
    if args.strict and clipped_total > 0:
        ds.close()
        fail(f"Strict mode: {clipped_total} value(s) clipped or adjusted; no file written.")

    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(staticmaps_path) + ".", suffix=".tmp",
        dir=os.path.dirname(staticmaps_path)
    )
    os.close(fd)
    ds.to_netcdf(tmp_path, mode="w", engine="netcdf4", format="NETCDF4", encoding=encoding)
    ds.close()
    shutil.move(tmp_path, staticmaps_path)
    log(f"Updated staticmaps written: {staticmaps_path}")

    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump({"applied": summary, "landuse_var": args.landuse_var}, f, indent=2)

if __name__ == "__main__":
    main()