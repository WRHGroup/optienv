#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Staticmaps Updater for Wflow.jl SBM (MOEA/NSGAII calibration)

Features:
- Updates target_vars: ["thetaS","SoilThickness","KsatHorFrac","Cfmax","WHC","TT","TTI","TTM"]
  from per class .tbl parameter files written by the calibration tool.
- Never modifies wflow_ldd during parameter updates.
- Hardens wflow_ldd at the end: NaNs -> nodata, clamp valid codes to 0..9, cast to int32
  (nodata cells are kept as nodata).
- Preserves dataset structure and metadata.
- Validates/clips parameters to the calibration parameter ranges; enforces thetaR < thetaS.
  --bounds_scope all        : clip the whole grid (default, original behaviour)
  --bounds_scope calibrated : clip only calibrated variables in calibrated cells
  --strict                  : abort if a .tbl value lies outside the calibration parameter ranges
- Atomic write with compression; optional backup and JSON summary.
"""
import argparse, json, os, sys, shutil, tempfile
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import xarray as xr

# -------------------
# Logging Helpers
# -------------------
def log(msg: str) -> None:
    print(f"[StaticmapsUpdater] {msg}", flush=True)
def warn(msg: str) -> None:
    print(f"[StaticmapsUpdater][WARN] {msg}", flush=True)
def fail(msg: str, code: int = 2) -> None:
    print(f"[StaticmapsUpdater][ERROR] {msg}", flush=True)
    sys.exit(code)

# -------------------
# .tbl Reader
# First column: calibration class id; last column: parameter value
# -------------------
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
    df = pd.DataFrame({"class_id": first, pname: last})
    df["class_id"] = df["class_id"].astype(int)
    df[pname] = df[pname].astype(float)
    return df

# -------------------
# Unit Handling
# -------------------
def normalize_unit(u: Optional[str]) -> Optional[str]:
    if u is None:
        return None
    u = u.lower().replace(" ", "")
    u = u.replace("per", "/").replace("s-1", "/s").replace("day-1", "/day")
    return u

def scale_value_to_ds_units(var: str, val: float, ds_unit: Optional[str], src_unit: Optional[str]) -> float:
    ds_u = normalize_unit(ds_unit)
    src_u = normalize_unit(src_unit)
    if var in ("thetaS", "thetaR", "N", "alpha_h1", "KsatHorFrac"):
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
            if src_u == "m/day": return float(val / 86400.0)
            if src_u == "mm/s": return float(val / 1000.0)
            if src_u == "m/s": return float(val)
        if ds_u in ("mm/s",):
            if src_u == "mm/day": return float(val / 86400.0)
            if src_u == "m/day": return float((val * 1000.0) / 86400.0)
            if src_u == "m/s": return float(val * 1000.0)
            if src_u == "mm/s": return float(val)
    return float(val)

# -------------------
# Validation
# -------------------
def clip_positive(arr: np.ndarray, minval: float = 1e-6) -> None:
    arr[np.isnan(arr) | (arr <= 0)] = minval
def enforce_bounds(arr: np.ndarray, lo: float, hi: float, mask: Optional[np.ndarray] = None) -> int:
    before = arr.copy()
    if mask is None:
        np.clip(arr, lo, hi, out=arr)
    else:
        arr[mask] = np.clip(arr[mask], lo, hi)
    return int(np.sum((arr != before) & ~(np.isnan(arr) & np.isnan(before))))
def ensure_theta_relation(thetaS: np.ndarray, thetaR: np.ndarray, margin: float = 0.02) -> int:
    bad = thetaR >= thetaS
    nbad = int(bad.sum())
    if nbad > 0:
        thetaR[bad] = np.maximum(0.0, thetaS[bad] - margin)
    return nbad

# -------------------
# Harden wflow_ldd at the end
# -------------------
def harden_ldd_da(ldd_da: xr.DataArray, nodata: int = -9999) -> xr.DataArray:
    arr = ldd_da.values.astype(np.float64)
    is_nodata = np.isnan(arr) | (arr == nodata)
    valid = np.clip(np.rint(np.where(is_nodata, 0, arr)), 0, 9)
    arr_i32 = np.where(is_nodata, nodata, valid).astype(np.int32)
    hardened = xr.DataArray(arr_i32, dims=ldd_da.dims, coords=ldd_da.coords,
                             attrs={"long_name": "Local Drain Direction", "units": "-"})
    return hardened

# -------------------
# Main
# -------------------
def main():
    ap = argparse.ArgumentParser(description="Robust Staticmaps Updater for Wflow.jl SBM (MOEA/NSGAII calibration)")
    ap.add_argument("--staticmap", required=True, help="Path to staticmaps.nc")
    ap.add_argument("--param_dir", required=True, help="Directory with .tbl parameter files")
    ap.add_argument("--landuse_var", default="wflow_landuse", help="Landuse variable name in staticmaps.nc")
    ap.add_argument("--mapping_file", default=None, help="Optional JSON mapping {class_id: wflow_landuse code}")
    ap.add_argument("--source_units", default=None, help="Optional JSON {var: unit}")
    ap.add_argument("--backup", action="store_true", help="Create staticmaps.nc.bak before writing")
    ap.add_argument("--summary_json", default=None, help="Write JSON summary to path")
    ap.add_argument("--strict", action="store_true",
                    help="Abort if a .tbl value lies outside the calibration parameter ranges (default: clip)")
    ap.add_argument("--bounds_scope", choices=["all", "calibrated"], default="all",
                    help="'all': clip bounded variables over the whole grid (original behaviour); "
                         "'calibrated': clip only calibrated variables in calibrated cells")
    ap.add_argument("--ldd_nodata", type=int, default=-9999, help="Nodata value for wflow_ldd")
    args = ap.parse_args()

    staticmaps_path = os.path.abspath(args.staticmap)
    param_dir = os.path.abspath(args.param_dir)
    if not os.path.isfile(staticmaps_path):
        fail(f"Staticmap not found: {staticmaps_path}")
    if not os.path.isdir(param_dir):
        fail(f"Param dir not found: {param_dir}")

    # Mapping: calibration class id -> wflow_landuse code (None = not calibrated)
    mapping = {1:10,2:30,3:60,4:70,5:80,6:100,7:None}
    if args.mapping_file:
        with open(args.mapping_file,"r",encoding="utf-8") as f:
            mm=json.load(f)
        mapping={int(k):(None if v is None else int(v)) for k,v in mm.items()}

    src_units={"SoilThickness":"mm","thetaS": None,"KsatHorFrac":None}
    if args.source_units:
        with open(args.source_units,"r",encoding="utf-8") as f:
            uu=json.load(f)
        for k,v in uu.items(): src_units[k]=v

    target_vars=["thetaS","SoilThickness","KsatHorFrac","Cfmax","WHC","TT","TTI","TTM"]

    # Calibration parameter ranges (source units)
    param_bounds={"Cfmax":(1.5,8.0),"TT":(-2.0,2.0),"TTI":(0.0,4.0),"TTM":(-2.0,2.0),"WHC":(0.02,0.20),
                  "KsatHorFrac":(10.0,2000.0),"SoilThickness":(200.0,3000.0),"thetaS":(0.30,0.65),"KsatVer":(50.0,3000.0)}

    # Load dataset
    ds=xr.load_dataset(staticmaps_path,engine="h5netcdf")

    if args.landuse_var not in ds:
        ds.close(); fail(f"Landuse variable '{args.landuse_var}' not found")
    landuse_da=ds[args.landuse_var]
    landuse=landuse_da.values
    if np.issubdtype(landuse.dtype,np.floating):
        landuse=np.where(np.isnan(landuse),-9999,np.rint(landuse)).astype(np.int32)
    else:
        landuse=landuse.astype(np.int32)

    # Read .tbl files
    tbl_paths={var:os.path.join(param_dir,f"{var}.tbl") for var in target_vars}
    dfs={}
    for var,path in tbl_paths.items():
        if os.path.exists(path):
            dfs[var]=read_tbl(path,var)
            log(f"Loaded {var}.tbl ({len(dfs[var])} rows)")
        else: warn(f"{var}.tbl not found; skipping")
    if not dfs: ds.close(); fail("No valid .tbl parameter files parsed")

    # Strict check: .tbl values must lie within the calibration parameter ranges
    if args.strict:
        for var,df in dfs.items():
            if var not in param_bounds: continue
            lo,hi=param_bounds[var]
            tol=1e-6*max(1.0,abs(hi-lo))
            bad=df[(df[var]<lo-tol)|(df[var]>hi+tol)]
            if not bad.empty:
                ds.close()
                fail(f"{var}: {len(bad)} value(s) outside [{lo}, {hi}] "
                     f"(class_id {bad['class_id'].tolist()}); aborting (--strict)")

    arrays={}
    for var in dfs.keys():
        if var in ds:
            base=ds[var].values
            if base.shape!=landuse.shape: ds.close(); fail(f"Shape mismatch for {var}")
            arrays[var]=base.astype(np.float32).copy()
        else:
            arrays[var]=np.full_like(landuse,np.nan,dtype=np.float32)

    summary={}
    calib_masks={}
    for var,df in dfs.items():
        applied_classes=0; applied_cells=0
        calib_masks[var]=np.zeros(landuse.shape,dtype=bool)
        ds_unit=ds[var].attrs.get("units") if var in ds else None
        src_unit=src_units.get(var)
        for _,row in df.iterrows():
            cid=int(row["class_id"]); val_src=float(row[var])
            wflow_id=mapping.get(cid,None)
            if wflow_id is None: continue
            mask=(landuse==int(wflow_id))
            if mask.any():
                val_scaled=scale_value_to_ds_units(var,val_src,ds_unit,src_unit)
                arrays[var][mask]=val_scaled
                calib_masks[var]|=mask
                applied_classes+=1; applied_cells+=int(mask.sum())
        summary[var]={"applied_classes":applied_classes,"applied_cells":applied_cells}
        log(f"{var}: applied to {applied_classes} classes; {applied_cells} cells")

    # Update only target_vars
    for var,arr in arrays.items():
        attrs=ds[var].attrs if var in ds else {"long_name":var,"units":src_units.get(var) or "-"}
        ds[var]=xr.DataArray(arr,dims=landuse_da.dims,coords=landuse_da.coords,attrs=attrs)

    # Validation: clip to calibration parameter ranges
    for var,(lo_src,hi_src) in param_bounds.items():
        if var not in ds: continue
        if args.bounds_scope=="calibrated" and var not in calib_masks: continue
        lo=scale_value_to_ds_units(var,lo_src,ds[var].attrs.get("units"),src_units.get(var))
        hi=scale_value_to_ds_units(var,hi_src,ds[var].attrs.get("units"),src_units.get(var))
        arr=ds[var].values
        mask=calib_masks[var] if args.bounds_scope=="calibrated" else None
        nclip=enforce_bounds(arr,lo,hi,mask)
        if nclip>0: log(f"{var}: {nclip} cells clipped to [{lo}, {hi}] (scope: {args.bounds_scope})")

    for var in ["SoilThickness","RootingDepth","KsatVer","alpha_h1"]:
        if var in ds:
            arr=ds[var].values
            clip_positive(arr,1e-6); ds[var].values[:]=arr

    if "thetaS" in ds and "thetaR" in ds:
        ts=ds["thetaS"].values; tr=ds["thetaR"].values
        ensure_theta_relation(ts,tr); ds["thetaR"].values[:]=tr

    # Harden wflow_ldd at the end
    if "wflow_ldd" in ds:
        ds["wflow_ldd"]=harden_ldd_da(ds["wflow_ldd"],nodata=args.ldd_nodata)
        log("Final hardening: wflow_ldd -> int32")

    # Encoding
    encoding={}
    for var in ds.data_vars:
        enc={"zlib":True,"complevel":4}
        if var=="wflow_ldd":
            enc["dtype"]="int32"; enc["_FillValue"]=np.int32(args.ldd_nodata)
        elif np.issubdtype(ds[var].dtype,np.floating):
            enc["dtype"]="float32"
        encoding[var]=enc

    if args.backup:
        shutil.copy2(staticmaps_path,staticmaps_path+".bak")

    fd,tmp_path=tempfile.mkstemp(prefix=os.path.basename(staticmaps_path)+".",suffix=".tmp",dir=os.path.dirname(staticmaps_path))
    os.close(fd)
    ds.to_netcdf(tmp_path,mode="w",engine="h5netcdf",format="NETCDF4",encoding=encoding)
    ds.close()
    shutil.move(tmp_path,staticmaps_path)
    log(f"Updated staticmaps written: {staticmaps_path}")

    if args.summary_json:
        with open(args.summary_json,"w",encoding="utf-8") as f:
            json.dump({"applied":summary,"landuse_var":args.landuse_var},f,indent=2)

if __name__=="__main__":
    main()
