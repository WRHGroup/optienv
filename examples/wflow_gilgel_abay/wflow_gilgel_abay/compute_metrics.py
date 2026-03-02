#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import pandas as pd

# ---------- Built-in GRDC station mapping (obs_name -> sim column) ----------
BUILTIN_STATION_MAP = {
    "Gabay":  "Q_200",
}

# -------- Metrics --------
def _safe(a):
    x = np.asarray(a, dtype=float)
    return x[np.isfinite(x)]

def nse(o, s):
    o = _safe(o); s = _safe(s)
    if o.size == 0 or s.size == 0:
        return np.nan
    n = min(len(o), len(s)); o = o[:n]; s = s[:n]
    denom = np.sum((o - np.mean(o))**2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((o - s)**2) / denom

def kge_prime_2012(o, s):
    """KGE' (2012), returns (-inf, 1], best = 1"""
    o = _safe(o); s = _safe(s)
    if o.size == 0 or s.size == 0:
        return np.nan
    n = min(len(o), len(s)); o = o[:n]; s = s[:n]
    mo, ms = np.mean(o), np.mean(s)
    so, ss = np.std(o, ddof=0), np.std(s, ddof=0)
    if mo == 0 or so == 0 or ms == 0:
        return np.nan
    r = np.corrcoef(o, s)[0, 1]
    beta = ms / mo
    gamma = (ss / ms) / (so / mo)
    if not np.isfinite(r) or not np.isfinite(beta) or not np.isfinite(gamma):
        return np.nan
    return float(1.0 - np.sqrt((r - 1.0)**2 + (beta - 1.0)**2 + (gamma - 1.0)**2))

def log_nse(o, s, eps_factor=0.08, eps_floor=1e-6):
    """
    NSE on log-transformed flows (emphasizes low flows).
    eps = max(eps_floor, eps_factor * mean(obs))
    """
    o = _safe(o); s = _safe(s)
    if o.size == 0 or s.size == 0:
        return np.nan
    n = min(len(o), len(s)); o = o[:n]; s = s[:n]
    eps = max(eps_floor, eps_factor * np.mean(o))
    return nse(np.log(o + eps), np.log(s + eps))

def pbias(o, s):
    """Percent bias: 100 * (sum(sim) - sum(obs)) / sum(obs)."""
    o = _safe(o); s = _safe(s)
    if o.size == 0 or s.size == 0:
        return np.nan
    n = min(len(o), len(s)); o = o[:n]; s = s[:n]
    so = np.sum(o)
    if so == 0:
        return np.nan
    return 100.0 * (np.sum(s) - so) / so

# -------- Helpers --------
def clean_header_names(cols):
    return [str(c).strip() for c in cols]

def write_bad(out_dir):
    outpath = os.path.join(out_dir, "metrics.txt")
    with open(outpath, "w", encoding="ascii") as f:
        f.write("-1000,-1000,-1000\n")
    sys.stderr.write("[compute_metrics] penalty (failure or NaNs)\n")

def write_metrics(out_dir, KGEp_value, logNSE_value, bias_score_value):
    outpath = os.path.join(out_dir, "metrics.txt")
    line = f"{KGEp_value},{logNSE_value},{bias_score_value}"
    with open(outpath, "w", encoding="ascii") as f:
        f.write(line + "\n")
    print(f"[compute_metrics] Wrote {outpath}: {line}")

# -------- Main --------
def main():
    ap = argparse.ArgumentParser(
        description="Outputs maximize-ready KGEp, logNSE, bias_score aggregated over multiple stations."
    )
    ap.add_argument("--sim", required=True, help="Path to simulation CSV")
    ap.add_argument("--obs", required=True, help="Path to observed CSV")
    ap.add_argument("--out_dir", required=True, help="Directory to write metrics.txt")
    ap.add_argument("--sep", default=",", help="CSV separator (default=',')")
    ap.add_argument("--warmup_years", type=int, default=1,
                    help="Drop 365*years rows from start (assumes daily).")
    ap.add_argument("--eps_factor", type=float, default=0.08,
                    help="Epsilon factor for log-NSE (default 0.08 for low-flow robustness).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Read
    try:
        obs = pd.read_csv(args.obs, sep=args.sep)
        sim = pd.read_csv(args.sim, sep=args.sep)
    except Exception as e:
        sys.stderr.write(f"[compute_metrics] CSV read error: {e}\n")
        write_bad(args.out_dir); return

    obs.columns = clean_header_names(obs.columns)
    sim.columns = clean_header_names(sim.columns)

    # Collect valid station pairs present in both files
    pairs = [(oc, sc) for oc, sc in BUILTIN_STATION_MAP.items()
             if oc in obs.columns and sc in sim.columns]
    if not pairs:
        sys.stderr.write("[compute_metrics] No valid station mapping found.\n")
        write_bad(args.out_dir); return

    # Per-station metrics
    kge_list, lognse_list, bias_list = [], [], []

    for oc, sc in pairs:
        try:
            o = pd.to_numeric(obs[oc], errors="coerce").to_numpy()
            s = pd.to_numeric(sim[sc], errors="coerce").to_numpy()
        except Exception as e:
            sys.stderr.write(f"[compute_metrics] Column extraction failed for {oc}->{sc}: {e}\n")
            write_bad(args.out_dir); return

        o = o[np.isfinite(o)]
        s = s[np.isfinite(s)]
        n = min(len(o), len(s))
        if n == 0:
            sys.stderr.write(f"[compute_metrics] Empty series after cleaning for {oc}->{sc}.\n")
            write_bad(args.out_dir); return
        o, s = o[:n], s[:n]

        # Warm-up trim (years-only)
        trim = 365 * int(args.warmup_years)
        if trim > 0:
            if len(o) <= trim or len(s) <= trim:
                sys.stderr.write(f"[compute_metrics] Warm-up exceeds length for {oc}->{sc}.\n")
                write_bad(args.out_dir); return
            o = o[trim:]; s = s[trim:]

        # Per-station metrics
        KGEp = kge_prime_2012(o, s)
        logNSE_val = log_nse(o, s, eps_factor=args.eps_factor)
        PBIAS_val = pbias(o, s)

        if any(np.isnan(x) for x in [KGEp, logNSE_val, PBIAS_val]):
            sys.stderr.write(f"[compute_metrics] NaN metric for {oc}->{sc}.\n")
            write_bad(args.out_dir); return

        bias_score = 1.0 - abs(PBIAS_val) / 100.0  # maximize (best = 1)

        kge_list.append(KGEp)
        lognse_list.append(logNSE_val)
        bias_list.append(bias_score)

    # ------- Aggregate across stations (equal-weight mean) -------
    KGEp   = round(float(np.mean(kge_list)),   6)
    logNSE = round(float(np.mean(lognse_list)), 6)
    bias_score  = round(float(np.mean(bias_list)),   6)

    # ------- Write maximize-ready triple (no headers, no JSON) -------
    write_metrics(args.out_dir, KGEp, logNSE, bias_score)

if __name__ == "__main__":
    main()