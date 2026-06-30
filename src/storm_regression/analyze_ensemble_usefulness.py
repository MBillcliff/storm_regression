"""
Phase 0.4 — compare experiment groups by ROLE detected from the directory path.

Designed to be robust to mislabelled dict keys: the experiment type (selection method
and loss) is read from each result directory's PATH, not from the label you give it.
Folder-name conventions recognised:
    loss      : '...crps...'            -> CRPS     else                     -> NLL
    selection : '...raw_ensemble(s)...' -> snap     '...per_timestep...'      -> per_timestep
                '...snap...'            -> snap

Two controlled comparisons are then formed automatically:
    SELECTION : snap+NLL        vs  per_timestep+NLL   (loss held fixed = NLL)
    LOSS      : per_timestep+CRPS vs per_timestep+NLL  (selection held fixed = per_timestep)

Comparisons use a bootstrap CI on the median difference over all files in each group
(robust to unequal n and to missing fold/seed pairing). Lower is better for every metric,
so a POSITIVE (median_a - median_b) means group_b is better.

CAVEAT (printed at the end): all groups here are 12h lead and contain no [50]-only run,
so this does NOT answer the core 0.4 question (median-vs-ensemble usefulness at long lead).
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import kstest

from storm_regression.predictive import forecast_from_results
from storm_regression.results_io import load_results

METRICS = ["CRPS_all", "CRPS_storm", "Brier_storm", "PITKS_all", "PITKS_storm"]


# --------------------------------------------------------------------------
# role detection from the directory path (the reliable source of truth)
# --------------------------------------------------------------------------
def _detect_role(path):
    p = str(path).lower()
    loss = "CRPS" if "crps" in p else "NLL"
    if "raw_ensemble" in p or "raw_ensembles" in p or "snap" in p:
        sel = "snap"
    elif "per_timestep" in p or "per_ts" in p:
        sel = "per_timestep"
    else:
        sel = "unknown"
    return sel, loss


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def _score_file(fp, family="auto", storm_threshold=4.5):
    try:
        results, config, _ = load_results(fp)
        f = forecast_from_results(results, family=family)
        y = np.asarray(results["y_test"], dtype=float)
    except Exception as e:
        print(f"  skip {os.path.basename(fp)}: {e}")
        return None

    storm = y > storm_threshold
    crps = f.crps(y)
    pit = np.clip(f.pit(y), 0, 1)
    exc = f.exceedance_prob(storm_threshold)
    return {
        "file": os.path.basename(fp),
        "CRPS_all": float(np.mean(crps)),
        "CRPS_storm": float(np.mean(crps[storm])) if storm.any() else np.nan,
        "Brier_storm": float(np.mean((exc - storm.astype(float)) ** 2)),
        "PITKS_all": float(kstest(pit, "uniform").statistic),
        "PITKS_storm": float(kstest(pit[storm], "uniform").statistic) if storm.any() else np.nan,
        "n_storm": int(storm.sum()),
    }


def collect_groups(group_dirs, family="auto", pattern="*.pkl", storm_threshold=4.5):
    """Score every file; tag each row with its detected role (sel+loss), not its label."""
    rows = []
    role_of = {}
    for label, d in group_dirs.items():
        sel, loss = _detect_role(d)
        role = f"{sel}+{loss}"
        role_of[role] = label
        files = glob.glob(os.path.join(str(d), "**", pattern), recursive=True)
        print(f"label '{label}'  ->  detected role '{role}'  ({len(files)} file(s))")
        for fp in files:
            m = _score_file(fp, family=family, storm_threshold=storm_threshold)
            if m is not None:
                m["role"], m["label"] = role, label
                rows.append(m)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No result files scored. Check paths / pattern.")
    return df, role_of


# --------------------------------------------------------------------------
# robust comparison: bootstrap CI on the median difference
# --------------------------------------------------------------------------
def _bootstrap_median_diff(a, b, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    boot = np.array([np.median(rng.choice(a, a.size)) - np.median(rng.choice(b, b.size))
                     for _ in range(n_boot)])
    return float(np.median(a) - np.median(b)), float(np.percentile(boot, 5)), float(np.percentile(boot, 95))


def _compare(df, role_a, role_b, metric, name):
    a = df[df["role"] == role_a][metric].dropna().values
    b = df[df["role"] == role_b][metric].dropna().values
    print("\n" + "-" * 78)
    print(f"{name}   [{metric}]   {role_a} (n={a.size})  vs  {role_b} (n={b.size})")
    print("-" * 78)
    if a.size == 0 or b.size == 0:
        print("  one group empty — skipped")
        return
    diff, lo, hi = _bootstrap_median_diff(a, b)
    sig = "CI excludes 0 -> meaningful" if (lo > 0 or hi < 0) else "CI spans 0 -> not distinguishable"
    print(f"  median({role_a}) = {np.median(a):.4f}")
    print(f"  median({role_b}) = {np.median(b):.4f}")
    print(f"  median diff (a-b) = {diff:+.4f}   90% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  (+ => {role_b} better, since lower is better)   {sig}")


def group_summary(df):
    print("\n" + "=" * 78)
    print("GROUP SUMMARY  (median over files; lower is better everywhere)")
    print("=" * 78)
    for role in df["role"].unique():
        sub = df[df["role"] == role]
        line = f"{role:<20} (n={sub.shape[0]:>2})  "
        line += "  ".join(f"{m}={sub[m].median():.4f}" for m in METRICS)
        print(line)


# --------------------------------------------------------------------------
# top-level entry point — takes ONLY the groups dict
# --------------------------------------------------------------------------
def run_abc_analysis(group_dirs, family="auto", storm_threshold=4.5):
    df, role_of = collect_groups(group_dirs, family=family, storm_threshold=storm_threshold)
    group_summary(df)

    have = set(df["role"].unique())
    per_ts_nll, per_ts_crps, snap_nll = "per_timestep+NLL", "per_timestep+CRPS", "snap+NLL"

    if {snap_nll, per_ts_nll} <= have:
        print("\n##### SELECTION EFFECT  (snap vs per_timestep, loss = NLL) #####")
        _compare(df, snap_nll, per_ts_nll, "CRPS_storm", "SELECTION")
        _compare(df, snap_nll, per_ts_nll, "CRPS_all", "SELECTION")
        _compare(df, snap_nll, per_ts_nll, "PITKS_storm", "SELECTION")
    else:
        print(f"\n[SELECTION skipped] need {snap_nll} and {per_ts_nll}; have {sorted(have)}")

    if {per_ts_crps, per_ts_nll} <= have:
        print("\n##### LOSS EFFECT  (CRPS vs NLL, selection = per_timestep) #####")
        _compare(df, per_ts_crps, per_ts_nll, "PITKS_all", "LOSS")
        _compare(df, per_ts_crps, per_ts_nll, "PITKS_storm", "LOSS")
        _compare(df, per_ts_crps, per_ts_nll, "CRPS_storm", "LOSS")
    else:
        print(f"\n[LOSS skipped] need {per_ts_crps} and {per_ts_nll}; have {sorted(have)}")

    print("\n" + "=" * 78)
    print("HOW TO READ")
    print("=" * 78)
    print(
        "SELECTION: diff ~0 / CI spans 0 => snap doesn't beat per_timestep; member coherence\n"
        "           is lost in the MLP's flatten+pool head (argues FOR a set encoder).\n"
        "LOSS:      lower PITKS for CRPS (positive diff, CI excluding 0) => CRPS flattens\n"
        "           calibration vs NLL -> supports the Phase 1 twCRPS switch.\n"
        "ROBUST:    PITKS_storm ~0.5 in EVERY group = structural storm-tail miscalibration,\n"
        "           untouched by selection/loss -> the case for the mixture + twCRPS.\n"
        "CAVEAT:    12h lead, no [50]-only run -> core 0.4 (median-vs-ensemble at long lead)\n"
        "           remains open; needs a [50] floor and a 36h pair."
    )
    return df