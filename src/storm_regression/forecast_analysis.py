# Ensemble analysis for regression forecasts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.colors import Normalize
from scipy.stats import weibull_min, kstest
from matplotlib.gridspec import GridSpec
from storm_regression.predictive import forecast_from_results
from scipy.stats import weibull_min
from scipy.integrate import quad
from scipy.special import gamma
import datetime
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
from storm_regression.results_io import load_results, recreate_dataset_from_results
from storm_regression.plotting import plot_comparative_case_study
import pickle
import logging
import pickle
import re


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_regression_forecast(
    y_pred: np.ndarray,
    y_true: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate regression forecast with multiple metrics.
    
    Args:
        y_pred: Model predictions
        y_true: True values
    
    Returns:
        Dictionary of metric names and values
    """
    # Ensure all arrays are 1D
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_pred - y_true))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    # Correlation (handle edge cases)
    if len(y_pred) > 1 and np.std(y_pred) > 0 and np.std(y_true) > 0:
        correlation = np.corrcoef(y_pred, y_true)[0, 1]
    else:
        correlation = 0.0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
    }


def evaluate_distribution_forecast(
    y_true,
    distribution: str,
    **dist_params
) -> Dict[str, float]:
    """Evaluate probabilistic forecasts using CRPS.

    Thin adapter over storm_regression.predictive: builds the appropriate
    PredictiveForecast and calls its closed-form crps(). A SINGLE CRPS
    implementation is now used everywhere. (The previous stand-alone lognormal
    branch scored the underlying Normal in log-space, giving values ~5x too
    small and rank-distorting; the weibull branch was also incorrect. Both are
    replaced by the verified closed forms in predictive.py.)

    Parameters
    ----------
    y_true : array-like
        Observed target values.
    distribution : {'weibull', 'normal', 'lognormal'}
    **dist_params :
        - 'weibull'   : lambda_pred, k_pred
        - 'normal'    : mu_pred, sigma_pred
        - 'lognormal' : log_mu_pred, log_sigma_pred

    Returns
    -------
    dict with keys 'crps' (float, mean over samples) and
    'crps_scores' (np.ndarray, per-sample).
    """
    from storm_regression.predictive import (
        LogNormalForecast, NormalForecast, WeibullForecast,
    )

    y_true = np.asarray(y_true, dtype=float).ravel()
    distribution = distribution.lower()

    if distribution == 'lognormal':
        log_mu = dist_params.get('log_mu_pred')
        log_sigma = dist_params.get('log_sigma_pred')
        if log_mu is None or log_sigma is None:
            raise ValueError("LogNormal CRPS requires 'log_mu_pred' and 'log_sigma_pred'")
        log_mu = np.asarray(log_mu, dtype=float).ravel()
        log_sigma = np.asarray(log_sigma, dtype=float).ravel()
        if not (len(y_true) == len(log_mu) == len(log_sigma)):
            raise ValueError("All arrays must have the same length")
        fc = LogNormalForecast(log_mu, log_sigma)

    elif distribution == 'normal':
        mu = dist_params.get('mu_pred')
        sigma = dist_params.get('sigma_pred')
        if mu is None or sigma is None:
            raise ValueError("Normal CRPS requires 'mu_pred' and 'sigma_pred'")
        mu = np.asarray(mu, dtype=float).ravel()
        sigma = np.asarray(sigma, dtype=float).ravel()
        if not (len(y_true) == len(mu) == len(sigma)):
            raise ValueError("All arrays must have the same length")
        fc = NormalForecast(mu, sigma)

    elif distribution == 'weibull':
        lam = dist_params.get('lambda_pred')
        k = dist_params.get('k_pred')
        if lam is None or k is None:
            raise ValueError("Weibull CRPS requires 'lambda_pred' and 'k_pred'")
        lam = np.asarray(lam, dtype=float).ravel()
        k = np.asarray(k, dtype=float).ravel()
        if not (len(y_true) == len(lam) == len(k)):
            raise ValueError("All arrays must have the same length")
        fc = WeibullForecast(lam, k)

    else:
        raise ValueError(
            f"Unknown distribution: '{distribution}'. "
            f"Must be 'weibull', 'normal', or 'lognormal'."
        )

    crps_scores = np.asarray(fc.crps(y_true), dtype=float)
    return {
        'crps': float(np.nanmean(crps_scores)),
        'crps_scores': crps_scores,
    }



def _climatology_crps(pool, y):
    """Per-observation CRPS of the (fixed) climatological distribution.

    Climatology is the marginal sample distribution `pool`, IDENTICAL for every
    window, so this is O(n log n). The naive approach of tiling `pool` into a
    (B, N) "ensemble" and calling EnsembleForecast.crps builds a (B, N, N) pairwise
    tensor (tens of GB) — that is what crashed the kernel. Don't do that.

    CRPS(F_clim, y) = E|C - y| - 0.5 E|C - C'|,  with C, C' ~ F_clim.
      - E|C - y| is computed per observation from the sorted pool (searchsorted).
      - E|C - C'| is the Gini mean difference of the pool: a single scalar.
    """
    x = np.sort(np.asarray(pool, dtype=float))
    n = x.size
    cs = np.cumsum(x)
    total = cs[-1]
    y = np.asarray(y, dtype=float)
    k = np.searchsorted(x, y, side="right")                    # # pool pts <= y, per obs
    S_le = np.where(k > 0, cs[np.clip(k - 1, 0, n - 1)], 0.0)   # sum of those k
    term1 = (y * k - S_le + (total - S_le) - y * (n - k)) / n   # E|C - y|, per obs
    i = np.arange(1, n + 1)
    mad = (2.0 / (n * n)) * np.sum((2 * i - n - 1) * x)         # E|C - C'| (scalar)
    return term1 - 0.5 * mad


def create_probabilistic_dashboard(results_path, family="auto",
                                    thresholds=(4.5, 6.5), save_path=None):
    """Distributional verification dashboard for the final forecasted distribution.

    Parameters
    ----------
    results_path : str
        Path to a saved results file (any model path: mlp, ensemble, set_mlp).
    family : str
        Which predictive representation to score ('auto' = headline forecast).
        See forecast_from_results.
    thresholds : tuple of float
        Hp30 thresholds for the exceedance-reliability panels (traffic-light levels).
    save_path : str or None
        If given, save the figure there.
    """
    results, config, _ = load_results(results_path)
    dataset = recreate_dataset_from_results(results_path)

    test_window_positions = list(config["test_indices"])
    targets = np.asarray(results["y_test"], dtype=float)

    # ONE forecast object over all test windows; metrics derive from it.
    forecast_all = forecast_from_results(results, family=family)

    # Climatological reference for CRPSS: the (sample) marginal distribution of the
    # target, applied identically to every window. Reuses EnsembleForecast as an
    # empirical distribution. NOTE: uses test targets as climatology (sample
    # climatology) — swap in training targets if you want a leakage-free reference.
    clim_pool = targets.copy()

    # ---- subsets -----------------------------------------------------------
    # Strength-based subsets are robust; event-type subsets are added only if the
    # dataset exposes the relevant filter method.
    subsets = {
        "All test data": {},
        "Storms (>4.5)": {"min_strength": 4.5},
        "Strong storms (>6.5)": {"min_strength": 6.5},
        "No storm (<4.5)": {"max_strength": 4.5},
    }
    if hasattr(dataset, "filter_indices_by_event_type"):
        subsets["All ICME"] = {"event_types": ["ICME"]}
        subsets["All SIR"] = {"event_types": ["SIR"]}

    pos_to_idx = {wp: i for i, wp in enumerate(test_window_positions)}

    def indices_for(filters):
        wp = np.array(test_window_positions)
        if any(k in filters for k in ("event_types", "exclude_quiet", "forecast_only")):
            wp = np.array(dataset.filter_indices_by_event_type(
                wp.tolist(),
                event_types=filters.get("event_types"),
                exclude_quiet=filters.get("exclude_quiet", False),
                forecast_only=filters.get("forecast_only", False),
            ))
        if "min_strength" in filters or "max_strength" in filters:
            wp = np.array(dataset.filter_indices_by_storm_strength(
                wp.tolist(),
                min_strength=filters.get("min_strength", 0.0),
                max_strength=filters.get("max_strength"),
            ))
        return np.array([pos_to_idx[w] for w in wp if w in pos_to_idx], dtype=int)

    # ---- per-subset metrics, all from the predictive distribution ----------
    def metrics_for(idx):
        if len(idx) < 5:
            return None
        f = forecast_all.subset(idx)
        y = targets[idx]

        model_crps = f.crps(y)                                   # (n,)
        clim_crps = _climatology_crps(clim_pool, y)              # (n,), O(n log n)
        crpss = 1.0 - np.mean(model_crps) / np.mean(clim_crps)   # vs climatology

        coverage, widths = {}, {}
        for p in [10, 25, 50, 75, 90, 95]:
            lo, hi = f.interval(p / 100.0)
            coverage[p] = np.mean((y >= lo) & (y <= hi)) * 100.0
            widths[p] = np.mean(hi - lo)

        pit = f.pit(y)
        ks = kstest(np.clip(pit, 0, 1), "uniform").statistic       # 0 = perfectly uniform

        reliability = {}
        for t in thresholds:
            pred = f.exceedance_prob(t)
            obs = (y > t).astype(float)
            bins = np.linspace(0, 1, 11)
            bp, bo = [], []
            for i in range(len(bins) - 1):
                m = (pred >= bins[i]) & (pred < bins[i + 1])
                if m.sum() > 0:
                    bp.append(pred[m].mean())
                    bo.append(obs[m].mean())
            reliability[t] = {"predicted": bp, "observed": bo}

        return {
            "n": len(idx), "crps": float(np.mean(model_crps)), "crpss": float(crpss),
            "sharpness": float(np.mean(f.std())), "coverage": coverage,
            "widths": widths, "pit": pit, "pit_ks": float(ks),
            "reliability": reliability,
        }

    prob = {}
    for name, filt in subsets.items():
        m = metrics_for(indices_for(filt))
        if m is not None:
            prob[name] = m
    names = list(prob.keys())
    all_m = prob["All test data"]

    # ---- figure ------------------------------------------------------------
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.30)
    fig.suptitle(
        f"Distributional Forecast Verification — {config.get('model_name', '')}\n"
        f"(Lead time: {config.get('lead_time', '?')}h, Fold: {config.get('test_fold', '?')}, "
        f"family={family})",
        fontsize=16, fontweight="bold",
    )

    # P1: CRPS by subset
    ax = fig.add_subplot(gs[0, 0])
    vals = [prob[s]["crps"] for s in names]
    ax.barh(range(len(names)), vals, color="steelblue", alpha=0.8, edgecolor="black")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("CRPS (lower is better)", fontweight="bold")
    ax.set_title("CRPS (closed-form)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3); ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)

    # P2: CRPSS vs climatology
    ax = fig.add_subplot(gs[0, 1])
    vals = [prob[s]["crpss"] for s in names]
    ax.barh(range(len(names)), vals,
            color=["green" if v > 0 else "red" for v in vals], alpha=0.8, edgecolor="black")
    ax.axvline(0, color="black", lw=1)
    ax.axvline(0.2, color="gray", ls="--", lw=1, label="0.2 (clear value)")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("CRPS Skill Score vs climatology", fontweight="bold")
    ax.set_title("Probabilistic Skill (CRPSS)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3); ax.invert_yaxis(); ax.legend(fontsize=8)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)

    # P3: Sharpness (predictive std)
    ax = fig.add_subplot(gs[0, 2])
    vals = [prob[s]["sharpness"] for s in names]
    ax.barh(range(len(names)), vals, color="orange", alpha=0.8, edgecolor="black")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean predictive std (lower = sharper)", fontweight="bold")
    ax.set_title("Sharpness", fontweight="bold")
    ax.grid(axis="x", alpha=0.3); ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)

    # P4: PIT histogram — all test data  (replaces rank histogram)
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(all_m["pit"], bins=10, range=(0, 1), density=True,
            color="steelblue", alpha=0.8, edgecolor="black")
    ax.axhline(1.0, color="red", ls="--", lw=2, label="Uniform (calibrated)")
    ax.set_xlabel("PIT value  F(y)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title(f"PIT Histogram — all test\n(KS={all_m['pit_ks']:.3f}; "
                 f"∪=under-dispersed, ∩=over-dispersed)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # P5: PIT histogram — storms (the tail is where a single LogNormal usually fails)
    ax = fig.add_subplot(gs[1, 1])
    if "Storms (>4.5)" in prob:
        sm = prob["Storms (>4.5)"]
        ax.hist(sm["pit"], bins=10, range=(0, 1), density=True,
                color="darkorange", alpha=0.8, edgecolor="black")
        ax.axhline(1.0, color="red", ls="--", lw=2)
        ax.set_title(f"PIT Histogram — storms >4.5\n(KS={sm['pit_ks']:.3f})",
                     fontsize=10, fontweight="bold")
    ax.set_xlabel("PIT value  F(y)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")

    # P6: Interval coverage diagram — all test
    ax = fig.add_subplot(gs[1, 2])
    exp = list(all_m["coverage"].keys())
    act = list(all_m["coverage"].values())
    ax.plot(exp, exp, "k--", alpha=0.5, label="Perfect")
    ax.plot(exp, act, "o-", color="steelblue", lw=2, ms=7, label="Actual")
    ax.set_xlabel("Nominal central coverage (%)", fontweight="bold")
    ax.set_ylabel("Empirical coverage (%)", fontweight="bold")
    ax.set_title("Prediction-Interval Coverage", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    # P7, P8: Exceedance reliability at the requested thresholds
    for j, t in enumerate(thresholds[:2]):
        ax = fig.add_subplot(gs[2, j])
        r = all_m["reliability"][t]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.plot(r["predicted"], r["observed"], "o-", color="darkred", lw=2, ms=7, label="Actual")
        ax.set_xlabel("Predicted P(Hp30 > %.1f)" % t, fontweight="bold")
        ax.set_ylabel("Observed frequency", fontweight="bold")
        ax.set_title(f"Reliability: P(Hp30 > {t})", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # P9: CRPS vs sharpness trade-off (sharpness subject to calibration)
    ax = fig.add_subplot(gs[2, 2])
    sx = [prob[s]["sharpness"] for s in names]
    cy = [prob[s]["crps"] for s in names]
    ax.scatter(sx, cy, s=130, alpha=0.8, edgecolor="black", c="steelblue")
    for i, s in enumerate(names):
        ax.annotate(s.split(" ")[0], (sx[i], cy[i]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Sharpness (predictive std)", fontweight="bold")
    ax.set_ylabel("CRPS", fontweight="bold")
    ax.set_title("Sharpness vs Accuracy", fontweight="bold")
    ax.grid(alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    # console summary table
    print(f"\n{'Subset':<22}{'n':>6}{'CRPS':>9}{'CRPSS':>9}{'Sharp':>9}{'Cov90':>8}{'PIT-KS':>8}")
    print("-" * 71)
    for s in names:
        m = prob[s]
        print(f"{s:<22}{m['n']:>6}{m['crps']:>9.3f}{m['crpss']:>9.3f}"
              f"{m['sharpness']:>9.3f}{m['coverage'][90]:>8.1f}{m['pit_ks']:>8.3f}")

    return fig, prob


def _run_label(file, config):
    """The identifier that actually distinguishes runs."""
    return config.get('run_name') or Path(file).stem


def _dataset_key(cfg, dsp):
    """Config tuple that determines window_labels/max_targets/valid_indices."""
    return (
        str(dsp['huxt_data_path']),
        str(dsp.get('discontinuity_path')),
        cfg['lead_time'],
        dsp['forecast_duration_hours'],
        dsp['stride_hours'],
        bool(cfg.get('remove_cmes', False)),
        bool(cfg.get('balance', False)),
    )


def _run_label_from_stem(stem):
    """Recover the human arm label from the filename (strip seed/lt/fold suffix)."""
    s = re.sub(r'_seed\d+_lt\d+_fold\d+$', '', stem)
    return s.replace('results_', '', 1)


def fast_batch_metrics(results_dir, threshold=4.5, point_key='y_pred_lognormal_median',
                       verify_alignment=True):
    files = sorted(Path(results_dir).glob('*.pkl'))
    if not files:
        raise FileNotFoundError(f"No .pkl files in {results_dir}")

    ds_cache = {}          # key -> dict(labels=, targets=, valid_indices=)
    rows = []

    for f in files:
        with open(f, 'rb') as fh:
            pkg = pickle.load(fh)
        r, cfg, dsp = pkg['results'], pkg['config'], pkg['dataset_params']
        key = _dataset_key(cfg, dsp)

        # build ONCE per unique dataset-determining config
        if key not in ds_cache:
            print(f"Building dataset for key: lead={cfg['lead_time']}, "
                  f"cmes={cfg.get('remove_cmes')}, balance={cfg.get('balance')} ...")
            ds = recreate_dataset_from_results(f)
            ds_cache[key] = {
                'labels':  np.asarray(ds.window_labels),
                'targets': np.asarray(ds.max_targets),
                'valid_indices': list(ds.valid_indices),
            }
            del ds  # free the 600-column DataFrame immediately

        cache = ds_cache[key]
        test_idx = np.asarray(cfg['test_indices'])
        labels_te  = cache['labels'][test_idx]
        targets_te = cache['targets'][test_idx]

        y = np.asarray(r['y_test']).ravel()

        # ALIGNMENT GUARD: sliced targets must equal the stored y_test
        if verify_alignment and not np.allclose(targets_te.astype(float), y, atol=1e-6):
            raise AssertionError(
                f"Alignment mismatch in {f.name}: cached targets[test_idx] != y_test. "
                f"test_indices are not positions into this dataset's valid_indices "
                f"(check lead/cme/balance match)."
            )

        fc = forecast_from_results(r, family='auto')
        crps = np.asarray(fc.crps(y)).ravel()
        pit  = np.asarray(fc.pit(y)).ravel()
        yhat = np.asarray(r[point_key]).ravel()

        subsets = {
            'all':   np.ones_like(y, dtype=bool),
            'storm': y > threshold,
            'quiet': y <= threshold,
            'ICME':  np.isin(labels_te, ['ICME_input', 'ICME_forecast']),
            'SIR':   np.isin(labels_te, ['SIR_input', 'SIR_forecast']),
        }
        run_label = _run_label_from_stem(f.stem)
        for name, mask in subsets.items():
            if mask.sum() == 0:
                continue
            pit_ks = float(kstest(pit[mask], 'uniform').statistic) if mask.sum() > 5 else np.nan
            rows.append({
                'run': run_label, 'fold': cfg['test_fold'], 'seed': cfg['random_seed'],
                'lead': cfg['lead_time'], 'subset': name, 'n': int(mask.sum()),
                'crps': float(np.mean(crps[mask])),
                'pit_ks': pit_ks,
                'mean_pred':   float(np.mean(yhat[mask])),
                'mean_target': float(np.mean(y[mask])),
                'bias':        float(np.mean(yhat[mask] - y[mask])),
                'rmse': float(np.sqrt(np.mean((yhat[mask] - y[mask])**2))),
                'mae':  float(np.mean(np.abs(yhat[mask] - y[mask]))),
            })

    print(f"Built {len(ds_cache)} dataset(s) for {len(files)} files.")
    return pd.DataFrame(rows)


def aggregate_folds(df):
    """Mean +/- std across folds, per run x subset (x lead if multiple)."""
    grp = ['run', 'subset'] + (['lead'] if df['lead'].nunique() > 1 else [])
    return (df.groupby(grp)
              .agg(crps_mean=('crps','mean'), crps_std=('crps','std'),
                 pit_ks_mean=('pit_ks','mean'), pit_ks_std=('pit_ks','std'),
                 bias_mean=('bias','mean'), bias_std=('bias','std'),
                 mean_pred_mean=('mean_pred','mean'),
                 n_folds=('fold','nunique'))
              .reset_index())


def auto_detect_and_compare(
    results_dir: Path,
    seed: int = 42,
    fold: int = 0,
    threshold: float = 4.5,
    test_mode: str = 'balanced',
    include_baselines: bool = True,
    subsets: Optional[Dict] = None,
    save: bool = False,
    save_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Detect every result FILE (not deduped model name) and compare them."""
    results_files = sorted(results_dir.glob('*.pkl'))
    if not results_files:
        print(f"No results files found in {results_dir}")
        return pd.DataFrame()
    print(f"Found {len(results_files)} result files")

    # One record per FILE, labelled by run_name/stem (the thing that varies).
    runs = []
    lead_times = set()
    for file in results_files:
        try:
            _, config, _ = load_results(file)
            label = _run_label(file, config)
            runs.append({'file': file, 'label': label, 'lead_time': config['lead_time']})
            lead_times.add(config['lead_time'])
        except Exception as e:
            print(f"  Could not load {file.name}: {e}")
            continue

    lead_times = sorted(lead_times)
    print("Detected runs:")
    for r in runs:
        print(f"  {r['label']:<40} (lead {r['lead_time']}h)  <- {r['file'].name}")
    print(f"Detected lead times: {lead_times}")

    return compare_models_and_lead_times(
        results_dir=results_dir,
        runs=runs,                       # pass explicit per-file run records
        lead_times=lead_times,
        seed=seed, fold=fold, threshold=threshold, test_mode=test_mode,
        subsets=subsets, include_baselines=include_baselines,
        save=save, save_dir=save_dir,
    )


def compare_models_and_lead_times(
    results_dir: Path,
    runs: List[Dict],                    # [{'file','label','lead_time'}, ...] — one per file
    lead_times: List[int],
    seed: int = 42,
    fold: int = 0,
    threshold: float = 4.5,
    test_mode: str = 'balanced',
    prediction_type: str = 'y_pred_lognormal_median',
    include_baselines: bool = True,
    subsets: Optional[Dict] = None,
    save: bool = False,
    save_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Compare across runs (one per file). Recreates dataset per file."""
    if subsets is None:
        subsets = {
            'All test data': {},
            'All ICME': {'event_types': ['ICME']},
            'All SIR': {'event_types': ['SIR']},
            'Strong storms (>6.5)': {'min_strength': 6.5},
            'Storms (>4.5)': {'min_strength': 4.5},
            'No storm (≤4.5)': {'min_strength': 0.0, 'max_strength': 4.5},
        }

    all_results = []
    baselines_added = set()

    for run in runs:
        results_file = run['file']
        label = run['label']                     # <- distinct per file
        try:
            print(f"\nProcessing: {label}  ({results_file.name})")
            results, config, _ = load_results(results_file)
            lead_time = config['lead_time']

            dataset = recreate_dataset_from_results(results_file)
            test_window_positions = config['test_indices']
            print(f"  Dataset size: {len(dataset)}, test samples: {len(test_window_positions)}")

            window_pos_to_test_idx = {wp: i for i, wp in enumerate(test_window_positions)}
            subset_filtered_positions = {}

            for subset_name, filters in subsets.items():
                filtered_wps = np.array(test_window_positions).copy()
                event_types   = filters.get('event_types')
                exclude_quiet = filters.get('exclude_quiet', False)
                forecast_only = filters.get('forecast_only', False)
                if event_types is not None or exclude_quiet or forecast_only:
                    filtered_wps = dataset.filter_indices_by_event_type(
                        filtered_wps.tolist(), event_types=event_types,
                        exclude_quiet=exclude_quiet, forecast_only=forecast_only)
                min_strength = filters.get('min_strength')
                max_strength = filters.get('max_strength')
                if min_strength is not None or max_strength is not None:
                    filtered_wps = dataset.filter_indices_by_storm_strength(
                        filtered_wps.tolist(),
                        min_strength=min_strength if min_strength is not None else 0.0,
                        max_strength=max_strength)
                filtered_test_indices = [
                    window_pos_to_test_idx[wp] for wp in filtered_wps
                    if wp in window_pos_to_test_idx]
                subset_filtered_positions[subset_name] = {
                    'window_positions': filtered_wps,
                    'test_indices': filtered_test_indices,
                }

            predictions = results[prediction_type]
            targets     = results['y_test']
            has_crps    = 'log_mu' in results and 'log_sigma' in results

            for subset_name, indices_dict in subset_filtered_positions.items():
                filtered_test_indices     = indices_dict['test_indices']
                filtered_window_positions = indices_dict['window_positions']
                if len(filtered_test_indices) == 0:
                    continue
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                filtered_preds   = predictions[filtered_test_indices]
                filtered_targets = targets[filtered_test_indices]
                result_dict = {
                    'model':           label,            # <- per-file label, not model_name
                    'lead_time':       lead_time,
                    'subset':          subset_name,
                    'n_samples':       len(filtered_window_positions),
                    'rmse':            np.sqrt(mean_squared_error(filtered_targets, filtered_preds)),
                    'mae':             mean_absolute_error(filtered_targets, filtered_preds),
                    'r2':              r2_score(filtered_targets, filtered_preds),
                    'mean_target':     float(np.mean(filtered_targets)),
                    'mean_prediction': float(np.mean(filtered_preds)),
                }
                if has_crps:
                    prob_metrics = evaluate_distribution_forecast(
                        filtered_targets, distribution='lognormal',
                        log_mu_pred=results['log_mu'][filtered_test_indices],
                        log_sigma_pred=results['log_sigma'][filtered_test_indices])
                    result_dict['crps'] = prob_metrics['crps']
                else:
                    result_dict['crps'] = np.nan
                all_results.append(result_dict)

            print(f"  {label}: done")

            # Baselines once per lead time (unchanged logic)
            if include_baselines and lead_time not in baselines_added:
                baseline_types = {
                    'Persistence':       'y_pred_persistence',
                    '27-Day Recurrence': 'y_pred_27_day_recurrence',
                    'Persistence Max':   'y_pred_persistence_max',
                }
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                for baseline_name, baseline_key in baseline_types.items():
                    if baseline_key not in results:
                        continue
                    baseline_preds = results[baseline_key]
                    for subset_name, indices_dict in subset_filtered_positions.items():
                        fti = indices_dict['test_indices']
                        fwp = indices_dict['window_positions']
                        if len(fti) == 0:
                            continue
                        fp = baseline_preds[fti]; ft = targets[fti]
                        all_results.append({
                            'model': baseline_name, 'lead_time': lead_time,
                            'subset': subset_name, 'n_samples': len(fwp),
                            'rmse': np.sqrt(mean_squared_error(ft, fp)),
                            'mae': mean_absolute_error(ft, fp),
                            'r2': r2_score(ft, fp),
                            'mean_target': float(np.mean(ft)),
                            'mean_prediction': float(np.mean(fp)),
                            'crps': np.nan,
                        })
                baselines_added.add(lead_time)
                print(f"  Baselines: done")

        except Exception as e:
            print(f"  {label}: ERROR — {e}")
            import traceback
            traceback.print_exc()
            continue

    df = pd.DataFrame(all_results)
    if len(df) == 0:
        print("\nWarning: No results were loaded!")
        return df

    print(f"\n{'='*80}\nAnalysis complete! {len(df)} run-subset combinations\n{'='*80}\n")
    all_models = df['model'].unique().tolist()
    create_comprehensive_visualization(df, all_models, lead_times, subsets, save, save_dir)
    if save and save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_dir / 'comprehensive_results.csv', index=False)
    return df


def create_comprehensive_visualization(
    df: pd.DataFrame,
    models: List[str],
    lead_times: List[int],
    subsets: Dict,
    save: bool = False,
    save_dir: Optional[Path] = None, 
):
    """Create comprehensive visualizations of results."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    def get_distinctive_names(names):
        if len(names) <= 1:
            return {name: name for name in names}
        result = {name: f'Model {i+1}' for i, name in enumerate(names)}
        return result

    BASELINE_MODELS = {'Persistence', 'Persistence Max', '27-Day Recurrence'}

    ml_models = [m for m in models if m not in BASELINE_MODELS]
    distinctive_map = get_distinctive_names(ml_models)
    
    # Add baselines back into the map unchanged
    for m in models:
        if m in BASELINE_MODELS:
            distinctive_map[m] = m
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Comprehensive Model Comparison Across Lead Times\n(Using LogNormal Median Predictions)', 
                 fontsize=18, fontweight='bold')
    
    # ===== PANEL 1: RMSE Heatmap =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    all_data = df[df['subset'] == 'All test data']
    pivot_rmse = all_data.pivot(index='model', columns='lead_time', values='rmse')
    
    # Sort by mean RMSE
    pivot_rmse['mean_rmse'] = pivot_rmse.mean(axis=1)
    pivot_rmse = pivot_rmse.sort_values('mean_rmse', ascending=True)
    pivot_rmse = pivot_rmse.drop('mean_rmse', axis=1)
    
    im1 = ax1.imshow(pivot_rmse.values, cmap='viridis_r', aspect='auto')
    ax1.set_xticks(range(len(lead_times)))
    ax1.set_xticklabels(lead_times)
    ax1.set_yticks(range(len(pivot_rmse.index)))
    
    # Format model names for display
    formatted_names = [distinctive_map[m] for m in pivot_rmse.index]
    ax1.set_yticklabels(formatted_names, fontsize=9)
    
    ax1.set_xlabel('Lead Time (hours)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax1.set_title('RMSE Heatmap (Lower is Better)', fontsize=13, fontweight='bold')
    
    for i in range(len(pivot_rmse.index)):
        for j in range(len(lead_times)):
            color = 'white' if pivot_rmse.values[i, j] > pivot_rmse.values.mean() else 'black'
            ax1.text(j, i, f'{pivot_rmse.values[i, j]:.2f}',
                    ha="center", va="center", color=color, fontsize=9)
    
    plt.colorbar(im1, ax=ax1, label='RMSE')
    
    # ===== PANEL 2: CRPS Heatmap (CHANGED FROM MAE) =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Check if CRPS data exists
    if 'crps' in all_data.columns:
        pivot_crps = all_data.pivot(index='model', columns='lead_time', values='crps')
        pivot_crps = pivot_crps.reindex(pivot_rmse.index)
        
        im2 = ax2.imshow(pivot_crps.values, cmap='viridis_r', aspect='auto')
        ax2.set_xticks(range(len(lead_times)))
        ax2.set_xticklabels(lead_times)
        ax2.set_yticks(range(len(pivot_crps.index)))
        ax2.set_yticklabels(formatted_names, fontsize=9)
        ax2.set_xlabel('Lead Time (hours)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax2.set_title('CRPS Heatmap (Lower is Better)', fontsize=13, fontweight='bold')
        
        for i in range(len(pivot_crps.index)):
            for j in range(len(lead_times)):
                if not np.isnan(pivot_crps.values[i, j]):
                    color = 'white' if pivot_crps.values[i, j] > np.nanmean(pivot_crps.values) else 'black'
                    ax2.text(j, i, f'{pivot_crps.values[i, j]:.2f}',
                            ha="center", va="center", color=color, fontsize=9)
        
        plt.colorbar(im2, ax=ax2, label='CRPS')
    else:
        # Fallback to MAE if CRPS not available
        pivot_mae = all_data.pivot(index='model', columns='lead_time', values='mae')
        pivot_mae = pivot_mae.reindex(pivot_rmse.index)
        
        im2 = ax2.imshow(pivot_mae.values, cmap='viridis_r', aspect='auto')
        ax2.set_xticks(range(len(lead_times)))
        ax2.set_xticklabels(lead_times)
        ax2.set_yticks(range(len(pivot_mae.index)))
        ax2.set_yticklabels(formatted_names, fontsize=9)
        ax2.set_xlabel('Lead Time (hours)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax2.set_title('MAE Heatmap (Lower is Better)', fontsize=13, fontweight='bold')
        
        for i in range(len(pivot_mae.index)):
            for j in range(len(lead_times)):
                color = 'white' if pivot_mae.values[i, j] > pivot_mae.values.mean() else 'black'
                ax2.text(j, i, f'{pivot_mae.values[i, j]:.2f}',
                        ha="center", va="center", color=color, fontsize=9)
        
        plt.colorbar(im2, ax=ax2, label='MAE')
    
    # ===== PANEL 3: Prediction Bias Heatmap =====
    ax3 = fig.add_subplot(gs[0, 2])
    
    bias_data = all_data.copy()
    bias_data['bias'] = bias_data['mean_prediction'] - bias_data['mean_target']
    pivot_bias = bias_data.pivot(index='model', columns='lead_time', values='bias')
    pivot_bias = pivot_bias.reindex(pivot_rmse.index)
    
    vmax = max(abs(pivot_bias.values.min()), abs(pivot_bias.values.max()))
    im3 = ax3.imshow(pivot_bias.values, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax3.set_xticks(range(len(lead_times)))
    ax3.set_xticklabels(lead_times)
    ax3.set_yticks(range(len(pivot_bias.index)))
    ax3.set_yticklabels(formatted_names, fontsize=9)
    ax3.set_xlabel('Lead Time (hours)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax3.set_title('Prediction Bias Heatmap (0 is Best)', fontsize=13, fontweight='bold')
    
    for i in range(len(pivot_bias.index)):
        for j in range(len(lead_times)):
            color = 'white' if abs(pivot_bias.values[i, j]) > vmax * 0.5 else 'black'
            ax3.text(j, i, f'{pivot_bias.values[i, j]:.2f}',
                    ha="center", va="center", color=color, fontsize=9)
    
    plt.colorbar(im3, ax=ax3, label='Bias (Pred - Target)')
    
    # ===== PANEL 4: RMSE by Storm Strength =====
    ax4 = fig.add_subplot(gs[1, 0])
    
    mid_lead_time = lead_times[len(lead_times)//2]
    storm_subsets = ['No storm (≤4.5)', 'Storms (>4.5)', 'Strong storms (>6.5)']
    
    x = np.arange(len(storm_subsets))
    display_models = list(pivot_rmse.index[:10])
    width = 0.8 / min(len(display_models), 10)
    
    for i, model in enumerate(display_models):
        rmse_values = []
        for subset in storm_subsets:
            subset_data = df[(df['model'] == model) & 
                            (df['lead_time'] == mid_lead_time) & 
                            (df['subset'] == subset)]
            if len(subset_data) > 0:
                rmse_values.append(subset_data['rmse'].values[0])
            else:
                rmse_values.append(0)

        formatted_name = distinctive_map[model]
        ax4.bar(x + i*width, rmse_values, width, label=formatted_name, alpha=0.8)
    
    ax4.set_xlabel('Storm Category', fontsize=12, fontweight='bold')
    ax4.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax4.set_title(f'RMSE by Storm Strength (Lead time: {mid_lead_time}h)', 
                 fontsize=13, fontweight='bold')
    ax4.set_xticks(x + width * (len(display_models)-1)/2)
    ax4.set_xticklabels(['No storm', 'Storm', 'Strong'], rotation=0)
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(axis='y', alpha=0.3)
    
    # ===== PANEL 5: Performance by Event Type =====
    ax5 = fig.add_subplot(gs[1, 1])
    
    event_subsets = ['All ICME', 'All SIR', 'Strong storms (>6.5)']
    
    x = np.arange(len(display_models))
    width = 0.25
    
    for i, subset in enumerate(event_subsets):
        subset_data = df[(df['lead_time'] == mid_lead_time) & (df['subset'] == subset)]
        rmse_values = [subset_data[subset_data['model'] == m]['rmse'].values[0] 
                      if len(subset_data[subset_data['model'] == m]) > 0 else 0 
                      for m in display_models]
        ax5.bar(x + i*width, rmse_values, width, label=subset, alpha=0.8)
    
    ax5.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax5.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax5.set_title(f'RMSE by Event Type (Lead time: {mid_lead_time}h)', 
                 fontsize=13, fontweight='bold')
    ax5.set_xticks(x + width)
    formatted_display = [distinctive_map[m] for m in display_models]
    ax5.set_xticklabels(formatted_display, rotation=45, ha='right', fontsize=8)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # ===== PANEL 6: Best model histogram =====
    ax6 = fig.add_subplot(gs[1, 2])

    # For each test sample, find which model's median was closest to y_true
    all_test_data  = df[df['subset'] == 'All test data']
    best_counts    = {m: 0 for m in pivot_rmse.index}

    # We need per-sample predictions — use mean_prediction as a proxy
    # since we only have aggregated metrics in df. Instead compute from
    # the bias: model whose mean_prediction is closest to mean_target wins.
    # For a proper per-sample count, iterate over lead times.
    for lt in lead_times:
        lt_data = all_test_data[all_test_data['lead_time'] == lt]
        if lt_data.empty:
            continue

        # Rank models by absolute bias for this lead time
        lt_data = lt_data.copy()
        lt_data['abs_bias'] = abs(lt_data['mean_prediction'] - lt_data['mean_target'])
        best_model = lt_data.loc[lt_data['abs_bias'].idxmin(), 'model']
        if best_model in best_counts:
            best_counts[best_model] += 1

    models_sorted  = sorted(best_counts.keys(), key=lambda m: best_counts[m], reverse=True)
    counts         = [best_counts[m] for m in models_sorted]
    labels_sorted  = [distinctive_map[m] for m in models_sorted]

    bars = ax6.bar(range(len(models_sorted)), counts, color='steelblue',
                   alpha=0.7, edgecolor='black')
    ax6.set_xticks(range(len(models_sorted)))
    ax6.set_xticklabels(labels_sorted, rotation=45, ha='right', fontsize=8)
    ax6.set_ylabel('Times closest to true value', fontsize=11, fontweight='bold')
    ax6.set_title('Best Model Count\n(closest mean prediction to mean target)',
                  fontsize=12, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, counts):
        if val > 0:
            ax6.text(bar.get_x() + bar.get_width() / 2, val + 0.05, str(val),
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ===== PANEL 7: Sample Sizes =====
    ax7 = fig.add_subplot(gs[2, 0])
    
    subset_data = df[(df['lead_time'] == mid_lead_time) & (df['model'] == pivot_rmse.index[0])]
    subset_names_short = [s.split(' (')[0] for s in subset_data['subset']]
    sample_sizes = subset_data['n_samples'].values
    
    ax7.barh(range(len(subset_names_short)), sample_sizes, 
            color='steelblue', alpha=0.7, edgecolor='black')
    ax7.set_yticks(range(len(subset_names_short)))
    ax7.set_yticklabels(subset_names_short, fontsize=9)
    ax7.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
    ax7.set_title(f'Sample Sizes (Lead time: {mid_lead_time}h)', 
                 fontsize=13, fontweight='bold')
    ax7.invert_yaxis()
    ax7.grid(axis='x', alpha=0.3)
    
    # ===== PANELS 8-9: Summary Table (spans 2 columns) =====
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('tight')
    ax8.axis('off')
    
    summary_models = list(pivot_rmse.index[:6])
    table_data = []
    
    for model in summary_models:
        model_data = all_data[all_data['model'] == model]
        avg_bias = (model_data['mean_prediction'] - model_data['mean_target']).mean()
        formatted_name = distinctive_map[model]
        
        # Get CRPS if available
        if 'crps' in model_data.columns and not model_data['crps'].isna().all():
            mean_crps = model_data['crps'].mean()
            crps_str = f"{mean_crps:.3f}"
        else:
            crps_str = "N/A"
        
        table_data.append([
            formatted_name,
            f"{model_data['rmse'].mean():.3f}",
            f"{model_data['mae'].mean():.3f}",
            crps_str,
            f"{avg_bias:.3f}",
        ])
    
    table = ax8.table(
        cellText=table_data,
        colLabels=['Model', 'Mean RMSE', 'Mean MAE', 'Mean CRPS', 'Avg Bias'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    ax8.set_title('Top 6 Models Summary', fontsize=13, fontweight='bold', pad=20)

    # ── Model name legend at bottom of figure ────────────────────────────────
    legend_lines = [f"Model {i+1}: {name}" for i, name in enumerate(ml_models)]
    legend_text  = '\n'.join(legend_lines)
    
    fig.text(
        0.1, 0.,
        legend_text,
        fontsize=14,
        verticalalignment='top',
        horizontalalignment='left',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.4)
    )
    
    plt.tight_layout()
    
    if save and save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / 'comprehensive_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_dir / 'comprehensive_comparison.png'}")
    
    plt.show()


def analyze_results(
    results_path: Path,
    n: Optional[int] = None,
    sort_by: str = 'y_test',
    descending: bool = True,
    event_types: Optional[List[str]] = None,
    min_strength: Optional[float] = None,
    max_strength: Optional[float] = None,
    exclude_quiet: bool = False,
    forecast_only: bool = False,
    plot_function: Optional[Callable] = None,
    verify_labels: bool = False,
) -> Dict:
    """Analyze results with optional filtering and case studies."""
    
    # Load results and recreate dataset
    results, config, _ = load_results(results_path)
    dataset = recreate_dataset_from_results(results_path)
    
    # Get test window positions from config
    test_window_positions = config['test_indices']  # [0, 1, 2, 3, ...]
    print(results.keys())
    predictions = results['y_pred_lognormal_median']
    targets = results['y_test']
    
    print(f"\n{'='*80}")
    if n is not None:
        print(f"Analyzing top {n} cases sorted by {sort_by}")
    else:
        print(f"Analyzing filtered results")
    print(f"{'='*80}")
    print(f"Config: seed={config['random_seed']}, lead_time={config['lead_time']}h, "
          f"fold={config['test_fold']}")
    print(f"Model: {config['model_name']}")
    print(f"Original test samples: {len(test_window_positions)}")
    
    # Start with all test window positions (NOT center indices!)
    filtered_window_positions = np.array(test_window_positions).copy()
    
    # Apply event type filter (filter functions expect window positions)
    if event_types is not None or exclude_quiet or forecast_only:
        filtered_window_positions = dataset.filter_indices_by_event_type(
            filtered_window_positions.tolist(),
            event_types=event_types,
            exclude_quiet=exclude_quiet,
            forecast_only=forecast_only
        )
        print(f"After event filter: {len(filtered_window_positions)} samples")
    
    # Apply storm strength filter (also expects window positions)
    if min_strength is not None:
        filtered_window_positions = dataset.filter_indices_by_storm_strength(
            filtered_window_positions.tolist(),
            min_strength=min_strength,
            max_strength=max_strength
        )
        print(f"After strength filter: {len(filtered_window_positions)} samples")
    
    # Map filtered window positions to test array positions
    window_pos_to_test_idx = {wp: i for i, wp in enumerate(test_window_positions)}
    filtered_test_indices = [window_pos_to_test_idx[wp] 
                            for wp in filtered_window_positions 
                            if wp in window_pos_to_test_idx]
    
    # Extract filtered predictions and targets
    filtered_preds = predictions[filtered_test_indices]
    filtered_targets = targets[filtered_test_indices]
    
    # Compute metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    if len(filtered_preds) > 0:
        metrics = {
            'n_samples': len(filtered_window_positions),
            'mse': mean_squared_error(filtered_targets, filtered_preds),
            'rmse': np.sqrt(mean_squared_error(filtered_targets, filtered_preds)),
            'mae': mean_absolute_error(filtered_targets, filtered_preds),
            'r2': r2_score(filtered_targets, filtered_preds),
            'mean_target': float(np.mean(filtered_targets)),
            'mean_prediction': float(np.mean(filtered_preds)),
            'std_target': float(np.std(filtered_targets)),
            'std_prediction': float(np.std(filtered_preds))
        }
        
        print(f"\n{'-'*80}")
        print(f"Filtered Metrics:")
        print(f"  RMSE: {metrics['rmse']:.3f}")
        print(f"  MAE: {metrics['mae']:.3f}")
        print(f"  R²: {metrics['r2']:.3f}")
        print(f"  Mean target: {metrics['mean_target']:.3f}")
        print(f"  Mean prediction: {metrics['mean_prediction']:.3f}")
        print(f"{'-'*80}\n")
    else:
        print("No samples matched the filter criteria!")
        metrics = None
    
    # Analyze top N cases if requested
    cases = []
    if n is not None and len(filtered_preds) > 0:
        # Sort within filtered results
        if sort_by == 'y_test':
            sort_values = filtered_targets
        else:
            sort_values = results[sort_by][filtered_test_indices]
        
        if descending:
            top_n_positions = np.argsort(sort_values)[-min(n, len(sort_values)):][::-1]
        else:
            top_n_positions = np.argsort(sort_values)[:min(n, len(sort_values))]
        
        # Analyze each case
        for i, pos_in_filtered in enumerate(top_n_positions):
            # Get indices
            test_idx = filtered_test_indices[pos_in_filtered]
            window_position = filtered_window_positions[pos_in_filtered]
            center_idx = results['window_idx'][test_idx]
            
            if verify_labels and 'ICME_flag' in dataset.df.columns:
                input_start_idx = center_idx + dataset.min_offset
                input_end_idx = center_idx + dataset.lead_time  # Input window end
                forecast_start_idx = center_idx + dataset.lead_time
                forecast_end_idx = center_idx + dataset.max_offset + 1
                
                # Check input window
                input_df = dataset.df.iloc[input_start_idx:input_end_idx]
                icme_in_input = input_df['ICME_flag'].any()
                sir_in_input = input_df['SIR_flag'].any()
                
                # Check forecast window
                forecast_df = dataset.df.iloc[forecast_start_idx:forecast_end_idx]
                icme_in_forecast = forecast_df['ICME_flag'].any()
                sir_in_forecast = forecast_df['SIR_flag'].any()
                
                # Determine recalculated label
                if icme_in_input and not icme_in_forecast:
                    recalc_label = 'ICME_input'
                elif icme_in_forecast and not icme_in_input:
                    recalc_label = 'ICME_forecast'
                elif icme_in_input and icme_in_forecast:
                    recalc_label = 'ICME_input'  # Or 'Both' if you have that category
                elif sir_in_input and not sir_in_forecast:
                    recalc_label = 'SIR_input'
                elif sir_in_forecast and not sir_in_input:
                    recalc_label = 'SIR_forecast'
                elif sir_in_input and sir_in_forecast:
                    recalc_label = 'SIR_input'
                else:
                    recalc_label = 'quiet'
                
                # Get stored label
                stored_label = dataset.window_labels[window_position]
                
                print(f"\n{'='*60}")
                print(f"LABEL VERIFICATION FOR CASE #{i+1}")
                print(f"{'='*60}")
                print(f"Window position: {window_position}")
                print(f"Center index: {center_idx}")
                print(f"Center timestamp: {dataset.df.index[center_idx]}")
                print(f"\nInput window: {dataset.df.index[input_start_idx]} to {dataset.df.index[input_end_idx-1]}")
                print(f"  ICME present: {icme_in_input}")
                print(f"  SIR present: {sir_in_input}")
                print(f"\nForecast window: {dataset.df.index[forecast_start_idx]} to {dataset.df.index[forecast_end_idx-1]}")
                print(f"  ICME present: {icme_in_forecast}")
                print(f"  SIR present: {sir_in_forecast}")
                print(f"\nStored label: {stored_label}")
                print(f"Recalculated label: {recalc_label}")
                print(f"Match: {stored_label == recalc_label}")
                print(f"{'='*60}\n")
            
            # Extract values
            case_info = {
                'rank': i + 1,
                'window_position': int(window_position),
                'center_idx': int(center_idx),
                'test_idx': int(test_idx),
                'y_true': float(results['y_test'][test_idx]),
                'y_pred_persistence': float(results['y_pred_persistence'][test_idx]),
                'y_pred_lognormal_median': float(results['y_pred_lognormal_median'][test_idx]),
            }

            try: 
                case_info['y_pred_weibull_median'] = float(results['y_pred_weibull_median'][test_idx])
                case_info['y_pred_normal_median'] = float(results['y_pred_normal_median'][test_idx])
                case_info['y_pred_weighted_mean'] = float(results['y_pred_weighted_mean'][test_idx])
                case_info['lambda'] = float(results['lambda'][test_idx])
                case_info['k'] = float(results['k'][test_idx])
                case_info['mu'] = float(results['mu'][test_idx])
                case_info['sigma'] = float(results['sigma'][test_idx])
            except: 
                print('Case info extracted')
            
            # Add recalculated label to case_info if verified
            if verify_labels and 'ICME_flag' in dataset.df.columns:
                case_info['stored_label'] = stored_label
                case_info['recalculated_label'] = recalc_label
                case_info['label_match'] = stored_label == recalc_label
            
            cases.append(case_info)
            
            if plot_function is not None:
                try:
                    plot_function(dataset, results, window_position, test_idx, config=config)
                except Exception as e:
                    logger.error(f"Error plotting case {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    print(f"\n{'='*80}\n")
    
    return {
        'metrics': metrics,
        'cases': cases,
        'filtered_window_positions': filtered_window_positions,
        'filtered_test_indices': filtered_test_indices,
        'predictions': filtered_preds,
        'targets': filtered_targets,
        'config': config,
        'results': results,
        'dataset': dataset,
        'filter_params': {
            'event_types': event_types,
            'min_strength': min_strength,
            'max_strength': max_strength,
            'exclude_quiet': exclude_quiet,
            'forecast_only': forecast_only
        }
    }
    

def auto_detect_and_plot_case_studies(
    results_dir: Path,
    seed: int = 42,
    fold: int = 0,
    threshold: float = 4.5,
    balance_mode: str = 'unbalanced',
    n_cases: int = 5,
    sort_by: str = 'y_test',
    descending: bool = True,
    event_types: Optional[List[str]] = None,
    min_strength: Optional[float] = None,
    max_strength: Optional[float] = None,
    exclude_quiet: bool = False,
    plot_function: Optional[Callable] = None,
    case_indices: Optional[List[int]] = None,
):
    pattern      = "*.pkl"
    result_files = sorted(results_dir.glob(pattern))

    if not result_files:
        print(f"No results files found in {results_dir}")
        return {}

    print(f"\n{'='*80}")
    print(f"Auto-detected {len(result_files)} results file(s):")
    for f in result_files:
        print(f"  {f.name}")
    print(f"{'='*80}\n")

    # ── Select case studies from anchor file ─────────────────────────────────
    anchor_file = result_files[0]
    print(f"Selecting cases from anchor file: {anchor_file.name}")

    anchor_output = analyze_results(
        results_path=anchor_file,
        n=n_cases if case_indices is None else None,
        sort_by=sort_by,
        descending=descending,
        event_types=event_types,
        min_strength=min_strength,
        max_strength=max_strength,
        exclude_quiet=exclude_quiet,
        forecast_only=False,
        plot_function=None,
        verify_labels=False,
    )

    if case_indices is not None:
        selected_window_positions = case_indices
    else:
        selected_window_positions = [c['window_position'] for c in anchor_output['cases']]

    print(f"Selected window positions: {selected_window_positions}")

    # ── Load all results ──────────────────────────────────────────────────────
    all_results    = {}
    all_configs    = {}
    shared_dataset = anchor_output['dataset']
    model_name_map = {}  # original_name -> Model{i}

    for i, results_file in enumerate(result_files, start=1):
        print(results_file.name)
        results, config, _ = load_results(results_file)
        original_name = results_file.name
        short_name    = f'Model{i}'

        all_results[short_name] = results
        all_configs[short_name] = config
        model_name_map[short_name] = original_name

    # ── Print model name lookup table ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print("Model name reference:")
    for short, original in model_name_map.items():
        print(f"  {short}: {original}")
    print(f"{'='*80}\n")

    # ── Plot one comparative figure per selected case ─────────────────────────
    for rank, window_position in enumerate(selected_window_positions):
        window_position = int(window_position)
        print(f"\n{'='*80}")
        print(f"Comparative case study #{rank + 1} | window_position={window_position}")
        print(f"{'='*80}")

        test_idx_map = {}
        for model_name, config in all_configs.items():
            wp_to_ti = {wp: i for i, wp in enumerate(config['test_indices'])}
            if window_position in wp_to_ti:
                test_idx_map[model_name] = wp_to_ti[window_position]
            else:
                print(f"  Skipping {model_name} — window position not in test set.")

        if not test_idx_map:
            print(f"  No models have this window position — skipping.")
            continue

        filtered_results = {m: all_results[m] for m in test_idx_map}
        filtered_configs = {m: all_configs[m] for m in test_idx_map}

        try:
            plot_comparative_case_study(
                forecasting_dataset=shared_dataset,
                all_results=filtered_results,
                all_configs=filtered_configs,
                window_position=window_position,
                test_idx_map=test_idx_map,
            )
        except Exception as e:
            logger.error(f"Error plotting comparative case {rank + 1}: {e}")
            import traceback
            traceback.print_exc()

    return {
        'files':                     result_files,
        'selected_window_positions': selected_window_positions,
        'all_results':               all_results,
        'all_configs':               all_configs,
        'dataset':                   shared_dataset,
        'model_name_map':            model_name_map,
    }


def _filter_indices(results, config, dataset, event_types, min_strength, max_strength, exclude_quiet):
    test_wps = config['test_indices']
    wps = np.array(test_wps).copy()
    if event_types is not None or exclude_quiet:
        wps = dataset.filter_indices_by_event_type(
            wps.tolist(), event_types=event_types, exclude_quiet=exclude_quiet)
    if min_strength is not None or max_strength is not None:
        wps = dataset.filter_indices_by_storm_strength(
            wps.tolist() if not isinstance(wps, list) else wps,
            min_strength=min_strength if min_strength is not None else 0.0,
            max_strength=max_strength)
    pos2idx = {wp: i for i, wp in enumerate(test_wps)}
    return [pos2idx[wp] for wp in wps if wp in pos2idx]


def _binned(x, y, edges, min_count=8):
    """Mean of y in bins of x; returns (centres, means, sems) skipping sparse bins."""
    centres, means, sems = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= min_count:
            centres.append(0.5 * (lo + hi))
            means.append(np.mean(y[m]))
            sems.append(np.std(y[m]) / np.sqrt(m.sum()))
    return np.array(centres), np.array(means), np.array(sems)


def plot_distribution_parameters(
    results, event_types=None, min_strength=None, max_strength=None,
    exclude_quiet=False, storm_threshold=4.5, bins=None,
    save=False, save_name=None,
):
    """
    results : Path | list[Path] | dict[label -> Path]
        One or several result files. Multiple are overlaid for comparison.
    storm_threshold : float
        Hp30 level separating storm / quiet for the diagnostic table.
    bins : array-like, optional
        y_true bin edges. Default emphasises the storm region.
    """
    # normalise to {label: path}
    if isinstance(results, (str, Path)):
        results = {Path(results).stem: results}
    elif isinstance(results, (list, tuple)):
        results = {Path(p).stem: p for p in results}

    if bins is None:
        bins = np.array([0, 2, 3, 4, 4.5, 5.5, 6.5, 8, 12])

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    colours = plt.cm.viridis(np.linspace(0, 0.85, len(results)))

    print(f"\n{'model':<24}{'obs μ(S)':>9}{'pred μ(S)':>10}{'σ storm':>9}"
          f"{'σ quiet':>9}{'σ gap':>8}{'corr':>7}{'std rat':>9}{'|z| S':>7}")
    print("-" * 92)

    for (label, path), col in zip(results.items(), colours):
        res, config, _ = load_results(path)
        dataset = recreate_dataset_from_results(path)
        idx = _filter_indices(res, config, dataset, event_types,
                              min_strength, max_strength, exclude_quiet)
        if len(idx) == 0:
            print(f"{label:<24} no samples matched filter")
            continue

        log_mu = np.asarray(res['log_mu'])[idx]
        log_sig = np.asarray(res['log_sigma'])[idx]
        y = np.asarray(res['y_test'])[idx]
        median = np.exp(log_mu)
        z = (np.log(np.maximum(y, 1e-9)) - log_mu) / log_sig   # ~N(0,1) iff calibrated
        storm = y > storm_threshold

        # Panel 1: predicted median vs strength
        c, m, s = _binned(y, median, bins)
        axes[0].errorbar(c, m, yerr=s, fmt='o-', color=col, label=label, capsize=3)

        # Panel 2: sigma vs strength
        c, m, s = _binned(y, log_sig, bins)
        axes[1].errorbar(c, m, yerr=s, fmt='o-', color=col, label=label, capsize=3)

        # Panel 3: dispersion calibration — std of z per bin
        c, m, _ = _binned(y, z, bins)
        zc, zstd = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mm = (y >= lo) & (y < hi)
            if mm.sum() >= 8:
                zc.append(0.5 * (lo + hi)); zstd.append(np.std(z[mm]))
        axes[2].plot(zc, zstd, 'o-', color=col, label=label)

        # diagnostic row
        sig_gap = log_sig[storm].mean() - log_sig[~storm].mean() if storm.any() else np.nan
        print(f"{label:<24}{y[storm].mean():>9.2f}{median[storm].mean():>10.2f}"
              f"{log_sig[storm].mean():>9.3f}{log_sig[~storm].mean():>9.3f}"
              f"{sig_gap:>8.3f}{np.corrcoef(median, y)[0,1]:>7.3f}"
              f"{median.std()/y.std():>9.3f}{np.mean(np.abs(z[storm])):>7.2f}")

    # references
    yr = np.linspace(bins[0] + 0.1, bins[-1], 100)
    axes[0].plot(yr, yr, 'k--', lw=1.2, alpha=0.6, label='perfect (y=x)')
    axes[0].axvline(storm_threshold, color='red', ls=':', alpha=0.5)
    axes[0].set(xlabel='y_true (Max Hp30)', ylabel='Predicted median Hp30',
                title='Centre: median vs strength\n(below y=x on storms = undershoot)')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].axvline(storm_threshold, color='red', ls=':', alpha=0.5, label=f'storm thr {storm_threshold}')
    axes[1].set(xlabel='y_true (Max Hp30)', ylabel='log_sigma (σ)',
                title='Spread: σ vs strength\n(σ should RISE on storms; falling = inverted)')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    axes[2].axhline(1.0, color='k', ls='--', lw=1.2, alpha=0.6, label='calibrated (z-std=1)')
    axes[2].axvline(storm_threshold, color='red', ls=':', alpha=0.5)
    axes[2].set(xlabel='y_true (Max Hp30)', ylabel='std of z = (ln y − μ)/σ',
                title='Dispersion calibration\n(>1 = under-dispersed in that range)')
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    if save and save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"\nSaved to: {save_name}")
    plt.show()
    return fig


# Example usage and testing
if __name__ == "__main__":
    from storm_regression.predictive import (
        LogNormalForecast, NormalForecast, WeibullForecast,
    )

    print("=" * 64)
    print("Testing forecast_analysis evaluation functions")
    print("=" * 64)

    rng = np.random.default_rng(42)
    n = 200

    # ---- 1. Deterministic metrics -------------------------------------------
    print("\n1. Deterministic metrics (evaluate_regression_forecast)")
    print("-" * 64)
    y_true = np.exp(rng.normal(1.4, 0.4, n))          # positive, right-skewed
    y_pred = y_true + rng.normal(0, 0.5, n)           # noisy predictions
    det = evaluate_regression_forecast(y_pred, y_true)
    for k, v in det.items():
        print(f"  {k.upper():<12}: {v:.4f}")

    # ---- 2. CRPS via the adapter, for each distribution ---------------------
    # These route through the verified closed forms in predictive.py.
    print("\n2. Probabilistic metrics (evaluate_distribution_forecast)")
    print("-" * 64)

    log_mu = rng.normal(1.4, 0.3, n)
    log_sigma = np.abs(rng.normal(0.5, 0.1, n)) + 0.05
    # draw observations from the forecast itself (well-calibrated case)
    y_ln = np.exp(log_mu + log_sigma * rng.standard_normal(n))

    ln = evaluate_distribution_forecast(
        y_ln, distribution="lognormal",
        log_mu_pred=log_mu, log_sigma_pred=log_sigma,
    )
    print(f"  LogNormal  mean CRPS : {ln['crps']:.4f}")

    mu = rng.normal(5.0, 1.0, n)
    sigma = np.abs(rng.normal(1.5, 0.3, n)) + 0.1
    y_no = mu + sigma * rng.standard_normal(n)
    no = evaluate_distribution_forecast(
        y_no, distribution="normal",
        mu_pred=mu, sigma_pred=sigma,
    )
    print(f"  Normal     mean CRPS : {no['crps']:.4f}")

    lam = np.abs(rng.normal(4.0, 0.5, n)) + 0.5
    k = np.abs(rng.normal(2.0, 0.3, n)) + 0.3
    y_wb = weibull_min.rvs(c=k, scale=lam, random_state=rng)
    wb = evaluate_distribution_forecast(
        y_wb, distribution="weibull",
        lambda_pred=lam, k_pred=k,
    )
    print(f"  Weibull    mean CRPS : {wb['crps']:.4f}")

    # ---- 3. Verify closed-form CRPS against Monte Carlo ---------------------
    # A single fixed case per family; closed form should match the energy-score
    # estimator to ~2-3 dp. This guards against a CRPS regression.
    print("\n3. Closed-form vs Monte-Carlo CRPS (sanity check)")
    print("-" * 64)

    def _mc_crps(sampler, y, n_samp=2_000_000):
        X = sampler(n_samp)
        Xp = sampler(n_samp)
        return np.mean(np.abs(X - y)) - 0.5 * np.mean(np.abs(X - Xp))

    checks = [
        ("LogNormal", LogNormalForecast(np.array([1.5]), np.array([0.5])),
         4.5, lambda m: np.exp(1.5 + 0.5 * rng.standard_normal(m))),
        ("Normal", NormalForecast(np.array([5.0]), np.array([1.5])),
         5.0, lambda m: 5.0 + 1.5 * rng.standard_normal(m)),
        ("Weibull", WeibullForecast(np.array([4.0]), np.array([2.0])),
         4.5, lambda m: weibull_min.rvs(c=2.0, scale=4.0, size=m, random_state=rng)),
    ]
    print(f"  {'family':<12}{'closed-form':>13}{'MC ref':>11}{'|diff|':>9}")
    for name, fc, yv, sampler in checks:
        cf = float(fc.crps(np.array([yv]))[0])
        mc = _mc_crps(sampler, yv)
        flag = "ok" if abs(cf - mc) < 0.02 else "CHECK"
        print(f"  {name:<12}{cf:>13.4f}{mc:>11.4f}{abs(cf - mc):>9.4f}  {flag}")

    print("\n" + "=" * 64)
    print("Done.")
    print("=" * 64)