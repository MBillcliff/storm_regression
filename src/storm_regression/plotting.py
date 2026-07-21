"""
Plotting utilities for regression forecast analysis.

Functions for visualizing case studies, ensemble comparisons, and distribution parameters.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from datetime import timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from scipy import stats
from scipy.stats import norm, lognorm
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from storm_utils.config_paths import get_project_paths

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================

def save_figure(fig_name, subfolder='exploratory', huxt_id=None, dpi=150):
    """
    Save figure with organized structure.
    
    Parameters
    ----------
    fig_name : str
        Descriptive figure name
    subfolder : str
        Purpose-based subfolder: 'model_comparison', 'case_studies', 
        'distribution_analysis', 'thesis', 'exploratory'
    huxt_id : int, optional
        If provided, includes HUXt ID in filename
    dpi : int
        Resolution for saved figure
    
    Returns
    -------
    Path
        Path to saved figure
    """
    paths = get_project_paths()
    figure_dir = paths['regression_figures'] / subfolder
    figure_dir.mkdir(parents=True, exist_ok=True)
    
    # Add HUXt ID to filename if specified
    if huxt_id is not None:
        fig_name = f'huxt{huxt_id}_{fig_name}'
    
    # Ensure .png extension
    if not fig_name.endswith('.png'):
        fig_name += '.png'
    
    save_path = figure_dir / fig_name
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    logger.info(f"Saved figure: {save_path}")
    
    return save_path


def weibull_exceeds(lam_pred, k_pred, thresholds):
    """Calculate probability of exceeding thresholds for Weibull distribution."""
    return np.exp(-(thresholds / lam_pred)**k_pred)


def weibull_pdf(x, k, lambda_):
    """Weibull probability density function."""
    return (k / lambda_) * (x / lambda_)**(k - 1) * np.exp(- (x / lambda_)**k)


def lognormal_exceeds(log_mu_pred, log_sigma_pred, thresholds):
    """
    Calculate probability of exceeding thresholds for LogNormal distribution.
    
    Parameters
    ----------
    log_mu_pred : float or array
        Mean of the underlying normal distribution (μ in log-space)
    log_sigma_pred : float or array
        Standard deviation of the underlying normal distribution (σ in log-space)
    thresholds : array-like
        Threshold values to evaluate
    
    Returns
    -------
    prob : array
        Probability of exceeding each threshold
    """
    from scipy.stats import lognorm
    
    thresholds = np.asarray(thresholds)
    return 1 - lognorm.cdf(thresholds, s=log_sigma_pred, scale=np.exp(log_mu_pred))


# ============================================================================
# Main Plotting Functions
# ============================================================================

def weibull_dual_plot(lam_pred, k_pred, weighted_mean_pred, true, 
                     save=False, save_name=None, huxt_id=None):
    """
    Plot Weibull exceedance probability and PDF.
    
    Parameters
    ----------
    lam_pred : float
        Weibull scale parameter
    k_pred : float
        Weibull shape parameter
    weighted_mean_pred : float
        Weighted mean prediction
    true : float
        True observed value
    save : bool
        Whether to save the figure
    save_name : str, optional
        Filename for saving (without extension)
    huxt_id : int, optional
        HUXt run ID for filename
    """
    thresholds = np.linspace(0, 13, 1000)
    integer_thresholds = np.arange(0, 13)
    
    # Exceedance probability
    p_exceed = weibull_exceeds(lam_pred, k_pred, thresholds)
    p_int = weibull_exceeds(lam_pred, k_pred, integer_thresholds)
    p_true = weibull_exceeds(lam_pred, k_pred, true)

    weibull_median = lam_pred * (np.log(2) ** (1/k_pred))
    
    # PDF
    pdf_vals = weibull_pdf(thresholds, k_pred, lam_pred)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Exceedance Probability
    ax1 = axes[0]
    ax1.plot(thresholds, p_exceed, label='Weibull exceedance probability')
    ax1.scatter(integer_thresholds, p_int, color='red', zorder=5)
    ax1.scatter(true, p_true, color='k', marker='x', s=100, zorder=10, 
               label=f'P(x > true)={p_true:.2f}')
    
    # Annotate integer points
    for x, y in zip(integer_thresholds, p_int):
        ax1.text(x, y + 0.02, f"{y:.2f}", ha='center', va='bottom', fontsize=8)

    fig.suptitle(f'λ: {lam_pred:.2f}, k: {k_pred:.2f}', fontsize=20)
    
    ax1.set_xticks(integer_thresholds)
    ax1.set_xlabel('Hp30 Threshold')
    ax1.set_ylabel('P(Hp30_max > threshold)')
    ax1.set_title('Exceedance Probability')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, 1.05)
    ax1.axvline(weighted_mean_pred, linestyle='--', color='red', 
               label=f'weighted_mean = {weighted_mean_pred:.2f}')
    ax1.axvline(true, linestyle='--', color='green', label=f'true = {true:.2f}')
    ax1.axvline(weibull_median, linestyle='--', color='k', label=f'Weibull Median')
    ax1.legend(loc='upper right', fontsize=10)
    
    # Right: PDF
    ax2 = axes[1]
    ax2.plot(thresholds, pdf_vals, color='green', label='Weibull PDF')
    ax2.set_xlabel('Hp30 Threshold')
    ax2.set_ylabel('PDF')
    ax2.set_title('Weibull Probability Density Function')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save and save_name:
        save_figure(save_name, subfolder='distribution_analysis', huxt_id=huxt_id)
    
    plt.show()


def plot_case_study(forecasting_dataset, results, window_position, test_idx,
                   config=None, thresholds=None, save=False, save_name=None, huxt_id=None):
    """
    Plot a single forecasting window with forecast probability gradient.

    Supports BOTH the single-LogNormal head (results have log_mu/log_sigma) and the
    mixture head (results have mu_m/sigma_m/alpha). For a mixture, the predictive
    distribution, the exceedance probabilities, and the probability contours are all
    computed from the full weighted mixture sum_k w_k LogNormal(mu_k, sigma_k); the
    histogram panel also overlays the individual weighted components (dashed) so the
    bimodality / component usage is visible per window.

    Layout: Bz_GSM (top), Solar Wind Velocity, Hp30 with probability contours (left),
    and predictive distribution (right side, vertical).
    """
    fontsize = 16
    ylim = 13
    ylen = ylim * 3 + 1
    if thresholds is None:
        thresholds = np.linspace(1/6, ylim + 1/6, ylen)

    # Get window data
    print(f"Window position: {window_position}")
    window = forecasting_dataset[window_position]

    # Extract data
    v = window['v']                          # (T, Nens)
    omni = window['omni_sw_plotting']
    omni_var = window['omni_plotting']
    target = window['target_plotting']
    max_target = float(window['max_target'])
    center_idx = window['center_idx']
    window_label = window['window_label']

    # Timestamps
    T0 = forecasting_dataset.df.index[center_idx]
    F_start = forecasting_dataset.df.index[center_idx + forecasting_dataset.lead_time]

    n_steps = forecasting_dataset.max_offset - forecasting_dataset.min_offset
    timestamps = pd.date_range(
        start=T0 + timedelta(minutes=30 * forecasting_dataset.min_offset),
        periods=n_steps,
        freq='30min',
    )
    print(f'T0: {T0}')

    # Determine model type (ensemble vs direct/mlp)
    model_type = results.get('model_type', 'ensemble' if 'ensemble_predictions' in results else 'direct')

    # ── Predictive distribution: mixture-aware ────────────────────────────────
    # Detect a mixture forecast by its keys; otherwise fall back to single LogNormal.
    is_mixture = ('mu_m' in results and 'sigma_m' in results and 'alpha' in results)

    # --- diagnostic ---
    print(f"[case study] result keys: {sorted(results.keys())}")
    print(f"[case study] is_mixture = {is_mixture}")
    if is_mixture:
        print(f"[case study] K = {np.atleast_1d(results['mu_m'][test_idx]).shape[0]}")
        print(f"[case study]   mu_m    = {np.atleast_1d(results['mu_m'][test_idx])}")
        print(f"[case study]   sigma_m = {np.atleast_1d(results['sigma_m'][test_idx])}")
        print(f"[case study]   alpha   = {np.atleast_1d(results['alpha'][test_idx])}")
    else:
        print("[case study] -> falling back to single LogNormal (no mu_m/sigma_m/alpha keys)")
    # --- end diagnostic ---

    if is_mixture:
        mu_k    = np.atleast_1d(results['mu_m'][test_idx]).astype(float)     # (K,)
        sigma_k = np.atleast_1d(results['sigma_m'][test_idx]).astype(float)  # (K,)
        w_k     = np.atleast_1d(results['alpha'][test_idx]).astype(float)    # (K,)
        K = len(mu_k)

        def mix_exceeds(thresh):
            """P(Y >= thresh) under the mixture. thresh scalar or 1-D -> matches shape."""
            thresh = np.atleast_1d(np.asarray(thresh, dtype=float))
            out = np.zeros_like(thresh, dtype=float)
            for k in range(K):
                out += w_k[k] * (1.0 - lognorm.cdf(thresh, s=sigma_k[k], scale=np.exp(mu_k[k])))
            return out

        def mix_pdf(y):
            y = np.asarray(y, dtype=float)
            return sum(w_k[k] * lognorm.pdf(y, s=sigma_k[k], scale=np.exp(mu_k[k]))
                       for k in range(K))

        # mixture median via numerical CDF inversion on a fine grid
        _grid = np.linspace(1e-3, ylim + 2, 4000)
        _cdf = sum(w_k[k] * lognorm.cdf(_grid, s=sigma_k[k], scale=np.exp(mu_k[k]))
                   for k in range(K))
        mixture_median = float(np.interp(0.5, _cdf, _grid))
        dist_label = "Mixture"
    else:
        log_mu_pred    = float(results['log_mu'][test_idx])
        log_sigma_pred = float(results['log_sigma'][test_idx])

        def mix_exceeds(thresh):
            thresh = np.atleast_1d(np.asarray(thresh, dtype=float))
            return np.asarray(lognormal_exceeds(log_mu_pred, log_sigma_pred, thresh)).ravel()

        def mix_pdf(y):
            return lognorm.pdf(y, s=log_sigma_pred, scale=np.exp(log_mu_pred))

        mixture_median = float(np.exp(log_mu_pred))
        dist_label = "LogNormal"

    # Exceedance at the observed maximum (the traffic-light number)
    p_exceeds = mix_exceeds([max_target])

    # Probability contours over the forecast window
    probs = mix_exceeds(thresholds)                                  # (n_thresh,)
    probs = np.tile(probs[:, None], (1, forecasting_dataset.forecast_steps))

    # ── Filter ensemble members if required ──────────────────────────────────
    if config is not None and config.get('filter_ensemble', False):
        n_keep   = config.get('n_ensemble_keep', 50)
        v_input  = v[:omni.shape[0], :]
        mae      = np.mean(np.abs(v_input - omni[:, None]), axis=0)
        best_idx = np.argsort(mae)[:n_keep]
        v        = v[:, best_idx]
        print(f"Filtered ensemble: kept {n_keep} of {mae.shape[0]} members")

    # ── Percentile ensemble members ───────────────────────────────────────────
    percentile_member_indices = {}
    if config is not None and 'mlp_ensemble_percentiles' in config:
        percentiles               = sorted(config['mlp_ensemble_percentiles'])
        n_members                 = v.shape[1]
        selection_method          = config.get('ensemble_selection_method', 'snap')
        print('selection method', selection_method)

        if selection_method == 'snap':
            member_mean_v = v.mean(axis=0)
            rank_order    = np.argsort(member_mean_v)
            for p in percentiles:
                rank_idx = min(int(np.floor(p / 100 * n_members)), n_members - 1)
                percentile_member_indices[p] = int(rank_order[rank_idx])
        elif selection_method == 'per_timestep':
            for p in percentiles:
                v_p      = np.percentile(v, p, axis=1, method='nearest')
                member_indices = np.argmin(np.abs(v - v_p[:, None]), axis=1)
                percentile_member_indices[p] = int(np.bincount(member_indices).argmax())

    # ── Pair outer percentiles symmetrically ─────────────────────────────────
    percentile_bands = []
    median_p         = None
    if percentile_member_indices:
        ps       = sorted(percentile_member_indices.keys())
        median_p = ps[len(ps) // 2]
        for i in range(len(ps) // 2):
            percentile_bands.append((ps[i], ps[-(i + 1)]))

    # GET ICME AND SIR FLAGS AND FIND PERIODS
    icme_periods = []
    sir_periods = []
    if 'ICME_flag' in forecasting_dataset.df.columns:
        window_start_idx = center_idx + forecasting_dataset.min_offset
        window_end_idx = center_idx + forecasting_dataset.max_offset + 1
        window_df = forecasting_dataset.df.iloc[window_start_idx:window_end_idx]
        icme_flags = window_df['ICME_flag'].values
        sir_flags = window_df['SIR_flag'].values

        in_icme = False
        icme_start = None
        for i in range(len(icme_flags)):
            if icme_flags[i] and not in_icme:
                icme_start = timestamps[i]; in_icme = True
            elif not icme_flags[i] and in_icme:
                icme_periods.append((icme_start, timestamps[i - 1])); in_icme = False
        if in_icme:
            icme_periods.append((icme_start, timestamps[-1]))

        in_sir = False
        sir_start = None
        for i in range(len(sir_flags)):
            if sir_flags[i] and not in_sir:
                sir_start = timestamps[i]; in_sir = True
            elif not sir_flags[i] and in_sir:
                sir_periods.append((sir_start, timestamps[i - 1])); in_sir = False
        if in_sir:
            sir_periods.append((sir_start, timestamps[-1]))

    # ── Figure layout ────────────────────────────────────────────────────────
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.5, 2, 2.5],
                  width_ratios=[3, 1], hspace=0.05, wspace=0.05)

    ax_bz   = fig.add_subplot(gs[0, 0])
    ax_v    = fig.add_subplot(gs[1, 0], sharex=ax_bz)
    ax_hp30 = fig.add_subplot(gs[2, 0], sharex=ax_bz)
    ax_hist = fig.add_subplot(gs[:, 1])

    xF = pd.date_range(
        start=F_start + pd.Timedelta(minutes=15),
        periods=forecasting_dataset.forecast_steps,
        freq='30min'
    )
    xG1 = pd.date_range(
        start=T0 + timedelta(minutes=30 * forecasting_dataset.min_offset),
        periods=forecasting_dataset.lead_time - forecasting_dataset.min_offset + 1,
        freq='30min',
    )
    xG = pd.date_range(
        start=T0 + timedelta(minutes=30 * forecasting_dataset.min_offset),
        periods=forecasting_dataset.max_offset - forecasting_dataset.min_offset + 1,
        freq='30min',
    )

    def add_event_bars(ax, y_position):
        for i, (start, end) in enumerate(icme_periods):
            cap = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
            ax.plot([start, end], [y_position, y_position],
                    color='darkred', lw=3, solid_capstyle='butt',
                    label='ICME' if i == 0 else '', alpha=0.8)
            for x in (start, end):
                ax.plot([x, x], [y_position - cap, y_position + cap],
                        color='darkred', lw=3, solid_capstyle='butt', alpha=0.8)
        if not icme_periods:
            y_off = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
            for i, (start, end) in enumerate(sir_periods):
                cap = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                ax.plot([start, end], [y_position - y_off, y_position - y_off],
                        color='darkblue', lw=3, solid_capstyle='butt',
                        label='SIR' if i == 0 else '', alpha=0.8)
                for x in (start, end):
                    ax.plot([x, x], [y_position - y_off - cap, y_position - y_off + cap],
                            color='darkblue', lw=3, solid_capstyle='butt', alpha=0.8)

    # ===== PANEL 1: Bz_GSM =====
    if omni_var.ndim == 2:
        ax_bz.plot(timestamps, omni_var[:, 0], lw=1, color='red', label='OMNI Bz_GSM')
    else:
        ax_bz.plot(timestamps, omni_var, lw=1, color='red', label='OMNI Bz_GSM')
    ax_bz.axvline(T0, color='black', linestyle='--', label='T0')
    ax_bz.axvspan(xF[0] - timedelta(minutes=15), xF[-1] + timedelta(minutes=15),
                  color='grey', alpha=0.1, label='Forecast Window')
    add_event_bars(ax_bz, ax_bz.get_ylim()[1] * 0.95)
    ax_bz.set_ylabel('Bz_GSM (nT)', fontsize=fontsize)
    ax_bz.legend(loc='upper right')
    ax_bz.set_xlim(timestamps[0], timestamps[-1])
    ax_bz.set_title(f"Event type: {window_label}", fontsize=fontsize + 2)

    # ===== PANEL 2: Solar Wind Velocity =====
    highlighted_members = set(percentile_member_indices.values())
    for i in range(v.shape[1]):
        if i in highlighted_members:
            continue
        ax_v.plot(timestamps, v[:, i], color='blue', lw=1, alpha=0.1,
                  label='v ensemble' if i == 0 else None)

    band_colours = ['steelblue', 'royalblue']
    band_alphas  = [0.2, 0.4]
    for j, (lo_p, hi_p) in enumerate(percentile_bands):
        lo_v = v[:, percentile_member_indices[lo_p]]
        hi_v = v[:, percentile_member_indices[hi_p]]
        colour = band_colours[j % len(band_colours)]
        alpha  = band_alphas[j % len(band_alphas)]
        ax_v.fill_between(timestamps, lo_v, hi_v,
                          alpha=alpha, color=colour, label=f'p{lo_p}-p{hi_p}')
        ax_v.plot(timestamps, lo_v, color=colour, lw=2.0, alpha=0.9, linestyle='--')
        ax_v.plot(timestamps, hi_v, color=colour, lw=2.0, alpha=0.9, linestyle='--')

    if median_p is not None:
        ax_v.plot(timestamps, v[:, percentile_member_indices[median_p]],
                  color='blue', lw=2, linestyle='-', alpha=0.9,
                  zorder=4, label=f'p{median_p}')

    ax_v.plot(timestamps, omni, 'k', lw=1.5, label='OMNI')
    ax_v.axvline(T0, color='black', linestyle='--', label='T0')
    ax_v.axvspan(xF[0] - timedelta(minutes=15), xF[-1] + timedelta(minutes=15),
                 color='grey', alpha=0.1)
    add_event_bars(ax_v, ax_v.get_ylim()[1] * 0.95)
    ax_v.set_ylabel('v (km/s)', fontsize=fontsize)
    ax_v.legend(loc='upper right')
    ax_v.set_xlim(timestamps[0], timestamps[-1])

    # ===== PANEL 3: Hp30 with probability contours =====
    Edges = [0, 5, 6, 7, 8, 9, ylim]
    nbands = len(Edges) - 1
    colors_arr = np.arange(1, nbands + 1)[:, None] + 1
    Z1 = np.tile(colors_arr, (1, len(xG) - 1))

    ax_hp30.pcolormesh(xG, Edges, Z1, shading='flat', cmap='Blues', alpha=0.3)
    ax_hp30.contourf(xF, thresholds, probs, levels=20, cmap='Reds', alpha=0.0)
    contour_lines = ax_hp30.contour(xF, thresholds, probs,
                                    levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                                    colors='darkred', linewidths=1)
    ax_hp30.clabel(contour_lines, inline=True, fontsize=8, fmt='%0.1f', rightside_up=True)

    G_labels = [5, 6, 7, 8, 9, 10]
    g_names  = ['G1', 'G2', 'G3', 'G4', 'G5']
    for i in range(len(G_labels) - 1):
        y_mid = (G_labels[i] + G_labels[i + 1]) / 2
        ax_hp30.text(xG1[0] + pd.Timedelta(minutes=120), y_mid, g_names[i],
                     va='center', ha='right', fontsize=10, color='black')
        ax_hp30.hlines(G_labels[i], timestamps[0], timestamps[-1],
                       color='lightblue', lw=0.5)

    max_hp30_colour = 'olive'
    ax_hp30.plot(timestamps, target, lw=1, label='Hp30')
    ax_hp30.plot([xF[0], xF[-1]], [max_target, max_target],
                 color=max_hp30_colour, linestyle='--', label='Max Hp30')
    ax_hp30.text(xF[0] - timedelta(minutes=10 * 60), max_target - 0.5,
                 f'p(X>={max_target:.2f})={p_exceeds[0]:.2f}',
                 verticalalignment='bottom', color=max_hp30_colour)
    ax_hp30.axvline(T0, color='black', linestyle='--', label='T0')
    ax_hp30.axvspan(xF[0] - timedelta(minutes=15), xF[-1] + timedelta(minutes=15),
                    color='grey', alpha=0.1, label='Forecast Window')
    add_event_bars(ax_hp30, ylim * 0.95)
    ax_hp30.set_ylabel('Hp30', fontsize=fontsize)
    ax_hp30.set_xlabel('Time (UTC)', fontsize=fontsize)
    ax_hp30.legend(loc='upper right')
    ax_hp30.set_ylim(0, ylim)
    ax_hp30.set_xlim(timestamps[0], timestamps[-1])
    ax_hp30.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    # ===== PANEL 4: Vertical predictive distribution =====
    y_range = np.linspace(0.01, ylim + 1, 300)
    pred_pdf = mix_pdf(y_range)

    if model_type == 'ensemble':
        if 'ensemble_predictions' not in results:
            raise ValueError("model_type is 'ensemble' but 'ensemble_predictions' not found")
        ensemble_preds = results['ensemble_predictions'][test_idx]
        ax_hist.hist(ensemble_preds, bins=30, alpha=0.7, color='steelblue',
                     edgecolor='black', density=True, orientation='horizontal',
                     label='Ensemble predictions')
        ax_hist.plot(pred_pdf, y_range, color='orange', linestyle='--', lw=2,
                     label=f'{dist_label} fit')
    else:  # direct / MLP
        ax_hist.plot(pred_pdf, y_range, color='orange', lw=2, label=f'{dist_label} pdf')
        ax_hist.fill_betweenx(y_range, pred_pdf, alpha=0.2, color='orange')

    # Overlay individual weighted components for a mixture (the bimodality view)
    if is_mixture:
        comp_colours = ['purple', 'green', 'brown', 'teal', 'magenta']
        for k in range(K):
            comp_pdf = w_k[k] * lognorm.pdf(y_range, s=sigma_k[k], scale=np.exp(mu_k[k]))
            ax_hist.plot(comp_pdf, y_range,
                         color=comp_colours[k % len(comp_colours)],
                         lw=1.2, ls='--', alpha=0.85,
                         label=f'comp {k}: w={w_k[k]:.2f}, μ={mu_k[k]:.2f}, σ={sigma_k[k]:.2f}')

    ax_hist.axhline(max_target, color=max_hp30_colour, linestyle='--', lw=2, label='True value')
    ax_hist.axhline(mixture_median, color='red', linestyle='-.', lw=1.5,
                    label=f'Median: {mixture_median:.2f}')
    ax_hist.yaxis.tick_right()
    ax_hist.yaxis.set_label_position("right")
    ax_hist.set_ylabel('Predicted Hp30', fontsize=fontsize)
    ax_hist.set_xlabel('Density', fontsize=fontsize - 2)
    ax_hist.legend(loc='upper left', fontsize=8, bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax_hist.set_ylim(0, ylim)

    ax_bz.tick_params(labelbottom=False)
    ax_v.tick_params(labelbottom=False)
    fig.autofmt_xdate()

    if save and save_name:
        save_figure(save_name, subfolder='case_studies', huxt_id=huxt_id)

    plt.show()
    


def plot_window_data(forecasting_dataset, window_position, thresholds=None,
                     save=False, save_name=None, huxt_id=None):
    """
    Plot a single forecasting window (data only, no predictions).

    Layout: Bz_GSM (top), Solar Wind Velocity (middle), Hp30 (bottom)

    Parameters
    ----------
    forecasting_dataset : ForecastingDataset
        Dataset object.
    window_position : int
        Position in valid_indices.
    thresholds : np.ndarray, optional
        Unused, kept for compatibility.
    save : bool
        Whether to save the figure
    save_name : str, optional
        Filename for saving
    huxt_id : int, optional
        HUXt run ID
    """
    fontsize = 16
    ylim = 13

    print(f"Window position: {window_position}")
    forecasting_dataset.set_omni_columns(['Bz_GSM'])
    window = forecasting_dataset[window_position]

    # Extract data
    v = window['v']
    omni = window['omni_sw_plotting']
    omni_var = window['omni_plotting']
    target = window['target_plotting']
    max_target = float(window['max_target'])
    center_idx = window['center_idx']
    window_label = window['window_label']

    # Timestamps
    T0 = forecasting_dataset.df.index[center_idx]
    F_start = forecasting_dataset.df.index[center_idx + forecasting_dataset.lead_time]

    n_steps = forecasting_dataset.max_offset - forecasting_dataset.min_offset
    timestamps = pd.date_range(
        start=T0 + timedelta(minutes=30 * forecasting_dataset.min_offset),
        periods=n_steps,
        freq='30min',
    )

    xF = pd.date_range(
        start=F_start + pd.Timedelta(minutes=15),
        periods=forecasting_dataset.forecast_steps,
        freq='30min'
    )

    xG = pd.date_range(
        start=T0 + timedelta(minutes=30 * forecasting_dataset.min_offset),
        periods=forecasting_dataset.max_offset - forecasting_dataset.min_offset + 1,
        freq='30min',
    )

    # Create figure
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.5, 2, 2.5], hspace=0.05)

    ax_bz = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1], sharex=ax_bz)
    ax_hp30 = fig.add_subplot(gs[2], sharex=ax_bz)

    # ===== PANEL 1: Bz_GSM =====
    if omni_var.ndim == 2:
        ax_bz.plot(timestamps, omni_var[:, 0], lw=1, label='OMNI Bz_GSM', color='red')
    else:
        ax_bz.plot(timestamps, omni_var, lw=1, label='OMNI Bz_GSM', color='red')
    
    # ADD THIS: Mark ICME/SIR regions
    if 'ICME_flag' in forecasting_dataset.df.columns:
        input_start_idx = center_idx + forecasting_dataset.min_offset
        input_end_idx = center_idx + forecasting_dataset.max_offset + 1
        window_df = forecasting_dataset.df.iloc[input_start_idx:input_end_idx]
        icme_flags = window_df['ICME_flag'].values
        sir_flags = window_df['SIR_flag'].values
        
        # Find ICME regions
        in_icme = False
        for i, flag in enumerate(icme_flags):
            if flag and not in_icme:
                start_time = timestamps[i]
                in_icme = True
            elif not flag and in_icme:
                end_time = timestamps[i-1]
                ax_bz.axvspan(start_time, end_time, alpha=0.3, color='red', 
                             label='ICME' if i == 1 else '', zorder=0)
                in_icme = False
        if in_icme:
            ax_bz.axvspan(start_time, timestamps[-1], alpha=0.3, color='red', 
                         label='ICME', zorder=0)
        
        # Find SIR regions
        in_sir = False
        for i, flag in enumerate(sir_flags):
            if flag and not in_sir:
                start_time = timestamps[i]
                in_sir = True
            elif not flag and in_sir:
                end_time = timestamps[i-1]
                ax_bz.axvspan(start_time, end_time, alpha=0.3, color='blue', 
                             label='SIR' if i == 1 else '', zorder=0)
                in_sir = False
        if in_sir:
            ax_bz.axvspan(start_time, timestamps[-1], alpha=0.3, color='blue', 
                         label='SIR', zorder=0)

    ax_bz.axvline(T0, color='black', linestyle='--', label='T0')
    ax_bz.axvspan(xF[0] - timedelta(minutes=15), xF[-1] + timedelta(minutes=15),
                  color='grey', alpha=0.1, label='Forecast Window')
    ax_bz.set_ylabel('Bz_GSM (nT)', fontsize=fontsize)
    ax_bz.legend(loc='upper right')
    ax_bz.set_title(f'event: {window_label}', fontsize=fontsize + 2)

    # Panel 2: Velocity
    for i in range(v.shape[1]):
        ax_v.plot(timestamps, v[:, i], color='blue', lw=1, alpha=0.1,
                 label='v ensemble' if i == 0 else None)
    ax_v.plot(timestamps, omni, 'k', lw=1, label='OMNI')
    ax_v.axvline(T0, color='black', linestyle='--')
    ax_v.axvspan(xF[0] - timedelta(minutes=15), xF[-1] + timedelta(minutes=15),
                 color='grey', alpha=0.1)
    ax_v.set_ylabel('v (km/s)', fontsize=fontsize)
    ax_v.legend(loc='upper right')

    # Panel 3: Hp30
    Edges = [0, 5, 6, 7, 8, 9, ylim]
    colors_arr = np.arange(1, len(Edges))[:, None] + 1
    Z1 = np.tile(colors_arr, (1, len(xG) - 1))

    ax_hp30.pcolormesh(xG, Edges, Z1, shading='flat', cmap='Blues', alpha=0.3)

    labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    for i in range(len(Edges) - 1):
        y_mid = 0.5 * (Edges[i] + Edges[i + 1])
        if i < len(labels):
            ax_hp30.text(xG[0] + pd.Timedelta(hours=2), y_mid, labels[i],
                        va='center', ha='right', fontsize=10, color='black')
        ax_hp30.hlines(Edges[i], timestamps[0], timestamps[-1], color='lightblue', lw=0.5)

    ax_hp30.plot(timestamps, target, lw=1, label='Hp30')
    ax_hp30.plot([xF[0], xF[-1]], [max_target, max_target],
                 linestyle='--', color='olive', label='Max Hp30')
    ax_hp30.axvline(T0, color='black', linestyle='--')
    ax_hp30.axvspan(xF[0] - timedelta(minutes=15), xF[-1] + timedelta(minutes=15),
                    color='grey', alpha=0.1)

    ax_hp30.set_ylabel('Hp30', fontsize=fontsize)
    ax_hp30.set_xlabel('Time (UTC)', fontsize=fontsize)
    ax_hp30.set_ylim(0, ylim)
    ax_hp30.legend(loc='upper right')

    ax_bz.tick_params(labelbottom=False)
    ax_v.tick_params(labelbottom=False)

    fig.autofmt_xdate()
    plt.tight_layout()
    
    if save and save_name:
        save_figure(save_name, subfolder='case_studies', huxt_id=huxt_id)
    
    plt.show()


def plot_preds_and_targets(preds, targets, title="Predictions vs. Targets", 
                           n_points=500, save=False, save_name=None, huxt_id=None):
    """
    Plot a subset of predictions vs. ground truth targets.
    
    Parameters
    ----------
    preds : array-like
        Predictions
    targets : array-like
        True values
    title : str
        Plot title
    n_points : int
        Number of points to plot (for large datasets)
    save : bool
        Whether to save the figure
    save_name : str, optional
        Filename for saving
    huxt_id : int, optional
        HUXt run ID
    """
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()

    if len(preds) > n_points:
        idx = np.linspace(0, len(preds) - 1, n_points).astype(int)
        preds = preds[idx]
        targets = targets[idx]
        x = idx
    else:
        x = np.arange(len(preds))

    plt.figure(figsize=(12, 6))
    plt.plot(x, preds, label='Predictions', color='tab:blue', alpha=0.7, linewidth=2)
    plt.plot(x, targets, label='Targets', color='tab:orange', alpha=0.7, linewidth=2)
    plt.fill_between(x, preds, targets, color='gray', alpha=0.2, label='Error')

    plt.title(title, fontsize=16)
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Hp30", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save and save_name:
        save_figure(save_name, subfolder='model_comparison', huxt_id=huxt_id)
    
    plt.show()


def plot_forecast_distribution(results, save=False, save_name=None, huxt_id=None):
    """
    Compare distributions of predicted vs true values.
    
    Parameters
    ----------
    results : dict
        Results dictionary with predictions and targets
    save : bool
        Whether to save the figure
    save_name : str, optional
        Filename for saving
    huxt_id : int, optional
        HUXt run ID
    """
    y_pred = results['y_pred_weibull_median']
    y_test = results['y_test']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Overlapping histograms
    ax = axes[0, 0]
    bins = np.arange(-1/6, 13 + 1/3 + 1/6, 1/3)
    ax.hist(y_test, bins=bins, alpha=0.6, label='True', density=True, edgecolor='black')
    ax.hist(y_pred, bins=bins, alpha=0.6, label='Forecast', density=True, edgecolor='black')
    
    ax.axvline(np.mean(y_test), color='blue', linestyle='--', linewidth=2, alpha=0.7, label='True Mean')
    ax.axvline(np.mean(y_pred), color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Forecast Mean')
    
    ax.set_xlabel('Hp30', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution Comparison', fontsize=14)
    ax.set_xlim(0, np.max(y_pred))
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Q-Q plot
    ax = axes[0, 1]
    quantiles = np.linspace(0, 100, 100)
    true_quantiles = np.percentile(y_test, quantiles)
    pred_quantiles = np.percentile(y_pred, quantiles)
    
    ax.scatter(true_quantiles, pred_quantiles, alpha=0.5, s=20)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            'r--', linewidth=2, label='Perfect Match')
    ax.set_xlabel('True Quantiles', fontsize=12)
    ax.set_ylabel('Forecast Quantiles', fontsize=12)
    ax.set_title('Q-Q Plot', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Cumulative distributions
    ax = axes[1, 0]
    sorted_true = np.sort(y_test)
    sorted_pred = np.sort(y_pred)
    cdf_true = np.arange(1, len(sorted_true) + 1) / len(sorted_true)
    cdf_pred = np.arange(1, len(sorted_pred) + 1) / len(sorted_pred)
    
    ax.plot(sorted_true, cdf_true, label='True', linewidth=2)
    ax.plot(sorted_pred, cdf_pred, label='Forecast', linewidth=2)
    ax.set_xlabel('Hp30', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution Functions', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight percentiles
    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:
        true_val = np.percentile(y_test, p)
        pred_val = np.percentile(y_pred, p)
        ax.scatter([true_val], [p/100], color='blue', s=100, zorder=5, edgecolors='black', linewidth=2)
        ax.scatter([pred_val], [p/100], color='orange', s=100, zorder=5, edgecolors='black', linewidth=2)
        ax.axhline(p/100, color='gray', linestyle=':', alpha=0.3)
    
    # 4. Box plots with statistical tests
    ax = axes[1, 1]
    bp = ax.boxplot([y_test, y_pred], tick_labels=['True', 'Forecast'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightsalmon')
    ax.set_ylabel('Hp30', fontsize=12)
    ax.set_title('Distribution Statistics', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Statistical tests
    ks_stat, ks_pval = stats.ks_2samp(y_test, y_pred)
    t_stat, t_pval = stats.ttest_ind(y_test, y_pred)
    mw_stat, mw_pval = stats.mannwhitneyu(y_test, y_pred)
    
    stats_text = f'KS Test: p={ks_pval:.4f}\n'
    stats_text += f'T-Test: p={t_pval:.4f}\n'
    stats_text += f'Mann-Whitney: p={mw_pval:.4f}\n\n'
    stats_text += f'True: μ={np.mean(y_test):.2f}, σ={np.std(y_test):.2f}\n'
    stats_text += f'Forecast: μ={np.mean(y_pred):.2f}, σ={np.std(y_pred):.2f}'
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save and save_name:
        save_figure(save_name, subfolder='distribution_analysis', huxt_id=huxt_id)
    
    plt.show()


def plot_ensemble_comparison(
    results_dict: Dict[str, Dict],
    test_idx: int,
    max_target: float,
    model_colors: Optional[Dict[str, str]] = None,
    figsize: tuple = (14, 6),
    save=False,
    save_name=None,
    huxt_id=None
):
    """
    Plot histogram comparison of raw ensemble forecasts across multiple models.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping model names to their results dicts.
    test_idx : int
        Index to access forecasts in results arrays
    max_target : float
        True observed Hp30 value
    model_colors : dict, optional
        Dictionary mapping model names to colors
    figsize : tuple
        Figure size
    save : bool
        Whether to save the figure
    save_name : str, optional
        Filename for saving
    huxt_id : int, optional
        HUXt run ID
    """
    from scipy.stats import norm, lognorm
    from matplotlib.lines import Line2D
    
    n_models = len(results_dict)
    
    # Default colors
    if model_colors is None:
        default_colors = ['steelblue', 'forestgreen', 'coral', 'purple', 'orange', 'crimson']
        model_colors = {name: default_colors[i % len(default_colors)] 
                       for i, name in enumerate(results_dict.keys())}
    
    # Create subplots
    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=True)
    if n_models == 1:
        axes = [axes]
    
    # Get ensemble predictions
    all_preds = []
    for model_name, results in results_dict.items():
        if 'ensemble_predictions' not in results:
            raise ValueError(f"Results for {model_name} do not contain 'ensemble_predictions'")
        preds = results['ensemble_predictions'][test_idx]
        all_preds.append(preds)
    
    x_min = min(p.min() for p in all_preds)
    x_max = max(p.max() for p in all_preds)
    x_range = np.linspace(max(0, x_min - 1), x_max + 1, 200)
    
    # Plot each model
    for idx, (model_name, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        ensemble_preds = all_preds[idx]
        
        # Histogram
        ax.hist(ensemble_preds, bins=30, alpha=0.7, 
               color=model_colors[model_name], 
               edgecolor='black', density=True,
               label='Ensemble Predictions')
        
        # Distributions
        if 'mu' in results:
            mu_pred = results['mu'][test_idx]
            sigma_pred = results['sigma'][test_idx]
            normal_pdf = norm.pdf(x_range, loc=mu_pred, scale=sigma_pred)
            ax.plot(x_range, normal_pdf, 'g--', lw=2)
        
        if 'log_mu' in results:
            log_mu_pred = results['log_mu'][test_idx]
            log_sigma_pred = results['log_sigma'][test_idx]
            lognormal_pdf = lognorm.pdf(x_range, s=log_sigma_pred, scale=np.exp(log_mu_pred))
            ax.plot(x_range, lognormal_pdf, 'orange', linestyle='--', lw=2)
        
        # Markers
        max_hp30_colour = 'olive'
        ax.axvline(max_target, color=max_hp30_colour, linestyle='--', lw=2.5, zorder=100)
        
        if 'y_pred_weighted_mean' in results:
            ax.axvline(results['y_pred_weighted_mean'][test_idx], 
                      color='purple', linestyle='-.', lw=2, alpha=0.8)
        
        # Labels
        ax.set_xlabel('Predicted Hp30', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Density', fontsize=12)
        ax.set_title(model_name, fontsize=14, fontweight='bold')
        ax.set_xlim(x_range[0], x_range[-1])
        ax.grid(alpha=0.3, ls='--')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # Shared legend
    legend_elements = [
        Line2D([0], [0], color='g', linestyle='--', lw=2, label='Normal Distribution'),
        Line2D([0], [0], color='orange', linestyle='--', lw=2, label='LogNormal Distribution'),
        Line2D([0], [0], color=max_hp30_colour, linestyle='--', lw=2.5, label='True Observed Value'),
        Line2D([0], [0], color='purple', linestyle='-.', lw=2, alpha=0.8, label='Weighted Mean Forecast'),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=4, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save and save_name:
        save_figure(save_name, subfolder='model_comparison', huxt_id=huxt_id)
    
    plt.show()


def plot_ensemble_comparison_from_files(
    results_files: Dict[str, Path],
    test_idx: int,
    model_colors: Optional[Dict[str, str]] = None,
    save=False,
    save_name=None,
    huxt_id=None
):
    """
    Load results from files and plot ensemble comparison.
    
    Parameters
    ----------
    results_files : dict
        Dictionary mapping model names to result file paths
    test_idx : int
        Index to access forecasts
    model_colors : dict, optional
        Model name to color mapping
    save : bool
        Whether to save the figure
    save_name : str, optional
        Filename for saving
    huxt_id : int, optional
        HUXt run ID
    """
    from storm_regression.results_io import load_results
    
    # Load all results
    results_dict = {}
    max_target = None
    
    for model_name, file_path in results_files.items():
        results, config, _ = load_results(file_path)
        results_dict[model_name] = results
        
        if max_target is None:
            max_target = results['y_test'][test_idx]
    
    # Plot
    plot_ensemble_comparison(results_dict, test_idx, max_target, model_colors,
                            save=save, save_name=save_name, huxt_id=huxt_id)


def compare_top_n_storms(
    results_files: Dict[str, Path],
    n: int = 5,
    sort_by: str = 'y_test',
    model_colors: Optional[Dict[str, str]] = None,
    save=False,
    save_prefix=None,
    huxt_id=None
):
    """
    Compare ensemble forecasts for top N storms across multiple models.
    
    Parameters
    ----------
    results_files : dict
        Dictionary mapping model names to result file paths
    n : int
        Number of top storms to plot
    sort_by : str
        What to sort by
    model_colors : dict, optional
        Model colors
    save : bool
        Whether to save figures
    save_prefix : str, optional
        Prefix for saved filenames
    huxt_id : int, optional
        HUXt run ID
    """
    from storm_regression.results_io import load_results
    
    # Load first model for ranking
    first_model = list(results_files.keys())[0]
    results_first, _, _ = load_results(results_files[first_model])
    
    # Get top N
    sort_values = results_first[sort_by].squeeze()
    test_top_n = np.argsort(sort_values)[-n:][::-1]
    
    # Load all models
    results_dict = {}
    for model_name, file_path in results_files.items():
        results, _, _ = load_results(file_path)
        results_dict[model_name] = results
    
    # Plot each storm
    for i, test_idx in enumerate(test_top_n):
        max_target = results_first['y_test'][test_idx]
        
        print(f"\n{'='*80}")
        print(f"Storm #{i+1} - True Hp30: {max_target:.2f}")
        print(f"{'='*80}")
        
        save_name = f"{save_prefix}_storm{i+1}" if save and save_prefix else None
        
        plot_ensemble_comparison(results_dict, test_idx, max_target, model_colors,
                                save=save, save_name=save_name, huxt_id=huxt_id)


def plot_distribution_params_joint(
    results_file: Path,
    aggregator: str = 'weibull_median',
    plot_type: str = 'scatter',
    apply_constraints: bool = True,
    figsize: tuple = (10, 10), 
    save=False, 
    save_name=None,
    huxt_id=None,
):
    """
    Plot joint distribution of fitted distribution parameters.
    
    Parameters
    ----------
    results_file : Path
        Path to saved results pickle file
    aggregator : str
        Which distribution to plot. Options:
        - 'weibull_median': Plot lambda vs k
        - 'lognormal_median': Plot exp(log_mu) vs log_sigma
        - 'normal_median': Plot mu vs sigma
    plot_type : str
        Either 'scatter' or 'heatmap'
    apply_constraints : bool
        If True, refit distribution parameters using constrained predictions.
    figsize : tuple
        Figure size (width, height)
    
    Examples
    --------
    >>> results_file = results_folder / 'results_seed42_..._LinearRegression.pkl'
    >>> plot_distribution_params_joint(results_file, aggregator='weibull_median')
    """
    from storm_regression.results_io import load_results
    from storm_regression.training_pipeline import fit_distribution_parameters, constrain_predictions
    
    # Load results
    results, config, _ = load_results(results_file)
    
    # Map aggregator to parameter names
    param_map = {
        'weibull_median': {
            'param1': 'lambda',
            'param2': 'k',
            'label1': 'λ (scale)',
            'label2': 'k (shape)',
            'color1': 'steelblue',
            'color2': 'coral',
            'log_param2': True,
            'transform_param1': None,
        },
        'lognormal_median': {
            'param1': 'log_mu',
            'param2': 'log_sigma',
            'label1': 'exp(μ) (median Hp30)',
            'label2': 'σ (log-space std)',
            'color1': 'orange',
            'color2': 'purple',
            'log_param2': False,
            'transform_param1': 'exp',
        },
        'normal_median': {
            'param1': 'mu',
            'param2': 'sigma',
            'label1': 'μ (mean)',
            'label2': 'σ (std dev)',
            'color1': 'green',
            'color2': 'red',
            'log_param2': False,
            'transform_param1': None,
        }
    }
    
    # Normalize aggregator name
    aggregator_lower = aggregator.lower().replace(' ', '_').replace('-', '_')
    
    if aggregator_lower not in param_map:
        raise ValueError(f"aggregator must be one of {list(param_map.keys())}, got '{aggregator}'")
    
    params = param_map[aggregator_lower]
    
    # Get constraint method from config
    constraint_method = config.get('constraint_method', None)
    model_name = config.get('model_name', 'Unknown')
    
    # Decide whether to refit with constraints
    if apply_constraints and constraint_method and 'ensemble_predictions' in results:
        logger.info(f"Refitting {aggregator} distribution with '{constraint_method}' constraints applied")
        
        ensemble_preds = results['ensemble_predictions']
        ensemble_preds_constrained = constrain_predictions(ensemble_preds, method=constraint_method)
        
        weibull_params, normal_params, lognormal_params = fit_distribution_parameters(ensemble_preds_constrained)
        
        if aggregator_lower == 'weibull_median':
            param1_data = weibull_params['lambda']
            param2_data = weibull_params['k']
        elif aggregator_lower == 'lognormal_median':
            param1_data = lognormal_params['log_mu']
            param2_data = lognormal_params['log_sigma']
        elif aggregator_lower == 'normal_median':
            param1_data = normal_params['mu']
            param2_data = normal_params['sigma']
        
        title_suffix = f" (with '{constraint_method}' constraints)"
    else:
        if params['param1'] not in results or params['param2'] not in results:
            raise ValueError(f"Results do not contain '{params['param1']}' or '{params['param2']}'. "
                            f"Available keys: {list(results.keys())}")
        
        param1_data = results[params['param1']]
        param2_data = results[params['param2']]
        title_suffix = " (original, unconstrained)" if constraint_method else ""
    
    # Apply transformation if needed
    if params['transform_param1'] == 'exp':
        param1_data_display = np.exp(param1_data)
    else:
        param1_data_display = param1_data
    
    # Remove NaN values
    valid_mask = ~(np.isnan(param1_data_display) | np.isnan(param2_data))
    param1_data_display = param1_data_display[valid_mask]
    param2_data = param2_data[valid_mask]
    
    if len(param1_data_display) == 0:
        raise ValueError("No valid data points after removing NaNs")
    
    # Calculate observed value statistics
    y_true = results['y_test']
    mean_observed = np.mean(y_true)
    median_observed = np.median(y_true)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.05, wspace=0.05, 
                          height_ratios=[1, 3, 0.2], width_ratios=[0.2, 3, 1])
    
    # Main joint distribution (center)
    ax_joint = fig.add_subplot(gs[1, 1])
    
    if plot_type == 'scatter':
        ax_joint.scatter(param1_data_display, param2_data, alpha=0.5, s=20, 
                        color=params['color1'], edgecolors='none')
    elif plot_type == 'heatmap':
        if params['log_param2']:
            bins_p2 = np.logspace(np.log10(param2_data.min()), 
                                  np.log10(param2_data.max()), 30)
        else:
            bins_p2 = 30
        
        h = ax_joint.hist2d(param1_data_display, param2_data, 
                           bins=[30, bins_p2], 
                           cmap='YlOrRd')
        
        bbox = ax_joint.get_position()
        cax = fig.add_axes([bbox.x1 + 0.02, bbox.y0, 0.02, bbox.height])
        plt.colorbar(h[3], cax=cax, label='Count')
    else:
        raise ValueError(f"plot_type must be 'scatter' or 'heatmap', got '{plot_type}'")
    
    # Add reference lines for observed values
    ax_joint.axvline(mean_observed, color='darkred', linestyle='--', 
                    linewidth=2, alpha=0.7, label=f'Mean Observed')
    ax_joint.axvline(median_observed, color='darkgreen', linestyle=':', 
                    linewidth=2, alpha=0.7, label=f'Median Observed')
    
    ax_joint.set_xlabel(params['label1'], fontsize=12)
    ax_joint.set_ylabel(params['label2'], fontsize=12)
    
    if params['log_param2']:
        ax_joint.set_yscale('log')
    
    ax_joint.grid(True, alpha=0.3)
    ax_joint.legend(loc='upper right', fontsize=9)
    
    # Param1 distribution (top)
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_joint)
    ax_top.hist(param1_data_display, bins=30, density=False, alpha=0.7, 
               color=params['color1'], edgecolor='black')
    ax_top.set_ylabel('Counts', fontsize=10)
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(True, alpha=0.3, axis='y')
    
    mean_p1 = np.mean(param1_data_display)
    median_p1 = np.median(param1_data_display)
    ax_top.axvline(mean_p1, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_p1:.2f}')
    ax_top.axvline(median_p1, color='green', linestyle='--', linewidth=2, 
                  label=f'Median: {median_p1:.2f}')
    ax_top.axvline(mean_observed, color='darkred', linestyle='--', 
                  linewidth=2, alpha=0.7, label=f'Obs Mean: {mean_observed:.2f}')
    ax_top.axvline(median_observed, color='darkgreen', linestyle=':', 
                  linewidth=2, alpha=0.7, label=f'Obs Median: {median_observed:.2f}')
    
    ax_top.legend(fontsize=8, loc='upper right')
    
    # Param2 distribution (right)
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_joint)
    
    if params['log_param2']:
        bins_p2 = np.logspace(np.log10(param2_data.min()), 
                             np.log10(param2_data.max()), 30)
    else:
        bins_p2 = 30
    
    ax_right.hist(param2_data, bins=bins_p2, density=False, alpha=0.7, 
                 color=params['color2'], edgecolor='black', orientation='horizontal')
    ax_right.set_xlabel('Counts', fontsize=10)
    ax_right.tick_params(labelleft=False)
    ax_right.grid(True, alpha=0.3, axis='x')
    
    mean_p2 = np.mean(param2_data)
    median_p2 = np.median(param2_data)
    ax_right.axhline(mean_p2, color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {mean_p2:.2f}')
    ax_right.axhline(median_p2, color='green', linestyle='--', linewidth=2, 
                    label=f'Median: {median_p2:.2f}')
    ax_right.legend(fontsize=8, loc='upper right')
    
    # Title
    title = f"{model_name} - {aggregator.replace('_', ' ').title()}{title_suffix}"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    if save and save_name:
        save_figure(save_name, subfolder='model_comparison', huxt_id=huxt_id)
    
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Distribution: {aggregator.replace('_', ' ').title()}")
    print(f"Model: {model_name}")
    if constraint_method:
        print(f"Constraint Applied: {constraint_method}")
    print(f"{'='*60}")
    print(f"\nObserved Values:")
    print(f"  Mean: {mean_observed:.3f} nT")
    print(f"  Median: {median_observed:.3f} nT")
    print(f"\n{params['label1']}:")
    print(f"  Mean: {mean_p1:.3f}, Median: {median_p1:.3f}")
    print(f"  Std: {np.std(param1_data_display):.3f}")
    print(f"  Range: [{np.min(param1_data_display):.3f}, {np.max(param1_data_display):.3f}]")
    print(f"\n{params['label2']}:")
    print(f"  Mean: {mean_p2:.3f}, Median: {median_p2:.3f}")
    print(f"  Std: {np.std(param2_data):.3f}")
    print(f"  Range: [{np.min(param2_data):.3f}, {np.max(param2_data):.3f}]")
    print(f"\nCorrelation: {np.corrcoef(param1_data_display, param2_data)[0, 1]:.3f}")
    print(f"Valid samples: {len(param1_data_display)} / {len(results[params['param1']])}")
    
    # Bias analysis
    pred_key = f'y_pred_{aggregator}'
    if pred_key in results:
        bias = np.mean(results[pred_key]) - mean_observed
        print(f"\nBias (predicted - observed mean): {bias:+.3f} nT")
        if bias > 0:
            print("  → Model tends to OVER-predict")
        else:
            print("  → Model tends to UNDER-predict")
    
    print(f"{'='*60}\n")


def compare_distribution_params_across_models(
    results_files: Dict[str, Path],
    aggregator: str = 'weibull_median',
    _max: Optional[float] = None,
    plot_type: str = 'scatter',
    figsize: tuple = (15, 5),
    apply_constraints: bool = True, 
    save=False,
    save_name=None,
    huxt_id=None,
):
    """
    Compare distribution parameter joint distributions across multiple models.
    
    Parameters
    ----------
    results_files : dict
        Dictionary mapping model names to result file paths
    aggregator : str
        Which distribution to plot ('weibull_median', 'lognormal_median', 'normal_median')
    _max : float, optional
        Maximum value to clip first parameter to
    plot_type : str
        Either 'scatter' or 'heatmap'
    figsize : tuple
        Figure size (width, height)
    apply_constraints : bool
        If True, apply the same prediction constraints used during training.
        This ensures the distribution parameters reflect the actual constrained predictions.
    
    Examples
    --------
    >>> files = {
    ...     'LinearRegression': results_folder / 'results_..._LinearRegression.pkl',
    ...     'Ridge': results_folder / 'results_..._Ridge.pkl'
    ... }
    >>> compare_distribution_params_across_models(files, aggregator='weibull_median')
    """
    from storm_regression.results_io import load_results
    from storm_regression.training_pipeline import fit_distribution_parameters, constrain_predictions
    
    n_models = len(results_files)
    
    # Map aggregator to parameter names
    param_map = {
        'weibull_median': {
            'param1': 'lambda', 'param2': 'k',
            'label1': 'λ (scale)', 'label2': 'k (shape)',
            'log_param2': True,
            'transform_param1': None,
        },
        'lognormal_median': {
            'param1': 'log_mu', 'param2': 'log_sigma',
            'label1': 'exp(μ) (median Hp30)', 'label2': 'σ (log-space)',
            'log_param2': False,
            'transform_param1': 'exp',
        },
        'normal_median': {
            'param1': 'mu', 'param2': 'sigma',
            'label1': 'μ (mean)', 'label2': 'σ (std)',
            'log_param2': False,
            'transform_param1': None,
        }
    }
    
    aggregator_lower = aggregator.lower().replace(' ', '_').replace('-', '_')
    if aggregator_lower not in param_map:
        raise ValueError(f"aggregator must be one of {list(param_map.keys())}")
    
    params = param_map[aggregator_lower]
    
    # Create subplots
    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=True, sharex=True)
    if n_models == 1:
        axes = [axes]
    
    # Calculate observed value for reference (from first model)
    first_results, _, _ = load_results(list(results_files.values())[0])
    y_true = first_results['y_test']
    mean_observed = np.mean(y_true)
    median_observed = np.median(y_true)
    
    # Plot each model
    for idx, (model_name, results_file) in enumerate(results_files.items()):
        ax = axes[idx]
        
        # Load results
        results, config, _ = load_results(results_file)
        
        # Get constraint method
        constraint_method = config.get('constraint_method', None)
        
        # Refit with constraints if applicable
        if apply_constraints and constraint_method and 'ensemble_predictions' in results:
            logger.info(f"Refitting {aggregator} for {model_name} with '{constraint_method}' constraints")
            
            ensemble_preds = results['ensemble_predictions']
            ensemble_preds_constrained = constrain_predictions(ensemble_preds, method=constraint_method)
            
            weibull_params, normal_params, lognormal_params = fit_distribution_parameters(ensemble_preds_constrained)
            
            if aggregator_lower == 'weibull_median':
                param1_data = weibull_params['lambda']
                param2_data = weibull_params['k']
            elif aggregator_lower == 'lognormal_median':
                param1_data = lognormal_params['log_mu']
                param2_data = lognormal_params['log_sigma']
            elif aggregator_lower == 'normal_median':
                param1_data = normal_params['mu']
                param2_data = normal_params['sigma']
        else:
            # Use saved parameters
            param1_data = results[params['param1']]
            param2_data = results[params['param2']]
        
        # Apply transformation if needed
        if params['transform_param1'] == 'exp':
            param1_data_display = np.exp(param1_data)
        else:
            param1_data_display = param1_data
        
        # Apply clipping if requested
        if _max is not None:
            param1_data_display = np.clip(param1_data_display, None, _max)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(param1_data_display) | np.isnan(param2_data))
        param1_data_display = param1_data_display[valid_mask]
        param2_data = param2_data[valid_mask]
        
        # Plot
        if plot_type == 'scatter':
            ax.scatter(param1_data_display, param2_data, alpha=0.5, s=20, edgecolors='none')
        elif plot_type == 'heatmap':
            bins_p2 = np.logspace(np.log10(param2_data.min()), 
                                 np.log10(param2_data.max()), 30) if params['log_param2'] else 30
            ax.hist2d(param1_data_display, param2_data, bins=[30, bins_p2], cmap='YlOrRd')
        
        # Add observed value reference line
        ax.axvline(mean_observed, color='darkred', linestyle='--', 
                  linewidth=1.5, alpha=0.6, label='Obs Mean')
        ax.axvline(median_observed, color='darkgreen', linestyle=':', 
                  linewidth=1.5, alpha=0.6, label='Obs Median')
        
        # Labels
        ax.set_xlabel(params['label1'], fontsize=11)
        if idx == 0:
            ax.set_ylabel(params['label2'], fontsize=11)
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        
        if params['log_param2']:
            ax.set_yscale('log')
        
        ax.grid(True, alpha=0.3)
        
        # Stats annotation
        corr = np.corrcoef(param1_data_display, param2_data)[0, 1]
        mean_p1 = np.mean(param1_data_display)
        bias = mean_p1 - mean_observed
        bias_pct = (bias / mean_observed) * 100 if mean_observed != 0 else np.nan
        
        stats_text = f'r={corr:.2f}\nn={len(param1_data_display)}\n'
        stats_text += f'Bias: {bias:+.2f}\n({bias_pct:+.1f}%)'
        
        ax.text(0.05, 0.95, stats_text, 
               transform=ax.transAxes, va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add legend only to first plot
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle(f"{aggregator.replace('_', ' ').title()} - Parameter Comparison Across Models", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save and save_name:
        save_figure(save_name, subfolder='model_comparison', huxt_id=huxt_id)
    
    plt.show()


def plot_comparative_case_study(
    forecasting_dataset,
    all_results: Dict[str, dict],
    all_configs: Dict[str, dict],
    window_position: int,
    test_idx_map: Dict[str, int],
    thresholds=None,
    save: bool = False,
    save_name: Optional[str] = None,
    huxt_id: Optional[int] = None,
    title_suffix: Optional[str] = None,
    show_components: bool = False,
):
    """
    Compare multiple models on one forecast window.

    Each model's predictive distribution is detected per-model: if the result has
    mu_m/sigma_m/alpha it is treated as a K-component LogNormal mixture; otherwise as a
    single LogNormal (log_mu/log_sigma). The right-hand panel overlays each model's
    *combined* predictive pdf. Set show_components=True to also dash in each mixture's
    individual weighted components (useful when inspecting one model's bimodality, noisy
    when several models are shown).
    """

    # ── Per-model predictive helpers (mixture-aware) ─────────────────────────
    def _model_dist(results, test_idx):
        """Return a dict of callables/values describing this model's forecast."""
        if all(k in results for k in ('mu_m', 'sigma_m', 'alpha')):
            mu_k    = np.atleast_1d(results['mu_m'][test_idx]).astype(float)
            sigma_k = np.atleast_1d(results['sigma_m'][test_idx]).astype(float)
            w_k     = np.atleast_1d(results['alpha'][test_idx]).astype(float)
            K = len(mu_k)

            def pdf(y):
                y = np.asarray(y, dtype=float)
                return sum(w_k[k] * lognorm.pdf(y, s=sigma_k[k], scale=np.exp(mu_k[k]))
                           for k in range(K))

            def exceeds(thresh):
                thresh = np.atleast_1d(np.asarray(thresh, dtype=float))
                out = np.zeros_like(thresh, dtype=float)
                for k in range(K):
                    out += w_k[k] * (1.0 - lognorm.cdf(thresh, s=sigma_k[k],
                                                       scale=np.exp(mu_k[k])))
                return out

            _grid = np.linspace(1e-3, 30, 4000)
            _cdf = sum(w_k[k] * lognorm.cdf(_grid, s=sigma_k[k], scale=np.exp(mu_k[k]))
                       for k in range(K))
            median = float(np.interp(0.5, _cdf, _grid))

            return {
                'is_mixture': True, 'K': K, 'mu_k': mu_k, 'sigma_k': sigma_k, 'w_k': w_k,
                'pdf': pdf, 'exceeds': exceeds, 'median': median,
                'label_params': f'K={K}, med={median:.2f}',
            }
        else:
            log_mu    = float(results['log_mu'][test_idx])
            log_sigma = float(results['log_sigma'][test_idx])
            median    = float(np.exp(log_mu))

            def pdf(y):
                return lognorm.pdf(y, s=log_sigma, scale=np.exp(log_mu))

            def exceeds(thresh):
                thresh = np.atleast_1d(np.asarray(thresh, dtype=float))
                return np.asarray(lognormal_exceeds(log_mu, log_sigma, thresh)).ravel()

            return {
                'is_mixture': False, 'pdf': pdf, 'exceeds': exceeds, 'median': median,
                'label_params': f'μ={log_mu:.2f}, σ={log_sigma:.2f}, med={median:.2f}',
            }

    # ── Distinctive label computation ────────────────────────────────────────
    def get_distinctive_labels(names):
        if len(names) <= 1:
            return {name: name for name in names}
        tokenized = [name.split('_') for name in names]
        all_tokens = set(tokenized[0])
        for tokens in tokenized[1:]:
            all_tokens &= set(tokens)
        result = {}
        for name, tokens in zip(names, tokenized):
            distinctive = [t for t in tokens if t not in all_tokens]
            result[name] = '_'.join(distinctive) if distinctive else name
        return result

    model_names = list(all_results.keys())
    label_map   = get_distinctive_labels(model_names)

    # Build per-model distribution descriptors once.
    model_dist = {m: _model_dist(all_results[m], test_idx_map[m]) for m in model_names}

    fontsize = 16
    ylim     = 13
    ylen     = ylim * 3 + 1
    if thresholds is None:
        thresholds = np.linspace(1/6, ylim + 1/6, ylen)

    print(f"Window position: {window_position}")
    window = forecasting_dataset[window_position]

    v            = window['v']
    omni_sw      = window['omni_sw']
    omni         = window['omni_sw_plotting']
    omni_var     = window['omni_plotting']
    target       = window['target_plotting']
    max_target   = float(window['max_target'])
    center_idx   = window['center_idx']
    window_label = window['window_label']

    T0      = forecasting_dataset.df.index[center_idx]
    F_start = forecasting_dataset.df.index[center_idx + forecasting_dataset.lead_time]

    n_steps    = forecasting_dataset.max_offset - forecasting_dataset.min_offset
    timestamps = pd.date_range(
        start=T0 + timedelta(minutes=30 * forecasting_dataset.min_offset),
        periods=n_steps, freq='30min',
    )
    print(f'T0: {T0}')

    # ── Select anchor model: closest median to true max_target ───────────────
    best_model = None
    best_error = np.inf
    for model_name in model_names:
        median = model_dist[model_name]['median']
        error  = abs(median - max_target)
        if error < best_error:
            best_error = error
            best_model = model_name

    anchor = model_dist[best_model]
    probs            = anchor['exceeds'](thresholds)                       # (n_thresh,)
    probs            = np.tile(probs[:, None], (1, forecasting_dataset.forecast_steps))
    p_exceeds_anchor = anchor['exceeds']([max_target])
    best_label       = label_map[best_model]

    print(f"Anchor model for contours: {best_label} "
          f"(median={anchor['median']:.2f}, error={best_error:.2f})")

    # ── Filter and select ensemble members using best model's config ──────────
    best_config      = all_configs[best_model]
    selection_method = best_config.get('ensemble_selection_method', 'snap')

    v_display = v.copy()
    if best_config.get('filter_ensemble', False):
        n_keep    = best_config.get('n_ensemble_keep', 50)
        v_input   = v_display[:omni_sw.shape[0], :]
        mae       = np.mean(np.abs(v_input - omni_sw[:, None]), axis=0)
        best_idx  = np.argsort(mae)[:n_keep]
        v_display = v_display[:, best_idx]
        print(f"Filtered ensemble to {n_keep} members using best model's config")

    if best_config.get('filter_ensemble', False):
        n_keep    = best_config.get('n_ensemble_keep', 50)
        v_input   = v[:omni_sw.shape[0], :]
        mae       = np.mean(np.abs(v_input - omni_sw[:, None]), axis=0)
        kept_idx  = set(np.argsort(mae)[:n_keep].tolist())
    else:
        kept_idx  = set(range(v.shape[1]))
    discarded_idx = set(range(v.shape[1])) - kept_idx

    # ── Percentile computation from v_display ─────────────────────────────────
    percentile_member_indices = {}
    percentile_bands          = []
    median_p                  = None

    if 'mlp_ensemble_percentiles' in best_config:
        percentiles = sorted(best_config['mlp_ensemble_percentiles'])
        n_members   = v_display.shape[1]
        if selection_method == 'snap':
            member_mean_v = v_display.mean(axis=0)
            rank_order    = np.argsort(member_mean_v)
            for p in percentiles:
                rank_idx = min(int(np.floor(p / 100 * n_members)), n_members - 1)
                percentile_member_indices[p] = int(rank_order[rank_idx])
        ps       = percentiles
        median_p = ps[len(ps) // 2]
        for i in range(len(ps) // 2):
            percentile_bands.append((ps[i], ps[-(i + 1)]))

    # ── ICME / SIR periods ───────────────────────────────────────────────────
    icme_periods = []
    sir_periods  = []
    if 'ICME_flag' in forecasting_dataset.df.columns:
        window_start_idx = center_idx + forecasting_dataset.min_offset
        window_end_idx   = center_idx + forecasting_dataset.max_offset + 1
        window_df        = forecasting_dataset.df.iloc[window_start_idx:window_end_idx]
        icme_flags       = window_df['ICME_flag'].values
        sir_flags        = window_df['SIR_flag'].values
        in_icme = False
        for i in range(len(icme_flags)):
            if icme_flags[i] and not in_icme:
                icme_start = timestamps[i]; in_icme = True
            elif not icme_flags[i] and in_icme:
                icme_periods.append((icme_start, timestamps[i - 1])); in_icme = False
        if in_icme:
            icme_periods.append((icme_start, timestamps[-1]))
        in_sir = False
        for i in range(len(sir_flags)):
            if sir_flags[i] and not in_sir:
                sir_start = timestamps[i]; in_sir = True
            elif not sir_flags[i] and in_sir:
                sir_periods.append((sir_start, timestamps[i - 1])); in_sir = False
        if in_sir:
            sir_periods.append((sir_start, timestamps[-1]))

    # ── Figure layout ────────────────────────────────────────────────────────
    from matplotlib.gridspec import GridSpec
    import matplotlib.cm as cm

    fig = plt.figure(figsize=(18, 10))
    gs  = GridSpec(3, 2, figure=fig, height_ratios=[1.5, 2, 2.5],
                   width_ratios=[3, 1], hspace=0.05, wspace=0.05)
    ax_bz   = fig.add_subplot(gs[0, 0])
    ax_v    = fig.add_subplot(gs[1, 0], sharex=ax_bz)
    ax_hp30 = fig.add_subplot(gs[2, 0], sharex=ax_bz)
    ax_hist = fig.add_subplot(gs[:, 1])

    xF = pd.date_range(start=F_start + pd.Timedelta(minutes=15),
                       periods=forecasting_dataset.forecast_steps, freq='30min')
    xG1 = pd.date_range(start=T0 + timedelta(minutes=30 * forecasting_dataset.min_offset),
                        periods=forecasting_dataset.lead_time - forecasting_dataset.min_offset + 1,
                        freq='30min')
    xG = pd.date_range(start=T0 + timedelta(minutes=30 * forecasting_dataset.min_offset),
                       periods=forecasting_dataset.max_offset - forecasting_dataset.min_offset + 1,
                       freq='30min')

    def add_event_bars(ax, y_position):
        for i, (start, end) in enumerate(icme_periods):
            cap = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
            ax.plot([start, end], [y_position, y_position], color='darkred', lw=3,
                    solid_capstyle='butt', label='ICME' if i == 0 else '', alpha=0.8)
            for x in (start, end):
                ax.plot([x, x], [y_position - cap, y_position + cap],
                        color='darkred', lw=3, solid_capstyle='butt', alpha=0.8)
        if not icme_periods:
            y_off = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
            for i, (start, end) in enumerate(sir_periods):
                cap = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                ax.plot([start, end], [y_position - y_off, y_position - y_off],
                        color='darkblue', lw=3, solid_capstyle='butt',
                        label='SIR' if i == 0 else '', alpha=0.8)
                for x in (start, end):
                    ax.plot([x, x], [y_position - y_off - cap, y_position - y_off + cap],
                            color='darkblue', lw=3, solid_capstyle='butt', alpha=0.8)

    # ===== PANEL 1: Bz_GSM =====
    if omni_var.ndim == 2:
        ax_bz.plot(timestamps, omni_var[:, 0], lw=1, color='red', label='OMNI Bz_GSM')
    else:
        ax_bz.plot(timestamps, omni_var, lw=1, color='red', label='OMNI Bz_GSM')
    ax_bz.axvline(T0, color='black', linestyle='--', label='T0')
    ax_bz.axvspan(xF[0] - timedelta(minutes=15), xF[-1] + timedelta(minutes=15),
                  color='grey', alpha=0.1, label='Forecast Window')
    add_event_bars(ax_bz, ax_bz.get_ylim()[1] * 0.95)
    ax_bz.set_ylabel('Bz_GSM (nT)', fontsize=fontsize)
    ax_bz.legend(loc='upper right')
    ax_bz.set_xlim(timestamps[0], timestamps[-1])
    title = f"Event type: {window_label}"
    if title_suffix:
        title += title_suffix
    ax_bz.set_title(title, fontsize=fontsize + 2)

    # ===== PANEL 2: Solar Wind Velocity =====
    for i in sorted(discarded_idx):
        ax_v.plot(timestamps, v[:, i], color='darkorange', lw=0.8, alpha=0.3,
                  label='Discarded' if i == min(discarded_idx, default=-1) else None)
    for i in sorted(kept_idx):
        ax_v.plot(timestamps, v[:, i], color='steelblue', lw=0.8, alpha=0.4,
                  label='Kept' if i == min(kept_idx, default=-1) else None)

    if selection_method == 'snap':
        if percentile_bands:
            lo_p_outer, hi_p_outer = percentile_bands[0]
            lo_v_outer = v_display[:, percentile_member_indices[lo_p_outer]]
            hi_v_outer = v_display[:, percentile_member_indices[hi_p_outer]]
            ax_v.fill_between(timestamps, lo_v_outer, hi_v_outer, alpha=0.15,
                              color='steelblue', label=f'p{lo_p_outer}-p{hi_p_outer} range')
        for j, (lo_p, hi_p) in enumerate(percentile_bands):
            lo_v = v_display[:, percentile_member_indices[lo_p]]
            hi_v = v_display[:, percentile_member_indices[hi_p]]
            lw = 2.0 if j == 0 else 1.5
            ls = '-' if j == 0 else '--'
            ax_v.plot(timestamps, lo_v, color='steelblue', lw=lw, alpha=0.9, linestyle=ls, label=f'p{lo_p}')
            ax_v.plot(timestamps, hi_v, color='steelblue', lw=lw, alpha=0.9, linestyle=ls, label=f'p{hi_p}')
        if median_p is not None:
            ax_v.plot(timestamps, v_display[:, percentile_member_indices[median_p]],
                      color='black', lw=2.5, linestyle='-', alpha=1.0, zorder=5, label=f'p{median_p} (median)')
    elif selection_method == 'per_timestep':
        if percentile_bands:
            lo_p_outer, hi_p_outer = percentile_bands[0]
            lo_v_outer = np.percentile(v_display, lo_p_outer, axis=1, method='nearest')
            hi_v_outer = np.percentile(v_display, hi_p_outer, axis=1, method='nearest')
            ax_v.fill_between(timestamps, lo_v_outer, hi_v_outer, alpha=0.15,
                              color='steelblue', label=f'p{lo_p_outer}-p{hi_p_outer} range')
        for j, (lo_p, hi_p) in enumerate(percentile_bands):
            lo_v = np.percentile(v_display, lo_p, axis=1, method='nearest')
            hi_v = np.percentile(v_display, hi_p, axis=1, method='nearest')
            lw = 2.0 if j == 0 else 1.5
            ls = '-' if j == 0 else '--'
            ax_v.plot(timestamps, lo_v, color='steelblue', lw=lw, alpha=0.9, linestyle=ls, label=f'p{lo_p}')
            ax_v.plot(timestamps, hi_v, color='steelblue', lw=lw, alpha=0.9, linestyle=ls, label=f'p{hi_p}')
        if median_p is not None:
            median_v = np.percentile(v_display, median_p, axis=1, method='nearest')
            ax_v.plot(timestamps, median_v, color='black', lw=2.5, linestyle='-',
                      alpha=1.0, zorder=5, label=f'p{median_p} (median)')

    ax_v.plot(timestamps, omni, 'k--', lw=1.5, label='OMNI V_sw')
    ax_v.axvline(T0, color='black', linestyle='--', label='T0')
    ax_v.axvspan(xF[0] - timedelta(minutes=15), xF[-1] + timedelta(minutes=15), color='grey', alpha=0.1)
    add_event_bars(ax_v, ax_v.get_ylim()[1] * 0.95)
    ax_v.set_ylabel('v (km/s)', fontsize=fontsize)
    ax_v.legend(loc='upper right', fontsize=8)
    ax_v.set_xlim(timestamps[0], timestamps[-1])

    # ===== PANEL 3: Hp30 with probability contours (anchor model) =====
    Edges      = [0, 5, 6, 7, 8, 9, ylim]
    nbands     = len(Edges) - 1
    colors_arr = np.arange(1, nbands + 1)[:, None] + 1
    Z1         = np.tile(colors_arr, (1, len(xG) - 1))
    ax_hp30.pcolormesh(xG, Edges, Z1, shading='flat', cmap='Blues', alpha=0.3)
    ax_hp30.contourf(xF, thresholds, probs, levels=20, cmap='Reds', alpha=0.0)
    contour_lines = ax_hp30.contour(xF, thresholds, probs, levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                                    colors='darkred', linewidths=1)
    ax_hp30.clabel(contour_lines, inline=True, fontsize=8, fmt='%0.1f', rightside_up=True)
    ax_hp30.text(0.01, 0.99, f'Contours: {best_label} *', transform=ax_hp30.transAxes,
                 fontsize=9, va='top', ha='left', color='darkred',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    G_labels = [5, 6, 7, 8, 9, 10]
    g_names  = ['G1', 'G2', 'G3', 'G4', 'G5']
    for i in range(len(G_labels) - 1):
        y_mid = (G_labels[i] + G_labels[i + 1]) / 2
        ax_hp30.text(xG1[0] + pd.Timedelta(minutes=120), y_mid, g_names[i],
                     va='center', ha='right', fontsize=10, color='black')
        ax_hp30.hlines(G_labels[i], timestamps[0], timestamps[-1], color='lightblue', lw=0.5)

    max_hp30_colour = 'olive'
    ax_hp30.plot(timestamps, target, lw=1, label='Hp30')
    ax_hp30.plot([xF[0], xF[-1]], [max_target, max_target], color=max_hp30_colour,
                 linestyle='--', label='Max Hp30')
    ax_hp30.text(xF[0] - timedelta(minutes=10 * 60), max_target - 0.5,
                 f'p(X>={max_target:.2f})={p_exceeds_anchor[0]:.2f}',
                 verticalalignment='bottom', color=max_hp30_colour)
    ax_hp30.axvline(T0, color='black', linestyle='--', label='T0')
    ax_hp30.axvspan(xF[0] - timedelta(minutes=15), xF[-1] + timedelta(minutes=15),
                    color='grey', alpha=0.1, label='Forecast Window')
    add_event_bars(ax_hp30, ylim * 0.95)
    ax_hp30.set_ylabel('Hp30', fontsize=fontsize)
    ax_hp30.set_xlabel('Time (UTC)', fontsize=fontsize)
    ax_hp30.legend(loc='upper right')
    ax_hp30.set_ylim(0, ylim)
    ax_hp30.set_xlim(timestamps[0], timestamps[-1])
    ax_hp30.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    # ===== PANEL 4: Overlaid predictive distributions for all models =====
    y_range      = np.linspace(0.01, ylim + 1, 300)
    colour_cycle = cm.tab10(np.linspace(0, 1, len(model_names)))

    for colour, model_name in zip(colour_cycle, model_names):
        d        = model_dist[model_name]
        label    = label_map[model_name]
        pdf      = d['pdf'](y_range)
        median   = d['median']
        is_best  = (model_name == best_model)
        lw       = 3.0 if is_best else 2.0
        alpha    = 1.0 if is_best else 0.6
        suffix   = ' *' if is_best else ''

        ax_hist.plot(pdf, y_range, lw=lw, color=colour, alpha=alpha,
                     label=f'{label}{suffix}\n({d["label_params"]})')
        ax_hist.fill_betweenx(y_range, pdf, alpha=0.15 if is_best else 0.08, color=colour)
        ax_hist.axhline(median, color=colour, linestyle='-.',
                        lw=1.5 if is_best else 1.2, alpha=0.9 if is_best else 0.7)

        # optional per-component dashed breakdown for mixtures
        if show_components and d['is_mixture']:
            for k in range(d['K']):
                comp = d['w_k'][k] * lognorm.pdf(y_range, s=d['sigma_k'][k],
                                                 scale=np.exp(d['mu_k'][k]))
                ax_hist.plot(comp, y_range, color=colour, lw=1.0, ls=':', alpha=0.5)

    ax_hist.axhline(max_target, color=max_hp30_colour, linestyle='--', lw=2,
                    label=f'True value: {max_target:.2f}')
    ax_hist.yaxis.tick_right()
    ax_hist.yaxis.set_label_position("right")
    ax_hist.set_ylabel('Predicted Hp30', fontsize=fontsize)
    ax_hist.set_xlabel('Density', fontsize=fontsize - 2)
    ax_hist.set_ylim(0, ylim)
    ax_hist.legend(loc='center left', fontsize=7, bbox_to_anchor=(1.15, 0.5), borderaxespad=0)

    ax_bz.tick_params(labelbottom=False)
    ax_v.tick_params(labelbottom=False)
    fig.autofmt_xdate()
    fig.subplots_adjust(right=0.72)

    if save and save_name:
        save_figure(save_name, subfolder='case_studies', huxt_id=huxt_id)

    plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Optional, List
from storm_regression.plotting import save_figure


def plot_metric_heatmap(
    metric_file: Path,
    metric: str,
    row_var: str = 'Ensemble Regressor',
    col_var: str = 'Aggregator',
    filters: Optional[dict] = None,
    sort_by_first_row: bool = True,
    figsize: tuple = (11, 6),
    save: bool = False,
    save_name: Optional[str] = None,
    save_subfolder: str = 'model_comparison',
    huxt_id: Optional[int] = None
):
    """
    Create a heatmap comparing a metric across two variables.
    
    Parameters
    ----------
    metric_file : Path
        Path to CSV file containing metrics
    metric : str
        Metric to plot (e.g., 'mae', 'rmse', 'correlation', 'crps', 'mean_ks')
    row_var : str
        Variable for rows (e.g., 'Ensemble Regressor', 'Lead Time', 'OMNI Parameters')
    col_var : str
        Variable for columns (e.g., 'Aggregator', 'Lead Time', 'Storm Test Threshold')
    filters : dict, optional
        Filters to apply. Example: {'Storm Test Threshold': 4.5, 'Test Mode': 'balanced'}
    sort_by_first_row : bool
        If True, sort columns by performance of first (best) row.
        If False, sort columns by mean across all rows.
    figsize : tuple
        Figure size (width, height)
    save : bool
        Whether to save the figure
    save_name : str, optional
        Filename for saving (without extension)
    save_subfolder : str
        Subfolder in figures/ for saving
    huxt_id : int, optional
        HUXt run ID for filename
    
    Returns
    -------
    pd.DataFrame
        The pivot table that was plotted
    
    Examples
    --------
    >>> # Model vs Aggregator
    >>> plot_metric_heatmap(
    ...     metric_file=paths['regression_metrics'] / 'HUXt1' / 'run_*.csv',
    ...     metric='mae',
    ...     row_var='Ensemble Regressor',
    ...     col_var='Aggregator',
    ...     filters={'Storm Test Threshold': 4.5, 'Test Mode': 'balanced', 'Lead Time': 12}
    ... )
    
    >>> # OMNI vars vs Lead Time
    >>> plot_metric_heatmap(
    ...     metric_file=paths['regression_metrics'] / 'HUXt1' / 'run_*.csv',
    ...     metric='rmse',
    ...     row_var='OMNI Parameters',
    ...     col_var='Lead Time',
    ...     filters={'Ensemble Regressor': 'RandomForest', 'Aggregator': 'weibull_median'}
    ... )
    """
    # Load data
    df = pd.read_csv(metric_file)
    
    print(f"Loaded {len(df)} rows from {metric_file.name}")
    
    # Apply filters
    if filters:
        print(f"\nApplying filters:")
        for key, value in filters.items():
            if key in df.columns:
                df = df[df[key] == value]
                print(f"  {key} == {value}: {len(df)} rows remaining")
            else:
                print(f"  Warning: Column '{key}' not found in data")
    
    # Group and aggregate
    grouped = (
        df
        .groupby([row_var, col_var])[metric]
        .median()  # Use median across seeds/folds
        .reset_index()
    )
    
    if len(grouped) == 0:
        print(f"Error: No data remaining after filtering and grouping")
        return None
    
    print(f"\nGrouped to {len(grouped)} combinations of {row_var} × {col_var}")
    print(f"Unique {row_var}: {sorted(grouped[row_var].unique())}")
    print(f"Unique {col_var}: {sorted(grouped[col_var].unique())}")
    
    # Determine metric-specific settings
    if metric in ['mae', 'rmse', 'crps', 'mean_ks']:
        vmin, vmax = None, None
        cmap = 'viridis_r'  # Lower is better
        upper_label = 'worse (higher)'
        lower_label = 'better (lower)'
        ascending = True
    elif metric == 'correlation':
        vmin, vmax = 0, 1
        cmap = 'viridis'  # Higher is better
        upper_label = 'better (higher)'
        lower_label = 'worse (lower)'
        ascending = False
    else:
        # Default for unknown metrics
        vmin, vmax = None, None
        cmap = 'viridis_r'
        upper_label = 'higher'
        lower_label = 'lower'
        ascending = True
    
    # Pivot to 2D table
    pivot = grouped.pivot(index=row_var, columns=col_var, values=metric)
    
    # Sort rows by mean performance
    row_means = pivot.mean(axis=1)
    pivot = pivot.loc[row_means.sort_values(ascending=ascending).index]
    
    # Sort columns
    if sort_by_first_row:
        # Sort by best model's performance
        col_means = pivot.iloc[0]
    else:
        # Sort by mean across all models
        col_means = pivot.mean(axis=0)
    
    pivot = pivot[col_means.sort_values(ascending=ascending).index]
    
    # Create heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(pivot, vmin=vmin, vmax=vmax, annot=True, cmap=cmap, fmt=".3f",
                    linewidths=0.5, linecolor='gray')
    
    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label(metric.upper(), fontsize=14)
    formatter = FuncFormatter(lambda x, pos: f"{x:.2f}")
    cbar.ax.yaxis.set_major_formatter(formatter)
    cbar.ax.text(0.5, -0.05, lower_label, ha="center", va="top", 
                transform=cbar.ax.transAxes, fontsize=11)
    cbar.ax.text(0.5, 1.05, upper_label, ha="center", va="bottom", 
                transform=cbar.ax.transAxes, fontsize=11)
    
    # Rotate labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    
    # Labels and title
    plt.xlabel(f'{col_var} (best → worst)', fontsize=14)
    plt.ylabel(f'{row_var} (best → worst)', fontsize=14)
    
    # Build title from filters
    filter_text = ', '.join([f'{k}: {v}' for k, v in filters.items()]) if filters else ''
    plt.title(f'{metric.upper()} - {filter_text}', fontsize=14, pad=15)
    
    plt.tight_layout()
    
    # Save if requested
    if save:
        if save_name is None:
            # Auto-generate name
            save_name = f'{metric}_{row_var.lower().replace(" ", "_")}_vs_{col_var.lower().replace(" ", "_")}'
        
        save_figure(save_name, subfolder=save_subfolder, huxt_id=huxt_id)
    
    plt.show()
    
    # Print summary statistics
    print(f"\n{metric.upper()} - {row_var} vs {col_var} (sorted best to worst):")
    print(f"{'='*80}")
    print(f"\nRow means ({row_var}):")
    for var, val in pivot.mean(axis=1).items():
        print(f"  {var}: {val:.3f}")
    print(f"\nColumn means ({col_var}):")
    for var, val in pivot.mean(axis=0).items():
        print(f"  {var}: {val:.3f}")
    print(f"{'='*80}\n")
    
    return pivot


def plot_loss_curve(results_path, save=False, save_name=None, huxt_id=None):
    """Plot train vs validation loss per epoch from a saved run, marking the
    best-validation epoch (where early stopping restored weights)."""
    from storm_regression.results_io import load_results
    results, config, _ = load_results(results_path)

    hist = results.get('loss_history')
    if not hist:
        print("No 'loss_history' in this results file "
              "(was it trained before loss-history saving was added?).")
        return

    train = hist['train']; val = hist['val']
    best_epoch = hist.get('best_epoch'); best_val = hist.get('best_val')
    epochs = range(1, len(train) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train, label='train', color='steelblue', lw=2)
    ax.plot(epochs, val,   label='validation', color='darkorange', lw=2)
    if best_epoch is not None:
        ax.axvline(best_epoch, color='red', ls='--', alpha=0.7,
                   label=f'best val (epoch {best_epoch}, {best_val:.3f})')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (NLL)', fontsize=12)
    ax.set_title(f"Training curve — {config.get('run_name', config.get('model_name',''))}",
                 fontsize=13)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save and save_name:
        save_figure(save_name, subfolder='training_curves', huxt_id=huxt_id)
    plt.show()


# ============================================================================
# Mixture Distribution Diagnostics
# ============================================================================
# Two entry points:
#   plot_mixture_single(results_file)  -> detailed view of ONE run
#       (weights, component separation, example PDFs, sigmas, param scatter matrix)
#   plot_mixture_directory(results_dir) -> pooled distributions + disparities
#       across MANY runs (parameter distributions split storm vs quiet)
#
# Append to plotting.py. Uses the existing load_results and save_figure.


def _mixture_arrays(results):
    """
    Pull (mu, sigma, alpha, y, K) from a results dict; None if not a mixture.

    Components are returned in their NATURAL (learned) order — NOT sorted. The
    network appears to learn a stable role per component (each output head
    specialising to a mode), so the raw per-component histograms are meaningful
    as-is. (Smeared/overlapping per-component histograms would instead indicate
    label switching — so the unsorted view doubles as a specialisation check.)
    """
    if 'mu_m' not in results:
        return None
    mu = np.asarray(results['mu_m'])
    sigma = np.asarray(results['sigma_m'])
    alpha = np.asarray(results['alpha'])
    y = np.asarray(results['y_test']).ravel()
    return mu, sigma, alpha, y, mu.shape[1]


def plot_mixture_single(
    results_file,
    storm_threshold: float = 4.5,
    huxt_id: Optional[int] = None,
    save_name: Optional[str] = None,
):
    """
    Mixture diagnostics for a SINGLE results file.

    Components are shown in their natural (learned) order — not sorted — so the
    per-component histograms reflect whatever roles the network has learned.

    Figure 1 (2x2 summary), all storm (red) vs quiet (blue):
        - per-component alpha (weights) histograms
        - per-component mu histograms
        - per-component sigma histograms
        - component 0 vs 1 median scatter (points on identity line = collapsed)
    Figure 2 (disparity): how multimodal forecasts get —
        - mu disparity  (max-min mu across components; large = well-separated modes)
        - alpha balance  (min alpha; large = both components genuinely used)
    Figure 3 (K panels of 3x3): per-component scatter of that component's own
        mu / sigma / alpha against each other.

    Parameters
    ----------
    results_file : str or Path
    """
    from storm_regression.results_io import load_results
    from matplotlib.gridspec import GridSpec

    results, config, _ = load_results(results_file)
    parsed = _mixture_arrays(results)
    if parsed is None:
        logger.warning("%s has no mixture params (mu_m); nothing to plot.", results_file)
        return
    mu, sigma, alpha, y, K = parsed
    is_storm = y > storm_threshold
    logger.info("Single-file mixture: %d forecasts, K=%d, lead=%sh, %d storms",
                len(y), K, config.get('lead_time'), int(is_storm.sum()))

    def _dual_hist(ax, data, title, xlabel):
        ax.hist(data[~is_storm], bins=40, alpha=0.5, color='steelblue', density=True, label='quiet')
        ax.hist(data[is_storm], bins=40, alpha=0.6, color='crimson', density=True, label='storm')
        ax.set_title(title, fontsize=10); ax.set_xlabel(xlabel, fontsize=9)
        ax.set_yticks([]); ax.legend(fontsize=8)

    # ---- Figure 1: 2x2 summary ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) alpha per component (overlay components; storm/quiet by colour intensity)
    ax = axes[0, 0]
    for k in range(K):
        ax.hist(alpha[:, k], bins=40, alpha=0.55, label=f'comp {k}', density=True)
    ax.set_title('Mixture weights (alpha) per component'); ax.set_xlabel('weight')
    ax.set_yticks([]); ax.legend(fontsize=8)

    # (b) mu per component — shown in REAL Hp30 space (component median = exp(mu))
    ax = axes[0, 1]
    comp_median = np.exp(mu)
    for k in range(K):
        ax.hist(comp_median[:, k], bins=40, alpha=0.55, label=f'comp {k}', density=True)
    ax.set_title('Component median (Hp30max) per component'); ax.set_xlabel('component median')
    ax.set_yticks([]); ax.legend(fontsize=8)

    # (c) sigma per component
    ax = axes[1, 0]
    for k in range(K):
        ax.hist(sigma[:, k], bins=40, alpha=0.55, label=f'comp {k}', density=True)
    ax.set_title('Component sigma per component'); ax.set_xlabel('sigma')
    ax.set_yticks([]); ax.legend(fontsize=8)

    # (d) component-0 vs component-1 median (collapse view), storm/quiet coloured
    ax = axes[1, 1]
    comp_med = np.exp(mu)
    ax.scatter(comp_med[~is_storm, 0], comp_med[~is_storm, 1],
               s=6, alpha=0.3, color='steelblue', label='quiet')
    ax.scatter(comp_med[is_storm, 0], comp_med[is_storm, 1],
               s=6, alpha=0.5, color='crimson', label='storm')
    lim = [0, float(np.percentile(comp_med, 99))]
    ax.plot(lim, lim, 'k--', lw=1, label='identical (collapse)')
    ax.set_xlabel('component 0 median (low mode)')
    ax.set_ylabel('component 1 median')
    ax.set_title('Component 0 vs 1 median (on line = collapsed)'); ax.legend(fontsize=8)

    plt.tight_layout()
    if save_name:
        save_figure(f'{save_name}_summary', subfolder='distribution_analysis', huxt_id=huxt_id)

    # ---- Figure 2: multimodality disparity ----
    comp_median = np.exp(mu)                     # real Hp30 space
    mu_disp = comp_median.max(1) - comp_median.min(1)   # separation of extreme modes (Hp30)
    alpha_bal = alpha.min(1)            # min weight: large => all comps genuinely used
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.5))
    _dual_hist(ax2[0], mu_disp,
               'median disparity = max-min component median (large = distinct modes)',
               'median disparity (Hp30max)')
    _dual_hist(ax2[1], alpha_bal,
               'min alpha (large = both/all components used, not collapsed)', 'min weight')
    fig2.suptitle('How multimodal do individual forecasts get? (red=storm blue=quiet)',
                  fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_name:
        save_figure(f'{save_name}_disparity', subfolder='distribution_analysis', huxt_id=huxt_id)

    logger.info("median disparity (Hp30)  storm mean=%.3f  quiet mean=%.3f "
                "(storm larger => more multimodal in storms)",
                mu_disp[is_storm].mean(), mu_disp[~is_storm].mean())
    logger.info("min alpha     storm mean=%.3f  quiet mean=%.3f",
                alpha_bal[is_storm].mean(), alpha_bal[~is_storm].mean())

    # ---- Figure 3: per-component 3x3 (mu/sigma/alpha within each component) ----
    fig3 = plt.figure(figsize=(4.0 * K + 1, 4.2))
    outer = GridSpec(1, K, figure=fig3, wspace=0.25)
    labels = ['mu', 'sigma', 'alpha']
    for k in range(K):
        P = np.column_stack([mu[:, k], sigma[:, k], alpha[:, k]])   # (N, 3)
        sub = outer[0, k].subgridspec(3, 3, hspace=0.06, wspace=0.06)
        for i in range(3):
            for j in range(3):
                ax = fig3.add_subplot(sub[i, j])
                if j > i:
                    # upper triangle is a mirror of the lower triangle — blank it
                    ax.axis('off')
                    continue
                if i == j:
                    ax.hist(P[~is_storm, i], bins=30, alpha=0.5, color='steelblue')
                    ax.hist(P[is_storm, i], bins=30, alpha=0.6, color='crimson')
                    ax.set_yticks([])
                else:
                    ax.scatter(P[~is_storm, j], P[~is_storm, i], s=2, alpha=0.2, color='steelblue')
                    ax.scatter(P[is_storm, j], P[is_storm, i], s=2, alpha=0.4, color='crimson')
                ax.set_xlabel(labels[j], fontsize=7) if i == 2 else ax.set_xticklabels([])
                ax.set_ylabel(labels[i], fontsize=7) if j == 0 else ax.set_yticklabels([])
                ax.tick_params(labelsize=5)
        fig3.text((k + 0.5) / K, 1.0, f'component {k}', ha='center', fontsize=11)
    fig3.suptitle(f'Per-component parameter relations (K={K}, red=storm blue=quiet)',
                  fontsize=12, y=1.08)
    if save_name:
        save_figure(f'{save_name}_percomponent', subfolder='distribution_analysis', huxt_id=huxt_id)
    plt.show()


def plot_mixture_directory(
    results_dir,
    storm_threshold: float = 4.5,
    max_points: int = 40000,
    huxt_id: Optional[int] = None,
    save_name: Optional[str] = None,
):
    """
    Pooled mixture-parameter distributions across MANY results files.

    Shows the full DISTRIBUTION of each parameter and of the between-component
    disparities (max-min across components within a forecast), split storm vs
    quiet — rather than a single collapse/no-collapse number. Prints the
    disparity summary (mean/median/max/95th, storm vs quiet).

    Only files sharing the first file's K are pooled (others skipped).

    Parameters
    ----------
    results_files : iterable of (str or Path)
        Paths to the results pickles to pool.
    """
    from storm_regression.results_io import load_results
 
    results_files = sorted(Path(results_dir).glob('*.pkl'))
    if not results_files:
        logger.warning("No .pkl files in %s; nothing to plot.", results_dir)
        return

    MU, SG, AL, Y, K0 = [], [], [], [], None
    for path in results_files:
        results, _, _ = load_results(path)
        parsed = _mixture_arrays(results)
        if parsed is None:
            continue
        mu, sigma, alpha, y, K = parsed
        if K0 is None:
            K0 = K
        if K != K0:
            logger.warning("Skipping %s: K=%d != %d", path, K, K0)
            continue
        MU.append(mu); SG.append(sigma); AL.append(alpha); Y.append(y)
    if not MU:
        logger.warning("No mixture results with mu_m found; nothing to plot.")
        return

    mu = np.concatenate(MU); sigma = np.concatenate(SG)
    alpha = np.concatenate(AL); y = np.concatenate(Y); K = K0
    is_storm = y > storm_threshold

    if mu.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(mu.shape[0], max_points, replace=False)
        mu, sigma, alpha, is_storm = mu[idx], sigma[idx], alpha[idx], is_storm[idx]

    # between-component disparities
    disparities = {
        'mu':             mu.max(1) - mu.min(1),
        'median (native)': np.exp(mu).max(1) - np.exp(mu).min(1),
        'sigma':          sigma.max(1) - sigma.min(1),
        'alpha':          alpha.max(1) - alpha.min(1),
    }

    def _dual_hist(ax, data, title, xlabel):
        ax.hist(data[~is_storm], bins=60, alpha=0.5, color='steelblue', density=True, label='quiet')
        ax.hist(data[is_storm], bins=60, alpha=0.6, color='crimson', density=True, label='storm')
        ax.set_title(title, fontsize=10); ax.set_xlabel(xlabel, fontsize=9)
        ax.set_yticks([]); ax.legend(fontsize=8)

    fig, axes = plt.subplots(3, K + 1, figsize=(3.2 * (K + 1), 9))
    for k in range(K):
        _dual_hist(axes[0, k], mu[:, k],    f'mu_{k}', 'mu')
        _dual_hist(axes[1, k], sigma[:, k], f'sigma_{k}', 'sigma')
        _dual_hist(axes[2, k], alpha[:, k], f'alpha_{k}', 'weight')
    _dual_hist(axes[0, K], disparities['mu'],    'mu disparity',    'range')
    _dual_hist(axes[1, K], disparities['sigma'], 'sigma disparity', 'range')
    _dual_hist(axes[2, K], disparities['alpha'], 'alpha disparity', 'range')
    fig.suptitle(f'Mixture parameter distributions (K={K}, pooled; red=storm blue=quiet)',
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_name:
        save_figure(save_name, subfolder='distribution_analysis', huxt_id=huxt_id)
    plt.show()

    logger.info("Between-component disparities (K=%d, pooled over %d files):", K, len(MU))
    for name, d in disparities.items():
        logger.info("  %-16s mean=%.3f median=%.3f max=%.3f 95th=%.3f | storm=%.3f quiet=%.3f",
                    name, d.mean(), np.median(d), d.max(), np.percentile(d, 95),
                    d[is_storm].mean(), d[~is_storm].mean())
    

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from storm_utils.config_paths import get_project_paths
    
    paths = get_project_paths()
    
    # Example 1: Model vs Aggregator
    plot_metric_heatmap(
        metric_file=paths['regression_metrics'] / 'HUXt1' / 'run_2026-02-06_16-58-18_metrics.csv',
        metric='mae',
        row_var='Ensemble Regressor',
        col_var='Aggregator',
        filters={'Storm Test Threshold': 4.5, 'Test Mode': 'balanced', 'Lead Time': 12},
        save=True,
        save_name='mae_model_vs_aggregator_lt12',
        huxt_id=1
    )
    
    # Example 2: OMNI Parameters vs Lead Time
    plot_metric_heatmap(
        metric_file=paths['regression_metrics'] / 'HUXt1' / 'run_2026-02-06_16-58-18_metrics.csv',
        metric='rmse',
        row_var='OMNI Parameters',
        col_var='Lead Time',
        filters={'Ensemble Regressor': 'RandomForest', 'Aggregator': 'weibull_median', 
                'Test Mode': 'balanced'},
        save=True,
        save_name='rmse_omni_vs_leadtime_randomforest',
        save_subfolder='feature_analysis',
        huxt_id=1
    )
    
    # Example 3: Lead Time vs Storm Threshold
    plot_metric_heatmap(
        metric_file=paths['regression_metrics'] / 'HUXt1' / 'run_2026-02-06_16-58-18_metrics.csv',
        metric='correlation',
        row_var='Lead Time',
        col_var='Storm Test Threshold',
        filters={'Ensemble Regressor': 'Ridge', 'Aggregator': 'lognormal_median'},
        sort_by_first_row=False,  # Sort by mean, not first row
        save=True,
        huxt_id=1
    )
    
    # Example 4: Compare all metrics for one configuration
    metrics = ['mae', 'rmse', 'correlation', 'crps']
    
    for metric in metrics:
        pivot = plot_metric_heatmap(
            metric_file=paths['regression_metrics'] / 'HUXt1' / 'run_2026-02-06_16-58-18_metrics.csv',
            metric=metric,
            row_var='Ensemble Regressor',
            col_var='Aggregator',
            filters={'Lead Time': 12, 'Test Mode': 'balanced'},
            save=True,
            save_subfolder='thesis',
            huxt_id=1
        )


