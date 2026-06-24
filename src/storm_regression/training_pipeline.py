"""
Improved ensemble regression training and evaluation pipeline.

Key improvements:
- Modular function-based design
- Configuration management
- Comprehensive error handling
- Memory optimization
- Better logging
- Progress tracking
"""

import numpy as np
import datetime
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import kstest, norm, weibull_min, lognorm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from storm_utils.data_structure import ForecastingDataset, ForecastingConfig
from storm_utils.config_paths import get_project_paths
from storm_regression.forecast_analysis import evaluate_regression_forecast, evaluate_distribution_forecast
from storm_regression.case_study_analysis import save_results


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LossConfig:
    """Base configuration for loss functions."""
    loss_type: str = 'nll'
    

@dataclass
class NLLLossConfig(LossConfig):
    """Configuration for NLL loss."""
    loss_type: str = 'nll'
    w_sigma: float = 1.0
    w_accuracy: float = 1.0
    intensity_type: str = 'none'
    intensity_strength: float = 1.0

@dataclass
class CRPSLossConfig(LossConfig):
    """Configuration for CRPS loss."""
    loss_type: str = 'crps'
    n_samples: int = 1000  # For Monte Carlo CRPS
    intensity_type: str = 'none'
    intensity_strength: float = 1.0

@dataclass
class AsymmetricLossConfig(LossConfig):
    """Configuration for asymmetric loss."""
    loss_type: str = 'asymmetric'
    underpred_penalty: float = 2.0
    overpred_penalty: float = 1.0

@dataclass
class TwCRPSLossConfig(LossConfig):
    loss_type: str = 'twcrps'
    threshold: float = 4.66
    sharpness: float = 2.0
    n_samples: int = 1000


def get_loss_function(loss_config: LossConfig):
    """
    Factory function to get the appropriate loss function.
    
    Args:
        loss_config: LossConfig instance (NLLLossConfig, CRPSLossConfig, etc.)
        
    Returns:
        Loss function callable
    """
    if loss_config.loss_type == 'nll':
        def nll_loss(log_mu_pred, log_sigma_pred, y_true):
            return negative_log_likelihood_lognormal(
                log_mu_pred, log_sigma_pred, y_true,
                w_sigma=loss_config.w_sigma,
                w_accuracy=loss_config.w_accuracy,
                intensity_weight_type=loss_config.intensity_type,
                intensity_weight_strength=loss_config.intensity_strength
            )
        return nll_loss
    
    elif loss_config.loss_type == 'crps':
        def crps_loss(log_mu_pred, log_sigma_pred, y_true):
            return crps_lognormal_loss(
                log_mu_pred, log_sigma_pred, y_true,
                n_samples=loss_config.n_samples,
                intensity_weight_type=loss_config.intensity_type,
                intensity_weight_strength=loss_config.intensity_strength
            )
        return crps_loss
    
    elif loss_config.loss_type == 'asymmetric':
        def asym_loss(log_mu_pred, log_sigma_pred, y_true):
            return asymmetric_nll_loss(
                log_mu_pred, log_sigma_pred, y_true,
                underpred_penalty=loss_config.underpred_penalty,
                overpred_penalty=loss_config.overpred_penalty
            )
        return asym_loss

    elif loss_config.loss_type == 'twcrps':
        def twcrps_loss(log_mu_pred, log_sigma_pred, y_true):
            return tw_crps_lognormal_loss(
                log_mu_pred, log_sigma_pred, y_true,
                threshold=loss_config.threshold,
                sharpness=loss_config.sharpness,
                n_samples=loss_config.n_samples
            )
        return twcrps_loss
            
    else:
        raise ValueError(f"Unknown loss type: {loss_config.loss_type}")


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    
    # Data paths
    huxt_run_id: int = 1
    huxt_data_path: Optional[Path] = None
    discontinuity_path: Optional[Path] = None
    run_name: Optional[str] = None
    
    # Model parameters
    model_type: str = 'ensemble'  # 'ensemble' or 'mlp'

    # Ensemble selection parameters
    n_ensembles: int = 100
    ensemble_selection_method: str = 'per_timestep'
    filter_ensemble: bool = False
    n_ensemble_keep: int = 50
    
    ensemble_regressors: Dict = field(default_factory=lambda: {
        'LinearRegression': LinearRegression
    })
    model_constraints: Dict = field(default_factory=lambda: {
        'LinearRegression': 'realistic',
    })
    n_jobs: int = -1
    
    # MLP-specific parameters
    mlp_ensemble_percentiles: List[float] = field(default_factory=lambda: [5, 50, 95])
    mlp_include_ensemble_spread: bool = False
    mlp_architecture: List[int] = field(default_factory=lambda: [50, 50, 50]) 
    mlp_n_epochs: int = 100
    mlp_learning_rate: float = 0.001
    mlp_patience: int = 10
    mlp_device: str = 'cpu'

    #loss function parameters
    loss_config: LossConfig = field(default_factory=NLLLossConfig)

    # Time parameters
    lead_times: List[int] = field(default_factory=lambda: [12])
    forecast_duration_hours: int = 24
    stride_hours: int = 24
    
    # Storm parameters
    storm_balance_threshold: float = ForecastingConfig.DEFAULT_STORM_THRESHOLD
    
    # Training parameters
    balance: bool = True
    remove_cmes: bool = True
    random_seeds: List[int] = field(default_factory=lambda: [42])
    test_folds: List[int] = field(default_factory=lambda: [0])
    train_ratio: float = 0.8
    batch_size: int = 128
    
    # OMNI feature subsets to test
    omni_subset: List[List[str]] = field(default_factory=lambda: ["Bz_GSM"])
    
    # Output parameters
    output_folder: Optional[Path] = None
    run_name: Optional[str] = None
    save_results: bool = True
    
    # Experiment tracking parameters
    experiment_phase: Optional[str] = None  # e.g., 'phase1_baseline', 'phase2_architecture'
    experiment_id: Optional[str] = None  # e.g., 'exp001_baseline', 'exp010_taper'
    save_metadata: bool = True  # Save JSON metadata file
    

class DataScaler:
    """Manages scaling of different data types."""
    
    def __init__(self):
        self.scalers = {}
    
    def fit_scaler(self, name: str, data: np.ndarray) -> MinMaxScaler:
        """Fit a scaler to data and store it."""
        flat = data.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler.fit(flat)
        self.scalers[name] = scaler
        return scaler
    
    def transform(self, name: str, data: np.ndarray) -> np.ndarray:
        """Transform data using a stored scaler."""
        if name not in self.scalers:
            raise ValueError(f"Scaler '{name}' not found. Fit it first.")
        
        B, T, N = data.shape
        flat = data.reshape(-1, 1)
        flat_scaled = self.scalers[name].transform(flat)
        return flat_scaled.reshape(B, T, N)


class MetricWriter:
    """Handles writing metrics to CSV file."""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self._write_header()
    
    def _write_header(self):
        """Write CSV header."""
        # Create dummy evaluation to get metric names
        dummy_metrics = evaluate_regression_forecast(
            np.array([[1], [0]]),
            np.array([[1], [0]])
        )
        
        header = (
            'Random Seed,Test Fold,Nens,Lead Time,Ensemble Regressor,'
            'Aggregator,Storm Test Threshold,Test Mode,OMNI Parameters,'
        )
        header += ','.join(dummy_metrics.keys())
        header += ',mean_ks,crps,'
        header += ('lambda_mean,lambda_median,lambda_std,lambda_min,lambda_max,'
                  'k_mean,k_median,k_std,k_min,k_max,'
                  'mu_mean,mu_median,mu_std,mu_min,mu_max,'
                  'sigma_mean,sigma_median,sigma_std,sigma_min,sigma_max')
        
        with open(self.output_path, 'w') as f:
            f.write(header)
        
        logger.info(f"Created metrics file: {self.output_path}")
    
    def write_metrics(
        self,
        config_params: Dict,
        metrics: Dict,
        crps: Optional[float] = None,
        dist_params: Optional[Dict] = None
    ):
        """Write a row of metrics to the CSV."""
        with open(self.output_path, 'a') as f:
            # Write configuration parameters
            f.write(f"\n{config_params['random_seed']}")
            f.write(f",{config_params['test_fold']}")
            f.write(f",{config_params['n_ensembles']}")
            f.write(f",{config_params['lead_time']}")
            f.write(f",{config_params['ensemble_regressor']}")
            f.write(f",{config_params['aggregator']}")
            f.write(f",{config_params['omni_subset']}")
            
            # Write metrics
            f.write(',' + ','.join([str(v) for v in metrics.values()]))
            
            # Write CRPS
            f.write(f',{crps if crps is not None else ""}')
            
            # Write distribution parameters (if applicable)
            if dist_params:
                for param_group in ['lambda', 'k', 'mu', 'sigma',]:
                    if param_group in dist_params:
                        f.write(',' + ','.join([str(v) for v in dist_params[param_group].values()]))
                    else:
                        # Write empty values for unused distribution parameters
                        n_stats = 5
                        f.write(',' + ','.join([''] * n_stats))
            else:
                # Write empty values for all distribution parameters
                f.write(',' + ','.join([''] * 16))
                

def prepare_features(
    batch: Dict,
    n_ensembles: int,
    scaler: Optional[DataScaler] = None,
    fit_scaler: bool = False
) -> Tuple[np.ndarray, Optional[DataScaler]]:
    """
    Prepare and scale features from a batch.
    
    Args:
        batch: Dictionary of batch data
        n_ensembles: Number of ensemble members
        scaler: Optional DataScaler to use for transformation
        fit_scaler: Whether to fit the scaler (True for training)
    
    Returns:
        Tuple of (scaled_features, scaler)
    """
    # Extract data
    v = batch['v'].numpy()
    vgrad = batch['vgrad'].numpy()
    vomni = batch['vomni'].numpy()
    omni = batch['omni'].numpy()
    hist = batch['historic_target'].numpy()
    
    # Expand OMNI and historical data to match ensemble dimension
    B, T, N_omni = omni.shape
    omni_flat = omni.reshape(B, T * N_omni, 1)
    omni_rep = np.repeat(omni_flat, n_ensembles, axis=2)
    hist_rep = np.repeat(hist, n_ensembles, axis=2)
    
    # Fit or use scaler
    if fit_scaler:
        scaler = DataScaler()
        scaler.fit_scaler('v', v)
        scaler.fit_scaler('vgrad', vgrad)
        scaler.fit_scaler('vomni', vomni)
        scaler.fit_scaler('omni', omni_flat)
        scaler.fit_scaler('hist_rep', hist)
    
    if scaler is None:
        raise ValueError("Scaler required for transformation")
    
    # Scale features
    v_scaled = scaler.transform('v', v)
    vgrad_scaled = scaler.transform('vgrad', vgrad)
    vomni_scaled = scaler.transform('vomni', vomni)
    omni_rep_scaled = scaler.transform('omni', omni_rep)
    hist_rep_scaled = scaler.transform('hist_rep', hist_rep)
    
    # Concatenate features
    X_scaled = np.concatenate([
        v_scaled, vgrad_scaled, vomni_scaled,
        omni_rep_scaled, hist_rep_scaled
    ], axis=1)
    
    # Transpose to (Nens, B, Features)
    X_scaled_transposed = np.transpose(X_scaled, (2, 0, 1))
    
    return X_scaled_transposed, scaler


def constrain_predictions(predictions: np.ndarray, method: str = 'zero') -> np.ndarray:
    """
    Constrain predictions to physically valid range for Hp30 index.
    
    This is a limitation of unconstrained LinearRegression which can predict
    negative or unrealistically high values. Future work should consider
    models with built-in constraints (e.g., Ridge with positive=True, GLMs).
    
    Args:
        predictions: Raw model predictions, shape (B, Nens) or (B,)
        method: Constraint method
            - 'zero': Clip to [0, inf)
            - 'small': Replace negatives with small positive value (0.01)
            - 'abs': Take absolute value
            - 'realistic': Clip to [0.01, 15] based on dataset range
    
    Returns:
        Constrained predictions with same shape as input
    """
    if method == 'zero':
        return np.clip(predictions, a_min=0, a_max=None)
    
    elif method == 'small':
        return np.where(predictions < 0, 0.01, predictions)
    
    elif method == 'abs':
        return np.abs(predictions)
    
    elif method == 'realistic':
        # Based on typical Hp30 range: 0-15
        return np.clip(predictions, a_min=0.01, a_max=15)
    
    else:
        raise ValueError(f"Unknown constraint method: {method}")


def _train_single_model(args):
    """Helper function for parallel training of a single ensemble member."""
    ens_idx, X_ens, y_train, regressor_class = args
    model = regressor_class()
    model.fit(X_ens, y_train)
    return ens_idx, model


def train_ensemble_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_ensembles: int,
    regressor_class,
    n_jobs: int = -1
) -> List:
    """
    Train ensemble of regression models in parallel.
    
    Args:
        X_train: Training features of shape (Nens, B, Features)
        y_train: Training targets of shape (B,)
        n_ensembles: Number of ensemble members
        regressor_class: Regression model class
        n_jobs: Number of parallel jobs (-1 for all CPUs, 1 for serial)
    
    Returns:
        List of trained models
    """
    logger.info(f"Training {n_ensembles} ensemble models...")
    
    # Determine number of workers
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    elif n_jobs == 1:
        # Serial training (original method)
        models = []
        for ens in tqdm(range(n_ensembles), desc="Training models"):
            X = X_train[ens]
            model = regressor_class()
            model.fit(X, y_train)
            models.append(model)
        return models
    
    logger.info(f"Using {n_jobs} parallel workers")
    
    # Prepare arguments for parallel processing
    train_args = [
        (ens, X_train[ens], y_train, regressor_class)
        for ens in range(n_ensembles)
    ]
    
    # Train in parallel
    models = [None] * n_ensembles
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(tqdm(
            executor.map(_train_single_model, train_args),
            total=n_ensembles,
            desc="Training models"
        ))
    
    # Sort results back into order
    for ens_idx, model in results:
        models[ens_idx] = model
    
    return models


def calculate_ensemble_weights(v_historic: np.ndarray, omni_V_sw: np.ndarray) -> np.ndarray:
    """
    Calculate weights for ensemble members based on historical performance.
    
    Args:
        v_historic: Historic velocity ensemble of shape (B, 48, Nens)
        omni_V_sw: OMNI solar wind velocity of shape (B, 48) or (B, 48, 1)
    
    Returns:
        Weights array of shape (B, Nens)
    """
    # Ensure omni_V_sw is (B, 48)
    if omni_V_sw.ndim == 3:
        omni_V_sw = omni_V_sw.squeeze(axis=-1)  # (B, 48, 1) -> (B, 48)
    
    # Expand for broadcasting: (B, 48) -> (B, 48, 1)
    omni_expanded = np.expand_dims(omni_V_sw, axis=-1)  # (B, 48, 1)
    
    # Calculate MAE per ensemble, per window
    # v_historic: (B, 48, Nens), omni_expanded: (B, 48, 1)
    mae = np.mean(np.abs(v_historic - omni_expanded), axis=1)  # (B, Nens)
    
    # Convert to inverse square weights
    inv_sq = 1 / (mae ** 2 + 1e-10)  # Add small constant for stability
    weights = inv_sq / np.sum(inv_sq, axis=1, keepdims=True)
    
    return weights


def fit_distribution_parameters(
    ensemble_predictions: np.ndarray
) -> Tuple[Dict, Dict]:
    """
    Fit Weibull and Normal distributions to ensemble predictions.
    
    Args:
        ensemble_predictions: Shape (B, Nens)
    
    Returns:
        Tuple of (weibull_params, normal_params) dictionaries
    """
    n_samples = ensemble_predictions.shape[0]
    
    # Initialize parameter arrays
    lambda_vals = np.zeros(n_samples)
    k_vals = np.zeros(n_samples)
    mu_vals = np.zeros(n_samples)
    sigma_vals = np.zeros(n_samples)
    log_mu_vals = np.zeros(n_samples)
    log_sigma_vals = np.zeros(n_samples)
    
    for i, sample_forecasts in enumerate(ensemble_predictions):
        # Fit Weibull
        try:
            c, loc, scale = weibull_min.fit(sample_forecasts, floc=0)
            k_vals[i] = c
            lambda_vals[i] = scale
        except Exception as e:
            logger.warning(f"Weibull fit failed for sample {i}: {e}")
            k_vals[i] = np.nan
            lambda_vals[i] = np.nan
        
        # Fit Normal
        try:
            mu, sigma = norm.fit(sample_forecasts)
            mu_vals[i] = mu
            sigma_vals[i] = sigma
        except Exception as e:
            logger.warning(f"Normal fit failed for sample {i}: {e}")
            mu_vals[i] = np.nan
            sigma_vals[i] = np.nan

        try:
            shape, loc, scale = lognorm.fit(sample_forecasts, floc=0)
    
            # Convert to standard lognormal parameters
            log_mu_vals[i] = np.log(scale)
            log_sigma_vals[i] = shape            
        except Exception as e:
            logger.info(f'Ensemble Forecast: min - {np.min(sample_forecasts.flatten())}, max - {np.max(sample_forecasts.flatten())}')
            logger.warning(f"LogNormal fit failed for sample {i}: {e}")
            log_mu_vals[i] = np.nan
            log_sigma_vals[i] = np.nan
    
    weibull_params = {
        'lambda': lambda_vals,
        'k': k_vals,
        'median': lambda_vals * (np.log(2) ** (1/k_vals))
    }
    
    normal_params = {
        'mu': mu_vals,
        'sigma': sigma_vals,
        'median': mu_vals
    }

    lognormal_params = {
        'mu': log_mu_vals,
        'sigma': log_sigma_vals,
        'median': np.exp(log_mu_vals),
    }
    
    return weibull_params, normal_params, lognormal_params


class LogNormalMLP(nn.Module):
    """
    MLP that predicts LogNormal distribution parameters (log_mu, log_sigma).
    
    Flexible architecture defined by hidden_layers list.
    Output: [log_mu, log_sigma] for LogNormal distribution
    """
    
    def __init__(self, input_size: int, hidden_layers: List[int] = [50, 50, 50]):
        """
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes (e.g., [100, 50, 25])
        """
        super(LogNormalMLP, self).__init__()
        
        # Build network dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 2))
        
        self.network = nn.Sequential(*layers)
        
        # Separate activation for log_sigma to ensure positivity
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features, shape (batch_size, input_size)
            
        Returns:
            Tuple of (log_mu, log_sigma), each shape (batch_size,)
        """
        output = self.network(x)
        log_mu = output[:, 0]
        log_sigma = self.softplus(output[:, 1]) + 0.01  # Ensure log_sigma > 0
        
        return log_mu, log_sigma


def negative_log_likelihood_lognormal(
    log_mu_pred, 
    log_sigma_pred, 
    y_true,
    w_sigma: float = 1.0,
    w_accuracy: float = 1.0,
    intensity_weight_type: str = 'none',
    intensity_weight_strength: float = 1.0
):
    """
    Negative log-likelihood loss for LogNormal distribution.
    
    Simplified version - removes constant terms that don't affect gradients.
    """
    # Avoid log(0) issues
    y_true = torch.clamp(y_true, min=1e-6)
    
    # Only the terms that matter for learning
    log_y = torch.log(y_true)
    term_sigma = torch.log(log_sigma_pred)  # Penalizes large sigma
    term_accuracy = ((log_y - log_mu_pred) ** 2) / (2 * log_sigma_pred ** 2)  # Penalizes errors
    
    # Apply term weights
    nll = (w_sigma * term_sigma) + (w_accuracy * term_accuracy)
    
    # Apply intensity-based weighting
    if intensity_weight_type == 'none':
        weights = torch.ones_like(y_true)
    elif intensity_weight_type == 'linear':
        weights = 1.0 + intensity_weight_strength * (y_true / 15.0)
    elif intensity_weight_type == 'quadratic':
        weights = 1.0 + intensity_weight_strength * ((y_true / 15.0) ** 2)
    elif intensity_weight_type == 'exponential':
        weights = torch.exp(intensity_weight_strength * y_true / 15.0)
    elif intensity_weight_type == 'threshold':
        storm_threshold = 4.5
        strong_threshold = 6.5
        weights = torch.where(
            y_true > strong_threshold,
            1.0 + 2.0 * intensity_weight_strength,
            torch.where(
                y_true > storm_threshold,
                1.0 + intensity_weight_strength,
                torch.ones_like(y_true)
            )
        )
    else:
        raise ValueError(f"Unknown intensity_weight_type: {intensity_weight_type}")
    
    weighted_nll = nll * weights
    
    return torch.mean(weighted_nll)


def tw_crps_lognormal_loss(log_mu_pred, log_sigma_pred, y_true,
                           threshold=4.66, sharpness=2.0, n_samples=2000):
    """Threshold-weighted CRPS (upper tail) for a LogNormal. Proper scoring rule.

    twCRPS with weight w(z)=1{z>=t} == ordinary CRPS of v(z), where v'=w, i.e.
    v(z)=(z-t)_+ (smoothed with softplus). Transform samples + obs, then energy score.
    Lower is better.
    """
    y_true = torch.clamp(y_true, min=1e-6)
    B = log_mu_pred.shape[0]
    dev = log_mu_pred.device

    def chain(z):                       # v(z) = softplus(beta (z - t)) / beta  ~  max(z-t, 0)
        return F.softplus(sharpness * (z - threshold)) / sharpness

    eps1 = torch.randn(n_samples, B, device=dev)
    eps2 = torch.randn(n_samples, B, device=dev)
    s1 = torch.exp(log_mu_pred.unsqueeze(0) + log_sigma_pred.unsqueeze(0) * eps1)
    s2 = torch.exp(log_mu_pred.unsqueeze(0) + log_sigma_pred.unsqueeze(0) * eps2)

    v1, v2, vy = chain(s1), chain(s2), chain(y_true).unsqueeze(0)
    term1 = torch.mean(torch.abs(v1 - vy), dim=0)      # E|v(X) - v(y)|
    term2 = 0.5 * torch.mean(torch.abs(v1 - v2), dim=0)  # 0.5 E|v(X) - v(X')|
    return torch.mean(term1 - term2)


def crps_lognormal_loss(
    log_mu_pred: torch.Tensor,
    log_sigma_pred: torch.Tensor,
    y_true: torch.Tensor,
    n_samples: int = 1000,
    intensity_weight_type: str = 'none',
    intensity_weight_strength: float = 1.0,
) -> torch.Tensor:
    """
    CRPS loss for a LogNormal distribution using Monte Carlo estimation.

    CRPS(F, y) = E[|X - y|] - 0.5 * E[|X - X'|]
    where X, X' are independent draws from the predicted distribution F.

    Parameters
    ----------
    log_mu_pred : torch.Tensor, shape (B,)
        Predicted log-space mean.
    log_sigma_pred : torch.Tensor, shape (B,)
        Predicted log-space standard deviation (must be > 0).
    y_true : torch.Tensor, shape (B,)
        Observed target values.
    n_samples : int
        Number of Monte Carlo samples to approximate the CRPS expectation.
    intensity_weight_type : str
        Same weighting options as NLL loss: 'none', 'linear', 'quadratic',
        'exponential', 'threshold'.
    intensity_weight_strength : float
        Strength of the intensity weighting.

    Returns
    -------
    torch.Tensor
        Scalar mean CRPS loss.
    """
    y_true = torch.clamp(y_true, min=1e-6)
    B = log_mu_pred.shape[0]

    # ── Monte Carlo samples from LogNormal(log_mu, log_sigma) ────────────────
    # Sample standard normals: (n_samples, B)
    eps = torch.randn(n_samples, B, device=log_mu_pred.device)

    # Transform to lognormal: X = exp(log_mu + log_sigma * eps)
    samples = torch.exp(
        log_mu_pred.unsqueeze(0) + log_sigma_pred.unsqueeze(0) * eps
    )  # (n_samples, B)

    # ── E[|X - y|] ───────────────────────────────────────────────────────────
    term1 = torch.mean(
        torch.abs(samples - y_true.unsqueeze(0)), dim=0
    )  # (B,)

    # ── 0.5 * E[|X - X'|] — approximate with two independent sample sets ────
    eps2    = torch.randn(n_samples, B, device=log_mu_pred.device)
    samples2 = torch.exp(
        log_mu_pred.unsqueeze(0) + log_sigma_pred.unsqueeze(0) * eps2
    )  # (n_samples, B)

    term2 = 0.5 * torch.mean(
        torch.abs(samples - samples2), dim=0
    )  # (B,)

    crps = term1 - term2  # (B,)

    # ── Intensity-based sample weighting ─────────────────────────────────────
    if intensity_weight_type == 'none':
        weights = torch.ones_like(y_true)
    elif intensity_weight_type == 'linear':
        weights = 1.0 + intensity_weight_strength * (y_true / 15.0)
    elif intensity_weight_type == 'quadratic':
        weights = 1.0 + intensity_weight_strength * ((y_true / 15.0) ** 2)
    elif intensity_weight_type == 'exponential':
        weights = torch.exp(intensity_weight_strength * y_true / 15.0)
    elif intensity_weight_type == 'threshold':
        storm_threshold  = 4.5
        strong_threshold = 6.5
        weights = torch.where(
            y_true > strong_threshold,
            1.0 + 2.0 * intensity_weight_strength,
            torch.where(
                y_true > storm_threshold,
                1.0 + intensity_weight_strength,
                torch.ones_like(y_true)
            )
        )
    else:
        raise ValueError(f"Unknown intensity_weight_type: {intensity_weight_type}")

    return torch.mean(crps * weights)


def select_best_ensemble_members(
    v: np.ndarray,        # (T, N_ens) or (B, T_full, N_ens)
    v_omni: np.ndarray,   # (T_input,) or (B, T_input) — V_sw only
    n_keep: int = 50,
) -> np.ndarray:
    """
    Filter ensemble members by MAE against OMNI V_sw in the input window,
    keeping only the n_keep best members. Applies selected indices to the
    full v array (including forecast window).

    Parameters
    ----------
    v : np.ndarray
        Ensemble velocity array. Shape (T_full, N_ens) or (B, T_full, N_ens).
    v_omni : np.ndarray
        OMNI observed V_sw for input window only. Shape (T_input,) or (B, T_input).
    n_keep : int
        Number of best members to retain.

    Returns
    -------
    np.ndarray
        Filtered ensemble array, shape (T_full, n_keep) or (B, T_full, n_keep).
    """
    if v.ndim == 2:
        # Single window: (T_full, N_ens)
        T_input = v_omni.shape[0]
        v_input = v[:T_input, :]                              # (T_input, N_ens)
        mae     = np.mean(np.abs(v_input - v_omni[:, None]), axis=0)  # (N_ens,)
        best_indices = np.argsort(mae)[:n_keep]
        return v[:, best_indices]                             # (T_full, n_keep)

    elif v.ndim == 3:
        # Batched: (B, T_full, N_ens)
        # v_omni is (B, T_input) — just V_sw, no extra feature dims
        B, T_full, N_ens = v.shape
        T_input = v_omni.shape[1]

        v_input = v[:, :T_input, :]                                       # (B, T_input, N_ens)
        mae     = np.mean(np.abs(v_input - v_omni[:, :, None]), axis=1)   # (B, N_ens)
        best_indices = np.argsort(mae, axis=1)[:, :n_keep]                # (B, n_keep)

        v_filtered = np.stack([v[b][:, best_indices[b]] for b in range(B)], axis=0) # (B, T_full, n_keep)
        assert v_filtered.shape == (B, T_full, n_keep), \
            f"expected (B, T_full, n_keep), got {v_filtered.shape}"

        return v_filtered


def prepare_mlp_features(
    batch: Dict,
    n_ensembles: int = 100,
    ensemble_percentiles: List[float] = [5, 50, 95],
    ensemble_selection_method: str = 'snap',
    filter_ensemble: bool =False,
    n_ensemble_keep: int=50,
    include_ensemble_spread: bool = False,
    scaler: Optional[MinMaxScaler] = None,
    fit_scaler: bool = False
) -> Tuple[np.ndarray, Optional[MinMaxScaler]]:
    """
    Prepare features for MLP training using solar wind velocity ensemble.
    
    Args:
        batch: Dictionary of batch data
        n_ensembles: Number of ensemble members
        ensemble_percentiles: Which percentiles to extract (e.g., [5, 50, 95])
        include_ensemble_spread: Whether to add std, IQR, range as features
        scaler: Optional MinMaxScaler
        fit_scaler: Whether to fit the scaler
        
    Returns:
        Tuple of (features, scaler) where features is (B, n_features)
    """
    # Extract solar wind velocity ensemble
    v = batch['v'].numpy()  # (B, T, Nens)
    omni = batch['omni'].numpy()  # (B, T, N_omni)
    omni_sw = batch['omni_sw'].numpy() 
    hist = batch['historic_target'].numpy()  # (B, hist_len)

    # logger.info('selection method:' + str(ensemble_selection_method))
    # logger.info('ensembles to keep:' + str(n_ensemble_keep))
    # logger.info('filter ensemble:' + str(filter_ensemble))
    # logger.info('percentiles:' + str(ensemble_percentiles))
    # logger.info('v shape:' + str(v.shape) + 'omni shape:' + str(omni.shape) + 'omni_sw shape' + str(omni_sw.shape) + 'historic hp30 shape' + str(hist.shape))
    
    # ── Optional: filter to best ensemble members ─────────────────────────
    if filter_ensemble:
        # Only use input window (up to lead_time) for MAE calculation
        v = select_best_ensemble_members(v, omni_sw, n_keep=n_ensemble_keep)
        # v is now (B, T, n_ensemble_keep)

    # ── Extract percentiles as before ─────────────────────────────────────
    B, T_v, Nens = v.shape
    v_percentiles = []
    for p in ensemble_percentiles:
        if ensemble_selection_method == 'snap':
            member_means = v.mean(axis=1)
            rank_order   = np.argsort(member_means, axis=1)
            rank_idx     = min(int(np.floor(p / 100 * Nens)), Nens - 1)
            member_idx   = rank_order[:, rank_idx]
            v_p          = v[np.arange(B), :, member_idx]
        elif ensemble_selection_method == 'per_timestep':
            v_p = np.percentile(v, p, axis=2, method='nearest')
        v_percentiles.append(v_p.reshape(B, -1))
    
    features_list = v_percentiles
    
    # Optionally add spread metrics
    if include_ensemble_spread:
        v_std = np.std(v, axis=2).reshape(B, -1)
        v_p25 = np.percentile(v, 25, axis=2)
        v_p75 = np.percentile(v, 75, axis=2)
        v_iqr = (v_p75 - v_p25).reshape(B, -1)  # Interquartile range
        v_range = (np.max(v, axis=2) - np.min(v, axis=2)).reshape(B, -1)
        
        features_list.extend([v_std, v_iqr, v_range])
    
    # Add OMNI features if present
    if omni.shape[-1] > 0:
        features_list.append(omni.reshape(B, -1))
    
    # Add historical Hp30
    features_list.append(hist.reshape(B, -1))
    
    # Concatenate all features
    features = np.concatenate(features_list, axis=1)
    
    # Scale features
    if fit_scaler:
        scaler = MinMaxScaler()
        scaler.fit(features)
    
    if scaler is None:
        raise ValueError("Scaler required for transformation")
    
    features_scaled = scaler.transform(features)
    
    return features_scaled, scaler


import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Tuple


def train_mlp(
    train_dataloader: DataLoader,
    n_ensembles: int,
    filter_ensemble: bool = False,
    n_ensemble_keep: int = 50,
    ensemble_selection_method: str = 'per_timestep',
    ensemble_percentiles: List[float] = [5, 50, 95],
    include_ensemble_spread: bool = False,
    architecture: List[int] = [50, 50, 50],
    n_epochs: int = 100,
    learning_rate: float = 0.001,
    loss_config=None,
    device: str = 'cpu',
    patience: int = 10,
    batch_size: int = 256,      # NEW: minibatch size over the cached tensors
    debug: bool = False,        # NEW: print shapes + per-epoch fwd/bwd timing
) -> Tuple['LogNormalMLP', 'MinMaxScaler']:
    """Train MLP to predict LogNormal distribution parameters.

    Features are built ONCE (cached) before the epoch loop instead of being rebuilt
    every batch of every epoch. Scaler is fit on the first batch (same as the original),
    then all features are transformed with it and concatenated into device tensors.
    """
    # ---- validate batch_size (a bad value here was the empty-data crash) ----
    if not isinstance(batch_size, int) or batch_size <= 0:
        if debug:
            print(f"DEBUG: bad batch_size={batch_size!r}; defaulting to 256")
        batch_size = 256

    feat_kwargs = dict(
        ensemble_percentiles=ensemble_percentiles,
        ensemble_selection_method=ensemble_selection_method,
        filter_ensemble=filter_ensemble,
        n_ensemble_keep=n_ensemble_keep,
        include_ensemble_spread=include_ensemble_spread,
    )

    # ---- fit scaler ONCE on the first batch (matches the original semantics) ----
    first_batch = next(iter(train_dataloader))
    _, scaler = prepare_mlp_features(
        first_batch, n_ensembles, fit_scaler=True, **feat_kwargs
    )

    # ---- cache ALL training features using that scaler (fit_scaler=False) ----
    X_list, y_list = [], []
    for bi, batch in enumerate(train_dataloader):
        X, _ = prepare_mlp_features(
            batch, n_ensembles, scaler=scaler, fit_scaler=False, **feat_kwargs
        )
        y = batch['max_target'].numpy()
        if debug and bi == 0:
            print(f"DEBUG: first cached batch -> X {np.asarray(X).shape}, y {np.asarray(y).shape}")
        X_list.append(np.asarray(X))
        y_list.append(np.asarray(y))

    if len(X_list) == 0:
        raise RuntimeError(
            "train_mlp: dataloader yielded no batches — training set is empty "
            "(check the fold split / CME removal / filters upstream)."
        )

    X_all = torch.as_tensor(np.concatenate(X_list, axis=0), dtype=torch.float32, device=device)
    y_all = torch.as_tensor(np.concatenate(y_list, axis=0), dtype=torch.float32, device=device)
    n = X_all.shape[0]
    input_size = X_all.shape[1]

    if debug:
        print(f"DEBUG: cached n={n}, input_size={input_size}, "
              f"batch_size={batch_size}, n_batches/epoch={int(np.ceil(n / batch_size))}, "
              f"X_all {tuple(X_all.shape)} ({X_all.element_size()*X_all.nelement()/1e6:.1f} MB)")

    if n == 0:
        raise RuntimeError(
            f"train_mlp: cached feature tensor is empty (n=0) although "
            f"{len(X_list)} batch(es) were collected — check prepare_mlp_features output."
        )

    logger.info(f"Input size: {input_size}")
    logger.info(f"Architecture: {input_size} → {' → '.join(map(str, architecture))} → 2")
    logger.info(f"Using percentiles: {ensemble_percentiles}")
    logger.info(f"Including spread metrics: {include_ensemble_spread}")
    logger.info(f"Cached features for {n} windows once (was rebuilt every epoch)")

    # ---- model / optimizer / loss ----
    model = LogNormalMLP(input_size, hidden_layers=architecture).to(device)
    logger.info(f"Total trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if loss_config is None:
        loss_config = NLLLossConfig()
    loss_fn = get_loss_function(loss_config)
    logger.info(f"Loss function: {loss_config.loss_type} — config: {vars(loss_config)}")

    # ---- training loop over cached tensors ----
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=device)     # reshuffle each epoch
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time(); t_fwd = 0.0; t_bwd = 0.0

        for start in range(0, n, batch_size):
            sel = perm[start:start + batch_size]

            t = time.time()
            log_mu_pred, log_sigma_pred = model(X_all[sel])
            loss = loss_fn(log_mu_pred, log_sigma_pred, y_all[sel])
            t_fwd += time.time() - t

            t = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_bwd += time.time() - t

            epoch_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError(
                f"train_mlp: no minibatches ran (n={n}, batch_size={batch_size})."
            )

        avg_loss = epoch_loss / n_batches

        if debug and (epoch < 2 or (epoch + 1) % 10 == 0):
            print(f"DEBUG epoch {epoch+1}: {time.time()-t0:.3f}s total | "
                  f"fwd+loss {t_fwd:.3f}s | bwd {t_bwd:.3f}s | {n_batches} batches")

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

        # early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    return model, scaler


def evaluate_mlp_test_set(
    model: LogNormalMLP,
    test_dataloader: DataLoader,
    scaler: DataScaler,
    n_ensembles: int = 100,
    include_ensemble_spread: bool = False,
    ensemble_percentiles: list = [5, 50, 95],
    ensemble_selection_method: str = 'per_timestep',
    filter_ensemble: bool = False,
    n_ensemble_keep: int = 50,
    device: str = 'cpu'
) -> Dict[str, np.ndarray]:
    """
    Evaluate MLP on test set.
    
    Args:
        model: Trained LogNormalMLP
        test_dataloader: DataLoader for test data
        scaler: Fitted DataScaler
        n_ensembles: Number of ensemble members
        include_ensemble_spread: Whether ensemble statistics were used
        device: Device for inference
        
    Returns:
        Dictionary of results
    """
    model.eval()
    all_outputs = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Test batches"):
            # Prepare features
            X, _ = prepare_mlp_features(
                batch, n_ensembles,
                include_ensemble_spread=include_ensemble_spread,
                ensemble_percentiles=ensemble_percentiles,
                ensemble_selection_method=ensemble_selection_method,
                filter_ensemble=filter_ensemble,
                n_ensemble_keep=n_ensemble_keep,
                scaler=scaler,
                fit_scaler=False
            )
            
            # Convert to torch
            X_torch = torch.FloatTensor(X).to(device)
            
            # Predict
            log_mu_pred, log_sigma_pred = model(X_torch)
            
            # Convert to numpy
            log_mu_np = log_mu_pred.cpu().numpy()
            log_sigma_np = log_sigma_pred.cpu().numpy()
            
            # Compute median (deterministic forecast)
            lognormal_median = np.exp(log_mu_np)
            
            # Extract additional data
            hist = batch['historic_target'].cpu().numpy().squeeze(axis=-1)
            y_true = batch['max_target'].cpu().numpy()
            recurrence = batch['27_day_target'].cpu().numpy().squeeze(axis=-1)
            center_indices = batch['center_idx'].cpu().numpy()
            
            # Ensure proper shapes
            if recurrence.ndim == 1:
                recurrence = recurrence.reshape(-1, 1)
            
            # Baseline forecasts
            y_pred_persistence = hist[:, -1]
            y_pred_persistence_max = np.max(hist, axis=-1)
            y_pred_27_day = np.max(recurrence, axis=-1)
            
            all_outputs.append({
                'y_test': y_true.ravel(),
                'y_pred_lognormal_median': lognormal_median,
                'y_pred_persistence': y_pred_persistence,
                'y_pred_persistence_max': y_pred_persistence_max,
                'y_pred_27_day_recurrence': y_pred_27_day,
                'log_mu': log_mu_np,
                'log_sigma': log_sigma_np,
                'center_idx': center_indices,
            })
    
    # Concatenate results
    results = {}
    for key in all_outputs[0]:
        results[key] = np.concatenate([d[key] for d in all_outputs], axis=0)
    
    results['window_idx'] = results['center_idx']
    
    return results
    

def compute_distribution_statistics(params: np.ndarray) -> Dict[str, float]:
    """Compute statistics for distribution parameters."""
    valid_params = params[~np.isnan(params)]
    
    if len(valid_params) == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan
        }
    
    return {
        'mean': np.mean(valid_params),
        'median': np.median(valid_params),
        'std': np.std(valid_params),
        'min': np.min(valid_params),
        'max': np.max(valid_params)
    }


def evaluate_test_set(
    models: List,
    test_dataloader: DataLoader,
    scaler: DataScaler,
    n_ensembles: int,
    test_indices: List[int],
    model_name: str,
    model_constraints: Dict[str, Optional[str]]
) -> Dict[str, np.ndarray]:
    """
    Evaluate models on test set.
    
    Args:
        model_name: Name of the model being evaluated
        model_constraints: Dict mapping model names to constraint methods
    """
    all_outputs = []
    
    # Check if this model needs constraints
    constraint_method = model_constraints.get(model_name, None)
    
    if constraint_method:
        logger.info(f"Will apply '{constraint_method}' constraints for {model_name}")
    else:
        logger.info(f"No constraints needed for {model_name}")
    
    for batch in tqdm(test_dataloader, desc="Test batches"):
        try:
            # Prepare features
            X_test, _ = prepare_features(batch, n_ensembles, scaler=scaler, fit_scaler=False)
            
            # Make predictions
            predictions = []
            for ens in range(n_ensembles):
                X = X_test[ens]
                pred = models[ens].predict(X)
                predictions.append(pred)
            
            predictions = np.array(predictions).T  # (B, Nens)
            
            # Apply constraints only if needed
            if constraint_method:
                n_negative = np.sum(predictions < 0)
                n_too_high = np.sum(predictions > 15)
                
                if n_negative > 0 or n_too_high > 0:
                    logger.debug(f"Batch has {n_negative} negative and {n_too_high} >15 predictions")
                
                predictions = constrain_predictions(predictions, method=constraint_method)
        
            # Extract additional data
            v_input = batch['v_input'].cpu().numpy()
            omni_V_sw = batch['omni_sw'].cpu().numpy()
            hist = batch['historic_target'].cpu().numpy().squeeze(axis=-1)
            y_true = batch['max_target'].cpu().numpy()
            recurrence = batch['27_day_target'].cpu().numpy().squeeze(axis=-1)
            
            # Ensure proper shapes for recurrence
            if recurrence.ndim == 1:
                recurrence = recurrence.reshape(-1, 1)
            
            # Calculate weights and weighted mean
            weights = calculate_ensemble_weights(v_input, omni_V_sw)
            y_pred_weighted_mean = np.sum(predictions * weights, axis=1)
            
            # Fit distributions
            weibull_params, normal_params, lognormal_params = fit_distribution_parameters(predictions)
            weibull_ks = []
            normal_ks = []
            lognormal_ks = []
            
            for i in range(len(predictions)):
                ensemble = predictions[i]
                ensemble_clean = ensemble[~np.isnan(ensemble)]
                
                if len(ensemble_clean) > 2:
                    # Weibull KS
                    ks_w, _ = kstest(ensemble_clean, 
                                     lambda x: weibull_min.cdf(x, c=weibull_params['k'][i], 
                                                              scale=weibull_params['lambda'][i]))
                    weibull_ks.append(ks_w)
                    
                    # Normal KS
                    ks_n, _ = kstest(ensemble_clean,
                                     lambda x: norm.cdf(x, loc=normal_params['mu'][i],
                                                       scale=normal_params['sigma'][i]))
                    normal_ks.append(ks_n)
                    
                    # LogNormal KS (only if all positive)
                    if np.all(ensemble_clean > 0):
                        ks_ln, _ = kstest(ensemble_clean,
                                         lambda x: lognorm.cdf(x, s=lognormal_params['sigma'][i],
                                                              scale=np.exp(lognormal_params['mu'][i])))
                        lognormal_ks.append(ks_ln)
                    else:
                        lognormal_ks.append(np.nan)
                else:
                    weibull_ks.append(np.nan)
                    normal_ks.append(np.nan)
                    lognormal_ks.append(np.nan)
            
            # Baseline forecasts
            y_pred_persistence = hist[:, -1]
            y_pred_persistence_max = np.max(hist, axis=-1)
            y_pred_27_day = np.max(recurrence, axis=-1)
            
            # Get center indices for this batch
            center_indices = batch['center_idx'].cpu().numpy()
            
            all_outputs.append({
                'y_test': y_true.ravel(),
                'y_pred_weighted_mean': y_pred_weighted_mean,
                'y_pred_weibull_median': weibull_params['median'],
                'y_pred_normal_median': normal_params['median'],
                'y_pred_lognormal_median': lognormal_params['median'],
                'y_pred_persistence': y_pred_persistence,
                'y_pred_persistence_max': y_pred_persistence_max,
                'y_pred_27_day_recurrence': y_pred_27_day,
                'lambda': weibull_params['lambda'],
                'k': weibull_params['k'],
                'mu': normal_params['mu'],
                'sigma': normal_params['sigma'],
                'log_mu': lognormal_params['mu'],
                'log_sigma': lognormal_params['sigma'],
                'center_idx': center_indices,
                'ensemble_predictions': predictions,
                'weibull_ks': weibull_ks,
                'normal_ks': normal_ks,
                'lognormal_ks': lognormal_ks,
            })
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            continue
    
    # Concatenate results
    results = {}
    for key in all_outputs[0]:
        results[key] = np.concatenate([d[key] for d in all_outputs], axis=0)
    
    # Map center_idx back to window indices in the filtered test set
    # This allows plotting code to work correctly
    results['window_idx'] = results['center_idx']
    
    return results


def run_training_pipeline(config: TrainingConfig):
    """
    Main training pipeline supporting both ensemble regression and MLP.
    
    Args:
        config: TrainingConfig object with all parameters
    """
    # Set up output paths
    if config.run_name is None:
        model_suffix = 'MLP' if config.model_type == 'mlp' else 'ensemble'
        config.run_name = datetime.datetime.now().strftime(f"run_{model_suffix}_%Y-%m-%d_%H-%M-%S")

    if config.output_folder is None:
        paths = get_project_paths()
        config.output_folder = (
            paths['regression_metrics'] / f'HUXt{config.huxt_run_id}'
        )
    
    config.output_folder.mkdir(parents=True, exist_ok=True)
    output_path = config.output_folder / f'{config.run_name}_metrics.csv'
    
    # Initialize metric writer
    metric_writer = MetricWriter(output_path)
    
    # Main training loops
    for random_seed in config.random_seeds:
        logger.info(f"\n{'='*80}\nRandom Seed: {random_seed}\n{'='*80}")
        
        for lead_time in config.lead_times:
            logger.info(f"\n{'-'*80}\nLead Time: {lead_time}h\n{'-'*80}")
            
            try:
                # Load dataset
                logger.info("Loading dataset...")
                dataset = ForecastingDataset(
                    parquet_path=str(config.huxt_data_path),
                    discontinuity_path=str(config.discontinuity_path) if config.discontinuity_path else None,
                    seed=random_seed,
                    Nens=config.n_ensembles,
                    lead_time_hours=lead_time,
                    forecast_duration_hours=config.forecast_duration_hours,
                    stride_hours=config.stride_hours
                )
                
                # Balance storms
                if config.balance:
                    logger.info("Balancing storm/non-storm samples...")
                    dataset.balance_storms(
                        threshold=config.storm_balance_threshold,
                        inplace=True,
                        random_state=random_seed
                    )

                if config.remove_cmes:
                    logger.info("Removing CME samples...")
                    dataset.remove_cmes(
                        inplace=True,
                    )
                
                # Log dataset statistics
                stats = dataset.get_storm_statistics()
                logger.info(f"Dataset stats: {stats['n_storms']} storms, "
                           f"{stats['n_non_storms']} non-storms")
                
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                continue
            
            # Test fold loop
            for test_fold in config.test_folds:
                logger.info(f"Test Fold: {test_fold}")

                try:
                    # Split data
                    train_indices, test_indices = dataset.rotation_aligned_train_test_split(
                        train_ratio=config.train_ratio,
                        test_fold=test_fold
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to split data: {e}")
                    continue
                
                try:
                    # Set OMNI columns
                    dataset.set_omni_columns(config.omni_subset)
                    
                    # ===== BRANCH: MLP vs Ensemble =====
                    if config.model_type == 'mlp':
                        logger.info("Using MLP model")
                        
                        # Create dataloaders
                        train_dataset = Subset(dataset, train_indices)
                        train_dataloader = DataLoader(
                            train_dataset,
                            batch_size=config.batch_size,
                            shuffle=True
                        )
                        
                        # Train MLP
                        mlp_model, scaler = train_mlp(
                            train_dataloader,
                            config.n_ensembles,
                            loss_config=config.loss_config,
                            include_ensemble_spread=config.mlp_include_ensemble_spread,
                            ensemble_percentiles = config.mlp_ensemble_percentiles,
                            ensemble_selection_method = config.ensemble_selection_method,
                            filter_ensemble = config.filter_ensemble,
                            n_ensemble_keep = config.n_ensemble_keep,
                            architecture=config.mlp_architecture,
                            n_epochs=config.mlp_n_epochs,
                            learning_rate=config.mlp_learning_rate,
                            device=config.mlp_device,
                            patience=config.mlp_patience,
                            debug=False,
                        )
                        
                        # Single "regressor" loop for MLP
                        regressor_name = f"MLP_{'stats' if config.mlp_include_ensemble_spread else 'flat'}"
                        models_dict = {regressor_name: (mlp_model, scaler)}
                        
                    else:  # ensemble regression
                        logger.info("Using ensemble regression")
                        
                        # Prepare training data
                        train_dataset = Subset(dataset, train_indices)
                        train_dataloader = DataLoader(
                            train_dataset,
                            batch_size=len(train_dataset),
                            shuffle=True
                        )
                        
                        train_batch = next(iter(train_dataloader))
                        X_train, scaler = prepare_features(
                            train_batch,
                            config.n_ensembles,
                            fit_scaler=True
                        )
                        y_train = train_batch['max_target'].numpy()
                        
                        # Train all ensemble regressors
                        models_dict = {}
                        for regressor_name, regressor_class in config.ensemble_regressors.items():
                            logger.info(f"Training {regressor_name}...")
                            models = train_ensemble_models(
                                X_train, y_train,
                                config.n_ensembles,
                                regressor_class,
                                n_jobs=config.n_jobs
                            )
                            models_dict[regressor_name] = (models, scaler)
                    
                except Exception as e:
                    logger.error(f"Failed to prepare training data: {e}")
                    continue
                    
                # Evaluate each model (MLP or ensemble regressors)
                for regressor_name, (model_or_models, scaler) in models_dict.items():
                    try:
                        logger.info(f"Evaluating {regressor_name}...")
                        logger.info(f"Test size: {len(test_indices)} ")
                        
                        # Create test dataloader
                        test_dataset = Subset(dataset, test_indices)
                        test_dataloader = DataLoader(
                            test_dataset,
                            batch_size=config.batch_size,
                            shuffle=False
                        )
                        
                        # Evaluate based on model type
                        if config.model_type == 'mlp':
                            results = evaluate_mlp_test_set(
                                model_or_models,
                                test_dataloader,
                                scaler,
                                config.n_ensembles,
                                include_ensemble_spread=config.mlp_include_ensemble_spread,
                                ensemble_percentiles=config.mlp_ensemble_percentiles,
                                ensemble_selection_method=config.ensemble_selection_method,
                                filter_ensemble=config.filter_ensemble,
                                n_ensemble_keep=config.n_ensemble_keep,
                                device=config.mlp_device,
                            )
                        else:
                            results = evaluate_test_set(
                                model_or_models,
                                test_dataloader,
                                scaler,
                                config.n_ensembles,
                                test_indices,
                                model_name=regressor_name,
                                model_constraints=config.model_constraints
                            )
                        
                        # Save results
                        if config.save_results:
                            save_results(
                                results=results,
                                dataset=dataset,
                                config=config,
                                random_seed=random_seed,
                                lead_time=lead_time,
                                test_fold=test_fold,
                                storm_threshold=4.5,
                                test_indices=test_indices,
                                regressor_name=regressor_name,
                                save_metadata=getattr(config, 'save_metadata', True)
                            )
                        
                        # Compute metrics
                        config_params = {
                            'random_seed': random_seed,
                            'test_fold': test_fold,
                            'n_ensembles': config.n_ensembles,
                            'lead_time': lead_time,
                            'ensemble_regressor': regressor_name,
                            'omni_subset': '-'.join(config.omni_subset),
                        }
                        
                        # For MLP, only evaluate LogNormal median
                        if config.model_type == 'mlp':
                            aggregators = {
                                'lognormal_median': results['y_pred_lognormal_median'],
                                'persistence': results['y_pred_persistence'],
                                'persistence_max': results['y_pred_persistence_max'],
                                '27_day_recurrence': results['y_pred_27_day_recurrence'],
                            }
                        else:
                            aggregators = {
                                'normal_median': results['y_pred_normal_median'],
                                'weibull_median': results['y_pred_weibull_median'],
                                'lognormal_median': results['y_pred_lognormal_median'],
                                'weighted_mean': results['y_pred_weighted_mean'],
                                'persistence': results['y_pred_persistence'],
                                'persistence_max': results['y_pred_persistence_max'],
                                '27_day_recurrence': results['y_pred_27_day_recurrence'],
                            }
                        
                        for agg_name, y_pred in aggregators.items():
                            config_params['aggregator'] = agg_name
                            
                            # Calculate deterministic metrics
                            metrics = evaluate_regression_forecast(y_pred, results['y_test'])
                            
                            # Calculate CRPS for LogNormal
                            crps_value = None
                            dist_params = None
                            
                            if agg_name == 'lognormal_median':
                                prob_metrics = evaluate_distribution_forecast(
                                    results['y_test'],
                                    distribution='lognormal',
                                    log_mu_pred=results['log_mu'],
                                    log_sigma_pred=results['log_sigma']
                                )
                                crps_value = prob_metrics['crps']
                                metrics['mean_ks'] = 0.0
                                dist_params = {
                                    'log_mu': compute_distribution_statistics(results['log_mu']),
                                    'log_sigma': compute_distribution_statistics(results['log_sigma'])
                                }
                            
                            elif config.model_type == 'ensemble':
                                if agg_name == 'weibull_median':
                                    prob_metrics = evaluate_distribution_forecast(
                                        results['y_test'],
                                        distribution='weibull',
                                        lambda_pred=results['lambda'],
                                        k_pred=results['k']
                                    )
                                    crps_value = prob_metrics['crps']
                                    metrics['mean_ks'] = np.nanmean(results['weibull_ks'])
                                    dist_params = {
                                        'lambda': compute_distribution_statistics(results['lambda']),
                                        'k': compute_distribution_statistics(results['k'])
                                    }
                                
                                elif agg_name == 'normal_median':
                                    prob_metrics = evaluate_distribution_forecast(
                                        results['y_test'],
                                        distribution='normal',
                                        mu_pred=results['mu'],
                                        sigma_pred=results['sigma']
                                    )
                                    crps_value = prob_metrics['crps']
                                    metrics['mean_ks'] = np.nanmean(results['normal_ks'])
                                    dist_params = {
                                        'mu': compute_distribution_statistics(results['mu']),
                                        'sigma': compute_distribution_statistics(results['sigma'])
                                    }
                            
                            metric_writer.write_metrics(config_params, metrics, crps_value, dist_params)
                        
                        logger.info(f"Evaluation complete for {regressor_name}")
                
                    except Exception as e:
                        logger.error(f"Failed evaluation for {regressor_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    
                    # Clean up models to free memory
                    if config.model_type == 'ensemble':
                        del model_or_models

    logger.info(f"\n{'='*80}\nTraining complete! Metrics saved to: {output_path}\n{'='*80}")

# Note: get_project_paths is imported from storm_utils.config_paths
# No need to redefine it here


if __name__ == "__main__":
    # Get project paths
    paths = get_project_paths()
    
    # Configure training
    config = TrainingConfig(
        huxt_run_id=1,
        huxt_data_path=paths['huxt_data_shared'] / f'HUXt{huxt_run_id}_modified' / 'full_df.parquet',
        discontinuity_path=paths['huxt_data_shared'] / f'HUXt{huxt_run_id}_modified' / 'discontinuities.npy',
        
        # Model parameters
        n_ensembles=100,
        ensemble_regressors={'LinearRegression': LinearRegression},
        
        # Experiment parameters
        lead_times=[12],
        random_seeds=[42],
        test_folds=[0],
        omni_subset=["Bz_GSM"],
        
        # Output
        output_folder=paths['regression_src'] / 'figures' / 'metric_tables'
    )
    
    # Run training
    try:
        run_training_pipeline(config)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)