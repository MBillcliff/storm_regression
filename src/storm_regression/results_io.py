"""
results_io.py — persistence for forecast results.

Saving/loading result pickles and reconstructing the dataset used for a run.
This module sits at the bottom of the dependency graph: it imports ONLY from
storm_utils + stdlib, never from training_pipeline / forecast_analysis / plotting,
so it can be safely imported anywhere without cycles.
"""
import logging
import pickle
import json
import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

from storm_utils.data_structure import ForecastingDataset, ForecastingConfig

logger = logging.getLogger(__name__)

def save_results(
    results: Dict,
    dataset: ForecastingDataset,
    config,  # TrainingConfig
    random_seed: int,
    lead_time: int,
    test_fold: int,
    storm_threshold: float,
    test_indices: np.array,
    regressor_name: str,
    constraint_method: Optional[str] = None,
    output_folder: Optional[Path] = None,
    save_metadata: Optional[bool] = True,
) -> Path:
    """
    Save results dictionary and metadata for later case study analysis.
    
    Model name is auto-generated based on config parameters.
    """
    from storm_utils.config_paths import get_project_paths
    paths = get_project_paths()
    
    # Determine output folder
    if output_folder is None:
        if hasattr(config, 'experiment_phase') and config.experiment_phase:
            output_folder = (
                paths['regression_results'] / f'HUXt{config.huxt_run_id}' / config.experiment_phase
            )
        else:
            output_folder = (
                paths['regression_results'] / f'HUXt{config.huxt_run_id}'
            )
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Build full model name based on model type
    if config.model_type == 'ensemble':
        model_name = regressor_name  # Just use base name: 'Ridge', 'LinearRegression'
        
    elif config.model_type == 'mlp':
        # Start with base name or custom name
        if hasattr(config, 'model_name') and config.model_name:
            base_name = config.model_name
        else:
            base_name = regressor_name  # Default to 'MLP'
        
        # Build descriptors list
        descriptors = []
        
        # Add ensemble selection info
        descriptors.append('filterEnsemble' + str(config.filter_ensemble))
        descriptors.append('nensToKeep' + str(config.n_ensemble_keep))
        descriptors.append('ensembleSelectionMethod' + str(config.ensemble_selection_method))

        perc_str = "p" + "-".join([str(int(p)) for p in config.mlp_ensemble_percentiles])
        descriptors.append(perc_str)
        
        # Add spread flag if used
        if config.mlp_include_ensemble_spread:
            descriptors.append("spread")
        
        # Add architecture if non-default (for Phase 2)
        if hasattr(config, 'mlp_architecture'):
            default_arch = [50, 50, 50]
            if config.mlp_architecture != default_arch:
                arch_str = "-".join([str(n) for n in config.mlp_architecture])
                descriptors.append(f"arch{arch_str}")
        
        # Add loss function if non-default (for Phase 3)
        if hasattr(config, 'mlp_loss_function') and config.mlp_loss_function != 'nll':
            descriptors.append(config.mlp_loss_function)

        if hasattr(config, 'loss_config'):
            descriptors.append('loss_type' + config.loss_config.loss_type)
            
            if config.loss_config.loss_type == 'nll':
                descriptors.append('w_sig' + str(config.loss_config.w_sigma))
                descriptors.append('w_acc' + str(config.loss_config.w_accuracy))
                descriptors.append('hp30_weighting' + str(config.loss_config.intensity_type))
                descriptors.append('hp30_scale' + str(config.loss_config.intensity_strength))

            elif config.loss_config.loss_type == 'crps':
                descriptors.append('nsamples' + str(config.loss_config.n_samples))
                descriptors.append('hp30_weighting' + str(config.loss_config.intensity_type))
                descriptors.append('hp30_scale' + str(config.loss_config.intensity_strength))

            elif config.loss_config.loss_type == 'asymmetric':
                descriptors.append('underpred' + str(config.loss_config.underpred_penalty))
                descriptors.append('overpred' + str(config.loss_config.overpred_penalty))
        
        # Combine base name with descriptors
        model_name = base_name + "_" + "_".join(descriptors)
    
    else:
        model_name = regressor_name  # Fallback
    
    # Determine balance mode
    balance_mode = 'balanced' if config.balance else 'unbalanced'
    
    # Create filename
    omni_str = '-'.join(config.omni_subset) if config.omni_subset else ''
    
    filename_parts = [
        'results',
        f"seed{random_seed}",
        f"lt{lead_time}",
        f"fold{test_fold}",
        f"thresh{storm_threshold}",
        f"{balance_mode}",
        f"cmesremoved{config.remove_cmes}",
        omni_str,
        model_name,
    ]

    if hasattr(config, 'run_name') and config.run_name:
        filename = f"results_{config.run_name}.pkl"
    else:
        # existing auto-generated filename logic
        filename = "_".join(filename_parts) + ".pkl"

    output_path = output_folder / filename
    
    # Build config dictionary
    config_dict = {
        'random_seed': random_seed,
        'lead_time': lead_time,
        'test_fold': test_fold,
        'n_ensembles': config.n_ensembles,
        'storm_threshold': storm_threshold,
        'balance_mode': balance_mode,
        'balance': config.balance,
        'omni_subset': config.omni_subset,
        'remove_cmes': getattr(config, 'remove_cmes', False),
        'model_name': model_name,  # Full constructed name
        'constraint_method': constraint_method,
        'test_indices': test_indices.tolist() if isinstance(test_indices, np.ndarray) else test_indices,
    }

    # Add loss function specific variables
    if hasattr(config, 'loss_config'):
        config_dict['loss_config'] = {
            k: v for k, v in vars(config.loss_config).items()
        }
    
    # Add MLP-specific metadata
    if config.model_type == 'mlp':
        config_dict.update({
            'mlp_ensemble_percentiles': config.mlp_ensemble_percentiles,
            'filter_ensemble': config.filter_ensemble,
            'n_ensemble_keep': config.n_ensemble_keep,
            'ensemble_selection_method': config.ensemble_selection_method,
            'mlp_include_ensemble_spread': config.mlp_include_ensemble_spread,
            'mlp_n_epochs': config.mlp_n_epochs,
            'mlp_learning_rate': config.mlp_learning_rate,
        })
        
        if hasattr(config, 'mlp_architecture'):
            config_dict['mlp_architecture'] = config.mlp_architecture
        if hasattr(config, 'mlp_loss_function'):
            config_dict['mlp_loss_function'] = config.mlp_loss_function
    
    # Package and save
    package = {
        'results': results,
        'test_indices': test_indices,
        'config': config_dict,
        'dataset_params': {
            'huxt_data_path': str(config.huxt_data_path),
            'discontinuity_path': str(config.discontinuity_path) if config.discontinuity_path else None,
            'forecast_duration_hours': config.forecast_duration_hours,
            'stride_hours': config.stride_hours,
            'storm_balance_threshold': config.storm_balance_threshold,
        }
    }
    
    # Save main pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(package, f)

    logger.info(f"Results saved to: {output_path}")

    if save_metadata:
        # Save JSON metadata for easy inspection
        metadata = {
            'model_name': model_name,
            'timestamp': datetime.datetime.now().isoformat(),
            'config': config_dict,
            'file_path': str(output_path),
            'experiment_phase': config.experiment_phase if hasattr(config, 'experiment_phase') else None,
        }
        
        metadata_file = output_folder / (filename.replace('.pkl', '_metadata.json'))
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
        logger.info(f"Metadata saved to: {metadata_file}")
    
    return output_path


def load_results(results_path: Path) -> Tuple[Dict, Dict, Dict]:
    """
    Load saved results for case study analysis.
    
    Args:
        results_path: Path to saved results pickle file
    
    Returns:
        Tuple of (results_dict, config_dict, dataset_params_dict)
    """
    with open(results_path, 'rb') as f:
        package = pickle.load(f)
    
    logger.info(f"Loaded results from: {results_path}")
    return package['results'], package['config'], package['dataset_params']


def recreate_dataset_from_results(results_path: Path) -> ForecastingDataset:
    """
    Recreate the ForecastingDataset from saved results.
    
    Applies the same preprocessing as during training:
        - balance_storms (if config['balance'] is True)
        - remove_cmes (if config['remove_cmes'] is True)
    
    Args:
        results_path: Path to saved results pickle file
    
    Returns:
        Reconstructed ForecastingDataset instance
    """
    _, config, dataset_params = load_results(results_path)
    
    logger.info("Recreating dataset from saved parameters...")
    
    dataset = ForecastingDataset(
        parquet_path=dataset_params['huxt_data_path'],
        discontinuity_path=dataset_params['discontinuity_path'],
        seed=config['random_seed'],
        Nens=config['n_ensembles'],
        lead_time_hours=config['lead_time'],
        forecast_duration_hours=dataset_params['forecast_duration_hours'],
        stride_hours=dataset_params['stride_hours']
    )
    
    # Apply same preprocessing as during training — order matters
    if config.get('balance', False):
        logger.info("Applying storm balancing...")
        dataset.balance_storms(
            threshold=dataset_params['storm_balance_threshold'],
            inplace=True,
            random_state=config['random_seed']
        )

    if config.get('remove_cmes', False):
        logger.info("Removing CME windows...")
        dataset.remove_cmes(
            inplace=True,
        )
    
    # Set OMNI columns
    dataset.set_omni_columns(config['omni_subset'])
    
    logger.info("Dataset recreated successfully")
    return dataset