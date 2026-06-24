"""
case_study_analysis.py

Case study analysis tools for loading and analyzing saved forecast results.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable, List
import numpy as np
import datetime

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


def analyze_top_n_cases(
    results_path: Path,
    n: int = 5,
    sort_by: str = 'y_test',
    descending: bool = True,
    plot_function: Optional[Callable] = None
) -> Dict:
    """
    Analyze top N cases from saved results.
    
    Args:
        results_path: Path to saved results file
        n: Number of top cases to analyze
        sort_by: What to sort by ('y_test', 'y_pred_weibull_median', 'lambda', etc.)
        descending: If True, sort largest to smallest (for strongest storms)
        plot_function: Optional plotting function(dataset, results, window_position, test_idx)
    
    Returns:
        Dictionary with case study information including:
            - cases: List of case dictionaries
            - config: Configuration used
            - results: Full results dictionary
            - dataset: Reconstructed dataset
    """
    # Load results and recreate dataset
    results, config, _ = load_results(results_path)
    dataset = recreate_dataset_from_results(results_path)
    
    print(f"\n{'='*80}")
    print(f"Analyzing top {n} cases sorted by {sort_by}")
    print(f"{'='*80}")
    print(f"Config: seed={config['random_seed']}, lead_time={config['lead_time']}h, "
          f"test_fold={config['test_fold']}")
    print(f"OMNI features: {config['omni_subset']}")
    print(f"Test mode: {config['test_mode']}, threshold: {config['storm_threshold']}")
    
    # Sort and get top N
    sort_values = results[sort_by].squeeze()
    if descending:
        test_top_n = np.argsort(sort_values)[-n:][::-1]
    else:
        test_top_n = np.argsort(sort_values)[:n]
    
    # Get center indices (DataFrame positions, i.e., index of timestep of T0 in the full dataframe)
    global_top_n = results['window_idx'][test_top_n]
    
    # Analyze each case
    cases = []
    for i, (center_idx, test_idx) in enumerate(zip(global_top_n, test_top_n)):
        # Convert DataFrame position (center_idx) to window position
        try:
            window_position = dataset.valid_indices.index(int(center_idx))
        except ValueError:
            logger.error(f"Center index {center_idx} not found in valid_indices - skipping")
            continue
        
        # Extract values
        case_info = {
            'rank': i + 1,
            'window_position': int(window_position),  # For dataset[window_position]
            'center_idx': int(center_idx),            # For df.index[center_idx]
            'test_idx': int(test_idx),                # For results[test_idx]
            'y_true': float(results['y_test'][test_idx]),
            'y_pred_weibull_median': float(results['y_pred_weibull_median'][test_idx]),
            'y_pred_normal_median': float(results['y_pred_normal_median'][test_idx]),
            'y_pred_weighted_mean': float(results['y_pred_weighted_mean'][test_idx]),
            'y_pred_persistence': float(results['y_pred_persistence'][test_idx]),
            'lambda': float(results['lambda'][test_idx]),
            'k': float(results['k'][test_idx]),
            'mu': float(results['mu'][test_idx]),
            'sigma': float(results['sigma'][test_idx]),
        }
        
        cases.append(case_info)
        
        # Print summary
        print(f"\n{'-'*80}")
        print(f"Case #{i+1} - Storm Strength: {case_info['y_true']:.2f}")
        print(f"{'-'*80}")
        print(f"  Window Position: {window_position} (in dataset)")
        print(f"  Center Index: {center_idx} (in DataFrame)")
        print(f"  Test Index: {test_idx} (in results)")
        print(f"  Weibull Distribution: λ={case_info['lambda']:.2f}, k={case_info['k']:.2f}")
        print(f"  Normal Distribution:  μ={case_info['mu']:.2f}, σ={case_info['sigma']:.2f}")
        print(f"  Forecasts:")
        print(f"    - Weibull median:  {case_info['y_pred_weibull_median']:.2f}"
              f"(error: {abs(case_info['y_true'] - case_info['y_pred_weibull_median']):.2f})")
        print(f"    - Normal median:   {case_info['y_pred_normal_median']:.2f}"
              f"(error: {abs(case_info['y_true'] - case_info['y_pred_normal_median']):.2f})")
        print(f"    - Weighted mean:   {case_info['y_pred_weighted_mean']:.2f}"
              f"(error: {abs(case_info['y_true'] - case_info['y_pred_weighted_mean']):.2f})")
        print(f"    - Persistence:     {case_info['y_pred_persistence']:.2f}"
              f"(error: {abs(case_info['y_true'] - case_info['y_pred_persistence']):.2f})")
        
        # Plot if function provided
        if plot_function is not None:
            try:
                # Pass window_position (for dataset access), test_idx (for results access)
                plot_function(dataset, results, window_position, test_idx, config=config)
            except Exception as e:
                logger.error(f"Error plotting case {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*80}\n")
    
    return {
        'cases': cases,
        'config': config,
        'results': results,
        'dataset': dataset
    }

def diagnose_dataset_integrity(dataset, config):
    """
    Check dataset integrity and alignment of indices/labels.
    
    Parameters
    ----------
    dataset : ForecastingDataset
        The dataset object
    config : dict
        Config dictionary from results
        
    Returns
    -------
    bool
        True if all checks pass, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"DATASET INTEGRITY CHECK")
    print(f"{'='*80}")
    
    issues = []
    
    # Check 1: Basic sizes
    print(f"\n1. Basic Size Checks:")
    print(f"   len(dataset): {len(dataset)}")
    print(f"   len(dataset.valid_indices): {len(dataset.valid_indices)}")
    print(f"   len(dataset.window_labels): {len(dataset.window_labels)}")
    print(f"   len(dataset.max_targets): {len(dataset.max_targets)}")
    print(f"   len(config['test_indices']): {len(config['test_indices'])}")
    
    if len(dataset.window_labels) != len(dataset.valid_indices):
        issues.append(f"window_labels ({len(dataset.window_labels)}) != valid_indices ({len(dataset.valid_indices)})")
    
    if len(dataset.max_targets) != len(dataset.valid_indices):
        issues.append(f"max_targets ({len(dataset.max_targets)}) != valid_indices ({len(dataset.valid_indices)})")
    
    # Check 2: Test indices validity
    print(f"\n2. Test Indices Validity:")
    test_indices = config['test_indices']
    min_test_idx = min(test_indices)
    max_test_idx = max(test_indices)
    print(f"   Min test index: {min_test_idx}")
    print(f"   Max test index: {max_test_idx}")
    print(f"   Dataset size: {len(dataset)}")
    
    if max_test_idx >= len(dataset):
        issues.append(f"Max test index ({max_test_idx}) >= dataset size ({len(dataset)})")
    
    if max_test_idx >= len(dataset.window_labels):
        issues.append(f"Max test index ({max_test_idx}) >= window_labels size ({len(dataset.window_labels)})")
    
    # Check 3: Sample a few test indices
    print(f"\n3. Sampling Test Indices (first 5):")
    for i in range(min(5, len(test_indices))):
        test_idx = test_indices[i]
        
        try:
            valid_idx = dataset.valid_indices[test_idx]
            label = dataset.window_labels[test_idx]
            max_target = dataset.max_targets[test_idx]
            print(f"   test_indices[{i}] = {test_idx}")
            print(f"     → valid_indices[{test_idx}] = {valid_idx}")
            print(f"     → window_labels[{test_idx}] = {label}")
            print(f"     → max_targets[{test_idx}] = {max_target:.2f}")
        except IndexError as e:
            issues.append(f"IndexError accessing test_indices[{i}] = {test_idx}: {e}")
            print(f"   test_indices[{i}] = {test_idx} → ERROR: {e}")
    
    # Check 4: Last few test indices
    print(f"\n4. Sampling Test Indices (last 5):")
    for i in range(max(0, len(test_indices) - 5), len(test_indices)):
        test_idx = test_indices[i]
        
        try:
            valid_idx = dataset.valid_indices[test_idx]
            label = dataset.window_labels[test_idx]
            max_target = dataset.max_targets[test_idx]
            print(f"   test_indices[{i}] = {test_idx}")
            print(f"     → valid_indices[{test_idx}] = {valid_idx}")
            print(f"     → window_labels[{test_idx}] = {label}")
            print(f"     → max_targets[{test_idx}] = {max_target:.2f}")
        except IndexError as e:
            issues.append(f"IndexError accessing test_indices[{i}] = {test_idx}: {e}")
            print(f"   test_indices[{i}] = {test_idx} → ERROR: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    if len(issues) > 0:
        print(f"FAILED: Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  ❌ {issue}")
        print(f"{'='*80}\n")
        return False
    else:
        print(f"✓ All integrity checks passed!")
        print(f"{'='*80}\n")
        return True

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
            
            # # Print summary
            # print(f"\n{'-'*80}")
            # print(f"Case #{i+1} - Storm Strength: {case_info['y_true']:.2f}")
            # print(f"{'-'*80}")
            # print(f"  Window Position: {window_position} (for dataset[{window_position}])")
            # print(f"  Center Index: {center_idx} (for df.index[{center_idx}])")
            # print(f"  Test Index: {test_idx} (for results[{test_idx}])")
            # print(f"  Timestamp: {dataset.df.index[center_idx]}")
            # try:
            #     print(f"  Weibull Distribution: λ={case_info['lambda']:.2f}, k={case_info['k']:.2f}")
            #     print(f"  Normal Distribution:  μ={case_info['mu']:.2f}, σ={case_info['sigma']:.2f}")
            #     print(f"  Forecasts:")
            
            #     print(f"    - Weibull median:  {case_info['y_pred_weibull_median']:.2f} "
            #           f"(error: {abs(case_info['y_true'] - case_info['y_pred_weibull_median']):.2f})")
            #     print(f"    - Normal median:   {case_info['y_pred_normal_median']:.2f} "
            #           f"(error: {abs(case_info['y_true'] - case_info['y_pred_normal_median']):.2f})")
            #     print(f"    - Weighted mean:   {case_info['y_pred_weighted_mean']:.2f} "
            #           f"(error: {abs(case_info['y_true'] - case_info['y_pred_weighted_mean']):.2f})")
            # except: 
            #     print('MLP - so no weibull, normal, or weighted mean')
            # print(f"    - Persistence:     {case_info['y_pred_persistence']:.2f} "
            #       f"(error: {abs(case_info['y_true'] - case_info['y_pred_persistence']):.2f})")
            # print(f"    - MLP:     {case_info['y_pred_lognormal_median']:.2f} "
            #       f"(error: {abs(case_info['y_true'] - case_info['y_pred_lognormal_median']):.2f})")
            
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


# Example usage
if __name__ == "__main__":
    from storm_utils.config_paths import get_project_paths
    
    paths = get_project_paths()
    
    # Example: Find a saved results file
    results_folder = paths['regression_src'] / 'figures' / 'results'
    
    # List available result files
    if results_folder.exists():
        result_files = list(results_folder.glob('*.pkl'))
        print(f"Found {len(result_files)} result files:")
        for f in result_files:
            print(f"  - {f.name}")
        
        # Analyze the first one as an example
        if result_files:
            print(f"\nAnalyzing: {result_files[0].name}")
            case_study = analyze_top_n_cases(
                result_files[0],
                n=5,
                sort_by='y_test',
                descending=True
            )
            
            # Show summary
            print("\nCase Study Summary:")
            for case in case_study['cases']:
                print(f"  Rank {case['rank']}: True={case['y_true']:.2f}, "
                      f"Predicted={case['y_pred_weibull_median']:.2f}")
    else:
        print(f"Results folder not found: {results_folder}")
        print("Run training with save_results=True first!")