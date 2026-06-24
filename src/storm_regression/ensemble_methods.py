# Model prediction function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import os

import storm_regression.ensemble_analysis as EnsA

from storm_utils.config_paths import get_project_paths
from storm_utils.data_loader import load_omni_data
from storm_utils.data_functions import train_test_val_split, balance_data


def run_experiment(
    overwrite,
    huxt_run_id,
    testing_run_id,
    target,
    variables,
    train_threshold,
    test_threshold,
    input_window_size,
    input_buffer_size,
    output_window_size,
    post_window_size,
    stride,
    n_ensembles,
    balance,
    scale_p,
    model_class,
    final_name,
    X_windows,
    y_windows,
    times,
    seed
):
    """
    Call this function to make a forecast model using the given input parameters

    input: 
    - overwrite           : boolean - Overwrite metrics when True
    - huxt_run_id         : int - id for HUXt run to use
    - testing_run_id      : int - id for saving model outputs
    - target              : str - target variable name ('hp30' or 'hp60')
    - variables           : np.array - array of variable names to include in the model
    - train_threshold     : float - hp30 threshold that defines a storm (for training purposes)
    - test_threshold      : float - hp30 threshold that defines a storm (for testing purposes)
    - input_window_size   : int - length of input window (in hours)
    - input_buffer_size   : int - length of buffer zone (in hours)
    - output_window_size  : int - length of output window (in hours)
    - post_window_size    : int - length of post window (for plotting)
    - stride              : int - time between starts of successive output_windows
    - n_ensembles         : int - number of HUXt ensembles to use
    - balance             : bool - balances storm and non-storm output windows when True
    - scale_p             : bool - scales ensemble probabilities when True
    - model_class         : sklearn model - Regression model from sklearn used to train regression ensemble
    - final_name          : string - model name for final_regression
    - X_windows           : windows of training parameters
    - y_windows           : windows of target variable
    - times               : times corresponding to the windows
    - seed                : int - random seed for numpy
    """
    paths = get_project_paths()
    # Prep file name
    save_name = f'run_{testing_run_id:03}'
    metric_table_path = paths['regression_src'] / 'figures' / f'{save_name}_metrics.csv'
    
    if overwrite:
        with open(metric_table_path, 'w') as f:
            # write run parameters to metric table
            f.write('Variables,Input Window Size (hours),Input Buffer Size (hours),Output Window Size (hours),Balanced,N_ensembles,Sorted,Scaled,Final Regressor,Storm Test Threshold,')
            # Write the metric names to metric table
            f.write(','.join(EnsA.evaluate_predictions(np.array([[1], [0]]), np.array([[1], [0]])).keys()))

    # Ensure 'target' is actual name, not a placeholder
    if 'target' in variables:
        variables.remove('target')
        variables.append(target)

    if final_name in ['persistence', '27_day_persistence']:
        nens = 1

    with open(metric_table_path, 'a') as f:

        # Balance the data
        if balance:
            X_windows_balanced, y_windows_balanced, times_balanced = balance_data(X_windows, y_windows, times, input_window_size=input_window_size, post_window_size=post_window_size, storm_threshold=train_threshold)
            # Set target variable as binary
            y = convert_hpo_to_boolean(y_windows_balanced, input_window_size=input_window_size, post_window_size=post_window_size, storm_threshold=train_threshold)
            # Split to train, val and test sets
            X_train, X_val, X_test, y_train_hpo, y_val_hpo, y_test_hpo, times_train, times_val, times_test = train_test_val_split(X_windows_balanced, y_windows_balanced, times_balanced, seed)
            
        else: 
            y = convert_hpo_to_boolean(y_windows, input_window_size=input_window_size, post_window_size=post_window_size)
            X_train, X_val, X_test, y_train_hpo, y_val_hpo, y_test_hpo, times_train, times_val, times_test = train_test_val_split(X_windows, y_windows, times, seed)
    
        scaler = MinMaxScaler(feature_range=(0, 1))                                                                                                                          
        def refactor_windows(X, y_hpo, name=''): 
            X[:,:,input_window_size-input_buffer_size:,-1] = 0   
    
            # Remove post_window from our input variables
            X_without_post = X[:,:,:-post_window_size]
            
            # Extract whether there was a storm or not based on hp60 values in the output window
            y_bool = convert_hpo_to_boolean(y_hpo, input_window_size=input_window_size, post_window_size=post_window_size)
    
            # Get target var during output window
            y = y_bool
            
            # Combine dimensions for rescaling
            X_reshaped = X_without_post.reshape(-1, X_without_post.shape[-1])
            
            # Scale the data and put it back to original shape
            scaled_X = scaler.fit_transform(X_reshaped).reshape(X_without_post.shape)
    
            # Combine last 2 dimensions
            scaled_X_reshaped = scaled_X.reshape(scaled_X.shape[:-2] + (-1,))
    
            # Remove (V - OMNI) for buffer and output window
            scaled_X_reshaped = scaled_X_reshaped[:, :, :-(input_buffer_size + output_window_size)]
    
            X_hpo = y_hpo[:, :input_window_size-input_buffer_size] if target in variables else None
    
            print(name, 'ensemble input shape', scaled_X_reshaped.shape)
    
            return X, scaled_X, scaled_X_reshaped, X_hpo, y, y_bool
    
        def remove_small_storms(X, y_hpo, times, storm_testing_threshold):
            ''' Removes storms above training threshold and below testing threshold '''
            # get the max hpo value for each output window
            y_max_hpo = np.max(y_hpo[:, input_window_size:-post_window_size], axis=1)
    
            # Extract indices for when we exceed large storm threshold and when we don't exceed storm threshold
            storm_indices = np.where(y_max_hpo >= storm_testing_threshold)[0]
            non_storm_indices = np.where(y_max_hpo < storm_training_threshold)[0]
            
            # Randomly drop non-storms to balance with the storm times
            non_storm_indices = np.random.choice(non_storm_indices, size=len(storm_indices), replace=False)
    
            # Combine indices
            all_indices = np.concatenate((storm_indices, non_storm_indices))
    
            # Extract correct parts of our arrays
            X_removed = X[all_indices]
            y_hpo_removed = y_hpo[all_indices]
            times_removed = times[all_indices]
    
            return X_removed, y_hpo_removed, times_removed
        
        def get_maes(X):
            X_input = X[:, :, :input_window_size - input_buffer_size, -1]
            maes = np.mean(np.abs(X_input), axis=-1)
            return maes
    
        # Remove storms based on testing threshold
        if test_threshold != train_threshold:
            X_test, y_test_hpo, times_test = remove_small_storms(X_test, y_test_hpo, times_test, storm_testing_threshold=test_threshold)
        
        # Extract arrays needed 
        print('Refactoring...')
        X_train, scaled_X_train, scaled_X_train_reshaped, X_train_hpo, y_train, y_train_bool = refactor_windows(X_train, y_train_hpo, 'train')
        X_val, scaled_X_val, scaled_X_val_reshaped, X_val_hpo, y_val, y_val_bool = refactor_windows(X_val, y_val_hpo, 'validation')
        X_test, scaled_X_test, scaled_X_test_reshaped, X_test_hpo, y_test, y_test_bool = refactor_windows(X_test, y_test_hpo, 'test')
        
        # Create ensemble models
        model_params = {'max_iter':2000}
        print("Training Regressions...")
        model_array = create_ensemble_models(scaled_X_train_reshaped, X_train_hpo, y_train, model_class, model_params)
    
        # Make ensemble predictions
        print('Making ensemble predictions...')
        
        train_predictions = make_ensemble_predictions(scaled_X_train_reshaped, X_train_hpo, y_train, model_array, predict_probabilities=True)
        test_predictions = make_ensemble_predictions(scaled_X_test_reshaped, X_test_hpo, y_test, model_array, predict_probabilities=True)
        val_predictions = make_ensemble_predictions(scaled_X_val_reshaped, X_val_hpo, y_val, model_array, predict_probabilities=True)
    
        # Decide whether to sort final_regression input by MAE
        if final_name in ['logreg_sorted', 'attention_NN'] or final_name[:11] == 'logreg_top_':
            sort=True
        else:
            sort=False
    
        # Find MAES for input window
        train_maes = get_maes(X_train)
        val_maes = get_maes(X_val)
        test_maes = get_maes(X_test)
    
        if sort: 
            # Sort indices by their associated MAE
            train_indices = np.argsort(train_maes, axis=1)
            val_indices = np.argsort(val_maes, axis=1)
            test_indices = np.argsort(test_maes, axis=1)
    
            # Apply sorted indices to arrays
            sorted_train_predictions = np.take_along_axis(train_predictions, train_indices, axis=1)
            sorted_val_predictions = np.take_along_axis(val_predictions, val_indices, axis=1)
            sorted_test_predictions = np.take_along_axis(test_predictions, test_indices, axis=1)
            sorted_train_maes = np.take_along_axis(train_maes, train_indices, axis=1)
            sorted_val_maes = np.take_along_axis(val_maes, val_indices, axis=1)
            sorted_test_maes = np.take_along_axis(test_maes, test_indices, axis=1)
    
            train_input = [sorted_train_predictions, sorted_train_maes]
            val_input = [sorted_val_predictions, sorted_val_maes]
            test_input = [sorted_test_predictions, sorted_test_maes]
    
        else:
            train_input = [train_predictions, train_maes]
            val_input = [val_predictions, val_maes]
            test_input = [test_predictions, test_maes]
    
        # Pass hpo for the input window 
        if final_name == 'persistence':
            test_input = y_test_hpo[:, :input_window_size-input_buffer_size]
    
        if final_name == '27_day_persistence':
            # Pass the times corresponding to the output window
            test_input = times_test[:, input_window_size:-post_window_size]
    
        # Train final regressor
        print('Training final regressor...')
        final_regressor = train_final_regressor(train_input, y_train, final_name)
    
        # Make probabilistic predictions
        print('Making final forecasts...')
        probabilistic_predictions = make_final_classifier_predictions(test_input, final_classifier, final_name, scale=scale_p)
        
        res = EnsA.evaluate_predictions(probabilistic_predictions, y_test)
        
        # Write metrics to file
        f.write('\n')
        f.write('-'.join(variables))
        cadence_factor = 2
        f.write(f',{input_window_size//cadence_factor},{input_buffer_size//cadence_factor},{output_window_size//cadence_factor},{balance},{n_ensembles},{sort},{scale_p},{final_name},{test_threshold},')
        f.write(','.join([str(i) for i in res.values()]))
    
        # Make plots from model output
        tag = f"i{input_window_size}_o{output_window_size}_s{stride}_buff{input_buffer_size}_{'_'.join(variables)}_nens{n_ensembles}_sorted_{sort}_scaled_{scale_p}_final_{final_name}_thresh_{test_threshold}"
            
        test_tupe = (probabilistic_predictions, X_test, y_test, y_test_hpo, times_test, train_predictions, train_maes, final_name)
    
        print(f'Finished training model - metrics saved to {metric_table_path}')
        return test_tupe


def create_ensemble_models(X_train, X_train_hpo, y_train, model_class, model_hyperparameters):
    """
    Calling this will create models of the specified class and fit them on the training data
    
    Input:
    - X_train     : array - training data of input variables
    - y_train     : array - training data of target variable
    - model_class : class - model Class imported from sklearn / sktime
    - model_hyperparameters : dict - contains the hyperparameters to be passed to the model

    Output:
    - model_array : array - ensemble of fitted models
    """
    
    model_array = []

    if X_train.shape[1] != y_train.shape[0]:
        X_train = np.transpose(X_train, (1, 0, 2))

    for X in X_train:
        if X_train_hpo is not None:
            X = np.hstack((X, X_train_hpo))
        model = model_class(**model_hyperparameters)
        model.fit(X, y_train)
        model_array.append(model)

    return model_array


def make_ensemble_predictions(X_train, X_train_hpo, y_train, model_array, predict_probabilities=False):
    """
    Calling this will train the array of models on the training data

    Input:
    - X_train     : array - training data of input variables
    - y_train     : array - training data of target variable
    - model_array : array - fitted models on the training data (i.e. the ensemble models)

    Output:
    - predictions : array - predictions based on the X and y arrays passed for each of the models
    """
    predictions = []

    if X_train.shape[1] != y_train.shape[0]:
        X_train = np.transpose(X_train, (1, 0, 2))

    for X, model in zip(X_train, model_array):
        if X_train_hpo is not None:
            X = np.hstack((X, X_train_hpo))
        p = model.predict(X)
        predictions.append(p)

    predictions = np.array(predictions).T
    return predictions


def train_final_regressor(X, y, model_type, X_val=None, y_val=None):
    """
    Calling this function will train a model on the given training data with the specified model type

    Input:
    - X          : array - training data of input variables
    - y          : array - training data of target variable
    - model_type : string - type of model to use. 
                    options:
                        - logreg
                        - simple_NN

    Output:
    - model : The trained model
    """

    if model_type in ['logreg', 'logreg_sorted']:
        p, MAE = X
        model = LogisticRegression()
        model.fit(p, y)

    elif model_type[:11] == 'logreg_top_':
        k = int(model_type[11:])
        Nens = X[0].shape[-1]
        n = int((k * Nens) // 100)
        X = np.concatenate([arr[:,:n] for arr in X], axis=-1)
        model = LogisticRegression()
        model.fit(X, y)

    elif model_type == 'persistence':
        model = None

    elif model_type == '27_day_persistence':
        model = None

    elif model_type == 'weighted_mean':
        model = None
    
    else:
        print(f'Model type: {model_type} not recognized')

    return model
    
    
def make_final_regressor_predictions(X, model, model_type, scale=False, storm_thresh=4.66):
    """
    Calling this function will train a model on the given training data with the specified model type

    Input:
    - X          : array - test data of input variables
    - model      : a trained model 
    - model_type : string - type of model as a string
    - scale      : bool - scales probabilities when True

    Output:
    - probabilistic_predictions : np.array - shape = (no. of samples, 1) 
    """
    # Scale data if we aren't using persistence
    if scale and model_type not in ['persistence', '27_day_persistence']: 
        s = StandardScaler()
        s.fit(X[0])
        X[0] = s.transform(X[0])
        
    if model_type == 'logreg' or model_type == 'logreg_sorted':
        p, MAE = X
        probabilistic_predictions = np.expand_dims(model.predict_proba(p)[:, 1], axis=1)

    elif model_type[:11] == 'logreg_top_':
        k = int(model_type[11:])
        Nens = X[0].shape[-1]
        n = int((k * Nens) // 100)
        X = np.concatenate([arr[:,:n] for arr in X], axis=-1)
        probabilistic_predictions = np.expand_dims(model.predict_proba(X)[:, 1], axis=1)

    elif model_type == 'persistence':
        probabilistic_predictions = np.expand_dims(np.max(X, axis=-1) > storm_thresh, axis=-1)

    elif model_type == '27_day_persistence':
        # Define paths
        data_dir = os.path.join(os.path.expanduser('~'), 'storm_forecasting', 'src', 'data')
        huxt_data_dir = os.path.join(data_dir, 'huxt', 'HUXt8_modified')

        # Read in hp30 only
        cols = ['hp30']
        twenty_seven_df = pd.read_parquet(os.path.join(huxt_data_dir, 'full_df.parquet'), engine='fastparquet', columns=cols)

        twenty_seven_df['27_day_offset_hp30'] = twenty_seven_df['hp30'].shift(27 * 24 * 2)
        
        shape = X.shape
        X = X.flatten()
    
        vals = twenty_seven_df['27_day_offset_hp30'].loc[X].to_numpy()
        vals = vals.reshape(shape)
    
        probabilistic_predictions = np.expand_dims(np.max(vals, axis=-1) > storm_thresh, axis=1)
    
    elif model_type == 'weighted_mean':
        p, mae = X
        p = np.clip(p, 0, 1)
        weights = 1 / mae**2
        weights /= np.sum(weights, axis=1, keepdims=True)
        probabilistic_predictions = np.sum(p * weights, axis=1, keepdims=True)

    return probabilistic_predictions

