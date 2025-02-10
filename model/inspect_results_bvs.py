import pymc as pm
import arviz as az
import xarray as xr
import pandas as pd
import sqlite3
import statsmodels.api as sm

import matplotlib.pyplot as plt


def load_inference_data(file_path):
    inference_df = pd.load_csv(file_path)
    return inference_df


def extract_inference_results(inference_results):
    posterior_avg_df  = inference_results.to_dataframe()
    sample_stats_avg_df = inference_results.sample_stats.to_dataframe()
    observed_data_avg_df = inference_results.observed_data.to_dataframe()
    
    return posterior_avg_df, sample_stats_avg_df, observed_data_avg_df


def plot_trace(inference_results, var_names=["beta", "beta_raw", "sigma", "ind", "mu"]):
    az.plot_trace(inference_results, var_names=var_names);

    
def plot_forest(inference_results, var_names=["beta"]):
    az.plot_forest(inference_results, var_names=var_names, combined=True, hdi_prob=0.95, r_hat=True);
    
    
def plot_ess(summary_df, variables, states):
    """Plots Effective Sample Size (ESS) for bulk and tail of
    specified variables across different covariate values.

    Args:
      summary_df (pd.DataFrame):
          A DataFrame containing the summary statistics 
          including ESS values for different variables and 
          covariates (states or variables).
          
      variables (list):
          A list of variable names to plot the ESS for.
          
      states (list):
          A list of state names (if plotting ESS by state).

    Returns:
        None:
            This function creates plots and does not return any value.
      
    Typical Usage Example: 
        This allows you to visually assess the convergence of the
        MCMC sampler for each variable and state, helping you identify
        potential issues with model fit or sampling efficiency.      
        
        plot_ess(summary_df, inference_variables, inference_states)
    """

    
    for i, var_name in enumerate(['beta', 'beta_raw', 'ind', 'mu']):
        pattern = f'{var_name}\[(\d+)\]'

        filtered_df = summary_df[summary_df.index.str.match(pattern)]
        
        fig, ax = plt.subplots(figsize=(20, 5), nrows=2, ncols=1, sharex=True)
        fig.suptitle(f"ESS bulk and tail: {var_name}")
            
        if var_name in ['beta', 'beta_raw', 'ind']:
            ax[0].plot(variables, filtered_df['ess_bulk'])
            ax[0].set_title("ess bulk")
            ax[1].plot(variables, filtered_df['ess_tail'])
            ax[0].set_title("ess tail")
            # plt.xlim(0, len(filtered_df['ess_bulk']))

            plt.xticks(rotation=90)
            plt.show()
            
        else: 
            ax[0].plot(states, filtered_df['ess_bulk'])
            ax[0].set_title("ess bulk")
            ax[1].plot(states, filtered_df['ess_tail'])
            ax[0].set_title("ess tail")
            # plt.xlim(0, len(filtered_df['ess_bulk']))

            plt.xticks(rotation=45)
            plt.show() 
            
            
            
def plot_importance(
    importance,
    x_labels,
    importance_type
):
    """Creates a bar plot to visualize the importance of features.

    Args:
        importance (list):
            A list containing the importance scores for each feature.

        x_labels (list):
            A list of corresponding labels for the features.

        importance_type (str):
            A string describing the type of importance (e.g., gain, weight, etc.).

    Returns:
        None:
            This function creates a plot and does not return any value.
          
    Typical Usage Example: 
        This code will generate a bar chart representing the importance
        scores for each feature. The x-axis will display the feature labels
        rotated for readability, and the y-axis will represent the importance score.
    
        plot_importance(normalized_importance_1, x_labels, importance_type="Importance: Beta & Ind")
    
    """   
    plt.plot(importance, 'o')
    plt.suptitle(f"{importance_type}")
    plt.xlabel(x_labels)
    plt.xticks(range(len(importance)), x_labels, rotation=90)
    plt.show()
    
    
def get_summary(inference_results):
    summary_df = az.summary(inference_results, round_to=4)
    return summary_df
    
    
def get_specific_inference_data(summary_df, col_type='mean', col='beta'):
    """
    col_type = mean or sd
    col = beta, beta_raw, mu, ind 
    """
    new_stat_df = summary_df[col_type]
    
    data = new_stat_df.loc[
    new_stat_df.index.str.startswith(col)
]
    return data
    
    
def get_post_samp_obs_data(inference_results):
    posterior_avg_df  = inference_results.to_dataframe()
    sample_stats_avg_df = inference_results.sample_stats.to_dataframe()
    observed_data_avg_df = inference_results.observed_data.to_dataframe()
    
    return posterior_avg_df, sample_stats_avg_df, observed_data_avg_df


def calculate_importance(ind_means, beta_raw):
    importance = np.multiply(ind_means.values.flatten(), beta_raw.values.flatten())
    
    return importance


import pandas as pd
import statsmodels.api as sm

def calculate_p_values(
    df_vars,
    series_target,
    scale_function="z-score",
    imputer_strategy: str = 'mean'
):
    """
    Calculates p-values for a linear regression model.

    Args:
        df_vars: Pandas DataFrame with predictor columns.
        series_target: Pandas Series with the target variable.

    Returns:
        A Pandas Series containing the p-values for each predictor,
        or None if there's an issue with the model.
    """
    # Extract data from DataFrames
    y = series_target.values  # Long COVID rates
    X = df_vars.values  # Variables affecting/not affecting LC rates

    # If scale predictor variables (df_vars)
    # Scale only the variables which are not binary
    # scaling predictor variables, especially non-binary ones, can be beneficial
    # for Bayesian variable selection using Monte Carlo simulations.
    # It improves convergence and interpretability. Scaling binary variables, however, 
    # Can introduce unecessary complexity.
    if scale_function: 
        # Z-score: If your data is normally distributed and you want to
        # preserve the relative distances between data points.
        if scale_function == "z-score": 
            scaler = StandardScaler() 
            
        # Min-Max: If you want to scale the data to a specific range
        elif scale_function == "min-max": 
            scaler = MinMaxScaler()
            
        # Robust: If your data contains outliers that might affect the scaling.
        elif scale_function == "robust": 
            scaler = RobustScaler()
            
        # Create a copy of X to avoid modifying the original DataFrame
        X_scaled = X.copy()

        # Identify non-binary columns
        non_binary_columns = [col for col in df_vars.columns if len(df_vars[col].unique()) > 2]
        
        print(f"scaling: {non_binary_columns}")

        # Scale only the specified non-binary columns
        columns_to_scale = [
            df_vars.columns.get_loc(col) for col in non_binary_columns
        ]

        X_scaled[:, columns_to_scale] = scaler.fit_transform(X[:, columns_to_scale])

    # Otherwise, don't scale them
    else: 
        X_scaled = X.copy()
    
    # Impute missing values using SimpleImputer, if there are any missing values.
    # Replace 'mean' with 'median' or 'most_frequent' if needed
    imputer = SimpleImputer(strategy=imputer_strategy)  
    X_imputed = imputer.fit_transform(X_scaled)          
    
    X_imputed = sm.add_constant(X_imputed)  # Add intercept
    model = sm.OLS(y, X_imputed).fit()  # Fit the model
    p_values = pd.Series(model.pvalues[1:], index=df_vars.columns) # Use original column names

    return p_values
