import pymc as pm
import arviz as az
import xarray as xr
import pandas as pd
import sqlite3
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


def load_inference_data(file_path):
    """Loads inference data from a CSV file into a Pandas DataFrame.

    Args:
        file_path: The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded inference data.
    """
    inference_df = pd.load_csv(file_path)
    return inference_df




def plot_trace(inference_results, var_names=["beta", "beta_raw", "sigma", "ind", "mu"]):
    """ Plots the trace of the MCMC samples using ArviZ.

    Args:
        inference_results: An ArviZ InferenceData object.
        var_names (optional): A list of variable names to plot. Defaults to
                              ["beta", "beta_raw", "sigma", "ind", "mu"].

    Returns:
        matplotlib.axes.Axes or numpy.ndarray: The axes on which the traceplot was drawn.

    """
    az.plot_trace(inference_results, var_names=var_names);

    
def plot_forest(inference_results, var_names=["beta"]):
    """ Generates a forest plot of credible intervals for specified variables
    from an ArviZ InferenceData object.

    Args:
        inference_results: An ArviZ InferenceData object.
        var_names (optional): A list of variable names to include in the forest plot.
                             Defaults to ["beta"].

    Returns:
        matplotlib.axes.Axes or numpy.ndarray: The axes on which the forest plot was drawn.

    """
    
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
    """
    Computes and returns the ArviZ summary of an InferenceData object.

    Args:
        inference_results: An ArviZ InferenceData object.

    Returns:
        pandas.DataFrame: A DataFrame containing the summary statistics.
    """
    summary_df = az.summary(inference_results, round_to=4)
    return summary_df
    
    
def get_specific_inference_data(summary_df, col_type='mean', col='beta'):
    """
    Extracts specific statistics (mean or standard deviation) for specified
    variables from an ArviZ summary DataFrame.

    Args:
        summary_df: A Pandas DataFrame generated by `az.summary()`.
        col_type (optional): The type of statistic to extract ('mean' or 'sd').
                           Defaults to 'mean'.
        col (optional): The prefix of the variable names to select (e.g., 'beta',
                        'beta_raw', 'mu', 'ind'). Defaults to 'beta'.

    Returns:
        pandas.Series: A Pandas Series containing the selected statistics for the
                       specified variables.
    """
    new_stat_df = summary_df[col_type]
    
    data = new_stat_df.loc[
    new_stat_df.index.str.startswith(col)
]
    return data
    


def extract_inference_results(inference_results):
    """ Extracts posterior, sample statistics, and observed data DataFrames from
    ArviZ InferenceData.

    Args:
        inference_results: An ArviZ InferenceData object.

    Returns:
        tuple: A tuple containing three Pandas DataFrames:
               (posterior_df, sample_stats_df, observed_data_df).
    """
    posterior_avg_df  = inference_results.to_dataframe()
    sample_stats_avg_df = inference_results.sample_stats.to_dataframe()
    observed_data_avg_df = inference_results.observed_data.to_dataframe()
    
    return posterior_avg_df, sample_stats_avg_df, observed_data_avg_df



def calculate_importance(ind_means, beta_raw):
    """
    *****
    *****NOT USED!!!*****
    *****
    
    Calculates feature importance by multiplying the mean of the indicator
    variables (`ind_means`) with the raw beta coefficients (`beta_raw`).

    Args:
        ind_means: Pandas Series or array-like containing the mean of the
                   indicator variables.
        beta_raw: Pandas Series or array-like containing the raw beta coefficients.

    Returns:
        numpy.ndarray: A NumPy array containing the calculated importance values.

    """
    
    importance = np.multiply(ind_means.values.flatten(), beta_raw.values.flatten())
    
    return importance



