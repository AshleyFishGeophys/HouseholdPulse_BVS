import pymc as pm
import arviz as az
import xarray as xr
import pandas as pd
import numpy as np
import re

import sqlite3
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from arviz import InferenceData  # Import InferenceData type


def load_inference_data(file_path: str) -> pd.DataFrame:
    """Loads inference data from a CSV file into a Pandas DataFrame.

    Args:
        file_path (str):
            The path to the CSV file.

    Returns:
        inference_df (pd.DataFrame):
            The loaded inference data.
            
    Typical Usage Example: 
        Load in the inference results which were saved after BVS. When there
        are multiple BVS model results, this is a useful function to use. Load 
        in the each of the multiple BVS inference trace data and compare. 
    """
    inference_df = pd.read_csv(file_path)
    
    return inference_df




def plot_trace(
    inference_results: InferenceData,
    var_names: list=["beta", "beta_raw", "sigma", "ind", "mu"]
) -> None:
    """ Plots the trace of the MCMC samples using ArviZ.

    Args:
        inference_results (InferenceData):
            An ArviZ InferenceData trace object. Contains the results from the 
            BVS modeling
        var_names (list):
            A list of variable names to plot from the inference trace data.
            Defaults to ["beta", "beta_raw", "sigma", "ind", "mu"].

    Returns:
        None
            This function creates a plot and does not return any value.
        
    Typical Usage Example: 
        Plot inference results using specified columns/data types. 
        matplotlib.axes.Axes or numpy.ndarray:
        The axes on which the traceplot was drawn.
    """
    
    az.plot_trace(inference_results, var_names=var_names);

    
def plot_forest(
    inference_results: InferenceData,
    var_names: list=["beta"]
) -> None:
    """ Generates a forest plot of credible intervals for specified variables
    from an ArviZ InferenceData trace object.

    Args:
        inference_results (InferenceData):
            An ArviZ InferenceData trace object. Contains the results from the 
            BVS modeling
        var_names (list):
            A list of variable names to plot from the inference trace data.
            Defaults to ["beta"].

    Returns:
        None
            This function creates a plot and does not return any value.

    Typical Usage Example:
        Plot forest data including r hat and hdi probability for the selected 
        column from the inference trace data in order to assess the quality of the 
        BVS modeling. 
        matplotlib.axes.Axes or numpy.ndarray: The axes on which the forest plot was drawn.

    """
    
    az.plot_forest(
        inference_results,
        var_names=var_names,
        combined=True,
        hdi_prob=0.95,
        r_hat=True
    );
    
    
def plot_ess(
    summary_df: pd.DataFrame,
    variables: list,
    states: list
) -> None:
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
        
        summary_df = az.summary(inference_results, round_to=4)
        inference_variables = lc_predictor_variables.columns  # variables affecting LC rates
        inference_states = lc_rates.index  # US states

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
    importance: list,
    x_labels: list,
    importance_type: str
) -> None:
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
    
    
def get_specific_inference_data(
    summary_df: pd.DataFrame,
    col_type: str='mean',
    col: str='beta'
) -> pd.Series:
    """Extracts specific statistics (mean or standard deviation) for specified
    variables from an ArviZ summary inference trace DataFrame.

    Args:
        summary_df (pd.DataFrame):
            A Pandas DataFrame generated by `az.summary()`. This is the 
            summary statistics of the BVS trace data. 
        col_type (optional):
            The type of statistic to extract ('mean' or 'sd').
            Defaults to 'mean'.
        col (optional):
            The prefix of the variable names to select (e.g., 'beta',
            'beta_raw', 'mu', 'ind'). Defaults to 'beta'.

    Returns:
        data (pd.Series):
            A Pandas Series containing the selected statistics for the 
            specified variables.
                       
                       
    """
    new_stat_df = summary_df[col_type]
    
    data = new_stat_df.loc[
        new_stat_df.index.str.startswith(col)
    ]
    
    return data
    


def extract_inference_results(
    inference_results: InferenceData
):
    """ Extracts posterior, sample statistics, and observed data DataFrames from
    ArviZ InferenceData BVS trace.

    Args:
        inference_results (InferenceData):
            An ArviZ InferenceData object containing the BVS trace model results.

    Returns:
        posterior_avg_df (pd.DataFrame):
            Pandas Dataframe containing the posteriors from the BVS trace inference
            model results.
        sample_stats_avg_df (pd.DataFrame):
            Pandas Dataframe containing the sample statistics from the BVS trace inference
            model results.
        observed_data_avg_df (pd.DataFrame):
            Pandas Dataframe containing the observed data from the BVS trace inference
            model results.
    """
    
    posterior_avg_df  = inference_results.to_dataframe()
    sample_stats_avg_df = inference_results.sample_stats.to_dataframe()
    observed_data_avg_df = inference_results.observed_data.to_dataframe()
    
    return posterior_avg_df, sample_stats_avg_df, observed_data_avg_df



# def calculate_importance(ind_means, beta_raw):
#     """
#     *****
#     *****NOT USED!!!*****
#     *****
    
#     Calculates feature importance by multiplying the mean of the indicator
#     variables (`ind_means`) with the raw beta coefficients (`beta_raw`).

#     Args:
#         ind_means: Pandas Series or array-like containing the mean of the
#                    indicator variables.
#         beta_raw: Pandas Series or array-like containing the raw beta coefficients.

#     Returns:
#         numpy.ndarray: A NumPy array containing the calculated importance values.

#     """
    
#     importance = np.multiply(ind_means.values.flatten(), beta_raw.values.flatten())
    
#     return importance





def restructure_mcmc_samples_custom(csv_file, output_file="restructured_posterior_samples.csv"):       
    try:
        df = pd.read_csv(csv_file, header=None, engine='python')

        header = df.iloc[0].tolist()  # Get header as a list of strings
        # print(header)
        df = df.iloc[1:]  # Remove the header row

        data = []
        for index, row in df.iterrows():
            chain = row[0]  # Chain is the first column
            draw = row[1]   # Draw is the second column
            for j in range(2, len(header)):  # Iterate over data columns
                col_name = header[j]

                match = re.search(r"\('(.+?)', '(.+?)'[,)]", col_name) # Improved regex

                if match:
                    group = match.group(1)
                    variable = match.group(2)
                    index_match = re.search(r"\[(\d+)\]", variable) # check for index in variable
                    if index_match:
                        index = index_match.group(1)
                        variable = variable.split('[')[0] + f"[{index}]"
                    value = row[j]
                    data.append({'chain': chain, 'draw': draw, 'variable': variable, 'value': value})
                elif "sample_stats" in col_name or "log_likelihood" in col_name: # single variable case
                    parts = col_name.split("('")
                    group = parts[1].split("'")[0]
                    variable = parts[1].split("'")[1]
                    value = row[j]
                    data.append({'chain': chain, 'draw': draw, 'variable': variable, 'value': value})
                else:
                    print(f"Skipping column {col_name} due to parsing issues.")


        long_df = pd.DataFrame(data)
        long_df['variable'] = long_df['variable'].astype(str)
        long_df.to_csv(output_file, index=False)
        print(f"Restructured data saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
        
def analyze_posterior_from_csv(csv_file, variable_pattern):  # variable_pattern is now a regex pattern
    try:
        df = pd.read_csv(csv_file)

        print(df.columns)
        
        # Select ALL columns that MATCH the variable_pattern (using regex)
        df_target = df[df['variable'].str.contains(variable_pattern, na=False)]  # na=False handles potential NaNs

        
        n_chains = df_target['chain'].nunique()
        n_draws = df_target['draw'].nunique()
        variables = df_target['variable'].unique()
        print(variables)

        posterior_data = {}

        for variable in variables:
            values = np.empty((n_chains, n_draws))

            for chain in range(n_chains):
                for draw in range(n_draws):
                    try:
                        val = df_target[(df_target['chain'] == chain) & (df_target['draw'] == draw) & (df_target['variable'] == variable)]['value'].iloc[0]
                        values[chain, draw] = val
                    except KeyError:
                        print(f"Warning: Missing data for variable: {variable}, chain: {chain}, draw: {draw}")
                        values[chain, draw] = np.nan
                    except Exception as e:
                        print(f"Error for variable: {variable}, chain: {chain}, draw: {draw}: {e}")
                        return None, None, None

            posterior_data[variable] = (("chain", "draw"), values)

        coords = {"chain": range(n_chains), "draw": range(n_draws)}
        posterior_ds = xr.Dataset(posterior_data, coords=coords)

        inference_data = az.InferenceData(posterior=posterior_ds)

        hdi_data = az.hdi(inference_data, hdi_prob=0.95)
        rhat_data = az.rhat(inference_data)

        return inference_data, hdi_data, rhat_data

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        return None, None, None
    except ValueError as e:
        print(f"Error: {e}")
        return None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None
