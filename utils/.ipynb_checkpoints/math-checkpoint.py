import pandas as pd
import numpy as np

def compute_zscore_df(
    df: pd.DataFrame,
    fill_missing_vals: bool = True
) -> pd.DataFrame:
    """Computes the z-score for all columns in a pandas DataFrame
  
    Args:
        df (pd.DataFrame):
            A pandas DataFrame.
            
        fill_missing_vals (bool):
            If true, then fill missing values with mean of zscores.
            Otherwise, fill missing values with NaNs (not a number)>

    Returns:
        df_z_score (pd.DataFrame): 
            A new pandas DataFrame with the z-scores of each column.

    Typical Usage Example:
        Take a DataFrame and a column name as input, calculate the mean
        and standard deviation of the specified column, and return a new
        pandas Series containing the z-scores. This allows you to easily
        apply the z-score calculation to specific columns within your
        DataFrame.    
        
        df_z_scores =  compute_zscore_df(df)
  """

    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate mean and standard deviation for numeric columns
    df_means = numeric_df.mean(skipna=True)
    df_stds = numeric_df.std(skipna=True)

    # Calculate z-scores for numeric columns
    z_scores = (df[numeric_df.columns] - df_means) / df_stds
    
    if fill_missing_vals:
        # Fill in missing values with mean of zscore
        z_scores.fillna(z_scores.mean(), inplace=True)
    else: 
        # Fill missing values in z-scores with NaN
        z_scores.fillna(np.nan, inplace=True)

    return z_scores



def compute_zscore_np(
    data: pd.Series,
) -> np.ndarray:
    """Computes the z-score for a Series or numpy array.

    Args:
        data (pd.Series):
            A pandas Series (flattened dataframe).

    Returns:
        z_scores (np.ndarray):
            A NumPy array containing the z-scores of the input data.
            
    Typical Usage Example: 
        Normalize data using zscore before generating a Q-Q plot. 
        
        lc_rates_flat_z_scores = compute_zscore_np(lc_rates_flat)
    """
    # Remove NA values from zscor calculations
    data = data[~np.isnan(data)]
    
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std

    return z_scores


