import pandas as pd
import re
import numpy as np
from imblearn.over_sampling import SMOTE, SMOTEN, ADASYN, SMOTENC

def load_data(
    data_path: str
) -> pd.DataFrame:
    """Loads a CSV file into a pandas DataFrame.

    Args:
        data_path (str):
            The path to the CSV file to be loaded.

    Returns:
        df (pd.DataFrame):
            A pandas DataFrame containing the data from the
            loaded CSV file.

    Raises:
        FileNotFoundError: If the specified `data_path` does
        not exist.
    
    Typical Usage Example: 
        Load in a csv file with columnar data. 
        
        df = load_data("root/path_to_your_data/your_data.csv")  
    """

    try:
        df = pd.read_csv(data_path)
        
    except FileNotFoundError:
        print("Error: Check your file path. File not found at path:", data_path)
        raise
        
    return df


def remove_columns(
    df:pd.DataFrame,
    columns_to_remove:list
) -> pd.DataFrame:
    """Removes specific columns from dataframe. 

    Args:
        df (pd.DataFrame):
            A pandas DataFrame.
            
        columns_to_remove (list):
            A list of columns names to remove from the dataframe.
            
    Returns:
        df_new (pd.DataFrame):
            A pandas DataFrame with specified columns removed.
    
    Typical Usage Example: 
        If one suspects that certain columns have bias in their 
        collection or are  duplicative or have other issues, 
        use this function to remove those columns from the 
        dataframe so that they are excluded from further
        analysis.
        
        df_new = remove_columns(df) 
    """
    
    df_new = df.drop(columns=columns_to_remove)

    return df_new


def set_states_as_idx(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Preprocesses a CSV file containing state-level data.
    
    Sets row names to values in the "State" column. Then, filters data
    to exclude rows where state column equals "United States".

    Args:
        df (pd.DataFrame):
            A pandas DataFrame containing the data from the
            loaded CSV file.

    Returns:
        df_states (pd.DataFrame):
            A preprocessed DataFrame with state names as row indices.

    Typical Usage Example: 
        This function is useful for preparing state-level data for
        analysis by:
            * Setting state names as row indices for easy identification
                and indexing.
            * Removing data for the United States if it's not relevant
                to your analysis.
            * Removing the "State" column to avoid redundancy.
            
        df_states= set_states_as_idx(df)

    The preprocessed DataFrame can then be used for various tasks, such
    as calculating statistics, creating visualizations, or performing
    machine learning models.
    """
    
    # Set state names as row indices
    df_states = df.copy()
    df_states.set_index("State", inplace=True)

    # Remove rows for the United States
    df_states = df_states.loc[df_states.index != "United States"]

    return df_states


def convert_dates_to_same_format(
    df: pd.DataFrame,
    col_name_patterns: list
) -> pd.DataFrame:
    """Converts date column names in a DataFrame to a consistent
    format (mm/dd/yyyy).

    This function iterates through the columns of the DataFrame
    and uses regular expressions to identify date columns based on
    provided patterns. Matching date column names are then converted
    to the format mm/dd/yyyy.  Non-matching column names are left unchanged.

    Args:
        df (pd.DataFrame):
            The input Pandas DataFrame.
        col_name_patterns (List[str]):
            A list of regular expression patterns to match date column names. 
            For example: `[r"\d{2}/\d{2}/\d{4}", r"\d/\d{2}/\d{4}"]`

    Returns:
        df (pd.DataFrame):
            A new Pandas DataFrame with the date columns renamed to the
            consistent format, or the original DataFrame if no date
            columns are found.
                      
    Typical Usage Example: 
        Extract the month, day, year from the column names and convert them
        to a consistent format. 
        
            col_name_patterns = [
                r"\d{2}/\d{2}/\d{4}",  # mm/dd/yyyy
                r"\d/\d{2}/\d{4}",  # m/dd/yyyy
                r"\d/\d/\d{4}",  # m/d/yyyy
                r"\d{2}/\d/\d{4}"  # mm/d/yyyy
            ]
           
        df_converted_dates = convert_dates_to_same_format(df_data, col_name_patterns)
    """
    new_headers = []
    # Iterate through each column name in the DataFrame
    for col in df.columns:
        # Iterate through the provided date patterns
        for pattern in col_name_patterns:
            # Attempt to match the pattern against the column name
            match = re.match(pattern, col)
            if match:
                print(match.groups())
                # print(f"Matched pattern: {pattern} for column: {col}")
                # Extract the month, day, and year groups from the match
                month, day, year = match.groups()
                # print(f"Groups: {month}, {day}, {year}")
                # Add the new header to the list
                new_header = f"{month.zfill(2)}/{day.zfill(2)}/{year}"
                new_headers.append(new_header)
                break
        else:
            new_headers.append(col)

     # Assign the new column names to the DataFrame
    df.columns = new_headers
    
    return df


def extract_lc_rates(
    df: pd.DataFrame,
    col_name_patterns: list
) -> pd.DataFrame:
    """Extracts the Long COVID rate columns from a pandas DataFrame.

    Args:
        df (pd.DataFrame):
            The input DataFrame containing the household
            pulse data.
            
        col_pattern (list):
            A list of regular expression patterns for matching column names.

    Returns:
        lc_rates (pd.DataFrame):
            A Pandas DataFrame containing the extracted Long COVID rate data.
            
    Typical Usage Example: 
        This function is useful for extracting specific columns from
        a DataFrame based on a regular expression pattern. It can be
        used to isolate Long COVID rate data from a dataset for further
        analysis or visualization.
    
        col_name_patterns = [
            r"\d{2}/\d{2}/\d{4}",  # mm/dd/yyyy
            r"\d/\d{2}/\d{4}",  # m/dd/yyyy
            r"\d/\d/\d{4}",  # m/d/yyyy
            r"\d{2}/\d/\d{4}",  # mm/d/yyyy
            r"Avg"  # Avg column
        ]

        lc_rates = extract_lc_rates(df_states, col_name_patterns)
    """
        
    headers_with_rates = []
    for header in df.columns:
        for pattern in col_name_patterns:
            if re.match(pattern, header):
                headers_with_rates.append(header)
    
    return df[headers_with_rates]


def convert_excel_dates(
    lc_rates,
    date_pattern="^[0-9]+$"
) -> pd.DataFrame:
    """Converts column headers which are in excel date format
    to human-redable dates. Prints out converion key, as well.
    
    Args:
        lc_rates (pd.DataFrame):
            The input DataFrame containing the household
            pulse data.
            
        date_pattern (str):
            A regular expression pattern indicating which columns
            are in excel date format.

    Returns:
        lc_rates_converted_dates (pd.DataFrame):
            A DataFrame containing the extracted Long COVID rate data.    
    
    Typical Usage Example: 
        If the raw data are exported from excel and the data contains
        dates, Excel converts those dates to it's own date format, making
        it difficult for humans to interpret. Use this function to convert
        Excel dates back to human-readable formats.
        
        lc_rates_dates_converted = convert_excel_dates(lc_rates)    
    """
    # Initialize lists for converted dates, original dates
    dates_converted = []
    original_dates = []
    # Dates have not been converted yet
    dates_have_been_converted = False

    # Loop through all columns in dataframe
    for col in lc_rates.columns:
        # Convert 5-digit column headers to dates and format as strings
        if re.match('^[0-9]+$', col):
            print("Found a match!")
            # Excel incorrectly treats 1900 as leap year AND starts at idx 1
            excel_date = int(col) - 2  
            # Convert Excel date to human readable date
            date_datetime = datetime.datetime(1900, 1, 1) + datetime.timedelta(days=excel_date)
            # Append converted date to list
            dates_converted.append(date_datetime.strftime('%m-%d-%Y'))
            original_dates.append(col)
            dates_have_been_converted = True
        # If not date format, then just append the column without converting
        else: 
            dates_converted.append(col)

    print("Excel dates converted:")
    for date_excel, date_converted in zip(original_dates, dates_converted):
        print(f"{date_excel} ------> {date_converted}")
    
    print(dates_converted)
    print(dates_have_been_converted)
    
    # If any dates have been found to be in Excel format and have been converted
    # Then indicate as such
    if dates_have_been_converted:
        print("Dates have been converted!!")
        lc_rates_converted_dates = lc_rates.set_axis(dates_converted, axis=1)
    else: 
        print("No Excel format dates found. Please check regular expressions.")
        lc_rates_converted_dates = lc_rates
            
    return lc_rates_converted_dates


def matrix_to_vector(
    df: pd.DataFrame,
) -> pd.Series:
    """Converts DataFrame to Series, or in other words, 
    flattens a matrix to a vector. 
    
    Args:
        df (pd.DataFrame):
            The input DataFrame containing the household
            pulse data.

    Returns:
        df_flat (pd.Series):
            The flattened DataFrame. 
            
    Typical Usage Example: 
        When wanting to generate a Q-Q plot, use this to flatten
        the DataFrame (matrix) to a Series (vector) first. 
        
        lc_rates_flat = matrix_to_vector(lc_rates_dates_converted)
    """
    
    # Flatten dataframe. 
    df_flat = df.melt(ignore_index=False)['value']
    
    return df_flat



def get_variables_df(
    df: pd.DataFrame,
    list_of_cols_to_remove: list,
    pattern_date="^[0-9]+$"
) -> pd.DataFrame:
    """ Gets variables from dataframe which may or may not
    contribute to the Long COVID rates.
    
    Args:
        df (pd.DataFrame):
            A pandas DataFrame containing the raw data with predictor
            and target variables
            
        list_of_cols_to_remove (list):
            A list of target variables to exlude from the dataframe.
            
        pattern_date (str): 
            A patterns of the LC rates date columns to exclude

    Returns:
        df_pred_vars (pd.DataFrame):
            A pandas DataFrame containing the predictor variables.           
        
    Typical Usage Example: 
        Use this to extract the predictor variables from the household pulse
        data before Baysean Variable Selection.

        df_variables = get_variables_df(df_states, list_of_cols_to_remove, pattern="^[0-9]+$")
    """
    
    # Exclude all columns which match the pattern
    exclude_mask = df.columns.astype(str).str.contains(pattern_date)
    
    # Filter columns based on the exclude mask
    df_pred_vars = df.loc[:, ~exclude_mask]
    
    # Remove specified columns
    df_pred_vars = df_pred_vars.drop(columns=list_of_cols_to_remove)

    return df_pred_vars



# def resample_class_imbalance(
#     df_variables: pd.DataFrame,
#     df_target: pd.DataFrame,
#     categorical_features: list,
#     feature_to_resample: str,
#     oversample: bool = True
# ):
    
#     """ ***NOT USED!!!! This was an attempt to understand and address
#     suspected class imbalance...
    
#     SMOTENC
#         Over-sample using SMOTE for continuous and categorical features.

#     SMOTEN
#         Over-sample using the SMOTE variant specifically for categorical features only.

#     BorderlineSMOTE
#         Over-sample using the borderline-SMOTE variant.

#     SVMSMOTE
#         Over-sample using the SVM-SMOTE variant.

#     ADASYN
#         Over-sample using ADASYN.

#     KMeansSMOTE
#         Over-sample applying a clustering before to oversample using SMOTE.
#     """
    
#     X_imbalanced = df_variables[feature_to_resample]
#     y_imbalanced = df_target
    
#     # print(X_imbalanced)
    
#     if oversample: 
#         # Oversample the imbalanced feature
#         sm = ADASYN(
#             categorical_features,
#             random_state=42
#         )
        
#         X_resampled, y_resampled = sm.fit_resample(X_imbalanced, y_imbalanced)
        
#         # Replace the imbalanced feature in the original DataFrame
#         df_variables[feature_to_resample] = X_resampled.flatten()
        
#     return df_variables        
    
    
    
def check_df_for_nan_inf_zero(df: pd.DataFrame) -> None:
    """Checks a Pandas DataFrame for NaN, infinite, and zero values.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.

    Returns: 
        None

    Typical Usage Example:
        has_nan, has_inf, has_zero = check_df_for_nan_inf_zero(my_dataframe)
        if has_nan:
            print("DataFrame contains NaN values.")
        if has_inf:
            print("DataFrame contains infinite values.")
        if has_zero:
            print("DataFrame contains zero values.")

    """    
    print(f"Is NaN: {np.isnan(df).any()}")  # Checks for NaNs
    print(f"Is INF: {np.isinf(df).any()}")  # Checks for infinities
    print(f"Is Zero: {(df.values == 0).any()}") # Check for zeros 

    
def calc_variable_correlations(
    df: pd.DataFrame,
    method: str = "pearson"
) -> pd.DataFrame:
    """Calculates correlations between all variables in a pandas DataFrame,
    sorts by the largest absolute values while maintaining polarity.

    Args:
      df (pd.DataFrame):
          The input pandas predictor variables DataFrame.
      method (str):
          The method of correlation to use. Example: pearson, spearman, etc
          
    Returns:
        sorted_corr_df (pd.DataFrame)
            A pandas DataFrame with sorted correlations.
    """

    # Calculate the correlation matrix
    corr_matrix = df.corr(method=method)
    
    # Unstack the correlation matrix to get a Series of correlations
    unstacked_corr = corr_matrix.unstack()

    # Drop duplicates (to avoid redundancy) and filter out self-correlations
    # (correlation of a variable with itself)
    sorted_corr = (
        unstacked_corr[
            # Filter out self-correlations            
            unstacked_corr.index.get_level_values(0) != unstacked_corr.index.get_level_values(1)
        ]
        .drop_duplicates() # Remove duplicate correlation pairs (e.g., corr(A, B) and corr(B, A))
        .abs() # Take the absolute value for sorting
        .sort_values(ascending=False) # Sort by absolute correlation value in descending order
    )

    # Create DataFrame with 'Variable 1', 'Variable 2', and 'Correlation' columns
    sorted_corr_df = sorted_corr.reset_index() # Convert the Series to a DataFrame
    sorted_corr_df.columns = ["Variable 1", "Variable 2", "Correlation"]

    # Restore original sign (polarity) of correlation
    sorted_corr_df["Correlation"] = sorted_corr_df.apply(
        # Look up the original signed correlation
        lambda row: corr_matrix.loc[row["Variable 1"], row["Variable 2"]], axis=1
    )

    return sorted_corr_df    



def calc_variable_correlations_with_removal(
    df: pd.DataFrame, 
    method: str = "pearson", 
    threshold: float = 0.7
):
    """ Calculates correlations between all variables in a pandas DataFrame, 
    sorts by the largest absolute values while maintaining polarity, 
    and removes variables with correlations exceeding the threshold.

    Args:
        df (pd.DataFrame):
            The input pandas predictor variables DataFrame.
        method (str):
            The method of correlation to use. Example: pearson, spearman, etc
        threshold (float):
            The correlation threshold for variable removal.

    Returns:
        removed_variables (list)
            List of removed variables based on threshold.
        df_cleaned (pd.DataFrame):
            A pandas DataFrame with sorted correlations
            
    Typical Usage Example: 
        If you have multicollinearity in the predictor variables, use correlation
        method and a threshold to remove the multicollinearity.
    """

    # Calculate the correlation matrix
    corr_matrix = df.corr(method=method)
    
    unstacked_corr = corr_matrix.unstack()

    # Drop duplicates (to avoid redundancy) and filter out self-correlations
    # (correlation of a variable with itself)
    sorted_corr = (
        unstacked_corr[
            # Filter out self-correlations            
            unstacked_corr.index.get_level_values(0) != unstacked_corr.index.get_level_values(1)
        ]
        .drop_duplicates() # Remove duplicate correlation pairs (e.g., corr(A, B) and corr(B, A))
        .abs() # Take the absolute value for sorting
        .sort_values(ascending=False) # Sort by absolute correlation value in descending order
    )

    # Create DataFrame with 'Variable 1', 'Variable 2', and 'Correlation' columns
    sorted_corr_df = sorted_corr.reset_index() # Convert the Series to a DataFrame
    sorted_corr_df.columns = ["Variable 1", "Variable 2", "Correlation"]
    
    # Initialize empty set of removed variables
    removed_variables = set()
    # Remove variables based on threshold
    for index, row in sorted_corr_df.iterrows():
        if abs(row["Correlation"]) >= threshold:
            
            if row["Variable 1"] not in removed_variables:
                removed_variables.add(row["Variable 2"])
                
            elif row["Variable 2"] not in removed_variables:
                removed_variables.add(row["Variable 1"])

    # Remove removed variables from the original DataFrame
    df_cleaned = df.drop(columns=list(removed_variables)) 

    return list(removed_variables), df_cleaned 


def calc_variable_correlations_with_removal_lc_rates(
    df: pd.DataFrame, 
    target_var: pd.Series,
    method: str = "pearson", 
    threshold: float = 0.7
):
    """Calculates correlations between all variables in a pandas DataFrame, 
    sorts by the largest absolute values while maintaining polarity, 
    and removes variables with correlations exceeding the threshold and
    utilizes prior information regarding urban vs rural and liberal vs
    conservative and their correlations to LC Rates to aid with removal of 
    the predictor variables.

    Args:
        df (pd.DataFrame):
            The input pandas predictor variables DataFrame.
        method (str):
            The method of correlation to use. Example: pearson, spearman, etc
        threshold (float):
            The correlation threshold for variable removal.

    Returns:
        removed_variables (list)
            List of removed variables based on threshold.
        df_cleaned (pd.DataFrame):
            A pandas DataFrame with sorted correlations
            
    Typical Usage Example: 
        *** THIS WAS USED IN THE WORKFLOW ***
        If you have multicollinearity in the predictor variables, use correlation
        method and a threshold to remove the multicollinearity, AND use the 
        correlation between the predictor variable and the lc rates to further
        aid in the removal of predictor variables with multicolinearity. In addition, 
        use prior knowledge about Liberal vs Conservative and Urban vs Rural to 
        aid in the removal of the variables. 
    """

    # Calculate correlations between each variable and the target
    corr_var_target = df.corrwith(target_var, method=method)
    
    # Calculate the correlation matrix within variables
    corr_matrix = df.corr(method=method)
    
    unstacked_corr = corr_matrix.unstack()

    # Drop duplicates (to avoid redundancy) and filter out self-correlations
    # (correlation of a variable with itself)
    sorted_corr = (
        unstacked_corr[
            # Filter out self-correlations            
            unstacked_corr.index.get_level_values(0) != unstacked_corr.index.get_level_values(1)
        ]
        .drop_duplicates() # Remove duplicate correlation pairs (e.g., corr(A, B) and corr(B, A))
        .abs() # Take the absolute value for sorting
        .sort_values(ascending=False) # Sort by absolute correlation value in descending order
    )

    # Create DataFrame with 'Variable 1', 'Variable 2', and 'Correlation' columns
    sorted_corr_df = sorted_corr.reset_index() # Convert the Series to a DataFrame
    sorted_corr_df.columns = ["Variable 1", "Variable 2", "Correlation"]

    # Remove variables based on threshold
    removed_variables = set()
    for index, row in sorted_corr_df.iterrows():
        if abs(row["Correlation"]) >= threshold:
            
            # Urban vs Rural vs LC Rates
            if row["Variable 1"] == "Urban" and row["Variable 2"] == "Rural" \
                or row["Variable 1"] == "Rural" and row["Variable 2"] == "Urban":
                print(f'corr_var_target["Urban"]: {corr_var_target["Urban"]}')
                print(f'corr_var_target["Rural"]: {corr_var_target["Rural"]}')
                if abs(corr_var_target["Urban"]) > abs(corr_var_target["Rural"]):
                    removed_variables.add("Rural")
                else:
                    removed_variables.add("Urban")
                    
            # Conservative vs Liberal vs LC Rates
            if row["Variable 1"] == "Liberal" and row["Variable 2"] == "Conservative" \
                or row["Variable 1"] == "Conservative" and row["Variable 2"] == "Liberal":
                print(f'corr_var_target["Liberal"]: {corr_var_target["Liberal"]}')
                print(f'corr_var_target["Conservative"]: {corr_var_target["Conservative"]}')
                if abs(corr_var_target["Liberal"]) > abs(corr_var_target["Conservative"]):
                    removed_variables.add("Conservative")
                else:
                    removed_variables.add("Liberal")
                    
            # Keep track of the removed variables
            elif row["Variable 1"] not in removed_variables:
                removed_variables.add(row["Variable 2"])
                
            # Keep track of the removed variables
            elif row["Variable 2"] not in removed_variables:
                removed_variables.add(row["Variable 1"])

    # Remove removed variables from the original DataFrame
    df_cleaned = df.drop(columns=list(removed_variables)) 

    return list(removed_variables), df_cleaned 


def clean_up_data(
    sample_stats: pd.DataFrame,
    df_variables: pd.DataFrame
) -> pd.DataFrame:
    """ Extracts beta mean statistics from sample statistics and aligns them with 
    variable labels from a DataFrame.

    This function filters rows in `sample_stats` that start with "beta" (but not "beta_raw"),
    and then uses the column names from `df_variables` as the new index for the filtered data.

    Args:
        sample_stats (pd.DataFrame):
            Pandas DataFrame containing sample statistics, likely including 
            beta values. The index of this DataFrame is assumed to contain
            strings that identify the statistic.
        df_variables (pd.DataFrame):
            Pandas DataFrame containing the variable names.  The *columns* of this
            DataFrame are assumed to correspond to the beta values in `sample_stats`.

    Returns:
        beta_means_results (pd.DataFrame):
            A DataFrame containing the extracted beta means, with the 
            variable names from `df_variables` as the index.
            
    Typical Usage Example: 
        Grab the beta data from the results and set the variable names as the 
        index.

    """
    # Filter rows starting with "beta" but not "beta_raw"
    beta_means_results  = sample_stats.loc[
        sample_stats.index.str.startswith('beta') & ~sample_stats.index.str.startswith('beta_raw')
    ]   
    # Get the variable names from df_variables' columns
    x_labels = list(df_variables.columns)
    
    # Set the new index using the variable names
    beta_means_results.set_index(pd.Index(x_labels), inplace=True)
    
    return beta_means_results


def save_cleaned_results(df: pd.DataFrame, save_path: str) -> None:
    """Save a Pandas DataFrame to a CSV file.

    Args:
        df (pd.DataFrame):
            The Pandas DataFrame to save.
        save_path (str):
            The path to the CSV file where the DataFrame will be saved.

    Returns:
        None
        
    Typical Usage Example: 
        Save results to a csv
    """
    
    df.to_csv(save_path, index=True) 