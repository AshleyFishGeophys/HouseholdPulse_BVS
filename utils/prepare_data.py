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
    df, 
    col_name_patterns
):
    """ Convert all dates to same re format for 
    compatability with later functions involving searching
    for date columns by pattern.
    
    col_name_patterns = [
        r"\d{2}/\d{2}/\d{4}",  # mm/dd/yyyy
        r"\d/\d{2}/\d{4}",  # m/dd/yyyy
        r"\d/\d/\d{4}",  # m/d/yyyy
        r"\d{2}/\d/\d{4}"  # mm/d/yyyy
    ]
    
    """
    new_headers = []
    for col in df.columns:
        for pattern in col_name_patterns:
            match = re.match(pattern, col)
            if match:
                print(match.groups())
                # print(f"Matched pattern: {pattern} for column: {col}")
                month, day, year = match.groups()
                # print(f"Groups: {month}, {day}, {year}")
                new_header = f"{month.zfill(2)}/{day.zfill(2)}/{year}"
                new_headers.append(new_header)
                break
        else:
            new_headers.append(col)

    df.columns = new_headers
    
    return df


def extract_lc_rates(
    df: pd.DataFrame,
    col_name_patterns: list
) -> pd.Series:
    """Extracts the Long COVID rate columns from a pandas DataFrame.

    Args:
        df (pd.DataFrame):
            The input DataFrame containing the household
            pulse data.
            
        col_pattern (str):
            A regular expression pattern for matching column names.

    Returns:
        lc_rates (pd.Series):
            A Series containing the extracted Long COVID rate data.
            
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
        
            ^ Matches the beginning of the string.
            [0-9]+ Matches one or more digits (0-9).
            $ Matches the end of the string.
    """

#         col_name_patterns = [
#             r"\d{2}/\d{2}/\d{4}",  # mm/dd/yyyy
#             r"\d/\d{2}/\d{4}",  # m/dd/yyyy
#             r"\d/\d/\d{4}",  # m/d/yyyy
#             r"\d{2}/\d/\d{4}",  # mm/d/yyyy
#             r"Avg"  # Avg column
#         ]
        
    headers_with_rates = []
    for header in df.columns:
        for pattern in col_name_patterns:
            if re.match(pattern, header):
                headers_with_rates.append(header)
    
    return df[headers_with_rates]


def convert_excel_dates(
    lc_rates,
    date_pattern="^[0-9]+$"
):
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
    dates_converted = []
    original_dates = []
    dates_have_been_converted = False

    # Loop through all columns in dataframe
    for col in lc_rates.columns:
        # Convert 5-digit column headers to dates and format as strings
        if re.match('^[0-9]+$', col):
            print("Found a match!")
            excel_date = int(col) - 2  # Excel incorrectly treats 1900 as leap year AND starts at idx 1
            date_datetime = datetime.datetime(1900, 1, 1) + datetime.timedelta(days=excel_date)
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
    
    if dates_have_been_converted:
        print("Dates have been converted!!")
        lc_rates_converted_dates = lc_rates.set_axis(dates_converted, axis=1)
    else: 
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
    df,
    list_of_cols_to_remove,
    pattern_date="^[0-9]+$"
):
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
    
    df_pred_vars = df_pred_vars.drop(columns=list_of_cols_to_remove)

    return df_pred_vars



def resample_class_imbalance(
    df_variables: pd.DataFrame,
    df_target: pd.DataFrame,
    categorical_features: list,
    feature_to_resample: str,
    oversample: bool = True
):
    
    """
    SMOTENC
        Over-sample using SMOTE for continuous and categorical features.

    SMOTEN
        Over-sample using the SMOTE variant specifically for categorical features only.

    BorderlineSMOTE
        Over-sample using the borderline-SMOTE variant.

    SVMSMOTE
        Over-sample using the SVM-SMOTE variant.

    ADASYN
        Over-sample using ADASYN.

    KMeansSMOTE
        Over-sample applying a clustering before to oversample using SMOTE.
    """
    
    X_imbalanced = df_variables[feature_to_resample]
    y_imbalanced = df_target
    
    # print(X_imbalanced)
    
    if oversample: 
        # Oversample the imbalanced feature
        sm = ADASYN(
            categorical_features,
            random_state=42
        )
        
        X_resampled, y_resampled = sm.fit_resample(X_imbalanced, y_imbalanced)
        
        # Replace the imbalanced feature in the original DataFrame
        df_variables[feature_to_resample] = X_resampled.flatten()
        
    return df_variables        
    
    
    
def check_df_for_nan_inf_zero(df):
    print(f"Is NaN: {np.isnan(df).any()}")  # Checks for NaNs
    print(f"Is INF: {np.isinf(df).any()}")  # Checks for infinities
