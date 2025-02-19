import pandas as pd
from typing import Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from statsmodels.stats.multitest import multipletests
from typing import Optional


def find_consv_lib_vs_paid_sick_leave(
    df_states: pd.DataFrame
) -> None:
    """Analyzes the relationship between political leaning
    (Conservative/Liberal) and paid sick leave availability
    across different states.

    Args:
        df_states (pd.DataFrame):
            A Pandas DataFrame containing data for each state,
            including columns for 'Conservative', 'Liberal',
            'Paid sick leave' (1 or 0), and 'Avg' (average value).

    Returns:
        None

    Typical Usage Example:
        Try and understand the relationship between political leaning and
        paid sick leave. Is there a relationship?
        find_consv_lib_vs_paid_sick_leave(df_states_data)

    """    
    df_new = df_states[['Conservative', 'Liberal', 'Paid sick leave', 'Avg']].copy() 
    
    # Create a new column 'Higher_Conservative'
    df_new['Higher_Conservative'] = df_new['Conservative'] > df_new['Liberal']    
    df_new['Higher_Liberal'] = df_new['Conservative'] < df_new['Liberal']    
    df_new['Equal_Conserv_Liberal'] = df_new['Conservative'] == df_new['Liberal']    

    
    df_higher_conserv_with_paid_sick = df_new[
        (df_new['Paid sick leave'] == 1) & (df_new['Higher_Conservative'] == True)
    ]
    df_higher_lib_with_paid_sick = df_new[
        (df_new['Paid sick leave'] == 1) & (df_new['Higher_Liberal'] == True)
    ]
    df_equal_conserv_lib_with_paid_sick = df_new[
        (df_new['Paid sick leave'] == 1) & (df_new['Equal_Conserv_Liberal'] == True)
    ]
    sick_leave_1 = df_new[(df_new['Paid sick leave'] == 1)]

    columns_to_drop = ['Higher_Conservative', 'Higher_Liberal', 'Equal_Conserv_Liberal'] 

    df_higher_conserv_with_paid_sick.drop(
        columns=columns_to_drop,
        axis=1,
        inplace=True
    ) 
    df_higher_lib_with_paid_sick.drop(
        columns=columns_to_drop,
        axis=1,
        inplace=True
    ) 
    df_equal_conserv_lib_with_paid_sick.drop(
        columns=columns_to_drop,
        axis=1,
        inplace=True
    ) 
    
    print(df_higher_conserv_with_paid_sick)
    print(f'Avg Consv rate: {df_higher_conserv_with_paid_sick["Avg"].mean()}')
    print()
    print(df_higher_lib_with_paid_sick)
    print(f'Avg Lib rate: {df_higher_lib_with_paid_sick["Avg"].mean()}')
    print()
    print(df_equal_conserv_lib_with_paid_sick)
    print(f'Avg Equal rate: {df_equal_conserv_lib_with_paid_sick["Avg"].mean()}') 
    # print()
    # print(sick_leave_1)
    print()
    
    

def evaluate_and_compare_models(
    df_vars: pd.DataFrame,
    target: pd.Series
) -> pd.DataFrame:
    """ Calculates Pearson, Spearman, and Kendall correlations
    and performs False Discovery Rate (FDR) correction.

    Args:
        df_vars (pd.DataFrame):
            Pandas DataFrame of independent predictor variables.
        target (pd.Series):
            Pandas Series of the target variable.

    Returns:
        A Pandas DataFrame with correlations and corrected p-values.
        Returns an empty DataFrame if no numeric columns are found.
        
    Typical Usage Example: 
        False Discover Rate proceedure from statsmodels multitest.
        https://islp.readthedocs.io/en/latest/labs/Ch13-multiple-lab.html
        It uses the Benjamini–Hochberg procedure.
    """

    X = df_vars.values # Convert DataFrame to NumPy array for efficiency
    numeric_vars = df_vars.select_dtypes(include='number') # Select only numeric columns

    if numeric_vars.empty:  # Check if numeric_vars is empty
        return pd.DataFrame()  # Return an empty DataFrame

    # Scale data (z-score scaling)
    X_scaled = X.copy()
    # Identify non-binary columns
    non_binary_columns = [col for col in df_vars.columns if len(df_vars[col].unique()) > 2]
    # Get the indices of the columns to scale
    columns_to_scale = [df_vars.columns.get_loc(col) for col in non_binary_columns]
    scaler = StandardScaler() # Create a StandardScaler object
    # Scale the selected columns
    X_scaled[:, columns_to_scale] = scaler.fit_transform(X[:, columns_to_scale])

    # Impute missing values using the mean
    imputer = SimpleImputer(strategy='mean') # Create a SimpleImputer object
    X_imputed = imputer.fit_transform(X_scaled) # Impute missing values

    # Initialize a list to store the results for each variable
    results_list = []
    # Get the number of tests (number of independent variables)
    num_tests = X_imputed.shape[1]

    # Initialize lists to store p-values for each test
    p_values_pearson = []
    p_values_spearman = []
    p_values_kendall = []

    # Iterate through each independent variable
    for i in range(X_imputed.shape[1]):
        var_name = numeric_vars.columns[i] # Get the name of the variable
        variable_data = X_imputed[:, i] # Get the data for the current variable
        var_results = {} # Initialize a dictionary
        var_results['Variable'] = var_name  # Store the variable name

        try:
            # Calculate Pearson, Spearman, Kendall tau correlations
            pearson_result = stats.pearsonr(variable_data, target)
            spearman_result = stats.spearmanr(variable_data, target)
            kendall_result = stats.kendalltau(variable_data, target)

            # Store the Pearson, Spearman, Kendall tau p-values
            var_results['Pearson_p_value'] = pearson_result[1]
            var_results['Spearman_p_value'] = spearman_result[1]
            var_results['Kendall_p_value'] = kendall_result[1]

            # Store the Pearson, Spearman, Kendall tau correlations
            var_results['Pearson_Correlation'] = pearson_result[0]
            var_results['Spearman_Correlation'] = spearman_result[0]
            var_results['Kendall_Tau'] = kendall_result[0]

            # Append the Pearson, Spearman, Kendall tau p-values to list
            p_values_pearson.append(pearson_result[1])
            p_values_spearman.append(spearman_result[1])
            p_values_kendall.append(kendall_result[1])

            # Append the results for the current variable to the list
            results_list.append(var_results)

        # Handle potential exceptions during correlation calculation
        except Exception as e:
            print(f"Error calculating correlation for {var_name}: {e}")
            results_list.append({'Variable': var_name, 'Error': str(e)})

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results_list)

    # False Discovery Rate correction using Benjamini-Hochberg
    reject_pearson, pvals_corrected_pearson, _, _ = multipletests(
        p_values_pearson,
        method='fdr_bh',
        is_sorted=False
    )
    reject_spearman, pvals_corrected_spearman, _, _ = multipletests(
        p_values_spearman,
        method='fdr_bh',
        is_sorted=False
    )
    reject_kendall, pvals_corrected_kendall, _, _ = multipletests(
        p_values_kendall,
        method='fdr_bh',
        is_sorted=False
    )

    # Add corrected p-values to the DataFrame
    results_df['Pearson_p_value_corrected'] = pvals_corrected_pearson
    results_df['Spearman_p_value_corrected'] = pvals_corrected_spearman
    results_df['Kendall_p_value_corrected'] = pvals_corrected_kendall


    # Decimal formatting (only for numeric columns) for better presentation
    for col in results_df.columns:
        if pd.api.types.is_numeric_dtype(results_df[col]):
            results_df[col] = results_df[col].apply(
                lambda x: f"{x:.6f}" if isinstance(x, float) else x
            )

    return results_df
