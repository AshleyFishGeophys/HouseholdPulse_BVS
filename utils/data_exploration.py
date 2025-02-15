def find_consv_lib_vs_paid_sick_leave(df_states):
    
    df_new = df_states[['Conservative', 'Liberal', 'Paid sick leave', 'Avg']].copy() 
    
    # Create a new column 'Higher_Conservative'
    df_new['Higher_Conservative'] = df_new['Conservative'] > df_new['Liberal']    
    df_new['Higher_Liberal'] = df_new['Conservative'] < df_new['Liberal']    
    df_new['Equal_Conserv_Liberal'] = df_new['Conservative'] == df_new['Liberal']    

    
    df_higher_conserv_with_paid_sick = df_new[(df_new['Paid sick leave'] == 1) & (df_new['Higher_Conservative'] == True)]
    df_higher_lib_with_paid_sick = df_new[(df_new['Paid sick leave'] == 1) & (df_new['Higher_Liberal'] == True)]
    df_equal_conserv_lib_with_paid_sick = df_new[(df_new['Paid sick leave'] == 1) & (df_new['Equal_Conserv_Liberal'] == True)]
    sick_leave_1 = df_new[(df_new['Paid sick leave'] == 1)]

    columns_to_drop = ['Higher_Conservative', 'Higher_Liberal', 'Equal_Conserv_Liberal'] 

    df_higher_conserv_with_paid_sick.drop(columns=columns_to_drop, axis=1, inplace=True) 
    df_higher_lib_with_paid_sick.drop(columns=columns_to_drop, axis=1, inplace=True) 
    df_equal_conserv_lib_with_paid_sick.drop(columns=columns_to_drop, axis=1, inplace=True) 
    
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
    df_vars,
    target
):
    """
    Calculates Pearson, Spearman, and Kendall correlations and performs FDR correction.

    Args:
        df_vars: Pandas DataFrame of independent variables.
        target: Pandas Series of the dependent variable.

    Returns:
        A Pandas DataFrame with correlations and corrected p-values.
        Returns an empty DataFrame if no numeric columns are found.
        
    Discussion: 
        False Discover Rate proceedure from statsmodels multitest.
        https://islp.readthedocs.io/en/latest/labs/Ch13-multiple-lab.html
        It uses the Benjamini–Hochberg procedure.
    """

    X = df_vars.values
    numeric_vars = df_vars.select_dtypes(include='number')

    if numeric_vars.empty:  # Check if numeric_vars is empty
        return pd.DataFrame()  # Return an empty DataFrame

    # Scale data (z-score scaling)
    X_scaled = X.copy()
    non_binary_columns = [col for col in df_vars.columns if len(df_vars[col].unique()) > 2]
    columns_to_scale = [df_vars.columns.get_loc(col) for col in non_binary_columns]
    scaler = StandardScaler()
    X_scaled[:, columns_to_scale] = scaler.fit_transform(X[:, columns_to_scale])

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_scaled)

    results_list = []
    num_tests = X_imputed.shape[1]

    p_values_pearson = []
    p_values_spearman = []
    p_values_kendall = []

    for i in range(X_imputed.shape[1]):
        var_name = numeric_vars.columns[i]
        variable_data = X_imputed[:, i]
        var_results = {}
        var_results['Variable'] = var_name

        try:
            pearson_result = stats.pearsonr(variable_data, target)
            spearman_result = stats.spearmanr(variable_data, target)
            kendall_result = stats.kendalltau(variable_data, target)

            var_results['Pearson_p_value'] = pearson_result[1]
            var_results['Spearman_p_value'] = spearman_result[1]
            var_results['Kendall_p_value'] = kendall_result[1]

            var_results['Pearson_Correlation'] = pearson_result[0]
            var_results['Spearman_Correlation'] = spearman_result[0]
            var_results['Kendall_Tau'] = kendall_result[0]

            p_values_pearson.append(pearson_result[1])
            p_values_spearman.append(spearman_result[1])
            p_values_kendall.append(kendall_result[1])

            results_list.append(var_results)

        except Exception as e:
            print(f"Error calculating correlation for {var_name}: {e}")
            results_list.append({'Variable': var_name, 'Error': str(e)})

    results_df = pd.DataFrame(results_list)

    # False Detection Rate correction
    reject_pearson, pvals_corrected_pearson, _, _ = multipletests(p_values_pearson, method='fdr_bh', is_sorted=False)
    reject_spearman, pvals_corrected_spearman, _, _ = multipletests(p_values_spearman, method='fdr_bh', is_sorted=False)
    reject_kendall, pvals_corrected_kendall, _, _ = multipletests(p_values_kendall, method='fdr_bh', is_sorted=False)

    results_df['Pearson_p_value_corrected'] = pvals_corrected_pearson
    results_df['Spearman_p_value_corrected'] = pvals_corrected_spearman
    results_df['Kendall_p_value_corrected'] = pvals_corrected_kendall


    # Decimal formatting (only for numeric columns)
    for col in results_df.columns:
        if pd.api.types.is_numeric_dtype(results_df[col]):
            results_df[col] = results_df[col].apply(lambda x: f"{x:.6f}" if isinstance(x, float) else x)

    return results_df
