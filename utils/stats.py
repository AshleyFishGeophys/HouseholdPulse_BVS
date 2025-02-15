import pandas as pd
import statsmodels.api as sm
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import warnings
from pygam import LinearGAM, s, f, te, terms  # Import LinearGAM, s, and f
from statsmodels.robust import norms
import io


def evaluate_and_compare_models(df_vars, y, transformations=None, model_types=None):
    """Evaluates and compares multiple regression models with different options."""

    transformations = ["None"] if transformations is None else transformations
    model_types = ["OLS"] if model_types is None else model_types

    results = []

    for model_type in model_types:
        for transformation in transformations:
            X = df_vars.copy()
            y_for_model = y.copy()

            # Transformations
            if transformation == "polynomial":
                X_poly = df_vars.copy()
                for col in df_vars.columns:
                    X_poly[col + "_sq"] = X_poly[col]**2
                X = X_poly.copy()
            elif transformation == "log":
                for col in X.columns:
                    X[col] = np.log1p(X[col]) if (X[col] <= 0).any() else np.log(X[col])
            elif transformation == "sqrt":
                for col in X.columns:
                    if (X[col] < 0).any():
                        print(f"Skipping sqrt transformation for column '{col}' (contains negative values).")
                    else:
                        X[col] = np.sqrt(X[col])
            elif transformation == "reciprocal":
                for col in X.columns:
                    if (X[col] == 0).any():
                        print(f"Skipping reciprocal transformation for column '{col}' (contains 0 values).")
                    else:
                        X[col] = 1 / X[col]
            elif transformation == "boxcox_y":
                try:
                    y_transformed, lambda_value = boxcox(y, lmbda=None)
                    y_for_model = pd.Series(y_transformed, index=y.index).values  # Keep as Series until needed
                except ValueError as e:
                    print(f"Error applying Box-Cox transformation to y: {e}")
                    continue
            elif transformation == "None":
                pass
            else:
                print(f"Invalid transformation specified: {transformation}")

            X_values = X.values

            # Scaling (StandardScaler only)
            scaler = StandardScaler()
            X_scaled = X_values.copy()
            non_binary_cols = [col for col in df_vars.columns if len(df_vars[col].unique()) > 2]
            cols_to_scale = [df_vars.columns.get_loc(col) for col in non_binary_cols]
            if cols_to_scale:
                X_scaled[:, cols_to_scale] = scaler.fit_transform(X_values[:, cols_to_scale])
            else:
                print("No non-binary columns found. Skipping scaling.")


            # Imputation
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X_scaled)
            X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns, index=df_vars.index)
            X_imputed_with_const = sm.add_constant(X_imputed_df)

            # Model Fitting and Metrics
            try:
                if model_type == "OLS":
                    rmse = mae = aic = bic = None # Initialize these!
                    try:
                        model = sm.OLS(y_for_model, X_imputed_with_const).fit()
                        r_squared = model.rsquared
                        adj_r_squared = model.rsquared_adj
                        y_pred = model.predict(X_imputed_with_const) #Make predictions
                        rmse = np.sqrt(np.mean((y_for_model - y_pred)**2))
                        mae = np.mean(np.abs(y_for_model - y_pred))
                        k = X_imputed_with_const.shape[1]
                        aic = -2 * model.llf + 2 * k
                        bic = -2 * model.llf + np.log(len(y_for_model)) * k
                        summary_io = io.StringIO(model.summary().as_text())
                        summary_df = pd.read_csv(
                            summary_io, delim_whitespace=True, header=None, skiprows=9, skipfooter=2,
                            names=['Variable', 'Coef', 'Std Err', 't', 'P>|t|', '[0.025', '0.975]']
                        )
                        p_values = pd.Series(summary_df['P>|t|'].values, index=summary_df['Variable'].values).drop('const')            
            
                    except Exception as e:
                        print(f"OLS fit error: {e}")
                        r_squared = adj_r_squared = rmse = mae = aic = bic = None  # Assign None in the except block
                        p_values = pd.Series(index=X_imputed_with_const.columns[1:], dtype='float64') * np.nan #Handle p-values

                elif model_type == "RobustLS":
                    try:
                        model = sm.RLM(y_for_model, X_imputed_with_const, M=norms.HuberT()).fit()
                        k = X_imputed_with_const.shape[1]  # Define k

                        null_model = sm.RLM(y_for_model, sm.add_constant(np.ones(len(y_for_model))), M=norms.LeastSquares()).fit()
                        model_ls = sm.RLM(y_for_model, X_imputed_with_const, M=norms.LeastSquares()).fit()
                        ll_null = null_model.llf
                        ll_model = model_ls.llf
                        r_squared = 1 - (ll_model / ll_null) if ll_null != 0 else None
                        adj_r_squared = None

                        aic = -2 * model_ls.llf + 2 * k
                        bic = -2 * model_ls.llf + np.log(len(y_for_model)) * k

                        # Wald test for p-values (RobustLS)
                        p_values = {}
                        for var in X_imputed_with_const.columns[1:]:  # Exclude the constant
                            r_matrix = np.zeros((1, model.params.size))  # Corrected Wald Test
                            r_matrix[0, X_imputed_with_const.columns.get_loc(var)] = 1
                            try:
                                wald_result = sm.stats.wald_test(model.params, model.cov_params(), r_matrix, scalar=False)
                                p_values[var] = wald_result.pvalue
                            except Exception as e:
                                print(f"Wald test error: {e}")
                                p_values[var] = None
                        p_values = pd.Series(p_values)


                    except Exception as e:
                        print(f"RobustLS fit error: {e}")
                        r_squared = adj_r_squared = rmse = mae = aic = bic = None
                        p_values = pd.Series(index=X_imputed_with_const.columns[1:], dtype='float64') * np.nan


                elif model_type == "GAM":
                    try:
                        num_vars = X_imputed_df.shape[1]
                        terms = terms.TermList([s(i) for i in range(num_vars)]) # Create TermList
                        gam = LinearGAM(terms=terms).fit(X_imputed_df, y_for_model)
                        
                        y_pred = gam.predict(X_imputed_df)
                        rmse = np.sqrt(np.mean((y_for_model - y_pred)**2))
                        mae = np.mean(np.abs(y_for_model - y_pred))
                        k = len(gam.terms) + 1

                        aic = gam.statistics_.get('AIC', None)
                        bic = gam.statistics_.get('BIC', None)
                        p_values = gam.statistics_.get('p_values', None)
                        r_squared = gam.statistics_.get('pseudo_r2', None)  # Get pseudo R-squared
                        adj_r_squared = None  # Adjusted R-squared not directly available
                    except Exception as e:
                        print(f"GAM fit error: {e}")
                        rmse = mae = aic = bic = r_squared = adj_r_squared = None
                        p_values = pd.Series(index=X_imputed_df.columns, dtype='float64') * np.nan


                        
                elif model_type in ["GLM", "Quantile"]:
                    if model_type == "GLM":
                        model = sm.GLM(y_for_model, X_imputed_with_const, family=sm.families.Gaussian()).fit()
                        ll_null = sm.GLM(y_for_model, sm.add_constant(np.ones(len(y_for_model))), family=sm.families.Gaussian()).fit().llf
                        ll_model = model.llf
                        r_squared = 1 - (ll_model / ll_null) if ll_null != 0 else None
                        adj_r_squared = None

                    else:  # Quantile
                        model = sm.QuantReg(y_for_model, X_imputed_with_const).fit()
                        r_squared = None
                        adj_r_squared = None

                    y_pred = model.predict(X_imputed_with_const)
                    rmse = np.sqrt(np.mean((y_for_model - y_pred)**2))
                    mae = np.mean(np.abs(y_for_model - y_pred))
                    k = X_imputed_with_const.shape[1]

                    aic = -2 * model.llf + 2 * k
                    bic = -2 * model.llf + np.log(len(y_for_model)) * k

                    p_values = getattr(model, 'pvalues', None)


                else:
                    raise ValueError("Invalid model type. Choose 'OLS', 'GAM', 'RobustLS', 'GLM', or 'Quantile'.")

                results.append({
                    'Model': model_type,
                    'Transformation': transformation,
                    'R-squared': r_squared,
                    'Adjusted R-squared': adj_r_squared,
                    'RMSE': rmse,
                    'MAE': mae,
                    'AIC': aic,
                    'BIC': bic,
                    'p-values': p_values
                })

            except Exception as e:  # Catch any exception during model fitting
                print(f"Error fitting model {model_type} with transformation {transformation}: {e}")
                continue  # Skip to the next model/transformation


    df_results = pd.DataFrame(results)

    for col in df_vars.columns:
        df_results[col + '_pvalue'] = df_results['p-values'].apply(lambda x: x.get(col) if (x is not None) else None)

    df_results = df_results.drop('p-values', axis=1)


    return df_results



def calculate_p_values(
    df_vars,
    series_target,
    scale_function="z-score",
    imputer_strategy: str = 'mean',
    model_type: str = "OLS", 
    transformation: str = "None"
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
    y_for_model = series_target.values  # Long COVID rates
    X = df_vars.copy() # Variables affecting/not affecting LC rates

    # Transformations
    if transformation == "polynomial":
        X_poly = df_vars.copy()
        for col in df_vars.columns:
            X_poly[col + "_sq"] = X_poly[col]**2
        X = X_poly.copy()
        
    elif transformation == "log":
        for col in X.columns:
            X[col] = np.log1p(X[col]) if (X[col] <= 0).any() else np.log(X[col])
            
    elif transformation == "sqrt":
        for col in X.columns:
            if (X[col] < 0).any():
                print(f"Skipping sqrt transformation for column '{col}' (contains negative values).")
            else:
                X[col] = np.sqrt(X[col])
                
    elif transformation == "reciprocal":
        for col in X.columns:
            if (X[col] == 0).any():
                print(f"Skipping reciprocal transformation for column '{col}' (contains 0 values).")
            else:
                X[col] = 1 / X[col]
                
    elif transformation == "boxcox_y":
        y_transformed, lambda_value = boxcox(y, lmbda=None)
        y_for_model = pd.Series(y_transformed, index=y.index).values  # Keep as Series until needed


    elif transformation == "None":
        pass
    
    else:
        print(f"Invalid transformation specified: {transformation}")

    X_values = X.values

    # Scaling (StandardScaler only)
    scaler = StandardScaler()
    X_scaled = X_values.copy()
    non_binary_cols = [col for col in df_vars.columns if len(df_vars[col].unique()) > 2]
    cols_to_scale = [df_vars.columns.get_loc(col) for col in non_binary_cols]
    
    if cols_to_scale:
        X_scaled[:, cols_to_scale] = scaler.fit_transform(X_values[:, cols_to_scale])
    else:
        print("No non-binary columns found. Skipping scaling.")


    # Imputation
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_scaled)
    X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns, index=df_vars.index)
    X_imputed_with_const = sm.add_constant(X_imputed_df)

    # Model Fitting and Metrics
    if model_type == "OLS":
        model = sm.OLS(y_for_model, X_imputed_with_const).fit()

    elif model_type == "RobustLS":
        model = sm.RLM(y_for_model, X_imputed_with_const).fit()

    elif model_type == "GLM":
        model = sm.GLM(y_for_model, X_imputed_with_const, family=sm.families.Gaussian()).fit()

    elif model_type == "Quantile":
        model = sm.QuantReg(y_for_model, X_imputed_with_const).fit()

    else:
        raise ValueError("Invalid model type. Choose 'OLS', 'RobustLS', 'GLM', or 'Quantile'.")

    
    # --- Model Assumption Checking ---
    print("\n--- Model Assumption Checks ---")

    print(f"MODEL: {model_type}, TRANSFORMATION: {transformation}")
    
    print("")

    # 1. Normality of Residuals
    residuals = model.resid
    statistic, p = shapiro(residuals)
    
    print(f"Shapiro-Wilk test: statistic={statistic:.3f}, p-value={p:.3f}")
    
    if p > 0.05: # Use > here to check for normality
        print("Residuals appear to be normally distributed (or close enough).")
    else:
        print("WARNING: Residuals do NOT appear to be normally distributed.")

    print("")
    
    # Additional Homoscedasticity Check (using Breusch-Pagan test)
    _, pval, _, fval = diag.het_breuschpagan(residuals, X_imputed_with_const)
    print(f"Breusch-Pagan test for heteroscedasticity: p-value = {pval:.4f}")
    
    if pval > 0.05: # Use > here to check for homoscedasticity
        print("Homoscedasticity assumption appears to be met via Breusch-Pagan.")
    else:
        print("WARNING: Heteroscedasticity detected Breusch-Pagan." \
              "Consider robust standard errors or a different model.")
        
    print("")

    # Homoscedasticity (Residuals vs. Fitted Values)
    fitted_values = model.fittedvalues
    plt.figure()
    plt.scatter(fitted_values, residuals)
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs. Fitted Values, Model: {model_type}, " \
              f"transform: {transformation}")
    plt.show()
    
    print("")

    # Normality Plots
    fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(residuals, bins=20)
    axes[0].set_title("Histogram of Residuals")
    sm.qqplot(residuals, line='s', ax=axes[1])
    axes[1].set_title("Q-Q Plot of Residuals")
    plt.tight_layout() 
    plt.show()
    
    print("")

    # 2. Linearity (Scatter plots of each predictor vs. the target)
    for col in df_vars.columns:
        plt.figure()
        plt.scatter(df_vars[col], series_target)
        plt.xlabel(col)
        plt.ylabel("Target Variable")
        plt.title(f"Scatter Plot: {col} vs. Target")
        plt.show()

    return p_values
