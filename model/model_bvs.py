import optuna
import logging
import sys
import datetime
import pickle

import pymc as pm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import arviz as az
import xarray as xr
import json
import pandas as pd
import numpy as np


def bayesian_variable_selection_multiple(
    df_vars: pd.DataFrame,
    df_values: pd.DataFrame,
    model_params: dict,
    experiment_name: str = "LC_household_pulse_v11",
    save_model: bool = True

) -> list:
    """Performs Bayesian Variable Selection for multiple target
    variables in two pandas DataFrames, one containing variables and
    the other containing Long COVID rates (for different dates and also
    an accumulated avg).

    Args:
        df_vars (pandas.DataFrame):
            The input DataFrame containing the variables.

        df_values (pandas.DataFrame):
            The input DataFrame containing the Long COVID rates.

        model_params (dict):
            Dictionary of model parameters.

        experiment_name (str, optional): 
            If you want to add a name to the saved file names to help 
            you more easily identify your experiment, then add it here.
          
    Returns:
        traces (list):
            A list of PyMC trace objects, one for each target variable, 
            containing the Baysian Variable Selection results. 
            
    Typical Usage Example: 
        Call this function to perform BVS using predictor variables 
        DataFrame and a dataframe containing the desired target variables
        to model.
        
        model_params = {
            "beta": 1,
            "mu": 0,
            "sigma": 2,
            "prob_of_success": 0.2,
            "draws": 1500,
            "tune": 1500,
            "cores": 4,
            "target_accept": 0.95,
            "max_treedepth": 25
        }  

        all_variables_avg_10_v2 = bayesian_variable_selection_multiple(
            df_variables,
            pd.DataFrame(lc_rates['Avg']),
            model_params
        )
    """

    traces = []  # init list to store results of BVS
    
    # Iterate through columns in Long COVID rate dataframe
    for target_col in df_values.columns:

        trace = bayesian_variable_selection_with_dfs(
            df_vars,  # dataframe containing variables
            df_values[target_col],  # series containing Long COVID rates
            target_col,  # corresponding column header to rate values
            model_params, # model parameters for BVS
            save_model=True
        )
        
        traces.append(trace)  # append BVS results to list
        
        save_inference_arviz_to_df(
            inferece_results_az=trace,
            target_name=f"{target_col}",
            model_params=model_params
        )
        
    return traces



# def pickle_model(output_path: str, model, trace):
#     """Pickles PyMC3 model and trace"""
#     with open(output_path, "wb") as buff:
#         pickle.dump({"model": model, "trace": trace}, buff)


def save_model(model, model_path):
    
    # pm.dump(model, f'{model_name}.h5')
    pm.save_model(model, f"./model_{experiment_name}.h5")
    

def save_trace(trace, experiment_name):
    pm.save_trace(
        trace,
        directory=f"./trace_{experiment_name}",
        overwrite=True
    )

    
def save_inference_arviz_to_df(
    inferece_results_az:  az.InferenceData,
    target_name: str,
    model_params: dict,
    # experiment_name: str
) -> None:
    """Saves BVS results and model parameters to csv.
        
    Args: 
        inferece_results_az (az.InferenceData):
            Inference results from BVS.
        
        target_name (str): 
            The name of the file to save the inference data. 
            
        model_params (dict): 
            Dictionary of model parameters associated with the 
            inference results.
    
        experiment_name (str, optional): 
            If you want to add a name to the saved file names to help 
            you more easily identify your experiment, then add it here.
            
    Typical Usage Example: 
        This function is called by the main baysean variable 
        selection funcion. No need to call this separately. 
    
    """
    
    experiment_name = model_params["experiment_name"]

    # Convert inference arviz data type to pandas datafram
    posterior_df  = inferece_results_az.to_dataframe()
    sample_stats_df = inferece_results_az.sample_stats.to_dataframe()
    observed_data_df = inferece_results_az.observed_data.to_dataframe()
    
    # Get current date so that we can keep track of when
    # the inference was run
    now = datetime.datetime.now()
    formatted_date = now.strftime('%m-%d-%Y_%H-%M-%S')
    
    suffix = f"{target_name}_{formatted_date}"
    
    # Save pandas dataframe to csv.
    posterior_df.to_csv(
        f'BVS_{experiment_name}_posterior_{suffix}.csv',
        index=False
    )
    
    # Save pandas dataframe to csv.
    sample_stats_df.to_csv(
        f'BVS_{experiment_name}_sample_stats_{suffix}.csv',
        index=False
    )    
    
    # Save pandas dataframe to csv.
    observed_data_df.to_csv(
        f'BVS_{experiment_name}_observed_data_{suffix}.csv',
        index=False
    )
    
    # Convert any non str or number values to str, For example, "kernel": pm.smc.IMH,
    # Convert pm.smc.IMH to "pm.smc.IMH" so that it's compatable with JSON saving
    
    params_to_save = model_params.copy()
    for param_key, param_value in params_to_save.items():
        if not isinstance(param_value, (str, int, float)):
            params_to_save[param_key] = str(param_value)      
    
    #Save parameters associated with the BVS
    with open(f'BVS_{experiment_name}_model_params_{suffix}.txt', 'w') as f:
        json.dump(params_to_save, f, indent=4)
        
        
        
def bayesian_variable_selection_with_dfs(
    df_vars: pd.DataFrame,
    series_values: pd.Series,
    target_col: str, 
    model_params: dict,
    imputer_strategy: str = 'mean',
    save_model: bool = False
) -> pm.backends.base.MultiTrace:
    """Performs Bayesian Variable Selection using PyMC3
    on a pandas DataFrame containing variables and a pandas
    Series containing Long COVID rates.

    Args:
        df_vars (pandas.DataFrame):
            DataFrame containing variables afftecting/not
            affecting Long COVID Rates.

        series_values (pd.Series):
            DataFrame containing Long COVID rates.

        target_col (str):
            The column name of the target column in df_values. 
            This can contains either a specific date or avg of 
            all dates of Long COVID rates.

        model_params (dict):
            Dictionary of model parameters.
        
        scale_predictors (bool):
            Booling indicating whether or not to scale the predictor
            variables or not. For z-score, use StandardScaler. 
            For min-max, use MinMaxScaler. For robust scaling using 
            interquartile range, use RobustScaler. 
            
    Returns:
        trace (pm.backends.base.MultiTrace):
            A PyMC3 trace object containing the Baysian Variable
            Selection results.
          
    Typical Usage Example: 
        This iterates through each target variable and performs BVS
        using all predictor variables. 

        Called by bayesian_variable_selection_multiple function

        sampling kernel options: 
            pm.smc.IMH: Independent Metropolis-Hastings sampler
            pm.nuts: No-U-Turn Sampler
            pm.hmc: Hamiltonian Monte Carlo
        """
    # Extract data from DataFrames
    y = series_values.values  # Long COVID rates
    X = df_vars.values  # Variables affecting/not affecting LC rates

    # If scale predictor variables (df_vars)
    # Scale only the variables which are not binary
    # scaling predictor variables, especially non-binary ones, can be beneficial
    # for Bayesian variable selection using Monte Carlo simulations.
    # It improves convergence and interpretability. Scaling binary variables, however, 
    # Can introduce unecessary complexity.
    if model_params["scale_predictors"]: 
        # Z-score: If your data is normally distributed and you want to
        # preserve the relative distances between data points.
        if model_params["scale_function"] == "z-score": 
            scaler = StandardScaler() 
            
        # Min-Max: If you want to scale the data to a specific range
        elif model_params["scale_function"] == "min-max": 
            scaler = MinMaxScaler()
            
        # Robust: If your data contains outliers that might affect the scaling.
        elif model_params["scale_function"] == "robust": 
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
#         # Scale only the specified columns (the non-binary columns)
#         columns_to_scale = [
#             df_vars.columns.get_loc(col) for col in model_params["predictors_to_scale"]
#         ]


        X_scaled[:, columns_to_scale] = scaler.fit_transform(X[:, columns_to_scale])

    # Otherwise, don't scale them
    else: 
        X_scaled = X.copy()
    
    # Impute missing values using SimpleImputer, if there are any missing values.
    # Replace 'mean' with 'median' or 'most_frequent' if needed
    imputer = SimpleImputer(strategy=imputer_strategy)  
    X_imputed = imputer.fit_transform(X_scaled)    
    
    n, p = X_imputed.shape

    with pm.Model() as model:
        # ------------------------------------------------------------------------------
        # Priors
        # ------------------------------------------------------------------------------
        # sigma defines a prior distribution for the std of the noise term in the model.
        # It uses a Gamma distribution with parameters alpha and beta. A higher alpha leads
        # to a distribution that is more concentrated towards lower values of sigma
        # (less noise). A higher beta leads to a distribution with larger
        # standard deviations (more noise).
        sigma = pm.Gamma(
            'sigma',
            alpha=model_params["alpha_sigma"],
            beta=model_params["beta_sigma"]
        )
        
        # beta_raw defines a prior distribution for a vector of raw coefficients 
        # associated with each feature in the data. It uses a normal distribution
        # with a mean of 0 and a std of 1 for each element of beta_raw. A mean of
        # 0 indicates a neutral prior, meaning it doesn't favor any particular
        # value of the parameter. Setting std to 1 means the prior distribution
        # is scale-invariant, meaning the choice of units doesn't affect the
        # inference. The shape=p argument specifies that beta_raw will be a vector of 
        # length p, where p is the number of features in your data.
        beta_raw = pm.Normal(
            'beta_raw',
            mu=model_params["mu"],
            sigma=model_params["sigma"],
            shape=p
        )
        # ind defines a prior distribution for a vector of indicator variables
        # for each feature. It uses a Bernoulli distribution with a probability of
        # success (p=0.2) for each element of ind. This means there's a 20% chance for
        # each element to be 1 (indicating inclusion) and an 80% chance to be 0
        # (exclusion). Setting p=0.2 can be seen as a sparsity assumption, a form of 
        # regularization, of for use in computational efficiency. Rgarding the sparsity
        # assumption, many real-world problems exhibit sparsity, meaning only a small
        # subset of features are truly informative. Rgarding regularization, setting 
        # p=0.2, the model is encouraged to favor simpler models with fewer active
        # features, which helps prevent overfitting. With respect to computational 
        # efficiency, using a smaller p value can lead to computational efficiency gains,
        # especially in high-dimensional problems, as it reduces the number of features
        # that need to be considered. The shape=p argument ensures ind has the same
        # length as beta_raw.
        ind = pm.Bernoulli('ind', p=model_params["prob_of_success"], shape=p)
        # This defines a deterministic variable (beta) that combines the raw
        # coefficients (beta_raw) and the indicator variables (ind). The element-wise
        # multiplication (*) ensures that features with corresponding ind values of 0
        # get effectively set to 0 in beta. This is how variable selection is achieved
        # through the interaction of priors. Features with low selection probability
        # (low ind values) will have their coefficients shrunk towards 0 in the posterior
        # distribution.
        beta = pm.Deterministic('beta', beta_raw * ind)

        # ------------------------------------------------------------------------------
        # Likelihood
        # ------------------------------------------------------------------------------
        # mu defines a deterministic variable (mu) that represents the predicted mean of
        # the response variable (y).It uses the dot product (pm.math.dot) between the
        # imputed data matrix (X_imputed) and the coefficient vector (beta) to calculate
        # the predicted means for each data point.
        mu = pm.Deterministic('mu', pm.math.dot(X_imputed, beta))
        # likelihood defines the likelihood function, representing the probability of 
        # observing the actual data (y) given the predicted means (mu) and the assumed 
        # noise distribution. It uses a normal distribution with the predicted means
        # (mu) as the expected values and the standard deviation (sigma) from the 
        # prior. The observed data (y) is used to compute the likelihood for each data
        # point.
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=y)
        
        # ------------------------------------------------------------------------------
        # Sampling
        # ------------------------------------------------------------------------------
        # This performs Sequential Monte Carlo (SMC) to obtain samples from
        # the posterior distribution of the model parameters. 
        # Draws specifies the number of samples that you want the sampler to
        # generate from the posterior distribution of your model's parameters.
        # Kernel determines the algorithm used to generate the samples.
        # Warmup is where initial samples are discarded before calculating diagnostics
        
        # Threshold 0.2-0.8. Default 0.5 
        # Use higher For complex models or when more fine-grained exploration is needed.
        # Correlation Thresold 0.001 to 0.1. Default 0.01
        # Use lower for more accurate sampling, especially in high-dimensional spaces.
        
        trace = pm.sample_smc(
            draws=model_params["draws"],  # Number of samples (previously nparticles)
            kernel=model_params["kernel"],  # pm uses Independent Metropolis Hastings kernel
            model=model,
            cores=model_params["cores"],
            random_seed=42,
            idata_kwargs = {'log_likelihood': True},       
        )
        
    
#         trace = pm.sample_smc(
#             draws=model_params["draws"],  # Number of samples (previously nparticles)
#             kernel=pm.smc.kernels.IMH(threshold=0.7, correlation_threshold=0.005),
#             model=model,
#             cores=model_params["cores"],
#             random_seed=42,
#             idata_kwargs = {'log_likelihood': True}
#             # kernel_kwargs={
#             #     'threshold': 0.7,
#             #     'correlation_threshold': 0.005
#             # }        
#         )
    
        if save_model:
            save_model(model, model_params["experiment_name"])
            save_trace(trace, model_params["experiment_name"])
    
    # return trace, model
    return trace



def run_optuna_hyperparameter_optimization(
    df_variables,
    df_target,
    model_params,
    n_trials=50,
    sampler_type='QMC',
    # experiment_name="LC_household_pulse_v11_with_race_zscore_norm_optuna",
    study_name = "LC_hyperparam_optimization_2cores" # Unique id of hyperparameters study.
):

    
    def objective(trial):
        """
        Why Negative Log-Likelihood?

        In statistical modeling, we often aim to find the parameters that maximize the likelihood of the observed data. The likelihood function measures the probability of observing the data given a specific set of parameter values. However, directly maximizing the likelihood can be computationally challenging, especially when dealing with products of probabilities.   

        To address this, we often work with the log-likelihood. Taking the logarithm of the likelihood function has several advantages:

        Transforms Product into Sum:

        The logarithm turns products into sums, which are computationally more efficient and numerically stable. This is particularly beneficial when dealing with a large number of data points.   
        Monotonic Transformation:

        The logarithm is a monotonic function, meaning that maximizing the log-likelihood is equivalent to maximizing the likelihood itself. This allows us to optimize the log-likelihood without losing any information.   
        Numerical Stability:

        By taking the logarithm, we can avoid underflow issues that may arise when multiplying very small probabilities.   
        However, most optimization algorithms are designed to minimize functions, not maximize them. Therefore, we introduce the negative sign to convert the maximization problem into a minimization problem.

        In summary, we use the negative log-likelihood as an objective function because:

        It simplifies the optimization process.
        It improves numerical stability.
        It allows us to use efficient optimization algorithms.   
        By minimizing the negative log-likelihood, we effectively maximize the likelihood of the observed data, leading to a better-fitting model.

        """
        # Define hyperparameters using trial.suggest_* methods
        
        # alpha_sigma Controls the "shape" of the distribution. 
        # Higher values lead to a distribution concentrated
        # towards lower sigma (less noise).
        alpha_sigma = trial.suggest_float(
            'alpha_sigma',
            1.0, 10.0, step=0.5 
        )
        
        # beta_sigma: Controls the "scale" of the distribution.
        # Higher values allow for larger stds (more noise).
        beta_sigma = trial.suggest_float(
            'beta_sigma',
            0.1, 1.0, step=0.1
        )
        
        # The actual standard deviation of the noise term in the model.
        sigma = trial.suggest_int(
            'sigma',
            1, 5, step=1
        )
        
        # This defines the prior distribution for the indicator
        # variables (ind) using a Bernoulli distribution. These
        # indicators control which features are included in the model. 
        prob_of_success = trial.suggest_float(
            'prob_of_success',
            0.1, 0.5, step=0.1
        )

        # Update model parameters
        model_params['alpha_sigma'] = alpha_sigma
        model_params['beta_sigma'] = beta_sigma
        model_params['sigma'] = sigma
        model_params['prob_of_success'] = prob_of_success

        trace = bayesian_variable_selection_with_dfs(
            df_variables,  # dataframe containing variables
            df_target,  # series containing Long COVID rates
            'Avg',  # corresponding column header to rate values
            model_params # model parameters for BVS
        )
        
        idata = pm.to_inference_data(trace, log_likelihood=True)
        
        log_likelihood = idata.log_likelihood.likelihood.values
        
        mean_log_likelihood = np.mean(log_likelihood)
        print(mean_log_likelihood)
        
        trial.set_user_attr("mean_log_likelihood", mean_log_likelihood)
        
        return mean_log_likelihood

    
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    
    # Save data in sqlite db
    storage_name = "sqlite:///{}.db".format(study_name) # DB to store hyperparameters study results

    # Use sampler with optuna optimization instead of random sampling
    
    if sampler_type == 'QMC':
        sampler = optuna.samplers.QMCSampler()  
    elif sampler_type == 'TPE':
        sampler = optuna.samplers.TPESampler()
    
    # Create optuna hyperparameters study
    study = optuna.create_study(
        direction='minimize',  # Optimize objective function
        storage=storage_name,  # Store results in local DB
        sampler=sampler,
        load_if_exists=True  # Resume hyperparameter study if exists
    )
    
    df_optuna_name = f'Optuna_best_params_{study_name}_sampler{sampler_type}.csv'
    df_optuna = study.trials_dataframe(attrs=("number", "value", "params", "state")) 
    df_optuna.to_csv(df_optuna_name, index=False) 

    # Optimize hyperparameters (sensitivity analysis)
    study.optimize(objective, n_trials=n_trials)

    # Save the sampler with pickle to be loaded later.
    # with open("sampler.pkl", "wb") as fout:
    #     pickle.dump(study.sampler, fout)

    # Get the best hyperparameters from the study
    best_params = study.best_params
    
    # # #Save parameters associated with the optuna experiment
    # with open(f'Optuna_best_params_{experiment_name}.txt', 'w') as f:
    #     json.dump(params_to_save, f, indent=4)
    
    return best_params

