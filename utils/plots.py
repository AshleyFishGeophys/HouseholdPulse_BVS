import scipy.stats as stats
import scipy.stats as stats_qq
import scipy.cluster.hierarchy as shc 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import seaborn as sns  
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
from matplotlib.ticker import MultipleLocator  # Import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap


def create_qq_plot(
    data: pd.Series
) -> None:
    """Creates a Q-Q plot of the specified Long COVID rate data.

    Args:
        lc_rates (pd.Series):
            A pandas Series containing the Z-Score of the 
            Long COVID rate data.

    Returns:
        None: The function directly creates the plot.
        
    Typical Usage Example: 
        It's often used to assess whether a dataset follows a particular
        theoretical distribution (e.g., normal, lognormal, uniform).
        If the plot is a straight line, it suggests that the two
        distributions are similar. If the plot is not a straight line, it
        indicates that the distributions are different. Deviations from the
        line can reveal the nature of the differences (e.g., skewness, kurtosis).
        
        create_qq_plot(lc_rates_flat_z_scores)

    """

    theoretical_quantiles = stats.norm.ppf(stats.rankdata(data) / (len(data) + 1))

    print(f"theoretical_quantiles min: {theoretical_quantiles.min()}, "\
          f"max: {theoretical_quantiles.max()}")
    print(f"data min: {data.min()}, max: {data.max()}")
    print(f"len(data): {len(data)}")
    
    plt.figure(figsize=(7, 7))

    # Add QQ line and data
    stats.probplot(data, dist="norm", plot=plt)
    
    # Access the scatter plot created by probplot
    scatter_plot = plt.gca().get_children()[0]
    
    # Modify the scatter plot properties
    scatter_plot.set_marker('o')  # Set marker to circles
    scatter_plot.set_markeredgecolor('black')  # Set edge color to black
    scatter_plot.set_markerfacecolor('white')  # Set face color to white for hollow
    scatter_plot.set_markersize(4)  # Adjust the marker size as needed

    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.title("Normal Q-Q Plot")
    plt.grid(False)
    
    plt.show()
    
    
    
def dendrogram(data: pd.DataFrame) -> None:
    """Creates a visualization of hierarchical clustering for
    household pulse long COVID rate data.

    Args:
        data (pd.DataFrame):
            A DataFrame containing z-score normalized household 
            pulse long COVID rate data. Each column represents a 
            state and the index represents the time period.

    Returns:
        None:
            This function creates a plot and does not return any value.
    
    Typical Usage Example: 
        It assumes the DataFrame contains z-score normalized household
        pulse long COVID rate data. Each column represents a state and
        the index represents the time period.
        
        Dendrogram: The dendrogram on the left side of the plot visually
        represents the hierarchical clustering of the data points
        (states, in this case). The height of the branches in the dendrogram
        indicates the similarity or dissimilarity between the clusters.
        
        Heatmap: The heatmap on the right side of the plot displays the values
        of the data points (long COVID rates) in a color-coded matrix. The
        colors represent the magnitude of the values, with darker colors
        indicating higher values.
        
        dendrogram(lc_rates_zscore)
    """
    
    
    # Calculate the linkage matrix
    distance_matrix = shc.distance.pdist(data)  # Use pairwise distances
    linkage_matrix = shc.linkage(distance_matrix, method='ward')

    # Create the figure with a smaller size
    fig = plt.figure(figsize=(11, 7))
    fig.suptitle("Dendrogram: ZScore Long COVID Rate Dates vs States")
    # Create a gridspec layout
    gs = gridspec.GridSpec(
        1,
        2,
        width_ratios=[15, 85],
        figure=fig,
        wspace=0
    )

    axs = gs.subplots(sharex='col')
        
    # Create the dendrogram
    # axs[0] = plt.subplot(gs[0])
    dendrogram = shc.dendrogram(
        linkage_matrix,
        orientation='left',
        no_labels=True,
        ax=axs[0]
    )
    
    # Plot heatmap
    # axs[1] = plt.subplot(gs[1], sharex=ax1)
    
    data_index = data.index
    data_col_names = data.columns
    
    # Remove "State" from idx
    data = data.reset_index(drop=True)
    
    axs[1] = sns.heatmap(
        data,
        cbar=True,  # Display colorbar
        cmap="Reds",  # Choose a colormap (adjust as needed)
        linewidths=0.0,  # No lines between cells
        xticklabels=True,  # Show x-axis labels
        yticklabels=False,  # Don't show y-axis labels
        cbar_kws={'pad': 0.22, 'fraction': 0.1},  # colorbar size and spacing
        center=True
    )
    
    # Add y-axis labels to the right
    # Set the y-axis ticks and labels for ax[1]
    axs[1].set_yticks(range(len(data)))
    axs[1].set_yticklabels(data_index)

    # Set the y-axis ticks and labels for ax[1]
    axs[1].set_xticks(range(len(data_col_names)))
    axs[1].set_xticklabels(data_col_names)

    # Adjust the y-axis position for ax[1]
    axs[1].yaxis.tick_right()

    axs[1].tick_params(axis="y", left=False, labelleft=False) 
    
    plt.show()

    
    
def plot_hist(df: pd.DataFrame) -> None:
    """Plots histograms for all columns in a Pandas DataFrame.

    Args:
        df (pd.DataFrame):
            The Pandas DataFrame to plot.

    Returns:
        None

    Typical Usage Example:
        plot_hist(my_dataframe)
    """
    for var in df.columns: 
        plt.hist(df[[var]])
        plt.title(var)
        plt.xlabel("Value")  # Add x-axis label for clarity
        plt.ylabel("Frequency")  # Add y-axis label
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        plt.show()        
        
def plot_scatter(
    df_variables: pd.DataFrame,
    df_target: pd.DataFrame,
    col_target: str = "Avg"
) -> None:
    """Plots scatter plots of independent predictor variables
    against a target variable.

    Args:
        df_variables (pd.DataFrame):
            Pandas DataFrame of independent predictor variables.
        df_target (pd.DataFrame):
            Pandas DataFrame containing the target variable.
        col_target (str):
            The name of the target variable column. Defaults to "Avg".

    Returns:
        None

    Typical Usage Example:
        Use this to see if there are any visible correlations between
        predictor variables and LC rates.
        
        plot_scatter(predictor_vars_df, LC_rates_df, col_target="Avg")
        
    """    
    for var in df_variables.columns: 
        plt.scatter(df_target[[col_target]], df_variables[[var]])
        plt.title(f"{var} vs. {col_target}")  # More descriptive title
        plt.xlabel(col_target)  # Add x-axis label
        plt.ylabel(var)  # Add y-axis label
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        plt.show()        
        
def plot_correlation_matrix(
    df: pd.DataFrame,
    method: str = 'pearson',
    plot_title: str = 'Correlation Matrix',
    corr_label_size: int = 4,
    axis_label_size: int = 4,
    fig_size: tuple = (10, 8),
    save_image: bool = False,
    image_path: str = None
) -> None:
    ''' Plots a correlation matrix as a heatmap with seaborn
    and optionally saves the figure as an image.

    Args:
        df (pd.DataFrame):
            The pandas DataFrame containing the data.
        method (str):
            Method of correlation between variables
            {‘pearson’, ‘kendall’, ‘spearman’} or callable
            pearson : standard correlation coefficient
            kendall : Kendall Tau correlation coefficient
            spearman : Spearman rank correlation
        save_image (bool):
            A boolean flag to indicate if the plot should be
            saved as an image (default: False).
        plot_title (str):
            The title of the plot (default: 'Correlation Matrix').
        corr_label_size (int):
            The font size for correlation values (default: 4).
        axis_label_size (int):
            The font size for axis labels (default: 4).
        figsize (tuple):
            A tuple specifying the width and height of the figure in
            inches (default: (10, 8)).
        save_image (bool):
            A boolean flag to indicate if the plot should be saved as
            an image (default: False).
        image_path (str):
            The file path to save the image (default: 'correlation_matrix.png').
            
    Typical Usage Example: 
        Understand the correlation between all predictor variables to 
        see if there is visible multicollinearity. 
    '''
    
    # Calculate the correlation matrix
    corr_matrix = df.corr(method=method)

    # Create a heatmap with annotated correlation values and tiny font
    fig, ax = plt.subplots(figsize=fig_size)
    ax = sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        square=True, 
        annot_kws={"fontsize": corr_label_size},
        cbar_kws={"shrink": 0.5}  # Adjust this value to control the colorbar size
    )

    # Customize the plot (optional)
    ax.set_title(f"Correlation Matrix: {method}", fontsize=12)
    
    # Invert the y-axis to position origin at lower left
    ax.invert_yaxis()
    
    ax.set_xticks(
        range(len(corr_matrix.columns)),
        corr_matrix.columns,
        rotation=45,
        ha='right'
    )
    
    ax.set_yticks(
        range(len(corr_matrix.columns)),
        corr_matrix.columns,
        # rotation=45,
        # ha='left'
    )
    
    # Set font size for both x and y ticks
    ax.tick_params(axis='both', labelsize=axis_label_size)  

    plt.tight_layout()
    
    if save_image:
        plt.savefig(image_path)
        print(f"Correlation matrix saved to image: {image_path}")
    
    plt.show()
    
    

def plot_p_values(
    df_results:pd.DataFrame,
    alpha: float=0.05
) -> None:
    """ Generates a grouped bar plot of corrected p-values
    (Pearson, Spearman, Kendall) for each variable, with a
    horizontal line indicating the alpha level.

    Args:
        df_results (pd.DataFrame):
            Pandas DataFrame containing the results of correlation tests,
            including p-values for Pearson, Spearman, and Kendall.
        alpha (float, optional):
            Significance level. A horizontal line will be drawn at
            this value. Defaults to 0.05.

    Returns:
        None (displays the plot).
    """
    
    variables = df_results['Variable']
    p_value_pearson = df_results['Pearson_p_value'].astype(float) #Convert to float.
    p_value_spearman = df_results['Spearman_p_value'].astype(float) #Convert to float.
    p_value_kendall = df_results['Kendall_p_value'].astype(float) #Convert to float.

    width = 0.2  # Width of each bar

    x = range(len(variables))  # X-axis positions for the bars

    plt.figure(figsize=(12, 6))  # Adjust figure size as needed

    # Plot the p-values as grouped bar charts
    plt.bar([i - width for i in x], p_value_pearson, width, label='Pearson p-value')
    plt.bar(x, p_value_spearman, width, label='Spearman p-value')
    plt.bar([i + width for i in x], p_value_kendall, width, label='Kendall p-value')

    # Set x-axis labels and rotate for readability
    plt.xticks(x, variables, rotation=45, ha='right')  # Set x-axis labels and rotate
    plt.ylabel("P-value")
    plt.title("NULL Hypothesis: P-values")
    plt.legend()
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.grid(axis='y', alpha=0.5)  # Add a subtle grid
    
    if alpha is not None:  # Add horizontal line if alpha is provided
        plt.axhline(y=alpha, color='r', linestyle='--', label=f"Alpha = {alpha}")
        plt.legend()  # Update the legend to include the alpha line

    plt.show()
    
    
def plot_p_values_false_discovery_corrected(
    df_results: pd.DataFrame,
    alpha: float=0.05
) -> None:
    """Generates a grouped bar plot of corrected p-values
    (Pearson, Spearman, Kendall) for each variable, with an optional
    horizontal line indicating the alpha level.

    Args:
        df_results (pd.DataFrame):
            Pandas DataFrame containing the results of correlation tests,
            including FD corrected p-values for Pearson, Spearman, and Kendall.
            
        alpha (float, Optional): 
            Significance level. If provided, a horizontal line will be drawn at
            this value. Defaults to 0.05.

    Returns:
        None (displays the plot).

    """
    
    variables = df_results['Variable']
    p_value_pearson = df_results['Pearson_p_value_corrected'].astype(float) #Convert to float.
    p_value_spearman = df_results['Spearman_p_value_corrected'].astype(float) #Convert to float.
    p_value_kendall = df_results['Kendall_p_value_corrected'].astype(float) #Convert to float.

    width = 0.2  # Width of each bar

    x = range(len(variables))  # X-axis positions for the bars

    plt.figure(figsize=(12, 6))  # Adjust figure size as needed

    # Plot the corrected p-values as grouped bar charts
    plt.bar([i - width for i in x], p_value_pearson, width, label='Pearson p-value corrected')
    plt.bar(x, p_value_spearman, width, label='Spearman p-value corrected')
    plt.bar([i + width for i in x], p_value_kendall, width, label='Kendall p-value corrected')

    plt.xticks(x, variables, rotation=45, ha='right')  # Set x-axis labels and rotate
    plt.ylabel("P-value corrected")
    plt.title("NULL Hypothesis: P-values corrected")
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.5)  # Add a subtle grid
    
    if alpha is not None:  # Add horizontal line if alpha is provided
        plt.axhline(y=alpha, color='r', linestyle='--', label=f"Alpha = {alpha}")
        plt.legend()  # Update the legend to include the alpha line

    plt.show()


# def plots_null_hypothesis(
#     df_vars: pd.DataFrame,
#     target: pd.Series
# ) -> None:
#     """ Generates plots to visually assess the relationship between
#     numeric variables and a target variable, including regression plots,
#     histograms, and Q-Q plots.

#     This function scales numeric variables, imputes missing values,
#     and then creates visualizations for each variable to help assess the
#     null hypothesis (typically that there is no relationship between the
#     variable and the target).

#     Args:
#         df_vars (pd.DataFrame):
#             Pandas DataFrame containing the independent variables.  Only numeric
#             columns will be used.
#         target (pd.Series):
#             Pandas Series or array-like containing the target variable.

#     Returns:
#         None (displays the plots).  Prints an error message if no numeric columns
#         are found in `df_vars`.
#     """
#     # Select numeric columns
#     numeric_vars = df_vars.select_dtypes(include='number')
#     if numeric_vars.empty: # Check if there are any numeric columns
#         print("Error: No numeric columns found in the DataFrame.")
#         return

#     # Convert numeric variables to a NumPy array for scaling and imputation
#     X = numeric_vars.values
#     target_array = np.array(target) # Target to numpy array.

#     # Scale data (z-score scaling)
#     X_scaled = X.copy()
#     scaler = StandardScaler() # Create a StandardScaler object
#     X_scaled = scaler.fit_transform(X) # Scale the numeric variables

#     # Impute missing values, if any
#     imputer = SimpleImputer(strategy='mean') # Create a SimpleImputer object
#     X_imputed = imputer.fit_transform(X_scaled) # Impute missing values using the mean

#     for i, var_name in enumerate(numeric_vars.columns):  # Iterate through columns
#         variable_data = X_imputed[:, i]  # Get the data for the current variable

#         # --- Visualizations ---
#         # 1. Scatter Plot & Regression Line
#         plt.figure(figsize=(10, 6))
#         sns.regplot(x=variable_data, y=target_array, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
#         plt.xlabel(var_name, fontsize=12)
#         plt.ylabel("Target Variable", fontsize=12)
#         plt.title(f"Scatter Plot & Regression Line of {var_name} vs. Target", fontsize=14)
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

#         # 2. Histogram for Normality Check
#         plt.figure(figsize=(8, 5))
#         sns.histplot(variable_data, kde=True)
#         plt.xlabel(var_name, fontsize=12)
#         plt.title(f"Histogram of {var_name}", fontsize=14)
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

#         # 3. Q-Q Plot for Normality Check
#         plt.figure(figsize=(8, 5))
#         sm.qqplot(variable_data, stats_qq.norm, line='s')
#         plt.title(f"Q-Q Plot of {var_name}", fontsize=14)
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

def plots_null_hypothesis(
    df_vars: pd.DataFrame,
    target: pd.Series
) -> None:
    """ Generates plots to visually assess the relationship between
    numeric variables and a target variable, including regression plots,
    histograms, and Q-Q plots.

    This function scales numeric variables, imputes missing values,
    and then creates visualizations for each variable to help assess the
    null hypothesis (typically that there is no relationship between the
    variable and the target).

    Args:
        df_vars (pd.DataFrame):
            Pandas DataFrame containing the independent variables.  Only numeric
            columns will be used.
        target (pd.Series):
            Pandas Series or array-like containing the target variable.

    Returns:
        None (displays the plots).  Prints an error message if no numeric columns
        are found in `df_vars`.
    """
    # Select numeric columns
    numeric_vars = df_vars.select_dtypes(include='number')
    if numeric_vars.empty: # Check if there are any numeric columns
        print("Error: No numeric columns found in the DataFrame.")
        return

#     # Convert numeric variables to a NumPy array for scaling and imputation
    X = numeric_vars.values
    target_array = np.array(target) # Target to numpy array.

    # Scale data (z-score scaling)
    X_scaled = X.copy()
    scaler = StandardScaler() # Create a StandardScaler object
    X_scaled = scaler.fit_transform(X) # Scale the numeric variables

    # Impute missing values, if any
    imputer = SimpleImputer(strategy='mean') # Create a SimpleImputer object
    X_imputed = imputer.fit_transform(X_scaled) # Impute missing values using the mean



    for i, var_name in enumerate(numeric_vars.columns):  # Iterate through columns
        # variable_data = X_imputed[:, i]  # Get the data for the current variable
        variable_data = X[:, i]  # Get the data for the current variable

        # --- Visualizations ---
        # 1. Scatter Plot & Regression Line
        plt.figure(figsize=(10, 6))
        sns.regplot(x=X[:, i], y=target_array, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
        plt.xlabel(var_name, fontsize=12)
        plt.ylabel("Long COVID Rates", fontsize=12)
        plt.title(f"Scatter Plot & Regression Line of {var_name} vs. Long COVID Rates", fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 2. Histogram for Normality Check
        plt.figure(figsize=(8, 5))
        sns.histplot(X[:, i], kde=True)
        plt.xlabel(var_name, fontsize=12)
        plt.title(f"Histogram of {var_name}", fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 3. Q-Q Plot for Normality Check
        plt.figure(figsize=(8, 5))
        sm.qqplot(X_imputed[:, i], stats_qq.norm, line='s')
        plt.title(f"Q-Q Plot of {var_name}", fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def clean_feature_name(name: str) -> str:
    """Cleans a feature name.
    args:
        name(str):
            name of the feature to clean up for readability
    """
    name = re.sub(r'[^a-zA-Z0-9_]+', ' ', name).strip()  # Replace non-alphanumeric with spaces
    name = name.replace(' ', '_')  # Replace spaces with underscores
    
    return name


def plot_importance_with_errors(
    df: pd.DataFrame,
    title: str="Feature Importance"
) -> None:
    """Plots feature importance with error bars, sorted by importance.

    Args:
        df (pd.DataFrame):
            Dataframe with 'Importance' and 'SEM' columns.
            The index of the DataFrame should be the feature names.
        title (str):
            Plot title.
    """

    df_results = pd.DataFrame(
        {'Importance': beta_means_results['mean'],
         'SEM': beta_means_results['mcse_mean']},
        index=beta_means_results.index
    )    
    
    # 1. Clean Feature Names FIRST
    df_results.index = df_results.index.map(clean_feature_name) #Clean the index

    # 2. Sort by Importance
    df_sorted = df_results.sort_values('Importance', ascending=False)

    # 3. Reset the index
    df_plot = df_sorted.reset_index().rename(columns={'index': 'Feature'})

    # 4. Convert 'Feature' to categorical
    df_plot['Feature'] = df_plot['Feature'].astype('category')

    # 5. Convert 'Importance' to numeric (just in case)
    df_plot['Importance'] = pd.to_numeric(df_plot['Importance'])

    # ***Explicitly set the order for barplot***
    feature_order = df_sorted.index.tolist() # Get the list of the sorted features

    # Create a custom colormap (red to blue)
    cmap = LinearSegmentedColormap.from_list("my_cmap", ["cyan", "orange"])

    # Normalize importance values for the colormap
    norm = plt.Normalize(df_plot['Importance'].min(), df_plot['Importance'].max())

    
    plt.figure(figsize=(8, 8))
    sns.set_style("whitegrid")

    ax = sns.barplot(
        x='Importance',  # Importance is now on the x-axis
        y='Feature',  # Features are now on the y-axis
        data=df_plot,
        xerr=df_plot['SEM'],  # Error bars are horizontal
        capsize=0,
        order=feature_order,
        palette=cmap(norm(df_plot['Importance'])),  # Apply colormap
    )
    
    for i, row in df_plot.iterrows():
        importance = row['Importance']
        sem = row['SEM']
        ax.errorbar(
            x=importance,  # Center of the error bar
            y=i ,  # Vertical position of the error bar (adjust 0.4 for centering)
            xerr=sem,  # Error bar length (SEM)
            fmt='none',  # Don't plot the data point itself
            ecolor='black',  # Error bar color
            capsize=5,  # Size of the caps (adjust as needed)
            elinewidth=1 # error line width
        )
    
    ax.grid(axis='x', linestyle='--', alpha=0.7)  # Vertical gridlines (x-axis)
    ax.grid(axis='y', linestyle='--', alpha=0.7)  # Horizontal gridlines (y-axis)


    ax.xaxis.set_major_locator(MultipleLocator(0.025))  # Set ticks at multiples of 0.5
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))  # Set ticks at multiples of 0.5

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.ylabel("Features")
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
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