import scipy.stats as stats
import scipy.cluster.hierarchy as shc 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns  

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

    
    
def plot_hist(df):
    for var in df.columns: 
        plt.hist(df[[var]])
        plt.title(var)
        plt.show()
        
        
def plot_scatter(df_variables, df_target, col_target="Avg"):
    for var in df_variables.columns: 
        plt.scatter(df_target[[col_target]], df_variables[[var]])
        plt.title(var)
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
    '''
    Plots a correlation matrix as a heatmap with seaborn and optionally saves the figure as an image.

    Args:
        df: The pandas DataFrame containing the data.
        method: Method of correlation between variables
            {‘pearson’, ‘kendall’, ‘spearman’} or callable
            pearson : standard correlation coefficient
            kendall : Kendall Tau correlation coefficient
            spearman : Spearman rank correlation
        save_image: A boolean flag to indicate if the plot should be saved as an image (default: False).
        plot_title: The title of the plot (default: 'Correlation Matrix').
        corr_label_size: The font size for correlation values (default: 4).
        axis_label_size: The font size for axis labels (default: 4).
        figsize: A tuple specifying the width and height of the figure in inches (default: (10, 8)).
        save_image: A boolean flag to indicate if the plot should be saved as an image (default: False).
        image_path: The file path to save the image (default: 'correlation_matrix.png').
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
    
    ax.tick_params(axis='both', labelsize=axis_label_size)  # Set font size for both x and y ticks

    plt.tight_layout()
    
    if save_image:
        plt.savefig(image_path)
        print(f"Correlation matrix saved to image: {image_path}")
    
    plt.show()