"""Functions for creating heatmaps of ENCODE ChIP-seq signal."""

from collections.abc import Iterable

import datashader as ds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datashader.mpl_ext import dsshow


def plot_components(
    dataset, component_1="PCA1", component_2="PCA2", cluster_col=None, name_col="name", filename=None, remove_ticks=True
):
    """Plot the components of a dataset in a grid.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to plot
    component_1 : str
        The name of the column containing the first component
    component_2 : str
        The name of the column containing the second component
    cluster_col : str
        The name of the column containing the cluster labels
    name_col : str
        The name of the column containing the dataset names
    filename : str
        The filename to save the plot to
    remove_ticks : bool
        Whether to remove the axis ticks

    """
    n_files = len(dataset[name_col].unique())
    grid_size = int(np.ceil(np.sqrt(n_files)))
    _, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    xmin = dataset[component_1].min()
    xmax = dataset[component_1].max()
    ymin = dataset[component_2].min()
    ymax = dataset[component_2].max()
    for i, name in enumerate(dataset[name_col].unique()):
        plotdf = dataset[dataset[name_col] == name]
        if cluster_col is not None:
            plotdf = plotdf.loc[plotdf.good_bin, [component_1, component_2, cluster_col]]
            plotdf[cluster_col] = pd.Categorical(plotdf[cluster_col])
            dsshow(
                plotdf,
                ds.Point(component_1, component_2),
                aggregator=ds.count_cat(cluster_col),
                aspect="auto",
                ax=axs[i // grid_size, i % grid_size],
                x_range=(xmin, xmax),
                y_range=(ymin, ymax),
            )
        else:
            plotdf = plotdf.loc[plotdf.good_bin, [component_1, component_2]]
            dsshow(
                plotdf,
                ds.Point(component_1, component_2),
                aspect="auto",
                ax=axs[i // grid_size, i % grid_size],
                x_range=(xmin, xmax),
                y_range=(ymin, ymax),
            )
        axs[i // grid_size, i % grid_size].set_title(name)

        if remove_ticks:
            axs[i // grid_size, i % grid_size].set_xticks([])
            axs[i // grid_size, i % grid_size].set_yticks([])

    if filename is not None:
        plt.savefig(filename)


def plot_signal(
    dataset,
    component_1: str,
    component_2: str,
    signals: Iterable,
    labels: list | None = None,
    filename=None,
    remove_ticks=True,
):
    """Plot signal values overlaying an embedding.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to plot
    component_1 : str
        The name of the column containing the first component
    component_2 : str
        The name of the column containing the second component
    signals : list
        The list of columns containing the signal values
    labels : list
        The list of columns containing the cluster labels
    filename : str
        The filename to save the plot to
    remove_ticks : bool
        Whether to remove the axis ticks

    """
    df = dataset.dropna().copy()

    # If length of unique label values is >20, set all values >20 to 20
    if labels is not None:
        for label in labels:
            if len(df[label].unique()) > 20:
                df[label] = df[label].apply(lambda x: x if x < 20 else 20)

    n_plots = len(signals) + len(labels)
    grid_size = int(np.ceil(np.sqrt(n_plots)))
    _, axs = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    xmin = df[component_1].min()
    xmax = df[component_1].max()
    ymin = df[component_2].min()
    ymax = df[component_2].max()

    # Plot categorical labels
    if labels is not None:
        for i, label in enumerate(labels):
            df[label] = pd.Categorical(df[label])
            dsshow(
                df,
                ds.Point(component_1, component_2),
                aggregator=ds.count_cat(label),
                aspect="auto",
                ax=axs[i // grid_size, i % grid_size],
                x_range=(xmin, xmax),
                y_range=(ymin, ymax),
            )
            axs[i // grid_size, i % grid_size].set_title(label)
            if remove_ticks:
                axs[i // grid_size, i % grid_size].set_xticks([])
                axs[i // grid_size, i % grid_size].set_yticks([])

    # Plot signal values
    for i, signal in enumerate(signals):
        dsshow(
            df,
            ds.Point(component_1, component_2),
            aggregator=ds.mean(signal),
            aspect="auto",
            ax=axs[(i + len(labels)) // grid_size, (i + len(labels)) % grid_size],
            x_range=(xmin, xmax),
            y_range=(ymin, ymax),
            cmap="seismic",
        )
        axs[(i + len(labels)) // grid_size, (i + len(labels)) % grid_size].set_title(signal)
        if remove_ticks:
            axs[(i + len(labels)) // grid_size, (i + len(labels)) % grid_size].set_xticks([])
            axs[(i + len(labels)) // grid_size, (i + len(labels)) % grid_size].set_yticks([])

    if filename is not None:
        plt.savefig(filename)


def create_dotplot(
    df: pd.DataFrame, cluster_column: str, signal_columns: list[str], threshold: float, filename: str | None = None
):
    """Create a dot plot of signal values for each cluster.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to plot
    cluster_column : str
        The name of the column containing the cluster labels
    signal_columns : List[str]
        The names of the columns containing the signal values
    threshold : float
        The threshold to use for the percentage calculation
    filename : str
        The filename to save the plot to

    """
    # Calculate fractions and means
    fractions_above_threshold = (
        df[[*signal_columns, cluster_column]]
        .groupby(cluster_column)
        .apply(lambda x: (x[signal_columns] > threshold).sum() / len(x))
        .reset_index()
    )
    means = df[[*signal_columns, cluster_column]].groupby(cluster_column).mean().reset_index()

    # Z-scale means by each signal
    for signal in signal_columns:
        means[signal] = (means[signal] - means[signal].mean()) / means[signal].std()

    # Prepare the plot data
    plot_data = []
    for col in signal_columns:
        for cluster in df[cluster_column].unique():
            fraction = fractions_above_threshold.loc[fractions_above_threshold[cluster_column] == cluster, col].values[
                0
            ]
            mean = means.loc[means[cluster_column] == cluster, col].values[0]
            plot_data.append({"Cluster": cluster, "Signal": col, "Fraction": fraction, "Mean": mean})

    # Plotting
    plot_df = pd.DataFrame(plot_data)
    _, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=plot_df, x="Cluster", y="Signal", size="Fraction", hue="Mean", ax=ax, legend=False, palette="seismic"
    )
    plt.xlabel("Cluster")
    plt.ylabel("Signal")

    if filename:
        plt.savefig(filename)
