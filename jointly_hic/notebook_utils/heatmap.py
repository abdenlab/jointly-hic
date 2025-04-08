"""Functions for creating heatmaps of ENCODE ChIP-seq signal."""

import matplotlib.pyplot as plt
import pandas as pd

from jointly_hic.notebook_utils.encode_utils import EncodeFile, get_signal


def encode_chip_heatmap(
    bins: pd.DataFrame, data_cols: list, name: str, bigwigs: dict, labels: list, output_prefix: str | None = None
):
    """Create a heatmap of ENCODE ChIP-seq signal for a given search term.

    Parameters
    ----------
    bins : pd.DataFrame
        The bins to plot
    data_cols : list
        The list of columns containing the data to plot
    name : str
        The name of the dataset to plot, since bins contains seeveral datasets
    bigwigs : dict["label": "accession"]
        A dictionary of encode bigwigs with the label as the key and the accession as the value.
    labels : str
        The name of the column containing the cluster labels
    output_prefix : str
        The prefix to use for the output files

    """
    df = bins.loc[bins.name == name, :].dropna().copy()

    # Get the signal for each bigwig
    for label, accession in bigwigs.items():
        with EncodeFile(accession, "bw") as bigwig:
            df[label] = get_signal(bigwig, df)
            data_cols.append(label)

    # Clip at 1% and 99% and Z-score data columns
    for col in data_cols:
        df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Convert label columns to Categorical
    for label in labels:
        df[label] = pd.Categorical(df[label])

    # Sort dataframe
    df = df.sort_values([*labels, "chrom", "start"]).reset_index(drop=True)

    # Save dataframe to parquet
    if output_prefix is not None:
        df.to_parquet(f"{output_prefix}.pq")

    # Set up plot
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(wspace=0.15)
    gs = plt.GridSpec(
        nrows=1 + len(data_cols),
        ncols=1,
        height_ratios=[2] * len(labels) + [1] * len(data_cols),
        hspace=0,
    )

    # Plot labels
    for i, label in enumerate(labels):
        # Plot label values
        ax = plt.subplot(gs[i])
        ax.imshow(
            df[label].to_numpy().reshape(1, -1),
            aspect="auto",
            cmap="tab20",
            rasterized=True,
        )
        ax.set_ylabel(label, rotation="horizontal", ha="right", va="center")
        ax.set_aspect("auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.set_visible(False)

    # Plot signals
    for i, signal in enumerate(data_cols):
        ax = plt.subplot(gs[i + len(labels)])
        ax.imshow(
            df[signal].to_numpy().reshape(1, -1),
            aspect="auto",
            cmap="seismic",
            rasterized=True,
        )
        y_label = ".".join(signal.split(".")[0:2])
        ax.set_ylabel(y_label, rotation="horizontal", ha="right", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    # Save figure
    if output_prefix is not None:
        fig.savefig(f"{output_prefix}_heatmap.png", dpi=600, bbox_inches="tight")

    return df
