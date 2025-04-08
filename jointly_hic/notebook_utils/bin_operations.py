"""Utility functions for working with genomic bins."""

import os

import bioframe
import numpy as np
import pandas as pd

# Set the cache directory, default to ~/cache if not set
CACHE_DIR = os.getenv("CACHE_DIR", os.path.expanduser("~/cache"))


def distance_traveled_by_bin(
    df: pd.DataFrame, time_column: str, time_points: list[str], prefix: str = "PCA"
) -> pd.DataFrame:
    """Calculate the distance traveled by each bin in the PCA space by each bin across time points.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with PCA coordinates for each bin at each time point.
    time_column : str
        Column name for the time points.
    time_points : List[str]
        List of time points to calculate distance traveled.
    prefix : str
        Prefix for the component columns.

    """
    # Filter rows to only those in time_points
    df = df.dropna().copy()

    # Pivot the dataframe to have one row per bin and one column per time point
    df = df.pivot(index=["chrom", "start"], columns=time_column, values=[c for c in df.columns if c.startswith(prefix)])

    # Calculate the distance between each time point
    distance_columns = []
    for i in range(1, len(time_points)):
        # All components for the first time point
        start = df.xs(time_points[i - 1], axis=1, level=1)
        # All components for the second time point
        end = df.xs(time_points[i], axis=1, level=1)
        # Calculate the distance between the two time points
        distance = np.linalg.norm(start - end, axis=1)
        # Add the distance to the dataframe
        df[f"{time_points[i - 1]}_{time_points[i]}_distance"] = distance
        distance_columns.append(f"{time_points[i - 1]}_{time_points[i]}_distance")

    # Sum the distance across all components
    df["total_distance"] = df[distance_columns].sum(axis=1)

    # Return the chrom, start, and total distance
    df = df[["total_distance"]].reset_index()

    # Reset column name (total_distance, None) to total_distance
    df.columns = df.columns.get_level_values(0)

    return df


def make_chromarms(chromsizes, mids, binsize=None, suffixes=("p", "q")):
    """Split chromosomes into chromosome arms.

    Parameters
    ----------
    chromsizes : pandas.Series
        Series mapping chromosomes to lengths in bp.
    mids : dict-like
        Mapping of chromosomes to midpoint locations.
    binsize : int, optional
        Round midpoints to nearest bin edge for compatibility with a given
        bin grid.
    suffixes : tuple, optional
        Suffixes to name chromosome arms. Defaults to p and q.

    Returns
    -------
    4-column BED-like DataFrame (chrom, start, end, name).
    Arm names are chromosome names + suffix.
    Any chromosome not included in ``mids`` will be omitted.

    """
    chromosomes = [chrom for chrom in chromsizes.index if chrom in mids]

    p_arms = [[chrom, 0, mids[chrom], chrom + suffixes[0]] for chrom in chromosomes]
    if binsize is not None:
        for x in p_arms:
            x[2] = round(x[2] / binsize) * binsize

    q_arms = [[chrom, mids[chrom], chromsizes[chrom], chrom + suffixes[1]] for chrom in chromosomes]
    if binsize is not None:
        for x in q_arms:
            x[1] = round(x[1] / binsize) * binsize

    interleaved = [*sum(zip(p_arms, q_arms), ())]

    return pd.DataFrame(interleaved, columns=["chrom", "start", "end", "name"])


def assign_arm_labels(df: pd.DataFrame, arms: pd.DataFrame) -> pd.DataFrame:
    """Assign chromosome arm labels to bins in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with chrom, start, and end columns.
    arms : pd.DataFrame
        DataFrame with chrom, start, end, and name columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with arm column added.

    """
    g = {
        arm["name"]: bioframe.select(df, (arm.chrom, arm.start, arm.end)).assign(arm=arm["name"])
        for _, arm in arms.iterrows()
    }
    return pd.concat(g.values(), ignore_index=True)


def assign_centel_distance(group: pd.DataFrame, arms: pd.DataFrame) -> pd.DataFrame:
    """Assign centromere-proximal distance to bins in a chromosome arm.

    Parameters
    ----------
    group : pd.DataFrame
        DataFrame with chrom, start, and end columns.
    arms : pd.DataFrame
        DataFrame with chrom, start, end, and name columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with centel column added.

    """
    arms = arms.set_index("name")
    this_arm = group.name
    if group.name.endswith("p"):
        arm_len = arms.loc[this_arm, "end"]
        return 1 - (group["end"] / arm_len)
    elif group.name.endswith("q"):
        arm_start = arms.loc[this_arm, "start"]
        arm_len = arms.loc[this_arm, "end"] - arm_start
        return (group["end"] - arm_start) / arm_len
    else:
        return group.assign(dummy=np.nan)["dummy"]


def centel_distance(assembly: str = "hg38", binsize: int = 50000) -> pd.DataFrame:
    """Calculate centromere-proximal distance for each bin in the genome.

    Parameters
    ----------
    assembly : str
        Genome assembly to use. Default is 'hg38'.
    binsize : int
        Bin size in bp. Default is 50,000.

    """
    # Check for cached results
    cache_path = f"{CACHE_DIR}/centel_distance_{assembly}_{binsize}.parquet"
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    chrom_sizes = bioframe.fetch_chromsizes(assembly)
    # SHORT_ARMS: list = ['chr13p', 'chr14p', 'chr15p', 'chr21p', 'chr22p', 'chrYp', 'chrYq']

    # Grab centromere locations and make a chromosome arms dataframe
    centros = bioframe.fetch_centromeres("hg38").set_index("chrom")
    arms = make_chromarms(chrom_sizes, centros["mid"], binsize)

    # Calculate mapping of chromosome arm labels to their lengths
    armlens: dict = arms.assign(length=arms["end"] - arms["start"]).set_index("name")["length"].to_dict()

    # Make a bin table
    df = bioframe.binnify(chrom_sizes, binsize)

    # Calculate GC content
    # fa_records = bioframe.load_fasta('/net/levsha/share/lab/genomes/hg38/hg38.fa')
    # df = bioframe.frac_gc(df, fa_records)

    # To each bin, assign arm label, arm length, and relative/absolute centel distance
    df = assign_arm_labels(df, arms)
    df["armlen"] = df["arm"].apply(armlens.get)
    df["centel"] = df.groupby("arm", sort=False).apply(assign_centel_distance, arms=arms).reset_index(drop=True)
    df["centel_abs"] = np.round(df["centel"] * df["armlen"]).astype(int)

    # Save bin table for caching
    df.to_parquet(cache_path)
    return df
