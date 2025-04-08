"""Utilties to convert a dataframe to a bed9 file with clusters."""

import bioframe
import click
import numpy as np
import pandas as pd
import seaborn as sns
from bioframe import ops
from bioframe.core.specs import _get_default_colnames


def mark_runs(
    df: pd.DataFrame,
    col: str,
    *,
    allow_overlaps: bool = False,
    reset_counter: bool = True,
    run_col: str = "run",
    cols: tuple[str, str, str] | None = None,
) -> pd.DataFrame:
    """Mark runs of immediately consecutive intervals.

    Parameters
    ----------
    df : DataFrame
        A bioframe dataframe.
    col : str
        The column to mark runs of values for.
    allow_overlaps : bool, optional [default: False]
        If True, allow intervals in ``df`` to overlap. This may cause
        unexpected results.
    reset_counter : bool, optional [default: True]
        If True, reset the run counter for each chromosome.
    run_col : str, optional [default: 'run']
        The name of the column to store the run numbers in.
    cols : tuple of str, optional
        The names of the columns in the input dataframe to use. If not
        provided, the default column names are used.

    Returns
    -------
    pandas.DataFrame
        A reordered copy the input dataframe with an additional column 'run'
        marking runs of values in the input column.

    See Also
    --------
    merge_runs

    """
    ck, sk, ek = _get_default_colnames() if cols is None else cols

    if not allow_overlaps and len(ops.overlap(df, df)) > len(df):
        raise ValueError("Not a proper bedGraph: found overlapping intervals.")

    result = []
    n_runs = 0

    for _, group in df.groupby(ck, sort=False):
        group = group.sort_values([sk, ek])
        starts = group[sk].to_numpy()
        ends = group[ek].to_numpy()

        # Extend ends by running max
        ends = np.maximum.accumulate(ends)

        # Find borders of interval clusters and assign cluster ids
        is_cluster_border = np.r_[True, starts[1:] > ends[:-1], False]

        # Find borders of consecutive equal values
        values = group[col].to_numpy()
        if values.dtype.kind == "f":
            is_value_border = np.r_[True, ~np.isclose(values[1:], values[:-1], equal_nan=True), False]
        else:
            is_value_border = np.r_[True, values[1:] != values[:-1], False]

        # Find index extents of runs
        is_border = is_cluster_border | is_value_border
        sum_borders = np.cumsum(is_border)
        run_ids = sum_borders[:-1] - 1

        # Assign run numbers to intervals
        if reset_counter:
            n_runs = 0
        group[run_col] = n_runs + run_ids
        n_runs += sum_borders[-1]

        result.append(group)

    return pd.concat(result)


def merge_runs(
    df: pd.DataFrame,
    col: str,
    *,
    allow_overlaps: bool = False,
    agg: dict | None = None,
    cols: tuple[str, str, str] | None = None,
) -> pd.DataFrame:
    """Merge runs of immediately consecutive intervals sharing the same value.

    Parameters
    ----------
    df : DataFrame
        A bioframe dataframe.
    col : str
        The column to compress runs of values for.
    allow_overlaps : bool, optional [default: False]
        If True, allow intervals in ``df`` to overlap. This may cause
        unexpected results.
    agg : dict, optional [default: None]
        A dictionary of additional column names and aggregation functions to
        apply to each run. Takes the format:
            {'agg_name': ('column_name', 'agg_func')}
    cols : tuple of str, optional
        The names of the columns in the input dataframe to use. If not
        provided, the default column names are used.

    Returns
    -------
    pandas.DataFrame
        Dataframe with consecutive intervals in the same run merged.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'chrom': ['chr1', 'chr1', 'chr1', 'chr1', 'chr1', 'chr1'],
    ...     'start': [0, 100, 200, 300, 400, 500],
    ...     'end': [100, 200, 300, 400, 500, 600],
    ...     'value': [1, 1, 1, 2, 2, 2],
    ... })

    >>> merge_runs(df, 'value')
        chrom  start  end  value
    0   chr1      0  300      1
    1   chr1    300  600      2

    >>> merge_runs(df, 'value', agg={'sum': ('value', 'sum')})
        chrom  start  end  value  sum
    0   chr1      0  300      1    3
    1   chr1    300  600      2    6

    See Also
    --------
    mark_runs

    """
    ck, sk, ek = _get_default_colnames() if cols is None else cols

    if agg is None:
        agg = {}

    df_runs = mark_runs(
        df,
        col,
        allow_overlaps=allow_overlaps,
        reset_counter=False,
        run_col="_run",
    )
    df_merged = df_runs.groupby("_run").agg(
        **{ck: (ck, "first"), sk: (sk, "min"), ek: (ek, "max"), col: (col, "first"), **agg}
    )
    return df_merged.reset_index(drop=True)


@click.command()
@click.option(
    "--input_file",
    "-i",
    required=True,
    help="Input file (parquet, pq or csv(.gz))",
)
@click.option(
    "--output_prefix",
    "-o",
    required=True,
    help="Output file prefix",
)
@click.option(
    "--cluster_col",
    "-c",
    default="leiden",
    help="Column name for clusters",
)
@click.option(
    "--palette",
    "-p",
    default="tab20",
    help="Color palette",
)
@click.option(
    "--name_column",
    "-n",
    default="filename",
    help="Column name for filenames",
)
def create_bed9_clusters(
    input_file: str,
    output_prefix: str,
    cluster_col: str = "leiden",
    palette: str = "tab20",
    name_column: str = "filename",
):
    """Create bed9 files for each cluster in the input file."""
    # Read input file (can be a parquet, pq or a csv(.gz) file)
    if input_file.endswith(".pq") or input_file.endswith(".parquet"):
        df = pd.read_parquet(input_file)
    elif input_file.endswith(".csv") or input_file.endswith(".csv.gz"):
        df = pd.read_csv(input_file)
    else:
        raise ValueError(f"Input file format not supported: {input_file}")

    # Convert start and end to integers, drop NAs, convert cluster_col to Categorical
    df = df.dropna(subset=cluster_col)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df[cluster_col] = df[cluster_col].astype(int).astype("category")

    # Create palette
    palette = sns.color_palette(palette, len(df[cluster_col].unique()))
    palette_mapping = {k: v for k, v in zip(df[cluster_col].unique(), palette)}
    colors = {k: bioframe.to_ucsc_colorstring(v) for k, v in palette_mapping.items()}

    # Iterate through all name_columns in df and create bed9 files
    for i, dataset_df in df.groupby(name_column):
        merged_df = merge_runs(dataset_df, cluster_col)
        result = pd.DataFrame(
            {
                "chrom": merged_df["chrom"],
                "start": merged_df["start"],
                "end": merged_df["end"],
                "name": merged_df[cluster_col],
                "score": 0,
                "strand": ".",
                "thickStart": merged_df["start"],
                "thickEnd": merged_df["end"],
                "itemRgb": merged_df[cluster_col].map(colors),
            }
        )
        result.to_csv(f"{output_prefix}_{i}.bed.gz", sep="\t", header=False, index=False)


# Create CLI entrypoint for create_bed9_clusters
if __name__ == "__main__":
    create_bed9_clusters()
