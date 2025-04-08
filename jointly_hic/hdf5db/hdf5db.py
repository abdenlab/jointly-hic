"""Create an HDF5 database of metadata, embeddings and tracks from jointly."""

import argparse
import concurrent
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pandas as pd
import yaml

from jointly_hic.notebook_utils.encode_utils import EncodeFile, get_signal

# HDF5 File Layout:
# Groups:
# bins/: Contains datasets for bin information (chrom, start, end).
#     /chrom
#     /start
#     /end
#     /good_bin
#     /bin_name
# embeddings/: Contains datasets for PCA, UMAP, and other embeddings.
#     /PCA1, /PCA2, /PCA3, ...
#     /umap1_n500, /umap2_n500, ...
#     /leiden1, /leiden2, ...
#     /kmeans1, /kmeans2, ...
# metadata/: Contains metadata for experiments, tracks, and other annotations.
#     /experiment
#     /track_metadata
# tracks/: Contains datasets for tracks from bigwig files.
#     /ENCFF0001, /ENCFF0002, ...


# Lock
lock = threading.Lock()

# Set up logging
logger = logging.getLogger("joint_pca")


def create_hdf5(output_file: str) -> None:
    """Create an empty HDF5 file with the required structure.

    Args:
    ----
        output_file (str): Path to the output HDF5 file.

    """
    with h5py.File(output_file, "w") as hdf:
        hdf.create_group("bins")
        hdf.create_group("embeddings")
        hdf.create_group("metadata")
        hdf.create_group("tracks")
    logger.info(f"HDF5 file created at {output_file}")


def load_bins_from_embeddings(embeddings_file: str) -> pd.DataFrame:
    """Load bin information from the embeddings file.

    Args:
    ----
        embeddings_file (str): Path to the parquet file containing embeddings.
        accession_column (str): Column name used as the accession identifier.

    Returns:
    -------
        pd.DataFrame: DataFrame containing bin information.

    """
    df = pd.read_parquet(embeddings_file)
    bins = df[["chrom", "start", "end", "good_bin"]].drop_duplicates().sort_values(["chrom", "start", "end"])
    return bins


def store_bins(hdf5_file: str, parquet_file: str) -> None:
    """Store bin information in the HDF5 file.

    Args:
    ----
        hdf5_file (str): Path to the HDF5 file.
        parquet_file (str): Path to the parquet file containing embeddings.
        accession_column (str): Column name used as the accession identifier.

    """
    df = load_bins_from_embeddings(parquet_file)

    with h5py.File(hdf5_file, "a") as hdf:
        bins_group = hdf.require_group("bins")
        bins_group.create_dataset("chrom", data=np.array(df["chrom"], dtype="S"))
        bins_group.create_dataset("start", data=df["start"].values.astype(np.int64))
        bins_group.create_dataset("end", data=df["end"].values.astype(np.int64))
        bins_group.create_dataset("good_bin", data=df["good_bin"].values.astype(np.bool_))
        bin_names = [f"{chrom}:{start}-{end}" for chrom, start, end in zip(df["chrom"], df["start"], df["end"])]
        bins_group.create_dataset("bin_name", data=np.array(bin_names, dtype="S"))
    logger.info("Bin information stored successfully.")


def load_bins(hdf5_file: str) -> pd.DataFrame:
    """Load bin information from the HDF5 file.

    Args:
    ----
        hdf5_file (str): Path to the HDF5 file.

    Returns:
    -------
        pd.DataFrame: DataFrame containing bin information.

    """
    with h5py.File(hdf5_file, "r") as hdf:
        bins = hdf["bins"]
        bin_df = pd.DataFrame(
            {
                "chrom": [chrom.decode("utf-8") for chrom in bins["chrom"]],
                "start": bins["start"][:],
                "end": bins["end"][:],
                "good_bin": bins["good_bin"][:],
                "bin_name": [name.decode("utf-8") for name in bins["bin_name"]],
            }
        )

    return bin_df


def store_experiment_metadata(hdf5_file: str, yaml_file: str) -> None:
    """Store experiment metadata in the HDF5 file.

    Args:
    ----
        hdf5_file (str): Path to the HDF5 file.
        yaml_file (str): Path to the YAML file containing experiment metadata.

    """
    with open(yaml_file) as file:
        metadata_list = yaml.safe_load(file)

    metadata_df = pd.DataFrame(metadata_list)

    # Convert all columns to string to avoid object dtype issues
    metadata_df = metadata_df.map(str)

    # Convert DataFrame to records
    metadata_records = metadata_df.to_records(index=False)

    # Define dtype explicitly to avoid issues with object dtype
    dtype = np.dtype([(col, h5py.special_dtype(vlen=str)) for col in metadata_df.columns])

    with h5py.File(hdf5_file, "a") as hdf:
        metadata_group = hdf.require_group("metadata")
        if "experiment_metadata" in metadata_group:
            del metadata_group["experiment_metadata"]
        metadata_group.create_dataset("experiment_metadata", data=metadata_records, dtype=dtype)
    logger.info("Experiment metadata stored successfully.")


def load_experiment_metadata(hdf5_file: str) -> pd.DataFrame:
    """Load experiment metadata from the HDF5 file.

    Args:
    ----
        hdf5_file (str): Path to the HDF5 file.

    Returns:
    -------
        pd.DataFrame: DataFrame containing experiment metadata.

    """
    with h5py.File(hdf5_file, "r") as hdf:
        metadata_group = hdf["metadata"]
        metadata = metadata_group["experiment_metadata"]

        # Create a DataFrame and decode byte arrays to strings
        metadata_df = pd.DataFrame(metadata[:])
        metadata_df = metadata_df.map(lambda x: x.decode("utf-8"))

    return metadata_df


def store_embeddings(hdf5_file: str, parquet_file: str, accession_column: str) -> None:
    """Store embeddings in the HDF5 file.

    Args:
    ----
        hdf5_file (str): Path to the HDF5 file.
        parquet_file (str): Path to the parquet file containing embeddings.
        accession_column (str): Column name used as the accession identifier.

    """
    df = pd.read_parquet(parquet_file)

    # Load bins
    bins_df = load_bins(hdf5_file)

    # Load experiment metadata to determine the order of experiments
    experiment_metadata = load_experiment_metadata(hdf5_file)
    accessions = experiment_metadata["accession"]

    # Get columns with embeddings
    embedding_columns = [
        c for c in df.columns if "PCA" in c or "NMF" in c or "SVD" in c or "umap" in c or "leiden" in c or "kmeans" in c
    ]

    # Store the embeddings in the HDF5 file
    with h5py.File(hdf5_file, "a") as hdf:
        embeddings_group = hdf.require_group("embeddings")
        for i, embedding in enumerate(embedding_columns):
            logger.info(f"[{i + 1}/{len(embedding_columns)}] Storing embeddings for {embedding}.")
            # Create an empty matrix with rows as bins and columns as experiments
            num_bins = bins_df.shape[0]
            num_experiments = len(accessions)
            embedding_matrix = np.full((num_bins, num_experiments), np.nan, dtype=np.float64)

            # Fill the matrix according to the order in the metadata table
            for i, accession in enumerate(accessions):
                group = df.loc[df[accession_column] == accession]
                # Add `bin_name` to the group
                group.loc[:, "bin_name"] = [
                    f"{chrom}:{start}-{end}" for chrom, start, end in zip(group["chrom"], group["start"], group["end"])
                ]
                # Reorder embedding rows based on bins_df using pandas indexes
                embedding_matrix[:, i] = group.set_index("bin_name").loc[bins_df["bin_name"], embedding].values

            # Store the matrix in the HDF5 file
            embeddings_group.create_dataset(embedding, data=embedding_matrix)


def load_embeddings(
    hdf5_file: str,
    embeddings: list[str],
    experiment_metadata: pd.DataFrame,
    bins: pd.DataFrame,
    drop_bad_bins: bool = False,
) -> pd.DataFrame:
    """Load and pivot embeddings into a `tall` DataFrame, horizontally merged with bins and experiment metadata.

    Args:
    ----
        hdf5_file (str): Path to the HDF5 file.
        embeddings (List[str]): List of embeddings to load.
        experiment_metadata (pd.DataFrame): DataFrame containing experiment metadata.
        bins (pd.DataFrame): DataFrame containing bin information.
        drop_bad_bins (bool): Whether to drop bad bins.

    Returns:
    -------
        pd.DataFrame: DataFrame containing combined embeddings, bins, and experiment metadata.

    """
    with h5py.File(hdf5_file, "r") as hdf:
        embeddings_group = hdf["embeddings"]

        # List to store data frames
        dfs = []

        for embedding in embeddings:
            if embedding in embeddings_group:
                # Load the embedding matrix
                embedding_matrix = embeddings_group[embedding][()]

                # Ensure that the number of columns matches the number of experiments
                if embedding_matrix.shape[1] != len(experiment_metadata):
                    raise ValueError("Mismatch between embedding and experiment metadata")

                # Ensure the bins match the number of rows
                if embedding_matrix.shape[0] != bins.shape[0]:
                    raise ValueError("Mismatch between embedding and bin information")

                # Create a DataFrame for the current embedding
                embedding_df = pd.DataFrame(embedding_matrix, columns=experiment_metadata["accession"])

                # Add bin information by horizontally concatenating the DataFrames, ignoring the index
                embedding_df = pd.concat([bins, embedding_df], axis=1)

                # Melt the DataFrame to a tall format
                embedding_df = embedding_df.melt(
                    id_vars=["bin_name", "chrom", "start", "end", "good_bin"],
                    var_name="accession",
                    value_name=embedding,
                ).reset_index()

                # Store the DataFrame in the list
                dfs.append(embedding_df)

    # Combine dataframes
    combined_df = dfs[0]
    for i, embedding in enumerate(embeddings):
        combined_df[embedding] = dfs[i][embedding]

    # Drop bad bins if required
    if drop_bad_bins:
        combined_df = combined_df[combined_df["good_bin"]]

    return combined_df.sort_values(["accession", "chrom", "start"])


def add_bigwig_track(
    hdf5_file: str, signal: pd.DataFrame, accession: str, assay: str, experiment: str, biosample: str
) -> None:
    """Add a bigwig track to the HDF5 file.

    Args:
    ----
        hdf5_file (str): Path to the HDF5 file.
        signal (pd.DataFrame): DataFrame containing the signal.
        accession (str): The accession identifier for the track.
        assay (str): The assay name.
        experiment (str): The experiment name.
        biosample (str): The biosample name.

    """
    with h5py.File(hdf5_file, "a") as hdf:
        # Add track data
        tracks_group = hdf.require_group("tracks")
        tracks_group.create_dataset(accession, data=np.array(signal, dtype=np.float64))

        # Add track metadata
        metadata_group = hdf.require_group("metadata/track_metadata")
        track_group = metadata_group.require_group(accession)
        track_group.attrs["assay"] = assay
        track_group.attrs["experiment"] = experiment
        track_group.attrs["biosample"] = biosample


def add_encode_tracks_to_hdf5(track_metadata_df: pd.DataFrame, hdf5_file: str) -> None:
    """Add tracks and their metadata to the HDF5 file.

    Args:
    ----
        track_metadata_df (pd.DataFrame): DataFrame containing track metadata.
        hdf5_file (str): Path to the HDF5 file.

    """
    number_of_tracks = track_metadata_df.shape[0]
    logger.info(f"Adding {number_of_tracks} tracks to the HDF5 file {hdf5_file}")

    # Load bin information from the HDF5 file
    bins_df = load_bins(hdf5_file)

    def process_entry(entry):
        """Process a single track entry."""
        acc = entry["accession"]
        logger.info(f"Loading signal for track: {acc}")
        with EncodeFile(acc, "bw") as bigwig:
            bw_signal = get_signal(bigwig, bins_df)
            return bw_signal, acc, entry["assay"], entry["experiment"], entry["biosample"]

    # Process each track entry in parallel
    with ThreadPoolExecutor(8) as executor:
        futures = [executor.submit(process_entry, entry) for _, entry in track_metadata_df.iterrows()]

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                signal, accession, assay, experiment, biosample = future.result()
                logger.info(f"[{i + 1}/{number_of_tracks}] Adding track {accession} to HDF5 file.")
                with lock:
                    add_bigwig_track(hdf5_file, signal, accession, assay, experiment, biosample)
            except Exception as e:
                logger.error(f"Error processing track: {e}")

    logger.info("All tracks and metadata added successfully.")


def load_track_metadata(hdf5_file: str) -> pd.DataFrame:
    """Load track metadata from the HDF5 file.

    Args:
    ----
        hdf5_file (str): Path to the HDF5 file.

    Returns:
    -------
        pd.DataFrame: DataFrame containing track metadata.

    """
    with h5py.File(hdf5_file, "r") as hdf:
        metadata_group = hdf["metadata/track_metadata"]
        track_metadata = []
        for accession in metadata_group:
            track_group = metadata_group[accession]
            track_metadata.append(
                {
                    "accession": accession,
                    "assay": track_group.attrs["assay"],
                    "experiment": track_group.attrs["experiment"],
                    "biosample": track_group.attrs["biosample"],
                }
            )
    return pd.DataFrame(track_metadata)


def load_tracks(hdf5_file: str, accessions: list[str], bins: pd.DataFrame, drop_bad_bins: bool = False) -> pd.DataFrame:
    """Load track signal from the HDF5 file.

    Args:
    ----
        hdf5_file (str): Path to the HDF5 file.
        accessions (List[str]): The accession identifiers for the tracks.
        bins (pd.DataFrame): DataFrame containing bin information.
        drop_bad_bins (bool): Whether to drop bad bins.

    Returns:
    -------
        pd.DataFrame: DataFrame containing the track signal.

    """
    tracks = bins.copy()

    with h5py.File(hdf5_file, "r") as hdf:
        tracks_group = hdf["tracks"]
        for i, accession in enumerate(accessions):
            signal = tracks_group[accession][()]
            tracks[accession] = signal
            if i % 5 == 0:
                tracks = tracks.copy()

        # Drop bad bins if required
        if drop_bad_bins:
            tracks = tracks[tracks["good_bin"]]

        return tracks


class JointDb:
    """Class to query the HDF5 database."""

    def __init__(self, hdf5_file: str):
        """Initialize the JointDb object."""
        self.hdf5_file = hdf5_file
        self._bins = None
        self._experiment_metadata = None
        self._track_metadata = None

    @property
    def bins(self) -> pd.DataFrame:
        """Load bin information from the HDF5 file.

        Returns
        -------
            pd.DataFrame: DataFrame containing bin information.

        """
        if self._bins is None:
            self._bins = load_bins(self.hdf5_file)
        return self._bins

    @property
    def experiment_metadata(self) -> pd.DataFrame:
        """Load experiment metadata from the HDF5 file.

        Returns
        -------
            pd.DataFrame: DataFrame containing experiment metadata.

        """
        if self._experiment_metadata is None:
            self._experiment_metadata = load_experiment_metadata(self.hdf5_file)
        return self._experiment_metadata

    @property
    def track_metadata(self) -> pd.DataFrame:
        """Load track metadata from the HDF5 file.

        Returns
        -------
            pd.DataFrame: DataFrame containing track metadata.

        """
        if self._track_metadata is None:
            self._track_metadata = load_track_metadata(self.hdf5_file)
        return self._track_metadata

    def get_embeddings(
        self, embeddings: list[str], accessions: list[str] | None = None, drop_bad_bins: bool = False
    ) -> pd.DataFrame:
        """Load embeddings from the HDF5 file.

        Args:
        ----
            embeddings (List[str]): List of embeddings to load.
            accessions (List[str]): The accession identifiers for the embeddings.
            drop_bad_bins (bool): Whether to drop bad bins.

        Returns:
        -------
            pd.DataFrame: DataFrame containing embeddings.

        """
        embeddings = load_embeddings(self.hdf5_file, embeddings, self.experiment_metadata, self.bins, drop_bad_bins)

        # Filter embeddings by accessions if required
        if accessions is not None:
            embeddings = embeddings[embeddings["accession"].isin(accessions)]

        return embeddings

    def get_tracks(self, accessions: list[str], drop_bad_bins: bool = False) -> pd.DataFrame:
        """Load track signal from the HDF5 file.

        Args:
        ----
            accessions (List[str]): The accession identifiers for the tracks.
            drop_bad_bins (bool): Whether to drop bad bins.

        Returns:
        -------
            pd.DataFrame: DataFrame containing the track signal.

        """
        return load_tracks(self.hdf5_file, accessions, self.bins, drop_bad_bins)

    def get_track_overlay(
        self,
        assay: str,
        embeddings: list[str],
        accessions: list[str] | None = None,
        drop_bad_bins: bool = False,
        scaler: Callable | None = None,
        biosample_mapper: Callable | None = None,
    ) -> pd.DataFrame:
        """Load embeddings & merge tracks on biosample, for a specific assay.

        Args:
        ----
            assay (str): The assay type.
            embeddings (List[str]): List of embeddings to load.
            accessions (List[str]): The embedding accessions to load.
            drop_bad_bins (bool): Whether to drop bad bins.
            scaler (Callable): A function to scale the signal.
            biosample_mapper (Callable): A function to map biosample names.

        Returns:
        -------
            pd.DataFrame: DataFrame containing embeddings, bins, and experiment metadata.

        """
        # Load embeddings
        embeddings_df = self.get_embeddings(embeddings, accessions=accessions, drop_bad_bins=drop_bad_bins)
        embeddings_df = pd.merge(self.experiment_metadata, embeddings_df, on="accession", how="inner")
        embeddings_df = embeddings_df[["bin_name", "biosample", "accession", *embeddings]]

        # If a biosample mapper is provided, apply it
        if biosample_mapper is not None:
            embeddings_df["biosample"] = embeddings_df["biosample"].map(biosample_mapper)
            # Example biosample mapper: lambda x: x.split("-")[0]

        # Load tracks for the given assay
        assay_tracks = self.track_metadata[self.track_metadata["assay"] == assay]
        track_df = self.get_tracks(assay_tracks["accession"].tolist(), drop_bad_bins)

        # Melt track_df and merge with track_metadata
        track_df = track_df.melt(
            id_vars=["bin_name", "chrom", "start", "end", "good_bin"], var_name="bw_accession", value_name="signal"
        ).reset_index(drop=True)
        track_df = pd.merge(track_df, self.track_metadata, left_on="bw_accession", right_on="accession", how="inner")

        # Scale the signal if a scaler is provided
        if scaler is not None:
            track_df = scaler(track_df)

        track_df = track_df[["bin_name", "biosample", "signal"]]

        # Merge with embeddings, then groupby and average signal so each sample-bin has a single signal
        result_df = (
            embeddings_df.merge(track_df, on=["bin_name", "biosample"], how="inner")
            .groupby(["bin_name", "biosample", "accession"])
            .mean()
            .reset_index()
        )
        return result_df

    def __repr__(self):
        """Return a string representation of the JointDb object."""
        return (
            f"JointDb(file='{self.hdf5_file}', bins={self.bins.shape[0]}, "
            f"experiments={self.experiment_metadata.shape[0]}, tracks={self.track_metadata.shape[0]})"
        )


def main(
    experiments_file: str, tracks_file: str | None, embeddings_file: str, accession_column: str, output_file: str
) -> None:
    """Create the HDF5 file with bins, metadata, embeddings, and tracks.

    Args:
    ----
        experiments_file (str): Path to the YAML file containing experiment metadata.
        tracks_file (str): Path to the YAML file containing track metadata.
        embeddings_file (str): Path to the parquet file containing embeddings.
        accession_column (str): Column name used as the accession identifier.
        output_file (str): Path to the output HDF5 file.

    """
    # Step 1: Create the HDF5 file
    logger.info(f"Creating HDF5 file at {output_file}")
    create_hdf5(output_file)

    # Step 2: Load and store bins (assuming bins are part of the embeddings file)
    logger.info("Loading and storing bins...")
    store_bins(output_file, embeddings_file)

    # Step 3: Load and store experiment metadata
    logger.info("Loading and storing experiment metadata...")
    store_experiment_metadata(output_file, experiments_file)

    # Step 4: Load and store embeddings
    logger.info("Loading and storing embeddings...")
    store_embeddings(output_file, embeddings_file, accession_column)

    # Step 5: Load and store track metadata and tracks
    if tracks_file is not None:
        logger.info("Loading and storing track metadata and tracks...")
        with open(tracks_file) as file:
            track_metadata_list = yaml.safe_load(file)

        track_metadata_df = pd.DataFrame(track_metadata_list)
        add_encode_tracks_to_hdf5(track_metadata_df, output_file)

    logger.info(f"HDF5 file '{output_file}' created successfully with all data.")


def run_main(arguments):
    """Run the main function with arguments."""
    main(arguments.experiments, arguments.tracks, arguments.embeddings, arguments.accession, arguments.output)


def configure_hdf5db_parser(parser):
    """Configure the HDF5 database parser."""
    parser.add_argument("--experiments", required=True, help="Path to the YAML file containing experiment metadata.")
    parser.add_argument("--tracks", required=False, help="Path to the YAML file containing track metadata.")
    parser.add_argument("--embeddings", required=True, help="Path to the parquet file containing embeddings.")
    parser.add_argument("--accession", required=True, help="Column name used as the accession identifier.")
    parser.add_argument("--output", required=True, help="Path to the output HDF5 file.")
    parser.set_defaults(func=run_main)


if __name__ == "__main__":
    # Define the CLI arguments
    local_parser = argparse.ArgumentParser(
        description="Create an HDF5 file with bins, metadata, embeddings, and tracks."
    )
    configure_hdf5db_parser(local_parser)

    # Parse the arguments
    args = local_parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.experiments, args.tracks, args.embeddings, args.accession, args.output)
