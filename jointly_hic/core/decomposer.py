"""Main command class for Joint PCA Hi-C."""

import gzip
import logging
import os
import pickle
from datetime import datetime

import bioframe
import numpy as np
import pandas as pd
from cooler import Cooler
from pandas import DataFrame
from sklearn.decomposition import IncrementalPCA, MiniBatchNMF

from jointly_hic.core.config import JointlyConfiguration, PostProcessingConfig, TrajectoryAnalysisConfig
from jointly_hic.methods.multivec import make_multivec, nanmean_agg
from jointly_hic.methods.sparse_incremental_svd import SparseIncrementalSVD
from jointly_hic.methods.trans_cooler_utils import normalized_affinity_matrix_from_trans

logger = logging.getLogger("joint_pca")


class JointlyDecomposer:
    """Main command class for Joint PCA Hi-C.

    JointlyDecomposer performs incremental PCA or NMF docomposition of multiple input 3D chromatin contact frequency
    matrices from mcool files. JointlyDecomposer can be run from the command line or imported as a module. It takes a
    JointlyConfig object as input and returns a tuple of the output embeddings and a PostProcessingConfig object,
    for use configuring the post-processing step. Use JointlyDecomposer.run() to run the decomposition.

    The fields in the configuration are:
        mcools: List of input mcool files
        output: Output file prefix
        resolution: Resolution of input mcool files
        assembly: Genome assembly of input mcool files
        components: Number of components to compute
        chrom_limit: Number of chromosomes to use
        method: Method to use for decomposition. Either 'PCA' or 'NMF'
        percentile_top: Top percentile of contact frequencies to threshold at
        percentile_bottom: Bottom percentile of contact frequencies to remove
        batch_size: Batch size for incremental decomposition
        log_level: Log level for logging

    Returns
    -------
        Tuple of output embeddings and PostProcessingConfig object

    """

    def __init__(self, configuration: JointlyConfiguration):
        """Initialize JointlyDecomposer."""
        self.configuration = configuration
        self.union_bad_bins: np.ndarray | None = None
        self._partition = None
        logger.info("JointlyDecomposer :: configuration: \n\n%s", configuration)

        # Initialize model
        if configuration.method == "PCA":
            self.model = IncrementalPCA(n_components=configuration.components)
        elif configuration.method == "NMF":
            self.model = MiniBatchNMF(n_components=configuration.components)
        elif configuration.method == "SVD":
            self.model = SparseIncrementalSVD(n_components=configuration.components)
        else:
            raise ValueError(f"Unknown method: {configuration.method}")

        # Get chromosome sizes
        self.chromosome_sizes = self.get_chromosome_sizes()

    def run(self) -> tuple[DataFrame, PostProcessingConfig, TrajectoryAnalysisConfig]:
        """Run joint PCA."""
        # Get start time
        start_time = datetime.now()

        # Get union set of bad bins
        self.set_union_bad_bins()

        # Iterate through files and compute low dimension embeddings
        for filename in self.configuration.mcools:
            # Decompose cooler file
            self.decompose_cooler_file(
                filename=filename,
            )
        logger.info("Model training complete. Training time: %s.", str(datetime.now() - start_time))

        # Compute embeddings using fully trained model
        logger.info("Computing embeddings using fully trained model...")
        output_embeddings = self.compute_output_embeddings().reset_index(drop=True)

        # Output file names
        output_prefix = (
            f"{self.configuration.output}_{self.configuration.method}-{self.configuration.components}_"
            f"{self.configuration.resolution}bp_{self.configuration.assembly}"
        )
        parquet_filename = f"{output_prefix}_embeddings.pq"
        csv_filename = f"{output_prefix}_embeddings.csv.gz"
        model_filename = f"{output_prefix}_model.pkl.gz"

        # Save results
        logger.info("Saving results...")
        self.save_model(model_filename)
        output_embeddings.to_csv(csv_filename, index=False)
        output_embeddings.to_parquet(parquet_filename, index=False)
        logger.debug(f"Output embeddings for filenames:\n{output_embeddings.filename.value_counts()}")
        logger.info("Saved embeddings to '%s' and '%s'.", csv_filename, parquet_filename)

        # Create configuration for post-processing
        post_config = PostProcessingConfig(
            parquet_file=parquet_filename,
            output=output_prefix,
            method=self.configuration.method,
            log_level=self.configuration.log_level,
        )
        trajectory_config = TrajectoryAnalysisConfig(
            parquet_file=parquet_filename,
            output=output_prefix,
            method=self.configuration.method,
            log_level=self.configuration.log_level,
        )

        # Log total time
        logger.info("Finished joint PCA in %s.", str(datetime.now() - start_time))
        return output_embeddings.copy(), post_config, trajectory_config

    @property
    def partition(self) -> np.ndarray:
        """Get partition numbers for each chromosome."""
        if self._partition is not None:
            return self._partition

        partition = None
        for cooler_file in self.configuration.mcools:
            clr = Cooler(cooler_file + f"::/resolutions/{self.configuration.resolution}")
            partition_sizes = [clr.offset(chrom) for chrom in self.chromosome_sizes.index]
            if partition is None:
                partition = np.r_[
                    partition_sizes,
                    clr.extent(self.chromosome_sizes.index[-1])[1],
                ]
            else:
                if not np.all(partition == np.r_[partition_sizes, clr.extent(self.chromosome_sizes.index[-1])[1]]):
                    raise ValueError("Chromosome partitions do not match across input files")
            logger.debug("Loaded partitions: [%s]", ", ".join([str(x) for x in partition]))
            self._partition = partition
        return self._partition

    def get_chromosome_sizes(self) -> pd.Series:
        """Download chromosome sizes using bioframe."""
        chromosome_sizes = bioframe.fetch_chromsizes(self.configuration.assembly.lower())
        # chromosome_sizes is a pandas Series with chromosome names as index and chromosome sizes as values
        # Filter to only the first chrom_limit chromosomes
        chromosome_sizes = chromosome_sizes.iloc[: self.configuration.chrom_limit]
        logger.info("Loaded chromosome sizes for specified assembly: %s", self.configuration.assembly)
        logger.info("Chromosome sizes:\n%s", chromosome_sizes)
        return chromosome_sizes

    def set_union_bad_bins(self) -> None:
        """Compute union of bad bins (where weight is NaN) across all input files.

        Returns
        -------
            Array of bad bins (where weight is NaN) across all input files

        """
        logger.info("Beginning to compute union set of NaN bins..")
        start_time = datetime.now()
        bad_bins: np.ndarray | None = None
        bins: pd.DataFrame | None = None

        # Bins are bad when the weight is NaN
        for filename in self.configuration.mcools:
            clr = Cooler(f"{filename}::/resolutions/{self.configuration.resolution}")
            lo = self.partition[0]
            hi = self.partition[-1]
            bins = clr.bins()[lo:hi]
            logger.debug("Loaded bins from cooler file '%s' with shape '%s'.", filename, bins.shape)
            if bad_bins is None:
                bad_bins = np.isnan(bins["weight"].to_numpy())
            else:
                bad_bins = np.logical_or(bad_bins, np.isnan(bins["weight"].to_numpy()))
            logger.debug("Percent bad bins: %s.", 100 * np.sum(bad_bins) / bad_bins.shape[0])

        # Load excluded_bins from exclusion list if provided
        if self.configuration.exclusion_list is not None:
            logger.info("Loading exclusion list from: %s", self.configuration.exclusion_list)
            excluded_bins = pd.read_csv(
                self.configuration.exclusion_list, sep="\t", header=None, names=["chrom", "start", "end"]
            )
            # Exclude bins overlapping an excluded_bin
            overlap = bioframe.overlap(bins, excluded_bins, how="inner", return_index=True, return_input=False)
            ix = overlap["index"].to_numpy(dtype="int64")
            bad_bins[ix] = True
            logger.debug("Percent bad bins: %s.", 100 * np.sum(bad_bins) / bad_bins.shape[0])

        # Preprocess all input files and include the bad bins from normalization
        assert isinstance(bad_bins, np.ndarray)
        self.union_bad_bins = bad_bins
        for filename in self.configuration.mcools:
            self.preprocess_matrix(filename, ignore_cache=True)
            logger.debug("Percent bad bins: %s.", 100 * np.sum(bad_bins) / bad_bins.shape[0])

        logger.info("Loaded union set of bad bins in: %s.", str(datetime.now() - start_time))
        logger.info(
            "Percent of bins that are bad: %s.", 100 * np.sum(self.union_bad_bins) / self.union_bad_bins.shape[0]
        )

        # Save bins to file
        logger.debug("Shape of bad_bin array: %s.", self.union_bad_bins.shape)
        bins["bad_bin"] = self.union_bad_bins
        logger.debug("Shape of bins dataframe: %s.", bins.shape)
        logger.debug("Saving bins to file...")
        bins.to_csv(f"{self.configuration.output}_bins.csv.gz", index=False)

    def preprocess_matrix(self, filename: str, ignore_cache: bool = False) -> np.ndarray:
        """Load matrix from cooler and preprocess it.

        Loads matrix from cooler file, computes cis-masked, balanced, affinity matrix, and removes bad bins.
        The result is cached to disk for future use.

        Args:
        ----
            filename: Input .mcool filename containing contact frequency matrix
            ignore_cache: If True, ignore cached matrix and recompute

        Returns:
        -------
            np.ndarray: Preprocessed contact frequency matrix

        """
        # Load matrix from cooler file
        start_time = datetime.now()
        clr = Cooler(filename + f"::/resolutions/{self.configuration.resolution}")
        partition = self.partition
        lo = partition[0]
        hi = partition[-1]

        # Load from disk if file exists
        matrix_filename = f"{filename.removesuffix('.mcool')}_preprocessed_{lo}_{hi}.npy"
        if os.path.exists(matrix_filename) and not ignore_cache:
            logger.info("Loading preprocessed matrix from disk...")
            matrix = np.load(matrix_filename)
            logger.debug(
                "Loaded preprocessed matrix with shape '%s' from disk in %s.",
                matrix.shape,
                str(datetime.now() - start_time),
            )

            # Remove bad bins
            bin_logical = np.logical_not(self.union_bad_bins)
            matrix = matrix[bin_logical, :]
            matrix = matrix[:, bin_logical]
            logger.debug("Removed bad bins from matrix. New shape: %s", matrix.shape)
            return matrix

        # Load contact frequency matrix
        matrix = clr.matrix(balance=True)[lo:hi, lo:hi]
        logger.debug(
            "Loaded contact frequency matrix with shape '%s' in %s.", matrix.shape, str(datetime.now() - start_time)
        )

        # Compute cis-masked, balanced, affinity matrix
        start_time = datetime.now()
        # Adjust partition so that it starts at 0
        partition = partition - partition[0]
        matrix, normalization_bad_bins = normalized_affinity_matrix_from_trans(
            matrix,
            partition,
            self.configuration.percentile_top,
            self.configuration.percentile_bottom,
        )

        # Save matrix to disk for future use
        np.save(matrix_filename, matrix)

        # Update self.union_bad_bins with normalization_bad_bins
        self.union_bad_bins = np.logical_or(self.union_bad_bins, normalization_bad_bins)
        assert isinstance(self.union_bad_bins, np.ndarray)

        logger.debug(
            "Computed cis-masked, balanced, affinity matrix of shape '%s' in %s.",
            matrix.shape,
            str(datetime.now() - start_time),
        )

        # Remove bad bins
        bin_logical = np.logical_not(self.union_bad_bins)
        matrix = matrix[bin_logical, :]
        matrix = matrix[:, bin_logical]
        logger.debug("Removed bad bins from matrix. New shape: %s", matrix.shape)
        return matrix

    def minibatch_fit(self, matrix: np.ndarray):
        """Fit model using partial fit and minibatching input matrix.

        Args:
        ----
            matrix: Input matrix
            batch_size: Batch size

        """
        num_bins = matrix.shape[0]
        num_batches = int(np.ceil(num_bins / self.configuration.batch_size))
        batch_indices = np.array_split(np.arange(num_bins), num_batches)
        logger.debug("Split matrix into %s batches.", num_batches)

        # Iterate over batches
        for i, batch_index in enumerate(batch_indices):
            start_time = datetime.now()
            batch = matrix[batch_index, :]

            # Fit batch
            logger.debug("Fitting batch: %s with nrows: %s.", i + 1, batch.shape[0])
            self.model.partial_fit(batch)
            logger.debug("Finished fitting batch: %s in %s seconds.", i + 1, str(datetime.now() - start_time))

    def decompose_cooler_file(self, filename):
        """Decompose a single cooler file."""
        logger.info("Computing dimensionality reduction for input file: '%s'...", filename)
        start_time = datetime.now()
        matrix = self.preprocess_matrix(filename)
        logger.debug("Finished preprocessing matrix for '%s' in %s.", filename, str(datetime.now() - start_time))

        # Compute incremental fit
        self.minibatch_fit(
            matrix,
        )
        logger.info(
            "Finished decomposition for '%s' in %s.",
            filename,
            str(datetime.now() - start_time),
        )

    def save_model(self, filename):
        """Save model to disk using pickle."""
        start_time = datetime.now()
        with gzip.open(filename, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("Saved model to '%s' in %s.", filename, str(datetime.now() - start_time))

    def compute_output_embeddings(self) -> pd.DataFrame:
        """Iterate through files and compute embeddings."""
        output = []
        for filename in self.configuration.mcools:
            output.append(self.compute_output_embeddings_single_file(filename))
        return pd.concat(output)

    def compute_output_embeddings_single_file(self, filename) -> pd.DataFrame:
        """Compute embeddings for a single input using the trained model."""
        logger.info("Computing embeddings for input file: '%s'...", filename)
        lo = self.partition[0]
        hi = self.partition[-1]

        # Load matrix from cooler file
        matrix = self.preprocess_matrix(filename)

        # Compute embeddings
        embeddings = self.model.transform(matrix)

        # Move embeddings into array of original shape
        full_embeddings = np.zeros((hi - lo, self.configuration.components), dtype=np.float64)
        full_embeddings[:] = np.nan
        full_embeddings[np.logical_not(self.union_bad_bins)] = embeddings

        # Add bin names and convert to dataframe
        bins = Cooler(f"{filename}::/resolutions/{self.configuration.resolution}").bins()[lo:hi]
        result = pd.DataFrame(
            full_embeddings,
            columns=[f"{self.configuration.method}{i + 1}" for i in range(self.configuration.components)],
            dtype=np.float64,
        )
        bins = bins.reset_index(drop=True)
        result = result.reset_index(drop=True)
        good_bin_df = pd.DataFrame({"good_bin": np.logical_not(self.union_bad_bins)}).reset_index(drop=True)
        filename_df = pd.DataFrame({"filename": [filename] * result.shape[0]}).reset_index(drop=True)
        result = pd.concat(
            [
                bins,
                good_bin_df,
                filename_df,
                result,
            ],
            axis=1,
        )

        # Make multivec output
        make_multivec(
            outpath=f"{filename.removesuffix('.mcool')}_{self.configuration.method}.mv5",
            df=result,
            feature_names=[f"{self.configuration.method}{i + 1}" for i in range(min(self.configuration.components, 8))],
            base_res=self.configuration.resolution,
            chromsizes=self.chromosome_sizes.to_dict(),
            tilesize=1024,
            agg=nanmean_agg,
            chunksize=int(1e6),
        )
        return result
