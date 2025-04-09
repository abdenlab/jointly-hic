"""Class to handle post-processing of embeddings."""

import logging

import datashader as ds
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
import umap.plot
from datashader.mpl_ext import dsshow
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from jointly_hic.core.config import PostProcessingConfig

logger = logging.getLogger("joint_pca")
MAX_PLOT_CLUSTERS = 20


class PostProcessor:
    """Class to handle post-processing of embeddings.

    PostProcessor is the main interface to handle basic Jointly-HiC post-processing. It takes a PostProcessingConfig
    and a pandas DataFrame of embeddings as input. It then runs KMeans clustering and UMAP on the embeddings and
    saves the results to parquet and csv files. It also creates basic plots of the embeddings.

    Configure the PostProcessor with a PostProcessingConfig.
    """

    def __init__(self, configuration: PostProcessingConfig, embeddings: pd.DataFrame | None = None):
        """Initialize PostProcessor."""
        self.configuration: PostProcessingConfig = configuration
        logger.info("PostProcessor :: configuration: \n\n%s", configuration)
        self.cluster_columns: list[str] = []
        if embeddings is None:
            logger.info("Loading embeddings from parquet file: %s", self.configuration.parquet_file)
            loaded_embeddings = pd.read_parquet(self.configuration.parquet_file)
        else:
            loaded_embeddings = embeddings.copy()

        # Ensure types
        loaded_embeddings["chrom"] = loaded_embeddings["chrom"].astype(str)
        loaded_embeddings["start"] = loaded_embeddings["start"].astype(int)
        loaded_embeddings["end"] = loaded_embeddings["end"].astype(int)
        self.embeddings: pd.DataFrame = loaded_embeddings

    @property
    def components(self) -> np.ndarray:
        """Return columns that start with 'PC#' and good_bin is True as numpy array."""
        try:
            return self.embeddings.loc[
                self.embeddings.good_bin,
                [col for col in self.embeddings.columns if col.startswith(self.configuration.method)],
            ].values
        except KeyError:
            logger.warning("No columns starting with %s found. Trying: PCA, NMF, SVD.", self.configuration.method)
            try:
                return self.embeddings.loc[
                    self.embeddings.good_bin, [col for col in self.embeddings.columns if col.startswith("PCA")]
                ].values
            except KeyError:
                logger.warning("No columns starting with PCA found. Trying: NMF, SVD.")
            try:
                return self.embeddings.loc[
                    self.embeddings.good_bin, [col for col in self.embeddings.columns if col.startswith("NMF")]
                ].values
            except KeyError:
                logger.warning("No columns starting with NMF found. Trying: SVD.")
            try:
                return self.embeddings.loc[
                    self.embeddings.good_bin, [col for col in self.embeddings.columns if col.startswith("SVD")]
                ].values
            except KeyError:
                logger.warning("No columns starting with SVD found. Trying: PC.")

        raise Exception("No columns starting with PCA, NMF, or SVD found.")

    def normalize_embeddings(self):
        """Normalize embeddings by dataset then rescale to overall norm."""
        # Compute overall norm
        norm = np.linalg.norm(self.components)

        # Group by 'filename' and rescale by group norm then overall norm
        for i, df in self.embeddings.groupby("filename"):
            logger.info(f"Normalizing embeddings for {i}")
            component_values = df.loc[
                df.good_bin, [col for col in df.columns if col.startswith(self.configuration.method)]
            ].values
            scaled_components = component_values / np.linalg.norm(component_values) * norm
            df.loc[df.good_bin, [col for col in df.columns if col.startswith(self.configuration.method)]] = (
                scaled_components
            )
            self.embeddings.update(df)
            logger.debug("Count of filenames: \n%s", self.embeddings.filename.value_counts())

    def add_column(self, column_name, values):
        """Add column to embeddings where good_bin is True and otherwise add NaN."""
        self.embeddings[column_name] = np.nan
        self.embeddings.loc[self.embeddings.good_bin, column_name] = values

    def run(self):
        """Run post-processing."""
        logger.info("Running post-processing")
        self.normalize_embeddings()

        # Set bin_name as chrom_start_end and compute bin distance variance
        self.embeddings["bin_name"] = (
            self.embeddings["chrom"]
            + "_"
            + self.embeddings["start"].astype(str)
            + "_"
            + self.embeddings["end"].astype(str)
        )
        self.embeddings["bin_variance"] = self.embeddings["bin_name"].map(self.compute_bin_distance_variance())

        components = self.components
        logger.info("Components shape: %s", components.shape)

        # Run kmeans clustering
        for n_clusters in self.configuration.kmeans_clusters:
            self.run_kmeans(components, n_clusters=n_clusters)

        # Run Leiden clustering
        for resolution in self.configuration.leiden_resolutions:
            self.run_leiden(components, resolution=resolution, n_neighbors=500)

        # Run UMAP and plot (requires clustering to be run first)
        for n_neighbors in self.configuration.umap_neighbours:
            self.run_umap(components, n_neighbors=n_neighbors)

        # Plot scores with clusters
        self.plot_scores()

        # Save embeddings to parquet and csv
        logger.info("Saving embeddings to parquet and csv")
        self.embeddings.to_csv(f"{self.configuration.output}_post_processed.csv.gz", index=False)
        self.embeddings.to_parquet(f"{self.configuration.output}_post_processed.pq", index=False)

        logger.info("Finished running post-processing")

    def plot_scores(self):
        """Plot the scores."""
        logger.info("Plotting scores")

        # Create plotdf from columns we want
        components = [f"{self.configuration.method}{i}" for i in range(1, 5)]
        plotdf = self.embeddings.loc[self.embeddings.good_bin, [*components, "good_bin", "filename"]]
        for cluster_column in self.cluster_columns:
            plotdf[cluster_column] = self.embeddings.loc[self.embeddings.good_bin, cluster_column].astype("category")

        # Truncate PC values at 1% and 99% percentile
        for pc in components:
            plotdf[pc] = plotdf[pc].clip(lower=plotdf[pc].quantile(0.005), upper=plotdf[pc].quantile(0.995))

        # Plot every pair of PCs using dsshow and a grid
        _, axs = plt.subplots(4, 4, figsize=(12, 12))
        for i, pc1 in enumerate(components):
            for j, pc2 in enumerate(components):
                if pc1 != pc2:
                    # Plot PC scores for pairs of PCs
                    dsshow(
                        plotdf,
                        ds.Point(pc1, pc2),
                        aspect="auto",
                        ax=axs[i, j],
                    )
                    axs[i, j].set_title(f"{pc1} vs {pc2}")
                if pc1 == pc2:
                    # Plot histogram of PC
                    axs[i, j].hist(plotdf[pc1], bins=100)
                    axs[i, j].set_title(pc1)
        plt.savefig(f"{self.configuration.output}_scores.png")
        plt.close()

        # Plot colored by cluster on a grid
        n_clusters = len(self.cluster_columns)
        grid_size = int(np.ceil(np.sqrt(n_clusters)))
        xmin = plotdf[components[0]].min()
        xmax = plotdf[components[0]].max()
        ymin = plotdf[components[1]].min()
        ymax = plotdf[components[1]].max()
        _, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        for i, cluster_column in enumerate(self.cluster_columns):
            # Plot if there are <= MAX_PLOT_CLUSTERS clusters
            if len(plotdf[cluster_column].unique()) <= MAX_PLOT_CLUSTERS:
                dsshow(
                    plotdf,
                    ds.Point(components[0], components[1]),
                    aggregator=ds.count_cat(cluster_column),
                    aspect="auto",
                    ax=axs[i // grid_size, i % grid_size],
                    x_range=(xmin, xmax),
                    y_range=(ymin, ymax),
                )
                axs[i // grid_size, i % grid_size].set_title(cluster_column)
        plt.savefig(f"{self.configuration.output}_scores_clustered.png")
        plt.close()

        # Plot PC1 vs PC2 colored by cluster for every filename
        n_files = len(plotdf.filename.unique())
        grid_size = int(np.ceil(np.sqrt(n_files)))
        _, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        for i, filename in enumerate(plotdf.filename.unique()):
            dsshow(
                plotdf[plotdf.filename == filename],
                ds.Point(components[0], components[1]),
                aspect="auto",
                ax=axs[i // grid_size, i % grid_size],
                x_range=(xmin, xmax),
                y_range=(ymin, ymax),
            )
            axs[i // grid_size, i % grid_size].set_title(filename)
        plt.savefig(f"{self.configuration.output}_scores_filenames.png")
        plt.close()

    def run_kmeans(self, components: np.ndarray, n_clusters: int = 8) -> None:
        """Run KMeans clustering on embeddings."""
        logger.info("Running KMeans clustering with %d clusters", n_clusters)
        # Initialize and fit the K-Means model
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=1000, random_state=42)
        kmeans.fit(components)
        column_name = f"kmeans_{n_clusters}"
        self.add_column(column_name, kmeans.labels_)
        self.cluster_columns.append(column_name)

    def run_leiden(
        self, components: np.ndarray, n_neighbors: int = 100, resolution: float = 1.0, preprocess: bool = False
    ) -> None:
        """Perform Leiden clustering on an array of embeddings.

        Arguments:
        ---------
        components: np.ndarray
            Array of embeddings
        n_neighbors: int
            Number of neighbors to use for KNN graph
        resolution: float
            Resolution parameter for Leiden clustering
        preprocess: bool
            Whether to preprocess the data with StandardScaler

        """
        logger.info("Running Leiden clustering with %d neighbors.", n_neighbors)
        # Preprocessing: Standardize features by removing the mean and scaling to unit variance
        if preprocess:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(components)
        else:
            data_scaled = components

        # Construct a k-nearest neighbors graph
        knn_graph = kneighbors_graph(data_scaled, n_neighbors=n_neighbors, mode="connectivity", include_self=True)

        # Convert to igraph
        sources, targets = knn_graph.nonzero()

        # Ensure the knn_graph is in a compatible sparse format
        if not isinstance(knn_graph, np.ndarray | np.generic):
            knn_graph = knn_graph.tocsr()

        # Extract weights directly from the sparse matrix
        weights = knn_graph[sources, targets].data

        # Flatten the weights to a 1D list if they are not already
        weights = np.array(weights).flatten().astype(float)

        # Create igraph Graph
        g = ig.Graph(
            n=data_scaled.shape[0],
            edges=list(zip(sources.tolist(), targets.tolist())),
            edge_attrs={"weight": weights.tolist()},
            directed=False,
        )

        # Perform Leiden clustering
        partition = leidenalg.find_partition(
            g, leidenalg.RBERVertexPartition, weights="weight", resolution_parameter=resolution
        )
        clusters = partition.membership

        # Save results
        column_name = f"leiden_{str(resolution).replace('.', '_')}_n{n_neighbors}"
        self.add_column(column_name, clusters)
        self.cluster_columns.append(column_name)
        logger.info("Leiden clustering complete.")

    def run_umap(self, scores: np.ndarray, n_neighbors: int = 50) -> None:
        """Run UMAP on embeddings."""
        logger.info("Running UMAP with %d neighbors", n_neighbors)
        mapper = umap.UMAP(n_neighbors=n_neighbors)
        umap_embedding = mapper.fit_transform(scores)
        self.add_column(f"umap1_n{n_neighbors}", umap_embedding[:, 0])
        self.add_column(f"umap2_n{n_neighbors}", umap_embedding[:, 1])

        # Plot a umap for each clustering_column
        n_clusters = len(self.cluster_columns)
        grid_size = int(np.ceil(np.sqrt(n_clusters)))
        _, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        for i, cluster_column in enumerate(self.cluster_columns):
            plotdf = self.embeddings.loc[self.embeddings.good_bin, [f"umap1_n{n_neighbors}", f"umap2_n{n_neighbors}"]]
            plotdf[cluster_column] = self.embeddings.loc[self.embeddings.good_bin, cluster_column].astype("category")
            # Only plot if <= MAX_PLOT_CLUSTERS clusters
            if len(plotdf[cluster_column].unique()) <= MAX_PLOT_CLUSTERS:
                dsshow(
                    plotdf,
                    ds.Point(f"umap1_n{n_neighbors}", f"umap2_n{n_neighbors}"),
                    aggregator=ds.count_cat(cluster_column),
                    aspect="auto",
                    ax=axs[i // grid_size, i % grid_size],
                )
                axs[i // grid_size, i % grid_size].set_title(cluster_column)
        plt.savefig(f"{self.configuration.output}_umap-n{n_neighbors}_clustered.png")
        plt.close()

        # Plot a umap for each filename on a grid
        n_files = len(self.embeddings.filename.unique())
        grid_size = int(np.ceil(np.sqrt(n_files)))
        logger.debug("Plotting UMAP for each filename in:\n %s", self.embeddings.filename.value_counts())
        _, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        for i, filename in enumerate(self.embeddings.filename.unique()):
            plotdf = self.embeddings.loc[self.embeddings.filename == filename]
            plotdf = plotdf.loc[plotdf.good_bin, [f"umap1_n{n_neighbors}", f"umap2_n{n_neighbors}"]]
            dsshow(
                plotdf,
                ds.Point(f"umap1_n{n_neighbors}", f"umap2_n{n_neighbors}"),
                aspect="auto",
                ax=axs[i // grid_size, i % grid_size],
            )
            axs[i // grid_size, i % grid_size].set_title(filename)
        plt.savefig(f"{self.configuration.output}_umap-n{n_neighbors}_filenames.png")
        plt.close()

    def compute_bin_distance_variance(self):
        """Compute the variance of a bin across all samples in component space.

        Group by bin_name, then compute the variance for each component across all samples, then sum the total variance.

        Returns
        -------
        bin_variance: dict
            A dictionary of {bin_name: variance} for each bin.

        """
        df = self.embeddings.loc[
            self.embeddings.good_bin,
            [col for col in self.embeddings.columns if col.startswith(self.configuration.method)] + ["bin_name"],
        ]
        bin_variance = df.groupby("bin_name").var().sum(axis=1)

        # Plot histogram of bin_variance
        plt.hist(bin_variance, bins=100)
        plt.title("Bin Distance Variance")
        plt.xlabel("Variance")
        plt.ylabel("Frequency")
        plt.savefig(f"{self.configuration.output}_bin_distance_variance.png")
        plt.close()

        return bin_variance.to_dict()
