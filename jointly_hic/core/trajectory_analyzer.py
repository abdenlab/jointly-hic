"""Class to handle trajectory analysis of embeddings."""

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

from jointly_hic.core.config import TrajectoryAnalysisConfig

logger = logging.getLogger("joint_pca")


class TrajectoryAnalyzer:
    """Class to handle trajectory analysis of embeddings.

    TrajectoryAnalyzer takes embeddings and runs trajectory analysis on them.
    """

    def __init__(self, configuration: TrajectoryAnalysisConfig, embeddings: pd.DataFrame | None = None):
        """Initialize PostProcessor."""
        self.configuration: TrajectoryAnalysisConfig = configuration
        logger.info("TrajectoryAnalyzer :: configuration: \n\n%s", configuration)
        self.cluster_columns: list[str] = []

        # Pivot all PCs wide with chrom, start, end, good_bin as index and all PC* as columns
        if embeddings is None:
            logger.info(
                "TrajectoryAnalyzer :: Loading embeddings from parquet file: %s", self.configuration.parquet_file
            )
            loaded_embeddings: pd.DataFrame = pd.read_parquet(self.configuration.parquet_file)
        else:
            loaded_embeddings = embeddings

        trajectory_df = loaded_embeddings.pivot(
            index=["chrom", "start", "end"],
            columns="filename",
            values=[col for col in loaded_embeddings.columns if col.startswith(self.configuration.method)],
        )
        # Flatten column names to string
        column_names = trajectory_df.columns.to_flat_index()
        trajectory_df.columns = ["_".join(col) if type(col) is tuple else col for col in column_names]
        # Add good_bin column
        trajectory_df["good_bin"] = trajectory_df.notna().all(axis=1)
        # Add chrom, start, end columns
        trajectory_df.reset_index(inplace=True)
        logger.info("Shape of pivoted trajectory embeddings: %s", trajectory_df.shape)
        self.trajectory_df = trajectory_df

    @property
    def data_values(self) -> np.ndarray:
        """Return columns that start with 'PC#' and good_bin is True as numpy array."""
        return (
            self.trajectory_df.dropna()
            .loc[
                self.trajectory_df.good_bin,
                [col for col in self.trajectory_df.columns if col.startswith(self.configuration.method)],
            ]
            .values
        )

    def add_column(self, column_name, values):
        """Add column to embeddings where good_bin is True and otherwise add NaN."""
        self.trajectory_df[column_name] = np.nan
        self.trajectory_df.loc[self.trajectory_df.good_bin, column_name] = values

    def run(self):
        """Run post-processing."""
        logger.info("Running trajectory analysis")

        # Run KMeans clustering
        for n_clusters in self.configuration.kmeans_clusters:
            self.run_kmeans(self.data_values, n_clusters=n_clusters)

        # Run Leiden clustering
        self.run_leiden(self.data_values, n_neighbors=self.configuration.leiden_neighbors, preprocess=True)

        # Run UMAP
        for n_neighbors in self.configuration.umap_neighbours:
            self.run_umap(self.data_values, n_neighbors=n_neighbors)

        # Save trajectory_df to parquet and csv
        self.trajectory_df.to_parquet(f"{self.configuration.output}_trajectories.pq", index=False)
        self.trajectory_df.to_csv(f"{self.configuration.output}_trajectories.csv.gz", index=False)

        logger.info("Finished trajectory analysis")

    def run_umap(self, data: np.ndarray, n_neighbors: int = 50) -> None:
        """Run UMAP on embeddings."""
        logger.info("Running UMAP with %d neighbors", n_neighbors)
        mapper = umap.UMAP(n_neighbors=n_neighbors)
        umap_embedding = mapper.fit_transform(data)
        self.add_column(f"umap1_n{n_neighbors}", umap_embedding[:, 0])
        self.add_column(f"umap2_n{n_neighbors}", umap_embedding[:, 1])

        # Plot a umap for each clustering_column
        n_clusters = len(self.cluster_columns)
        grid_size = int(np.ceil(np.sqrt(n_clusters)))
        _, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        for i, cluster_column in enumerate(self.cluster_columns):
            plotdf = self.trajectory_df.loc[
                self.trajectory_df.good_bin, [f"umap1_n{n_neighbors}", f"umap2_n{n_neighbors}"]
            ]
            plotdf[cluster_column] = self.trajectory_df.loc[self.trajectory_df.good_bin, cluster_column].astype(
                "category"
            )
            # Plot if <20 clusters
            if len(plotdf[cluster_column].unique()) < 20:
                dsshow(
                    plotdf,
                    ds.Point(f"umap1_n{n_neighbors}", f"umap2_n{n_neighbors}"),
                    aggregator=ds.count_cat(cluster_column),
                    aspect="auto",
                    ax=axs[i // grid_size, i % grid_size],
                )
                axs[i // grid_size, i % grid_size].set_title(cluster_column)
        plt.savefig(f"{self.configuration.output}_umap-n{n_neighbors}_trajectories_clustered.png")
        plt.close()

    def run_kmeans(self, data: np.ndarray, n_clusters: int = 100) -> None:
        """Run KMeans on data array."""
        logger.info("Running KMeans clustering with %d clusters", n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=1000, random_state=42)
        kmeans.fit(data)
        column_name = f"kmeans_{n_clusters}"
        self.add_column(column_name, kmeans.labels_)
        self.cluster_columns.append(column_name)

    def run_leiden(self, data: np.ndarray, n_neighbors: int = 100, preprocess: bool = False) -> None:
        """Perform Leiden clustering on an array.

        Arguments:
        ---------
        data: np.ndarray
            Array of embeddings
        n_neighbors: int
            Number of neighbors to use for KNN graph
        preprocess: bool
            Whether to preprocess the data with StandardScaler

        """
        # Preprocessing: Standardize features by removing the mean and scaling to unit variance
        if preprocess:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
        else:
            data_scaled = data

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
        partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights="weight")
        clusters = partition.membership

        # Save results
        self.add_column("leiden", clusters)
        self.cluster_columns.append("leiden")
