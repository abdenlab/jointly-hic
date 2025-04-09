"""Configuration for jointly_hic."""

from datetime import datetime

from pydantic import BaseModel, Field


class JointlyConfiguration(BaseModel):
    """Configuration for Jointly.

     Run PCA or NMF on Hi-C contact matrices (in mcool format) from multiple samples using
        incremental embedding algorithm.

    Parameters
    ----------
    mcools: List[str]
        Path to mcool file(s) containing Hi-C contact matrices
    resolution: int
        Bin size to use for contact frequency matrix
    assembly: str
        Genome assembly name used for alignment (e.g. hg38, mm10)
    output: str
        Prefix for output files (Default: output_<datetime>)
    components: int
        Number of components to use for PCA (Default: 32)
    chrom_limit: int
        Limit number of chromosomes to load (Example: '23' to limit to human chr1-chrX and exclude chrY and chrM).
    method: str
        Method to use for decomposition, either PCA or NMF (Default: PCA)
    percentile_top: float
        Top percentile to filter based off of (Default: 99.5)
    percentile_bottom: float
        Bottom percentile to filter based off of (Default: 1.0)
    batch_size: int
        Batch size for mini-batches (Default: 10000)
    log_level: str
        Logging level verbosity (Default: INFO)

    """

    mcools: list[str] = Field(description="Path to mcool file(s) containing Hi-C contact matrices")
    resolution: int = Field(description="Bin size to use for contact frequency matrix")
    assembly: str = Field(description="Genome assembly name used for alignment (e.g. hg38, mm10)")
    output: str = Field(
        default_factory=lambda: f"output_{datetime.now().strftime('%Y_%m_%d_%H_%M')}",
        description="Prefix for output files",
    )
    components: int = Field(default=32, description="Number of components to use for PCA")
    chrom_limit: int = Field(default=-1, description="Limit number of chromosomes to load.")
    method: str = Field(default="PCA", description="Method to use for PCA")
    exclusion_list: str | None = Field(default=None, description="File containing regions to exclude from analysis")
    percentile_top: float = Field(default=99.5, description="Top percentile to filter based off of")
    percentile_bottom: float = Field(default=1.0, description="Bottom percentile to filter based off of")
    batch_size: int = Field(default=10000, description="Batch size for mini-batches")
    log_level: str = Field(default="INFO", description="Logging level verbosity")

    def __str__(self):
        """Pretty print configuration with an indented line per field."""
        result = "====================  Joint PCA Configuration  ====================\n"
        config_dict = self.model_dump()

        max_key_length = max(len(key) for key in config_dict.keys())
        for key, value in config_dict.items():
            if isinstance(value, list):
                value = "\n".join(f"\t\t\t\t- {v}" for v in value)
                result += f"\t{key: <{max_key_length}}  :\n{value}\n"
            else:
                result += f"\t{key: <{max_key_length}}  :\t{value}\n"
        result += "===================================================================\n"
        return result


class PostProcessingConfig(BaseModel):
    """Configuration for post-processing.

    Parameters
    ----------
    parquet_file: str
        Path to parquet file containing embedding results
    output: str
        Prefix for output files (Default: output_<datetime>)
    umap_neighbours: List[int]
        Number of neighbours to use for UMAP
    kmeans_clusters: List[int]
        Number of clusters to use for KMeans
    leiden_resolutions: List[float]
        Resolution to use for Leiden clustering
    method: str
        Method used for decomposition
    log_level: str
        Logging level verbosity (Default: INFO)

    """

    parquet_file: str = Field(description="Path to parquet file containing embedding results")
    output: str = Field(
        default_factory=lambda: f"output_{datetime.now().strftime('%Y_%m_%d_%H_%M')}",
        description="Prefix for output files",
    )
    umap_neighbours: list[int] = Field(default=[30, 100, 500], description="Number of neighbours to use for UMAP")
    kmeans_clusters: list[int] = Field(
        default=[5, 6, 7, 8, 9, 10, 15, 20], description="Number of clusters to use for KMeans"
    )
    leiden_resolutions: list[float] = Field(
        default=[0.1, 0.2, 0.3, 0.5, 0.8, 1.0], description="Resolution to use for Leiden clustering"
    )
    method: str = Field(default="PCA", description="Method to use for decomposition")
    log_level: str = Field(default="INFO", description="Logging level verbosity")

    def __str__(self):
        """Pretty print configuration with an indented line per field."""
        result = "====================  Post Processor Configuration  ====================\n"
        config_dict = self.model_dump()

        max_key_length = max(len(key) for key in config_dict.keys())
        for key, value in config_dict.items():
            result += f"\t{key: <{max_key_length}}  :\t{value}\n"
        result += "========================================================================\n"
        return result


class TrajectoryAnalysisConfig(BaseModel):
    """Configuration for trajectory analysis.

    Parameters
    ----------
    parquet_file: str
        Path to parquet file containing embedding results
    output: str
        Prefix for output files (Default: output_<datetime>)
    umap_neighbours: List[int]
        Number of neighbours to use for UMAP
    kmeans_clusters: List[int]
        Number of clusters to use for KMeans
    log_level: str
        Logging level verbosity (Default: INFO)

    """

    parquet_file: str = Field(description="Path to parquet file containing embedding results")
    output: str = Field(
        default_factory=lambda: f"output_{datetime.now().strftime('%Y_%m_%d_%H_%M')}",
        description="Prefix for output files",
    )
    kmeans_clusters: list[int] = Field(
        default=[5, 6, 7, 8, 9, 10, 15, 20], description="Number of clusters to use for KMeans"
    )
    leiden_neighbors: int = Field(default=100, description="Number of neighbors to use for Leiden clustering")
    umap_neighbours: list[int] = Field(default=[30, 100, 500], description="Number of neighbours to use for UMAP")
    method: str = Field(default="PCA", description="Method to use for decomposition")
    log_level: str = Field(default="INFO", description="Logging level verbosity")

    def __str__(self):
        """Pretty print configuration with an indented line per field."""
        result = "====================  Trajectory Analysis Configuration  ====================\n"
        config_dict = self.model_dump()

        max_key_length = max(len(key) for key in config_dict.keys())
        for key, value in config_dict.items():
            result += f"\t{key: <{max_key_length}}  :\t{value}\n"
        result += "=============================================================================\n"
        return result
