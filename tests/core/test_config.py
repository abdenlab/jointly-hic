import pytest
from pydantic_core._pydantic_core import ValidationError

from jointly_hic.core.config import JointlyConfiguration, PostProcessingConfig, TrajectoryAnalysisConfig


def test_joint_pca_config():
    config = JointlyConfiguration(
        mcools=["test1.mcool", "test2.mcool"],
        output="test",
        resolution=10000,
        assembly="hg38",
        components=32,
        chrom_limit=23,
        method="PCA",
        percentile_top=99.5,
        percentile_bottom=1,
        batch_size=20000,
        log_level="INFO",
    )
    assert config.mcools == ["test1.mcool", "test2.mcool"]
    assert config.output == "test"
    assert config.resolution == 10000
    assert config.assembly == "hg38"
    assert config.components == 32
    assert config.chrom_limit == 23
    assert config.method == "PCA"
    assert config.percentile_top == 99.5
    assert config.percentile_bottom == 1
    assert config.batch_size == 20000
    assert config.log_level == "INFO"
    assert isinstance(str(config), str)

    # Test error for wrong types
    with pytest.raises(ValidationError):
        JointlyConfiguration(
            mcools="test1.mcool",
            output="test",
            resolution=10000,
            assembly="hg38",
            components=32,
            chrom_limit=23,
            method="PCA",
            percentile_top=99.5,
            percentile_bottom=1,
            batch_size=20000,
            log_level="INFO",
        )


def test_post_processing_config():
    config = PostProcessingConfig(
        parquet_file="test.parquet",
        output="test",
        umap_neighbours=[30, 100, 500],
        kmeans_clusters=[5, 10, 15, 20],
        log_level="INFO",
    )
    assert config.parquet_file == "test.parquet"
    assert config.output == "test"
    assert config.umap_neighbours == [30, 100, 500]
    assert config.kmeans_clusters == [5, 10, 15, 20]
    assert config.log_level == "INFO"
    assert isinstance(str(config), str)

    # Test error for wrong types
    with pytest.raises(ValidationError):
        PostProcessingConfig(
            parquet_file="test.parquet",
            output="test",
            umap_neighbours=30,
            kmeans_clusters=[5, 10, 15, 20, 25, 30],
            log_level="INFO",
        )


def test_trajectory_analysis_config():
    config = TrajectoryAnalysisConfig(
        parquet_file="test.parquet",
        output="test",
        umap_neighbours=[30, 100, 500],
        kmeans_clusters=[5, 10, 15, 20, 25, 30],
        log_level="INFO",
    )
    assert config.parquet_file == "test.parquet"
    assert config.output == "test"
    assert config.umap_neighbours == [30, 100, 500]
    assert config.kmeans_clusters == [5, 10, 15, 20, 25, 30]
    assert config.log_level == "INFO"
    assert isinstance(str(config), str)

    # Test error for wrong types
    with pytest.raises(ValidationError):
        TrajectoryAnalysisConfig(
            parquet_file="test.parquet",
            output="test",
            umap_neighbours=30,
            kmeans_clusters=[5, 10, 15, 20, 25, 30],
            log_level="INFO",
        )
