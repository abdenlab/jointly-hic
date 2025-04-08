import numpy as np
import pandas as pd
import pytest

from jointly_hic.core.config import TrajectoryAnalysisConfig
from jointly_hic.core.trajectory_analyzer import TrajectoryAnalyzer


@pytest.fixture
def config():
    return TrajectoryAnalysisConfig(
        parquet_file="test.parquet",
        output="test_output",
        kmeans_clusters=[2, 3],
        leiden_neighbors=10,
        umap_neighbors=[10, 15],
    )


@pytest.fixture
def embeddings():
    return pd.DataFrame(
        {
            "chrom": ["chr1"] * 1000,
            "start": list(range(1, 501)) + list(range(1, 501)),
            "end": list(range(2, 502)) + list(range(2, 502)),
            "PCA1": np.random.rand(1000),
            "PCA2": np.random.rand(1000),
            "PCA3": np.random.rand(1000),
            "PCA4": np.random.rand(1000),
            "PCA5": np.random.rand(1000),
            "PCA6": np.random.rand(1000),
            "good_bin": np.random.choice([True, False], size=1000, p=[0.95, 0.05]),
            "filename": ["file1"] * 500 + ["file2"] * 500,
        }
    )


@pytest.fixture
def trajectory_analyzer(config, embeddings):
    return TrajectoryAnalyzer(config, embeddings)


def test_init_no_embeddings(mocker, config):
    # Patch pd.read_parquet
    mock_read_parquet = mocker.patch("pandas.read_parquet")
    analyzer = TrajectoryAnalyzer(config)
    assert analyzer.configuration == config
    mock_read_parquet.assert_called_once_with(config.parquet_file)
    assert analyzer.trajectory_df == mock_read_parquet.return_value.pivot.return_value


def test_init_with_embeddings(mocker, config, embeddings):
    # Patch pd.read_parquet
    mock_read_parquet = mocker.patch("pandas.read_parquet")
    analyzer = TrajectoryAnalyzer(config, embeddings)
    assert analyzer.configuration == config
    mock_read_parquet.assert_not_called()
    assert analyzer.trajectory_df.shape == (500, 16)
    assert "PCA1_file1" in analyzer.trajectory_df.columns
    assert "PCA1_file2" in analyzer.trajectory_df.columns
    assert "PCA6_file1" in analyzer.trajectory_df.columns
    assert "PCA6_file2" in analyzer.trajectory_df.columns
    assert "chrom" in analyzer.trajectory_df.columns
    assert "start" in analyzer.trajectory_df.columns
    assert "end" in analyzer.trajectory_df.columns
    assert "good_bin" in analyzer.trajectory_df.columns


def test_run(mocker, trajectory_analyzer):
    # Patch run_kmeans, run_leiden, run_umap, to_parquet, to_csv
    mock_run_kmeans = mocker.patch("jointly_hic.core.trajectory_analyzer.TrajectoryAnalyzer.run_kmeans")
    mock_run_leiden = mocker.patch("jointly_hic.core.trajectory_analyzer.TrajectoryAnalyzer.run_leiden")
    mock_run_umap = mocker.patch("jointly_hic.core.trajectory_analyzer.TrajectoryAnalyzer.run_umap")
    mock_to_parquet = mocker.patch("pandas.DataFrame.to_parquet")
    mock_to_csv = mocker.patch("pandas.DataFrame.to_csv")
    trajectory_analyzer.run()
    assert mock_run_kmeans.call_count == 2
    mock_run_leiden.assert_called_once()
    assert mock_run_umap.call_count == 3
    mock_to_parquet.assert_called_once_with(f"{trajectory_analyzer.configuration.output}_trajectories.pq", index=False)
    mock_to_csv.assert_called_once_with(f"{trajectory_analyzer.configuration.output}_trajectories.csv.gz", index=False)


def test_run_umap(mocker, trajectory_analyzer):
    # Patch plt.subplots
    mock_subplots = mocker.patch("matplotlib.pyplot.subplots", return_value=(None, None))
    # Patch savefig and close
    mock_savefig = mocker.patch("matplotlib.pyplot.savefig")
    mock_close = mocker.patch("matplotlib.pyplot.close")
    trajectory_analyzer.run_umap(trajectory_analyzer.data_values, n_neighbors=10)
    assert "umap1_n10" in trajectory_analyzer.trajectory_df.columns
    assert "umap2_n10" in trajectory_analyzer.trajectory_df.columns
    mock_subplots.assert_called_once()
    mock_savefig.assert_called_once_with(
        f"{trajectory_analyzer.configuration.output}_umap-n10_trajectories_clustered.png"
    )
    mock_close.assert_called_once()


def test_run_kmeans(mocker, trajectory_analyzer):
    trajectory_analyzer.run_kmeans(trajectory_analyzer.data_values, n_clusters=10)
    assert "kmeans_10" in trajectory_analyzer.trajectory_df.columns


@pytest.mark.parametrize("preprocess", [True, False])
def test_run_leiden(trajectory_analyzer, preprocess):
    trajectory_analyzer.run_leiden(trajectory_analyzer.data_values, n_neighbors=10, preprocess=preprocess)
    assert "leiden" in trajectory_analyzer.trajectory_df.columns
