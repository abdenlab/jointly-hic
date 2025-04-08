import numpy as np
import pandas as pd
import pytest

from jointly_hic.core.config import PostProcessingConfig
from jointly_hic.core.post_processor import PostProcessor


@pytest.fixture
def embeddings():
    """Fixture for embeddings."""
    return pd.DataFrame(
        {
            "chrom": ["chr1"] * 1000,
            "start": np.arange(1, 1001),
            "end": np.arange(2, 1002),
            "weight": np.ones(1000),
            "good_bin": [True] * 1000,
            # Randomly create PC 1-4 as floating point columns
            "PCA1": np.random.rand(1000),
            "PCA2": np.random.rand(1000),
            "PCA3": np.random.rand(1000),
            "PCA4": np.random.rand(1000),
            "filename": ["test1.mcool"] * 1000,
        }
    )


@pytest.fixture
def configuration():
    return PostProcessingConfig(
        parquet_file="test.parquet",
        output="test",
        umap_neighbours=[30, 100],
        kmeans_clusters=[2, 3],
        trajectory_kmeans=[2],
        trajectory_umap_neighbours=[30],
        leiden_resolution=[0.5],
        log_level="INFO",
        method="PCA",
    )


@pytest.fixture
def post_processor(mocker, embeddings, configuration):
    """Fixture for PostProcessor."""
    mocker.patch("jointly_hic.core.post_processor.logger")
    # Mock all plotting and IO functions
    mocker.patch("jointly_hic.core.post_processor.PostProcessor.plot_scores")
    mocker.patch("matplotlib.pyplot.savefig")
    mocker.patch("pandas.DataFrame.to_csv")
    mocker.patch("pandas.DataFrame.to_parquet")
    return PostProcessor(configuration, embeddings)


def test_post_processor_init(mocker, embeddings, configuration):
    """Test PostProcessor.init."""
    mocker.patch("jointly_hic.core.post_processor.logger")
    post_processor = PostProcessor(configuration, embeddings)
    assert post_processor.configuration == configuration
    assert post_processor.embeddings.equals(embeddings)
    assert post_processor.cluster_columns == []


def test_post_processor_init_no_embeddings(mocker, configuration):
    """Test PostProcessor.init without embeddings."""
    mocker.patch("jointly_hic.core.post_processor.logger")
    mock_read_parquet = mocker.patch("pandas.read_parquet")
    post_processor = PostProcessor(configuration)
    assert post_processor.configuration == configuration
    assert post_processor.embeddings == mock_read_parquet.return_value
    assert post_processor.cluster_columns == []


def test_post_processor_components(mocker, post_processor, embeddings):
    """Test PostProcessor.components."""
    assert np.all(
        post_processor.components == embeddings.loc[embeddings.good_bin, ["PCA1", "PCA2", "PCA3", "PCA4"]].values
    )


def test_post_processor_add_column(mocker, post_processor, embeddings):
    """Test PostProcessor.add_column."""
    test_data = np.random.rand(embeddings.shape[0])
    post_processor.add_column("test", test_data)
    assert np.all(post_processor.embeddings["test"] == test_data)
    assert np.all(post_processor.embeddings.loc[~post_processor.embeddings.good_bin, "test"].isna())


def test_post_processor_run(mocker, post_processor):
    """Test PostProcessor.run."""
    post_processor.run_kmeans = mocker.MagicMock()
    post_processor.run_leiden = mocker.MagicMock()
    post_processor.run_umap = mocker.MagicMock()
    post_processor.plot_scores = mocker.MagicMock()

    post_processor.run()
    components = post_processor.components

    # Assert run_kmeans called with scores and n_clusters=2 and 3
    assert post_processor.run_kmeans.call_count == 2
    assert np.all(post_processor.run_kmeans.call_args_list[0][0][0] == components)
    assert post_processor.run_kmeans.call_args_list[0][1]["n_clusters"] == 2
    assert np.all(post_processor.run_kmeans.call_args_list[1][0][0] == components)
    assert post_processor.run_kmeans.call_args_list[1][1]["n_clusters"] == 3

    # Assert run_umap called with scores and n_neighbors=30 and 100
    assert post_processor.run_umap.call_count == 2
    assert np.all(post_processor.run_umap.call_args_list[0][0][0] == components)
    assert post_processor.run_umap.call_args_list[0][1]["n_neighbors"] == 30
    assert np.all(post_processor.run_umap.call_args_list[1][0][0] == components)
    assert post_processor.run_umap.call_args_list[1][1]["n_neighbors"] == 100

    # Assert plot_scores called with scores
    post_processor.plot_scores.assert_called_once()


def test_post_processor_plot_scores(mocker, configuration, embeddings):
    """Test PostProcessor.plot_scores."""
    savefig_patch = mocker.patch("matplotlib.pyplot.savefig")
    mocker.patch("datashader.mpl_ext.dsshow")
    ceil_patch = mocker.patch("numpy.ceil", return_value=2)
    dsshow_patch = mocker.patch("datashader.mpl_ext.dsshow")
    hist_patch = mocker.patch("matplotlib.pyplot.hist")
    subplots_patch = mocker.patch("matplotlib.pyplot.subplots", return_value=(mocker.MagicMock(), mocker.MagicMock()))
    post_processor = PostProcessor(configuration, embeddings)
    post_processor.plot_scores()

    # Assert savefig called with output + "_filename_scores.png" and "_clustered_scores.png"
    savefig_patch.call_count == 2
    savefig_patch.assert_any_call("test_scores.png")
    savefig_patch.assert_any_call("test_scores_filenames.png")
    ceil_patch.call_count == 2
    dsshow_patch.call_count == 2
    hist_patch.call_count == 2
    subplots_patch.call_count == 3


def test_post_processor_run_kmeans(mocker, configuration, embeddings):
    """Test PostProcessor.run_kmeans."""
    post_processor = PostProcessor(configuration, embeddings)
    post_processor.add_column = mocker.MagicMock()
    post_processor.run_kmeans(post_processor.components, n_clusters=2)

    # Assert add_column called with "kmeans_2"
    assert post_processor.add_column.call_count == 1
    assert post_processor.add_column.call_args_list[0][0][0] == "kmeans_2"
    assert post_processor.cluster_columns == ["kmeans_2"]


def test_post_processor_run_umap(mocker, configuration, embeddings):
    """Test PostProcessor.run_umap."""
    mock_umap = mocker.patch("umap.UMAP")
    savefig_patch = mocker.patch("matplotlib.pyplot.savefig")
    subplots_patch = mocker.patch("matplotlib.pyplot.subplots", return_value=(mocker.MagicMock(), mocker.MagicMock()))
    ceil_patch = mocker.patch("numpy.ceil", return_value=2)
    mocker.patch("pandas.DataFrame.loc")
    post_processor = PostProcessor(configuration, embeddings)
    post_processor.add_column = mocker.MagicMock()
    post_processor.cluster_columns = []
    # Patch dsshow and ds.Point and SclarDSArtist
    mocker.patch("datashader.mpl_ext.dsshow")
    mocker.patch("datashader.mpl_ext.ScalarDSArtist")
    mocker.patch("datashader.Point")

    post_processor.run_umap(post_processor.components, n_neighbors=30)

    mock_umap.assert_called_once_with(n_neighbors=30)
    mock_umap.return_value.fit_transform.assert_called_once_with(post_processor.components)
    assert post_processor.add_column.call_count == 2
    assert post_processor.add_column.call_args_list[0][0][0] == "umap1_n30"
    assert post_processor.add_column.call_args_list[1][0][0] == "umap2_n30"
    assert savefig_patch.call_count == 2
    assert ceil_patch.call_count == 2
    savefig_patch.assert_any_call("test_umap-n30_filenames.png")
    subplots_patch.call_count == 2


def test_run_leiden(mocker, configuration, embeddings):
    """Test PostProcessor.run_leiden."""
    post_processor = PostProcessor(configuration, embeddings)
    post_processor.add_column = mocker.MagicMock()
    post_processor.run_leiden(post_processor.components, n_neighbors=20)

    # Assert add_column called with "leiden"
    assert post_processor.add_column.call_count == 1
    assert post_processor.add_column.call_args_list[0][0][0] == "leiden_1_0_n20"
    assert post_processor.cluster_columns == ["leiden_1_0_n20"]
