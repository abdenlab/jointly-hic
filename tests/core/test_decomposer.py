import bioframe
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import IncrementalPCA, MiniBatchNMF

from jointly_hic.core.config import JointlyConfiguration
from jointly_hic.core.decomposer import JointlyDecomposer
from jointly_hic.methods.sparse_incremental_svd import SparseIncrementalSVD


@pytest.fixture
def configuration():
    """Fixture for JointPCAConfig."""
    return JointlyConfiguration(
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


@pytest.fixture
def joint_pca_decomposer(mocker, configuration):
    """Fixture for JointPCADecomposer."""
    mocker.patch(
        "jointly_hic.core.decomposer.JointlyDecomposer.partition",
        return_value=[0, 1, 2, 3, 4, 5, 6, 7],
    )
    return JointlyDecomposer(configuration)


def test_joint_pca_decomposer(mocker, configuration):
    """Test JointPCADecomposer init."""
    # Mock JointPCADecomposer.get_partition
    get_partition_patch = mocker.patch(
        "jointly_hic.core.decomposer.JointlyDecomposer.partition",
        return_value=[0, 1, 2, 3, 4, 5, 6, 7],
    )
    joint_pca_decomposer = JointlyDecomposer(configuration)

    # Test init with PCA model
    assert joint_pca_decomposer.configuration == configuration
    assert np.all(joint_pca_decomposer.chromosome_sizes == bioframe.fetch_chromsizes("hg38")[0:23])
    assert isinstance(joint_pca_decomposer.model, IncrementalPCA)
    assert joint_pca_decomposer.partition == get_partition_patch


@pytest.mark.parametrize(
    "method, model",
    [
        ("PCA", IncrementalPCA),
        ("NMF", MiniBatchNMF),
        ("SVD", SparseIncrementalSVD),
    ],
)
def test_joint_nmf_decomposer(mocker, configuration, method, model):
    """Test JointPCADecomposer init when method is NMF."""
    configuration.method = method
    joint_pca_decomposer = JointlyDecomposer(configuration)
    assert isinstance(joint_pca_decomposer.model, model)


def test_run(mocker, joint_pca_decomposer, configuration):
    """Test the JointPCADecomposer.run."""
    joint_pca_decomposer = JointlyDecomposer(configuration)
    joint_pca_decomposer.save_model = mocker.Mock()
    joint_pca_decomposer.decompose_cooler_file = mocker.Mock()
    joint_pca_decomposer.set_union_bad_bins = mocker.MagicMock()
    joint_pca_decomposer.compute_output_embeddings = mocker.Mock()
    joint_pca_decomposer.run()
    joint_pca_decomposer.save_model.assert_called_once()
    joint_pca_decomposer.decompose_cooler_file.assert_has_calls(
        [
            mocker.call(
                filename="test1.mcool",
            ),
            mocker.call(
                filename="test2.mcool",
            ),
        ]
    )
    joint_pca_decomposer.set_union_bad_bins.assert_called_once()


def test_partition(mocker, configuration):
    # Test when _partition is set
    joint_pca_decomposer = JointlyDecomposer(configuration)
    joint_pca_decomposer._partition = np.ndarray([0, 1, 2, 3, 4, 5, 6, 7])
    assert np.all(joint_pca_decomposer.partition == np.ndarray([0, 1, 2, 3, 4, 5, 6, 7]))

    # Test when _partition is not set
    joint_pca_decomposer._partition = None
    # Patch Cooler
    cooler_patch = mocker.patch("jointly_hic.core.decomposer.Cooler")
    cooler_patch.return_value.offset.return_value = 10

    partition = joint_pca_decomposer.partition
    for x in partition:
        assert x == 10

    assert cooler_patch.call_count == 2


def test_get_chromosome_sizes(mocker, joint_pca_decomposer):
    """Test JointPCADecomposer.get_chromosome_sizes."""
    mocker.patch(
        "jointly_hic.core.decomposer.bioframe.fetch_chromsizes",
        return_value=pd.Series([1, 2, 3, 4, 5, 6]),
    )
    assert np.all(joint_pca_decomposer.get_chromosome_sizes() == pd.Series([1, 2, 3, 4, 5, 6]))


def test_set_union_bad_bins(mocker, joint_pca_decomposer):
    """Test JointPCADecomposer.take_union_bad_bins."""
    # Mock JointPCADecomposer.preprocess_matrix
    preprocess_matrix_patch = mocker.patch("jointly_hic.core.decomposer.JointlyDecomposer.preprocess_matrix")
    # Mock JointPCADecomposer.preprocess_matrix to return a dataframe with shape (50, 50) and a union_bad_bins
    preprocess_matrix_patch.return_value = (
        pd.DataFrame(np.random.random((50, 50))),
        pd.DataFrame(np.array([False, True] * 25)),
    )
    # Mock Cooler.cooler
    mocker.patch("jointly_hic.core.decomposer.Cooler")

    # Test take_union_bad_bins
    joint_pca_decomposer.set_union_bad_bins()
    # Assert that preprocess_matrix_patch was called with test1.mcool and test2.mcool
    preprocess_matrix_patch.assert_has_calls(
        [
            mocker.call("test1.mcool", ignore_cache=True),
            mocker.call("test2.mcool", ignore_cache=True),
        ]
    )


def test_preprocess_matrix(mocker, joint_pca_decomposer):
    # Patch numpy.save
    numpy_save_patch = mocker.patch("jointly_hic.core.decomposer.np.save")
    cooler_patch = mocker.patch("jointly_hic.core.decomposer.Cooler")
    normalized_affinity_matrix_patch = mocker.patch("jointly_hic.core.decomposer.normalized_affinity_matrix_from_trans")
    normalized_affinity_matrix_patch.return_value = (mocker.MagicMock(), mocker.MagicMock())
    logical_not_patch = mocker.patch("jointly_hic.core.decomposer.np.logical_not")

    matrix = joint_pca_decomposer.preprocess_matrix("test.mcool")
    cooler_patch.assert_called_once_with(f"test.mcool::/resolutions/{joint_pca_decomposer.configuration.resolution}")
    normalized_affinity_matrix_patch.assert_called_once_with(
        cooler_patch.return_value.matrix.return_value.__getitem__.return_value,
        joint_pca_decomposer.partition.__sub__(),
        joint_pca_decomposer.configuration.percentile_top,
        joint_pca_decomposer.configuration.percentile_bottom,
    )
    logical_not_patch.assert_called_once()
    numpy_save_patch.assert_called_once()


def test_minibatch_fit(mocker, joint_pca_decomposer):
    """Test JointPCADecomposer.minibatch_fit."""
    # Mock IncrementalPCA.partial_fit
    partial_fit_patch = mocker.patch("jointly_hic.core.decomposer.IncrementalPCA.partial_fit")
    matrix = np.random.random((100, 100))

    # Test minibatch_fit
    joint_pca_decomposer.minibatch_fit(matrix)
    partial_fit_patch.assert_called_once()


def test_decompose_cooler_file(mocker, joint_pca_decomposer):
    """Test JointPCADecomposer.decompose_cooler_file."""
    # Mock JointPCADecomposer.minibatch_fit
    minibatch_fit_patch = mocker.patch("jointly_hic.core.decomposer.JointlyDecomposer.minibatch_fit")
    # Mock JointPCADecomposer.preprocess_matrix
    preprocess_matrix_patch = mocker.patch("jointly_hic.core.decomposer.JointlyDecomposer.preprocess_matrix")

    joint_pca_decomposer.decompose_cooler_file("test.mcool")
    minibatch_fit_patch.assert_called_once()
    preprocess_matrix_patch.assert_called_once()


def test_save_model(mocker, joint_pca_decomposer):
    """Test JointPCADecomposer.save_model."""
    joint_pca_decomposer.model = mocker.Mock()
    # Mock gzip.open
    gzip_open_patch = mocker.patch("gzip.open")
    # Mock pickle.dump
    pickle_dump_patch = mocker.patch("pickle.dump")

    # Test save_model with gzip
    joint_pca_decomposer.save_model("testout.pkl.gz")
    gzip_open_patch.assert_called_once_with("testout.pkl.gz", "wb")
    pickle_dump_patch.assert_called_once()


def test_compute_output_embeddings(mocker, joint_pca_decomposer):
    """Test JointPCADecomposer.compute_output_embeddings."""
    joint_pca_decomposer.compute_output_embeddings_single_file = mocker.Mock(
        return_value=pd.DataFrame({"test": [1, 2, 3]})
    )
    embeddings = joint_pca_decomposer.compute_output_embeddings()
    joint_pca_decomposer.compute_output_embeddings_single_file.assert_has_calls(
        [
            mocker.call("test1.mcool"),
            mocker.call("test2.mcool"),
        ]
    )
    assert np.all(embeddings == pd.concat([pd.DataFrame({"test": [1, 2, 3]}), pd.DataFrame({"test": [1, 2, 3]})]))


def test_compute_output_embeddings_single_file(mocker, joint_pca_decomposer):
    """Test JointPCADecomposer.compute_output_embeddings_single_file."""
    # Mock JointPCADecomposer.preprocess_matrix
    preprocess_matrix_patch = mocker.patch("jointly_hic.core.decomposer.JointlyDecomposer.preprocess_matrix")
    # Mock IncrementalPCA.transform and return array with shape (50, 32)
    transform_patch = mocker.patch(
        "jointly_hic.core.decomposer.IncrementalPCA.transform",
        return_value=np.random.random((25, 32)),
    )
    # Patch make_multivec
    mocker.patch("jointly_hic.core.decomposer.make_multivec")
    # Add partitions
    joint_pca_decomposer.partition = np.array([0, 25, 40, 50])
    # Mock JointPCADecomposer.union_bad_bins of length 50
    joint_pca_decomposer.union_bad_bins = np.array([False, True] * 25)

    # Mock Cooler to return bins with length 50
    bins = pd.DataFrame({"chrom": ["chr1"] * 50, "start": np.arange(50), "end": np.arange(1, 51)})
    cooler_patch = mocker.patch("jointly_hic.core.decomposer.Cooler")
    # Return bins for Cooler.bins()[lo:hi]
    cooler_patch.return_value.bins.return_value.__getitem__.return_value = bins

    # Test compute_output_embeddings_single_file
    embeddings = joint_pca_decomposer.compute_output_embeddings_single_file("test.mcool")

    # Assertions
    preprocess_matrix_patch.assert_called_once_with("test.mcool")
    transform_patch.assert_called_once_with(preprocess_matrix_patch.return_value)
    assert embeddings.shape == (50, 37)
    assert embeddings.columns.tolist() == [
        "chrom",
        "start",
        "end",
        "good_bin",
        "filename",
    ] + [f"PCA{i + 1}" for i in range(32)]
    assert embeddings.good_bin.tolist() == [True, False] * 25
    assert embeddings.start.tolist() == np.arange(50).tolist()
    assert embeddings.end.tolist() == np.arange(1, 51).tolist()
    assert embeddings.chrom.tolist() == ["chr1"] * 50

    # Loop over embeddings, every other row matches a row in transorm_patch.return_value
    for i in range(25):
        assert np.all(embeddings.iloc[2 * i, 5:] == transform_patch.return_value[i])
