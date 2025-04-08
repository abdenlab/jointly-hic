import numpy as np
import pytest
import scipy
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import gen_batches

from jointly_hic.methods.sparse_incremental_svd import SparseIncrementalSVD

N_COMPONENTS = 20
BATCH_SIZE = 50
MATRIX_SIZE = 200


@pytest.fixture(scope="module")
def toeplitz():
    mat = scipy.linalg.toeplitz(np.arange(MATRIX_SIZE)).astype(np.float64)
    return mat


@pytest.fixture(scope="module")
def sparse_toeplitz(toeplitz):
    return scipy.sparse.csr_matrix(toeplitz)


@pytest.fixture(scope="module")
def n_samples(sparse_toeplitz):
    return sparse_toeplitz.shape[0]


def truncated_svd(matrix, n_components=N_COMPONENTS):
    svd = TruncatedSVD(n_components=n_components)
    svd = svd.fit(matrix)
    return svd


def sparse_incremental_svd(matrix, n_components=N_COMPONENTS):
    svd = SparseIncrementalSVD(n_components=n_components, batch_size=BATCH_SIZE)
    svd = svd.partial_fit(matrix)
    return svd


def test_truncated_svd(toeplitz):
    svd = truncated_svd(toeplitz)
    assert svd.components_.shape == (N_COMPONENTS, MATRIX_SIZE)


def test_sparse_truncated_svd(sparse_toeplitz):
    svd = truncated_svd(sparse_toeplitz)
    assert svd.components_.shape == (N_COMPONENTS, MATRIX_SIZE)


def test_incremental_svd_partial_fit(toeplitz):
    svd = SparseIncrementalSVD(n_components=N_COMPONENTS, batch_size=BATCH_SIZE)
    svd.partial_fit(toeplitz)
    assert svd.components_.shape == (N_COMPONENTS, MATRIX_SIZE)


def test_sparse_incremental_svd_partial_fit(sparse_toeplitz):
    svd = sparse_incremental_svd(sparse_toeplitz)
    assert svd.components_.shape == (N_COMPONENTS, MATRIX_SIZE)


def test_sparse_incremental_svd_fit(sparse_toeplitz):
    svd = SparseIncrementalSVD(n_components=N_COMPONENTS, batch_size=BATCH_SIZE)
    svd.fit(sparse_toeplitz)
    assert svd.components_.shape == (N_COMPONENTS, MATRIX_SIZE)


def test_incremental_svd_minibatch_fit(sparse_toeplitz, n_samples):
    svd = SparseIncrementalSVD(n_components=N_COMPONENTS, batch_size=BATCH_SIZE)
    for i, batch in enumerate(gen_batches(n_samples, BATCH_SIZE, min_batch_size=N_COMPONENTS)):
        svd.partial_fit(sparse_toeplitz[batch])
        if i == 0:
            assert svd.components_.shape == (N_COMPONENTS, MATRIX_SIZE)
        else:
            assert svd.components_.shape == (N_COMPONENTS, MATRIX_SIZE)


def test_same_results(sparse_toeplitz):
    """Test that SparseIncrementalSVD and TruncatedSVD give the same results (R^2>0.999)."""
    svd = truncated_svd(sparse_toeplitz)
    svd2 = sparse_incremental_svd(sparse_toeplitz)

    # Assert components_ are correlated > 0.999
    for i in range(N_COMPONENTS):
        corr = np.corrcoef(svd.components_[i], svd2.components_[i])[0, 1]
        assert np.abs(corr) > 0.999

    assert np.allclose(np.abs(svd.singular_values_), np.abs(svd2.singular_values_), rtol=1e-3)


def test_transform(sparse_toeplitz):
    svd = sparse_incremental_svd(sparse_toeplitz, N_COMPONENTS)
    transformed = svd.transform(sparse_toeplitz)
    assert transformed.shape == (MATRIX_SIZE, N_COMPONENTS)
