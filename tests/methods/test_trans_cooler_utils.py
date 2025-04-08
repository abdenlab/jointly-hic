import numpy as np
import pytest

from jointly_hic.methods.trans_cooler_utils import normalized_affinity_matrix_from_trans


def test_normalized_affinity_matrix_from_trans(mocker):
    # Check that a non-symmetric matrix raises an error
    contact_matrix = np.array([[1, 2, 3], [4, 5, 6]])
    partition = np.array([0, 2, 3])
    percentile_top = 95
    percentile_bottom = 5
    with pytest.raises(ValueError):
        normalized_affinity_matrix_from_trans(contact_matrix, partition, percentile_top, percentile_bottom)

    # Check that a non-monotonic partition raises an error
    partition = np.array([0, 2, 1])
    with pytest.raises(ValueError):
        normalized_affinity_matrix_from_trans(contact_matrix, partition, percentile_top, percentile_bottom)

    # Check that a partition that doesn't start at 0 raises an error
    partition = np.array([1, 2])
    with pytest.raises(ValueError):
        normalized_affinity_matrix_from_trans(contact_matrix, partition, percentile_top, percentile_bottom)

    # Create an 8x8 test matrix
    contact_matrix = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [2, 1, 2, 3, 4, 5, 6, 7],
            [3, 2, 1, 2, 3, 4, 5, 6],
            [4, 3, 2, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 2, 3, 4],
            [6, 5, 4, 3, 2, 1, 2, 3],
            [7, 6, 5, 4, 3, 2, 1, 2],
            [8, 7, 6, 5, 4, 3, 2, 1],
        ],
        dtype=np.float64,
    )
    partition = np.array([0, 4, 8])
    percentile_top = 95
    percentile_bottom = 5
    matrix, bad_bins = normalized_affinity_matrix_from_trans(
        contact_matrix, partition, percentile_top, percentile_bottom
    )

    # Check that the matrix is symmetric
    assert np.allclose(matrix, matrix.T)
    # Check that the matrix is finite
    assert np.all(np.isfinite(matrix))
    # Check that the matrix is stochastic
    assert np.allclose(np.sum(matrix, axis=0), np.ones(8), atol=0.01)
    assert np.allclose(np.sum(matrix, axis=1), np.ones(8), atol=0.01)
    # Check that the matrix is balanced
    assert np.allclose(np.sum(matrix, axis=0), np.sum(matrix, axis=1), atol=0.01)
    # Check that bad_bins is empty
    assert bad_bins.size == 8
