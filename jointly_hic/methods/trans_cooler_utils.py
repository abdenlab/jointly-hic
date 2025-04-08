"""Utils for operating on trans- Hi-C data in cooler format."""

import logging
from itertools import pairwise

import numpy as np
from cooltools.api.eigdecomp import _fake_cis, _filter_heatmap
from cooltools.lib import numutils

logger = logging.getLogger("joint_pca")


def normalized_affinity_matrix_from_trans(
    contact_matrix, partition, percentile_top, percentile_bottom
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize and balance Hi-C affinity matrix.

    Produce an affinity matrix based on trans data by filling in cis regions with randomly sampled trans pixels from
    the same row or column. The resulting matrix is rebalanced and uniformly scaled such that all rows and columns sum
    to 1 (a.k.a. a stochastic matrix).

    Parameters
    ----------
    contact_matrix : 2D array (n, n)
        Whole genome contact matrix.
    partition : 1D array (n_chroms+1,)
        An offset array providing the starting bin of each chromosome and
        whose last element is the last bin of the last chromosome.
    percentile_top : float
        Clip trans blowout pixels above this cutoff.
    percentile_bottom :
        Mask bins with trans coverage below this cutoff.

    Returns
    -------
    2D array (n, n)
        Normalized affinity matrix
    1D array (n,)
        Boolean mask of bins that are not used in the affinity matrix.

    """
    contact_matrix = np.array(contact_matrix)

    # Ensure contact_matrix is square
    if contact_matrix.shape[0] != contact_matrix.shape[1]:
        logger.critical("contact_matrix.shape: %s", contact_matrix.shape)
        raise ValueError("Matrix A is not square.")

    # Adjust partitions if partition[0] != 0
    if partition[0] != 0:
        logger.error("partition[0] != 0.")
        raise ValueError("partition[0] != 0.")

    logger.debug("contact_matrix.shape: %s", contact_matrix.shape)
    n_bins = contact_matrix.shape[0]
    if not (partition[0] == 0 and partition[-1] == n_bins and np.all(np.diff(partition) > 0)):
        logger.warning("partition: %s", partition)
        logger.warning("partition[0]: %s", partition[0])
        logger.warning("partition[-1]: %s", partition[-1])
        logger.warning("np.diff(partition) > 0: %s", np.diff(partition) > 0)
        raise ValueError(f"Not a valid partition. Must be a monotonic sequence from 0 to {n_bins}.")

    # Zero out cis data and create mask for trans
    extents = pairwise(partition)
    part_ids = []
    for n, (i0, i1) in enumerate(extents):
        contact_matrix[i0:i1, i0:i1] = 0
        part_ids.extend([n] * (i1 - i0))
    part_ids = np.array(part_ids)
    is_trans = part_ids[:, None] != part_ids[None, :]

    # Zero out bins nulled out using NaNs
    is_bad_bin = np.nansum(contact_matrix, axis=0) == 0
    contact_matrix[is_bad_bin, :] = 0
    contact_matrix[:, is_bad_bin] = 0

    if np.any(~np.isfinite(contact_matrix)) or np.any(np.isnan(contact_matrix)):
        raise ValueError("Matrix A contains point-wise NaNs or infinite values, not expected for cooler-loaded maps")

    # Filter the heatmap
    is_good_bin = ~is_bad_bin
    is_valid = np.logical_and.outer(is_good_bin, is_good_bin)
    contact_matrix = _filter_heatmap(contact_matrix, is_trans & is_valid, percentile_top, percentile_bottom)
    is_bad_bin = np.nansum(contact_matrix, axis=0) == 0
    contact_matrix[is_bad_bin, :] = 0
    contact_matrix[:, is_bad_bin] = 0

    # Inject decoy cis data, balance and rescale margins to 1
    contact_matrix = _fake_cis(contact_matrix, ~is_trans)
    numutils.set_diag(contact_matrix, 0, 0)
    contact_matrix = numutils.iterative_correction_symmetric(contact_matrix)[0]
    marg = np.r_[np.sum(contact_matrix, axis=0), np.sum(contact_matrix, axis=1)]
    marg = np.mean(marg[marg > 0])
    contact_matrix /= marg

    contact_matrix = _fake_cis(contact_matrix, ~is_trans)
    numutils.set_diag(contact_matrix, 0, 0)
    contact_matrix = numutils.iterative_correction_symmetric(contact_matrix)[0]
    marg = np.r_[np.sum(contact_matrix, axis=0), np.sum(contact_matrix, axis=1)]
    marg = np.mean(marg[marg > 0])
    contact_matrix /= marg

    contact_matrix[is_bad_bin, :] = 0
    contact_matrix[:, is_bad_bin] = 0

    return contact_matrix, is_bad_bin
