import numpy as np
import pandas as pd
import pytest

from jointly_hic.notebook_utils.bin_operations import make_chromarms, distance_traveled_by_bin


def test_distance_traveled_by_bin():
    # Mock input DataFrame
    df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 100, 200],
            "PCA1_T0": [1, 2, 3],
            "PCA2_T0": [4, 5, 6],
            "PCA1_T1": [1, 2, 4],
            "PCA2_T1": [4, 5, 7],
            "Time": ["T0", "T0", "T0"],
        }
    )
    time_column = "Time"
    time_points = ["T0"]
    prefix = "PCA"

    result_df = distance_traveled_by_bin(df, time_column, time_points, prefix)

    # Expected results
    expected_df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 100, 200],
            "total_distance": [0.0, 0.0, 0.0],
        }
    )

    pd.testing.assert_frame_equal(result_df, expected_df)


def test_make_chromarms():
    chromsizes = pd.Series({"chr1": 1000, "chr2": 2000})
    mids = {"chr1": 500, "chr2": 1000}
    binsize = 100

    result_df = make_chromarms(chromsizes, mids, binsize)

    expected_df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2", "chr2"],
            "start": [0, 500, 0, 1000],
            "end": [500, 1000, 1000, 2000],
            "name": ["chr1p", "chr1q", "chr2p", "chr2q"],
        }
    )

    pd.testing.assert_frame_equal(result_df, expected_df)
