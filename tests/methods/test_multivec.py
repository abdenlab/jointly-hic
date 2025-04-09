from os import remove
from os.path import isfile
from uuid import uuid4

import h5py
import numpy as np
import pandas as pd

from jointly_hic.methods.multivec import nansum_agg, nanmean_agg, logsumexp_agg, make_multivec


def test_nansum_agg():
    # Create 3D array of test data
    data = np.array([[[1, np.nan], [3, 4], [5, 6]], [[7, np.nan], [9, 10], [11, 12]]])
    result = nansum_agg(data)
    assert np.all(result == np.array([[8, 16, 14], [12, 0, 18]]))


def test_nanmean_agg():
    # Create 3D array of test data
    data = np.array([[[1, np.nan], [3, 4], [5, 6]], [[7, np.nan], [9, 10], [11, 12]]])
    result = nanmean_agg(data)
    assert np.allclose(result, np.array([[4, 8, 7], [6, np.nan, 9]]), equal_nan=True)


def test_logsumexp_agg():
    # Create 3D array of test data
    data = np.array([[[1, np.nan], [3, 4], [5, 6]], [[7, np.nan], [9, 10], [11, 12]]])
    result = logsumexp_agg(data)
    assert result.shape == (2, 3)


def test_make_multivec():
    # Create a test dataframe for chr1 and chr2 that is 5 mb long
    df = pd.DataFrame(
        {
            "chrom": ["chr1"] * 10000 + ["chr2"] * 10000,
            "start": list(range(0, 5000000, 500)) * 2,
            "end": list(range(500, 5000500, 500)) * 2,
            "feature1": np.random.rand(20000),
            "feature2": np.random.rand(20000) * 100 + 3,
        }
    )
    feature_names = ["feature1", "feature2"]
    base_res = 500
    chrom_sizes = {"chr1": 5000500, "chr2": 5000500}
    outpath = f"test-{uuid4()!s}.h5"

    try:
        make_multivec(outpath, df, feature_names, base_res, chrom_sizes)
        assert isfile(outpath), "Output file was not created"

        with h5py.File(outpath, "r") as f:
            # Verify metadata
            assert f["info"].attrs["tile-size"] == 1024, "Tile-size metadata is incorrect"
            chrom_names = f["chroms"]["name"][:]
            chrom_lengths = f["chroms"]["length"][:]
            assert len(chrom_names) == 2, "Incorrect number of chromosomes"
            assert all(name.decode() in chrom_sizes for name in chrom_names), "Chromosome names mismatch"
            assert all(length == chrom_sizes[name.decode()] for name, length in zip(chrom_names, chrom_lengths)), (
                "Chromosome lengths mismatch"
            )

            # Check data structure
            assert "resolutions" in f, "Resolutions group is missing"
            res_group = f["resolutions"][str(base_res)]
            assert set(feature_names).issubset(set(res_group.attrs["row_infos"].astype(str))), (
                "Feature names mismatch in attributes"
            )

            # Validate data integrity for the highest resolution level
            for chrom in chrom_sizes.keys():
                dset = res_group["values"][str.encode(chrom)]
                # Ensure dataset shape matches expected shape based on input DataFrame and chrom_sizes
                expected_shape = (chrom_sizes[chrom] // base_res - 1, len(feature_names))
                assert dset.shape == expected_shape, f"Dataset shape for {chrom} is incorrect"
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    finally:
        remove(outpath) if isfile(outpath) else None
