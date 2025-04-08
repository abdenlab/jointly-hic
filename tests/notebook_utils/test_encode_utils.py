import tempfile

import pandas as pd
import pytest

import jointly_hic.notebook_utils.encode_utils
from jointly_hic.notebook_utils.encode_utils import get_bigwigs_from_experiment_report, EncodeFile, get_signal


@pytest.fixture
def bins():
    return pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 100, 200],
            "end": [100, 200, 300],
        }
    )


def test_encode_utils(bins, tmpdir):
    bigwigs = get_bigwigs_from_experiment_report("liver")
    assert len(bigwigs) > 0

    # Create a new temporary directory for testing the cache
    with tempfile.TemporaryDirectory(dir="./") as cache:
        jointly_hic.notebook_utils.encode_utils.CACHE_DIR = cache
        with EncodeFile(list(bigwigs.values())[0], "bw") as bigwig:
            signal = get_signal(bigwig, bins)
            assert len(signal) == len(bins)
