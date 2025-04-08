import os.path
from glob import glob
from uuid import uuid4

import bioframe
import numpy as np
import pytest

from jointly_hic.notebook_utils.heatmap import encode_chip_heatmap


@pytest.fixture
def bins():
    bins = bioframe.binnify(bioframe.fetch_chromsizes("hg38"), binsize=1000000)
    bins["feature1"] = np.random.rand(len(bins))
    bins["feature2"] = np.random.rand(len(bins))
    bins["name"] = "sample1"
    bins["cluster"] = np.random.choice([1, 2, 3], len(bins))
    return bins


def test_encode_chip_heatmap(bins):
    output_prefix = str(uuid4())
    encode_chip_heatmap(
        bins,
        data_cols=["feature1", "feature2"],
        name="sample1",
        bigwigs={},
        labels=["cluster"],
        output_prefix=output_prefix,
    )

    assert os.path.exists(f"{output_prefix}.pq")
    assert os.path.exists(f"{output_prefix}_heatmap.png")

    # Remove output files
    for f in glob(f"{output_prefix}*"):
        os.remove(f)
