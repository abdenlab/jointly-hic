import os
from uuid import uuid4

import pytest
import bioframe
import numpy as np

from jointly_hic.notebook_utils.plotting import plot_components, plot_signal, create_dotplot


@pytest.fixture
def bins():
    bins = bioframe.binnify(bioframe.fetch_chromsizes("hg38"), binsize=1000000)
    bins["PCA1"] = np.random.rand(len(bins))
    bins["PCA2"] = np.random.rand(len(bins))
    bins["good_bin"] = [True] * len(bins)
    bins["name"] = np.random.choice(["sample1", "sample2"], len(bins))
    bins["cluster"] = np.random.choice([1, 2, 3], len(bins))
    return bins


def test_plot_components(bins):
    filename = f"{uuid4()!s}.png"
    plot_components(
        bins,
        component_1="PCA1",
        component_2="PCA2",
        cluster_col="cluster",
        name_col="name",
        filename=filename,
        remove_ticks=True,
    )
    os.remove(filename)


def test_plot_signal(bins):
    filename = f"{uuid4()!s}.png"
    plot_signal(
        bins,
        component_1="PCA1",
        component_2="PCA2",
        signals=["PCA1"],
        labels=["cluster"],
        filename=filename,
        remove_ticks=True,
    )
    os.remove(filename)


def test_create_dotplot(bins):
    filename = f"{uuid4()!s}.png"
    create_dotplot(bins, "cluster", ["PCA1", "PCA2"], filename=filename, threshold=0.5)
    os.remove(filename)
