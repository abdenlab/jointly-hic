import os.path
import uuid
from glob import glob

import numpy as np
import pandas as pd
import pytest

from jointly_hic.core.config import JointlyConfiguration
from jointly_hic.core.decomposer import JointlyDecomposer
from jointly_hic.core.post_processor import PostProcessor
from jointly_hic.core.trajectory_analyzer import TrajectoryAnalyzer

SKIP_END2END = False
DATA_FILES = [
    "data/DE-FA-DSG-DdeI-DpnII-P2.hg38.mapq_30.1000.mcool",
    "data/ESC-FA-DSG-DdeI-DpnII-P2.hg38.mapq_30.1000.mcool",
    "data/HB-FA-DSG-DdeI-DpnII-P2.hg38.mapq_30.1000.mcool",
]
EXCLUSION_LIST = "data/blacklist.bed"


@pytest.fixture
def configuration():
    """Fixture for JointPCAConfig."""
    return JointlyConfiguration(
        mcools=DATA_FILES,
        output=f"{uuid.uuid4()!s}",
        resolution=250000,
        assembly="hg38",
        components=32,
        chrom_limit=4,
        method="PCA",
        percentile_top=99.5,
        percentile_bottom=1,
        batch_size=20000,
        log_level="DEBUG",
        exclusion_list=EXCLUSION_LIST,
    )


def check_data_files():
    """Check for data/ files and return True/False."""
    if SKIP_END2END:
        return False
    return all(os.path.exists(data_file) for data_file in DATA_FILES)


@pytest.mark.parametrize("method", ["PCA", "NMF", "SVD"])
@pytest.mark.skipif(not check_data_files(), reason="Data files not found")
def test_end_to_end_decomposition(configuration, method):
    """Test the JointPCADecomposer complete pipeline."""
    try:
        configuration.method = method
        decomposer = JointlyDecomposer(configuration)
        output_embeddings, post_config, trajectory_config = decomposer.run()

        # Make sure output files exist
        assert os.path.exists(post_config.parquet_file)

        # Load embeddings from parquet and check on them
        embeddings = pd.read_parquet(post_config.parquet_file)
        assert np.all(
            embeddings.columns
            == ["chrom", "start", "end", "weight", "good_bin", "filename"] + [f"{method}{i + 1}" for i in range(32)]
        )
        assert embeddings.shape == (10560, 38)
        assert embeddings.dropna().shape == (9498, 38)

        # Run post-processing
        post_processor = PostProcessor(post_config)
        post_processor.run()

        # Check output files
        assert os.path.exists(f"{post_config.output}_post_processed.csv.gz")
        assert os.path.exists(f"{post_config.output}_post_processed.pq")
        assert os.path.exists(f"{post_config.output}_scores.png")
        assert os.path.exists(f"{post_config.output}_scores_clustered.png")
        assert os.path.exists(f"{post_config.output}_scores_filenames.png")

        for n_neighbors in post_config.umap_neighbours:
            assert os.path.exists(f"{post_config.output}_umap-n{n_neighbors}_clustered.png")
            assert os.path.exists(f"{post_config.output}_umap-n{n_neighbors}_filenames.png")

        # Run TrajectoryAnalyzer
        trajectory_analyzer = TrajectoryAnalyzer(trajectory_config)
        trajectory_analyzer.run()

        # Check output files
        assert os.path.exists(f"{trajectory_config.output}_trajectories.csv.gz")
        assert os.path.exists(f"{trajectory_config.output}_trajectories.pq")

    except Exception as e:
        raise e

    finally:
        # Remove  output files
        filelist = glob(configuration.output + "*")

        # Remove mv5 and npy files
        for f in DATA_FILES:
            filelist.extend(glob(f.replace(".mcool", "*mv5")))
            filelist.extend(glob(f.replace(".mcool", "*npy")))

        # Remove  output files
        for f in filelist:
            os.remove(f)
