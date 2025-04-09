from unittest.mock import MagicMock

from jointly_hic.parser import JointlyCommandLineInterface


def test_joint_pca_cli_embedding(mocker):
    """Test JointPCACommandLineInterface for embedding."""
    parser = JointlyCommandLineInterface()

    # Test parsing arguments for embedding
    args = parser.parser.parse_args(
        "embed --mcools data.mcool --chrom-limit 5 --resolution 100000 --assembly hg38 --method NMF".split()
    )
    assert args.func.__name__ == "run_embedding"
    assert args.mcools == ["data.mcool"]
    assert args.chrom_limit == 5
    assert args.resolution == 100000
    assert args.assembly == "hg38"
    assert args.log_level == "INFO"
    assert args.method == "NMF"


def test_joint_pca_cli_post(mocker):
    """Test JointPCACommandLineInterface fpr parser."""
    parser = JointlyCommandLineInterface()

    # Test parsing arguments for post-processing
    args = parser.parser.parse_args(
        "post-process --parquet-file data.parquet --output test --umap-neighbours 30 100 500 "
        "--kmeans-clusters 5 10 15 20".split()
    )
    assert args.func.__name__ == "run_post_process"
    assert args.parquet_file == "data.parquet"
    assert args.output == "test"
    assert args.umap_neighbours == [30, 100, 500]
    assert args.kmeans_clusters == [5, 10, 15, 20]
    assert args.log_level == "INFO"

    # Patch PostProcessor
    mocker.patch("jointly_hic.core.post_processor.PostProcessor")
    args.func(args)


def test_joint_pca_cli_trajectory(mocker):
    """Test JointPCACommandLineInterface for trajectory analyzer."""
    parser = JointlyCommandLineInterface()

    # Test parsing arguments for trajectory analysis
    args = parser.parser.parse_args(
        "trajectory --parquet-file data.parquet --output test --umap-neighbours 30 100 500 "
        "--kmeans-clusters 5 10 15 20".split()
    )
    assert args.func.__name__ == "run_trajectory"
    assert args.parquet_file == "data.parquet"
    assert args.output == "test"
    assert args.umap_neighbours == [30, 100, 500]
    assert args.kmeans_clusters == [5, 10, 15, 20]
    assert args.log_level == "INFO"

    # Patch TrajectoryAnalyzer
    mocker.patch("jointly_hic.core.trajectory_analyzer.TrajectoryAnalyzer")
    args.func(args)
