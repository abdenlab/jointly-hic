"""Parser for jointly_hic."""

import argparse
from datetime import datetime

from jointly_hic.core.config import JointlyConfiguration, PostProcessingConfig, TrajectoryAnalysisConfig
from jointly_hic.hdf5db.experiments2yaml import configure_embedding_metadata_parser
from jointly_hic.hdf5db.hdf5db import configure_hdf5db_parser
from jointly_hic.hdf5db.trackcsv2yaml import configure_trackcsv2yaml_parser


class JointlyCommandLineInterface:
    """Main CLI for jointly_hic."""

    def __init__(self):
        """Initialize parser."""
        parser = argparse.ArgumentParser(
            prog="jointly",
            description="Jointly decompose Hi-C contact matrices using PCA or NMF and analyze the embeddings",
            epilog="For more information, visit https://github.com/abdenlab/jointly-hic",
        )
        parser.add_argument(
            "--log",
            help="Log file",
            required=False,
            type=str,
            default=f"output_{datetime.now().strftime('%Y_%m_%d_%H_%M')}_log.txt",
        )
        parser.add_argument(
            "--log-level",
            help="Logging level",
            required=False,
            type=str,
            choices=["DEBUG", "INFO", "WARNING", "CRITICAL", "ERROR"],
            default="INFO",
        )
        subparsers = parser.add_subparsers(help="List of commands.", required=True)

        # Embed parser
        embed_parser = subparsers.add_parser("embed", help="Embed Hi-C contact matrices using PCA or NMF.")
        self.configure_embed_parser(embed_parser)

        # Post-process parser
        post_process_parser = subparsers.add_parser("post-process", help="Post process embeddings from jointly-hic.")
        self.configure_post_process_parser(post_process_parser)

        # Trajectory parser
        trajectory_parser = subparsers.add_parser(
            "trajectory", help="Trajectory analysis of embeddings from jointly-hic."
        )
        self.configure_trajectory_parser(trajectory_parser)

        # HDF5 database parser
        hdf5db_parser = subparsers.add_parser(
            "hdf5db", help="Create HDF5 database from Jointly-hic output and signal files."
        )
        configure_hdf5db_parser(hdf5db_parser)

        # embedding-metadata parser
        embedding_metadata_parser = subparsers.add_parser(
            "embedding2yaml", help="Create metadata YAML from embeddings."
        )
        configure_embedding_metadata_parser(embedding_metadata_parser)

        # trackcsv2yaml parser
        trackcsv2yaml_parser = subparsers.add_parser(
            "tracks2yaml", help="Convert a 4-column CSV metadata file to a YAML file."
        )
        configure_trackcsv2yaml_parser(trackcsv2yaml_parser)

        self.parser = parser

    def configure_embed_parser(self, embed_parser):
        """Configure embed parser."""
        embed_parser.add_argument(
            "--mcools",
            help="Path to mcool file(s) containing Hi-C contact matrices",
            required=True,
            nargs="+",
            type=str,
        )
        embed_parser.add_argument(
            "--resolution",
            help="Bin size to use for contact frequency matrix",
            required=True,
            type=int,
        )
        embed_parser.add_argument(
            "--assembly",
            help="Genome assembly name used for alignment (e.g. hg38, mm10)",
            required=True,
            type=str,
            choices=["hg38", "mm10", "hg19", "mm9"],
        )
        embed_parser.add_argument(
            "--output",
            help="Prefix for output files (Default: output_<datetime>)",
            required=False,
            type=str,
            default=f"output_{datetime.now().strftime('%Y_%m_%d_%H_%M')}",
        )
        embed_parser.add_argument(
            "--components",
            help="Number of components to use for PCA (Default: 32)",
            required=False,
            type=int,
            default=32,
        )
        embed_parser.add_argument(
            "--chrom-limit",
            help="Limit number of chromosomes to load "
            "(Example: '23' to limit to human chr1-chrX and exclude chrY and chrM).",
            required=False,
            type=int,
            default=-1,
        )
        embed_parser.add_argument(
            "--method",
            help="Method to use for decomposition, either PCA, NMF or SVD (Default: PCA)",
            required=False,
            type=str,
            choices=["PCA", "NMF", "SVD"],
            default="PCA",
        )
        embed_parser.add_argument(
            "--exclusion-list",
            help="File containing regions to exclude from analysis",
            required=False,
            type=str,
            default=None,
        )
        embed_parser.add_argument(
            "--percentile-top",
            help="Top percentile for filtering (Default: 99.5)",
            required=False,
            type=float,
            default=99.5,
        )
        embed_parser.add_argument(
            "--percentile-bottom",
            help="Bottom percentile for filtering (Default: 1)",
            required=False,
            type=float,
            default=1,
        )
        embed_parser.add_argument(
            "--batch-size",
            help="Batch size for mini-batches (Default: 10000)",
            required=False,
            type=int,
            default=10000,
        )
        embed_parser.set_defaults(func=self.run_embedding)

    def configure_post_process_parser(self, post_process_parser):
        """Configure post-process parser."""
        post_process_parser.add_argument(
            "--parquet-file",
            help="Path to parquet file containing embeddings",
            required=True,
            type=str,
        )
        post_process_parser.add_argument(
            "--output",
            help="Prefix for output files (Default: output_<datetime>)",
            required=False,
            type=str,
            default=f"output_{datetime.now().strftime('%Y_%m_%d_%H_%M')}",
        )
        post_process_parser.add_argument(
            "--umap-neighbours",
            help="Number of neighbours for UMAP (Default: [30, 100, 500])",
            required=False,
            type=int,
            nargs="+",
            default=[30, 100, 500],
        )
        post_process_parser.add_argument(
            "--kmeans-clusters",
            help="Number of clusters for k-means (Default: [5, 10, 15, 20])",
            required=False,
            type=int,
            nargs="+",
            default=[5, 10, 15, 20],
        )
        post_process_parser.set_defaults(func=self.run_post_process)

    def configure_trajectory_parser(self, trajectory_parser):
        """Configure trajectory parser."""
        trajectory_parser.add_argument(
            "--parquet-file",
            help="Path to parquet file containing embeddings",
            required=True,
            type=str,
        )
        trajectory_parser.add_argument(
            "--output",
            help="Prefix for output files (Default: output_<datetime>)",
            required=False,
            type=str,
            default=f"output_{datetime.now().strftime('%Y_%m_%d_%H_%M')}",
        )
        trajectory_parser.add_argument(
            "--umap-neighbours",
            help="Number of neighbours for UMAP (Default: [30, 100, 500])",
            required=False,
            type=int,
            nargs="+",
            default=[30, 100, 500],
        )
        trajectory_parser.add_argument(
            "--kmeans-clusters",
            help="Number of clusters for k-means (Default: [5, 10, 15, 20])",
            required=False,
            type=int,
            nargs="+",
            default=[5, 10, 15, 20],
        )
        trajectory_parser.set_defaults(func=self.run_trajectory)

    def run_embedding(self, args):
        """Run embedding."""
        from jointly_hic.core.decomposer import JointlyDecomposer
        from jointly_hic.core.post_processor import PostProcessor
        from jointly_hic.core.trajectory_analyzer import TrajectoryAnalyzer

        # Run embedding
        configuration = JointlyConfiguration(**args.__dict__)
        decomposer = JointlyDecomposer(configuration)
        embeddings, post_config, trajectory_config = decomposer.run()

        # Run post-processing
        post_processor = PostProcessor(post_config, embeddings)
        post_processor.run()

        # Run trajectory analysis
        trajectory_analyzer = TrajectoryAnalyzer(trajectory_config, embeddings)
        trajectory_analyzer.run()

    def run_post_process(self, args):
        """Run post-processing."""
        from jointly_hic.core.post_processor import PostProcessor

        config = PostProcessingConfig(**args.__dict__)
        post_processor = PostProcessor(config)
        post_processor.run()

    def run_trajectory(self, args):
        """Run trajectory analysis."""
        from jointly_hic.core.trajectory_analyzer import TrajectoryAnalyzer

        config = TrajectoryAnalysisConfig(**args.__dict__)
        trajectory_analyzer = TrajectoryAnalyzer(config)
        trajectory_analyzer.run()
