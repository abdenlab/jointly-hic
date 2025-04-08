"""Convert a 4-column CSV metadata file to a YAML file."""

import argparse
import csv
import logging

import yaml

# Set up logging
logger = logging.getLogger("joint_pca")


def csv_to_yaml(arguments: argparse.Namespace):
    """Convert a CSV file with biosample, assay, experiment, accession columns to a YAML file.

    Args:
    ----
        arguments (argparse.Namespace): Arguments from the command line.

    """
    csv_file = arguments.csv_file  # Path to the input CSV file.
    yaml_file = arguments.yaml_file  # Path to the output YAML file.

    metadata = []
    with open(csv_file) as f:
        reader = csv.reader(f)
        for row in reader:
            biosample, assay, experiment, accession = row
            metadata.append({"biosample": biosample, "assay": assay, "experiment": experiment, "accession": accession})

    with open(yaml_file, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)


def configure_trackcsv2yaml_parser(parser):
    """Configure the parser for the trackcsv2yaml script."""
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument("yaml_file", help="Path to the output YAML file.")
    parser.set_defaults(func=csv_to_yaml)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a 4-column CSV metadata file to a YAML file.")
    configure_trackcsv2yaml_parser(parser)
    args = parser.parse_args()
    csv_to_yaml(args.csv_file, args.yaml_file)
    logger.info(f"Metadata successfully written to {args.yaml_file}")
