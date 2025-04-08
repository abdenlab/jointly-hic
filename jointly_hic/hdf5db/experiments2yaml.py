"""Script to produce a YAML file of experiment metadata from a parquet file."""

import argparse
import logging

import pandas as pd
import yaml

# Set up logging
logger = logging.getLogger("joint_pca")


def verify_metadata_consistency(df: pd.DataFrame, accession_column: str, metadata_columns: list[str]) -> list[dict]:
    """Verify that each metadata value is the same for all rows with the same accession.

    Args:
    ----
        df (pd.DataFrame): DataFrame containing the data.
        accession_column (str): Column name used as the accession identifier.
        metadata_columns (List[str]): List of other metadata columns to include.

    Returns:
    -------
        List[dict]: List of dictionaries with consistent metadata per accession.

    """
    metadata_list = []

    for accession, group in df.groupby(accession_column):
        metadata_dict = {"accession": accession}

        for column in metadata_columns:
            unique_values = group[column].unique()
            if len(unique_values) > 1:
                raise ValueError(f"Inconsistent values found in column '{column}' for accession '{accession}'.")
            metadata_dict[column] = unique_values[0]

        metadata_list.append(metadata_dict)

    return metadata_list


def write_metadata_to_yaml(metadata_list: list[dict], yaml_file: str) -> None:
    """Write metadata to a YAML file.

    Args:
    ----
        metadata_list (List[dict]): List of dictionaries containing metadata.
        yaml_file (str): Path to the output YAML file.

    """
    with open(yaml_file, "w") as file:
        yaml.dump(metadata_list, file, default_flow_style=False)


def main(parquet_file: str, accession_column: str, metadata_columns: list[str], yaml_file: str) -> None:
    """Load parquet data, verify metadata, and write to YAML.

    Args:
    ----
        parquet_file (str): Path to the parquet file containing data.
        accession_column (str): Column name used as the accession identifier.
        metadata_columns (List[str]): List of other metadata columns to include.
        yaml_file (str): Path to the output YAML file.

    """
    df = pd.read_parquet(parquet_file)
    metadata_list = verify_metadata_consistency(df, accession_column, metadata_columns)
    write_metadata_to_yaml(metadata_list, yaml_file)
    logger.info(f"Metadata successfully written to {yaml_file}")


def run_main(args):
    """Run the main function with arguments."""
    main(args.parquet_file, args.accession_column, args.metadata_columns, args.yaml_file)


def configure_embedding_metadata_parser(embedding_metadata_parser):
    """Configure the parser for the embedding2yaml script."""
    embedding_metadata_parser.add_argument(
        "--parquet-file", required=True, help="Path to the parquet file containing data."
    )
    embedding_metadata_parser.add_argument("--accession-column", required=True, help="Name of the accession column.")
    embedding_metadata_parser.add_argument(
        "--metadata-columns", nargs="+", required=True, help="List of other metadata columns."
    )
    embedding_metadata_parser.add_argument("--yaml-file", required=True, help="Path to the output YAML file.")
    embedding_metadata_parser.set_defaults(func=run_main)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Produce a YAML file of experiment metadata.")
    configure_embedding_metadata_parser(parser)
    arguments = parser.parse_args()
    run_main(arguments)
