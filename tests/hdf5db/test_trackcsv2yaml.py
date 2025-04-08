import argparse

import pytest
import yaml

from jointly_hic.hdf5db.trackcsv2yaml import csv_to_yaml


def test_csv_to_yaml(tmp_path):
    # Prepare a mock CSV input
    csv_content = """biosample1,assay1,experiment1,accession1
biosample2,assay2,experiment2,accession2
biosample3,assay3,experiment3,accession3"""
    csv_file = tmp_path / "input.csv"
    csv_file.write_text(csv_content)

    # Define the expected YAML output
    expected_metadata = [
        {"biosample": "biosample1", "assay": "assay1", "experiment": "experiment1", "accession": "accession1"},
        {"biosample": "biosample2", "assay": "assay2", "experiment": "experiment2", "accession": "accession2"},
        {"biosample": "biosample3", "assay": "assay3", "experiment": "experiment3", "accession": "accession3"},
    ]

    # Define the output YAML file
    yaml_file = tmp_path / "output.yaml"

    # Call the function
    args = argparse.Namespace(csv_file=csv_file, yaml_file=yaml_file)
    csv_to_yaml(args)

    # Read and assert the YAML output
    with open(yaml_file, "r") as file:
        content = yaml.safe_load(file)

    assert content == expected_metadata


def test_csv_to_yaml_empty_file(tmp_path):
    # Prepare an empty CSV file
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("")

    # Define the output YAML file
    yaml_file = tmp_path / "empty_output.yaml"

    # Call the function
    args = argparse.Namespace(csv_file=csv_file, yaml_file=yaml_file)
    csv_to_yaml(args)

    # Read and assert the YAML output
    with open(yaml_file, "r") as file:
        content = yaml.safe_load(file)

    assert content == []


def test_csv_to_yaml_incorrect_format(tmp_path):
    # Prepare a CSV file with incorrect format (missing one column)
    csv_content = """biosample1,assay1,experiment1
biosample2,assay2"""
    csv_file = tmp_path / "incorrect.csv"
    csv_file.write_text(csv_content)

    # Define the output YAML file
    yaml_file = tmp_path / "incorrect_output.yaml"

    # Call the function and expect it to raise an error
    with pytest.raises(ValueError):
        # Call the function
        args = argparse.Namespace(csv_file=csv_file, yaml_file=yaml_file)
        csv_to_yaml(args)
