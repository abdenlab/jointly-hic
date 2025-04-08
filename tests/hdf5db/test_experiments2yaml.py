import pytest
import pandas as pd
import yaml

from jointly_hic.hdf5db.experiments2yaml import verify_metadata_consistency, write_metadata_to_yaml, main


def test_verify_metadata_consistency():
    # Test with consistent metadata
    data = {
        "accession": ["acc1", "acc1", "acc2", "acc2"],
        "metadata1": ["value1", "value1", "value2", "value2"],
        "metadata2": ["value3", "value3", "value4", "value4"],
    }
    df = pd.DataFrame(data)
    result = verify_metadata_consistency(df, "accession", ["metadata1", "metadata2"])
    expected = [
        {"accession": "acc1", "metadata1": "value1", "metadata2": "value3"},
        {"accession": "acc2", "metadata1": "value2", "metadata2": "value4"},
    ]
    assert result == expected

    # Test with inconsistent metadata
    data = {
        "accession": ["acc1", "acc1", "acc2", "acc2"],
        "metadata1": ["value1", "value1", "value2", "value5"],
        "metadata2": ["value3", "value3", "value4", "value4"],
    }
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="Inconsistent values found in column 'metadata1' for accession 'acc2'"):
        verify_metadata_consistency(df, "accession", ["metadata1", "metadata2"])


def test_write_metadata_to_yaml(tmp_path):
    metadata_list = [
        {"accession": "acc1", "metadata1": "value1", "metadata2": "value3"},
        {"accession": "acc2", "metadata1": "value2", "metadata2": "value4"},
    ]
    yaml_file = tmp_path / "output.yaml"
    write_metadata_to_yaml(metadata_list, yaml_file)

    with open(yaml_file, "r") as file:
        content = yaml.safe_load(file)

    assert content == metadata_list


# Mock data for end-to-end test
def test_main_end_to_end(tmp_path, monkeypatch):
    data = {
        "accession": ["acc1", "acc1", "acc2", "acc2"],
        "metadata1": ["value1", "value1", "value2", "value2"],
        "metadata2": ["value3", "value3", "value4", "value4"],
    }
    df = pd.DataFrame(data)
    parquet_file = tmp_path / "input.parquet"
    df.to_parquet(parquet_file)

    yaml_file = tmp_path / "output.yaml"

    def mock_read_parquet(file):
        return df

    monkeypatch.setattr(pd, "read_parquet", mock_read_parquet)
    main(parquet_file, "accession", ["metadata1", "metadata2"], yaml_file)

    with open(yaml_file, "r") as file:
        content = yaml.safe_load(file)

    expected = [
        {"accession": "acc1", "metadata1": "value1", "metadata2": "value3"},
        {"accession": "acc2", "metadata1": "value2", "metadata2": "value4"},
    ]

    assert content == expected
