import os

import pandas as pd
import pytest
from click.testing import CliRunner

from jointly_hic.notebook_utils.bed9_conversion import mark_runs, merge_runs, create_bed9_clusters


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2", "chr2", "chr1"],
            "start": [100, 200, 100, 200, 300],
            "end": [150, 250, 150, 250, 350],
            "value": [1, 2, 3, 4, 5],
        }
    )


def test_mark_runs_no_overlap(sample_dataframe):
    result_df = mark_runs(sample_dataframe, "value")
    assert "run" in result_df.columns


def test_merge_runs(sample_dataframe):
    merged_df = merge_runs(sample_dataframe, "value")
    assert len(merged_df) == len(sample_dataframe)


@pytest.fixture
def input_data_csv(tmpdir):
    data = """chrom,start,end,value,filename
chr1,100,150,1,file1
chr1,200,250,2,file1
chr2,100,150,3,file2
chr2,200,250,4,file2
chr1,300,350,5,file1"""
    file_path = tmpdir.join("input.csv")
    file_path.write(data)
    return str(file_path)


@pytest.fixture
def runner():
    return CliRunner()


def test_create_bed9_clusters_csv_input(runner, input_data_csv, tmpdir):
    output_prefix = str(tmpdir.join("output"))
    result = runner.invoke(
        create_bed9_clusters, ["-i", input_data_csv, "-o", output_prefix, "-c", "value", "-n", "filename"]
    )

    assert result.exit_code == 0
    assert os.path.exists(f"{output_prefix}_file1.bed.gz")
    assert os.path.exists(f"{output_prefix}_file2.bed.gz")


def test_unsupported_file_format(runner, tmpdir):
    unsupported_file = tmpdir.join("input.xyz")
    unsupported_file.write("dummy data")
    output_prefix = str(tmpdir.join("output"))

    result = runner.invoke(create_bed9_clusters, ["-i", str(unsupported_file), "-o", output_prefix])

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
