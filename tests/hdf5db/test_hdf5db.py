import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import pandas as pd
import h5py
import yaml

from jointly_hic.hdf5db.hdf5db import (
    create_hdf5,
    load_bins_from_embeddings,
    store_bins,
    load_bins,
    store_experiment_metadata,
    load_experiment_metadata,
    store_embeddings,
    load_embeddings,
    add_bigwig_track,
    add_encode_tracks_to_hdf5,
    load_track_metadata,
    load_tracks,
    JointDb,
)

from jointly_hic.hdf5db.experiments2yaml import main as experiments2yaml_main

EMBEDDINGS_FILE = "test-data.pq"
TEST_EMBEDDINGS = os.path.join(os.path.dirname(__file__), EMBEDDINGS_FILE)
ACCESSION_COLUMN = "hic_accession"


@pytest.fixture(scope="function")
def test_file():
    with tempfile.NamedTemporaryFile(suffix=".yaml") as temp_file:
        temp_file_path = temp_file.name
        yield temp_file_path

    assert not os.path.exists(temp_file_path)


@pytest.fixture
def test_data():
    return pd.read_parquet(TEST_EMBEDDINGS)


@pytest.fixture
def test_metadata_yaml():
    # Define the temporary YAML file path
    with tempfile.NamedTemporaryFile(suffix=".yaml") as temp_file:
        temp_file_path = temp_file.name

        # Call the main function from the experiments2yaml script to generate the YAML file
        metadata_columns = ["biosample", "hic_accession", "filename"]  # Define your metadata columns

        # Generate the YAML file
        experiments2yaml_main(
            parquet_file=TEST_EMBEDDINGS,
            accession_column=ACCESSION_COLUMN,
            metadata_columns=metadata_columns,
            yaml_file=temp_file_path,
        )

        yield temp_file_path

    assert not os.path.exists(temp_file_path)


def test_create_hdf5(test_file):
    # Create an empty HDF5 file
    create_hdf5(test_file)
    assert os.path.exists(test_file)

    # Open the file and check if the groups exist
    with h5py.File(test_file, "r") as hdf:
        assert "bins" in hdf
        assert "embeddings" in hdf
        assert "metadata" in hdf
        assert "tracks" in hdf


def test_load_bins_from_embeddings():
    # Load bins from the test data
    bins = load_bins_from_embeddings(TEST_EMBEDDINGS)

    # Check that the dataframe is not empty and contains the expected columns
    assert not bins.empty
    assert "chrom" in bins.columns
    assert "start" in bins.columns
    assert "end" in bins.columns
    assert "good_bin" in bins.columns


def test_bin_loading(test_file, test_data):
    # Create an empty HDF5 file
    create_hdf5(test_file)

    # Store bins information into the HDF5 file
    store_bins(test_file, TEST_EMBEDDINGS)

    # Open the HDF5 file and check if the bins datasets exist
    with h5py.File(test_file, "r") as hdf:
        bins_group = hdf["bins"]
        assert "chrom" in bins_group
        assert "start" in bins_group
        assert "end" in bins_group
        assert "good_bin" in bins_group
        assert "bin_name" in bins_group

    # Load the bins from the HDF5 file
    bins_df = load_bins(test_file)

    # Check that the dataframe is not empty and contains the expected columns
    assert not bins_df.empty
    assert "chrom" in bins_df.columns
    assert "start" in bins_df.columns
    assert "end" in bins_df.columns
    assert "good_bin" in bins_df.columns
    assert "bin_name" in bins_df.columns

    # Load originals, sort by chrom, start, end, and verify columns chrom, start, end, good_bin
    original_df = test_data[["chrom", "start", "end", "good_bin"]]
    original_df = original_df.drop_duplicates().sort_values(by=["chrom", "start", "end"])
    original_df = original_df[["chrom", "start", "end", "good_bin"]]
    original_df = original_df.reset_index(drop=True)
    bins_df = bins_df.sort_values(by=["chrom", "start", "end"])
    bins_df = bins_df[["chrom", "start", "end", "good_bin"]]
    bins_df = bins_df.reset_index(drop=True)
    pd.testing.assert_frame_equal(original_df, bins_df)


def test_experiment_metadata(test_file, test_metadata_yaml):
    # Create an empty HDF5 file
    create_hdf5(test_file)

    # Store experiment metadata
    store_experiment_metadata(test_file, test_metadata_yaml)

    # Open the HDF5 file and check if the experiment metadata is stored
    with h5py.File(test_file, "r") as hdf:
        metadata_group = hdf["metadata"]
        assert "experiment_metadata" in metadata_group

    # Load the experiment metadata
    metadata_df = load_experiment_metadata(test_file)

    # Check that the dataframe is not empty and contains the expected columns
    assert not metadata_df.empty
    assert "accession" in metadata_df.columns
    assert "biosample" in metadata_df.columns

    # Load original data from YAML for comparison
    with open(test_metadata_yaml) as file:
        original_metadata_list = yaml.safe_load(file)
    original_metadata_df = pd.DataFrame(original_metadata_list).astype(str)

    # Compare the loaded data with the original data
    pd.testing.assert_frame_equal(original_metadata_df, metadata_df)


def test_store_embeddings(test_file, test_metadata_yaml, test_data):
    # Create an empty HDF5 file
    create_hdf5(test_file)

    # Store experiment metadata
    store_experiment_metadata(test_file, test_metadata_yaml)

    # Store bins
    store_bins(test_file, TEST_EMBEDDINGS)

    # Store embeddings
    store_embeddings(test_file, TEST_EMBEDDINGS, ACCESSION_COLUMN)

    # Open the HDF5 file and check if the embeddings datasets exist
    with h5py.File(test_file, "r") as hdf:
        embeddings_group = hdf["embeddings"]
        # Verify if the embeddings datasets are created
        for column in test_data.columns:
            if (
                "PCA" in column
                or "NMF" in column
                or "SVD" in column
                or "umap" in column
                or "leiden" in column
                or "kmeans" in column
            ):
                assert column in embeddings_group


def test_load_embeddings(test_file, test_metadata_yaml, test_data):
    # Create an empty HDF5 file
    create_hdf5(test_file)

    # Store experiment metadata
    store_experiment_metadata(test_file, test_metadata_yaml)

    # Store bins
    store_bins(test_file, TEST_EMBEDDINGS)

    # Store embeddings
    store_embeddings(test_file, TEST_EMBEDDINGS, ACCESSION_COLUMN)

    # Load experiment metadata and bins
    experiment_metadata = load_experiment_metadata(test_file)
    bins = load_bins(test_file)

    # Define embeddings to load
    embeddings_to_load = ["PCA1", "PCA2"]

    # Load the embeddings
    embeddings_df = load_embeddings(test_file, embeddings_to_load, experiment_metadata, bins)

    # Check that the dataframe is not empty and contains the expected columns
    assert not embeddings_df.empty
    assert "accession" in embeddings_df.columns
    assert "bin_name" in embeddings_df.columns
    assert "PCA1" in embeddings_df.columns
    assert "PCA2" in embeddings_df.columns

    # Verify the content of the loaded embeddings matches the stored data
    original_data = test_data[["hic_accession", "chrom", "start", "end", *embeddings_to_load]].rename(
        columns={"hic_accession": "accession"}
    )
    original_data["bin_name"] = [
        f"{chrom}:{start}-{end}"
        for chrom, start, end in zip(original_data["chrom"], original_data["start"], original_data["end"])
    ]
    original_data = original_data.sort_values(["accession", "bin_name"]).set_index(["accession", "bin_name"])
    original_data = original_data[embeddings_to_load]
    print(original_data)
    loaded_data = embeddings_df.sort_values(["accession", "bin_name"]).set_index(["accession", "bin_name"])[
        embeddings_to_load
    ]
    print(loaded_data)

    pd.testing.assert_frame_equal(original_data, loaded_data)

    # Test JointDb access
    db = JointDb(test_file)
    embeddings = db.get_embeddings(embeddings_to_load)
    pd.testing.assert_frame_equal(embeddings, embeddings_df)


class MockEncodeFile:
    """Mock class to simulate the opening of a file."""

    def __init__(self, accession, filetype):
        self.accession = accession
        self.filetype = filetype

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def mock_get_signal(bigwig, bins_df):
    return np.random.rand(len(bins_df))


@pytest.fixture
def mock_track_metadata():
    data = [
        {"accession": "track1", "assay": "H3K27ac", "experiment": "exp1", "biosample": "cell1"},
        {"accession": "track2", "assay": "H3K4me3", "experiment": "exp2", "biosample": "cell2"},
    ]
    return pd.DataFrame(data)


def test_bigwig_tracks(test_file, test_data):
    # Create an empty HDF5 file
    create_hdf5(test_file)

    # Store bins
    store_bins(test_file, TEST_EMBEDDINGS)
    bins = load_bins(test_file)

    # Mock signal data
    signal = pd.DataFrame({"signal": np.random.rand(bins.shape[0])})
    accession = "track1"
    assay = "H3K27ac"
    experiment = "exp1"
    biosample = "cell1"

    # Add bigwig track
    add_bigwig_track(test_file, signal, accession, assay, experiment, biosample)

    # Check if the track and metadata are stored correctly
    with h5py.File(test_file, "r") as hdf:
        tracks_group = hdf["tracks"]
        assert accession in tracks_group
        loaded_data = np.array(tracks_group[accession][:], dtype=float).reshape(-1)
        expected_data = np.array(signal["signal"].values, dtype=float).reshape(-1)
        assert np.allclose(loaded_data, expected_data)

        metadata_group = hdf["metadata/track_metadata"]
        track_group = metadata_group[accession]
        assert track_group.attrs["assay"] == assay
        assert track_group.attrs["experiment"] == experiment
        assert track_group.attrs["biosample"] == biosample

    # Load the track signal data
    bins_df = load_bins(test_file)
    track_signals = load_tracks(test_file, [accession], bins_df, drop_bad_bins=False)
    loaded_signal = track_signals[accession].values

    # Assert signal matches loaded track_signal column
    assert np.allclose(loaded_signal, expected_data)


@patch("jointly_hic.hdf5db.hdf5db.EncodeFile", new=MockEncodeFile)
@patch("jointly_hic.hdf5db.hdf5db.get_signal", new=mock_get_signal)
def test_add_encode_tracks_to_hdf5(test_file, mock_track_metadata):
    # Create an empty HDF5 file
    create_hdf5(test_file)

    # Store bins (mock data)
    store_bins(test_file, TEST_EMBEDDINGS)

    # Add mock tracks and their metadata
    add_encode_tracks_to_hdf5(mock_track_metadata, test_file)

    # Verify that the tracks and their metadata are correctly added
    with h5py.File(test_file, "r") as hdf:
        tracks_group = hdf["tracks"]
        metadata_group = hdf["metadata/track_metadata"]

        for track in mock_track_metadata.itertuples():
            assert track.accession in tracks_group
            track_group = metadata_group[track.accession]
            assert track_group.attrs["assay"] == track.assay
            assert track_group.attrs["experiment"] == track.experiment
            assert track_group.attrs["biosample"] == track.biosample

    # Test JointDb accession
    db = JointDb(test_file)
    track_metadata = db.track_metadata
    assert track_metadata["accession"].tolist() == mock_track_metadata["accession"].tolist()


@patch("jointly_hic.hdf5db.hdf5db.EncodeFile", new=MockEncodeFile)
@patch("jointly_hic.hdf5db.hdf5db.get_signal", new=mock_get_signal)
def test_load_track_metadata(test_file, mock_track_metadata):
    # Create an empty HDF5 file
    create_hdf5(test_file)

    # Store bins (mock data)
    store_bins(test_file, TEST_EMBEDDINGS)

    # Add mock tracks and their metadata
    add_encode_tracks_to_hdf5(mock_track_metadata, test_file)

    # Load the track metadata
    loaded_metadata = load_track_metadata(test_file)

    # Check that the metadata matches the original mock data
    pd.testing.assert_frame_equal(
        loaded_metadata.sort_values(by="accession").reset_index(drop=True),
        mock_track_metadata.sort_values(by="accession").reset_index(drop=True),
    )


def test_jointdb_initialization(test_file):
    # Create an empty HDF5 file
    create_hdf5(test_file)
    joint_db = JointDb(test_file)
    assert joint_db.hdf5_file == test_file


def test_jointdb_bins(test_file, test_data):
    # Create an empty HDF5 file and store bins
    create_hdf5(test_file)
    store_bins(test_file, TEST_EMBEDDINGS)

    # Initialize JointDb and access bins
    joint_db = JointDb(test_file)
    bins_df = joint_db.bins

    # Check if the bins data matches the test data
    original_bins = (
        test_data[["chrom", "start", "end", "good_bin"]].drop_duplicates().sort_values(by=["chrom", "start", "end"])
    )
    original_bins.reset_index(drop=True, inplace=True)
    bins_df = bins_df.sort_values(by=["chrom", "start", "end"]).reset_index(drop=True)[
        ["chrom", "start", "end", "good_bin"]
    ]
    pd.testing.assert_frame_equal(original_bins, bins_df)


def test_jointdb_experiment_metadata(test_file, test_metadata_yaml):
    # Create an empty HDF5 file and store experiment metadata
    create_hdf5(test_file)
    store_experiment_metadata(test_file, test_metadata_yaml)

    # Initialize JointDb and access experiment metadata
    joint_db = JointDb(test_file)
    experiment_metadata_df = joint_db.experiment_metadata

    # Load original data from YAML for comparison
    with open(test_metadata_yaml) as file:
        original_metadata_list = yaml.safe_load(file)
    original_metadata_df = pd.DataFrame(original_metadata_list).astype(str)

    # Compare the loaded data with the original data
    pd.testing.assert_frame_equal(original_metadata_df, experiment_metadata_df)


def test_jointdb_get_embeddings(test_file, test_metadata_yaml, test_data):
    # Create an empty HDF5 file, store metadata, bins, and embeddings
    create_hdf5(test_file)
    store_experiment_metadata(test_file, test_metadata_yaml)
    store_bins(test_file, TEST_EMBEDDINGS)
    store_embeddings(test_file, TEST_EMBEDDINGS, ACCESSION_COLUMN)

    # Initialize JointDb and get embeddings
    joint_db = JointDb(test_file)
    embeddings_to_load = ["PCA1", "PCA2"]
    embeddings_df = joint_db.get_embeddings(embeddings_to_load)

    # Check that the dataframe is not empty and contains the expected columns
    assert not embeddings_df.empty
    assert "accession" in embeddings_df.columns
    assert "bin_name" in embeddings_df.columns
    assert "PCA1" in embeddings_df.columns
    assert "PCA2" in embeddings_df.columns

    # Verify the content of the loaded embeddings matches the stored data
    original_data = test_data[["hic_accession", "chrom", "start", "end", *embeddings_to_load]].rename(
        columns={"hic_accession": "accession"}
    )
    original_data["bin_name"] = [
        f"{chrom}:{start}-{end}"
        for chrom, start, end in zip(original_data["chrom"], original_data["start"], original_data["end"])
    ]
    original_data = original_data.sort_values(["accession", "bin_name"]).set_index(["accession", "bin_name"])
    original_data = original_data[embeddings_to_load]
    loaded_data = embeddings_df.sort_values(["accession", "bin_name"]).set_index(["accession", "bin_name"])[
        embeddings_to_load
    ]

    pd.testing.assert_frame_equal(original_data, loaded_data)
