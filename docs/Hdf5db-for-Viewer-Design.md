# Design Document: HDF5 Storage and Query System for PCA Embeddings and Metadata

## Overview

This design document outlines the implementation of a system for storing and querying PCA embeddings, metadata, and associated tracks within an HDF5 file.
The system is designed to efficiently handle large-scale genomic data, allowing for flexible storage, retrieval, and analysis.

## Objectives

1. **Store PCA embeddings, UMAP, kmeans, and Leiden embeddings in a structured and scalable manner.**
2. **Store experiment metadata, track metadata, and bin information in an HDF5 file.**
3. **Provide efficient querying capabilities for retrieving embeddings and metadata.**
4. **Ensure modular and reusable code for easy maintenance and extensibility.**

## System Architecture

### Data Storage

The data is stored in an HDF5 file with the following structure:

- **Groups:**
  - `bins/`: Contains datasets for bin information, including chromosome, start, end, and bin names.
  - `embeddings/`: Contains datasets for various embeddings such as PCA, UMAP, kmeans, and Leiden.
  - `metadata/`: Contains metadata for experiments and tracks.
  - `tracks/`: Contains datasets for tracks from bigwig files.

- **Datasets:**
  - `bins/chrom`, `bins/start`, `bins/end`, `bins/good_bin`: Store bin details.
  - `embeddings/PCA1`, `embeddings/UMAP1`, etc.: Store embeddings for each experiment.
  - `metadata/experiments`: Stores a table of metadata for experiments, with each row corresponding to an experiment.
  - `metadata/tracks`: Stores metadata for tracks, including details like assay, experiment, and biosample.
  - `tracks/ACCESSION1`, `tracks/ACCESSION2`, etc.: Store signal track data from bigwig files.

### Key Functional Components

1. **Data Ingestion**
   - **Loading YAML Metadata**: A function to load metadata from a YAML file.
   - **Writing Experiment Metadata**: Stores metadata for experiments in the HDF5 file.
   - **Writing Embeddings**: Writes embeddings from a parquet file to the HDF5 file.
   - **Adding Tracks**: Integrates track data from bigwig files into the HDF5 file.

2. **Data Querying**
   - **Retrieving Bins**: Retrieves bin information with optional filtering of bad bins.
   - **Fetching Experiment Metadata**: Retrieves metadata for experiments.
   - **Fetching Track Metadata**: Retrieves metadata for tracks.
   - **Retrieving Embeddings**: Retrieves embeddings for all or individual experiments, pivoted into tall or wide formats.
   - **Merging Metadata and Tracks**: Merges metadata from experiments and tracks, with optional filtering by assay.

3. **Main Process Flow**
   - A main function coordinates the loading of embeddings and metadata, adding tracks, and finalizing the HDF5 file.
