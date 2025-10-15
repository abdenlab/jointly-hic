# Jointly-HiC

[![PyPI](https://img.shields.io/pypi/v/jointly-hic)](https://pypi.org/project/jointly-hic/)
[![Docker Image](https://img.shields.io/badge/ghcr.io-abdenlab%2Fjointly--hic-blue)](https://github.com/abdenlab/jointly-hic/pkgs/container/jointly-hic)
[![CI](https://github.com/abdenlab/jointly-hic/actions/workflows/python-pytest.yaml/badge.svg)](https://github.com/abdenlab/jointly-hic/actions)
[![DOI](https://zenodo.org/badge/962766794.svg)](https://zenodo.org/badge/latestdoi/962766794)

Welcome to `jointly-hic`, a Python tool for jointly embedding Hi-C 3D chromatin contact matrices into the same vector space.
This toolkit is designed to help you analyze multi-sample Hi-C datasets efficiently and integrate epigenetic data (ATAC-seq, RNA-seq, ChIP-seq) effectively.

![graphics](./jointly-graphical-abstract.png)

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Input Data Preparation](#input-data-preparation)
  - [Embedding](#embedding)
  - [Post-processing](#post-processing)
  - [Trajectory Inference](#trajectory-inference)
  - [Metadata Integration](#metadata-integration)
  - [Joint HDF5 Database](#joint-hdf5-database)
- [Output](#output)
- [Contributing](#contributing)
- [CI/CD and Release Process](#cicd-and-release-process)
- [License](#license)

## Introduction

The three-dimensional organization of the genome plays a critical role in regulating gene expression and establishing cell identity.
Hi-C and related chromosome conformation capture technologies have enabled genome-wide profiling of chromatin contacts, revealing compartmental domains and long-range interactions that orchestrate regulatory programs across development and disease.
However, as the scale and diversity of Hi-C datasets grow—from tissue atlases to time courses and in vitro differentiation models—there remains a lack of computational tools that can integrate dozens to hundreds of Hi-C experiments into a unified analytical space while preserving biological signal and enabling comparative analyses.

Jointly-HiC is a scalable, Python-based toolkit for the joint analysis of Hi-C data across many biosamples.
It enables efficient pre-processing, fixed-memory joint decomposition using incremental principal component analysis (PCA), singular value decomposition (SVD), or non-negative matrix factorization (NMF), and downstream clustering and visualization.
Designed with scalability and interpretability in mind, jointly-hic extracts low-dimensional embeddings that can be used to identify shared and sample-specific chromatin interaction profiles across conditions, cell types, and developmental stages.
By operating in a mini-batched fashion, it avoids the memory constraints of traditional matrix factorization techniques, making it suitable for large-scale studies such as tissue atlases or differentiation trajectories.

In addition to providing core Hi-C analysis workflows, jointly-hic integrates seamlessly with other epigenomic data through the optional JointDb module, a compressed HDF5 database format supporting ChIP-seq, RNA-seq, and ATAC-seq signal tracks at the same resolution.
This unified framework enables users to analyze compartmentalization dynamics, correlate chromatin interactions with regulatory activity, and discover structural genome features such as nuclear speckle-associated regions or heterochromatic domains.
With jointly-hic, researchers can uncover patterns of 3D genome organization at scale, shedding light on the structural underpinnings of gene regulation and chromatin state transitions across biological contexts.

## Installation

Python version 3.10 or higher is required. `jointly-hic` can be installed via `pip`. A pre-built docker image is available on `GHRC`. The package  can also be installed from source.

### Install from PyPI

```bash
# Requires python version >= 3.10
pip install jointly-hic
```

### Run with Docker

```bash
docker pull ghcr.io/abdenlab/jointly-hic
```

### Install from source (for development)

```bash
git clone https://github.com/abdenlab/jointly-hic.git
cd jointly-hic
python3 -m venv venv
source venv/bin/activate
pip install -e '.[dev,notebook]'
```

## Usage

You can run `jointly` from the command line to embed, post-process, analyze trajectories, or create metadata and databases from Hi-C matrices.

To get help on available subcommands:

```bash
jointly -h
```

### Input Data Preparation

1. Prepare your Hi-C data as `.mcool` files, binned and balanced at your planned analysis resolution. (We recommend using: [Distiller for alignment](https://github.com/open2c/distiller-nf), [cooler for pre-processing](https://github.com/open2c/cooler), and [hictk for file conversion](https://github.com/paulsengroup/hictk))
2. Balance your data using `cooler balance`.
3. (Optional) Create metadata CSV or YAML files with ENCODE accessions of experiment metadata and signal tracks (examine the example notebooks for more information).

### Embedding

The primary compute module of `jointly-hic` is through `embed`.
This will take a list of input `mcool` files and create a joint decomposition using the provided method, resolution, genome and number of components.
`jointly-embed` will run the post-processing and trajectory modules with default parameters, which is good for many use cases.
The output files are vertically stacked tables of bins, embeddings, clustering and UMAP visualizations for all samples, stacked on top of each other.
Some useful plots, logs, and information will be printed and saved.

```bash
jointly embed \
  --mcools sample1.mcool sample2.mcool \
  --resolution 50000 \
  --assembly hg38 \
  --method PCA \
  --components 32
```

### Post-processing

Post-processing is usually performed as part of the `embed` pipeline, but can also be run separately if necessary.

```bash
jointly post-process \
  --parquet-file jointly_embeddings.pq \
  --umap-neighbours 30 100 500 \
  --kmeans-clusters 5 10 15
```

### Trajectory Analysis

Trajectory analysis is usually performed as part of the `embed` pipeline, but can also be run separately if necessary.

```bash
jointly trajectory \
  --parquet-file jointly_embeddings.pq
  --kmeans-clusters 5 10 15
```

### JointDb Database

Part of `jointly-hic` is the `JointDb` database module, a powerful way to integrate embeddings from `jointly embed` with ChIP-seq, ATAC-seq, RNA-seq or other epigenetic signal tracks.
This requires extensive metadata, and we recommend examining the example notebooks and hdf5db source code for more information.
Creation of a `JointDb` database requires 1) Experiment Metadata in YAML format and 2) (Optional) ENCODE track metadata in YAML format.
Use `embedding2yaml` to extract experiment metadata from the post processed embeddings.

```bash
jointly embedding2yaml \
  --parquet-file jointly_embeddings_updated.pq \
  --accession-column hic_accession \
  --metadata-columns condition stage \  # Assuming these have been added to jointly_embeddings_updated.pq
  --yaml-file experiments.yaml
```

Then use `tracks2yaml` to convert a CSV table of ENCODE metadata to YAML format.

```bash
jointly tracks2yaml track_meta.csv tracks.yaml
```

Finally, create the `JointDb`.

```bash
jointly hdf5db \
  --experiments experiments.yaml \
  --tracks tracks.yaml \
  --embeddings jointly_embedding_embeddings.pq \
  --accession sample_id \
  --output jointly_output.h5
```

## Output

The output of the `jointly-hic` tool includes a set of files that contain the results of the analysis. The files are saved with the prefix specified by the `--output` option.

### Data Files
- `*_post_processed.pq` / `*_post_processed.csv.gz`: Rescaled embeddings, clustering and visualization table. (This is the main output)
- `*_embeddings.pq` and `*_embeddings.csv.gz`: Raw Hi-C embeddings.
- `*_model.pkl.gz`: Trained sklearn decomposition model (PCA, NMF or SVD).
- `*_log.txt`: Execution log.
- `*_trajectories.pq` / `*_trajectories.csv.gz`: Trajectory analysis results.
- `*jointly_output.h5`: HDF5 database of all embeddings, metadata, and track info (if using `hdf5db`).

### Plots
- Component score plots: `*_scores.png`, `*_scores_clustered.png`, `*_scores_filenames.png`
- UMAP plots: `*_umap-n##_clustered.png`, `*_umap-n##_filenames.png`
- Trajectory UMAP: `*_trajectory_umap-n##_kmeans.png`

## Contributing

We welcome contributions to this project!
If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

### Setup (Development)

```bash
git clone https://github.com/abdenlab/jointly-hic.git
cd jointly-hic
pip install -e '.[dev,notebook]'
pre-commit install
```

### Running Tests and Linting

```bash
ruff check jointly_hic
pytest --cov=jointly_hic --cov-report=term-missing tests
```

## CI/CD and Release Process

We use GitHub Actions for continuous integration and deployment.

### Workflow Summary

1. **CI Tests**: On every push or pull request to `main`, we run tests and linting (`python-pytest.yaml`).
2. **Versioning + Release**: If tests pass, the version is pulled from `jointly_hic/__init__.py`, and a GitHub release is created (`auto-release.yaml`).
3. **Package Publishing**:
   - The `pypi-publish.yaml` workflow builds and publishes the package to [PyPI](https://pypi.org/project/jointly-hic).
   - The `docker-publish.yaml` workflow builds a Docker image and pushes it to [GHCR](https://github.com/orgs/abdenlab/packages/container/package/jointly-hic).

## License

This project is licensed under the GNU GPL (version 3). See the [LICENSE](LICENSE) file for details.

## Citation

Please cite this work if you use it:

```
Reimonn, Thomas & Abdennur, Nezar. (2025). abdenlab/jointly-hic: Release v1.0.1 (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.15198530
```

---

Thank you for your interest in `jointly-hic`. We hope this tool aids your research and helps you uncover new insights into chromatin organization and 3D genome structure.
