# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [1.0.3] - Octoboer 15, 2025

### Fixed
- Python 3.10 or higher is required. Update pyproject.toml & documentation to reflect this.

---

## [1.0.2] - April 30, 2025

### Added

#### Examples
- `immune-joint-analysis`: Demonstrates the use of the `jointly-hic` toolkit for joint analysis of CD4+ and CD8+ T cell Hi-C and ChIP-seq data available via the ENCODE portal.
- `breast-tissue-joint-analysis`: Demonstrates the use of the `jointly-hic` toolkit for joint analysis of human breast tissue Hi-C data (*Choppavarapu et al.*), along with breast tissue and MCF-7 cell ChIP-seq, RNA-seq, DNase-seq, and ATAC-seq data from the ENCODE portal.

---

## [1.0.0] - April 11, 2025

### Added

#### Core Features
- Command-line interface (`jointly`) with six core subcommands:
  - `embed`: Joint decomposition of Hi-C matrices using PCA, NMF, or SVD
  - `post-process`: UMAP dimensionality reduction and k-means clustering
  - `trajectory`: Trajectory inference using k-means and UMAP projections
  - `embedding2yaml`: Extract metadata from embeddings to generate experiment YAML
  - `tracks2yaml`: Convert CSV metadata into YAML for signal track ingestion
  - `hdf5db`: Build a compressed HDF5 database (JointDb) integrating Hi-C embeddings and epigenetic signal data

#### Input & Output
- Support for multi-resolution `.mcool` Hi-C input files
- Outputs include `.parquet`, `.csv.gz`, `.pkl.gz`, `.png`, and `.h5` formats
- Compatibility with ENCODE bigwig tracks
- Configurable percentile-based filtering, chromosome restrictions, and mini-batch sizing

#### Visualization
- Generates UMAP visualizations with color-coded cluster and sample metadata
- Supports multi-neighbor UMAP and multi-k clustering in a single run

#### Integration
- `hdf5db` output for joint Hi-C + epigenomics database format (JointDb)

#### Scalability
- Out-of-core, fixed-memory matrix decomposition
- Designed for time courses, tissue atlases, and multi-condition comparisons

#### Distribution
- Published on [PyPI](https://pypi.org/project/jointly-hic)
- Available as a Docker image on [GitHub Container Registry (GHCR)](https://github.com/orgs/abdenlab/packages/container/package/jointly-hic)

#### CI/CD
- Continuous integration with test and linting checks on every PR and push to `main`
- Automatic GitHub release tagging, PyPI publication, and Docker publishing via GitHub Actions

---

## [Unreleased] - MMMM DD, YYYY

_TBD: Next milestone features will be added here as development progresses._
