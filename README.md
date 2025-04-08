# Jointly-HIC

![logo](./logo.png)

Welcome to `jointly-hic`, a Python tool for jointly embedding Hi-C 3D chromatin contact matrices into the same vector space.
Whether you're a researcher, developer, or enthusiast, this toolkit is designed to help you integrate and analyze multi-sample Hi-C datasets efficiently and effectively.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Input Data](#input-data)
- [Output](#output)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Hi-C data analysis is a powerful assay to probe chromatin organization and 3D genome structure.
`jointly-hic` facilitates the integration of multiple Hi-C datasets by jointly embedding them into the same vector space using Principal Component Analysis (PCA), Non-Negative Matrix Factorization (NMF), or Singular Value Decomposition (SVD).
The tool also includes a post-processing and visualization pipeline to help you make sense of the results.

## Installation

### Install with pip
To install `jointly-hic` using `pip`, follow these steps:

```bash
# Clone the source code
git clone https://github.com/abdenlab/jointly-hic.git

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the package
pip install .

# Install development and notebook dependencies if you develop with this project or run the notebooks
pip install -e '.[dev, notebook]'
```

### Run with Docker
Everything required to run this tool is included in the Docker image.

```bash
# Pull the Docker image
docker pull treimonn/jointly-hic:latest
```

## Input Data
To use `jointly-hic`, follow these steps:

1. Prepare your Hi-C data in the form of `.mcool` files. If your data is in `.hic` format, you can use the `hictk` [Hi-C toolkit](https://hictk.readthedocs.io/en/latest/) to convert it to `.mcool` format.
2. Balance the Hi-C data using `cooler balance` from [cooler](https://cooler.readthedocs.io/en/latest/cli.html#cooler-balance).
3. Run `jointly-hic` with the appropriate parameters (see below).

Each `.mcool` file should contain Hi-C contact matrices for a specific sample or condition.

## Usage
The usage options for the `jointly-hic` tool are specified in the command-line interface (CLI) as follows:

```
usage: jointly embed [-h] --mcools MCOOLS [MCOOLS ...] --resolution RESOLUTION --assembly {hg38,mm10,hg19,mm9}
                     [--output OUTPUT] [--components COMPONENTS] [--chrom-limit CHROM_LIMIT] [--method {PCA,NMF,SVD}]
                     [--exclusion-list EXCLUSION_LIST] [--percentile-top PERCENTILE_TOP]
                     [--percentile-bottom PERCENTILE_BOTTOM] [--batch-size BATCH_SIZE]

options:
  -h, --help            show this help message and exit
  --mcools MCOOLS [MCOOLS ...]
                        Path to mcool file(s) containing Hi-C contact matrices
  --resolution RESOLUTION
                        Bin size to use for contact frequency matrix
  --assembly {hg38,mm10,hg19,mm9}
                        Genome assembly name used for alignment (e.g. hg38, mm10)
  --output OUTPUT       Prefix for output files (Default: output_<datetime>)
  --components COMPONENTS
                        Number of components to use for PCA (Default: 32)
  --chrom-limit CHROM_LIMIT
                        Limit number of chromosomes to load (Example: '23' to limit to human chr1-chrX and exclude
                        chrY and chrM).
  --method {PCA,NMF,SVD}
                        Method to use for decomposition, either PCA, NMF or SVD (Default: PCA)
  --exclusion-list EXCLUSION_LIST
                        File containing regions to exclude from analysis
  --percentile-top PERCENTILE_TOP
                        Top percentile for filtering (Default: 99.5)
  --percentile-bottom PERCENTILE_BOTTOM
                        Bottom percentile for filtering (Default: 1)
  --batch-size BATCH_SIZE
                        Batch size for mini-batches (Default: 10000)
```

## Output
The output of the `jointly-hic` tool includes a set of files that contain the results of the analysis. The files are saved with the prefix specified by the `--output` option.

### Data Files
- `prefix_embeddings.pq`: Contains the embeddings of the Hi-C data saved as a parquet file.
- `prefix_embeddings.csv.gz`: Contains the embeddings of the Hi-C data saved as a CSV file.
- `prefix_model.pkl.gz`: Contains the trained PCA or NMF model used for the decomposition.
- `prefix_log.txt`: Contains the log of the analysis, including the parameters used and the time taken for each step.
- `prefix_post_processed.csv.gz`: Contains post-processing results, including UMAP embedding and clustering labels, saved as a CSV.
- `prefix_post_processed.pq`: Contains post-processing results, including UMAP embedding and clustering labels, saved as a parquet file.
- `prefix_trajectories.csv.gz`: Contains the trajectory analysis results saved as a CSV file.
- `prefix_trajectories.pq`: Contains the trajectory analysis results saved as a parquet file.

### Plots
- `prefix_scores.png`: A plot of the PCA or NMF scores for each sample.
- `prefix_scores_clustered.png`: A plot of the PCA or NMF scores for each sample, with points colored by cluster.
- `prefix_scores_filenames.png`: A plot of the PCA or NMF scores for each sample, with points colored by filename.
- `prefix_umap-n##_clustered.png`: A plot of the UMAP embedding for each sample, with points colored by cluster.
- `prefix_umap-n##_filenames.png`: A plot of the UMAP embedding for each sample, with points colored by filename.
- `prefix_trajectory_umap-n##_kmeans.png`: A plot of the trajectory analysis UMAP embedding for each sample, with points colored by k-means cluster.

## Contributing
We welcome contributions to this project!
If you have any suggestions or questions, please feel free to open an issue or submit a pull request.
Whether it's fixing bugs, adding new features, or improving documentation, your contributions are highly valued.

## License
This project is licensed under the GNU GPL (version 3). See the [LICENSE](LICENSE) file for details.

---

Thank you for your interest in `jointly-hic`!
We hope this tool aids your research and helps you uncover new insights into chromatin organization and 3D genome structure.
Happy analyzing!
