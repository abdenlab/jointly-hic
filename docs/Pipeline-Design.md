### Methods Summary for Jointly-HIC

#### Overview
The Jointly-HIC toolkit facilitates the analysis of Hi-C 3D chromatin contact matrices by embedding them into a unified vector space.
This process involves data preprocessing, decomposition using PCA, NMF, or SVD, and extensive post-processing, including normalization, clustering, and visualization.
Below is a detailed explanation of the steps involved in the pipeline.

#### 1. **Data Preprocessing**
- **Input Data Preparation:** The input data consists of `.mcool` files, which must be balanced using the `cooler` tool. These files are expected to contain genome-wide Hi-C contact frequency matrices.
- **Chromosome Sizes:** The sizes of chromosomes are fetched using the `bioframe` library based on the specified genome assembly (e.g., hg38, mm10). This information is crucial for accurately partitioning the contact matrices.
- **Union of Bad Bins:** A union of "bad bins" is computed across all input files. Bad bins are those with NaN values in their weights, potentially due to noise or low coverage. If an exclusion list is provided, bins overlapping these regions are also marked as bad.

#### 2. **Core Decomposition**
- **Model Initialization:** Depending on the configuration, an incremental model is selected:
  - **IncrementalPCA:** For PCA-based decomposition.
  - **MiniBatchNMF:** For non-negative matrix factorization.
  - **SparseIncrementalSVD:** For SVD-based decomposition.
- **Matrix Preprocessing:** For each `.mcool` file, the Hi-C matrix is loaded and preprocessed. This involves computing a cis-masked, balanced, affinity matrix and removing bad bins. The result is cached to disk for efficiency.
- **Incremental Fitting:** The preprocessed matrices are used for incremental fitting of the model. The matrix is divided into minibatches, and each batch is partially fitted to the model, allowing the processing of large datasets.
- **Embedding Computation:** Once the model is fully trained, it is used to compute the embeddings for each dataset. These embeddings represent the low-dimensional projections of the Hi-C contact matrices in a unified space.

#### 3. **Post-Processing**
- **Normalization:** Embeddings are normalized by dataset to a consistent overall norm, ensuring comparability.
- **Clustering:** Clustering is performed using KMeans and Leiden algorithms:
  - **KMeans Clustering:** Applied to the embeddings with a specified number of clusters, identifying major groupings within the data.
  - **Leiden Clustering:** Uses a k-nearest neighbors graph to identify communities within the data, offering a resolution-adjustable clustering method.
- **UMAP Embedding:** UMAP (Uniform Manifold Approximation and Projection) is applied to further reduce the dimensionality of the embeddings. This aids in the visualization of high-dimensional data.
- **Variance Analysis:** The variance of bins across all samples in the component space is computed, providing insights into the stability and variability of chromatin interactions.

#### 4. **Visualization and Output**
- **Score Plots:** Visualize the PCA/NMF scores for each sample, colored by clusters or filenames. This helps in understanding the data distribution and cluster separations.
- **UMAP Plots:** Display UMAP embeddings, highlighting the clustering of samples and the relationships between different datasets.
- **Output Files:** The results are saved in various formats, including:
  - **Parquet and CSV:** Contain the embeddings and clustering results.
  - **Model File:** The trained model is saved as a pickle file for future use.

#### 5. **Implementation Details**
- **Programming Language:** Python, utilizing libraries such as `numpy`, `pandas`, `cooler`, `sklearn`, `umap-learn`, and `leiden`.
- **Incremental Learning:** Key to handling large datasets, allowing the model to learn from data in chunks.
- **Extensibility:** The modular design allows for the addition of new decomposition methods, clustering algorithms, and visualization tools.

This pipeline ensures a comprehensive and scalable approach to analyzing Hi-C datasets, enabling the study of chromatin architecture across different biological conditions.
The results facilitate the understanding of genome organization and its variation across tissues and developmental stages.
