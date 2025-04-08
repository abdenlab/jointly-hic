### Jointly-HiC Design Document

#### 1. **Overview**
`jointly-hic` is a comprehensive Python toolkit designed for the joint embedding of Hi-C 3D chromatin contact matrices from multiple datasets into a unified vector space.
It uses incremental embedding algorithms, combined with data loading/unloading during model training to handle atlas-scale datasets using fixed amounts of memory.
The tool provides decomposition methods including as Principal Component Analysis (PCA), Non-Negative Matrix Factorization (NMF), and Singular Value Decomposition (SVD) to enable the integration and comparative analysis of chromatin structures across various biological conditions.

#### 2. **Goals**
1. **Data Integration:** Develop a tool capable of embedding multiple Hi-C datasets into a common latent space.
2. **Dimensionality Reduction:** Provide robust methods for reducing the dimensionality of large-scale Hi-C data.
3. **Visualization and Analysis:** Offer post-processing tools for visualizing and analyzing the resulting embeddings.
4. **Facilitate Research:** Enable researchers to study chromatin architecture differences across tissues, cell types, and developmental stages.

#### 3. **Modules and Features**
##### 3.1. **Data Preprocessing**
- **Data Balancing:** Use cooler's balancing tool to normalize Hi-C matrices.
- **File Format Support:** Accept `.mcool` files. For conversion options from `.hic` to `.mcool` use [hictk](https://hictk.readthedocs.io/en/latest/).

##### 3.2. **Core Decomposition**
- **Methods:** Implements PCA, NMF, and SVD for matrix decomposition.
- **Incremental Learning:** These algorithms are extensible to support incremental fitting to handle additional datasets.

##### 3.3. **Post-Processing**
- **Normalization:** Normalize embeddings across datasets to a consistent scaling.
- **Clustering:** Perform clustering using KMeans and Leiden algorithms.
- **UMAP Embedding:** Apply UMAP for dimensionality reduction and visualization.

##### 3.4. **Visualization**
- **Score Plots:** Generate plots of PCA/NMF scores, colored by clusters or filenames.
- **UMAP Plots:** Visualize UMAP embeddings with different clustering annotations.
- **Variance Analysis:** Compute and visualize variance across bins.

#### 4. **Implementation Plan**
1. **Development Phase**
   - **Core Functionality:** Develop the core decomposition and incremental learning features.
   - **Post-Processing:** Develop post-processing pipelines for clustering and visualization.
   - **Testing and Validation:** Validate the tool using synthetic and real datasets.

2. **Deployment Phase**
   - **Packaging:** Package the tool for installation via pip and Docker.
   - **Documentation:** Provide comprehensive documentation and usage examples.
   - **Community Engagement:** Encourage contributions and feedback from the research community.

#### 5. **Analysis Pipeline**
1. **Input Data Preparation**
   - Convert raw Hi-C data to `.mcool` format and balance using `cooler`.
   - Define the parameters for analysis, including resolution and decomposition method.

2. **Joint Embedding**
   - Use the selected decomposition method (PCA, NMF, SVD) to embed Hi-C contact matrices.
   - Handle large datasets incrementally to manage computational resources efficiently.

3. **Post-Processing**
   - Normalize the resulting embeddings and perform clustering using KMeans and Leiden.
   - Apply UMAP for further dimensionality reduction and visualization.

4. **Visualization and Reporting**
   - Generate plots for the embeddings and save results to disk.
   - Provide comprehensive visualizations to interpret the data, including score plots and variance analysis.

5. **Interpretation and Analysis**
    - Create a track database using ENCODE ChIP-seq data to aid interpretation.
    - Provide an interactive viewing tool to analyze results.

#### 6. **Implementation Details**
- **Programming Language:** Python
- **Core Libraries:** numpy, pandas, cooler, sklearn, umap-learn, igraph, leidenalg
- **Data Handling:** Support for large datasets using incremental methods and efficient I/O operations.
- **Scalability:** Designed to handle atlas-scale datasets through incremental fitting and parallel processing.
- **Extensibility:** Modular design to allow easy addition of new features and methods.

#### 7. **Conclusion**
`jointly-hic` is a powerful toolkit for the integrative analysis of Hi-C datasets, providing researchers with the ability to jointly embed and compare chromatin contact matrices across multiple conditions.
By offering a comprehensive suite of tools for data preprocessing, decomposition, post-processing, and visualization, `jointly-hic` enables the detailed study of chromatin architecture and its variation across different biological contexts.
The tool's development and application to large-scale datasets demonstrate its potential to facilitate new discoveries in the field of 3D genome organization.
