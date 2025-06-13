---
title: 'Jointly-HiC: A toolkit for integrative analysis of Hi-C genome architecture data'
tags:
  - Python
  - bioinformatics
  - Hi-C
  - genome architecture
  - epigenomics
authors:
  - name: Thomas M. Reimonn
    orcid: 0000-0002-5129-7989
    affiliation: 1
  - name: Vedat Yilmaz
    affiliation: 1
  - name: Guoyun Chen
    affiliation: 1
  - name: Nezar Abdennur
    orcid: 0000-0001-5814-0864
    corresponding: true
    affiliation: 1
affiliations:
 - name: Department of Genomics and Computational Biology, UMass Chan Medical School, Worcester 01655, MA, USA
   index: 1
date: 16 June 2025
bibliography: paper.bib

---

# Summary

Jointly-HiC is an open-source Python toolkit designed for the joint embedding and integrative analysis of Hi-C chromosome conformation data across multiple biosamples.
The software addresses critical limitations in current analytical methods by enabling simultaneous dimensionality reduction and decomposition of chromatin contact matrices from numerous samples, significantly improving comparative genomic analysis at scale.
Jointly-HiC efficiently integrates Hi-C data with complementary epigenetic modalities such as RNA-seq, ATAC-seq, and ChIP-seq, facilitating a unified understanding of chromatin architecture dynamics and regulatory processes across diverse biological contexts.


# Statement of need



# Mathematics



# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References








## Summary

Jointly-HiC is an open-source Python toolkit designed for the joint embedding and integrative analysis of Hi-C chromosome conformation data across multiple biosamples. The software addresses critical limitations in current analytical methods by enabling simultaneous dimensionality reduction and decomposition of chromatin contact matrices from numerous samples, significantly improving comparative genomic analysis at scale. Jointly-HiC efficiently integrates Hi-C data with complementary epigenetic modalities such as RNA-seq, ATAC-seq, and ChIP-seq, facilitating a unified understanding of chromatin architecture dynamics and regulatory processes across diverse biological contexts.

## Statement of Need

Hi-C and other chromosome conformation capture techniques have greatly advanced our understanding of the three-dimensional organization of genomes, highlighting its importance in gene regulation, cell identity, and developmental processes. Current analytical tools often perform separate decomposition of Hi-C contact matrices per sample, limiting the capacity for direct comparative analysis due to introduced normalization biases and batch effects. Furthermore, existing approaches struggle with scalability, computational memory constraints, and integration of complementary genomic datasets. Jointly-HiC addresses these limitations by providing a scalable, efficient, and integrative framework for jointly analyzing and embedding large-scale, multi-sample Hi-C datasets.

## Software Description

### Functionality

Jointly-HiC offers an end-to-end pipeline, including:

* Joint preprocessing (filtering, balancing, and normalization).
* Incremental decomposition (PCA, SVD, or NMF) performed jointly on multiple samples.
* Integration with RNA-seq, ATAC-seq, and ChIP-seq datasets through the JointDb module.
* Clustering and visualization via UMAP and Leiden clustering.
* Trajectory inference and metadata integration for developmental analyses.

### Implementation Details

Jointly-HiC is developed in Python and designed for scalability and efficiency through incremental and mini-batch algorithms. It leverages widely-used scientific computing libraries including NumPy, SciPy, scikit-learn, UMAP-learn, pandas, and h5py. The software is packaged via PyPI, supports Docker containers, and employs continuous integration workflows.

### Installation and Usage

Jointly-HiC can be easily installed via pip or Docker:

```bash
pip install jointly-hic
```

```bash
docker pull ghcr.io/abdenlab/jointly-hic
```

### Example Usage

```bash
jointly embed \
  --mcools sample1.mcool sample2.mcool \
  --resolution 50000 \
  --assembly hg38 \
  --method PCA \
  --components 32
```

Post-processing for visualization and clustering:

```bash
jointly post-process --parquet-file jointly_embeddings.pq
```

## Acknowledgments

We acknowledge the contributions of the ENCODE project and 4D Nucleome Consortium for providing high-quality datasets.

## References

* Lieberman-Aiden et al. (2009). Comprehensive mapping of long-range interactions reveals folding principles of the human genome. *Science*, 326(5950), 289-293.
* Rao et al. (2014). A 3D map of the human genome at kilobase resolution reveals principles of chromatin looping. *Cell*, 159(7), 1665-1680.
* Imakaev et al. (2012). Iterative correction of Hi-C data reveals hallmarks of chromosome organization. *Nature Methods*, 9(10), 999-1003.
* Durand et al. (2016). Juicer provides a one-click system for analyzing loop-resolution Hi-C experiments. *Cell Systems*, 3(1), 95-98.
* Buenrostro et al. (2015). ATAC-seq: A method for assaying chromatin accessibility genome-wide. *Current Protocols in Molecular Biology*, 109(1), 21-29.

## Citation

Reimonn, Thomas & Abdennur, Nezar. (2025). abdenlab/jointly-hic: Release v1.0.1. Zenodo. [https://doi.org/10.5281/zenodo.15198530](https://doi.org/10.5281/zenodo.15198530)
