# Jointly Analysis of Hi-C, ATAC-seq, ChIP-seq, RNA-seq, and DNase-seq Data in Human Breast Tissues and MCF-7 Cells

## Overview
This repository provides the analysis of multi-omic data generated from **Hi-C** profiling on healthy/malignant breast tissues from human donors and **ATAC-seq**, **ChIP-seq**, **RNA-seq**, and **DNase-seq** profiling on the **MCF-7** breast cancer cell line and healthy breast tissues from human donors. 

These **ATAC-seq**, **ChIP-seq**, **RNA-seq**, and **DNase-seq** datasets are sourced from the **ENCODE** database. 

The **HiC** profiling data are from the study:

*Choppavarapu L, Fang K, Liu T, Ohihoin AG, Jin VX. Hi-C profiling in tissues reveals 3D chromatin-regulated breast tumor heterogeneity informing a looping-mediated therapeutic avenue. Cell Rep. 2025 Apr 22; 44(4):115585.*  
[DOI: 10.1016/j.celrep.2025.115450](https://doi.org/10.1016/j.celrep.2025.115450)

Using Hi-C profiling, *Choppavarapu et al.* analyzed **a cohort of 12 breast tissue samples**, including 2 normal tissues, 5 estrogen receptor-positive (ER+) primary breast tumors, and 5 tamoxifen-treated recurrent tumors.

## Scientific Background

### Hi-C
**Hi-C** is a technique that maps the 3D structure of the genome by identifying physical interactions between distant chromatin regions, allowing for the analysis of chromatin folding and genome architecture.

### ATAC-seq
**ATAC-seq** (Assay for Transposase-Accessible Chromatin with high-throughput sequencing) measures chromatin accessibility by detecting regions of open chromatin, which are crucial for transcriptional regulation.

### ChIP-seq
**ChIP-seq** (Chromatin Immunoprecipitation followed by sequencing) identifies DNA-protein interactions by capturing and sequencing regions of the genome bound by specific proteins, such as transcription factors and histones.

### RNA-seq
**RNA-seq** (RNA sequencing) quantifies gene expression levels by sequencing the RNA transcribed from the genome, providing a comprehensive view of gene activity and regulation.

### DNase-seq
**DNase-seq** detects regions of open chromatin by using the DNase I enzyme to preferentially cleave accessible DNA regions, helping to map active regulatory elements in the genome.

### Chromatin Architecture Heterogeneity in Estrogen Receptor-Positive Breast Cancer
**Breast cancer** is one of the most common and heterogeneous cancers worldwide, affecting both men and women, though it predominantly occurs in women. The disease arises from the uncontrolled growth of breast cells and can spread (metastasize) to other parts of the body. Breast cancer is classified into different subtypes based on molecular characteristics, one of the most significant being **estrogen receptor-positive (ER+)** breast cancer. ER+ breast cancer is characterized by the presence of estrogen receptors on tumor cells, making these tumors responsive to estrogen signaling, which promotes cancer cell growth. 
Endocrine therapy, such as tamoxifen, is a standard treatment for estrogen receptor-positive (ER⁺) breast cancer. However, tamoxifen resistance remains a major clinical challenge. Many factors contribute to this resistance, such as mutations in the estrogen receptor, changes in co-regulatory proteins, etc.. *Choppavarapu et al.* highlighted that although recent studies have shed light on genetic drivers and differences among breast cancers, they have primarily focused on copy number and gene expression data. To gain a more comprehensive understanding of treatment resistance and tumor diversity, *Choppavarapu et al.* set out to investigate breast cancer at the three-dimensional (3D) genome level, particularly the differences in chromatin architecture across patient tumors.

### Jointly analysis 
In the analysis of this demo, ```Jointly-HiC``` serves as the **analytical backbone** for integrating chromatin interaction data from **Hi-C** with regulatory signals from **ATAC-seq**, **ChIP-seq**, **RNA-seq**, and **DNase-seq**. It helps to identify tumor-specific structural features and correlate them with **chromatin accessibility** and **gene expression changes**.

