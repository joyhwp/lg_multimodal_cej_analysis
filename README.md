# Multimodal CEJ Analysis via Text-Image Clustering

## Overview

This project was conducted as part of an industry-academia collaboration, analyzing customer experience data across the CEJ (Customer Experience Journey) using both text and image modalities. The goal was to uncover customer segments and satisfaction patterns that text-only analysis alone could not fully capture.

## Data

Data was collected via web crawling from publicly available sources (official brand website reviews, Naver Blog posts, and Ohouse platform reviews). Raw data files are not included in this repository due to file size and data licensing considerations; the code demonstrates the full data processing and analysis pipeline.

## Approach

**1. Initial Approach: LLM-based CEJ Mapping** (`01_llm_cej_mapping_initial_approach.ipynb`)
Text data from official brand website reviews was mapped to CEJ stages using LLM prompting.

**2. Validation** (`02_eda_cej_validation.ipynb`)
Exploratory data analysis was conducted to validate the CEJ mapping results. This step revealed that VLM-based mapping alone lacked sufficient reliability for the intended use case, prompting a shift in methodology.

**3. Pivot: Embedding-based Clustering Pipeline** (`03_embedding_clustering_pipeline.ipynb`)
After identifying reliability limitations in the initial LLM-mapping approach, the methodology was redesigned around a five-stage embedding-based clustering pipeline:

**(1) Chunking**
Naver Blog posts were split into smaller text chunks to preserve local context and improve embedding quality, rather than embedding entire posts as single units.

**(2) Embedding**
Each text chunk was embedded using Qwen3-VL-Embedding, chosen for its ability to represent both textual and visual semantics in a shared embedding space.

**(3) Text-Image Alignment Filtering**
Cosine similarity between text and paired image embeddings was computed to assess semantic alignment. Chunks in the bottom 5–10% of similarity scores (threshold: 0.25) were flagged and filtered out, removing cases where the review text and its associated image were poorly aligned (e.g., generic captions paired with unrelated product photos).

**(4) Text Clustering & CEJ Mapping**
The filtered, embedded chunks were clustered using UMAP for dimensionality reduction followed by HDBSCAN (`min_cluster_size=20`, `min_samples=5`) to identify density-based text clusters. Each resulting cluster was mapped to a corresponding CEJ stage.

**(5) Image Sub-Clustering**
Within each text cluster, images were further sub-clustered to surface visual patterns not captured by text alone. For example, within a single text cluster centered on product satisfaction (T5), image sub-clustering revealed two distinct visual profiles — one focused on the product itself (T5_I0) and another focused on the surrounding space/interior context (T5_I2) — despite both sharing similar text-level sentiment. This distinction would have been invisible under a text-only analysis.


**4. Final Analysis** (`04_cluster_insight_analysis.ipynb`)
The finalized pipeline was applied to Ohouse platform review data. Image sub-clustering within each text cluster revealed differentiated customer satisfaction profiles that were not visible from text alone (e.g., product-focused vs. space/interior-focused sub-clusters within the same text cluster).


## Tech Stack
`Python` `Qwen3-VL-Embedding` `UMAP` `HDBSCAN` `LLM Prompting` `Selenium` `pandas`
