# visium_analysis

[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.1101%2F2024.09.09.610361-B31B1B.svg)](https://doi.org/10.1101/2024.09.09.610361)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

> **Informing biologically relevant signal from spatial transcriptomic data**

`visium_analysis` is a Python tool for downstream processing of Visium spatial transcriptomics data. It enables neighborhood distance analysis, cell-type deconvolution evaluation, and pathway activity scoring across spatial spots.

---

## Reference and Links

* **Author:** Emir Radkevich
* **Preprint:** [bioRxiv (2024.09.09.610361)](https://www.biorxiv.org/content/10.1101/2024.09.09.610361v1) | **DOI:** [10.1101/2024.09.09.610361](https://doi.org/10.1101/2024.09.09.610361)
* **Publication:** Used in a 2026 *Science* [publication](https://science.org).
* **Claude skill:** [Link](https://github.com/emir-radkevich/claude-skills/tree/main/visium-gradient-analysis) to the skill.
* **Tutorials & Extensions:** Example notebooks are available below and pending pull request to [`squidpy_notebooks`](https://github.com/emir-radkevich/squidpy_notebooks/blob/add-neighborhood-notebook/tutorials/tutorial_neighborhood.ipynb).

--- 

## Usage and Workflow

The `visium_analysis/` directory contains the core scripts to perform downstream analysis on Visium spatial transcriptomics slides:

### 1. Spatial Neighborhood Analysis (`distance.py`)
Analyzes cell distribution and gene expression dynamics as a function of radial distance from a region of interest (in 100 um increments, moving both inward and outward):
* **Cell Composition:** Maps changes in cell-type composition from the core to surrounding neighborhoods.
* **Expression Dynamics:** Runs linear regression to track how gene expression changes across spatial distance gradients.

### 2. Deconvolution and Structural Comparison (`deconvolution.py`)
Uses deconvolution results to compute PCA and UMAP embeddings, comparing structural compartments (e.g., distinct immune aggregates) by:
* Calculating cell-type proportion differences.
* Computing **Mann-Whitney U-tests** per cell type across compartments.
* Assessing expression of key receptor-ligand pairs within aggregates.

### 3. Pathway Activity Scoring (`run_decoupler.py`)
Leverages [decoupler](https://github.com/saezlab/decoupler-py) to infer footprint-based activity scores across Visium spots using established reference databases:
* **PROGENy** (Pathway activities)
* **DoRothEA** (Transcription factor activities)
* **CytoSig** (Cytokine signaling)
* **MSigDB** (Molecular signatures)

## Installation

We suggest using a separate conda environment for the package:  

```
conda create --name va python=3.12
conda activate va
pip install git+https://github.com/ismms-himc/visium_analysis.git
```  

Or it can be installed in a python virtual environment:  
```
mkdir venvs && cd venvs
python3.12 -m venv va
source va/bin/activate
pip install git+https://github.com/ismms-himc/visium_analysis.git
```    

All the necessary Python packages are listed in `setup.cfg` file.
