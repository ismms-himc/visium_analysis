# Informing biologically relevant signal from spatial transcriptomic data

Deposited in bioRxiv: https://www.biorxiv.org/content/10.1101/2024.09.09.610361v1  
doi: https://doi.org/10.1101/2024.09.09.610361

# Usage and tutorials

This is an example of how you can work with Visium slides.

`visium_analysis` folder contains all the necessary code to run the downstream 
analysis on Visium slides and consists of a few Python scripts:

1. `distance.py` performs  **neighborhood** analysis. You can pick all spots at a certain 
distance from area of interest (with an increment of 100 micrometers). 
It works outside the defined area and inside, so you can assess the neighborhood, 
as well as the core of the area. You can plot cell composition of the neighborhood over the distance.
You can also assess how gene expression changes over the distance using linear regression analysis.

2. `deconvolution.py` utilizes **deconvolution** results to calculate PCA, 
UMAP and compare obtained results to results based on gene expression.
If you have structurally-resolved compartment, like immune aggregates, you can 
assess the difference between them on a slide by: 
- calculating cell composition; 
- calculating Mann-Whitney U-test for one immune aggregate vs other for every cell type; 
- calculating expression of receptor-ligand pairs inside immune aggregates.

3. `run_decoupler.py` uses [decoupler](https://doi.org/10.1093/bioadv/vbac016) 
Python library to calculate **pathways** activities from different databases such as 
[PROGENy](https://doi.org/10.1038/s41467-017-02391-6), 
[DoRothEA](https://doi.org/10.1101%2Fgr.240663.118), 
[CytoSig](https://doi.org/10.1038/s41592-021-01274-5) and 
[MSigDB](https://doi.org/10.1073/pnas.0506580102) on all Visium spots.

# Installation

We suggest using a separate python virtual environment for the package.\
You can install it using pip:  
`pip install git+https://github.com/ismms-himc/visium_analysis.git`.\
All the necessary Python packages are listed in `setup.cfg` file.
