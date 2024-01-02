# Usage and tutorials

This is an example of how you can work with visium slides.

`visium_example.ipynb` file contains an example of Python Notebook 
for visium slides analysis.

1. First block is called **neighborhood**. You can pick all spots at a certain 
distance from area of interest. It works outside the defined area and inside, so
you can assess core of the area. It also plots the composition of neighborhood.
You can assess how gene expression changes over the distance using regression.

2. Second block, **deconvolution**, utilizes deconvolution results, specifically 
matrix from [cell2location](https://doi.org/10.1038/s41587-021-01139-4), though it
can be any tool, to calculate PCA, UMAP and compare to gene expression results.
If you have nicely structurally-resolved compartment, like immune aggregates, you can 
assess the difference between them by: 
- calculating composition, 
- calculating U-test for one immune aggregate vs other for every cell subtype, 
- calculating cell-cell interaction inside immune aggregates and 
comparing top expressed receptor-ligand pairs.

3. Third block, **decoupler**, calculates [PROGENy](https://doi.org/10.1038/s41467-017-02391-6), 
[DoRothEA](https://doi.org/10.1101%2Fgr.240663.118), 
[CytoSig](https://doi.org/10.1038/s41592-021-01274-5) and 
[MSigDB](https://doi.org/10.1073/pnas.0506580102) collections on all spots.

# Installation

We suggest using a separate python virtual environment  
for the package. You can install it using pip:  
`pip install git+https://github.com/ismms-himc/visium_analysis.git`.
