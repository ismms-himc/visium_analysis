import scanpy as sc
import pandas as pd

from visium_analysis.run_decoupler import calc_progeny

def test_run_mlm_func():
    '''
    Predefined dataset in `data/expected_run_mlm.csv` can be obtained:

    import squidpy as sq
    from visium_analysis.run_decoupler import calc_progeny
    adata=sq.datasets.visium('Visium_FFPE_Human_Breast_Cancer')
    test=calc_progeny(adata,False).obsm['progeny_mlm_estimate']
    '''

    # load predefined dataset with expected results
    test=pd.read_csv('data/expected_run_mlm.csv',index_col=0)

    # calculate results
    adata=sc.read_h5ad('data/test_run_mlm.h5ad')
    result=calc_progeny(adata,False).obsm['progeny_mlm_estimate']

    test=test.astype(result.dtypes)
    assert test.round(2).equals(result.round(2))
