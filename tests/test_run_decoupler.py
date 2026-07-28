from pathlib import Path
import squidpy as sq
import pandas as pd

from visium_analysis.run_decoupler import calc_progeny

def test_run_mlm_func():
    '''
    Predefined dataset in `data/expected_run_mlm.csv` can be obtained:

    import squidpy as sq
    from visium_analysis.run_decoupler import calc_progeny
    adata=sq.datasets.visium('Visium_FFPE_Human_Breast_Cancer')
    test=calc_progeny(adata,False).obsm['progeny_score_mlm']
    '''

    test_dir = Path(__file__).parent

    # load predefined dataset with expected results
    test=pd.read_csv(f'{test_dir}/data/expected_run_mlm.csv',index_col=0)

    # calculate results
    adata=sq.datasets.visium('Visium_FFPE_Human_Breast_Cancer')
    adata.var_names_make_unique()
    result=calc_progeny(adata,False).obsm['progeny_score_mlm']

    test=test.astype(result.dtypes)
    result.to_csv('/sc/arion/work/radkee01/mlm.csv')
    assert test.round(2).equals(result.round(2))
