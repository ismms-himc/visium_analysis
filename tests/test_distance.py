import numpy as np
import pandas as pd

from visium_analysis.distance import gen_spot_circle, calc_lm

def test_gen_spot_func():
    spot=np.array([0,1])
    spot_scalefactor=100
    n=1
    x,y,r=gen_spot_circle(spot,spot_scalefactor,n)
    assert r==100

def test_calc_lm_func():
    '''
    Predefined dataset in `data/test_calc_lm.csv` can be obtained:
    
    import squidpy as sq
    from visium_analysis.distance import calc_all_neighbors, gen_data_for_lm
    adata=sq.datasets.visium_hne_adata_crop()
    ad=calc_all_neighbors(adata=adata,
                          cat='cluster',
                          celltype='Fiber_tract',
                          n=4,
                          sample_obs_key=None,
                          samples=None,
                          noncumulative=True,
                          add_intra=True,
                          intra_only=False)
    calc_lm_df=gen_data_for_lm(ad,'Sox17','')
    '''

    # load predefined dataset with expected results
    calc_lm_df=pd.read_csv('data/test_calc_lm.csv',index_col=0)
    pv_expected=\
    [3.770258657226368e-09,
     0.20055055318735338,
     0.7892467284945289,
     0.636560347915213,
     0.7397360658409451]
    regression_expected=pd.read_csv('data/expected_calc_lm.csv',index_col=0)['0']\
    .values.tolist()
    # round all floats
    pv_expected_round=[round(pv,5) for pv in pv_expected]
    regression_expected_round=[round(regr,5) for regr in regression_expected]
   
    # calculate results 
    r=calc_lm(data=calc_lm_df, 
              use_mixedlm=False, 
              return_regression_coef=True)
    pv_result=r[0].copy()
    regression_result=r[1].copy()
    # round all floats
    pv_result_round=[round(pv,5) for pv in pv_result]
    regression_result_round=[round(regr,5) for regr in regression_result]

    assert pv_expected_round==pv_result_round
    assert regression_expected_round==regression_result_round
