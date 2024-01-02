import tqdm
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from typing import List,Tuple,Union,Optional

import statsmodels.api as sm
from scipy.stats import linregress

plt.rcdefaults()
sc.set_figure_params()
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (5,5)

def gen_spot_circle(spot: np.array,
                    spot_scalefactor: float,
                    n: int) -> Tuple[float,float,float]:
    # spot_scalefactor is the theoritcal distance between centers of spots
    assert len(spot)==2
    spot_distance=spot_scalefactor*n
    error=0
    center_x,center_y=spot[0],spot[1]
    radius=spot_distance+error
    return center_x,center_y,radius

def gen_neighborhood(n: int, 
                     test_spots: np.array,
                     test_spots_bool: np.array,
                     all_spot_coords: np.array,
                     spot_scalefactor: float) -> np.array:
    '''
    Return a np.array of [x, y] coordinates of neighbors 
    within a certain distance.

    Parameters:
    -----------
    n : int
        The number of layers of interest. 
        For example, if n=2, it includes spots 
        that are two layers away from test spots.
    test_spots : np.array
        [x, y] coordinates of spots of interest.
    test_spots_bool: np.array
        bool array of spots of interest.
    all_spot_coords : np.array
        [x, y] coordinates of all spots.
    spot_scalefactor: float
        ratio of spot_diameter_fullres in .uns and 65 micrometers.

    Returns:
    --------
    np.array
        bool array of neighbors within a certain distance.
    '''

    if n == 0:
        return test_spots_bool

    ngh=np.empty((0,len(all_spot_coords)), dtype=bool)
    for spot in test_spots:
        center_x, center_y, radius = gen_spot_circle(spot,spot_scalefactor,n)
        dist = np.square(all_spot_coords[:, 0]-center_x) \
        + np.square(all_spot_coords[:, 1]-center_y)
        filt_coords=dist<=radius**2
        filt_coords=filt_coords & ~test_spots_bool
        ngh = np.append(ngh, [filt_coords], axis=0)
    neighborhood=np.any(ngh, axis=0)
    
    return neighborhood

def gen_neighborhood_nc(adata: anndata.AnnData,
                        n: int,
                        test_spots: np.array,
                        test_spots_bool: np.array,
                        all_spot_coords: np.array,
                        spot_scalefactor: float) -> np.array:
    '''
    Returns list of [x,y] coordinates of noncumulative (one layer) neighbors.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData with one visium image
    n : int
        Number of layers of interest. For example, 2.
    test_spots : np.array
        [x, y] coordinates of spots of interest.
    test_spots_bool: np.array
        bool array of spots of interest.
    all_spot_coords : np.array
        [x, y] coordinates of all spots.
    spot_scalefactor: float                                          
        ratio of spot_diameter_fullres in .uns and 65 micrometers.

    --------
    np.array
        bool array of noncumulative (one layer) neighbors.
    '''
    
    if n==0: 
        add_neighborhood2obs(adata,f'{n}_c',test_spots_bool)
        return test_spots_bool
    
    # check if cumulative neighborhood at this distance is already calculated
    try:
        len(adata.uns['neighborhood'][f'neighborhood_{n}_c'])!=0
        print(f'Cumulative neighborhood at the {n*100} micrometers '\
        'distance is already calculated.')
        cur_calc=adata.uns['neighborhood'][f'neighborhood_{n}_c']
        cur_neighbors=(cur_calc=='neighbor').values
    except:
        print(f'Cumulative neighborhood at the {n*100} micrometers '\
        'distance is not calculated. Running it now.')
        cur_neighbors=gen_neighborhood(n,test_spots,
                                       test_spots_bool,
				       all_spot_coords,
				       spot_scalefactor)
        add_neighborhood2obs(adata,f'{n}_c',cur_neighbors)
    
    prev_calc=adata.uns['neighborhood'][f'neighborhood_{n-1}_c']
    previous_neighbors=(prev_calc=='neighbor').values
    r=cur_neighbors & ~previous_neighbors
    
    return r
    
def clc_ngh(adata: anndata.AnnData,
            cat: str,
            celltype: str,
            n: int,
            noncumulative: bool=True,
            return_adata: bool=False,
            ) -> Union[anndata.AnnData,None]:
    '''
    Calculate neighborhood of specified group of spots.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData with one visium image
    cat : str
        Category in .obs you want to test. 
        For example 'pathologist_annotaion' or 'singler'.
    celltype : str
        Celltype in cat. For example 'Immune cells'.
    n : int
        Number of layers of interest. For example 2.
    noncumulative : bool, optional (Default: `True`)
        Whether to calculate noncumulative (True) 
        or cumulative (False) neighborhood.
    return_adata : bool, optional (Default: `False`)
        Whether to return adata.
    
    IMPORTANT: when noncumulative=True, both neighborhoods are calculated.
    Until cumulative neighborhood at `n` is not already calculated.
    
    Returns:
    --------
    anndata.AnnData
        if `return_adata=True` returns AnnData with .uns['neighborhood'] 
        for all n for cumulative and noncumulative neighborhoods. 
        Otherwise, return None.
    '''
                        
    start=0
    
    # check if there are previous runs
    # if yes, start from the latest computed layer
    if noncumulative: 
        typ='_nc'
        try:
            last_calc=\
            sorted(adata.uns['neighborhood'].columns[
                adata.uns['neighborhood'].columns.str.contains('_nc')
            ])[-1]
            start=int(last_calc.split('_')[1])+1
        except: pass
    else: 
        typ='_c'
        try:
            last_calc=\
            sorted(adata.uns['neighborhood'].columns[
                (adata.uns['neighborhood'].columns.str.startswith('neighborhood'))\
                 & (adata.uns['neighborhood'].columns.str.contains('_c'))
            ])[-1]
            start=int(last_calc.split('_')[1])+1
        except: pass

    # generating and adding neighborhoods to adata
    test_spots=np.array(adata[adata.obs[cat]==celltype].obsm['spatial'])
    test_spots_bool=(adata.obs[cat]==celltype).values
    all_spot_coords=np.array(adata.obsm['spatial'])

    # check if sample has this category in annotation
    if len(test_spots_bool[test_spots_bool==True])==0: 
        print(f'Sample does not have {celltype} in {cat}.')
        return adata

    # get the distance between spots centers
    sample_name_uns=list(adata.uns['spatial'].keys())[0]
    spot_scalefactor=\
    adata.uns['spatial'][sample_name_uns]['scalefactors']['fiducial_diameter_fullres']

    for j in tqdm.tqdm(range(start,n+1)):
        if noncumulative:
            neighborhood=gen_neighborhood_nc(adata,j,test_spots,
                                             test_spots_bool,
                                             all_spot_coords,
					     spot_scalefactor)
        else: neighborhood=gen_neighborhood(j,test_spots,
                                            test_spots_bool,
                                            all_spot_coords,
					    spot_scalefactor)

        # check when to stop if you reach the end of the slide
        if len(neighborhood[neighborhood==True])==0: 
            print(f'{j}th layer is empty. Stopping the calculation.')    
            break

        add_neighborhood2obs(adata,f'{j}{typ}',neighborhood)
    
    if return_adata: return adata
        
def calc_neighborhood(adata: anndata.AnnData,
                      cat: str,
                      celltype: str,
                      n: int,
                      sample_obs_key: Union[str,None]=None,
                      samples: Union[list,None]=None,
                      noncumulative: bool=True) -> anndata.AnnData:
    '''
    Calculate the neighborhood of a specified group of spots 
    for one or more samples in an AnnData.
    This function is a wrapper of the `clc_ngh` function.

    Parameters:
    -----------
    adata : anndata.AnnData
        An AnnData object with one or more visium images.
    cat : str
        A category in `.obs` that you want to test. 
        For example, "pathologist_annotaion" or "singler".
    celltype : str
        A cell type in `cat`. For example, "Immune cells".
    n : int
        The number of layers of interest. For example, 2.
    sample_obs_key : str, optional (Default: `None`)
        The `.obs` key where the sample names are stored. 
        If `None`, AnnData contains one sample.
    samples : list, optional (Default: `None`)
        A list of the specific samples you want to process 
        if you have several images in the AnnData. 
        By default, all the samples in AnnData are processed.
    noncumulative : bool, optional (default: `True`)
        Whether to calculate the noncumulative (True) 
        or cumulative (False) neighborhood.

    IMPORTANT: When `noncumulative=True`, both neighborhoods are calculated 
    until the cumulative neighborhood at the distance `n` is already calculated.

    Returns:
    --------
    anndata.AnnData
        An updated AnnData with calculated `.uns['neighborhood'] 
        with columns ['neighborhood_*_sample_name']`
        for >1 samples for both cumulative and noncumulative neighborhoods. 
        For AnnData with one sample, it returns `.uns['neighborhood'] 
        with columns ['neighborhood_*']`.
    '''

    if sample_obs_key==None: 
        return clc_ngh(adata,cat,celltype,n,noncumulative,return_adata=True)
    
    if samples: samples_list=samples
    else: samples_list=adata.obs[sample_obs_key].unique()

    r=[]
    uns_r=[]
    for sample in samples_list:
        
        adatatmp=adata[adata.obs[sample_obs_key]==sample]
        adatatmp.uns['spatial']={
            s:adatatmp.uns['spatial'][s] \
            for s,v in adatatmp.uns['spatial'].items() \
            if s==sample
        }
        print(f'Running {sample} sample')
        adatatmp=clc_ngh(adatatmp,cat,celltype,n,noncumulative,return_adata=True)
        try:
            adatatmp.uns['neighborhood'].columns=[
                i+f'_{sample}' \
                for i in adatatmp.uns['neighborhood'].columns
            ]
            uns_r.append(adatatmp.uns['neighborhood'])
        except: pass
        r.append(adatatmp)

    r=sc.concat(r,join='outer',uns_merge='unique')
    r.uns['neighborhood']=\
    pd.concat(uns_r).reindex(r.obs_names).fillna('not neighbor')
    return r

def gen_intraneighborhood(adata: anndata.AnnData,
                          test_spots: np.array,
                          test_spots_bool: np.array,
                          n: int,
                          sample: Union[str,None]=None) -> np.array:
    '''
    Returns numpy array of [x,y] coordinates of intra-neighbors, 
    nth layer of area of interest itself.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData with one visium image.
    test_spots : np.array
        [x,y] coordinates of spots of interest.
    test_spots_bool: np.array
        bool array of spots of interest.
    n : int
        Number of layers of interest. For example 2.
    sample : str, optional (default: `None`)
        Sample name.
    
    Returns:
    --------
    np.array
        bool array of intra-neighbors.
    '''
    
    all_spot_coords=np.array(adata.obsm['spatial'])
    
    if sample is None: suffix=''
    else: suffix=f'_{sample}'
    
    # start from the nth layer from the outer layer
    if n==0: return test_spots_bool
    if n>1:
        # collect previous neighbors
        prev_spots=np.empty((0,len(adata.obs_names)), dtype=bool)
        for i in range(1,n):
            i=f'-{i}'
            prev_row=\
            (adata.uns['neighborhood']\
             [f'neighborhood_{i}_intra{suffix}']=='neighbor').values
            prev_spots=np.append(prev_spots, [prev_row], axis=0)
        prev_spots=np.any(prev_spots, axis=0)
        # subtract previous neighbors from current ones
        test_spots_bool=test_spots_bool & ~prev_spots
            
    # if you eventually got to the core on the previous step
    # do not calculate further
    if len(test_spots_bool[test_spots_bool==True])==0: return []

    test_spots=np.array(adata.obsm['spatial'])[test_spots_bool]

    # get the distance between spots centers
    sample_name_uns=list(adata.uns['spatial'].keys())[0]
    spot_scalefactor=\
    adata.uns['spatial'][sample_name_uns]['scalefactors']['fiducial_diameter_fullres']

    neighborhood=np.empty(0, dtype=int)
    for spot in test_spots:
        center_x, center_y, radius = gen_spot_circle(spot, spot_scalefactor, 1)
        # test each spot vs `test spots`
        dist = np.square(test_spots[:, 0]-center_x) \
        + np.square(test_spots[:, 1]-center_y)
        # sum all neighbors from `test spots`
        neighborhood=\
        np.append(neighborhood, 
                  np.sum(np.less(dist,radius**2)))
    # if spot has less than 6 neighbors from the `test_spots`
    # than this spot at the border. we write 7, because
    # the spot of interest itself is counted also in `test_spots`
    neighborhood=np.where(neighborhood<7, True, False)
    # write results to all-spots-length array
    true_indices = np.where(test_spots_bool)[0]
    newngh = test_spots_bool.copy()
    newngh[true_indices] = neighborhood

    return newngh

def clc_intrangh(adata: anndata.AnnData,
                 cat: str,
                 celltype: str,
                 n: int,
                 return_adata: bool=False,
                 sample: Union[str,None]=None) -> Union[anndata.AnnData,None]:
    '''
    Calculate intraneighborhood of specified group of spots.
    Intra-neighbors are spots of nth layer of area of interest itself.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData with one visium image
    cat : str
        Category in .obs you want to test. 
        For example 'pathologist_annotaion' or 'singler'.
    celltype : str 
        Celltype in `cat`. For example 'Immune cells'.
    n : int 
        Number of layers of interest. For example 2.
    return_adata : bool, optinonal (Default: `False`) 
        Whether to return adata. 
    sample : str, optional (Default: `None`)
        Sample name.
    
    Returns:
    --------
    anndata.AnnData 
        If `return_adata==True` return AnnData 
        with .uns['neighborhood'] ith columns ['neighborhood_*_intra'] for all n.
    '''
                        
    start=0
    
    # check if there are previous runs
    # if yes, start from the latest computed layer
    if sample is None: typ='_intra'
    else: typ=f'_intra_{sample}'
    try:
        last_calc=sorted(adata.uns['neighborhood'].columns[
            adata.uns['neighborhood'].columns.str.contains('_intra')
        ])[-1]
        start=int(last_calc.split('_')[1])+1
    except: pass

    # generating and adding neighborhoods to adata
    test_spots=np.array(adata[adata.obs[cat]==celltype].obsm['spatial'])
    test_spots_bool=(adata.obs[cat]==celltype).values

    # check if sample has this category in annotation
    if len(test_spots_bool[test_spots_bool==True])==0: 
        print(f'Sample does not have {celltype} in {cat}.')
        return adata

    for j in tqdm.tqdm(range(start,n+1)):
        neighborhood=gen_intraneighborhood(adata,test_spots,
                                           test_spots_bool,j,sample)
        # check when to stop
        # if you reach core in the previous step
        if len(neighborhood)==0: 
            print(f'-{j}th layer is empty. Previous one was already the core. '\
            'Stopping the calculation.')
            break
        if j!=0: j=f'-{j}'
        add_neighborhood2obs(adata,f'{j}{typ}',neighborhood)
    
    if return_adata: return adata
        
def calc_intraneighborhood(adata: anndata.AnnData,
                           cat: str,
                           celltype: str,
                           n: int,
                           sample_obs_key: Union[str,None]=None,
                           samples: Union[List,None]=None
                          ) -> anndata.AnnData:
    '''
    Calculate intraneighborhood of specified group of spots 
    for one or more samples in AnnData object.
    Intra-neighbors are spots of nth layer of area of interest itself.
    Wrapper of `clc_intrangh` function.
    
    Parameters:
    -----------
    adata : anndata.AnnData 
        AnnData with one or more visium image.
    cat : str 
        Category in .obs you want to test. 
        For example 'pathologist_annotaion' or 'singler'.
    celltype : str 
        Celltype in `cat`. For example 'Immune cells'.
    n : int 
        Number of layers of interest. For example 2.
    sample_obs_key : str, optional (Default is `None`)
        .obs name where sample names are stored.
    samples : list, optional (Default is `None`)
        Names of specific samples you want to plot in AnnData. 
        Default is to plot all the samples in AnnData.
    
    Returns:
    --------
    anndata.AnnData 
        AnnData with calculated .uns['neighborhood']
        with columns ['neighborhood_*_intra_{sample_name}'] 
        for >1 samples. columns ['neighborhood_*_intra'] for AnnData with one sample.
    '''

    if sample_obs_key==None: 
        return clc_intrangh(adata,cat,celltype,n,return_adata=True)
    
    if samples: samples_list=samples
    else: samples_list=adata.obs[sample_obs_key].unique()

    r=[]
    uns_r=[]
    for sample in samples_list:
        
        adatatmp=adata[adata.obs[sample_obs_key]==sample]
        adatatmp.uns['spatial']={
            s:adatatmp.uns['spatial'][s] \
            for s,v in adatatmp.uns['spatial'].items() \
            if s==sample
        }
        print(f'Running {sample} sample.')
        adatatmp=clc_intrangh(adatatmp,cat,celltype,
                              n,return_adata=True,sample=sample)
        try: uns_r.append(adatatmp.uns['neighborhood'])
        except: pass
        r.append(adatatmp)

    r=sc.concat(r,join='outer',uns_merge='unique')
    r.uns['neighborhood']=\
    pd.concat(uns_r).reindex(r.obs_names).fillna('not neighbor')
    return r

def calc_all_neighbors(adata: anndata.AnnData, 
                       cat: str, 
                       celltype: str,
                       n: int, 
                       sample_obs_key: Union[str,None] = None, 
                       samples: Union[List,None] = None,
                       noncumulative: bool = True,
                       add_intra: bool = True,
                       intra_only: bool = False) -> anndata.AnnData:
    '''
    Calculates neighborhood and intra-neighborhood.
    Wrapper of `calc_neighborhood` and `calc_intraneighborhood`.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData with or more visium image.
    cat : str
        Category in .obs you want to test. 
        For example 'pathologist_annotaion' or 'singler'.
    celltype : str
        Celltype in `cat`. For example 'Immune cells'.
    n : int
        Number of layers of interest. For example 2.
    sample_obs_key : str, optional (Default is `None`)
        .obs name where sample names are stored.
    samples : list, optional (Default is `None`)
        Names of specific samples you want to plot in AnnData. 
        Default is to plot all the samples in AnnData.
    noncumulative : bool, optional (Default: `True`)
        Whether to calculate the noncumulative (True) 
        or cumulative (False) neighborhood.
    add_intra : bool, optional (Default: `True`)
        Whether to add intra-neighborhood in calculations.
    intra_only : bool, optional (Default: `False`)
        Whether to calculate intra-neighborhood only.
    
    Returns:
    --------
    anndata.AnnData
        An updated AnnData with calculated `.uns['neighborhood']` 
        with columns ['neighborhood_*_sample_name'] 
        for >1 samples where `*` means cumulative/noncumulative neighborhoods 
        and/or intra-neighbors. 
        For AnnData with one sample, it returns `.uns['neighborhood']`
        with columns ['neighborhood_*'].
    '''
    
    if intra_only:
        ad2=calc_intraneighborhood(adata,cat,celltype,
                                   n,sample_obs_key,samples)
        return ad2
    
    else:
        ad1=calc_neighborhood(adata,cat,celltype,n,sample_obs_key,
                              samples,noncumulative)
        if add_intra:
            ad2=calc_intraneighborhood(adata,cat,celltype,
                                       n,sample_obs_key,samples)
            try:
                for i in ad2.uns['neighborhood'].columns:
                    if 'intra' in i:
                        ad1.uns['neighborhood'][i]=ad2.uns['neighborhood'][i]
            except: pass
        return ad1
    
def add_neighborhood2obs(adata: anndata.AnnData,
                         n: int,
                         neighborhood: np.array) -> None:
    '''
    Add neighborhood information to the observation metadata of an AnnData.

    Parameters:
    -----------
    adata : anndata.AnnData
        The object to which the neighborhood information is to be added.
    n : int
        The index of the neighborhood.
    neighborhood : np.array
        bool array of the neighborhood.

    Returns:
    --------
    None.
    '''

    series_to_add=\
    pd.Series(index=adata.obs_names,                                   
              data=np.where(neighborhood,'neighbor','not neighbor'))   
    
    if 'neighborhood' not in adata.uns.keys():
        adata.uns['neighborhood']=pd.DataFrame(index=adata.obs_names)
    else: pass
    adata.uns['neighborhood'][f'neighborhood_{n}']=series_to_add

    print(f'Annotation of neighbors is added '\
    f'in adata.uns[\'neighborhood\'], column \'neighborhood_{n}\'')

    
# code for boxplots in plot_funcs

def prepare_ngh_composition(adata: anndata.AnnData,
                            feature_mtx: pd.DataFrame,
                            noncumulative: bool=True,
                            sample: Union[str,None]=None,
                            subtypes: list=None,
                            intra: bool=False) -> Tuple[list,pd.DataFrame]:
    '''
    Calculates neighborhood composition till specified distance.
    
    Parameters:
    -----------
    adata : anndata.Anndata 
        AnnData
    feature_mtx : pd.DataFrame 
        pd.DataFrame with barcodes in rows and features in columns.
        For example, deconvolution or gene expression matrix.
    noncumulative : bool, optional (Default is `True`)
        Whether to plot noncumulative (True) or cumulative (False) neighborhood.
    sample : str, optional (Default is `None`)
        Sample name. 
        Is used only in AnnData with several images.
    subtypes : list, optional (Default is `None`)
        list of cell types you want to visualise.
    intra : bool, optional (Default is `False`)
        Whether to add intra neighborhood to the plot.
    
    Returns:
    --------
    Tuple:
        - list: List of subtypes to show
        - pd.DataFrame: pd.DataFrame with composition at different distances
    '''
    
    # get sample name
    if sample is None: sample_name=''
    else: sample_name=f'_{sample}'
    
    # get all distances
    all_dists=set()
    all_dists.update([i.split('_')[1] \
                      for i in adata.uns['neighborhood'].columns \
                      if (i.startswith('neighbor')) \
                      and (sample_name in i)])
    all_dists=sorted(list(map(int,all_dists)))

    # check if celltype is not in sample
    if len(all_dists)==0:
        return [], pd.DataFrame()
    
    # get all types
    types=[]
    if intra: types.append('_intra')
    if noncumulative: types.append('_nc')
    else: types.append('_c')
    
    if subtypes is not None:
        sbtps=subtypes
    else:    
        # picks top 8 subtypes in the 0 layer
        if '_intra' in types: typ='_intra'
        else: typ='_c'
        sbtps=\
        feature_mtx.reindex(adata\
                            [adata.uns['neighborhood'].reindex(adata.obs_names)\
                             [f'neighborhood_0{typ}{sample_name}']=='neighbor']\
                             .obs_names
                           ).sum().sort_values(ascending=False).index[:8].tolist()
        
    dst=[]
    existing_dists=[]
    i_old=1000
    for typ in types:
        for i in all_dists:
            # check that this distance exists
            # it is meaningful because of intra-neighbors 
            try:
                neighbor_index=\
                adata[adata.uns['neighborhood'].reindex(adata.obs_names)[
                    f'neighborhood_{i}{typ}{sample_name}'
                ]=='neighbor']\
                .obs_names
                
                # silly check for two zeros
                if i==i_old: continue
                i_old=i
                
                existing_dists.append(i)
            except: continue
            dst.append(feature_mtx.reindex(neighbor_index).reindex(sbtps,axis=1))
    dst_df=pd.concat(dst,axis=1,keys=[i for i in existing_dists])
    
    return sbtps,dst_df

# code for assessing gene expression across the distance down there

def get_series_w_distances(adata: anndata.AnnData,
                           sample: str) -> pd.Series:
    '''
    Return pd.Series with spot names and distance category.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData with one or more images.
    sample : str
        Sample name. If neighborhood was calculated on AnnData 
        with one sample without .obs specifying 'samples', pass '' here. 
        If you select one sample from AnnData, pass sample name.
        
    Returns:
    --------
    pd.Series
        pd.Series with spot names and distance category.
    '''
    
    if sample!='': sample=f'_{sample}'
    # get all distances
    a=sorted([
        int(i.split('_')[1]) for i in adata.uns['neighborhood'].columns \
        if (sample in i) and ('neighbor' in i) \
        and ('neighborhood_0' not in i) \
        and ('nc' in i or 'intra' in i)
    ])

    # check if sample does not have calculated distances
    if len(a)==0: return pd.Series()
    
    # get series with spots and distance category
    r=[]
    for i in a:
        if i<0: typ='_intra'
        else: typ='_nc'
        ixs=\
        adata[adata.uns['neighborhood'].reindex(adata.obs_names)[
            f'neighborhood_{i}{typ}{sample}']=='neighbor'
             ].obs_names
        r.append(pd.Series(index=ixs,data=[i]*len(ixs)))
    r=pd.concat(r)
    
    # replace all distances that have <10 spots 
    # to distances+1
    to_rep=r.value_counts()[r.value_counts()<10].index
    for i in to_rep: r=r.replace({i:i+1})
    
    return r

def get_goi(adata: anndata.AnnData,
            sample: str,
            nonzero_counts: float) -> list:
    '''
    Return list of genes of interest (goi).
    GOIs are genes that has nonzero counts in >70 % of cells
    (by default).
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData with one or more images.
    sample : str
        Sample name. If neighborhood was calculated on AnnData 
        with one sample without .obs specifying 'samples', pass '' here. 
        If you select one sample from AnnData, pass sample name.
        
    Returns:
    --------
    list
        list of genes of interest.
    '''
    
    r=get_series_w_distances(adata,sample)

    # check if sample does not have calculated distances
    if len(r)==0: return []

    adata=adata[r.index]
    # get genes that has nonzero counts in >70 % of cells (default)
    nonzero=adata[:,adata.to_df().sum()!=0].var_names
    nonzero_filtered=\
    adata[:,nonzero].to_df().apply(lambda x: (x!=0).sum()/len(r))
    goi=nonzero_filtered[nonzero_filtered>nonzero_counts].index
    return goi

def gen_data_for_lm(adata: anndata.AnnData,
                    gene: str,
                    sample: str,
                    use_mixedlm: bool=False,
                    samples: Union[List,None]=None)->pd.DataFrame:
    '''
    Generates pd.DataFrame with gene expression across distances.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData with one image.
    gene : str
        Gene name.
    sample : str
        Sample name.
    use_mixedlm : bool, optional (Default: `False`)
        Whether to use MixedLM for samples with different conditions.
        For example, you have 10 samples, some of them responders, some of them
        are non-responders.
        WARNING: it is not implemented yet.
    samples : list, optional (Default: `None`)
        List of sample names you want to test. 
        By default calculates on all samples in AnnData.
        
    Return:
    -------
    pd.DataFrame
        Long dataframe with gene expression across distances.
    '''
    
    if use_mixedlm:
        # get sample names
        if samples is None: 
            sample_list=\
            set([i.split('_')[-1] 
                 for i in adata.uns['neighborhood'].columns 
                 if i.startswith('neighborhood')])
        else:
            sample_list=samples

        fin_df=[]
        for sample in samples:
            r=get_series_w_distances(adata,sample)
            data=adata[r.index,gene].to_df()
            data['distance']=r.values
            data['groups']=[sample]*data.shape[0]
            data=data.reset_index()
            data=data.melt(id_vars=['index','distance','groups'])
            fin_df.append(data)
        fin_df=pd.concat(fin_df)
        fin_df['groups']=fin_df['groups'].astype('category')
        fin_df['distance']=fin_df['distance'].astype('category')
        return fin_df
    else:
        r=get_series_w_distances(adata,sample)
        data=adata[r.index,gene].to_df()
        data['distance']=r.values
        data=data.reset_index()
        data=data.melt(id_vars=['index','distance'])
        return data

def calc_lm(data: pd.DataFrame,
            use_mixedlm: bool=False,
            return_regression_coef: bool=False) -> Union[List,pd.Series]:
    '''
    Test dependency of gene expression of specific gene on distance.
    By default generalized linear model is used.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData with one or more images.
    use_mixedlm : bool, optional (Default: `False`)
        Whether to use MixedLM to use information of sample identity.
        Instead of `statsmodels.api.GLM` use `statsmodels.api.mixedlm`.
    return_regression_coef : bool, optional (Default: `False`)
        Whether to return regression coefficients per each layer.
        
    Returns:
    --------
    List
        List of p-values of tested gene for intercept 
        and coefficients (distances).
    '''
    
    mindist=int(sorted(data['distance'])[0])
    
    if use_mixedlm:
        formula = f'value ~ C(distance,Treatment({mindist}))'
        model = sm.MixedLM.from_formula(formula, data=data,
                                        groups=data['groups'],
                                        re_formula=\
                                        f'1+C(distance,Treatment({mindist}))')
    else: 
        formula = f'value ~ C(distance,Treatment({mindist}))'
        model = sm.GLM.from_formula(formula, data=data, family=sm.families.NegativeBinomial())
    try: result = model.fit()
    except: return np.nan

    if return_regression_coef:
        return result.pvalues.values.tolist(), result.params
    else:
        return result.pvalues.values.tolist()

	
def calc_lm_s(adata: anndata.AnnData,
              sample: str,
              nonzero_counts: float,
              use_mixedlm: bool=False,
              samples: Union[List,None]=None,
              return_regression_coef: bool=False) -> pd.Series:
    '''
    Calculates adjusted p-values (Benjamini/Hochberg) for genes of interest, 
    genes that has nonzero counts in >70 % of cells (by default).
    Generalized linear model tests dependency of gene expression on distance
    for every gene from the list.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData with one or more images.
    sample : str
        Sample name. If neighborhood was calculated on AnnData 
        with one sample without .obs specifying 'samples', pass '' here. 
        If you select one sample from AnnData, pass sample name.
    use_mixedlm : bool, optional (Default: `False`)
        Whether to use MixedLM for samples with different conditions.
        For example, you have 10 samples, some of them responders, 
        some of them are non-responders.
        WARNING: it is not implemented yet.
    samples : list, optional (Default: `None`)
        List of sample names you want to test. 
        By default calculates on all samples in AnnData.
    return_regression_coef : bool, optional (Default: `False`)          
         Whether to return regression coefficients per each layer.

    Returns:
    --------
    ps.Series
        pd.Series with gene names and slope against flat line.
    '''
    
    if use_mixedlm: 
        # get sample for extracting goi
        if samples is None:
            sample=\
            set([i.split('_')[-1] 
                 for i in adata.uns['neighborhood'].columns 
                 if i.startswith('neighborhood')])[0]
        else: sample=samples[0]
        goi=get_goi(adata,sample,nonzero_counts)
    else:
        goi=get_goi(adata,sample,nonzero_counts)

    # check if sample does not have calculated distances
    if len(goi)==0: 
        print(f'{sample} does not have calculated distances.')
        return pd.Series()
    
    # calculate p-values for gene of interests
    print(f'Length of genes of interest = {len(goi)}')  
    if return_regression_coef:
        p_values_and_coef=[calc_lm(data=gen_data_for_lm(adata=adata,
                                                        gene=g,
                                                        sample=sample,
                                                        use_mixedlm=use_mixedlm,
                                                        samples=samples),
                                   use_mixedlm=use_mixedlm,
                                   return_regression_coef=return_regression_coef)
                           for g in tqdm.tqdm(goi,leave=True,position=0)]
        p_values=[pair[0] for pair in p_values_and_coef]
        coefs=pd.concat([pair[1] for pair in p_values_and_coef],
                         axis=1).T
        coefs.index=goi
        
    # calculate p-values for gene of interests
    else:
        p_values=[calc_lm(data=gen_data_for_lm(adata=adata,
                                               gene=g,
                                               sample=sample,
                                               use_mixedlm=use_mixedlm,
                                               samples=samples),
                           use_mixedlm=use_mixedlm,
                           return_regression_coef=return_regression_coef)
                  for g in tqdm.tqdm(goi,leave=True,position=0)]
    
    # filter out np.nan and gene names as well
    p_values_filt=[]
    goi_filt=[]
    for g,pv in zip(goi,p_values):
        if pv==pv:
            p_values_filt.append(pv)
            goi_filt.append(g)

    def get_trend(values):
        slope, _, _, _, _ = linregress(range(len(values)), values)
        return slope
    
    # calculate slope of -1,1,2 or the first three layers
    try:
        ngh_cols=coefs.columns
        elem=ngh_cols[ngh_cols.str.contains('T.-1')]
        elem_ix=ngh_cols.tolist().index(elem)
    except:
        elem_ix=1

    slope=\
    coefs.iloc[:,elem_ix:elem_ix+3].apply(lambda x: get_trend(x), axis=1)\
    .sort_values()
    
    return slope

def calc_lm_on_samples(adata: anndata.AnnData,
                       sample_obs_key: str,
                       samples: Union[List,None]=None,
                       nonzero_counts: float=0.7,
                       use_mixedlm: bool=False,
                       return_regression_coef: bool=False
                      ) -> dict:
    '''
    Calculates adjusted p-values (Benjamini/Hochberg) for genes of interest, 
    genes that has nonzero counts in >70 % of cells (default).
    Generalized linear model tests dependency of gene expression on distance
    for every gene from the list.
    WARNING: mixedlm is not implemented yet.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData with one or more images.
    sample_obs_key : str
        .obs name where 'samples' identity is stoted.
        If AnnData has only one sample without such object, pass `None`.
    samples : list, optional (Default: `None`)
        List of sample names you want to test. 
        By default calculates on all samples in AnnData.
    nonzero_counts : float (Default: `0.7`)
        The minimum fraction of barcodes expressing a certain gene.
    use_mixedlm : bool, optional (Default: `False`)
        Whether to use MixedLM for samples with different conditions.
        For example, you have 10 samples, some of them responders, 
        some of them are non-responders.
        WARNING: it is not implemented yet.
    return_regression_coef : bool, optional (Default: `False`)                 
          Whether to return regression coefficients per each layer.

    Returns:
    --------
    Union[pd.Series,dict]
        - pd.Series with gene names and corresponding adjusted p-values,
        if `sample_obs_key==None`.
        - dict where keys are sample names and values are 
        pd.Series with gene names and corresponding adjusted p-values.
    '''
    
    if use_mixedlm:
        print('Running MixedLM.')
        return calc_lm_s(adata,'',nonzero_counts,use_mixedlm,
                         samples,return_regression_coef)
    
    print('Running GLM.')
    
    if sample_obs_key is None: 
        print('Calculating linear model on one sample.')
        return calc_lm_s(adata,'',nonzero_counts,use_mixedlm,
                         None,return_regression_coef)
    
    if samples is None: sample_list=adata.obs[sample_obs_key].unique()
    else: sample_list=samples
    
    r={}
    for sample in sample_list:
        adatatmp=adata[adata.obs[sample_obs_key]==sample]
        adatatmp.uns['spatial']={
            s:adatatmp.uns['spatial'][s] \
            for s,v in adatatmp.uns['spatial'].items() \
            if s==sample
        }
        print(f'Calculating linear model on {sample} sample.')
        r[sample]=calc_lm_s(adatatmp,sample,nonzero_counts,use_mixedlm,
                            None,return_regression_coef)
    return r

