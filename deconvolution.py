import tqdm
import anndata
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

from typing import List,Union

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from sklearn.cluster import DBSCAN

plt.rcdefaults()
sc.set_figure_params()
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (5,5)


def get_optimal_num_clusters(adata: anndata.AnnData) -> anndata.AnnData:
    '''
    Calculates optimal number of Leiden clusters.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData with one visium image.
    
    Returns:
    -------
    anndata.AnnData
        AnnData with calculated Leiden clustering.
    '''
    
    for i in [j/10 for j in range(1,11,1)]:
        adata_tmp=sc.tl.leiden(adata,resolution=i,copy=True)
        num_cl=len(adata_tmp.obs['leiden'].unique())
        if 9<=num_cl<=11: break
    print(f'Optimal number of leiden clusters {num_cl}, res={i}')
    return adata_tmp

def calc_leiden(adata: anndata.AnnData,
                deconv: pd.DataFrame,
                sample: str,
                pre_computed_adata: bool=False) -> anndata.AnnData:
    '''
    Calculates leiden clustering on deconvolution 
    and gene expression matrices.
    
    Parameters:
    -----------
    adata : anndata.AnnData 
        AnnData with one visium image, expected raw counts.
    deconv : pd.DataFrame
        pd.DataFrame with barcodes in rows 
        and deconvoluted cell subtypes in columns.
    sample : str
        Sample name.
    pre_computed_adata : bool, optional (Default is `False`)
        True if normalization, PCA, UMAP and leiden clustering
        are already done on adata.
    
    Returns:
    --------
    anndata.AnnData
        AnnData with .obs['leiden_deconv'] and .obs['leiden'].
    '''
    
    deconv_adata=anndata.AnnData(deconv)
    
    sc.tl.pca(deconv_adata)
    sc.pp.neighbors(deconv_adata)
    sc.tl.umap(deconv_adata)
    print('PCA and UMAP are calculated for deconvoluted matrix.')
    
    deconv_adata=get_optimal_num_clusters(deconv_adata)
    adata.obs[f'leiden_deconv{sample}']=\
    deconv_adata.obs['leiden'].reindex(adata.obs_names)
    print(f'Results for leiden clustering on deconvoluted matrix '\
    f'is saved in .obs[\'leiden_deconv{sample}\'].')
    
    if pre_computed_adata==True: 
        print('PCA, UMAP and Leiden clustering \
        are already calculated for gene expression matrix.')
        pass
    else:
        sc.pp.normalize_total(adata,target_sum=1e6)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata)
        sc.tl.pca(adata,use_highly_variable=True)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        print('PCA, UMAP are calculated for gene expression matrix.')
        g_adata=get_optimal_num_clusters(adata)
        adata.obs[f'leiden{sample}']=\
        g_adata.obs['leiden'].reindex(adata.obs_names)
        print(f'Results for leiden clustering on gene expression matrix ' \
        f'are saved in .obs[\'leiden{sample}\'].')
        
    return adata

def calc_leiden_on_deconv(adata: anndata.AnnData,
                          deconv: pd.DataFrame,
                          sample_obs_key: str=None,
                          samples: Union[list,None]=None,
                          pre_computed_adata: bool=False) -> anndata.AnnData:
    '''
    Calculate leiden clustering on deconvolution and gene expression matrices.
    Wrapper of `calc_leiden` that supports calculation 
    in AnnData with one or more images.
    
    Parameters:
    -----------
    adata : anndata.AnnData 
        AnnData object, expected raw counts. Can contain one or more samples.
    deconv : pd.DataFrame 
        pd.DataFrame with barcodes in rows 
        and deconvoluted cell subtypes in columns.
    sample_obs_key : str, optional (Default is `None`)
        .obs name where sample names are stored. 
        `None` if AnnData contains one sample.
    samples : list, optional (Default is `None`)
        Names of specific samples you want to plot 
        if you have several images in AnnData.
    pre_computed_adata : bool, optional (Default is `False`) 
        True if normalization, PCA, UMAP and leiden clustering 
        are already done on adata.
    
    Returns:
    --------
    anndata.AnnData
        AnnData with .obs['leiden_deconv'] and .obs['leiden'].
    '''
    
    if sample_obs_key==None: 
        sample=''
        return calc_leiden(adata,deconv,sample,pre_computed_adata=False)
    
    if samples: samples_list=samples
    else: samples_list=adata.obs[sample_obs_key].unique()
    
    r=[]
    for sample in samples_list:
        adatatmp=adata[adata.obs[sample_obs_key]==sample]
        adatatmp.uns['spatial']={s:adatatmp.uns['spatial'][s]\
                                 for s,v in adatatmp.uns['spatial'].items()\
                                 if s==sample}
        deconvtmp=deconv.reindex(adatatmp.obs_names)
        sample=f'_{sample}'
        print(f'Running {sample} sample.')
        r.append(calc_leiden(adatatmp,deconvtmp,\
                             sample,pre_computed_adata=False))
        
    r=sc.concat(r,join='outer',uns_merge='unique')
    return r
    
def get_dense_clusters(adata: anndata.AnnData,
                       group_name: str,
                       cluster: str) -> anndata.AnnData:
    '''
    Calculate DBSCAN on specified group of spots in `.obs`.
    Can be helpful if you want to assess difference in structured group. 
    For example, immune aggregates.
    
    Parameters:
    -----------
    adata : anndata.AnnData 
        AnnData object with one image.
    group_name : str 
        Name of `.obs` in adata, which clusters you want to assess. 
        For example .obs['leiden'].
    cluster : str 
        Cluster name in group_name, which you want to separate using DBSCAN.
        For example '0' in 'leiden'.
    
    Returns:
    --------
    anndata.AnnData
        AnnData with .obs['denselabels'], where `-1` defines outliers, 
        `1000` defines other clusters in group_name.
    '''
    
    indexofinterest=adata.obs[group_name][adata.obs[group_name]==cluster].index
    otherindexes=list(set(adata.obs_names)-set(indexofinterest))
    data=adata[indexofinterest].obsm['spatial']
    
    dbscan=DBSCAN(eps=130, min_samples=5)
    dbscan.fit(data)
    labels=dbscan.labels_
    # minus 2 == minus outliers and `other`
    print(f'Number of dense clusters = {len(set(labels))-2}')
    
    adata.obs['denselabels']=\
    pd.concat([\
               pd.Series(index=indexofinterest,data=labels),\
               pd.Series(index=otherindexes,data=[1000]*len(otherindexes))\
              ]).reindex(adata.obs_names).astype('category')
    print('Results for DBSCAN clustering is saved in .obs[\'denselabels\'].')
    return adata

def calc_utest(adata: anndata.AnnData,
               deconv: pd.DataFrame,
               denselabels: str,
               mcorr: bool=True,
               normalize: bool=True) -> pd.DataFrame:
    '''
    Calcute Mann-Whitney U test on 1 vs other cell subtypes.
    
    Parameters:
    -----------
    adata : anndata.AnnData 
        AnnData object with one image.
    deconv : pd.DataFrame
        pd.DataFrame with barcodes in rows 
        and deconvoluted cell subtypes in columns.
    denselabels : str 
        .obs name with dense labels.
    mcorr : bool, optional (Default is `True`) 
        Whether to apply `Benjamini-Hochberg` 
        multiple correction to `compare celltypes`.
    normalize : bool, optional (Default is `True`)
        Whether to normalize absolute number of deconvoluted cells 
        on mean number of cells in spot.
    
    Returns:
    --------
    pd.DataFrame
        pd.DataFrame of p-values per dense cluster per celltype.
    '''
    
    deconv['group']=adata.obs[denselabels].reindex(deconv.index)
    
    # generating unique (1) vs other
    # len of each list should be the same and equal to the number of dense clusters
    uniq,other,l=[],[],[]
    for i in deconv['group'].unique():
        if i not in [-1,1000]:
            l.append(i)
            
            if normalize: 
                uniq.extend([
                    group.astype(float)/group.sum(axis=1).mean()\
                    for group_name, group in deconv.groupby('group')\
                    if group_name==i
                ])
                
                other.append(pd.concat([
                    group.astype(float)/group.sum(axis=1).mean()\
                    for group_name, group in deconv.groupby('group')\
                    if group_name not in [-1,1000,i]
                ]))
            else:
                uniq.extend([
                    group.astype(float)\
                    for group_name, group in deconv.groupby('group')\
                    if group_name==i
                ])
                
                other.append(pd.concat([
                    group.astype(float)\
                    for group_name, group in deconv.groupby('group')\
                    if group_name not in [-1,1000,i]
                ]))

    # do not take into account 'group' column
    sbtps=deconv.columns[:-1]
    rctmp=[]
    counter=0
    for u,o in zip(uniq,other):
        rc=pd.DataFrame(index=[l[counter]],columns=sbtps)
        # get rid of `group` column
        u=u.iloc[:,:-1]
        o=o.iloc[:,:-1]
        # compare cell subtypes of 1 vs other
        for ix,c in enumerate(sbtps):
            # filter out columns with a majority of zeros
            # due to very small number we operating
            # there is a rounding up of values to the magnitude of 5
            if round(u.iloc[:,ix],3).value_counts(normalize=True).\
            values[0]<=0.5:
                rc.loc[l[counter],c]=\
                mannwhitneyu(u.iloc[:,ix],o.iloc[:,ix]).pvalue
            else:
                rc.loc[l[counter],c]=np.nan
        if mcorr:
            rcna=rc.loc[:,rc.isna().any()]
            rcnotna=rc.loc[:,~rc.isna().any()]
            notnasbtps=rcnotna.columns
            
            rcnotna=rcnotna.apply(lambda x: multipletests(x,method='fdr_bh')[1],\
                                  axis=1,result_type='expand')
            rcnotna.columns=notnasbtps
            
            rcconcat=pd.concat([rcna,rcnotna],axis=1).reindex(sbtps,axis=1)
            rctmp.append(rcconcat)
        else:
            rctmp.append(rc)
        counter+=1    
    rc=pd.concat(rctmp)
    return rc