import os
import anndata
import pandas as pd
import scanpy as sc
import squidpy as sq
import seaborn as sns
import decoupler as dc
import matplotlib.pyplot as plt

from typing import List,Optional,Union

plt.rcdefaults()
sc.set_figure_params()
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (5,5)

def prep_data(adata: anndata.AnnData) -> anndata.AnnData:
    '''
    Normalizing, log-transforming.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData with one or more visium images.
    
    Returns:
    --------
    anndata.AnnData
        AnnData with calculated PCA and UMAP.
    '''
    
    print('Normalizing, log-transforming.')
    sc.pp.normalize_total(adata,target_sum=1e6)
    sc.pp.log1p(adata)
    return adata

def check_preloaded_db(db_name: str) -> Union[pd.DataFrame,None]:
    '''
    Check whether database was already loaded locally.
    '''
    
    path_to_preloaded_dbs='preloaded_dbs'
    if os.path.isdir(path_to_preloaded_dbs):
        if os.path.isfile(f'{path_to_preloaded_dbs}/{db_name}.csv'):
            print(f'Loading pre-loaded {db_name} database')
            return pd.read_csv(f'{path_to_preloaded_dbs}/{db_name}.csv',
                               index_col=0)
        else:
            return None
    else:
        os.mkdir(path_to_preloaded_dbs)
        return None

def calc_progeny(adata: anndata.AnnData,
                 precomp: bool=False) -> anndata.AnnData:
    '''
    Calculate progeny score.
    
    Parameters:
    -----------
    adata : anndata.AnnData 
        AnnData object with one or several visium images.
    precomp : bool, optional (Default is `False`) 
        Whether adata log-transformed and PCA, UMAP are calculated.
    
    Returns:
    --------
    anndata.AnnData
        AnnData with .obsm['progeny_mlm_estimate'] 
        and .obsm['progeny_mlm_pvals'].
    '''
    
    if not precomp: prep_data(adata)
    
    prg=check_preloaded_db('progeny')
    if prg is None:
        print('Loading progeny.')
        prg=dc.get_progeny(organism='human',top=100)
        print('Saving progeny to `preloaded_dbs`.')
        prg.to_csv('preloaded_dbs/progeny.csv')
        
    print('Running progeny')
    dc.run_mlm(mat=adata, net=prg, source='source', target='target', 
               weight='weight', verbose=True, use_raw=False)
    adata.obsm['progeny_mlm_estimate']=adata.obsm['mlm_estimate']
    adata.obsm['progeny_mlm_pvals']=adata.obsm['mlm_pvals']
    del adata.obsm['mlm_estimate']; del adata.obsm['mlm_pvals']
    
    return adata

def calc_dorothea(adata: anndata.AnnData,
                  precomp: bool=False) -> anndata.AnnData:
    '''
    Calculate dorothea score.
    
    Parameters:
    -----------
    adata : anndata.AnnData 
        AnnData object with one or several visium images.
    precomp : bool, optional (Default is `False`) 
        Whether adata log-transformed and PCA, UMAP are calculated.
    
    Returns:
    --------
    anndata.AnnData
        AnnData with .obsm['dorothea_mlm_estimate'] 
        and .obsm['dorothea_mlm_pvals']
    '''
    
    if not precomp: prep_data(adata)
    
    drt=check_preloaded_db('dorothea')
    if drt is None:
        print('Loading dorothea.')
        drt=dc.get_dorothea(organism='human',levels=['A','B','C'])
        print('Saving dorothea to `preloaded_dbs`.')
        drt.to_csv('preloaded_dbs/dorothea.csv')
    
    print('Running dorothea')
    dc.run_mlm(mat=adata, net=drt, source='source', target='target', 
               weight='weight', verbose=True, use_raw=False)
    adata.obsm['dorothea_mlm_estimate']=adata.obsm['mlm_estimate']
    adata.obsm['dorothea_mlm_pvals']=adata.obsm['mlm_pvals']
    del adata.obsm['mlm_estimate']; del adata.obsm['mlm_pvals']
    
    return adata

def calc_cytosig(adata: anndata.AnnData,
                 precomp: bool=False) -> anndata.AnnData:
    '''
    Calculate CytoSig score.
    
    Parameters:
    -----------
    adata : anndata.AnnData 
        AnnData object with one or several visium images.
    precomp : bool, optional (Default is `False`) 
        Whether adata log-transformed and PCA, UMAP are calculated.
    
    Returns:
    --------
    anndata.AnnData
        AnnData with .obsm['cytosig_mlm_estimate'] 
        and .obsm['cytosig_mlm_pvals']
    '''
    
    if not precomp: prep_data(adata)
    
    cts=check_preloaded_db('cytosig')
    if cts is None:
        print('Loading CytoSig.')
        cts=dc.get_resource('CytoSig')
        cts=cts[~cts.duplicated(['cytokine_genesymbol','target_genesymbol'])]
        cts['score']=cts['score'].astype(float)
        print('Saving CytoSig to `preloaded_dbs`.')
        cts.to_csv('preloaded_dbs/cytosig.csv')
    
    print('Running CytoSig')
    dc.run_mlm(mat=adata, net=cts, source='cytokine_genesymbol', 
               target='target_genesymbol', 
               weight='score', verbose=True, use_raw=False)
    adata.obsm['cytosig_mlm_estimate']=adata.obsm['mlm_estimate']
    adata.obsm['cytosig_mlm_pvals']=adata.obsm['mlm_pvals']
    del adata.obsm['mlm_estimate']; del adata.obsm['mlm_pvals']
    
    return adata

def calc_msigdb(adata: anndata.AnnData,
                ora: bool=True,
                precomp: bool=False,
                msigdb_collection: Optional[List]=['hallmark']
               ) -> anndata.AnnData:
    '''
    Calculate msigdb score.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData object with one or several visium images.
    ora : bool (Default is `True`) 
        Whether to use over representation analysis (True) or GSEA (False). 
        WARNING: GSEA is very slow.
    precomp : bool (Default is `False`)
        Whether adata log-transformed and PCA, UMAP are calculated.
    msigdb_collection : list, optional (Default is `['hallmark']`)
        Names of msigdb collection. 
        For example ['kegg_pathways','hallmark','reactome_pathways'].
    
    Returns:
    --------
    anndata.AnnData
        AnnData with .obsm['{msigdb_collection}_ora_estimate'] 
        and .obsm['{msigdb_collection}_ora_pvals'].
    '''
    
    msg=check_preloaded_db('msigdb')
    if msg is None:
        print('Loading MSigDB.')
        msg=dc.get_resource('MSigDB')
        print('Saving MSigDB to `preloaded_dbs`.')
        msg.to_csv('preloaded_dbs/msigdb.csv')
    
    # check all names exist in msigdb
    all_collections=msg.collection.unique().tolist()
    existing_names=[i for i in msigdb_collection if i in all_collections]
    if len(existing_names)!=len(msigdb_collection):
        nonexisting_names=list(set(msigdb_collection)-set(existing_names))
        print(f'{nonexisting_names} not in MSigDB collection')
        return print(f'Choose from collections: {all_collections}')
    
    if not precomp: prep_data(adata)
    if not ora: print('You are trying to run GSEA. It may run very slowly.')

    for name in msigdb_collection:
        
        print(f'Extracting {name}')
        msg_col=msg[msg['collection']==name]
        msg_col=msg_col[~msg_col.duplicated(['geneset','genesymbol'])]

        if ora:
            print('Running ora')
            try:
                dc.run_ora(mat=adata, net=msg_col, source='geneset', 
                           target='genesymbol', 
                           verbose=True, use_raw=False)
            except ValueError as e:
                print(f'{e}\nContinue with the next collection.')
                continue
            adata.obsm[f'{name}_ora_estimate']=adata.obsm['ora_estimate']
            adata.obsm[f'{name}_ora_pvals']=adata.obsm['ora_pvals']
            del adata.obsm['ora_estimate']; del adata.obsm['ora_pvals']
        else:
            print('Running GSEA')
            try:
                dc.run_gsea(mat=adata, net=msg_col, source='geneset', 
                            target='genesymbol', 
                            verbose=True, use_raw=False)
            except ValueError as e:
                print(f'{e}\nContinue with the next collection.')
                continue
            adata.obsm[f'{name}_gsea_estimate']=adata.obsm['gsea_estimate']
            adata.obsm[f'{name}_gsea_pvals']=adata.obsm['gsea_pvals']
            del adata.obsm['gsea_estimate']
            del adata.obsm['gsea_pvals']
            del adata.obsm['gsea_norm']

    return adata

def calc_decoupler(adata: anndata.AnnData,
                   ora: bool=True,
                   precomp: bool=False,
                   msigdb_collection: Optional[List]=['hallmark']
                  ) -> anndata.AnnData:
    '''
    Calculates progeny, dorothea, cytosig and msigdb scores. 
    If you want to calculate them separately 
    use functions `calc_{name_of_db}` from the same package.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData object with one or several visium images.
    ora : bool (Default is `True`) 
        Whether to use over representation analysis (True) or GSEA (False). 
        WARNING: GSEA is very slow.
    precomp : bool (Default is `False`)
        Whether adata log-transformed and PCA, UMAP are calculated.
    msigdb_collection : list (Default is `['hallmark']`)
        Names of msigdb collection. 
        For example ['kegg_pathways','hallmark','reactome_pathways'].
    
    Returns:
    --------
    anndata.AnnData
        AnnData with new .obsm objects.
    '''
    
    if not precomp: prep_data(adata)
    
    try: adata=calc_progeny(adata,True)
    except: print('Progeny failed')
    
    try: adata=calc_dorothea(adata,True)
    except: print('Dorothea failed')
    
    try: adata=calc_cytosig(adata,True)
    except: print('Cytosig failed')
    
    if len(msigdb_collection)>0:
        try: adata=calc_msigdb(adata=adata,
                               msigdb_collection=msigdb_collection,
                               ora=ora,precomp=True)
        except: print('MSigDB failed')
    
    return adata

def get_activities(adata: anndata.AnnData,
                   obsm_key: str) -> anndata.AnnData:
    '''
    Return AnnData object with activities of certain .obsm object.
    
    Parameters:
    -----------
    adata : anndata.AnnData 
        AnnData object with one or several visium images.
    obsm_key : str 
        .obsm object you want to extract. 
        For example .obsm['msigdb_ora_estimate'].
    
    Returns:
    --------
    anndata.AnnData
        AnnData object of specified .obsm object.
    '''
    
    return dc.get_acts(adata, obsm_key=obsm_key)

def get_activities_per_group(acts: anndata.AnnData,
                             group_name: str,
                             mode: str='mean') -> pd.DataFrame:
    '''
    Return AnnData object with mean/median activities per specified group.
    
    Parameters:
    -----------
    acts : anndata.AnnData 
        AnnData object with activities of certain .obsm object.
    group_name : str 
        Group of interest you want to calculate activities on. 
        For example 'leiden'.
    mode : str, optional (Default is `mean`)
        Take either 'mean' or 'median' while summarizing activities per group.
    
    Returns:
    --------
    pd.DataFrame
        pd.DataFrame with activities per certain group.
    '''
    
    return dc.summarize_acts(acts,groupby=group_name,mode=mode,min_std=0.35)

def get_series_w_lists(series: pd.Series) -> pd.Series:
    '''
    Transform pd.Series from long to wide format with list as a value.
    
    Parameters:
    -----------
    series : pd.Series
        pd.Series.
    
    Returns:
    --------
    pd.Series
        pd.Series with list as a value for each row.
    '''
    
    return series.groupby(series.index).apply(list)

def get_geneset_pairs(adata: anndata.AnnData,
                      msigdb_collection: Optional[List]=[
                          'kegg_pathways','hallmark','reactome_pathways'
                      ]) -> pd.Series:
    '''
    Return progeny, dorothea and msigdb gene-geneset pairs.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData object with one or several visium images.
    msigdb_collection : list, optional (Default is 
    `['kegg_pathways','hallmark','reactome_pathways']`)
        Names of msigdb collection.
    
    Returns:
    --------
    pd.Series
        pd.Series with gene lists for genesets.
    '''

    prg=check_preloaded_db('progeny')
    if prg is None:
        print('Loading progeny.')
        prg=dc.get_progeny(organism='human',top=100)
        print('Saving progeny to `preloaded_dbs`.')
        prg.to_csv('preloaded_dbs/progeny.csv')
    
    drt=check_preloaded_db('dorothea')
    if drt is None:
        print('Loading dorothea.')
        drt=dc.get_dorothea(organism='human',levels=['A','B','C'])
        print('Saving dorothea to `preloaded_dbs`.')
        drt.to_csv('preloaded_dbs/dorothea.csv')

    prgseries=prg.set_index('source')['target']
    drtseries=drt.set_index('source')['target']

    if len(msigdb_collection)>0:
        
        msg=check_preloaded_db('msigdb')
        if msg is None:
            print('Loading MSigDB.')
            msg=dc.get_resource('MSigDB')
            print('Saving MSigDB to `preloaded_dbs`.')
            msg.to_csv('preloaded_dbs/msigdb.csv')

        r=[]
        for c in msigdb_collection:
            msgtmp=msg[msg['collection']==c]
            msgtmp=msgtmp[~msgtmp.duplicated(['geneset','genesymbol'])]
            r.append(msgtmp.set_index('geneset')['genesymbol'])
        msgseries=pd.concat(r)
    
        return pd.concat(list(map(get_series_w_lists,[prgseries,drtseries,
                                                      msgseries])))
    
    else: return pd.concat(list(map(get_series_w_lists,
                                    [prgseries,drtseries])))