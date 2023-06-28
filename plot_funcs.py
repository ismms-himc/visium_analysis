import tqdm
import anndata
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import squidpy as sq
import matplotlib.pyplot as plt

from distance import prepare_ngh_composition#,uns2obs,process_ngh_colors

from typing import Tuple,List,Optional,Union

plt.rcdefaults()
sc.set_figure_params()
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (5,5)

# @uns2obs
def plot_spatial_scatter(adata: anndata.AnnData,
                         sample_obs_key: Union[str,None],
                         samples: Union[List,None]=None,
                         decoupler_params: bool=False,
			 figsize: tuple=(10,10),
                         *args) -> None:
    '''
    Plot spatial scatter plots for any specified .obs keys as *args.
    Each row is one sample with its own .obs.

    Parameters:
    -----------
    adata : anndata.AnnData 
        AnnData object. Can contain one or more samples.
    sample_obs_key : str 
        .obs name where sample names are stored. 
        If you pass AnnData with one image, specify `None`.
    samples : list, optional (Default is `None`)
        Names of specific samples you want to plot in AnnData. 
        Default is to plot all the samples in AnnData.
    decoupler_params : bool, optional (Default is `False`)
        If `True`, applied `coolwarm` cmap 
        and  enlarged size of spots for decoupler output.
    figsize: tuple, optional (Default is (10,10))
	figsize like in squidpy.pl.spatial_scatter.
        For example (7,7).
    args : *args
        Any .obs keys delimited by comma or passed as *list. 
        For example 'pathologist_annotation', 'denselabels' 
        or *['pathologist_annotation','denselabels'].
    
    Returns:
    --------
    None.
    
    Example of command:
    -------------------
    plot_spatial_scatter(adata,'samples',
    ['MIME22_BIC21-A6_0','MIME22_BIC21-A5_0'],
    *['pathologist_annotation','denselabels'])
    '''
    
    colors=[i for i in args]
    
    # loading decoupler params
    if decoupler_params: 
        size=1.5
        cmap='coolwarm'
    else:
        size,cmap=None,None
    
    if samples is not None: sample_list=samples
    else: sample_list=adata.obs[sample_obs_key].unique()
    
    # plot image if adata has one sample
    if sample_obs_key is None:
        sq.pl.spatial_scatter(adata,color=colors,img_res_key='lowres',
                              size=size,cmap=cmap,
                              wspace=.2,ncols=len(colors))
        return None

    colors_common=[i for i in colors if ('neighborhood' not in i) \
                   and ('leiden' not in i)]
    
    for sample in sample_list:
        cols=[]
        # remove unnecessary samples from adata
        adatatmp=adata[adata.obs[sample_obs_key]==sample]
        adatatmp.uns['spatial']={
            s:adatatmp.uns['spatial'][s] \
            for s,v in adatatmp.uns['spatial'].items() \
            if s==sample
        }
        # check that all columns have values
        # crucial step for neighborhoods
        for i in colors:
            try:
                unique_values=adatatmp.obs[i].dropna()
                if len(unique_values)!=0: cols.append(i)
            except: 
                pass
        cols=colors_common+[i for i in cols if i.endswith(sample)]
     
        sq.pl.spatial_scatter(adatatmp,color=cols,
                              library_key=sample_obs_key,
                              img_res_key='lowres',wspace=.2,
                              size=size,cmap=cmap,
                              ncols=len(cols),figsize=figsize)
        
def clustermap_of_activities(acts_gr: anndata.AnnData) -> None:
    '''
    Plot clustermap of mean/median activities per specified group
    
    Parameters:
    -----------
    acts_gr : anndata.AnnData 
        AnnData object of with activities per certain group.
    
    Returns:
    --------
    None.
    '''
    
    x=acts_gr.shape[1]/2
    y=acts_gr.shape[0]/2
#     x0,y0,w,h=.2,.7,x/2000,y/20
    sns.clustermap(acts_gr,xticklabels=acts_gr.columns,figsize=(x,y),
                   dendrogram_ratio=(.03,.1),
                   cmap='coolwarm',z_score=1,
                   vmax=2,vmin=-2,)
#                    cbar_pos=(x0,y0,w,h))
    
def plot_clusters_composition(adata: anndata.AnnData,
                              group_name: str,
                              deconv: pd.DataFrame,
                              groups_of_interest: Union[List,None]=None,
                              normalize: bool=True) -> None:
    '''
    Plot composition (boxplots) of each cluster.

    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData object with one image.
    group_name : str 
        Name of `.obs` in adata, which clusters you want to assess. 
        For example .obs['leiden'].
    deconv : pd.DataFrame
        pd.DataFrame with barcodes in rows 
        and deconvoluted cell subtypes in columns.
    groups_of_interest : list, optinal (Default is `None`) 
        Specific groups in group_name to plot. Default is all groups.
    normalize : bool, optional (Default is `True`) 
        Whether to report normalized number of cells. 
        Normalization per mean spot size.
    
    Returns:
    --------
    None.
    '''
    
    if groups_of_interest: clusters=groups_of_interest
    else: clusters=sorted(adata.obs[group_name].dropna().unique())
    
    n=len(clusters)
    _,axs=plt.subplots(nrows=n,ncols=1,
                       figsize=(.2*len(deconv.columns),3*n+n),
                       sharex=False)
    for ax,i in tqdm.tqdm(zip(axs.flatten(),clusters),total=n):
        adatatmp=adata.obs[group_name][adata.obs[group_name]==i].index
        z=deconv.reindex(adatatmp)
        if normalize: z=z.astype(float)/z.sum(axis=1).mean()
        z.boxplot(ax=ax,rot=90,grid=False,flierprops={'markeredgewidth':.5})
        ax.set_xticklabels([])
        ax.set_title(i)
    ax.set_xticklabels(z.columns)
    plt.show()
    
def plot_ngh_cmps(adata: anndata.AnnData,
                  deconv: pd.DataFrame,
                  noncumulative: bool=True,
                  sample: Union[str,None]=None,
                  intra: bool=False) -> None:
    '''
    Plot neighborhood composition till specified distance.
    
    Parameters:
    -----------
    adata : anndata.AnnData 
        AnnData with one image.
    deconv : pd.DataFrame 
        pd.DataFrame with barcodes in rows 
        and deconvoluted cell subtypes in columns.
    noncumulative : bool, optional (Default is `True`)
        Whether to plot noncumulative (True) or cumulative (False) neighborhood.
    sample : str, optional (Default is `None`)
        Sample name. 
        Is used only in AnnData with several images.
    intra : bool, optional (Default is `False`) 
        Whether to add intra-neighborhood to plot.
    
    Returns:
    --------
    None.
    '''
    
    sbtps,dst_df=\
    prepare_ngh_composition(adata,deconv,noncumulative,sample,intra)

    all_dists=dst_df.columns.get_level_values(0).unique()

    # check if sample does not have celltype in the sample
    if len(all_dists)==0:
        print(f'{sample} does not have calculated distance for this celltype.')
        return None

    plt.figure(figsize=(len(all_dists)/1.5,5))
    leg=[]
    for i in sbtps:
        y=dst_df.iloc[:,dst_df.columns.get_level_values(1)==i].mean().values
        plt.plot(all_dists,y)

    plt.legend(sbtps, loc=(1,.4), edgecolor='black')
    plt.xticks(all_dists,[i*100 for i in all_dists])
    plt.xlabel('Distance, micrometers')
    plt.ylabel('Mean number of cells')
    plt.title(f'{sample}')

    for i in sbtps:
        ymin=dst_df.iloc[:,dst_df.columns.get_level_values(1)==i].quantile(.25)
        ymax=dst_df.iloc[:,dst_df.columns.get_level_values(1)==i].quantile(.75)
        plt.fill_between(all_dists,ymin,ymax,alpha=.1)

    plt.show()
    
def plot_neighborhood_composition(adata: anndata.AnnData,
                                  deconv: pd.DataFrame,
                                  sample_obs_key: Union[str,None],
                                  samples: Union[List,None]=None,
                                  noncumulative: bool=True,
                                  intra: bool=False) -> None:
    '''
    Plot neighborhood composition till specified distance.
    
    Parameters:
    -----------
    adata : anndata.AnnData 
        AnnData with one or more images.
    deconv : pd.DataFrame 
        pd.DataFrame with barcodes in rows 
        and deconvoluted cell subtypes in columns.
    sample_obs_key : str 
        .obs name where sample names are stored. 
        If you do not have such field, pass `None`.
    samples : list, optional (Default is `None`)
        Names of specific samples you want to plot in AnnData. 
        Default is to plot for all samples.
    noncumulative : bool, optional (Default is `True`)
        Whether to plot noncumulative (True) or cumulative (False) neighborhood.
    intra : bool, optional (Default is `False`) 
        Whether to add intra-neighborhood to plot.
    
    Returns:
    --------
    None.
    '''
    
    if sample_obs_key is None:
        plot_ngh_cmps(adata,deconv,noncumulative,None,intra)
        return None
    
    if samples: samples_list=samples
    else: samples_list=adata.obs[sample_obs_key].unique()
    
    for sample in samples_list:
        adatatmp=adata[adata.obs[sample_obs_key]==sample]
        adatatmp.uns['spatial']={
            s:adatatmp.uns['spatial'][s] \
            for s,v in adatatmp.uns['spatial'].items() \
            if s==sample
        }
        plot_ngh_cmps(adatatmp,deconv,noncumulative,sample,intra)
        
# code for plotting receptor-ligand interactions
        
def calc_cellphonedb(adata: anndata.AnnData,
                     obs_key: str,
                     denselabels: bool=True) -> Tuple[pd.DataFrame,set,set]:
    '''
    Calculate receptor-ligand interactions using CellPhoneDB.
    
    Parameters:
    -----------
    adata : anndata.AnnData 
        AnnData with one visium image.
    obs_key : str
        Any .obs key name.
    denselabels : bool, optional (Default is `True`) 
        Whether `obs_key` is dense labels or not. 
        If True, calculations are not performed for -1 and 100 clusters.
    
    Returns:
    --------
    Three variables:
        - pd.DataFrame: pd.DataFrame with mean expression 
        of receptor-ligand pairs
        - set: Set of pair names to display on x axis
        - set: Set of pair names to subset from pd.DataFrame
    '''
    
    print('Calculating CellPhoneDB on AnnData.')
    res=sq.gr.ligrec(adata=adata, cluster_key=obs_key, use_raw=False,
                     fdr_method=None, copy=True,
                     corr_method='fdr_bh',
                     interactions_params={"resources":"CellPhoneDB",
                                          "organism":"human"},
                     threshold=0.05, seed=0, n_perms=10000, n_jobs=1)
    
    # check if calculating for dense labels
    if denselabels: 
        print('Calculating for dense labels. ' \
        'Excluding clusters `-1` and `1000`.')
        to_check=[-1,1000]
    else: to_check=[]
    
    # concatenate results of calculations in one dataframe
    r=[]
    for i in adata.obs[obs_key].unique():
        if i not in to_check:
            i=str(i)
            mask=res['pvalues'][i][i].apply(lambda x: x<0.05)
            m=res['means'][i][i]
            m=m.reindex(mask[mask].index)
            r.append(m)
    r=pd.concat(r,axis=1).fillna(0)
    
    # get pairs to display
    xlabs,topindex=[],[]
    for c in r.columns:
        tmp=r[c].sort_values()[-10:].index.tolist()
        topindex.extend(tmp)
        for i in tmp:
            xlabs.append('|'.join(i))
    xlabs,topindex=set(xlabs),set(topindex)
    
    return r,xlabs,topindex

def plot_heatmap_rl_pairs(adata: anndata.AnnData,
                          obs_key: str,
                          topindex: set,
                          denselabels: bool=True,
                          genes_heatmap: Union[List,None]=None) -> None:
    '''
    Plot gene expression of genes of specified receptor-ligand interactions.
    
    Parameters:
    -----------
    adata : anndata.AnnData 
        AnnData with one visium image.
    obs_key : str
        Any .obs key name.
    topindex : set 
        Set of pair names to subset from pd.DataFrame 
        with mean expression of receptor-ligand pairs. 
        If you want to use as a standalone function, you can pass `None`.
    denselabels : bool, optional (Default is `True`)
        Whether `obs_key` is dense labels or not. 
        If True, calculations are not performed for -1 and 100 clusters.
    genes_heatmap : list, optional (Default is `None`)
        List of genes you want to assess. Default is to plot all gene pairs.
    
    Returns:
    --------
    None.
    '''
    
    if genes_heatmap is not None: m=sorted(genes_heatmap)
    else: m=sorted(list(set([j for i in topindex for j in i])))
    
    if denselabels: to_check=[-1,1000]
    else: to_check=[]
    
    adata_tmp=\
    adata[adata.obs[obs_key][~adata.obs[obs_key].isin(to_check)].index]
    sc.pl.heatmap(adata_tmp,m,obs_key,swap_axes=True,show_gene_labels=True)

def plot_rl_pairs(adata: anndata.AnnData,
                  obs_key: str,
                  denselabels: bool=True,
                  plot_heatmap: bool=True,
                  genes_heatmap: Union[List,None]=None) -> None:
    '''
    Plot receptor-ligand interactions using CellPhoneDB.
    Perform all calculations under the hood. 
    If you want to extract pd.DataFrame with pairs expression
    use `calc_cellphonedb` from the same package.
    If you want to plot heatmap with gene expressions separately, 
    you can either use `plot_heatmap_rl_pairs` function or `scanpy.pl.heatmap`.
    
    Parameters:
    -----------
    adata : anndata.AnnData 
        AnnData with one visium image.
    obs_key : str
        Any .obs key name.
    denselabels : bool, optional (Default is `True`)
        Whether `obs_key` is dense labels or not. 
        If True, calculations are not performed for -1 and 100 clusters.
    plot_heatmap : bool, optional (Default is `True`) 
        Additionally whether to plot heatmap of gene expressions 
        of all genes in receptor-ligand pairs.
    genes_heatmap : list, optional (Default is `None`)
        List of genes you want to assess. 
        Default is to plot all gene pairs.
    
    Returns:
    --------
    None.
    '''
    
    r,xlabs,topindex=calc_cellphonedb(adata,obs_key,denselabels)
    
    rtmp=r.reindex(topindex)
    x=list(xlabs)*len(r.columns)
    y=[]
    for i in r.columns: y.extend([i]*len(rtmp.index))
    vals=rtmp.melt()['value']

    plt.figure(figsize=(len(xlabs)/4,5))
    ax=plt.scatter(x=x,y=y,s=vals*50,
                   edgecolors='black',alpha=.6)
    ax.axes.set_ylabel(obs_key)
    ax.axes.set_xticklabels(['|'.join(i) for i in rtmp.index],rotation=90)

    # plot legend
    maxsize=round(vals.max(),2)
    minsize=round(vals.replace({0:np.nan}).dropna().sort_values().iloc[0],2)
    line1 = plt.Line2D([],[],linewidth=0,marker='o',ms=np.sqrt(maxsize*50),
                       markeredgecolor='black',alpha=.6)
    line2 = plt.Line2D([],[],linewidth=0,marker='o',ms=np.sqrt(minsize*50),
                       markeredgecolor='black',alpha=.6)
    ax.axes.legend(handles=(line1, line2), 
                   labels=(f'{maxsize}',f'{minsize}'), 
                   bbox_to_anchor=(1,1),
                   edgecolor='black',
                   title='Mean expression,\nlog1p',
                   alignment='left')

    plt.show()
    
    if plot_heatmap: plot_heatmap_rl_pairs(adata,obs_key,topindex,
                                           denselabels=True,
                                           genes_heatmap=genes_heatmap)
    
def plot_hm_gs(activities: anndata.AnnData,
               adata: anndata.AnnData,
               label: str,
               gs: pd.Series) -> None:
    '''
    Plot heatmap with gene expression of genes in specified geneset.
    All spots are binned into 'high' and 'low' 
    based on mean value of decoupler feature.
    WARNING: Do not use it for CytoSig.
    
    Parameters:
    -----------
    activities : anndata.AnnData 
        AnnData with decoupler features.
    adata : anndata.AnnData 
        AnnData with gene expression.
    label : str
        Name of decoupler feature. 
        For example 'REACTOME_TERMINAL_PATHWAY_OF_COMPLEMENT' or 'SRF'.
    gs : pd.Series
        pd.Series with genesets with list of genes as values.
    
    Return:
    -------
    None.
    '''
    
    totreat=activities[:,label].X
    adata.obs['bins']=np.where(totreat>totreat.mean(),'high','low')
    
    to_plot=set(gs[label])&set(adata.var_names)
    print(f'Length of genes to plot = {len(to_plot)}')
    sc.pl.heatmap(adata,list(to_plot),'bins',show_gene_labels=True)
    del adata.obs['bins']
