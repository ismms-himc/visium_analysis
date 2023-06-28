def collapse_obs2uns(adata: anndata.AnnData,
                     n: int,
                     sample_obs_key: str,
                     samples: list,
                     noncumulative: bool=True,
                     add_intra: bool=True,
                     intra_only: bool=False):
    # infer types
    types=[]
    if intra_only: types.append('intra')
    else:
        if noncumulative: types.extend(['c','nc'])
        else: types.append('c')
        if add_intra: types.append('intra')
        
    # parse extreme cases
    if sample_obs_key is None: 
        samples=['']
    else:
        if samples is None: 
            samples=adata.obs[sample_obs_key].unique()
    
    # create dict for all neighborhoods
    d0={}
    for s in samples:
        d1={}
        for i in range(-n,n+1):
            d2={}
            for t in types:
                try:
                    d2[t]=adata.obs[f'neighborhood_{i}_{t}_{s}'].dropna()
                except: continue
            d1[i]=d2
        d0[s]=d1
        
    for i in adata.obs.columns:
        if i.startswith('neighborhood'):
            del adata.obs[i]
            
    adata.uns['neighborhood']=d0
    return adata

def process_ngh_colors(adata: anndata.AnnData, 
                       *args) -> anndata.AnnData:
    d=adata.uns['neighborhood']
    series_list,true_ngh_colors=[],[]
    for s in d:
        for n in d[s]:
            for t in d[s][n]:
                name=f'neighborhood_{n}_{t}_{s}'
                try:
                    series_list.append(d[s][n][t])
                    true_ngh_colors.append(name)
                except: continue
                
    for i,j in zip(true_ngh_colors,series_list):
        adata.obs[i]=j
    return adata

def uns2obs(func):
    def wrapper(adata,*args):
        adata=process_ngh_colors(adata,*args)
        return func(adata,*args)
    return wrapper
