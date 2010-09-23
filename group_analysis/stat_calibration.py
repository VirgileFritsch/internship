"""
utility functions to calibrate statistics through permutations

Normally there should be something to do that in nipy,
but I'm not completely sure about it...

Bertrand
"""

#autoindent
import numpy as np
import nipy.neurospin.graph as fg
import nipy.neurospin.group.onesample as fos
from nipy.neurospin.utils.reproducibility_measures import fttest, conjunction
import mixed_effects_stat as mes

def select_voxel_level_threshold( effects, var, nbsamp=1024, pval=0.001, 
                                  method='rfx', corrected=False, 
                                  group_size=-1, verbose=0):
    """
    Calibrating the first-level threshold through permutations    

    Parameters
    ----------
    effects, array of shape (nvox,nsubj)
             input effects
    var, array of shape (nvox,nsubj)
         input variance
    nbsamp=1024, number of random draws
    pval=0.001, frequentist threshold
    method='crfx', string 'crfx' or 'cmfx'
    corrected=False, bool correaction for mcp or not
    group_size=-1, int,
                   when positive the subjects are drawn in the input population
                   without replaecment if group_size<nsubj
                   with replacement otherwise
    verbose=0, verbosity mode

    Returns
    -------
    zc, float, the firest-level threshold
    
    """
    nvox = effects.shape[0]
    nsubj = effects.shape[1]
    zmax = np.zeros(nbsamp)
    if group_size==-1:
        group_size=nsubj
    
    for irs in range(nbsamp):   
        # data splitting
        group = np.arange(nsubj)
        if group_size<nsubj:
           # draw randomly group_size subjects among nsubj
           group = np.argsort(np.random.rand(nsubj))[:group_size]
        if group_size>nsubj:
           group = (nsubj*np.random.rand(group_size)).astype(np.int)

        x = effects[:,group]
        if (method=='mfx') or (method=='ffx') or (method=='cjt'):
            vx = var[:,group]

        sswap = np.reshape(2*(np.random.rand(group_size)>0.5)-1, (1,group_size))
        # analysis
        if method=='rfx':        
            #y = fos.stat(sswap*x, id='student', axis=1)
            y = mes.t_stat(sswap*x)
        elif method=='mfx':
            #y = fos.stat_mfx(sswap*x, vx, id='student_mfx', axis=1)
            y = mes.mfx_t_stat(sswap*x, vx)
        elif method=='ffx':           
           y = fttest(sswap*x,vx)
        elif method=='cjt':
           sswap = np.reshape(2 * (np.random.rand(group_size) > 0.5) -1, (1, group_size))
           y = conjunction(sswap*x, vx, group_size/2)
        else: raise ValueError, 'unknown method'
        
        # statistic extraction
        if corrected:
            zmax[irs] = y.max() 
        else:
            aux = np.sort(y,0)
            zmax[irs] = aux[int((1-pval)*nvox)]
        if verbose:
           print irs, aux.min(), zmax[irs], aux.max()
    
    # derive the thresholds
    if corrected:
        aux = np.sort(zmax)
        zc = aux[int((1-pval)*nbsamp)]
    else:
        zc = zmax[zmax!=0].mean()# fix

    return zc

def select_cluster_level_threshold(
    effects, var, xyz, nbsamp=1024, pval=0.001,  method='rfx', threshold=0.,
    group_size=-1, verbose=0, graph=None, volume=None):
    """
    Calibrating the first-level threshold through permutations    

    Parameters
    ----------
    effects, array of shape (nvox,nsubj)
             input effects
    var, array of shape (nvox,nsubj)
         input variance
    xyz, array of shape(nvox,3)
         comon grid coorinates
    nbsamp=1024, number of random draws
    pval=0.001,frequentist threshold
    method='crfx', string 'crfx' or 'cmfx'
    thresold=0., cluster-forming threshold
    group_size=-1, int,
                   when positive the subjects are drawn in the input population
                   without replacement if group_size<nsubj
                   with replacement otherwise
    verbose=0, verbosity mode
    graph: graph instance, optional
           represents the dataset topology (used to form clusters)
    volume: array, optional
            yields for each voxle, node the surface/volume (1 by default)

    Returns
    -------
    kc, float, the firest-level threshold

    Note
    ----
    the p-value is systematically corrected
    """
    nvox = effects.shape[0]
    nsubj = effects.shape[1]
    k = np.zeros(nbsamp)
    if group_size==-1:
        group_size=nsubj

    if volume==None:
        volume = np.ones(nvox)
       
    for irs in range(nbsamp):   
        # data splitting
        group = np.arange(nsubj)
        if group_size<nsubj:
           # draw randomly group_size subjects among nsubj
           group = np.argsort(np.random.rand(nsubj))[:group_size]
        if group_size>nsubj:
           group = (nsubj*np.random.rand(group_size)).astype(np.int)

        x = effects[:,group]
        if method=='mfx':
            vx = var[:,group]

        sswap = np.reshape(2*(np.random.rand(group_size)>0.5)-1, (1,group_size))
        # analysis
        if method=='rfx':        
           #y = fos.stat(x,id='student',axis=1, Magics=np.array([irs]))
           y = mes.t_stat(sswap*x)
        elif method=='mfx':
           #y = fos.stat_mfx(x, vx, id='student_mfx', axis=1, Magics=np.array([irs]))
           y = mes.mfx_t_stat(sswap*x, vx)
        else: raise ValueError, 'unknown method'
        
        # stat extraction
        y = np.reshape(y,np.size(y))
        xyzl = xyz[y>threshold,:]
        n1 = xyzl.shape[0]
        if n1>0:
            if graph==None:
                lg = fg.WeightedGraph(n1)
                lg.from_3d_grid(xyzl.astype(np.int))
            else:
                lg = graph.subgraph(y>threshold)
            u = lg.cc()
            lvolume = volume[y>threshold]
            cluster_vol = np.array([np.sum(lvolume[u==uu]) for uu in np.unique(u)])
            mk = cluster_vol.max()
        else:
            mk=0
        
        k[irs] = mk
        if verbose:
           print irs, n1, mk, y.min(), y.max()

    # derive the thresholds
    aux = np.sort(k)
    kc = aux[int((1-pval)*nbsamp)]

    return kc
