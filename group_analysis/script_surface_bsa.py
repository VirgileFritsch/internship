
import numpy as	 np
import os.path as op

import mesh_processing2 as mep

from gifti import loadImage
# to be replaced with nibabel.gifti

import nipy.neurospin.glm_files_layout.tio as tio

from nipy.neurospin.spatial_models.discrete_domain import domain_from_mesh
import nipy.neurospin.spatial_models.bayesian_structural_analysis as bsa
import nipy.neurospin.spatial_models.structural_bfls as sbf
from nipy.neurospin.clustering.von_mises_fisher_mixture import select_vmm, \
     select_vmm_cv, VonMisesMixture
from nipy.neurospin.spatial_models import hroi


def bsa_vmm(bf, gf0, sub, gfc, dmax, thq, ths, verbose=0):
    """
    Estimation of the population level model of activation density using 
    dpmm and inference
    
    Parameters
    ----------
    bf list of nipy.neurospin.spatial_models.hroi.HierarchicalROI instances
       representing individual ROIs
       let nr be the number of terminal regions across subjects
    gf0, array of shape (nr)
         the mixture-based prior probability 
         that the terminal regions are true positives
    sub, array of shape (nr)
         the subject index associated with the terminal regions
    gfc, array of shape (nr, coord.shape[1])
         the coordinates of the of the terminal regions
    dmax float>0:
         expected cluster std in the common space in units of coord
    thq = 0.5 (float in the [0,1] interval)
        p-value of the prevalence test
    ths=0, float in the range [0,nsubj]
        null hypothesis on region prevalence that is rejected during inference
    verbose=0, verbosity mode

    Returns
    -------
    crmap: array of shape (nnodes):
           the resulting group-level labelling of the space
    LR: a instance of sbf.LandmarkRegions that describes the ROIs found
        in inter-subject inference
        If no such thing can be defined LR is set to None
    bf: List of  nipy.neurospin.spatial_models.hroi.Nroi instances
        representing individual ROIs
    p: array of shape (nnodes):
       likelihood of the data under H1 over some sampling grid
       
    """
    dom = bf[0].domain
    n_subj = len(bf)
    
    crmap = -np.ones(dom.size, np.int)
    LR = None
    p = np.zeros(dom.size)
    if len(sub)<1:
        return crmap, LR, bf, p

    sub = sub.astype(np.int) 
    #gfc = np.concatenate(gfc)
    #gf0 = np.concatenate(gf0)

    # launch the VMM
    precision = 200.
    #vmm = select_vmm(range(10, 40, 5 ), precision, True, gfc)
    range_cv = range(3, min(30,gfc.shape[0])+1, 1)
    vmm = select_vmm_cv(range_cv, precision, True, gfc, sub)
    if verbose:
        vmm.show(gfc)

    print vmm.k
    z = vmm.responsibilities(gfc)    
    label = np.argmax(vmm.responsibilities(dom.coord), 1)-1
    
    # append some information to the hroi in each subject
    for s in range(n_subj):
        bfs = bf[s]
        if bfs.k>0 :
            leaves = bfs.isleaf()
            us = -np.ones(bfs.k).astype(np.int)

            # set posterior proba
            lq = np.zeros(bfs.k)
            lq[leaves] = 1-z[sub==s, 0]
            bfs.set_roi_feature('posterior_proba', lq)

            # set prior proba
            lq = np.zeros(bfs.k)
            lq[leaves] = 1-gf0[sub==s]
            bfs.set_roi_feature('prior_proba', lq)

            us[leaves] = z[sub==s].argmax(1)-1
            
            # when parent regions has similarly labelled children,
            # include it also
            us = bfs.make_forest().propagate_upward(us)
            bfs.set_roi_feature('label',us)
                        
    # derive the group-level landmarks
    # with a threshold on the number of subjects
    # that are represented in each one 
    LR, nl = sbf.build_LR(bf, thq, ths, dmax, verbose=verbose)

    # make a group-level map of the landmark position        
    crmap = bsa._relabel_(label, nl)   
    return crmap, LR, bf, p




def make_surface_BSA(meshes, texfun, texlat, texlon, theta=3.,
                     ths = 0, thq = 0.5, smin = 0, swd = "/tmp/",
                     contrast_id='cid'):
    """
    Perform the computation of surface landmarks
    this function deals mainly with io

    fixme
    -----
    Write the doc
    replace with nibabel gifti io
    """
    nbsubj = len(meshes)
    coord = []
    r0 = 1.

    mesh_dom = domain_from_mesh(meshes[0])
    ## get the surface-based coordinates
    latitude = tio.Texture(texlat[0]).read(texlat[0]).data
    latitude = latitude-latitude.min()
    longitude = tio.Texture('').read(texlon[0]).data

    #latitude = np.random.rand(mesh_dom.size) * 2  * np.pi
    #longitude = np.random.rand(mesh_dom.size) * np.pi
    coord = r0*np.vstack((np.sin(latitude) * np.cos(longitude),
                          np.sin(latitude) * np.sin(longitude),
                          np.cos(latitude))).T
    mesh_dom.coord = coord
    
    mesh_doms = []
    bf = []
    gf0 = np.array([], ndmin=1)
    sub = np.array([], ndmin=1)
    gfc = np.tile(np.array([], ndmin=2), (3,0)).T
    for s in range(nbsubj):
        
        """
        # this is for subject-specific domains
        mesh_dom = domain_from_mesh(meshes[s])
        
        #import Mesh
        mesh = loadImage(meshes[s])
        vertices = mesh.getArrays()[0].getData()

        ## get the surface-based coordinates
        #latitude = tio.Texture(texlat[s]).read(texlat[s]).data
        #latitude = latitude-latitude.min()
        #longitude = tio.Texture(texlat[s]).read(texlon[s]).data
        #print latitude.min(),latitude.max(),longitude.min(),longitude.max()
        latitude = np.random.rand(vertices.shape[0]) * 2  * np.pi
        longitude = np.random.rand(vertices.shape[0]) * np.pi
        lcoord = r0*np.vstack((np.sin(latitude) * np.cos(longitude),
                               np.sin(latitude) * np.sin(longitude),
                               np.cos(latitude))).T
        
        mesh_dom.coord = lcoord
        mesh_doms.append(mesh_dom)
        """
        
        functional_data = tio.Texture(texfun[s]).read(texfun[s]).data
        rois_confidence = functional_data[functional_data != -1]

        nroi = hroi.HROI_as_discrete_domain_blobs(
            mesh_dom, functional_data, threshold=0., smin=0)
        nroi.make_feature('position', mesh_dom.coord)
        bfc = nroi.representative_feature('position', 'mean')
        nroi.set_roi_feature('position', bfc)
        bf.append(nroi)
        gf0 = np.concatenate((gf0, rois_confidence))
        sub = np.concatenate((sub, s*np.ones(rois_confidence.size)))
        gfc = np.concatenate(
            (gfc, mesh_dom.get_coord()[np.where(functional_data != -1)[0]]))
    
    #lbeta = np.array(lbeta).T
    #bf, gf0, sub, gfc = bsa.compute_individual_regions (
    #    mesh_dom, lbeta, smin, theta, method='prior')
    
    
    verbose = 1
    crmap, LR, bf, p = bsa_vmm(bf, gf0, sub, gfc, dmax, thq, ths, verbose)
    
    if LR != None:
        defindex = LR.k+2
    else:
        defindex = 0
    
    # write the resulting labelling
    tex_labels_name = op.join(swd, "CR_%s.tex" % contrast_id)
    tio.Texture('', data=crmap).write(tex_labels_name)
    
    #write the corresponding density
    tex_labels_name = op.join(swd, "density_%s.tex" % contrast_id) 
    tio.Texture('', data=p).write(tex_labels_name)
    
    for s in range(nbsubj):
        tex_labels_name = op.join(swd,"AR_s%04d_%s.tex" % (s, contrast_id))
        label = -np.ones(mesh_dom.size, 'int32')
        #
        if bf[s]!=None:
            label = bf[s].label.astype('int32')
        tio.Texture('', data=label).write(tex_labels_name)
    return LR, bf
    


theta = 0.
smin = 0

dmax = 10.
ths = 4
thq = 0.9
BOOTSTRAP = False

subj_id = ['s12069', 's12300', 's12401', 's12431', 's12508', 's12532', 's12539', 's12562','s12590', 's12635', 's12636', 's12898', 's12081', 's12165', 's12207', 's12344', 's12352', 's12370', 's12381', 's12405', 's12414', 's12432']
nbsubj = len(subj_id)
datadir = "/volatile/subjects_database"
texlat = [op.join(datadir,"ico100_7_lat.tex") for s in subj_id]
texlon = [op.join(datadir,"ico100_7_lon.tex") for s in subj_id]
tex_type = "fcoord"
contrast = "left-right"

if not BOOTSTRAP:
    # left hemisphere
    texfun = [op.join(datadir,"%s/experiments/smoothed_FWHM5/%s_z_map/results_%s_level001/left_%s_z_map_FWHM2D5_smin2D5_FWHM3D5_smin3D5.tex"%(s, contrast, tex_type, contrast)) for s in subj_id]
    meshes = [op.join(datadir,"s12069/surf/lh.r.white.gii") for s in subj_id]
    swd = "/volatile/subjects_database/group_analysis/smoothed_FWHM5/surface_bsa"
    contrast_id = '%s_left_%s' %(tex_type, contrast)
    
    LR, bf = make_surface_BSA(
        meshes, texfun, texlat, texlon, theta, ths, thq, smin, swd, contrast_id)
    
    # right hemisphere
    texfun = [op.join(datadir,"%s/experiments/smoothed_FWHM5/%s_z_map/results_%s_level001/right_%s_z_map_FWHM2D5_smin2D5_FWHM3D5_smin3D5.tex"%(s, contrast, tex_type, contrast)) for s in subj_id]
    meshes = [op.join(datadir,"s12069/surf/rh.r.white.gii") for s in subj_id]
    swd = "/volatile/subjects_database/group_analysis/smoothed_FWHM5/surface_bsa"
    contrast_id = '%s_right_%s' %(tex_type, contrast)
    
    LR, bf = make_surface_BSA(
        meshes, texfun, texlat, texlon, theta, ths, thq, smin, swd, contrast_id)
else:
    nb_iter = 5
    for i in range(nb_iter):
        pick = np.random.randint(0, high=nbsubj, size=nbsubj)
        sample_subj_id = [subj_id[s] for s in pick]
        
        # left hemisphere
        texfun = [op.join(datadir,"%s/experiments/smoothed_FWHM5/%s_z_map/results_%s_level001/left_%s_z_map_FWHM2D5_smin2D5_FWHM3D5_smin3D5.tex"%(s, contrast, tex_type, contrast)) for s in sample_subj_id]
        meshes = [op.join(datadir,"s12069/surf/lh.r.white.gii") for s in sample_subj_id]
        swd = "/volatile/subjects_database/group_analysis/smoothed_FWHM5/surface_bsa/reproducibility"
        contrast_id = 'iter%d_%s_left_%s' %(i, tex_type, contrast)
        
        LR, bf = make_surface_BSA(
            meshes, texfun, texlat, texlon, theta, ths, thq, smin, swd, contrast_id)
        
        # right hemisphere
        texfun = [op.join(datadir,"%s/experiments/smoothed_FWHM5/%s_z_map/results_%s_level001/right_%s_z_map_FWHM2D5_smin2D5_FWHM3D5_smin3D5.tex"%(s, contrast, tex_type, contrast)) for s in sample_subj_id]
        meshes = [op.join(datadir,"s12069/surf/rh.r.white.gii") for s in sample_subj_id]
        swd = "/volatile/subjects_database/group_analysis/smoothed_FWHM5/surface_bsa/reproducibility"
        contrast_id = 'iter%d_%s_right_%s' %(i, tex_type, contrast)
        
        LR, bf = make_surface_BSA(
            meshes, texfun, texlat, texlon, theta, ths, thq, smin, swd, contrast_id)
        
    

