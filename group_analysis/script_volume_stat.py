"""
Script to calibrate RFX/MFX statistic in the volume

"""
import os.path as op
import numpy as np
from configobj import ConfigObj
from nipy.neurospin.utils.mask import intersect_masks
from nipy.io.imageformats import load, save, Nifti1Image
from stat_calibration import select_voxel_level_threshold, select_cluster_level_threshold
import nipy.neurospin.group.onesample as fos
import nipy.neurospin.graph.graph as fg
import mixed_effects_stat as mes


################################################################
# first define paths etc.
################################################################

subj = ['s12069', 's12300', 's12401', 's12431', 's12508', 's12532', 's12539', 's12562','s12590', 's12635', 's12636', 's12898', 's12081', 's12165', 's12207', 's12344', 's12352', 's12370', 's12381', 's12405', 's12414', 's12432']
nsubj = len(subj)
db_path = '/volatile/subjects_database'
mask_images = [op.join(db_path,"%s/fMRI/default_acquisition/Minf/mask.nii") % s for s in subj]

# create the mask
mask = intersect_masks(mask_images, output_filename=None,
                       threshold=0.5, cc=True)
affine = load(mask_images[0]).get_affine()
grp_mask = Nifti1Image(mask, load(mask_images[0]).get_affine())
ijk = np.array(np.where(mask)).T
nvox = np.sum(mask)

# output dir
b_smooth = False
if b_smooth:
    print "smoothed data"
    threshold_path = 'volume_threshold_smooth.con'
    swd = '/volatile/subjects_database/group_analysis/smoothed_FWHM5'
else:
    print "unsmoothed data"
    threshold_path = 'volume_threshold.con'
    swd = '/volatile/subjects_database/group_analysis/smoothed_FWHM0'

save(grp_mask, op.join(swd,'grp_mask.nii'))

################################################################
# Load the effects and variance
################################################################

def load_images(con_images, var_images):
    """
    """
    nsubj = len(con_images)
    beta = []
    varbeta = []
    tiny = 1.e-15
    for s in range(nsubj): 
        rbeta = load(con_images[s])
        temp = (rbeta.get_data())[mask]
        beta.append(temp)
        rvar = load(var_images[s])
        temp = (rvar.get_data())[mask]
        varbeta.append(temp)

    VarFunctional = np.array(varbeta).T
    Functional = np.array(beta).T
    Functional[np.isnan(Functional)] = 0
    VarFunctional[np.isnan(VarFunctional)] = 0
    VarFunctional = np.maximum(VarFunctional, tiny)
    return Functional,  VarFunctional

contrast = ['audio-video', 'reading-visual', 'left-right', 'right-left', 'computation-sentences']
#contrast = ['audio-video']
contrast_id = contrast

##########################################################################
# Compute statistical thresholds
##########################################################################

for cid in contrast_id:
    print cid
    if b_smooth:
        con_images = [
            op.join(db_path,
        "%s/fMRI/default_acquisition/glm/smoothed_FWHM5/Contrast/%s_con.nii")
            % (s,cid) for s in subj]
        var_images = [
            op.join(db_path,
        "%s/fMRI/default_acquisition/glm/smoothed_FWHM5/Contrast/%s_ResMS.nii")
            % (s,cid) for s in subj]
    else:
        con_images = [
            op.join(db_path,
        "%s/fMRI/default_acquisition/glm/smoothed_FWHM0/Contrast/%s_con.nii")
            % (s,cid) for s in subj]
        var_images = [
            op.join(db_path,
        "%s/fMRI/default_acquisition/glm/smoothed_FWHM0/Contrast/%s_ResMS.nii")
            % (s,cid) for s in subj]

    Functional,  VarFunctional = load_images(con_images, var_images)
    
    zc_rfx = select_voxel_level_threshold(
        Functional, VarFunctional, nbsamp=1024, pval=0.05, method='rfx',
        corrected=True,group_size=nsubj)

    zc_mfx = select_voxel_level_threshold(
        Functional, VarFunctional, nbsamp=1024, pval=0.05, method='mfx',
        corrected=True,group_size=nsubj)

    z1_rfx = select_voxel_level_threshold(
        Functional, VarFunctional, nbsamp=1024, pval=0.001, method='rfx',
        corrected=False, group_size=nsubj)
    
    z2_rfx = select_cluster_level_threshold(
        Functional, VarFunctional, ijk, nbsamp=1024, pval=0.05, method='rfx',
        threshold=z1_rfx, group_size=nsubj)
    
    z1_mfx = select_voxel_level_threshold(
        Functional, VarFunctional, nbsamp=1024, pval=0.001, method='mfx',
        corrected=False, group_size=nsubj)
    
    z2_mfx = select_cluster_level_threshold(
        Functional, VarFunctional, ijk, nbsamp=1024, pval=0.05, method='mfx',
        threshold=z1_mfx, group_size=nsubj)
    
    thresholds={cid:{ 'zc_rfx':zc_rfx, 'zc_mfx':zc_mfx, 'z1_rfx':z1_rfx,
                      'z2_rfx':z2_rfx, 'z1_mfx':z1_mfx, 'z2_mfx':z2_mfx}}
    
    x = ConfigObj(threshold_path)
    for k in thresholds.keys():
        x[k] = thresholds[k]
            
    x.write()


##########################################################################
# Do the group analysis
##########################################################################

for cid in contrast_id:
    print cid
    if b_smooth:
        con_images = [
            op.join(db_path,
        "%s/fMRI/default_acquisition/glm/smoothed_FWHM5/Contrast/%s_con.nii")
            % (s,cid) for s in subj]
        var_images = [
            op.join(db_path,
        "%s/fMRI/default_acquisition/glm/smoothed_FWHM5/Contrast/%s_ResMS.nii")
            % (s,cid) for s in subj]
    else:
        con_images = [
            op.join(db_path,
        "%s/fMRI/default_acquisition/glm/smoothed_FWHM0/Contrast/%s_con.nii")
            % (s,cid) for s in subj]
        var_images = [
            op.join(db_path,
        "%s/fMRI/default_acquisition/glm/smoothed_FWHM0/Contrast/%s_ResMS.nii")
            % (s,cid) for s in subj]
    
    Functional,  VarFunctional = load_images(con_images, var_images)
    
    #------------------------------------------------------------------------
    # load the thresholds
    #------------------------------------------------------------------------
    
    thresholds = ConfigObj(threshold_path)
    zc_rfx = float(thresholds[cid]['zc_rfx'])
    zc_mfx = float(thresholds[cid]['zc_mfx'])
    z1_rfx = float(thresholds[cid]['z1_rfx'])
    z1_mfx = float(thresholds[cid]['z1_mfx'])
    z2_rfx = float(thresholds[cid]['z2_rfx'])
    z2_mfx = float(thresholds[cid]['z2_mfx'])
    
    #------------------------------------------------------------------------
    # rfx
    #------------------------------------------------------------------------
    
    #y = fos.stat(Functional, id='student', axis=1)
    y = mes.t_stat(Functional)
    
    #------------------------------------------------------------------------
    # voxel-level rfx
    #------------------------------------------------------------------------
    
    Label = np.zeros(grp_mask.get_shape())
    Label[mask] =  np.squeeze(y*(y>zc_rfx))
    wim = Nifti1Image(Label, grp_mask.get_affine())
    save(wim, op.join(swd,"vrfx_%s.nii"%cid))
    print "Number of active voxels: %04d"% np.sum(y>zc_rfx)
    
    #------------------------------------------------------------------------
    # cluster-level rfx
    #------------------------------------------------------------------------
    
    y = np.reshape(y,np.size(y))
    y = y * (y>z1_rfx)
    idx = np.nonzero(y)[0]
    ijkl = ijk[y>z1_rfx,:]
    n1 = ijkl.shape[0]
    if n1>0:
        gr = fg.WeightedGraph(ijkl.shape[0])
        gr.from_3d_grid(ijkl.astype(np.int))
        u = gr.cc()
        su = np.array([np.sum(u==ic) for ic in range (u.max()+1)])
        y[idx[su[u]<=z2_rfx]] = 0
    else:
        y *= 0
    
    print "Number of clusters: %04d"% np.sum(su>z2_rfx)
    
    Label = np.zeros(grp_mask.get_shape())
    Label[mask] =  np.squeeze(y)
    wim = Nifti1Image(Label, grp_mask.get_affine())
    save(wim, op.join(swd,"crfx_%s.nii"%cid))
    
    #------------------------------------------------------------------------
    # mfx
    #------------------------------------------------------------------------
    
    #y = fos.stat_mfx(Functional, VarFunctional, id='student_mfx', axis=1)
    y = mes.mfx_t_stat(Functional, VarFunctional)
    
    #------------------------------------------------------------------------
    # voxel-level mfx
    #------------------------------------------------------------------------
    
    Label = np.zeros(grp_mask.get_shape())
    Label[mask] =  np.squeeze(y*(y>zc_mfx))
    wim = Nifti1Image(Label, grp_mask.get_affine())
    save(wim, op.join(swd,"vmfx_%s.nii"%cid))
    
    print "Number of active voxels: %04d"% np.sum(y>zc_mfx)
    
    #------------------------------------------------------------------------
    # cluster-level mfx
    #------------------------------------------------------------------------
    
    y = np.reshape(y,np.size(y))
    y = y * (y>z1_mfx)
    idx = np.nonzero(y)[0]
    ijkl = ijk[y>z1_mfx,:]
    n1 = ijkl.shape[0]
    if n1>0:
        gr = fg.WeightedGraph(ijkl.shape[0])
        gr.from_3d_grid(ijkl.astype(np.int))
        u = gr.cc()
        su = np.array([np.sum(u==ic) for ic in range (u.max()+1)])
        y[idx[su[u]<=z2_mfx]] = 0
    else:
        y *= 0
    
    print "Number of clusters: %04d"% np.sum(su>z2_mfx)
    
    Label = np.zeros(grp_mask.get_shape())
    Label[mask] =  np.squeeze(y)
    wim = Nifti1Image(Label, grp_mask.get_affine())
    save(wim, op.join(swd,"cmfx_%s.nii"%cid))
