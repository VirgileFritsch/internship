"""
script to calibrate RFX/MFX statistic in the volume

Note
----
stat_calibration and mesh_processing_dev modules are not publicly available

Bertrand Thirion
"""
import os.path as op
import numpy as np
from configobj import ConfigObj

import nipy.neurospin.group.onesample as fos
import nipy.neurospin.graph.graph as fg

import mixed_effects_stat as mes
import mesh_processing as mep 
from stat_calibration import *

from soma import aims
from nipy.neurospin.glm_files_layout import tio

################################################################
# User Parameters
################################################################
# b_cached:
# - False = compute and store area
# - True = use cached area
# - None = are = 1
b_cached = False

# whether to use smoothed data or not
b_smooth = False

################################################################
# first define paths etc.
################################################################
# list of subjects
subj = ['s12069', 's12300', 's12401', 's12431', 's12508', 's12532', 's12539', 's12562','s12590', 's12635', 's12636', 's12898', 's12081', 's12165', 's12207', 's12344', 's12352', 's12370', 's12381', 's12405', 's12414', 's12432']
nsubj = len(subj)

# paths to mesh
db_path = '/data/home/virgile/virgile_internship'
left_meshes = [op.join(db_path,"%s/surf/lh.r.aims.white.mesh") %s for s in subj]
right_meshes = [op.join(db_path,"%s/surf/rh.r.aims.white.mesh") %s for s in subj]

# output path
if b_smooth:
    threshold_path = 'surface_threshold_smooth.con'
    # output dir
    swd = '/data/home/virgile/virgile_internship/group_analysis/smoothed_FWHM5'
else:
    threshold_path = 'surface_threshold.con'
    # output dir
    swd = '/data/home/virgile/virgile_internship/group_analysis/smoothed_FWHM0'


################################################################
# Create a topological model from the mesh
################################################################
# buils graph from meshes
R = aims.Reader()
left_mesh = R.read(left_meshes[0])
lg = mep.mesh_to_graph(left_mesh)
right_mesh = R.read(right_meshes[0])
rg = mep.mesh_to_graph(right_mesh)
gg = fg.concatenate_graphs(lg,rg)

area = np.ones(gg.V)
if b_cached==True:
    area = np.load('area.npz')
elif b_cached==False:
    area = np.zeros(gg.V)
    for s in range(nsubj):
        area[:lg.V] += mep.node_area(R.read(left_meshes[s]))
        area[lg.V:] += mep.node_area(R.read(right_meshes[s]))
    area /= nsubj
    area.dump('area.npz')

ijk = np.reshape(np.arange(gg.V), (gg.V, 1))


################################################################
# Load the effects and variance
################################################################
# list of contrasts
contrast = ['audio-video']
contrast_id = contrast

def load_textures(left_con_tex, right_con_tex, left_var_tex, right_var_tex):
    """
    """
    
    nsubj = len(left_con_tex)
    beta = []
    varbeta = []
    tiny  = 1.e-15
    for s in range(nsubj):
        ltex = tio.Texture(left_con_tex[s]).read(left_con_tex[s]).data
        rtex = tio.Texture(right_con_tex[s]).read(right_con_tex[s]).data
        beta.append(np.hstack((ltex,rtex)))
    
        ltex = tio.Texture(left_var_tex[s]).read(left_var_tex[s]).data
        rtex = tio.Texture(right_var_tex[s]).read(right_var_tex[s]).data
        varbeta.append(np.hstack((ltex,rtex)))
    
    VarFunctional = np.array(varbeta).T
    Functional = np.array(beta).T
    Functional[np.isnan(Functional)] = 0
    VarFunctional[np.isnan(VarFunctional)] = 0
    VarFunctional = np.maximum(VarFunctional, tiny)
    return Functional,  VarFunctional



################################################################
# Compute the thresholds
################################################################

for cid in contrast_id:
    print cid
    # build textures paths
    if b_smooth:
        left_con_tex = [
            op.join(db_path,
                    "%s/fct/glm/smoothed_FWHM5/Contrast/left_%s_con.tex")
            % (s,cid) for s in subj]
        right_con_tex = [
            op.join(db_path,
                    "%s/fct/glm/smoothed_FWHM5/Contrast/right_%s_con.tex")
            % (s,cid) for s in subj]
        left_var_tex = [
            op.join(db_path,
                    "%s/fct/glm/smoothed_FWHM5/Contrast/left_%s_ResMS.tex")
            % (s,cid) for s in subj]
        right_var_tex = [
            op.join(db_path,
                    "%s/fct/glm/smoothed_FWHM5/Contrast/right_%s_ResMS.tex")
            % (s,cid) for s in subj]
    else:
        left_con_tex = [
            op.join(db_path,
                    "%s/fct/glm/smoothed_FWHM0/Contrast/left_%s_con.tex")
            % (s,cid) for s in subj]
        right_con_tex = [
            op.join(db_path,
                    "%s/fct/glm/smoothed_FWHM0/Contrast/right_%s_con.tex")
            % (s,cid) for s in subj]
        left_var_tex = [
            op.join(db_path,
                    "%s/fct/glm/smoothed_FWHM0/Contrast/left_%s_ResMS.tex")
            % (s,cid) for s in subj]
        right_var_tex = [
            op.join(db_path,
                    "%s/fct/glm/smoothed_FWHM0/Contrast/right_%s_ResMS.tex")
            % (s,cid) for s in subj]

    # organize functional and variance functional data in an array structure
    Functional,  VarFunctional = load_textures(left_con_tex, right_con_tex, left_var_tex, right_var_tex)
    
    
    zc_rfx = select_voxel_level_threshold(
        Functional, VarFunctional, nbsamp=1024, pval=0.05, method='rfx',
        corrected=True, group_size=nsubj)
    print zc_rfx

    zc_mfx = select_voxel_level_threshold(
        Functional, VarFunctional, nbsamp=1024,pval=0.05, method='mfx',
        corrected=True, group_size=nsubj)
    print zc_mfx

    z1_rfx = select_voxel_level_threshold(
        Functional, VarFunctional, nbsamp=1024, pval=0.001, method='rfx',
        corrected=False, group_size=nsubj)
    print z1_rfx

    z2_rfx = select_cluster_level_threshold(
        Functional, VarFunctional, ijk, nbsamp=1024, pval=0.05, method='rfx',
        threshold=z1_rfx, group_size=nsubj, graph=gg, volume=area)
    print z2_rfx

    z1_mfx = select_voxel_level_threshold(
        Functional, VarFunctional, nbsamp=1024, pval=0.001, method='mfx',
        corrected=False, group_size=nsubj)
    print z1_mfx

    z2_mfx = select_cluster_level_threshold(
        Functional, VarFunctional, ijk, nbsamp=1024, pval=0.05, method='mfx',
        threshold=z1_mfx, group_size=nsubj, graph=gg, volume=area)
    print z2_mfx

    # save the thresholds in a ConfigObj structure
    thresholds={cid:{'zc_rfx':zc_rfx, 'zc_mfx':zc_mfx, 'z1_rfx':z1_rfx,\
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
    # build textures paths
    if b_smooth:
        left_con_tex = [
            op.join(db_path,
                    "%s/fct/glm/smoothed_FWHM5/Contrast/left_%s_con.tex")
            % (s,cid) for s in subj]
        right_con_tex = [
            op.join(db_path,
                    "%s/fct/glm/smoothed_FWHM5/Contrast/right_%s_con.tex")
            % (s,cid) for s in subj]
        left_var_tex = [
            op.join(db_path,
                    "%s/fct/glm/smoothed_FWHM5/Contrast/left_%s_ResMS.tex")
            % (s,cid) for s in subj]
        right_var_tex = [
            op.join(db_path,
                    "%s/fct/glm/smoothed_FWHM5/Contrast/right_%s_ResMS.tex")
            % (s,cid) for s in subj]
    else:
        left_con_tex = [
            op.join(db_path,
                    "%s/fct/glm/smoothed_FWHM0/Contrast/left_%s_con.tex")
            % (s,cid) for s in subj]
        right_con_tex = [
            op.join(db_path,
                    "%s/fct/glm/smoothed_FWHM0/Contrast/right_%s_con.tex")
            % (s,cid) for s in subj]
        left_var_tex = [
            op.join(db_path,
                    "%s/fct/glm/smoothed_FWHM0/Contrast/left_%s_ResMS.tex")
            % (s,cid) for s in subj]
        right_var_tex = [
            op.join(db_path,
                    "%s/fct/glm/smoothed_FWHM0/Contrast/right_%s_ResMS.tex")
            % (s,cid) for s in subj]

    Functional,  VarFunctional = load_textures(left_con_tex, right_con_tex, left_var_tex, right_var_tex)

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
    
    thy = y*(y>zc_rfx).astype(np.float)
    
    tex_labels_name = op.join(swd,"left_vrfx_%s.tex"%cid) 
    textureW = tio.Texture(tex_labels_name, data = thy[:lg.V])
    textureW.write(tex_labels_name)
    
    tex_labels_name = op.join(swd,"right_vrfx_%s.tex"%cid) 
    textureW = tio.Texture(tex_labels_name, data = thy[lg.V:])
    textureW.write(tex_labels_name)
    
    print 'number of active nodes: left %d, right: %d' \
          %(np.sum(thy[:lg.V]), np.sum(thy[lg.V:lg.V+rg.V]))
    
    #------------------------------------------------------------------------
    # cluster-level rfx
    #------------------------------------------------------------------------
    
    y = np.reshape(y,np.size(y))
    y = y * (y>z1_rfx)
    idx = np.nonzero(y)[0]
    n1 = len(idx)
    if n1>0:
        sg = gg.subgraph(y>z1_rfx)
        u = sg.cc()
        larea = area[y>z1_rfx]
        su = np.array([np.sum(larea[u==k]) for k in np.unique(u)])
        y[idx[su[u]<=z2_rfx]] = 0
        
        print "Number of clusters: %04d"% np.sum(su>z2_rfx)
        
    tex_labels_name = op.join(swd,"left_crfx_%s.tex"%cid) 
    textureW = tio.Texture(tex_labels_name, data = y[:lg.V])
    textureW.write(tex_labels_name)
        
    tex_labels_name = op.join(swd,"right_crfx_%s.tex"%cid)
    textureW = tio.Texture(tex_labels_name, data = y[lg.V:])
    textureW.write(tex_labels_name)
    
    #------------------------------------------------------------------------
    # mfx
    #------------------------------------------------------------------------
    
    #y = fos.stat_mfx(Functional, VarFunctional, id='student_mfx', axis=1)
    y = mes.mfx_t_stat(Functional, VarFunctional)
    
    #------------------------------------------------------------------------
    # voxel-level mfx
    #------------------------------------------------------------------------
    
    thy = y*(y>zc_mfx).astype(np.float)
    
    tex_labels_name = op.join(swd,"left_vmfx_%s.tex"%cid) 
    textureW = tio.Texture(tex_labels_name, data = thy[:lg.V])
    textureW.write(tex_labels_name)
        
    tex_labels_name = op.join(swd,"right_vmfx_%s.tex"%cid) 
    textureW = tio.Texture(tex_labels_name, data = thy[lg.V:])
    textureW.write(tex_labels_name)
    
    print 'number of active nodes: left %d, right: %d' \
          %(np.sum(thy[:lg.V]), np.sum(thy[lg.V:lg.V+rg.V]))

    #------------------------------------------------------------------------
    # cluster-level mfx
    #------------------------------------------------------------------------
    
    y = np.reshape(y,np.size(y))
    y = y * (y>z1_mfx)
    idx = np.nonzero(y)[0]
    n1 = len(idx)
    if n1>0:
        sg = gg.subgraph(y>z1_mfx)
        u = sg.cc()
        larea = area[y>z1_mfx]
        su = np.array([np.sum(larea[u==k]) for k in np.unique(u)])
        y[idx[su[u]<=z2_mfx]] = 0

    print "Number of clusters: %04d"% np.sum(su>z2_mfx)
    
    
    tex_labels_name = op.join(swd,"left_cmfx_%s.tex"%cid)
    textureW = tio.Texture(tex_labels_name, data = y[:lg.V])
    textureW.write(tex_labels_name)
    
    tex_labels_name = op.join(swd,"right_cmfx_%s.tex"%cid) 
    textureW = tio.Texture(tex_labels_name, data = y[lg.V:])
    textureW.write(tex_labels_name)
    


