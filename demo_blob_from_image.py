"""
This scipt extracts the blobs from a 3D nifti image

This create as output
- a label image representing the nested blobs,
- an image of the averge signal per blob and
- an image with the terminal blob only

Author : Virgile Fritsch, 2010,
         adapted from original Bertrand Thirion's script, 2009
"""

import os.path as op
import os
import numpy as np
from nipy.io.imageformats import load, save, Nifti1Image
import tempfile
import nipy.neurospin.graph.field as ff
import nipy.neurospin.spatial_models.hroi as hroi

# -----------------------------------------------------------
# --------- Paths and Parameters ----------------------------
# -----------------------------------------------------------
from database_archi import *
# Group analysis paths and parameters
if SUBJECT == "group":
    # type of analysis
    GA_TYPE = "vrfx"
    # path to group analysis results in the database
    r_path = "/volatile/subjects_database"
    m_path = "%s/group_analysis/smoothed_FWHM5" % r_path
    # path to data to extract the blobs from
    glm_data_path = "%s/%s_%s.nii" % (m_path, GA_TYPE, CONTRAST)
    # output directory
    BLOBS3D_DIR = "%s/blobs3D_%s" % (m_path, CONTRAST)

# path to data to extract the blobs from
#glm_data_path = ?

# output directory
swd = BLOBS3D_DIR
if not os.path.exists(swd):
    os.makedirs(swd)
# type of leaves (end-blobs) to write :
# - "blobs": index-labelled blobs
# - "mblobs": mean-labelled blobs
#BLOB3D_TYPE = ?

# blob-forming threshold
threshold = THETA3D
# size threshold on bblobs
smin = SMIN3D

# -----------------------------------------------------------
# --------- Script starts -----------------------------------
# -----------------------------------------------------------
# Prepare the data
nim = load(glm_data_path)
affine = nim.get_affine()
shape = nim.get_shape()
data = nim.get_data()
values = data[data!=0]
xyz = np.array(np.where(data)).T
F = ff.Field(xyz.shape[0])
F.from_3d_grid(xyz)
F.set_field(values)

# Compute the nested roi object
label = -np.ones(F.V)
nroi = hroi.NROI_from_field(F, affine, shape, xyz, 0, threshold, smin)
bmap = -np.zeros(F.V)
if nroi != None:
    nroi.set_discrete_feature_from_index('activation', values)
    bfm = nroi.discrete_to_roi_features('activation')
    idx = nroi.discrete_features['index']
    for k in range(nroi.k):
        label[idx[k]] = k
        bmap[idx[k]] = bfm[k]
	
# saving the blob image,i. e. a label image 
wlabel = -2 * np.ones(shape)
wlabel[data!=0] = label + 1
wim = Nifti1Image(wlabel, affine)
wim.get_header()['descrip'] = 'blob image extracted from %s' %glm_data_path 
save(wim, op.join(swd,"blob.nii"))

# saving the image of the average-signal-per-blob
wlabel = np.zeros(shape)
wlabel[data!=0] = bmap
wim = Nifti1Image(wlabel, affine)
wim.get_header()['descrip'] = 'blob average signal extracted from %s' \
                              %(glm_data_path) 
save(wim, op.join(swd,"bmap.nii"))

# saving the image of the end blobs or leaves
label = -np.ones(F.V)
mean_value = -np.ones(F.V)
if nroi != None:
    nroi.set_discrete_feature_from_index('average', values)
    lroi = nroi.reduce_to_leaves()
    idx = lroi.discrete_features['index']
    leaves_avg = lroi.discrete_to_roi_features('average')
    for k in range(lroi.k):
        label[idx[k]] = k
        mean_value[idx[k]] =  leaves_avg[k,0]

if BLOB3D_TYPE == "mblobs":  #mean-labelled blobs
    wlabel = -2 * np.ones(shape)
    wlabel[data!=0] = mean_value
    wim = Nifti1Image(wlabel, affine)
    wim.get_header()['descrip'] = 'blob image extracted from %s' %glm_data_path 
    save(wim, op.join(swd,"mleaves.nii"))
    
    print "Wrote the blob image in %s" %op.join(swd, "blob.nii")
    print "Wrote the blob-average signal image in %s" %op.join(swd, "bmap.nii")
    print "Wrote the end-blob image in %s" %op.join(swd, "mleaves.nii")
else:  #index-labelled blobs
    wlabel = -2 * np.ones(shape)
    wlabel[data!=0] = label + 1
    wim = Nifti1Image(wlabel, affine)
    wim.get_header()['descrip'] = 'blob image extracted from %s' %glm_data_path 
    save(wim, op.join(swd,"leaves.nii"))
    
    print "Wrote the blob image in %s" %op.join(swd, "blob.nii")
    print "Wrote the blob-average signal image in %s" %op.join(swd, "bmap.nii")
    print "Wrote the end-blob image in %s" %op.join(swd, "leaves.nii")
