"""
This scipt extracts the blobs from a 3D nifti image

This create as output
- a label image representing the nested blobs,
- an image of the averge signal per blob and
- an image with the terminal blob only

Author : Virgile Fritsch, 2010,
         adapted from original Bertrand Thirion's script, 2009
"""

import os
import numpy as np
from nipy.io.imageformats import load, save, Nifti1Image
import nipy.neurospin.spatial_models.hroi as hroi
from nipy.neurospin.spatial_models.discrete_domain import domain_from_image

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

#----- Path to the data mask
#brain_mask_path = ?
#----- Path to data to extract the blobs from
#glm_data_path = ?

#----- Output directory
swd = BLOBS3D_DIR
if not os.path.exists(swd):
    os.makedirs(swd)

#----- Blob-forming threshold (min significant intensity value)
#THETA3D = ?
#----- Size threshold on blobs
#SMIN3D = ?

# -----------------------------------------------------------
# --------- Script starts -----------------------------------
# -----------------------------------------------------------
### Compute the nested roi object
# get data domain (~ data mask)
mask = load(brain_mask_path)
mask_data = mask.get_data()
domain = domain_from_image(mask)
# get data
nim = load(glm_data_path)
glm_data = nim.get_data()[mask_data != 0]
# construct the blobs hierarchy
nroi = hroi.HROI_as_discrete_domain_blobs(domain, glm_data.ravel(),
                                          threshold=THETA3D, smin=SMIN3D)

#------------------------------------------------------------
### Extract blobs maps as data arrays
blobs_labels = -np.zeros(domain.size)
blobs_means = -np.zeros(domain.size)
if nroi != None:
    nroi.make_feature('activation', glm_data.ravel())
    bfm = nroi.representative_feature('activation')
    for k in range(nroi.k):
        blobs_labels[nroi.label == k] = k
        blobs_means[nroi.label == k] = bfm[k]
# saving the blobs image, i.e. a label image
label_data = np.zeros(mask_data.shape)
label_data[mask_data != 0] = (blobs_labels + 1)
label_image = Nifti1Image(label_data, mask.get_affine())
label_image.get_header()['descrip'] = 'blob image extracted from %s' \
                                      %glm_data_path 
save(label_image, os.path.join(swd,"blob.nii"))
print "Wrote the blobs label image in %s" \
      %os.path.join(swd, "blob.nii")
# saving the image of the average-signal-per-blob
avg_data = np.zeros(mask_data.shape)
avg_data[mask_data != 0] = blobs_means
avg_image = Nifti1Image(avg_data, mask.get_affine())
avg_image.get_header()['descrip'] = 'blob average signal extracted from %s' \
                                      %glm_data_path 
save(avg_image, os.path.join(swd,"bmap.nii"))
print "Wrote the blobs average signal image in %s" \
      %os.path.join(swd, "bmap.nii")

#------------------------------------------------------------
### Extract end-blobs (or leaves) maps as data arrays
leaves_labels = -np.zeros(domain.size)
leaves_means = -np.zeros(domain.size)
if nroi != None:
    lroi = nroi.reduce_to_leaves()
    bfm = lroi.representative_feature('activation')
    for k in range(lroi.k):
        leaves_labels[lroi.label == k] = k
        leaves_means[lroi.label == k] = bfm[k]
# saving the end-blobs image, i.e. a label image
label_data = np.zeros(mask_data.shape)
label_data[mask_data != 0] = (leaves_labels + 1)
label_image = Nifti1Image(label_data, mask.get_affine())
label_image.get_header()['descrip'] = 'blob image extracted from %s' \
                                      %glm_data_path 
save(label_image, os.path.join(swd,"leaves.nii"))
print "Wrote the end-blobs label image in %s" \
      %os.path.join(swd, "leaves.nii")
# saving the end-blobs image of the average-signal-per-blob
avg_data = np.zeros(mask_data.shape)
avg_data[mask_data != 0] = leaves_means
avg_image = Nifti1Image(avg_data, mask.get_affine())
avg_image.get_header()['descrip'] = 'blob average signal extracted from %s' \
                                      %glm_data_path 
save(avg_image, os.path.join(swd,"mleaves.nii"))
print "Wrote the end-blobs average signal image in %s" \
      %os.path.join(swd, "mleaves.nii")
