"""
This scipt extracts the blobs from a 2D texture.

This creates as output two textures with the terminal blobs only
(activation means and labels).

Author : Virgile Fritsch, 2010,
         adapted from original Bertrand Thirion's script, 2009
         
"""

import os, sys
import numpy as np

from nipy.neurospin.spatial_models import hroi
from nipy.neurospin.glm_files_layout import tio
from nipy.neurospin.spatial_models.discrete_domain import domain_from_mesh

# -----------------------------------------------------------
# --------- Paths and Parameters ----------------------------
# -----------------------------------------------------------
from database_archi import *

#----- Path to the meshes
#rmesh_path_gii = ?
#lmesh_path_gii = ?

#----- Path to the textures from which to form blobs
#glm_rtex_path = ?
#glm_ltex_path = ?

#----- Blob-forming threshold (min significant intensity value)
#THETA = ?
#----- Size threshold on blobs
#SMIN = ?

#----- Type of leaves (end-blobs) to write :
# - "blobs": index-labelled blobs
# - "mblobs": mean-labelled blobs
#BLOB2D_TYPE = ?

#----- Output path for the end-blobs textures
#smoothed_rtex_path = ?
#smoothed_ltex_path = ?


#-------------------------------------------------
#--- Right hemisphere processing -----------------
#-------------------------------------------------
print "Processing right hemisphere."
sys.stdout.flush()
# Compute the nested roi object
domain = domain_from_mesh(rmesh_path_gii)
# Get initial texture (from which to create the blobs)
rtex = tio.Texture(glm_rtex_path).read(glm_rtex_path)
# Create blobs
nroi = hroi.HROI_as_discrete_domain_blobs(
    domain, rtex.data, threshold=THETA, smin=SMIN)

# Get the end-blobs textures (labels and activation mean)
leaves_labels = -np.ones(domain.size)
leaves_means = -np.ones(domain.size)
if nroi is not None:
    nroi.make_feature('activation', rtex.data)
    leaves = nroi.reduce_to_leaves()
    leaves_avg = leaves.representative_feature('activation')
    for k in range(leaves.k):
        leaves_means[leaves.label == k] = k
        leaves_labels[leaves.label == k] = leaves_avg[k]

# Write output 2D blobs texture
if not os.path.exists('%s/%s' %(MAIN_PATH, BLOBS2D_SUBDIR)):
    os.makedirs('%s/%s' %(MAIN_PATH, BLOBS2D_SUBDIR))
# we only write one kind of texture (mean or label)
if BLOB2D_TYPE == "mblobs":
    output_tex = tio.Texture(blobs2D_rtex_path, data=leaves_means)
else:
    output_tex = tio.Texture(blobs2D_rtex_path, data=leaves_labels)    
output_tex.write()

#-------------------------------------------------
#--- Left hemisphere processing ------------------
#-------------------------------------------------
print "Processing left hemisphere."
sys.stdout.flush()
# Compute the nested roi object
domain = domain_from_mesh(lmesh_path_gii)
# Get initial texture (from which to create the blobs)
ltex = tio.Texture(glm_ltex_path).read(glm_ltex_path)
# Create blobs
nroi = hroi.HROI_as_discrete_domain_blobs(
    domain, ltex.data, threshold=THETA, smin=SMIN)

# Get the end-blobs textures (labels and activation mean)
leaves_labels = -np.ones(domain.size)
leaves_means = -np.ones(domain.size)
if nroi is not None:
    nroi.make_feature('activation', ltex.data)
    leaves = nroi.reduce_to_leaves()
    leaves_avg = leaves.representative_feature('activation')
    for k in range(leaves.k):
        leaves_means[leaves.label == k] = k
        leaves_labels[leaves.label == k] = leaves_avg[k]

# Write output 2D blobs texture
if not os.path.exists('%s/%s' %(MAIN_PATH, BLOBS2D_SUBDIR)):
    os.makedirs('%s/%s' %(MAIN_PATH, BLOBS2D_SUBDIR))
# we only write one kind of texture (mean or label)
if BLOB2D_TYPE == "mblobs":
    output_tex = tio.Texture(blobs2D_ltex_path, data=leaves_means)
else:
    output_tex = tio.Texture(blobs2D_ltex_path, data=leaves_labels)    
output_tex.write()
