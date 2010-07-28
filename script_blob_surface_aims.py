"""
Script to create blobs on a texture 
"""

import numpy as np
import os.path as op
import sys

import nipy.neurospin.graph.field as ff
from nipy.neurospin.spatial_models import hroi
import neurospy.bvfunc.mesh_processing as mep

from soma import aims

from database_archi import *

#-------------------------------------------------
#--- Left hemisphere processing ------------------
#-------------------------------------------------
print "Processing left hemisphere.",
sys.stdout.flush()
# read the mesh
R = aims.Reader()
lmesh = R.read(lmesh_path_aims)
vertices = lmesh.vertex()
print ".",
sys.stdout.flush()
# read the contrast texture
ltex = R.read(glm_ltex_path)
Fun = np.array(ltex[0].data())
Fun[np.isnan(Fun)] = 0
G = mep.mesh_to_graph(lmesh)
F = ff.Field(G.V, G.get_edges(), G.get_weights(), Fun)
print ".",
sys.stdout.flush()
"""
if theta<Fun.max():
    idx,height, father,label = F.threshold_bifurcations(0, theta)
else:F
    idx = []
    father = []
    label = -np.ones(np.shape(Fun))
"""

# construct ROI
affine = np.eye(4)
shape = None
disc = np.reshape(np.arange(F.V), (F.V, 1)).astype(int)
nroi = hroi.NROI_from_field(F, affine, shape, disc, refdim=0,
                            th=THETA, smin=SMIN)
label = -np.ones(np.shape(Fun))
if nroi is not None:
    # select either mean-labeled 2D blobs...
    if BLOB2D_TYPE == "mblobs":
        nroi.set_discrete_feature_from_index('average', Fun)
        leaves = nroi.reduce_to_leaves()
        idx = leaves.discrete_features['index']
        leaves_avg = leaves.discrete_to_roi_features('average')
        for k in range(leaves.k):
            label[idx[k]] =  leaves_avg[k,0]
    # ...or index-labeled ones
    else:
        leaves = nroi.reduce_to_leaves()
        idx = leaves.discrete_features['index']
        for k in range(leaves.k):
            label[idx[k]] =  k
print ".",
sys.stdout.flush()

# write output 2D blobs texture
W = aims.Writer()
nnode = G.V
textureW = aims.TimeTexture_FLOAT()
tex = textureW[0] # First Time sample
tex.reserve(nnode)
for i in range(nnode): tex.append(label[i])
W.write(textureW, blobs2D_ltex_path)
print "done."
sys.stdout.flush()

#-------------------------------------------------
#--- Right hemisphere processing -----------------
#-------------------------------------------------
print "Processing right hemisphere.",
sys.stdout.flush()
# read the mesh
R = aims.Reader()
rmesh = R.read(rmesh_path_aims)
vertices = rmesh.vertex()
print ".",
sys.stdout.flush()
# read the contrast texture
rtex = R.read(glm_rtex_path)
Fun = np.array(rtex[0].data())
Fun[np.isnan(Fun)] = 0
G = mep.mesh_to_graph(rmesh)
F = ff.Field(G.V, G.get_edges(), G.get_weights(), Fun)
print ".",
sys.stdout.flush()

"""
if theta<Fun.max():
    idx,height, father,label = F.threshold_bifurcations(0, theta)
else:F
    idx = []
    father = []
    label = -np.ones(np.shape(Fun))
"""

# construct ROI
affine = np.eye(4)
shape = None
disc = np.reshape(np.arange(F.V), (F.V, 1)).astype(int)
nroi = hroi.NROI_from_field(F, affine, shape, disc, refdim=0,
                            th=THETA, smin=SMIN)
label = -np.ones(np.shape(Fun))
if nroi is not None:
    # select either mean-labeled 2D blobs...
    if BLOB2D_TYPE == "mblobs":
        nroi.set_discrete_feature_from_index('average', Fun)
        leaves = nroi.reduce_to_leaves()
        idx = leaves.discrete_features['index']
        leaves_avg = leaves.discrete_to_roi_features('average')
        for k in range(leaves.k):
            label[idx[k]] =  leaves_avg[k,0]
    # ...or index-labeled ones
    else:
        leaves = nroi.reduce_to_leaves()
        idx = leaves.discrete_features['index']
        for k in range(leaves.k):
            label[idx[k]] =  k
print ".",
sys.stdout.flush()

# write output 2D blobs texture
W = aims.Writer()
nnode = G.V
textureW = aims.TimeTexture_FLOAT()
tex = textureW[0] # First Time sample
tex.reserve(nnode)
for i in range(nnode): tex.append(label[i])
W.write(textureW, blobs2D_rtex_path)
print "done."
sys.stdout.flush()
