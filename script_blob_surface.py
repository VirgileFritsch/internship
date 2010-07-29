"""
Script to create blobs on a texture 
"""

import numpy as np
import os
import sys

import nipy.neurospin.graph.field as ff
import nipy.neurospin.graph.graph as fg
from nipy.neurospin.spatial_models import hroi

from nipy.neurospin.glm_files_layout import tio
from gifti import loadImage

from database_archi import *

def mesh_to_graph(vertices, poly):
    """
    This function builds an fff graph from a mesh
    (Taken from nipy mesh_processing.py but removed the aims dependancy)
    """
    V = len(vertices)
    E = poly.shape[0]
    edges = np.zeros((3*E,2))
    weights = np.zeros(3*E)

    for i in range(E):
        sa = poly[i,0]
        sb = poly[i,1]
        sc = poly[i,2]
        
        edges[3*i] = np.array([sa,sb])
        edges[3*i+1] = np.array([sa,sc])
        edges[3*i+2] = np.array([sb,sc])    
            
    G = fg.WeightedGraph(V, edges, weights)

    # symmeterize the graph
    G.symmeterize()

    # remove redundant edges
    G.cut_redundancies()

    # make it a metric graph
    G.set_euclidian(vertices)

    return G

#-------------------------------------------------
#--- Left hemisphere processing ------------------
#-------------------------------------------------
print "Processing left hemisphere.",
sys.stdout.flush()
# read the mesh
lmesh = loadImage(lmesh_path_gii)
if SUBJECT == "group":
    c, t = lmesh.getArrays()
else:
    c, n, t = lmesh.getArrays()
lvertices = c.getData()
ltriangles = t.getData()
print ".",
sys.stdout.flush()
# read the contrast texture
ltex = tio.Texture(glm_ltex_path).read(glm_ltex_path)
Fun = np.array(ltex.data)
Fun[np.isnan(Fun)] = 0
G = mesh_to_graph(lvertices, ltriangles)
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

# write output 2D blobs textur
if not os.path.exists('%s/%s' %(MAIN_PATH, BLOBS2D_SUBDIR)):
    os.makedirs('%s/%s' %(MAIN_PATH, BLOBS2D_SUBDIR))
output_tex = tio.Texture(blobs2D_ltex_path, data=label)
output_tex.write()
print "done."
sys.stdout.flush()

#-------------------------------------------------
#--- Right hemisphere processing -----------------
#-------------------------------------------------
print "Processing right hemisphere.",
sys.stdout.flush()
# read the mesh
rmesh = loadImage(rmesh_path_gii)
if SUBJECT == "group":
    c, t = rmesh.getArrays()
else:
    c, n, t = rmesh.getArrays()
rvertices = c.getData()
rtriangles = t.getData()
print ".",
sys.stdout.flush()
# read the contrast texture
rtex = tio.Texture(glm_rtex_path).read(glm_rtex_path)
Fun = np.array(rtex.data)
Fun[np.isnan(Fun)] = 0
G = mesh_to_graph(rvertices, rtriangles)
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
if not os.path.exists('%s/%s' %(MAIN_PATH, BLOBS2D_SUBDIR)):
    os.makedirs('%s/%s' %(MAIN_PATH, BLOBS2D_SUBDIR))
output_tex = tio.Texture(blobs2D_rtex_path, data=label)
output_tex.write()
