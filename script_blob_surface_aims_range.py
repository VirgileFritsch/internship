"""
Script to create blobs on a texture 
"""

import numpy as np
import os.path as op
import nipy.neurospin.graph.field as ff
from nipy.neurospin.spatial_models import hroi
import neurospy.bvfunc.mesh_processing as mep

from soma import aims


def extract_blobs_surface(input_mesh_graph, activation_data, theta):
    """

    """
    edges = input_mesh_graph.get_edges()
    F = ff.Field(input_mesh_graph.V, edges, input_mesh_graph.get_weights(),
                 activation_data)

    affine = np.eye(4)
    shape = ()
    disc = np.reshape(np.arange(F.V), (F.V, 1)).astype(int)
    nroi = hroi.NROI_from_field(F, affine, shape, disc, refdim=0,
                                th=theta, smin=5)
    label = -np.ones(np.shape(activation_data))
    idx = nroi.discrete_features['index']
    for k in range(nroi.k):
        label[idx[k]] =  k

    return label


# set the paths
path = '/data/home/virgile/virgile_internship/'
left_mesh = path + 's12069/surf/lh.r.aims.white.mesh'

# contrast 
c = 22
img_id = 1
left_fun_text = path + 'experiments/diffusion_smoothing_range/' + \
                'SL_spmT_%04d_%04d.tex' %(c, img_id)

# output path
opath = '/data/home/virgile/virgile_internship/experiments/blobs_range/'
tex_labels_name = opath + 'left_blobs_%04d_%04d.tex' %(c, img_id)

# threshold
theta = 3.3

# read the mesh
R = aims.Reader()
mesh = R.read(left_mesh)
vertices = mesh.vertex()
input_mesh_graph = mep.mesh_to_graph(mesh)

for img_id in range(1, 21):
    left_fun_text = path + 'experiments/diffusion_smoothing_range/' + \
                'SL_spmT_%04d_%04d.tex' %(c, img_id)
    tex_labels_name = opath + 'left_blobs_%04d_%04d.tex' %(c, img_id)
        
    #import Texture
    Funtex = R.read(left_fun_text)
    Fun = np.array(Funtex[0].data())
    Fun[np.isnan(Fun)] = 0
    
    label = extract_blobs_surface(input_mesh_graph, Fun, theta)
    
    W = aims.Writer()
    
    nnode = input_mesh_graph.V
    textureW = aims.TimeTexture_FLOAT()
    tex1 = textureW[0] # First Time sample
    tex1.reserve(nnode)
    for i in range(nnode): tex1.append( label[i] )
    W.write( textureW, tex_labels_name)


