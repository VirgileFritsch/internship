"""
This script performs a diffusion smoothing on brain surface
using heat equation.
It implements the method described in the Chung & Taylor paper :
Diffusion Smoothing on Brain Surface via Finite Element Method.

Author: Virgile Fritsch, 2010

"""

import numpy as np
import scipy as sp
import sys

import library_smoothing as smooth

from nipy.neurospin.glm_files_layout import tio
from gifti import loadImage

from database_archi import *

def get_edges_from_polygons(polygons, vertices):
    """

    """
    nb_edges = 3 * polygons.shape[0]
    edges = np.zeros((nb_edges, 2))
    # get the polygons edges as tuples
    permutator = np.array([(0,0,1),(1,0,0),(0,1,0)], dtype=int)
    edges[:,0] = np.ravel(polygons)
    edges[:,1] = np.ravel(np.dot(polygons, permutator))
    ind = np.lexsort((edges[:,1], edges[:,0]))
    edges = edges[ind]

    return edges

# -----------------------------------------------------------
# --------- Processing hemispheres separately ---------------
# -----------------------------------------------------------
for hemisphere in ['left','right']:
    print "Processing %s hemisphere:" %hemisphere
    sys.stdout.flush()
    if hemisphere == "right":
        mesh_path = rmesh_path_gii
        orig_tex_path = orig_rtex_path
        smoothed_tex_path = smoothed_rtex_path
    else:
        mesh_path = lmesh_path_gii
        orig_tex_path = orig_ltex_path
        smoothed_tex_path = smoothed_ltex_path
    
    ### Get information from input mesh
    # /!\ fixme : check that input_mesh is a triangular mesh
    print "  * Getting information from input mesh"
    sys.stdout.flush()
    # load mesh
    input_mesh = loadImage(mesh_path)
    # get vertices
    c, n, t = input_mesh.getArrays()
    vertices = c.getData()
    nb_vertices = vertices.shape[0]
    # get polygons
    polygons = t.getData()
    nb_polygons = polygons.shape[0]
    # get edges
    edges = get_edges_from_polygons(polygons, vertices)
    
    ### Get information from input texture
    # /!\ fixme : check that input_tex corresponds to the mesh
    print "  * Getting information from input texture"
    sys.stdout.flush()
    # load texture
    input_tex = tio.Texture(orig_tex_path).read(orig_tex_path)
    # get activation data
    activation_data = input_tex.data
    activation_data[np.isnan(activation_data)] = 0
        
    ### Construct the weights matrix
    print "  * Computing the weights matrix"
    sys.stdout.flush()
    weights_matrix = smooth.compute_weights_matrix(polygons, vertices, edges)
        
    ### Define the Laplace-Beltrami operator
    LB_operator = smooth.define_LB_operator(weights_matrix)
        
    ### Compute the number of iterations needed
    N, dt = smooth.compute_smoothing_parameters(weights_matrix, FWHM)
        
    ### Apply smoothing
    print "  * Smoothing...(FWHM = " + str(FWHM) + ")"
    sys.stdout.flush()
    smoothed_activation_data = smooth.diffusion_smoothing(activation_data,
                                                          LB_operator, N, dt)
    
    ### Write smoothed data into a new texture file
    print "  * Writing output texture"
    sys.stdout.flush()
    # create a new texture object
    output_tex = tio.Texture(smoothed_tex_path, data=smoothed_activation_data)
    # write the file
    output_tex.write()

"""
# -----------------------------------------------------------
# --------- DEBUG -------------------------------------------
# -----------------------------------------------------------

max_neighbors_values = np.array([])
for p in range(nb_vertices):
    p_neighbors = np.concatenate(([p], edges[edges[:,0]==p, 1]), 1)
    p_neighbors_values = smoothed_activation_data[0,:][p_neighbors]
    p_max_neighbor = np.amin(p_neighbors_values)
    max_neighbors_values = \
            np.concatenate((max_neighbors_values,[p_max_neighbor]), 1)

### Create a map showing the reluctant vertices
diff = smoothed_activation_data[1,:] - max_neighbors_values
diff[diff > 0] = 0

# create a new texture object
output_tex_err = aims.TimeTexture_FLOAT()
# add data to the texture object
tex_0 = output_tex_err[0]
tex_0.reserve(nb_vertices)
for i in np.arange(nb_vertices): tex_0.append(diff[i])

# write the file
W = aims.Writer()
output_tex_err_path = output_tex_folder_path + 'lh.esaloc1_FWHM5_sbs_biased.tex'
W.write(output_tex_err, output_tex_err_path)
"""

