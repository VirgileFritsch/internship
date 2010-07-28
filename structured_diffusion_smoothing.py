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

from soma import aims

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
        mesh_path_aims = rmesh_path_aims
        orig_tex_path = orig_rtex_path
        smoothed_tex_path = smoothed_rtex_path
    else:
        mesh_path_aims = lmesh_path_aims
        orig_tex_path = orig_ltex_path
        smoothed_tex_path = smoothed_ltex_path
    
    ### Get information from input mesh
    # /!\ fixme : check that input_mesh is a triangular mesh
    print "  * Getting information from input mesh"
    sys.stdout.flush()
    # load mesh
    R1 = aims.Reader()
    input_mesh = R1.read(mesh_path_aims)
    # get vertices
    vertices = np.asarray(input_mesh.vertex())
    nb_vertices = vertices.shape[0]
    # get polygons
    polygons = np.asarray(input_mesh.polygon())
    nb_polygons = polygons.shape[0]
    # get edges
    edges = get_edges_from_polygons(polygons, vertices)
    
    ### Get information from input texture
    # /!\ fixme : check that input_tex corresponds to the mesh
    print "  * Getting information from input texture"
    sys.stdout.flush()
    # load texture
    R2 = aims.Reader()
    input_tex = R2.read(orig_tex_path)
    # get activation data
    nb_time_samples = int(input_tex.size())
    activation_data = np.zeros((nb_time_samples, nb_vertices))
    for ts in np.arange(0, nb_time_samples):
        activation_data[ts,:] = np.asarray(input_tex[ts].data())
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
    output_tex = aims.TimeTexture_FLOAT()
    # add data to the texture object
    for ts in np.arange(0, smoothed_activation_data.shape[0]):
        tex_ts = output_tex[ts]
        tex_ts.reserve(nb_vertices)
        for i in np.arange(nb_vertices):
            tex_ts.append(smoothed_activation_data[ts,i])
                
    # write the file
    W = aims.Writer()
    W.write(output_tex, smoothed_tex_path)

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

