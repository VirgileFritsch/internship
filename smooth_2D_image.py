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

# -----------------------------------------------------------
# --------- Paths -------------------------------------------
# -----------------------------------------------------------
from database_archi import *

# Path to the meshes
#rmesh_path_gii = ?
#lmesh_path_gii = ?

# Path to the original textures
#orig_rtex_path = ?
#orig_ltex_path = ?

# Output path for the smoothed images
#smoothed_rtex_path = ?
#smoothed_ltex_path = ?


# -----------------------------------------------------------
# --------- Parameters  -------------------------------------
# -----------------------------------------------------------

# Amount of (gaussian) smoothing
#FWHM = ?

# -----------------------------------------------------------
# --------- Define some useful functions --------------------
# -----------------------------------------------------------
def get_edges_from_polygons(polygons, vertices):
    """Builds a mesh edges list from its polygons and vertices.

    """
    nb_edges = 3 * polygons.shape[0]
    edges = np.zeros((nb_edges, 2))
    # get the polygons edges as tuples
    permut = np.array([(0,0,1),(1,0,0),(0,1,0)], dtype=int)
    edges[:,0] = np.ravel(polygons)
    edges[:,1] = np.ravel(np.dot(polygons, permut))
    ind = np.lexsort((edges[:,1], edges[:,0]))
    edges = edges[ind]

    return edges

# -----------------------------------------------------------
# --------- Process hemispheres separately ------------------
# -----------------------------------------------------------
for hemisphere in ['left','right']:
    print "Smoothing: processing %s hemisphere:" %hemisphere
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
    input_mesh = loadImage(mesh_path)
    input_mesh_arrays = input_mesh.getArrays()
    if len(input_mesh_arrays) == 2:
        c, t = input_mesh_arrays
    elif len(input_mesh_arrays) == 3:
        c, n, t = input_mesh_arrays
    else:
        raise ValueError("Error during gifti data extraction")
    vertices = c.getData()
    nb_vertices = vertices.shape[0]
    polygons = t.getData()
    nb_polygons = polygons.shape[0]
    edges = get_edges_from_polygons(polygons, vertices)
    
    ### Get information from input texture
    # /!\ fixme : check that input_tex corresponds to the mesh
    print "  * Getting information from input texture"
    sys.stdout.flush()
    input_tex = tio.Texture(orig_tex_path).read(orig_tex_path)
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
    print "  * Smoothing...(FWHM = %g)" %FWHM
    sys.stdout.flush()
    smoothed_activation_data = smooth.diffusion_smoothing(
        activation_data, LB_operator, N, dt)
    
    ### Write smoothed data into a new texture file
    print "  * Writing output texture"
    sys.stdout.flush()
    output_tex = tio.Texture(smoothed_tex_path, data=smoothed_activation_data)
    output_tex.write()

