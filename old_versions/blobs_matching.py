"""


Author: Virgile Fritsch, 2010

"""

import re
import numpy as np
import scipy as sp
import scikits as sck
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pdb

from soma import aims

def get_2D_blobs_centers(right_input_2D_blobs, right_vertices,
                         left_input_2D_blobs, left_vertices):
    """
    
    
    Note
    ----
    AIMS dependant
    
    """
    # process right hemisphere blobs
    right_blobs = np.asarray(right_input_2D_blobs[0].data())
    nb_right_blobs = int(np.amax(right_blobs)) + 1
    right_blobs_centers = np.zeros((nb_right_blobs, 3))
    for blob in np.arange(nb_right_blobs):
        blob_vertices = right_vertices[np.where(right_blobs == blob)]
        blob_center = get_2D_blob_center(blob_vertices)
        right_blobs_centers[blob,:] = blob_center

    # process left hemisphere blobs
    left_blobs = np.asarray(left_input_2D_blobs[0].data())
    nb_left_blobs = int(np.amax(left_blobs)) + 1
    left_blobs_centers = np.zeros((nb_left_blobs, 3))
    for blob in np.arange(nb_left_blobs):
        blob_vertices = left_vertices[np.where(left_blobs == blob)]
        blob_center = get_2D_blob_center(blob_vertices)
        left_blobs_centers[blob,:] = blob_center

    blobs_centers = np.concatenate((right_blobs_centers, left_blobs_centers),0)
    
    return blobs_centers


def get_2D_blob_center(blob_voxels):
    """
    To be improved (take blob geometry into account)
    """
    return np.mean(blob_voxels, 0)


def get_3D_blobs_centers(input_3D_blobs):
    """
    
    
    Note
    ----
    AIMS dependant
    
    """
    pattern = re.compile("leaves_(\d*)_.*")
    R = aims.Reader()
    blobs_centers = np.zeros((input_3D_blobs.vertices().size(), 3))
    for blob_path in input_3D_blobs.vertices():
        blob = R.read(blob_path['name'])
        blob_id = int(re.findall(pattern, blob_path['name'])[0])
        blob_voxels = np.asarray(blob.vertex())
        blob_center = get_3D_blob_center(blob_voxels)
        blobs_centers[blob_id-1,:] = blob_center
    
    return blobs_centers


def get_3D_blob_center(blob_voxels):
    """
    To be improved (take blob geometry into account)
    """
    return np.mean(blob_voxels, 0)

# -----------------------------------------------------------
# --------- User's parameters -------------------------------
# -----------------------------------------------------------

# -----------------------------------------------------------
# --------- Environnement settings --------------------------
# -----------------------------------------------------------

### Set the paths

# input paths
input_mesh_folder_path = '/data/home/virgile/virgile_internship/s12069/surf/'
input_lmesh_path = input_mesh_folder_path + 'lh.r.aims.white.mesh'
input_rmesh_path = input_mesh_folder_path + 'rh.r.aims.white.mesh'
input_2D_blobs_folder_path = '/data/home/virgile/virgile_internship/s12069/experiments/test_aims/'
input_2D_blobs_endfilename = 'audio-video_z_map_blobs_FWHM5_smin5_theta3d3.tex'
right_input_2D_blobs_path = input_2D_blobs_folder_path + 'right_' + \
                            input_2D_blobs_endfilename
left_input_2D_blobs_path = input_2D_blobs_folder_path + 'left_' + \
                           input_2D_blobs_endfilename

input_3D_blobs_folder_path = '/data/home/virgile/virgile_internship/s12069/experiments/smoothed_05/audio-video/'
input_3D_blobs_path = input_3D_blobs_folder_path + 'leaves_.arg'

# -----------------------------------------------------------
# --------- Script starts -----------------------------------
# -----------------------------------------------------------

### Get information from input mesh
# load mesh
R1 = aims.Reader()
input_lmesh = R1.read(input_lmesh_path)
input_rmesh = R1.read(input_rmesh_path)
# get vertices
right_vertices = np.asarray(input_rmesh.vertex())
left_vertices = np.asarray(input_lmesh.vertex())

### Get the list of 2D blobs centers
R2D = aims.Reader()
right_input_2D_blobs = R2D.read(right_input_2D_blobs_path)
left_input_2D_blobs = R2D.read(left_input_2D_blobs_path)
blobs2D_centers = get_2D_blobs_centers(right_input_2D_blobs, right_vertices,
                                       left_input_2D_blobs, left_vertices)
nb_2D_blobs = blobs2D_centers.shape[0]

### Get the list of 3D blobs centers
R3D = aims.Reader()
input_3D_blobs = R3D.read(input_3D_blobs_path)
blobs3D_centers = get_3D_blobs_centers(input_3D_blobs)
nb_3D_blobs = input_3D_blobs.vertices().size()

### Compute distances between each pair of 3D-2D centers
blobs3D_centers_aux = np.tile(blobs3D_centers.reshape(-1,1,3),
                              (1,blobs2D_centers.shape[0],1))
blobs2D_centers_aux = np.tile(blobs2D_centers,
                              (blobs3D_centers.shape[0],1,1))
dist = ((blobs3D_centers_aux - blobs2D_centers_aux)**2).sum(2)

#####################################################
### Match each 2D blob with one or several 3D blob(s)
#####################################################
match_2D_blobs = np.argmin(dist, 0)

fig = plt.figure(1)
ax = Axes3D(fig)
ax.set_aspect(0.5, 'datalim')

for blob3D in np.arange(nb_3D_blobs):
    neighbors = dist[blob3D, np.where(match_2D_blobs == blob3D)[0]]
    std_deviation = np.sqrt(np.var(neighbors))
    mean_length = np.mean(neighbors)
    for blob2D in np.where(match_2D_blobs == blob3D)[0]:
        edge_length = ((blobs3D_centers[blob3D,:] - \
                 blobs2D_centers[blob2D,:])**2).sum()
        if edge_length - mean_length > std_deviation or edge_length > 2000:
            edge_type = 'red'
#            print "Blob %d is far from its corresponding 3D blob (%d)" \
#                  %(blob2D, blob3D)
        else:
            edge_type = 'blue'
        ax.plot3D([blobs3D_centers[blob3D,0],blobs2D_centers[blob2D,0]],
                  [blobs3D_centers[blob3D,1],blobs2D_centers[blob2D,1]],
                  [blobs3D_centers[blob3D,2],blobs2D_centers[blob2D,2]],
                  color=edge_type, linestyle='solid')

### Define an association probability (gaussian kernel)
sigma = 5.
dist_gauss = np.exp(-0.5 * (dist / sigma**2))
norm_aux = np.tile(dist_gauss.sum(0), nb_3D_blobs).reshape((nb_3D_blobs,
                                                            nb_2D_blobs))
norm_aux[norm_aux == 0] = 1
proba = dist_gauss / norm_aux

### Define another association probability (gaussian kernel)
### taking possible lonelyness into account
FWHM = 10.
sigma  = FWHM/(2 * np.sqrt(2 * np.log(2)))
dist_gauss_2 = np.exp(-0.5 * (dist / sigma**2))
# function "1-max" can be replaced by another
coeff = 0.1
last_row = coeff * (1 - np.amax(dist_gauss_2, 0))
dist_gauss_2 = np.vstack((dist_gauss_2, last_row))
norm_aux = np.tile(dist_gauss_2.sum(0), nb_3D_blobs+1).reshape((nb_3D_blobs+1,
                                                              nb_2D_blobs))
norm_aux[norm_aux == 0] = 1
proba_2 = dist_gauss_2 / norm_aux
proba = proba_2

### Plot the results
fig = plt.figure(2)
ax = Axes3D(fig)
ax.set_aspect(0.5, 'datalim')

"""
for blob3D in np.arange(nb_3D_blobs):
    for blob2D in np.arange(nb_2D_blobs):
        if proba[blob3D, blob2D] > 1e-1:
            ax.plot3D([blobs3D_centers[blob3D,0],blobs2D_centers[blob2D,0]],
                      [blobs3D_centers[blob3D,1],blobs2D_centers[blob2D,1]],
                      [blobs3D_centers[blob3D,2],blobs2D_centers[blob2D,2]],
                      color='blue', linestyle='solid')
        if proba[blob3D, blob2D] > 5e-1:
             print "3D blob %d is thought to be related to 2D blob %d at %f%%" %(blob3D+1, blob2D, 100*proba[blob3D, blob2D])
"""

for blob2D in np.arange(nb_2D_blobs):
    for blob3D in np.arange(nb_3D_blobs):
        if proba[blob3D, blob2D] > 7.5e-1:
            ax.plot3D([blobs3D_centers[blob3D,0],blobs2D_centers[blob2D,0]],
                      [blobs3D_centers[blob3D,1],blobs2D_centers[blob2D,1]],
                      [blobs3D_centers[blob3D,2],blobs2D_centers[blob2D,2]],
                      color='red', linestyle='solid')
            print "2D blob %d is thought to be related to 3D blob %d at %f%%" %(blob2D, blob3D+1, 100*proba[blob3D, blob2D])
        elif proba[blob3D, blob2D] > 5.0e-1:
            ax.plot3D([blobs3D_centers[blob3D,0],blobs2D_centers[blob2D,0]],
                      [blobs3D_centers[blob3D,1],blobs2D_centers[blob2D,1]],
                      [blobs3D_centers[blob3D,2],blobs2D_centers[blob2D,2]],
                      color='blue', linestyle='solid')
            print "2D blob %d is thought to be related to 3D blob %d at %f%%" %(blob2D, blob3D+1, 100*proba[blob3D, blob2D])
    significant_neighbors = np.where(proba[:,blob2D] > 1.e-2)[0]
    print "Blob 2D", blob2D, ":"
    significant_neighbors_id = significant_neighbors + 1
    print np.vstack((significant_neighbors_id,
                     100.*proba[significant_neighbors,blob2D])).T, "\n"

#plt.show()

