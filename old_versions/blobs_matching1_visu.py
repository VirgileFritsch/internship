"""


Author: Virgile Fritsch, 2010

"""

import re, sys
import numpy as np
import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.text as mtxt
from mpl_toolkits.mplot3d import Axes3D

import pdb

from nipy.neurospin.glm_files_layout import tio
from nipy.io.imageformats import load
from nipy.neurospin.spatial_models.roi import DiscreteROI, MultipleROI
from nipy.neurospin.viz_tools.maps_3d import affine_img_src
import enthought.mayavi.mlab as mayavi
from gifti import loadImage
from scikits.learn import BallTree

from database_archi import *

def get_2D_blobs_centers(right_blobs, right_vertices,
                         left_blobs, left_vertices):
    """
    
    """
    # process right hemisphere blobs
    nb_right_blobs = int(np.amax(right_blobs)) + 1
    right_blobs_centers = np.zeros((nb_right_blobs, 3))
    for blob in np.arange(nb_right_blobs):
        blob_vertices = right_vertices[np.where(right_blobs == blob)]
        blob_center = get_2D_blob_center(blob_vertices)
        right_blobs_centers[blob,:] = blob_center

    # process left hemisphere blobs
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

    current version = 1 : return a point that is indeeed part of the blob
    
    """
    blob_center = np.mean(blob_voxels, 0)
    center_neighbor = np.argmin(((blob_voxels - blob_center)**2).sum(1))
    #return blob_voxels[center_neighbor]
    return blob_center

def phi(dik):
    dik[dik > 150.] = 150.
    return (15./16.)*(1-((dik-75.)/75.)**2)**2


# eventually choose 3D blobs to show (from 1 to ...)
blobs3D_to_show = [4]
# eventually choose 2D blobs to show (from 0 to ...)
blobs2D_to_show = [2]

# -----------------------------------------------------------
# --------- Script starts -----------------------------------
# -----------------------------------------------------------

### Load left hemisphere data
lmesh = loadImage(lmesh_path_gii)
c, n, t  = lmesh.getArrays()
lvertices = c.getData()
triangles = t.getData()
blobs2D_ltex = tio.Texture(blobs2D_ltex_path).read(blobs2D_ltex_path)
alt_ltex = blobs2D_ltex.data.copy()
alt_ltex[alt_ltex != -1.] = 2.

### Load right hemisphere data
rmesh = loadImage(rmesh_path_gii)
c, n, t  = rmesh.getArrays()
rvertices = c.getData()
triangles = t.getData() 
blobs2D_rtex = tio.Texture(blobs2D_rtex_path).read(blobs2D_rtex_path)
alt_rtex = blobs2D_rtex.data.copy()
alt_rtex[alt_rtex != -1.] = 2.

### Get the list of 2D blobs centers
# /!\ fixme : select a 2D blob center that is actually part of the blob
blobs2D_centers = get_2D_blobs_centers(blobs2D_rtex.data, rvertices,
                                       blobs2D_ltex.data, lvertices)
nb_2D_blobs = blobs2D_centers.shape[0]

### Get the list of 3D blobs centers
blobs3D = load(blobs3D_path)
affine = blobs3D.get_affine()
shape = blobs3D.get_shape()
mroi = MultipleROI(affine=affine, shape=shape)
mroi.from_labelled_image(blobs3D_path)
mroi.compute_discrete_position()
blobs3D_centers = mroi.discrete_to_roi_features('position')

### Plot the 3D blobs
nb_3D_blobs = mroi.k
"""
data = blobs3D.get_data() > 4.
i = 3.
while data[data == True].size == 0 or i<=1.:
    data = blobs3D.get_data() > i
    i -= 1.
a = blobs3D.get_data()[data]
ijk = np.array(np.where(data)).T
nvox = ijk.shape[0]
rcoord = np.dot(np.hstack((ijk,np.ones((nvox,1)))), affine.T)
# plot glyphes
mayavi.points3d(rcoord[:,0], rcoord[:,1], rcoord[:,2], a,
                colormap="summer", scale_factor=1)
"""

data = blobs3D.get_data()
labels = sp.ndimage.measurements.label(data)[0]
#data = np.asarray(data).copy()
if blobs3D_to_show[0] != -2:
    data = np.asarray(data).copy()
    toto = data.copy()
    for k in blobs3D_to_show:
        mayavi.points3d(blobs3D_centers[k-1,0], blobs3D_centers[k-1,1],
                        blobs3D_centers[k-1,2], scale_factor=1)
        data[data == k] = 110.
    data[data < 100.] = -2.
blobs3D_mayavi_src = affine_img_src(data, affine) 
blobs3D_mayavi_src.image_data.point_data.add_array(
    labels.T.ravel().astype(data.dtype))
blobs3D_mayavi_src.image_data.point_data.get_array(1).name = 'labels'
blobs3D_mayavi_src.image_data.update()
src2 = mayavi.pipeline.set_active_attribute(blobs3D_mayavi_src, point_scalars='scalar')
contour = mayavi.pipeline.contour(src2)
contour2 = mayavi.pipeline.set_active_attribute(contour, point_scalars='labels')
mayavi.pipeline.surface(contour2, colormap='spectral')
# disable rendering for acceleration purpose
blobs3D_mayavi_src.scene.disable_render = True

#blobs3D_mayavi_warp = mayavi.pipeline.warp_scalar(blobs3D_mayavi_src)
#blobs3D_mayavi_normals = mayavi.pipeline.poly_data_normals(blobs3D_mayavi_warp)
#blobs3D_mayavi_surf = mayavi.pipeline.surface(blobs3D_mayavi_normals)
#voi = mayavi.pipeline.extract_grid(blobs3D_mayavi_src)
#blobs3D_mayavi_iso_surf = mayavi.pipeline.iso_surface(blobs3D_mayavi_src, contours=[1.], colormap="hot")


### Compute distances between each pair of (3D blobs)-(2D blobs centers)
dist = np.zeros((nb_2D_blobs, nb_3D_blobs))
dist_arg = np.zeros((nb_2D_blobs, nb_3D_blobs), dtype='int')
for k in np.arange(nb_3D_blobs):
    blob3D_vertices = mroi.discrete_features['position'][k]
    blob3D_vertices_aux = np.tile(blob3D_vertices.reshape(-1,1,3),
                              (1,nb_2D_blobs,1))
    blobs2D_centers_aux = np.tile(blobs2D_centers,
                              (blob3D_vertices.shape[0],1,1))
    dist_all = ((blob3D_vertices_aux - blobs2D_centers_aux)**2).sum(2)
    dist_arg[:,k] = np.argmin(dist_all, 0)
    dist[:,k] = np.amin(dist_all, 0)

#####################################################
### Match each 2D blob with one or several 3D blob(s)
#####################################################
dist_display = np.zeros((nb_2D_blobs, nb_3D_blobs+1))
dist_display[:,0:-1] = dist.copy()
dist_display = np.sqrt(dist_display)
dist_display[:,nb_3D_blobs] = -1.

### Apply the model to data
gamma = (0.9/nb_3D_blobs)
## H0 (no link)
phi_all = phi(dist_display[:,0:-1])
probaH0 = np.tile(phi_all.prod(1),(phi_all.shape[1],1)).T
## H1 (link)
# "exponential" part
sigma  = 5.
dist_exp = np.exp(-0.5 * (dist / sigma**2))
#Zi = 1. / (phi_all.prod(1) * (dist_exp/phi_all).sum(1))
Zi = 2. / (sigma * np.sqrt(2. * np.pi))
#norm_aux = np.repeat(Zi, nb_3D_blobs).reshape((nb_2D_blobs, nb_3D_blobs))
#norm_aux[norm_aux == 0.] = 1.
proba_exp = dist_exp * Zi
# "volume repartition" part
phi_all[phi_all == 0.] = 1.
proba_rep = probaH0 / phi_all
# combine the two parts
probaH1 = proba_exp * proba_rep
## "final" proba
proba = np.zeros((nb_2D_blobs, nb_3D_blobs+1))
proba[:,0:-1] = (probaH1 * gamma) / \
                ((1.-nb_3D_blobs*gamma)*probaH0 + \
                 gamma*np.tile(probaH1.sum(1), (nb_3D_blobs,1)).T)
proba[:,nb_3D_blobs] = (probaH0[:,0]*(1-nb_3D_blobs*gamma)) / \
                       ((1.-nb_3D_blobs*gamma)*probaH0[:,0] + \
                        gamma*probaH1.sum(1))


### Plot the results
# set matplotlib figure basis
fig = plt.figure(1)
ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=0.8)
blobs2D_x = 0.
blobs2D_spacing = 5
blobs2D_radius = 1.
blobs3D_x = 35.
blobs3D_radius = 1.
ax.text((blobs3D_x-blobs2D_x)/2, 6,
        "Subject %s, Contrast %s, gamma=%g" %(SUBJECT, CONTRAST, gamma),
        horizontalalignment='center')
ax.set_xlim(blobs2D_x-10, blobs3D_x+4)
ax.set_ylim(-blobs2D_spacing*np.amax([nb_2D_blobs-1, nb_3D_blobs-1]),
            2*blobs2D_radius)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
for blob2D in np.arange(nb_2D_blobs):
    p = mpatches.Circle((blobs2D_x,-(blobs2D_spacing*blob2D)), blobs2D_radius)
    ax.text(blobs2D_x-5, -(blobs2D_spacing*blob2D)-1, '%d'%blob2D,
            horizontalalignment='center')
    ax.add_patch(p)
blobs3D_pos = np.zeros(nb_3D_blobs)
for blob3D in np.arange(nb_3D_blobs):
    significant_neighbors = np.where(proba[:,blob3D] > 2.5e-1)[0]
    pos = 0.
    for i in significant_neighbors:
        pos += -(blobs2D_spacing*i)*proba[i,blob3D]
    pos /= proba[significant_neighbors,blob3D].sum()
    blobs3D_pos[blob3D] = pos
    p = mpatches.Circle((blobs3D_x,pos), blobs3D_radius)
    ax.text(blobs3D_x+5, pos-1, '%d'%(blob3D+1),
            horizontalalignment='center')
    ax.add_patch(p)
blobs3D_lines_colors = np.tile(['b','g','c','m','k','y'], 10)
# customize the output text results
res_file = open('./results/ver0/res.txt', 'w')
sys.stdout = res_file
print "### SUBJECT: %s" %SUBJECT
print "# Contrast: %s" %CONTRAST
print "# gamma: %g\n" %gamma
print "nb_2D_blobs = %d" %nb_2D_blobs
print "nb_3D_blobs = %d\n" %nb_3D_blobs

for blob2D in np.arange(nb_2D_blobs):
    for blob3D in np.arange(nb_3D_blobs):
        if proba[blob2D, blob3D] > 7.5e-1:
            ax.plot([blobs2D_x,blobs3D_x],
                    [-(blobs2D_spacing*blob2D),blobs3D_pos[blob3D]],
                    color=blobs3D_lines_colors[blob3D])
            point2D = blobs2D_centers[blob2D,:]
            blob3D_points = mroi.discrete_features['position'][blob3D]
            point3D = blob3D_points[dist_arg[blob2D, blob3D]]
            match = np.vstack((point2D, point3D)).T
            mayavi.plot3d(match[0,:], match[1,:], match[2,:],
                          color=(0.,1.,0.), tube_radius=0.2)
            print "2D blob %d is thought to be related to 3D blob %d at %f%%" %(blob2D, blob3D+1, 100*proba[blob2D, blob3D])
        elif proba[blob2D, blob3D] > 2.5e-1:
            ax.plot([blobs2D_x,blobs3D_x],
                    [-(blobs2D_spacing*blob2D),blobs3D_pos[blob3D]],
                    '--', color=blobs3D_lines_colors[blob3D])
            point2D = blobs2D_centers[blob2D,:]
            blob3D_points = mroi.discrete_features['position'][blob3D]
            point3D = blob3D_points[dist_arg[blob2D, blob3D]]
            match = np.vstack((point2D, point3D)).T
            mayavi.plot3d(match[0,:], match[1,:], match[2,:],
                          color=(0.,1.,1.), tube_radius=0.2)
    if proba[blob2D, nb_3D_blobs] > 7.5e-1:
        print "2D blob %d is thought to be related to 3D blob %d at %f%%" %(blob2D, nb_3D_blobs+1, 100*proba[blob2D, nb_3D_blobs])
    significant_neighbors = np.where(proba[blob2D,:] > 5.e-2)[0]
    print "Blob 2D", blob2D, ":"
    significant_neighbors_id = significant_neighbors + 1
    significant_neighbors_id[significant_neighbors_id == nb_3D_blobs + 1] = -1.
    print np.vstack((significant_neighbors_id,
                     100.*proba[blob2D,significant_neighbors],
                     dist_display[blob2D,significant_neighbors])).T
    """
    # post-treatment for ambigous blobs
    if significant_neighbors.size > 1:
        if blob2D <= np.amax(blobs2D_rtex.data):
            blob2D_vertices = rvertices[np.where(blobs2D_rtex.data == blob2D)]
        else:
            blob2D_vertices = lvertices[np.where(blobs2D_ltex.data == blob2D - np.amax(blobs2D_rtex.data) - 1)]
        i = 0
        new_dist = np.zeros(significant_neighbors.size)
        for j in significant_neighbors[significant_neighbors != nb_3D_blobs]:
            blob3D_vertices = mroi.discrete_features['position'][j]
            blob3D_ball = BallTree.BallTree(blob3D_vertices)
            new_dist[i] = np.mean(blob3D_ball.query(blob2D_vertices))
            i = i + 1
        new_dist[i] = np.sqrt(d0)
        print "Refinement:"
        new_dist_aux = np.exp(-0.5*(new_dist/sigma)**2)
        print np.vstack((significant_neighbors_id,
                         100. * new_dist_aux / new_dist_aux.sum())).T
    """
    print "\n"

res_file.close()
sys.stdout = sys.__stdout__

# choose textures
ltex = blobs2D_ltex.data.copy()
rtex = blobs2D_rtex.data.copy()

### Plot left hemisphere
#ltex = alt_ltex
if blobs2D_to_show[0] != -2.:
    for b in blobs2D_to_show:
        mayavi.points3d(blobs2D_centers[b,0], blobs2D_centers[b,1],
                        blobs2D_centers[b,2],
                        scale_factor=1, color=(0.5, 0.5, 0.))
        if (b - np.amax(rtex) - 1) != -1.:
            ltex[ltex == b - np.amax(rtex) - 1] = 100.
    #ltex[ltex != 100.] = -1.
    ltex[(ltex != 100.)*(ltex != -1.)] = 10.
mayavi.triangular_mesh(lvertices[:,0], lvertices[:,1], lvertices[:,2],
                       triangles, scalars=ltex,
                       transparent=False, opacity=1.)

### Plot right hemisphere
#rtex = alt_rtex
if blobs2D_to_show[0] != -2.:
    for b in blobs2D_to_show:
        mayavi.points3d(blobs2D_centers[b,0], blobs2D_centers[b,1],
                        blobs2D_centers[b,2],
                        scale_factor=1, color=(0.5, 0.5, 0.))
        rtex[rtex == b] = 100.
    #rtex[rtex != 100.] = -1.
    rtex[(rtex != 100.)*(rtex != -1.)] = 10.
mayavi.triangular_mesh(rvertices[:,0], rvertices[:,1], rvertices[:,2],
                       triangles, scalars=rtex,
                       transparent=False, opacity=1.)

# enable rendering (because we have disabled it)
blobs3D_mayavi_src.scene.disable_render = False 
plt.show()

