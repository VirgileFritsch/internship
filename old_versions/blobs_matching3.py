"""


Author: Virgile Fritsch, 2010

"""

import sys, copy
import numpy as np
import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.text as mtxt

import pdb

from nipy.neurospin.glm_files_layout import tio
from nipy.io.imageformats import load
from nipy.neurospin.spatial_models.roi import DiscreteROI, MultipleROI
from nipy.neurospin.viz_tools.maps_3d import affine_img_src
import nipy.neurospin.graph.field as ff
from nipy.neurospin.spatial_models import hroi
import enthought.mayavi.mlab as mayavi
from gifti import loadImage
from scikits.learn import BallTree

from database_archi import *

# -----------------------------------------------------------
# --------- Classes definition ------------------------------
# -----------------------------------------------------------

class Blob2D:
    '''
    Representation of a 2D blob with:
        - id: its id
        - ...
        
    '''
    # total number of 2D blobs (including parents and children)
    nb_blobs = 0
    # store all the 2D blobs instanced
    all_blobs = {}
    
    def __init__(self, vertices_list, vertices_id_list,
                 hemisphere, parent=None):
        '''
        '''
        # set blob id and increase total number of 2D blobs
        self.id = Blob2D.nb_blobs
        Blob2D.nb_blobs +=1
        Blob2D.all_blobs[self.id] = self

        # set blob parent
        self.parent = parent

        # set vertices list
        self.hemisphere = hemisphere
        self.vertices = vertices_list
        self.vertices_id = vertices_id_list

        # compute blob center
        self.center = np.mean(vertices_list, 0)

        # set blob parent
        self.parent = parent

        # set blob children
        self.children = []

        # set default associated 3D blob
        self.associated_3Dblob = -1.
        self.potentialy_associated_3Dblobs = []
        

    def associate_3Dblob(self, blob3D_id):
        '''
        '''
        self.associated_3Dblob = blob3D_id

    def associate_potential(self, blob3D_id):
        '''
        '''
        self.potentialy_associated_3Dblobs.append(blob3D_id)

    def set_association_probas(self, probas):
        '''
        '''
        self.association_probas = probas

    def add_children(self, children):
        '''
        '''
        self.children.append(children)

class Blob2DDisplay(Blob2D):
    '''
    Representation a of 2D blob for ploting
    '''
    # results displaying parameters
    default_x_pos = 0.
    spacing = 8
    radius = 1.
    # store all the 2D blobs instanced
    all_blobs = {}

    def __init__(self, blob2D):
        self.blob = blob2D
        Blob2DDisplay.all_blobs[blob2D.id] = self

        # set y position
        self.y_pos = -(blob2D.id * Blob2DDisplay.spacing) - 1

        # set x position
        if blob2D.children:
            self.x_pos = -5*Blob2DDisplay.spacing
        else:
            self.x_pos = Blob2DDisplay.default_x_pos

        # tag to avoid displaying the same blob several times
        self.already_displayed = False

    def set_ypos(self, new_y):
        '''
        '''
        if self.already_displayed:
            print "Warning: 2D blob %d already displayed" %self.id
        self.y_pos = new_y

    def display(self, ax, text_color='black', circle_color='blue'):
        if (not self.already_displayed):
            p = mpatches.Circle((self.x_pos,self.y_pos), Blob2DDisplay.radius,
                                color=circle_color)
            if self.blob.parent:
                display_color = 'red'
            else:
                display_color = text_color
            ax.text(self.x_pos-5, self.y_pos-2, '%d'%self.blob.id,
                    horizontalalignment='center', color=display_color)
            ax.add_patch(p)
        self.already_displayed = True


class Blob3DDisplay:
    '''
    Representation a of 3D blob for ploting
    '''
    # display parameters
    x_pos = 35.
    radius = 1.

    all_blobs = {}

    def __init__(self, id):
        '''
        '''
        self.id = id
        Blob3DDisplay.all_blobs[id] = self
        self.y_pos_values = 0.
        self.y_pos_coeffs = 0.

    def add_link(self, value, proba):
        '''
        '''
        self.y_pos_values += proba * value
        self.y_pos_coeffs += proba

    def get_pos(self):
        if self.y_pos_coeffs == 0:
            res = 0.
        else:
            res = self.y_pos_values / self.y_pos_coeffs

        return res


# -----------------------------------------------------------
# --------- Routines definition -----------------------------
# -----------------------------------------------------------

def compute_distances(blobs3D, blobs2D_list):
    blobs2D_centers = np.zeros((blobs2D_list.__len__(), 3))
    i = 0
    for blob in blobs2D_list:
        blobs2D_centers[i,:] = blob.center
        i += 1
    return compute_distances_aux(blobs3D, blobs2D_centers)

def compute_distances_aux(blobs3D, blobs2D_centers):
    nb_2Dblobs = blobs2D_centers.__len__()
    nb_3Dblobs = blobs3D.__len__()
    dist = np.zeros((nb_2Dblobs, nb_3Dblobs))
    dist_arg = np.zeros((nb_2Dblobs, nb_3Dblobs), dtype='int')
    for k in np.arange(nb_3Dblobs):
        blob3D_vertices = blobs3D[k]
        blob3D_vertices_aux = np.tile(blob3D_vertices.reshape(-1,1,3),
                                  (1,nb_2Dblobs,1))
        blobs2D_centers_aux = np.tile(blobs2D_centers,
                                  (blob3D_vertices.shape[0],1,1))
        dist_all = ((blob3D_vertices_aux - blobs2D_centers_aux)**2).sum(2)
        dist_arg[:,k] = np.argmin(dist_all, 0)
        dist[:,k] = np.amin(dist_all, 0)

    return dist, dist_arg

def compute_association_proba(blobs2D_list, nb_3Dblobs, gamma_aux, sigma, dist,
                              exclusion=False):
    nb_2Dblobs = blobs2D_list.__len__()
    
    dist2 = dist.copy()
    if exclusion:
        for i in np.arange(nb_2Dblobs):
            brothers = blobs2D_list[i].parent.children
            may_associated = blobs2D_list[i].potentialy_associated_3Dblobs
            for j in brothers:
                if (j.id != blobs2D_list[i].id) and (j.associated_3Dblob > 0):
                    dist2[i,j.associated_3Dblob-1] = 100.
                    if (j.associated_3Dblob in may_associated):
                        may_associated.remove(j.associated_3Dblob)
    
    gamma = (gamma_aux/nb_3Dblobs)
    ## H0 (no link)
    phi_all = phi(dist2)
    probaH0 = np.tile(phi_all.prod(1),(phi_all.shape[1],1)).T
    ## H1 (link)
    # "exponential" part
    dist2_exp = np.exp(-0.5 * (dist2 / sigma)**2)
    Zi = 2. / (sigma * np.sqrt(2. * np.pi))
    proba_exp = dist2_exp * Zi
    # "volume repartition" part
    phi_all[phi_all == 0.] = 1.
    proba_rep = probaH0 / phi_all
    # combine the two parts
    probaH1 = proba_exp * proba_rep
    ## "final" proba
    proba = np.zeros((nb_2Dblobs, nb_3Dblobs+1))
    proba[:,0:-1] = (probaH1 * gamma) / \
                    ((1.-nb_3Dblobs*gamma)*probaH0 + \
                     gamma*np.tile(probaH1.sum(1), (nb_3Dblobs,1)).T)
    proba[:,nb_3Dblobs] = (probaH0[:,0]*(1-nb_3Dblobs*gamma)) / \
                           ((1.-nb_3Dblobs*gamma)*probaH0[:,0] + \
                            gamma*probaH1.sum(1))

    return proba, gamma

def phi(dik):
    dik[dik > 150.] = 150.
    return (15./16.)*(1-((dik-75.)/75.)**2)**2

def plot_matching_results(nb_2Dblobs, nb_3Dblobs, proba, dist,
                          gamma, gamma_prime, sigma, file, blobs2D_list,
                          explode=False):
    global SUBJECT, CONTRAST, threshold_sure, threshold_maybe

    nb_2Dblobs = dist.shape[0]
    nb_3Dblobs = dist.shape[1] - 1
    res_file = open(file, 'w')
    sys.stdout = res_file
    print "### SUBJECT: %s" %SUBJECT
    print "# Contrast: %s" %CONTRAST
    print "# gamma: %g\n" %gamma
    print "nb_2D_blobs = %d" %nb_2Dblobs
    print "nb_3D_blobs = %d\n" %nb_3Dblobs

    new_blobs2D_list = copy.deepcopy(blobs2D_list)
    for blob2D in np.arange(nb_2Dblobs):
        if blobs2D_list[blob2D].parent:
            parent_id = blobs2D_list[blob2D].parent.id
        else:
            parent_id = -1
        for blob3D in np.arange(nb_3Dblobs):
            if proba[blob2D, blob3D] > threshold_sure:
                print "2D blob %d (%d) is thought to be related to 3D blob %d at %f%%" \
                      %(blobs2D_list[blob2D].id, parent_id,
                        blob3D+1, 100*proba[blob2D, blob3D])
                blobs2D_list[blob2D].associate_3Dblob(blob3D+1)
            elif proba[blob2D, blob3D] > threshold_maybe:
                blobs2D_list[blob2D].associate_potential(blob3D+1)
        if proba[blob2D, nb_3Dblobs] > threshold_sure:
            print "2D blob %d (%d) is not thought to be related to any 3D blob at %f%%" \
                  %(blobs2D_list[blob2D].id, parent_id,
                    100*proba[blob2D, nb_3Dblobs])
            blobs2D_list[blob2D].associate_3Dblob(0)
        elif proba[blob2D, nb_3Dblobs] > threshold_maybe:
            blobs2D_list[blob2D].associate_potential(0)
        significant_neighbors = np.where(proba[blob2D,:] > 5.e-2)[0]
        print "Blob 2D %d (%d):" %(blobs2D_list[blob2D].id, parent_id)
        significant_neighbors_id = significant_neighbors + 1
        significant_neighbors_id[significant_neighbors_id == nb_3Dblobs+1] = -1
        blobs2D_list[blob2D].set_association_probas(
            np.vstack((significant_neighbors_id,
                       100.*proba[blob2D,significant_neighbors],
                       dist[blob2D,significant_neighbors])).T)
        print blobs2D_list[blob2D].association_probas
        # post-treatment for ambigous blobs
        if (significant_neighbors.size > 2 and explode):
            print "--> exploding blob"
            new_blobs2D_list = explode_blob(blobs2D_list[blob2D],
                         significant_neighbors[significant_neighbors_id != -1],
                         new_blobs2D_list)
        else:
            for i in np.arange(new_blobs2D_list.__len__()):
                if new_blobs2D_list[i].id == blobs2D_list[blob2D].id:
                    del new_blobs2D_list[i]
                    break
        print ""
    
    res_file.close()
    sys.stdout = sys.__stdout__
    
    return new_blobs2D_list

def explode_blob(blob2D, significant_neighbors, blobs2D_list):
    neighbors_list = []
    for neighbor in significant_neighbors:
        neighbors_list.append(
            mroi.discrete_features['position'][neighbor])
    local_dist = compute_distances_aux(neighbors_list, blob2D.vertices)[0]
    associated_3Dblobs = np.argmin(local_dist, 1)
    for i in np.unique(associated_3Dblobs):
        # create a new 2D blob
        new_blob_vertices = blob2D.vertices[associated_3Dblobs == i]
        new_blob_vertices_id = blob2D.vertices_id[associated_3Dblobs == i]
        new_blob = Blob2D(new_blob_vertices, new_blob_vertices_id,
                          blob2D.hemisphere, blob2D)
        # add it to 2D blobs list
        blobs2D_list.append(new_blob)
        # update parent blob
        blob2D.add_children(new_blob)
        # remove parent blob from 2D blobs list
        for i in np.arange(blobs2D_list.__len__()):
            if blobs2D_list[i].id == blob2D.id:
                del blobs2D_list[i]
                break

    return blobs2D_list

def sort_list_by_link(my_list, blob3D_id):
    for i in np.arange(my_list.__len__()):
        links = my_list[i].blob.association_probas
        max_blob = links[links[:,0] == blob3D_id,:][0,1]
        max_index = i
        for j in np.arange(i+1, my_list.__len__()):
            links = my_list[j].blob.association_probas
            link = links[links[:,0] == blob3D_id,:][0,1]
            if (link > max_blob):
                max_blob = link
                max_index = j
        tmp_swap = my_list[i]
        my_list[i] = my_list[max_index]
        my_list[max_index] = tmp_swap

    return my_list

def mesh_to_graph(vertices, poly):
    """
    This function builds an fff graph from a mesh
    (Taken from nipy mesh_processing.py but removed the aims dependancy)
    """
    V = len(vertices)
    E = poly.size()
    edges = np.zeros((3*E,2))
    weights = np.zeros(3*E)

    for i in range(E):
        sa = poly[i][0]
        sb = poly[i][1]
        sc = poly[i][2]
        
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


# -----------------------------------------------------------
# --------- Parameters --------------------------------------
# -----------------------------------------------------------

# eventually choose 3D blobs to show (from 1 to ..., -3 to show everything)
blobs3D_to_show = [-3]
# eventually choose 2D blobs to show (from 0 to ..., -3 to show everything,
# -2 for original blobs)
blobs2D_to_show = [-3]

gamma_prime = 0.9
sigma = 5.

# thresholds for blobs matching
threshold_sure = 7.0e-1
threshold_maybe = 2.5e-1

# -----------------------------------------------------------
# --------- Script (part 1): IO and data handling -----------
# -----------------------------------------------------------

### Load left hemisphere data
lmesh = loadImage(lmesh_path_gii)
c, n, t  = lmesh.getArrays()
lvertices = c.getData()
ltriangles = t.getData()
blobs2D_ltex = tio.Texture(blobs2D_ltex_path).read(blobs2D_ltex_path)

### Load right hemisphere data
rmesh = loadImage(rmesh_path_gii)
c, n, t  = rmesh.getArrays()
rvertices = c.getData()
rtriangles = t.getData() 
blobs2D_rtex = tio.Texture(blobs2D_rtex_path).read(blobs2D_rtex_path)

### Construct 2D blobs hierarchy
# left hemisphere processing
G = mesh_to_graph(lvertices, ltriangles)
F = ff.Field(G.V, G.get_edges(), G.get_weights(), Fun)
affine = np.eye(4)
shape = ()
disc = np.reshape(np.arange(F.V), (F.V, 1)).astype(int)
lnroi = hroi.NROI_from_field(F, affine, shape, disc, refdim=0,
                            th=THETA, smin=SMIN)
# right hemisphere processing
G = mep.mesh_to_graph(rvertices, rtriangles)
F = ff.Field(G.V, G.get_edges(), G.get_weights(), Fun)
affine = np.eye(4)
shape = ()
disc = np.reshape(np.arange(F.V), (F.V, 1)).astype(int)
rnroi = hroi.NROI_from_field(F, affine, shape, disc, refdim=0,
                            th=THETA, smin=SMIN)

children = rnroi.get_children()
parents = rnroi.get_parents()
nroi_vertices_id = nroi.discrete_feature['index']
for i in np.arange(rnroi.k):
    if (not children[k]):
        if (parent[k} not in Hierarchical2DBlob.all_blobs.keys):
            new_hblob = Hierarchical2DBlob(nroi_vertices_id, k)

### Construct the list of 2D blobs
blobs2D_list = []
nb_right_blobs = int(np.amax(blobs2D_rtex.data)) + 1
for blob in np.arange(nb_right_blobs):
    blob_vertices_id = np.where(blobs2D_rtex.data == blob)[0]
    blob_vertices = rvertices[blob_vertices_id,:]
    new_blob2D = Blob2D(blob_vertices, blob_vertices_id, "right")
    blobs2D_list.append(new_blob2D)
nb_left_blobs = int(np.amax(blobs2D_ltex.data)) + 1
for blob in np.arange(nb_left_blobs):
    blob_vertices_id = np.where(blobs2D_ltex.data == blob)[0]
    blob_vertices = lvertices[blob_vertices_id,:]
    new_blob2D = Blob2D(blob_vertices, blob_vertices_id, "left")
    blobs2D_list.append(new_blob2D)

### Get the list of 3D blobs centers
blobs3D = load(blobs3D_path)
affine = blobs3D.get_affine()
shape = blobs3D.get_shape()
mroi = MultipleROI(affine=affine, shape=shape)
mroi.from_labelled_image(blobs3D_path)
mroi.compute_discrete_position()
blobs3D_centers = mroi.discrete_to_roi_features('position')
nb_3D_blobs = mroi.k

### Plot the 3D blobs
data = blobs3D.get_data()
labels = sp.ndimage.measurements.label(data)[0]
if blobs3D_to_show[0] == -3:
    blobs3D_to_show = np.arange(1,nb_3D_blobs+1)
data = np.asarray(data).copy()
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
# disable rendering for script acceleration purpose
blobs3D_mayavi_src.scene.disable_render = True
contour = mayavi.pipeline.contour(src2)
contour2 = mayavi.pipeline.set_active_attribute(contour, point_scalars='labels')
mayavi.pipeline.surface(contour2, colormap='spectral', color=(0., 0., 0.))

# -----------------------------------------------------------
# --------- Script (part 2): blobs matching -----------------
# -----------------------------------------------------------

### Compute distances between each pair of (3D blobs)-(2D blobs centers)
dist, dist_arg = compute_distances(mroi.discrete_features['position'],
                                   blobs2D_list)
dist_display = np.zeros((Blob2D.nb_blobs, nb_3D_blobs+1))
dist_display[:,0:-1] = dist.copy()
dist_display = np.sqrt(dist_display)
dist_display[:,nb_3D_blobs] = -1.

### Match each 2D blob with one or several 3D blob(s)
proba, gamma = compute_association_proba(blobs2D_list, nb_3D_blobs,
                                         gamma_prime, sigma,
                                         dist_display[:,0:-1])

### Post-processing the results
new_blobs2D_list = plot_matching_results(Blob2D.nb_blobs, nb_3D_blobs,
                                         proba, dist_display, gamma_prime,
                                         gamma, sigma, './results/ver0/res.txt',
                                         blobs2D_list, True)

### Compute distances between each pair of (3D blobs)-(new 2D blobs centers)
dist, dist_arg = compute_distances(mroi.discrete_features['position'],
                                   new_blobs2D_list)
dist_display = np.zeros((new_blobs2D_list.__len__(), nb_3D_blobs+1))
dist_display[:,0:-1] = dist.copy()
dist_display = np.sqrt(dist_display)
dist_display[:,nb_3D_blobs] = -1.

### Match each new 2D blobs with one or several 3D blob(s)
proba, gamma = compute_association_proba(new_blobs2D_list,
                                         nb_3D_blobs,
                                         gamma_prime, sigma,
                                         dist_display[:,0:-1])

### Post-processing the results
plot_matching_results(new_blobs2D_list.__len__(),
                                         nb_3D_blobs,
                                         proba, dist_display, gamma_prime,
                                         gamma, sigma,
                                         './results/ver0/new_res.txt',
                                         new_blobs2D_list, False)
#-------> TEST BEGIN
### Match each new 2D blobs with one or several 3D blob(s)
proba, gamma = compute_association_proba(new_blobs2D_list,
                                         nb_3D_blobs,
                                         gamma_prime, sigma,
                                         dist_display[:,0:-1],
                                         exclusion=True)
### Post-processing the results
plot_matching_results(new_blobs2D_list.__len__(), nb_3D_blobs, proba,
                      dist_display, gamma_prime, gamma, sigma,
                      './results/ver0/new_res.txt', new_blobs2D_list, False)
#-------> TEST END

### Mayavi Plot
# choose textures
ltex = blobs2D_ltex.data.copy()
rtex = blobs2D_rtex.data.copy()
if blobs2D_to_show[0] != -2.:
    if blobs2D_to_show[0] == -3.:
        blobs2D_to_show = np.arange(0,Blob2D.nb_blobs)
    ltex[:] = -1.
    rtex[:] = -1.
    for i in blobs2D_to_show:
        blob = Blob2D.all_blobs[i]
        if (blob.associated_3Dblob != -1):
            value = blob.associated_3Dblob
        else:
            value = -0.7
        if blob.hemisphere == "left":
            ltex[blob.vertices_id] = value
        else:
            rtex[blob.vertices_id] = value


# plot left hemisphere
mayavi.triangular_mesh(lvertices[:,0], lvertices[:,1], lvertices[:,2],
                       triangles, scalars=ltex,
                       transparent=False, opacity=1.)

# plot right hemisphere
mayavi.triangular_mesh(rvertices[:,0], rvertices[:,1], rvertices[:,2],
                       triangles, scalars=rtex,
                       transparent=False, opacity=1.)

# enable rendering (because we have disabled it)
blobs3D_mayavi_src.scene.disable_render = False 
plt.show()


### Plot the results in a nice way
nb_linked = 0
for blob2D in Blob2D.all_blobs.values():
    Blob2DDisplay(blob2D)
    if (not blob2D.children):
        nb_linked += 1

# construct a list of associated 2D blobs for each 3D blobs
# and sort it by their link probability value
nested_association_lists = []
for i in np.arange(1,nb_3D_blobs+1):
    new_blob3D = Blob3DDisplay(i)
    association_list = []
    for blob2D in Blob2D.all_blobs.values():
        if (not blob2D.children):
            if (blob2D.associated_3Dblob == i):
                association_list.insert(0,Blob2DDisplay.all_blobs[blob2D.id])
            elif (i in blob2D.potentialy_associated_3Dblobs):
                association_list.append(Blob2DDisplay.all_blobs[blob2D.id])
    association_list = sort_list_by_link(association_list, i)
    if association_list:
        association_list.insert(0,i)
        nested_association_lists.append(association_list)

# set matplotlib figure basis
fig = plt.figure(1)
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.text((Blob3DDisplay.x_pos-Blob2DDisplay.default_x_pos)/2, 15,
        "Subject %s, Contrast %s, gamma=%g" %(SUBJECT, CONTRAST, gamma),
        horizontalalignment='center')
ax.set_xlim(Blob2DDisplay.default_x_pos-15, Blob3DDisplay.x_pos+4)
ax.set_ylim(-Blob2DDisplay.spacing*np.amax([nb_linked,
                                            nb_3D_blobs-1]),25)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
blobs3D_lines_colors = np.tile(['b','g','c','m','k','y'], 10)

# display the associated blobs
id_display = 0
for l in nested_association_lists:
    i = l[0]
    for b in l[1:]:
        if (not b.already_displayed):
            b.set_ypos(-(id_display * Blob2DDisplay.spacing) - 1)
            b.display(ax)
            id_display += 1
        coeff = b.blob.association_probas[b.blob.association_probas[:,0]==i,:][0,1]
        Blob3DDisplay.all_blobs[i].add_link(b.y_pos, coeff)
    mean_position =  Blob3DDisplay.all_blobs[i].get_pos()
    p = mpatches.Circle((Blob3DDisplay.x_pos, mean_position),
                        Blob3DDisplay.radius)
    ax.text(Blob3DDisplay.x_pos+5, mean_position-1, '%d'%(i),
            horizontalalignment='center')
    ax.add_patch(p)
    for b in l[1:]:
        probas = b.blob.association_probas
        link = probas[probas[:,0]==i,:][0,1]
        if link > 100*threshold_sure:
            ax.plot([b.x_pos,Blob3DDisplay.x_pos],
                    [b.y_pos,mean_position],
                    color=blobs3D_lines_colors[i])
        elif link > 100*threshold_maybe:
            ax.plot([b.x_pos,Blob3DDisplay.x_pos],
                    [b.y_pos,mean_position], '--',
                    color=blobs3D_lines_colors[i])

# display 2D blobs that have no children and no association
for blob2D in Blob2DDisplay.all_blobs.values():
    blob = blob2D.blob
    if (not blob.children):
        if (blob.associated_3Dblob == 0):
            if (not blob2D.already_displayed):
                blob2D.set_ypos(-(id_display * Blob2DDisplay.spacing) - 1)
                blob2D.display(ax, circle_color='red')
                id_display += 1
        elif (0 in blob.potentialy_associated_3Dblobs):
            if (not blob2D.already_displayed):
                blob2D.set_ypos(-(id_display * Blob2DDisplay.spacing) - 1)
                blob2D.display(ax, circle_color='red')
                id_display += 1

# display 2D blobs that have been exploded
for blob2D in Blob2DDisplay.all_blobs.values():
    if ((not blob2D.already_displayed) and (blob2D.blob.children)):
        values = 0.
        coeffs = 0.
        text_color = 'gray'
        for child in blob2D.blob.children:
            values += Blob2DDisplay.all_blobs[child.id].y_pos
            coeffs += 1.
            if child.associated_3Dblob > 0:
                text_color = 'black'
        blob2D.set_ypos(values/coeffs)
        blob2D.display(ax, text_color)
        for child in blob2D.blob.children:
            p = mpatches.Circle((Blob2DDisplay.all_blobs[child.id].x_pos-10,
                                 Blob2DDisplay.all_blobs[child.id].y_pos),
                                Blob2DDisplay.radius)
            ax.add_patch(p)
            ax.plot([Blob2DDisplay.all_blobs[child.id].x_pos-10,blob2D.x_pos],
                    [Blob2DDisplay.all_blobs[child.id].y_pos,blob2D.y_pos],
                    color='black')

### Finally write output (right and left) textures
output_rtex = tio.Texture(rresults_output, data=rtex)
output_rtex.write()
output_ltex = tio.Texture(lresults_output, data=ltex)
output_ltex.write()

# show graphics
plt.show()
