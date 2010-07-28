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
import nipy.neurospin.graph.graph as fg
from nipy.neurospin.spatial_models import hroi
import enthought.mayavi.mlab as mayavi
from gifti import loadImage
from scikits.learn import BallTree

from database_archi import *


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
# --------- Classes definition ------------------------------
# -----------------------------------------------------------

class Blob:
    '''
    Represents a 2D or 3D blob with:
        - vertices, float[3][]: list of its vertices
    
    '''

    def __init__(self, vertices_list, vertices_id_list, parent=None):
        '''
        Construct a new blob
        
        '''
        self.vertices = vertices_list
        self.vertices_id = vertices_id_list
        self.children = []
        
        # set parent 3D blob
        self.parent = parent
        
        # attributes for plotting
        self.y_pos_values = 0.
        self.y_pos_coeffs = 0.

    def add_child(self, child_to_add):
        '''
        '''
        if (child_to_add not in self.children):
            self.children.append(child_to_add)
        self.change_to_node()

    def remove_child(self, child_to_remove):
        '''
        '''
        self.children.remove(child_to_remove)
        if (not self.children):
            self.change_to_leaf()

    def set_parent(self, parent):
        '''
        '''
        self.parent = parent
    

class Blob3D(Blob):
    '''
    Represents a 3D blob with:
      (inherited from Blob)
        - vertices, float[3][]: list of its vertices
      (new to Blob3D)
        - parent, Blob3D: parent blob, "None" if no parent
        - children, Blob3D[]: children list
      
    '''
    # total number of 3D blobs
    nb_blobs = -1
    # store all the 3D blobs instantiated
    all_blobs = {}
    # leaves
    leaves = {}
    # nodes
    nodes = {}

    # Plotting parameters
    spacing = 8.
    default_xpos = 35.
    radius = 1.
    
    def __init__(self, vertices_list, vertices_id_list, parent=None):
        '''
        Construct a new 3D blob
        '''
        # create a new blob
        Blob.__init__(self, vertices_list, vertices_id_list, parent)

        # set blob id and increase total number of 3D blobs
        self.id = Blob3D.nb_blobs + 1
        Blob3D.nb_blobs += 1
        Blob3D.all_blobs[self.id] = self
        
        # compute and set blob center
        self.center = self.compute_center()

        # tag to avoid displaying the same blob several times
        self.already_displayed = False

    def compute_center(self):
        '''
        Compute blob center
        '''
        if not isinstance(self.vertices, None.__class__):
            res = np.mean(self.vertices, 0)
        else:
            res = np.nan
        
        return res

    def is_leaf(self):
        if self.id in Blob3D.leaves.keys():
            res = True
        else:
            res = False

        return res

    def is_node(self):
        if self.id in Blob3D.nodes.keys():
            res = True
        else:
            res = False

        return res

    def change_to_leaf(self):
        '''
        '''
        if self.is_node():
            del  Blob3D.nodes[self.id]
        if not self.is_leaf():
            Blob3D.leaves[self.id] = self

    def change_to_node(self):
        '''
        '''
        if self.is_leaf():
            del  Blob3D.leaves[self.id]
        if not self.is_node():
            Blob3D.nodes[self.id] = self

    ### Methods for plotting
    def get_xpos(self):
        if self.already_displayed:
            res = self.xpos
        else:
            if self.is_node():
                max_children_pos = Blob3D.default_xpos
                for child in self.children:
                    if child.get_xpos() > max_children_pos:
                        max_children_pos = child.get_xpos()
                res = max_children_pos + 30.
            else:
                res = Blob3D.default_xpos
        
        return res

    def get_ypos(self):
        if self.already_displayed:
            res = self.ypos
        else:
            if self.is_leaf():
                yvalues = 0.
                ycoeffs = 0.
                for linked_blob in Blob2D.leaves.values():
                    if (linked_blob.associated_3D_blob == self \
                        or self in linked_blob.potentialy_associated):
                        link =  linked_blob.association_probas
                        coeff = link[link[:,0]==self.id,1]
                        ycoeffs += coeff
                        yvalues +=  linked_blob.get_ypos() * coeff
                if ycoeffs != 0.:
                    res = yvalues/ycoeffs
                else:
                    res = -(self.id * Blob3D.spacing) - 1
            elif self.is_node():
                yvalues = 0.
                ycoeffs = 0.
                for child in self.children:
                    ycoeffs += 1
                    yvalues += child.get_ypos()
                res = yvalues/ycoeffs
            else:
                print "Warning"
                res = -(self.id * Blob3D.spacing) - 1

        return res

    def display(self, ax, circle_color='blue'):
        if (not self.already_displayed):
            self.xpos = self.get_xpos()
            self.ypos = self.get_ypos()
            p = mpatches.Circle((self.xpos,self.ypos),
                                Blob3D.radius, color=circle_color)
            ax.text(self.xpos+5, self.ypos-1, '%d'%self.id,
                    horizontalalignment='center', color='black')
            ax.add_patch(p)
        self.already_displayed = True


class Blob2D(Blob):
    '''
    Represents a 2D blob with:
      (inherited from Blob)
        - vertices, float[3][]: list of its vertices
      (new to Blob2D)
        - vertices_id, int[]: list of its vertices id
        - parent, Blob2D : parent blob, "None" if no parent
        - children, Blob2D[]: children list
        - associated_3D_blob, Blob3D: associated 3D blob
        - potentialy_associated, Blob3D[]: potentialy associated 3D blobs
        - association_probas, float[][]: association probability values
        - hemisphere, string: hemishpere in which to find the blob
      
    '''
    # total number of 2D blobs
    nb_blobs = 0
    # store all the 2D blobs instantiated
    all_blobs = {}
    # leaves
    leaves = {}
    # nodes
    nodes = {}
    
    # Plotting parameters
    display_id = 0.
    default_xpos = 0.
    radius = 1.
    spacing = 8.
    
    def __init__(self, vertices_list, vertices_id_list,
                 hemisphere, parent=None, sub_blob=False):
        '''
        Construct a new 2D blob
        '''
        # create a new blob
        Blob.__init__(self, vertices_list, vertices_id_list, parent)
        self.hemisphere = hemisphere
        
        # set blob id and increase total number of 2D blobs
        self.id = Blob2D.nb_blobs + 1
        Blob2D.nb_blobs += 1
        Blob2D.all_blobs[self.id] = self
        
        # compute and set blob center
        self.center = self.compute_center()
        
        # set default associated 3D blob (-1: associated to nothing,
        # O is "we don't know about the association")
        self.associated_3D_blob = None
        self.potentialy_associated = []
        
        # tag to avoid displaying the same blob several times
        self.already_displayed = False
        
        #
        self.is_sub_blob = sub_blob
    
    def compute_center(self):
        '''
        Compute blob center
        '''
        return np.mean(self.vertices, 0)
    
    def change_to_leaf(self):
        '''
        '''
        if self.is_node():
            del  Blob2D.nodes[self.id]
        if not self.is_leaf():
            Blob2D.leaves[self.id] = self
    
    def change_to_node(self):
        '''
        '''
        if self.is_leaf():
            del  Blob2D.leaves[self.id]
        if not self.is_node():
            Blob2D.nodes[self.id] = self
    
    def associate_3Dblob(self, blob3D):
        '''
        '''
        self.associated_3D_blob = blob3D
        
    def associate_potential(self, blob3D):
        '''
        '''
        if (blob3D not in self.potentialy_associated):
            self.potentialy_associated.append(blob3D)
            
    def set_association_probas(self, probas):
        '''
        '''
        self.association_probas = probas

    def is_leaf(self):
        if self.id in Blob2D.leaves.keys():
            res = True
        else:
            res = False
            
        return res
    
    def is_node(self):
        if self.id in Blob2D.nodes.keys():
            res = True
        else:
            res = False
            
        return res
    
    ### Methods for plotting
    def get_xpos(self):
        if self.already_displayed:
            res = self.xpos
        else:
            if self.children:
                min_children_pos = Blob2D.default_xpos
                for child in self.children:
                    if child.get_xpos() < min_children_pos:
                        min_children_pos = child.get_xpos()
                res = min_children_pos - 30.
            else:
                res = Blob2D.default_xpos
        
        return res
    
    def get_ypos(self):
        if self.already_displayed:
            res = self.ypos
        else:
            if self.is_node():
                yvalues = 0.
                ycoeffs = 0.
                for child in self.children:
                    ycoeffs += 1.
                    yvalues += child.get_ypos()
                res = yvalues/ycoeffs
            else:
                res = self.ypos
            
        return res
    
    def display(self, ax, text_color='black', circle_color='blue'):
        if ((not self.already_displayed) and self.is_leaf()):
            self.xpos = self.get_xpos()
            self.ypos = -(Blob2D.display_id * Blob2D.spacing) - 1
            Blob2D.display_id += 1.
            p = mpatches.Circle((self.xpos,self.ypos),
                                Blob2D.radius, color=circle_color)
            if self.is_sub_blob:
                display_color = 'red'
            else:
                display_color = text_color
            ax.text(self.xpos-5, self.ypos-2, '%d'%self.id,
                    horizontalalignment='center', color=display_color)
            ax.add_patch(p)
        elif ((not self.already_displayed) and self.is_node()):
            self.xpos = self.get_xpos()
            self.ypos = self.get_ypos()
            Blob2D.display_id += 1.
            p = mpatches.Circle((self.xpos,self.ypos),
                                Blob2D.radius, color=circle_color)
            ax.text(self.xpos-5, self.ypos-2, '%d'%self.id,
                    horizontalalignment='center')
            ax.add_patch(p)
        self.already_displayed = True


# -----------------------------------------------------------
# --------- Routines definition -----------------------------
# -----------------------------------------------------------

def compute_distances(blobs3D_vertices, blobs2D_list):
    blobs2D_centers = np.zeros((blobs2D_list.__len__(), 3))
    i = 0
    for blob in blobs2D_list:
        blobs2D_centers[i,:] = blob.center
        i += 1
    return compute_distances_aux(blobs3D_vertices, blobs2D_centers)

def compute_distances_aux(blobs3D_vertices, blobs2D_centers):
    nb_2Dblobs = blobs2D_centers.__len__()
    nb_3Dblobs = blobs3D_vertices.__len__()
    dist = np.zeros((nb_2Dblobs, nb_3Dblobs))
    dist_arg = np.zeros((nb_2Dblobs, nb_3Dblobs), dtype='int')
    for k in np.arange(nb_3Dblobs):
        vertices = blobs3D_vertices[k]
        blob3D_vertices_aux = np.tile(vertices.reshape(-1,1,3),
                                  (1,nb_2Dblobs,1))
        blobs2D_centers_aux = np.tile(blobs2D_centers,
                                  (vertices.shape[0],1,1))
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
            may_associated = blobs2D_list[i].potentialy_associated
            for j in brothers:
                if ((not isinstance(j.associated_3D_blob, None.__class__)) \
                    and (j.associated_3D_blob.id > 0)
                    and (j.id != blobs2D_list[i].id)):
                    # get index of j.associated_3D_blob in blobs3D_list
                    blob_index = blobs3D_list.index(j.associated_3D_blob)
                    # remove potential association
                    dist2[i,blob_index] = 100.
                    if (j.associated_3D_blob in may_associated):
                        may_associated.remove(j.associated_3D_blob)
    
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

def plot_matching_results(proba, dist, gamma, gamma_prime, sigma,
                          file, blobs2D_list, blobs3D_list, explode=False):
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
                        blobs3D_list[blob3D].id, 100*proba[blob2D, blob3D])
                blobs2D_list[blob2D].associate_3Dblob(blobs3D_list[blob3D])
            elif proba[blob2D, blob3D] > threshold_maybe:
                blobs2D_list[blob2D].associate_potential(blobs3D_list[blob3D])
        if proba[blob2D, nb_3Dblobs] > threshold_sure:
            print "2D blob %d (%d) is not thought to be related to any 3D blob at %f%%" \
                  %(blobs2D_list[blob2D].id, parent_id,
                    100*proba[blob2D, nb_3Dblobs])
            blobs2D_list[blob2D].associate_3Dblob(Blob3D.all_blobs[0])
        elif proba[blob2D, nb_3Dblobs] > threshold_maybe:
            blobs2D_list[blob2D].associate_potential(Blob3D.all_blobs[0])
        significant_neighbors = np.where(proba[blob2D,:] > 5.e-2)[0]
        print "Blob 2D %d (%d):" %(blobs2D_list[blob2D].id, parent_id)
        significant_neighbors_id = significant_neighbors.copy()
        for i in np.arange(significant_neighbors_id.__len__()):
            if significant_neighbors[i] != nb_3Dblobs:
                significant_neighbors_id[i] = blobs3D_list[significant_neighbors[i]].id
            else:
                significant_neighbors_id[i] = -1
        blobs2D_list[blob2D].set_association_probas(
            np.vstack((significant_neighbors_id,
                       100.*proba[blob2D,significant_neighbors],
                       dist[blob2D,significant_neighbors])).T)
        print blobs2D_list[blob2D].association_probas
        # post-treatment for ambigous blobs
        if (significant_neighbors.size > 2 and explode):
            print "--> exploding blob"
            new_blobs2D_list = explode_blob(blobs2D_list[blob2D],
                    significant_neighbors_id[significant_neighbors_id != -1],
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

def explode_blob(blob2D, significant_neighbors_id, blobs2D_list):
    neighbors_list = []
    for neighbor in significant_neighbors_id:
        neighbors_list.append(Blob3D.leaves[neighbor].vertices)
    local_dist = compute_distances_aux(neighbors_list, blob2D.vertices)[0]
    associated_3D_blobs = np.argmin(local_dist, 1)
    for i in np.unique(associated_3D_blobs):
        # create a new 2D blob
        new_blob_vertices = blob2D.vertices[associated_3D_blobs == i]
        new_blob_vertices_id = blob2D.vertices_id[associated_3D_blobs == i]
        new_blob = Blob2D(new_blob_vertices, new_blob_vertices_id,
                          blob2D.hemisphere, blob2D, True)
        new_blob.change_to_leaf()
        # add it to 2D blobs list
        blobs2D_list.append(new_blob)
        # update parent blob
        blob2D.add_child(new_blob)
        # remove parent blob from 2D blobs list
        for i in np.arange(blobs2D_list.__len__()):
            if blobs2D_list[i].id == blob2D.id:
                del blobs2D_list[i]
                break

    return blobs2D_list


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


# -----------------------------------------------------------
# --------- Script (part 1): IO and data handling -----------
# -----------------------------------------------------------

### Load left hemisphere data
lmesh = loadImage(lmesh_path_gii)
c, n, t  = lmesh.getArrays()
lvertices = c.getData()
ltriangles = t.getData()
glm_ltex = tio.Texture(glm_ltex_path).read(glm_ltex_path)
blobs2D_ltex = tio.Texture(blobs2D_ltex_path).read(blobs2D_ltex_path)

### Load right hemisphere data
rmesh = loadImage(rmesh_path_gii)
c, n, t  = rmesh.getArrays()
rvertices = c.getData()
rtriangles = t.getData()
glm_rtex = tio.Texture(glm_rtex_path).read(glm_rtex_path)
blobs2D_rtex = tio.Texture(blobs2D_rtex_path).read(blobs2D_rtex_path)

### Construct 2D blobs hierarchy
# right hemisphere processing
G = mesh_to_graph(rvertices, rtriangles)
F = ff.Field(G.V, G.get_edges(), G.get_weights(), glm_rtex.data)
affine = np.eye(4)
shape = ()
disc = np.reshape(np.arange(F.V), (F.V, 1)).astype(int)
rnroi = hroi.NROI_from_field(F, affine, shape, disc, refdim=0,
                            th=THETA, smin=SMIN)
# create the right number of blobs
for i in np.arange(rnroi.k):
    vertices_id = np.ravel(rnroi.discrete_features['index'][i])
    Blob2D(rvertices[vertices_id], vertices_id, "right")
# update associations between blobs
children = rnroi.get_children()
parents = rnroi.get_parents()
for i in np.arange(rnroi.k):
    current_blob = Blob2D.all_blobs[i+1]
    # set parent
    if (parents[i] != i):
        current_blob.parent = Blob2D.all_blobs[parents[i]+1]
    # set children
    if (children[i].size == 0):
        current_blob.change_to_leaf()
    else:
        current_blob.change_to_node()
        for child in children[i]:
            current_blob.add_child(Blob2D.all_blobs[child+1])

# left hemisphere processing
G = mesh_to_graph(lvertices, ltriangles)
F = ff.Field(G.V, G.get_edges(), G.get_weights(), glm_ltex.data)
affine = np.eye(4)
shape = ()
disc = np.reshape(np.arange(F.V), (F.V, 1)).astype(int)
lnroi = hroi.NROI_from_field(F, affine, shape, disc, refdim=0,
                            th=THETA, smin=SMIN)

# create the right number of blobs
for i in np.arange(lnroi.k):
    vertices_id = np.ravel(lnroi.discrete_features['index'][i])
    Blob2D(lvertices[vertices_id], vertices_id, "left")
# update associations between blobs
children = lnroi.get_children()
parents = lnroi.get_parents()
for i in np.arange(lnroi.k):
    current_blob = Blob2D.all_blobs[i+1+rnroi.k]
    # set parent
    if (parents[i] != i):
        current_blob.parent = Blob2D.all_blobs[parents[i]+1+rnroi.k]
    # set children
    if (children[i].size == 0):
        current_blob.change_to_leaf()
    else:
        current_blob.change_to_node()
        for child in children[i]:
            current_blob.add_child(Blob2D.all_blobs[child+1+rnroi.k])

# finally get the 2D blobs that are leaves
blobs2D_list = Blob2D.leaves.values()

### Get 3D blobs hierarchy
# get data from activation map
nim = load(glm_data_path)
affine = nim.get_affine()
shape = nim.get_shape()
data = nim.get_data()
values = data[data!=0]
xyz = np.array(np.where(data)).T
F = ff.Field(xyz.shape[0])
F.from_3d_grid(xyz)
F.set_field(values)

# create the right number of blobs
Blob3D(None, None)
nroi3D = hroi.NROI_from_field(F, affine, shape, xyz, 0, THETA3D, SMIN3D)
nroi3D.compute_discrete_position()
for i in np.arange(nroi3D.k):
    vertices = nroi3D.discrete_features['position'][i]
    vertices_id = np.ravel(nroi3D.discrete_features['index'][i])
    Blob3D(vertices, vertices_id)
# update associations between blobs
children = nroi3D.get_children()
parents = nroi3D.get_parents()
for i in np.arange(nroi3D.k):
    current_blob = Blob3D.all_blobs[i+1]
    # set parent
    if (parents[i] != i):
        current_blob.parent = Blob3D.all_blobs[parents[i]+1]
    # set children
    if (children[i].size == 0):
        current_blob.change_to_leaf()
    else:
        current_blob.change_to_node()
        for child in children[i]:
            current_blob.add_child(Blob3D.all_blobs[child+1])

# finally get the 2D blobs that are leaves
blobs3D_vertices = []
blobs3D_list = []
for b in Blob3D.leaves.values():
    blobs3D_vertices.append(b.vertices)
    blobs3D_list.append(b)

### Get the list of 3D blobs centers
blobs3D = load(blobs3D_path)
"""
affine = blobs3D.get_affine()
shape = blobs3D.get_shape()
mroi = MultipleROI(affine=affine, shape=shape)
mroi.from_labelled_image(blobs3D_path)
mroi.compute_discrete_position()
for i in np.arange(mroi.k):
    mroi.discrete_feature

blobs3D_centers = mroi.discrete_to_roi_features('position')
nb_3D_blobs = mroi.k
"""
"""
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
"""

# -----------------------------------------------------------
# --------- Script (part 2): blobs matching -----------------
# -----------------------------------------------------------
nb_3D_blobs = blobs3D_vertices.__len__()
### Compute distances between each pair of (3D blobs)-(2D blobs centers)
dist, dist_arg = compute_distances(blobs3D_vertices, blobs2D_list)
dist_display = np.zeros((blobs2D_list.__len__(), nb_3D_blobs+1))
dist_display[:,0:-1] = dist.copy()
dist_display = np.sqrt(dist_display)
dist_display[:,nb_3D_blobs] = -1.

### Match each 2D blob with one or several 3D blob(s)
proba, gamma = compute_association_proba(blobs2D_list, nb_3D_blobs,
                                         gamma_prime, sigma,
                                         dist_display[:,0:-1])

### Post-processing the results
new_blobs2D_list = plot_matching_results(proba, dist_display, gamma_prime,
                                         gamma, sigma, './results/ver0/res.txt',
                                         blobs2D_list, blobs3D_list,
                                         explode=True)

### Compute distances between each pair of (3D blobs)-(new 2D blobs centers)
dist, dist_arg = compute_distances(blobs3D_vertices, new_blobs2D_list)
dist_display = np.zeros((new_blobs2D_list.__len__(), nb_3D_blobs+1))
dist_display[:,0:-1] = dist.copy()
dist_display = np.sqrt(dist_display)
dist_display[:,nb_3D_blobs] = -1.

### Match each new 2D blobs with one or several 3D blob(s)
proba, gamma = compute_association_proba(new_blobs2D_list, nb_3D_blobs,
                                         gamma_prime, sigma,
                                         dist_display[:,0:-1])
### Post-processing the results
plot_matching_results(proba, dist_display, gamma_prime, gamma, sigma,
                      './results/ver0/new_res.txt', new_blobs2D_list,
                      blobs3D_list, explode=False)

### Match each new 2D blobs with one or several 3D blob(s)
proba, gamma = compute_association_proba(new_blobs2D_list, nb_3D_blobs,
                                         gamma_prime, sigma,
                                         dist_display[:,0:-1],
                                         exclusion=True)
### Post-processing the results
plot_matching_results(proba, dist_display, gamma_prime, gamma, sigma,
                      './results/ver0/newnew_res.txt', new_blobs2D_list,
                      blobs3D_list, explode=False)

"""
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
"""

def sort_list_by_link(my_list, blob3D_id):
    for i in np.arange(my_list.__len__()):
        links = my_list[i].association_probas
        max_blob = links[links[:,0] == blob3D_id,:][0,1]
        max_index = i
        for j in np.arange(i+1, my_list.__len__()):
            links = my_list[j].association_probas
            link = links[links[:,0] == blob3D_id,:][0,1]
            if (link > max_blob):
                max_blob = link
                max_index = j
        tmp_swap = my_list[i]
        my_list[i] = my_list[max_index]
        my_list[max_index] = tmp_swap

    return my_list

# construct a list of associated 2D blobs for each 3D blobs
# and sort it by their link probability value
nested_association_lists = []
for i in np.arange(nb_3D_blobs):
    association_list = []
    for blob2D in Blob2D.leaves.values():
        if (blobs3D_list[i] in blob2D.potentialy_associated):
            association_list.append(blob2D)
        elif (not isinstance(blob2D.associated_3D_blob, None.__class__) and \
              blob2D.associated_3D_blob.id == blobs3D_list[i].id):
            association_list.insert(0,blob2D)
    association_list = sort_list_by_link(association_list, blobs3D_list[i].id)
    if association_list:
        association_list.insert(0,blobs3D_list[i].id)
        nested_association_lists.append(association_list)

# set matplotlib figure basis
fig = plt.figure(1)
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.text((Blob3D.default_xpos-Blob2D.default_xpos)/2, 15,
        "Subject %s, Contrast %s, gamma=%g" %(SUBJECT, CONTRAST, gamma),
        horizontalalignment='center')
ax.set_xlim(Blob2D.default_xpos-15, Blob3D.default_xpos+15)
#ax.set_ylim(-Blob2DDisplay.spacing*np.amax([nb_linked,
#                                            nb_3D_blobs-1]),25)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
blobs3D_lines_colors = np.tile(['b','g','c','m','k','y'], 10)

# display the associated blobs
for l in nested_association_lists:
    blob3D_id = l[0]
    for b in l[1:]:
        b.display(ax)
    Blob3D.leaves[blob3D_id].display(ax)
    for b in l[1:]:
        probas = b.association_probas
        link = probas[probas[:,0]==blob3D_id,:][0,1]
        if link > 100*threshold_sure:
            ax.plot([b.get_xpos(),Blob3D.leaves[blob3D_id].get_xpos()],
                    [b.get_ypos(),Blob3D.leaves[blob3D_id].get_ypos()],
                    color=blobs3D_lines_colors[blob3D_id])
        elif link > 100*threshold_maybe:
            ax.plot([b.get_xpos(),Blob3D.leaves[blob3D_id].get_xpos()],
                    [b.get_ypos(),Blob3D.leaves[blob3D_id].get_ypos()],
                    '--', color=blobs3D_lines_colors[blob3D_id])

# display 2D blobs that have no children and no association
for blob2D in Blob2D.leaves.values():
    if (blob2D.associated_3D_blob == Blob3D.all_blobs[0]):
        blob2D.display(ax, circle_color='red')
    else:
        blob2D.display(ax, circle_color='green')

# display 3D blobs hierarchy
for blob3D in Blob3D.nodes.values():
    blob3D.display(ax)
    for child in blob3D.children:
        ax.plot([child.get_xpos(),blob3D.get_xpos()],
                [child.get_ypos(),blob3D.get_ypos()],
                color='black')

# display 2D blobs hierarchy
for blob2D in Blob2D.nodes.values():
    blob2D.display(ax)
    for child in blob2D.children:
        ax.plot([child.get_xpos(),blob2D.get_xpos()],
                [child.get_ypos(),blob2D.get_ypos()],
                color='black')
