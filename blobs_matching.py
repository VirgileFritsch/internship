"""


Author: Virgile Fritsch, 2010

"""

import sys, copy, os
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
import nipy.neurospin.clustering.clustering as cl
from nipy.neurospin.spatial_models import hroi
import enthought.mayavi.mlab as mayavi
from gifti import loadImage
from scikits.learn import ball_tree

from database_archi import *


# -----------------------------------------------------------
# --------- Parameters --------------------------------------
# -----------------------------------------------------------

# eventually choose 3D blobs to show (from 1 to ..., -3 to show everything)
blobs3D_to_show = [-3]
# eventually choose 2D blobs to show (from 0 to ..., -3 to show everything,
# -2 for original blobs)
blobs2D_to_show = [-3]
blobs2D_to_show_bckup = np.array(blobs2D_to_show).copy()
# choose kind of texture to plot with mayavi
mayavi_outtex_type = "coord"
# choose level of texture to plot with mayavi
mayavi_outtex_level = 1

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

    def __init__(self, vertices_list, vertices_id_list,
                 activation, parent=None):
        '''
        Construct a new blob
        
        '''
        self.vertices = vertices_list
        self.vertices_id = vertices_id_list
        self.activation = activation
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

    def get_max_activation(self):
        '''
        '''
        return np.amax(self.activation)

    def get_argmax_activation(self):
        '''
        '''
        return np.argmax(self.activation)

    def get_max_activation_location(self):
        '''
        '''
        return self.vertices[np.argmax(self.activation)]
    

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
    radius = 8.
    
    def __init__(self, vertices_list, vertices_id_list,
                 activation, parent=None):
        '''
        Construct a new 3D blob
        '''
        # create a new blob
        Blob.__init__(self, vertices_list, vertices_id_list,
                      activation, parent)

        # set blob id and increase total number of 3D blobs
        self.id = Blob3D.nb_blobs + 1
        Blob3D.nb_blobs += 1
        Blob3D.all_blobs[self.id] = self

        # tag to avoid displaying the same blob several times
        self.already_displayed = False

    def compute_center(self):
        '''
        Compute blob center
        '''
        if not isinstance(self.vertices, None.__class__):
            tmp_vertices = self.vertices.copy()
            for child in self.children:
                tmp_vertices = np.vstack((tmp_vertices, child.vertices.copy()))
            res = np.mean(tmp_vertices, 0)
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

    def merge_child(self, child):
        if (child in self.children):
            self.remove_child(child)
            self.vertices = np.vstack((self.vertices, child.vertices))
            self.vertices_id = np.hstack((self.vertices_id, child.vertices_id))
            self.activation = np.concatenate((self.activation,
                                              child.activation))
            del Blob3D.leaves[child.id]
        else:
            print "Warning, not a child of this 3D blob."

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

    def display(self, ax, circle_color='#3399ff', text_color='black'):
        if (not self.already_displayed):
            self.xpos = self.get_xpos()
            self.ypos = self.get_ypos()
            p = mpatches.Circle((self.xpos,self.ypos),
                                Blob3D.radius, color=circle_color)
            ax.text(self.xpos, self.ypos, '%d'%self.id,
                    horizontalalignment='center', verticalalignment='center',
                    color=text_color)
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
    radius = 8.
    spacing = 30.
    
    def __init__(self, vertices_list, vertices_id_list, activation,
                 hemisphere, parent=None, sub_blob=False, meta_blob=False):
        '''
        Construct a new 2D blob
        '''
        # create a new blob
        Blob.__init__(self, vertices_list, vertices_id_list,
                      activation, parent)
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
        self.association_probas = None
        
        # tag to avoid displaying the same blob several times
        self.already_displayed = False
        
        #
        self.is_sub_blob = sub_blob
        self.is_meta_blob = meta_blob
    
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

    def add_vertices(self, new_vertices):
        self.vertices = np.vstack((self.vertices, new_vertices))

    def add_vertices_id(self, new_vertices_id):
        self.vertices_id = np.hstack((self.vertices_id, new_vertices_id))

    def merge_child(self, child):
        if (child in self.children):
            self.remove_child(child)
            self.vertices = np.vstack((self.vertices, child.vertices))
            self.vertices_id = np.hstack((self.vertices_id, child.vertices_id))
            self.activation = np.concatenate((self.activation,
                                              child.activation))
            del Blob2D.leaves[child.id]
        else:
            print "Warning, not a child of this 2D blob."
    
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
    
    def display(self, ax, text_color='black', circle_color='#3399ff'):
        if ((not self.already_displayed) and self.is_leaf()):
            self.xpos = self.get_xpos()
            self.ypos = -(Blob2D.display_id * Blob2D.spacing) - 1
            Blob2D.display_id += 1.
            p = mpatches.Circle((self.xpos,self.ypos),
                                Blob2D.radius, color=circle_color)
            if self.is_sub_blob:
                display_color = 'red'
            elif self.is_meta_blob:
                display_color = 'white'
            else:
                display_color = text_color
            ax.text(self.xpos, self.ypos, '%d'%self.id,
                    horizontalalignment='center', verticalalignment='center',
                    color=display_color, fontsize=11)
            ax.add_patch(p)
        elif ((not self.already_displayed) and self.is_node()):
            self.xpos = self.get_xpos()
            self.ypos = self.get_ypos()
            Blob2D.display_id += 1.
            p = mpatches.Circle((self.xpos,self.ypos),
                                Blob2D.radius, color=circle_color)
            ax.text(self.xpos, self.ypos, '%d'%self.id,
                    horizontalalignment='center', verticalalignment='center',
                    color=text_color, fontsize=11)
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
            if not isinstance(blobs2D_list[i].parent, None.__class__):
                brothers = blobs2D_list[i].parent.children
            else:
                brothers = []
            if not blobs2D_list[i].is_sub_blob:
                brothers = []
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
    #new_blobs2D_list = blobs2D_list
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
        new_activation = blob2D.activation[associated_3D_blobs == i]
        new_blob = Blob2D(new_blob_vertices, new_blob_vertices_id,
                          new_activation, blob2D.hemisphere, blob2D, True)
        new_blob.change_to_leaf()
        # add it to 2D blobs list
        blobs2D_list.append(new_blob)
        # update parent blob
        blob2D.add_child(new_blob)
        blob2D.associate_3Dblob(None)
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

if SUBJECT == "group":
    GA_TYPE = "vrfx"
    r_path = "/data/home/virgile/virgile_internship"
    m_path = "%s/group_analysis/smoothed_FWHM0" % r_path
    lmesh_path_gii = "%s/group_analysis/surf/lh.r.white.normalized.gii" % r_path
    rmesh_path_gii = "%s/group_analysis/surf/rh.r.white.normalized.gii" % r_path
    glm_ltex_path = "%s/left_%s_%s.tex" % (m_path, GA_TYPE, CONTRAST)
    glm_rtex_path = "%s/right_%s_%s.tex" % (m_path, GA_TYPE, CONTRAST)
    glm_data_path = "%s/%s_%s.nii" % (m_path, GA_TYPE, CONTRAST)
    OUTPUT_DIR = "%s/%s/results" %(m_path, CONTRAST)
    lresults_output = "left_%s_%s_results.tex" % (GA_TYPE, CONTRAST)
    rresults_output = "right_%s_%s_results.tex" %(GA_TYPE, CONTRAST)
    OUTPUT_ENTIRE_DOMAIN_DIR = "%s/%s/results_entire_domain" %(m_path, CONTRAST)
    lresults_entire_domain_output = "left_%s_%s_results_entire_domain.tex" % (GA_TYPE, CONTRAST)
    rresults_entire_domain_output = \
        "right_%s_%s_results_entire_domain.tex" % (GA_TYPE, CONTRAST)
    OUTPUT_AUX_DIR = "%s/%s/results_aux" %(m_path, CONTRAST)
    lresults_aux_output = "left_%s_%s_results_aux.tex" \
                          % (GA_TYPE, CONTRAST)
    rresults_aux_output = "right_%s_%s_results_aux.tex" \
                          % (GA_TYPE, CONTRAST)
    OUTPUT_LARGE_AUX_DIR = "%s/%s/results_aux_large" %(m_path, CONTRAST)
    lresults_aux_large_output = "left_%s_%s_results_aux_large.tex" \
                          % (GA_TYPE, CONTRAST)
    rresults_aux_large_output = "right_%s_%s_results_aux_large.tex" \
                          % (GA_TYPE, CONTRAST)
    OUTPUT_COORD_DIR = "%s/%s/results_coord" %(m_path, CONTRAST)
    lresults_coord_output = "left_%s_%s_results_coord.tex" \
                            % (GA_TYPE, CONTRAST)
    rresults_coord_output = "right_%s_%s_results_coord.tex" \
                            % (GA_TYPE, CONTRAST)
    blobs3D_path = "%s/blobs3D_%s/leaves.nii" % (m_path, CONTRAST)
    
### Load left hemisphere data
lmesh = loadImage(lmesh_path_gii)
if SUBJECT == "group":
    c, t  = lmesh.getArrays()
else:
    c, n, t  = lmesh.getArrays()
lvertices = c.getData()
ltriangles = t.getData()
glm_ltex = tio.Texture(glm_ltex_path).read(glm_ltex_path)
blobs2D_ltex = -np.ones(glm_ltex.data.shape[0])

### Load right hemisphere data
rmesh = loadImage(rmesh_path_gii)
if SUBJECT == "group":
    c, t  = rmesh.getArrays()
else:
    c, n, t  = rmesh.getArrays()
rvertices = c.getData()
rtriangles = t.getData()
glm_rtex = tio.Texture(glm_rtex_path).read(glm_rtex_path)
blobs2D_rtex = -np.ones(glm_rtex.data.shape[0])

### Construct 2D blobs hierarchy
# right hemisphere processing
G = mesh_to_graph(rvertices, rtriangles)
F = ff.Field(G.V, G.get_edges(), G.get_weights(), glm_rtex.data)
affine = np.eye(4)
shape = None
disc = np.reshape(np.arange(F.V), (F.V, 1)).astype(int)
rnroi = hroi.NROI_from_field(F, affine, shape, disc, refdim=0,
                             th=THETA, smin=SMIN)
# create the right number of blobs
if rnroi:
    for i in np.arange(rnroi.k):
        vertices_id = np.ravel(rnroi.discrete_features['index'][i])
        new_blob = Blob2D(rvertices[vertices_id], vertices_id,
                          glm_rtex.data[vertices_id], "right")
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
            current_blob.is_meta_blob = True
            for child in children[i]:
                current_blob.add_child(Blob2D.all_blobs[child+1])
    nb_rnroi = rnroi.k

    rleaves = rnroi.reduce_to_leaves()
    idx = rleaves.discrete_features['index']
    for k in range(rleaves.k):
        blobs2D_rtex[idx[k]] =  k
else:
    nb_rnroi = 0
    
# left hemisphere processing
G = mesh_to_graph(lvertices, ltriangles)
F = ff.Field(G.V, G.get_edges(), G.get_weights(), glm_ltex.data)
affine = np.eye(4)
shape = None
disc = np.reshape(np.arange(F.V), (F.V, 1)).astype(int)
lnroi = hroi.NROI_from_field(F, affine, shape, disc, refdim=0,
                            th=THETA, smin=SMIN)
# create the right number of blobs
if lnroi:
    for i in np.arange(lnroi.k):
        vertices_id = np.ravel(lnroi.discrete_features['index'][i])
        new_blob = Blob2D(lvertices[vertices_id], vertices_id,
                          glm_ltex.data[vertices_id], "left")
    # update associations between blobs
    children = lnroi.get_children()
    parents = lnroi.get_parents()
    for i in np.arange(lnroi.k):
        current_blob = Blob2D.all_blobs[i+1+nb_rnroi]
        # set parent
        if (parents[i] != i):
            current_blob.parent = Blob2D.all_blobs[parents[i]+1+nb_rnroi]
        # set children
        if (children[i].size == 0):
            current_blob.change_to_leaf()
        else:
            current_blob.change_to_node()
            for child in children[i]:
                current_blob.add_child(Blob2D.all_blobs[child+1+nb_rnroi])

    lleaves = lnroi.reduce_to_leaves()
    idx = lleaves.discrete_features['index']
    for k in range(lleaves.k):
        blobs2D_ltex[idx[k]] =  k
    
# finally get the 2D blobs that are leaves
blobs2D_list = Blob2D.leaves.values()
# 
max_pos = np.zeros((blobs2D_list.__len__(),3))
rindex = []
lindex = []
for i in np.arange(blobs2D_list.__len__()):
    max_pos[i,:] = blobs2D_list[i].get_max_activation_location()
    if blobs2D_list[i].hemisphere == "right":
        rindex.append(i)
    else:
        lindex.append(i)

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
Blob3D(None, None, None)
nroi3D = hroi.NROI_from_field(F, affine, shape, xyz, 0, THETA3D, SMIN3D)
if nroi3D:
    nroi3D.compute_discrete_position()
    for i in np.arange(nroi3D.k):
        vertices = nroi3D.discrete_features['position'][i]
        vertices_id = np.ravel(nroi3D.discrete_features['index'][i])
        Blob3D(vertices, vertices_id, np.asarray(values)[vertices_id])
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
    
# finally get the 3D blobs that are leaves
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

### Plot the 3D blobs
data = blobs3D.get_data()
labels = sp.ndimage.measurements.label(data)[0]
if blobs3D_to_show[0] == -3:
    blobs3D_to_show = []
    for b in Blob3D.leaves.values():
        blobs3D_to_show.append(b.id)
data = np.asarray(data).copy()
for k in blobs3D_to_show:
    blob_center = Blob3D.all_blobs[k].compute_center()
    mayavi.points3d(blob_center[0], blob_center[1], blob_center[2],
                    scale_factor=1)
    data[data == k] = 110.
#    for child in Blob3D.all_blobs[k].children:
#        data[data == child.id] = 110.
data[data < 100.] = -2.
blobs3D_mayavi_src = affine_img_src(data, affine)
blobs3D_mayavi_src.image_data.point_data.add_array(
    labels.T.ravel().astype(data.dtype))
blobs3D_mayavi_src.image_data.point_data.get_array(1).name = 'labels'
blobs3D_mayavi_src.image_data.update()
src2 = mayavi.pipeline.set_active_attribute(blobs3D_mayavi_src,
                                            point_scalars='scalar')
# disable rendering for script acceleration purpose
blobs3D_mayavi_src.scene.disable_render = True
contour = mayavi.pipeline.contour(src2)
contour2 = mayavi.pipeline.set_active_attribute(contour, point_scalars='labels')
mayavi.pipeline.surface(contour2, colormap='spectral', color=(0., 0., 0.))


# -----------------------------------------------------------
# --------- Script (part 2): blobs matching -----------------
# -----------------------------------------------------------
nb_3D_blobs = blobs3D_vertices.__len__()
if nb_3D_blobs == 0:
    nb_3D_blobs = 1

#--------------------
#- FIRST ASSOCIATION
    
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

#-----------------------------------------
#- NEW ASSOCIATION AFTER 2D BLOBS DIVISION

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

#-----------------------------------------------
#- RECOMPUTE ASSOCIATION PREVENTING TWO BROTHERS
#- TO BE LINKED TO THE SAME 3D BLOB

### Match each new 2D blobs with one or several 3D blob(s)
proba, gamma = compute_association_proba(new_blobs2D_list, nb_3D_blobs,
                                         gamma_prime, sigma,
                                         dist_display[:,0:-1],
                                         exclusion=True)
### Post-processing the results
plot_matching_results(proba, dist_display, gamma_prime, gamma, sigma,
                      './results/ver0/newnew_res.txt', new_blobs2D_list,
                      blobs3D_list, explode=False)

#-----------------------------------------
#- MERGING SOME 2D BLOBS INTO THEIR PARENT

### Replace severals 2D blobs linked to the same 3D one by their hierarchichal
### parent
old_proba = []
while np.any(old_proba != proba):
    old_proba = proba
    for leaf in Blob2D.leaves.values():
        if not isinstance(leaf.parent, None.__class__):
            brothers = leaf.parent.children
        else:
            brothers = []
        all_linked_to_the_same = True
        brothers_id = []
        for j in brothers:
            brothers_id.append(j.id)
            # brother is associated to the same 3D blob
            if ((j.associated_3D_blob == leaf.associated_3D_blob) and \
                (not isinstance(j.associated_3D_blob, None.__class__))):
                all_linked_to_the_same &= True
            # brother has only one potentially associated blob (and it is
            # the which "leaf" is associated to)
            #elif ((np.shape(j.potentially_associated)[0] == 1) and \
            #     (not isinstance(leaf.associated_3D_blob, None.__class__)) and \
            #      (leaf.associated_3D_blob in j.potentially_associated)):
            #    all_linked_to_the_same &= True
            # maybe current leaf is not associated with a blob but has
            # a potentially associated one
            #elif ((np.shape(leaf.potentially_associated)[0] == 1) and \
            #     (not isinstance(j.associated_3D_blob, None.__class__)) and \
            #      (j.associated_3D_blob in leaf.potentially_associated)):
            #    all_linked_to_the_same &= True
            # brother can be linked to another 3D blob
            else:
                all_linked_to_the_same &= False
        if (all_linked_to_the_same and \
            (not isinstance(leaf.parent, None.__class__))):
            linked_3D_blob = leaf.associated_3D_blob
            parent_blob = leaf.parent
            parent_blob.associate_3Dblob(linked_3D_blob)
            for i in brothers_id:
                parent_blob.merge_child(Blob2D.leaves[i])
  
    ### Compute distances between each pair of (3D blobs)-(new 2D blobs centers)
    dist, dist_arg = compute_distances(blobs3D_vertices, Blob2D.leaves.values())
    dist_display = np.zeros((len(Blob2D.leaves.values()), nb_3D_blobs+1))
    dist_display[:,0:-1] = dist.copy()
    dist_display = np.sqrt(dist_display)
    dist_display[:,nb_3D_blobs] = -1.
    
    ### Match each new 2D blobs with one or several 3D blob(s)
    proba, gamma = compute_association_proba(Blob2D.leaves.values(),
                                             nb_3D_blobs, gamma_prime, sigma,
                                             dist_display[:,0:-1],
                                             exclusion=False)

### Post-processing the results
new_blobs2D_list = plot_matching_results(proba, dist_display, gamma_prime,
                                         gamma, sigma,
                                         './results/ver0/new_res.txt',
                                         Blob2D.leaves.values(), blobs3D_list,
                                         explode=False)


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
if blobs3D_list:
    for i in np.arange(nb_3D_blobs):
        association_list = []
        for blob2D in Blob2D.leaves.values():
            if (blobs3D_list[i] in blob2D.potentialy_associated):
                association_list.append(blob2D)
            elif (not isinstance(blob2D.associated_3D_blob,None.__class__) and \
                  blob2D.associated_3D_blob.id == blobs3D_list[i].id):
                association_list.insert(0,blob2D)
        association_list = sort_list_by_link(association_list,
                                             blobs3D_list[i].id)
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
lines_colors = ['b','g','c','m','k','y', (1.,0.5,0.), (0.5,0.5,0.), (0.,0.5,0.)]

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
            ax.plot([b.get_xpos()+Blob2D.radius/2.,
                     Blob3D.leaves[blob3D_id].get_xpos()-Blob3D.radius/2.],
                    [b.get_ypos(),Blob3D.leaves[blob3D_id].get_ypos()],
                    color=lines_colors[blob3D_id%lines_colors.__len__()])
        elif link > 100*threshold_maybe:
            ax.plot([b.get_xpos()+Blob2D.radius/2.,
                     Blob3D.leaves[blob3D_id].get_xpos()-Blob3D.radius/2.],
                    [b.get_ypos(),Blob3D.leaves[blob3D_id].get_ypos()],
                    '--', color=lines_colors[blob3D_id%lines_colors.__len__()])

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
        ax.plot([child.get_xpos()+Blob3D.radius/2.,
                 blob3D.get_xpos()-Blob3D.radius/2.],
                [child.get_ypos(),blob3D.get_ypos()],
                color='black')

# display 2D blobs hierarchy
for blob2D in Blob2D.nodes.values():
    blob2D.display(ax)
    for child in blob2D.children:
        if child.is_sub_blob:
            ax.plot([child.get_xpos()-Blob2D.radius/2.,
                     blob2D.get_xpos()+Blob2D.radius/2.],
                    [child.get_ypos(),blob2D.get_ypos()],
                    color=lines_colors[blob2D.id%lines_colors.__len__()])
        else:
            ax.plot([child.get_xpos()-Blob2D.radius/2.,
                     blob2D.get_xpos()+Blob2D.radius/2.],
                    [child.get_ypos(),blob2D.get_ypos()],
                    color='black')

# choose textures
ltex = blobs2D_ltex.copy()
rtex = blobs2D_rtex.copy()
if blobs2D_to_show_bckup[0] != -2.:
    if blobs2D_to_show_bckup[0] == -3.:
        blobs2D_to_show = []
        for b in Blob2D.leaves.values():
            blobs2D_to_show.append(b.id)
    ltex[:] = -1.
    rtex[:] = -1.
    for i in blobs2D_to_show:
        blob = Blob2D.all_blobs[i]
        if (not isinstance(blob.associated_3D_blob, None.__class__)):
            value = blob.associated_3D_blob.id
        else:
            value = -0.7
        if blob.hemisphere == "left":
            ltex[blob.vertices_id] = value
        else:
            rtex[blob.vertices_id] = value

if blobs2D_to_show_bckup[0] == -3.:
    ### Finally write output (right and left) textures
    out_dir = "%s_level%03d" %(OUTPUT_DIR, 1)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_rtex = tio.Texture("%s/%s" %(out_dir,rresults_output), data=rtex)
    output_rtex.write()
    output_ltex = tio.Texture("%s/%s" %(out_dir,lresults_output), data=ltex)
    output_ltex.write()
    
    ### Output textures with entire domain
    # fill the entire blob domain
    ltex_entire = ltex.copy()
    rtex_entire = rtex.copy()
    for b in Blob2D.nodes.values():
        if b.hemisphere == "left":
            the_tex = ltex_entire
        else:
            the_tex = rtex_entire
        for i in b.vertices_id:
            if the_tex[i] == -1:
                the_tex[i] = -0.7
    # write results
    out_dir = "%s_level%03d" %(OUTPUT_ENTIRE_DOMAIN_DIR, 1)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_entire_domain_rtex = tio.Texture("%s/%s" %(out_dir,rresults_entire_domain_output), data=rtex_entire)
    output_entire_domain_rtex.write()
    output_entire_domain_ltex = tio.Texture("%s/%s" %(out_dir,lresults_entire_domain_output), data=ltex_entire)
    output_entire_domain_ltex.write()
    
    ### Auxiliary results large domain
    all_rvertices = np.array([[],[],[]], ndmin=2).T
    all_lvertices = np.array([[],[],[]], ndmin=2).T
    all_rvertices_id = np.array([], dtype=int)
    all_lvertices_id = np.array([], dtype=int)
    for b in Blob2D.all_blobs.values():
        if b.hemisphere == "right":
            all_rvertices = np.concatenate((all_rvertices, b.vertices))
            all_rvertices_id = np.concatenate((all_rvertices_id, b.vertices_id))
        else:
            all_lvertices = np.concatenate((all_lvertices, b.vertices))
            all_lvertices_id = np.concatenate((all_lvertices_id, b.vertices_id))
    # right hemisphere cluster
    rassignment = cl.voronoi(all_rvertices, max_pos[rindex])
    rtex_aux_large = -np.ones(rtex.shape[0])
    rtex_aux_large[all_rvertices_id] = rassignment
    # left hemisphere cluster
    lassignment = cl.voronoi(all_lvertices, max_pos[lindex])
    ltex_aux_large = -np.ones(ltex.shape[0])
    ltex_aux_large[all_lvertices_id] = lassignment
    # write results
    out_dir = "%s_level%03d" %(OUTPUT_LARGE_AUX_DIR, 1)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_aux_large_rtex = tio.Texture("%s/%s" %(out_dir,rresults_aux_large_output), data=rtex_aux_large)
    output_aux_large_rtex.write()
    output_aux_large_ltex = tio.Texture("%s/%s" %(out_dir,lresults_aux_large_output), data=ltex_aux_large)
    output_aux_large_ltex.write()
    
    ### Auxiliary results restricted domain
    # right hemisphere cluster
    all_rblobs_vertices = rvertices[rtex != -1]
    rassignment = cl.voronoi(all_rblobs_vertices, max_pos[rindex])
    rtex_aux = -np.ones(rtex.shape[0])
    rtex_aux[rtex != -1] = rassignment
    # left hemisphere cluster
    all_lblobs_vertices = lvertices[ltex != -1]
    lassignment = cl.voronoi(all_lblobs_vertices, max_pos[lindex])
    ltex_aux = -np.ones(ltex.shape[0])
    ltex_aux[ltex != -1] = lassignment
    # write results
    out_dir = "%s_level%03d" %(OUTPUT_AUX_DIR, 1)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_aux_rtex = tio.Texture("%s/%s" %(out_dir,rresults_aux_output), data=rtex_aux)
    output_aux_rtex.write()
    output_aux_ltex = tio.Texture("%s/%s" %(out_dir,lresults_aux_output), data=ltex_aux)
    output_aux_ltex.write()

    ### Coordinates results
    rtex_coord = -np.ones(rtex.size)
    ltex_coord = -np.ones(ltex.size)
    max_region = {}
    max_region_location = {}
    max_region_hemisphere = {}
    for b in Blob2D.leaves.values():
        if b.associated_3D_blob is not None and \
           b.associated_3D_blob.id != 0:
            if b.associated_3D_blob.id in max_region.keys():
                if max_region[b.associated_3D_blob.id] < b.get_argmax_activation():
                    max_region[b.associated_3D_blob.id] = \
                                b.get_argmax_activation()
                    max_region_location[b.associated_3D_blob.id] = \
                                b.vertices_id[b.get_argmax_activation()]
                    max_region_hemisphere[b.associated_3D_blob.id] = \
                                b.hemisphere
            else:
                max_region[b.associated_3D_blob.id] = \
                                b.get_argmax_activation()
                max_region_location[b.associated_3D_blob.id] = \
                                b.vertices_id[b.get_argmax_activation()]
                max_region_hemisphere[b.associated_3D_blob.id] = \
                                b.hemisphere
    for r in max_region.keys():
        if max_region_hemisphere[r] == "right":
            rtex_coord[max_region_location[r]] = 10.
        else:
            ltex_coord[max_region_location[r]] = 10.
    # write results
    out_dir = "%s_level%03d" %(OUTPUT_COORD_DIR, 1)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_coord_rtex = tio.Texture("%s/%s" %(out_dir,rresults_coord_output), data=rtex_coord)
    output_coord_rtex.write()
    output_coord_ltex = tio.Texture("%s/%s" %(out_dir,lresults_coord_output), data=ltex_coord)
    output_coord_ltex.write()

    ### Coordinates former results
    rtex_fcoord = -np.ones(rtex.size)
    ltex_fcoord = -np.ones(ltex.size)
    for b in Blob2D.leaves.values():
        if b.hemisphere == "right":
            rtex_fcoord[b.vertices_id[b.get_argmax_activation()]] = 10.
        else:
            ltex_fcoord[b.vertices_id[b.get_argmax_activation()]] = 10.
    # write results
    out_dir = "%s_level%03d" %(OUTPUT_FCOORD_DIR, 1)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_fcoord_rtex = tio.Texture("%s/%s" %(out_dir,rresults_fcoord_output), data=rtex_fcoord)
    output_fcoord_rtex.write()
    output_fcoord_ltex = tio.Texture("%s/%s" %(out_dir,lresults_fcoord_output), data=ltex_fcoord)
    output_fcoord_ltex.write()

    if mayavi_outtex_level == 1:
        if mayavi_outtex_type == "aux":
            mayavi_routtex = rtex_aux_large
            mayavi_louttex = ltex_aux
        elif mayavi_outtex_type == "aux_large":
            mayavi_routtex = rtex_aux_large
            mayavi_louttex = ltex_aux_large
        elif mayavi_outtex_type == "coord":
            mayavi_routtex = rtex_coord
            mayavi_louttex = ltex_coord
        elif mayavi_outtex_type == "fcoord":
            mayavi_routtex = rtex_fcoord
            mayavi_louttex = ltex_fcoord
        elif mayavi_outtex_type == "entire":
            mayavi_routtex = rtex_entire
            mayavi_louttex = ltex_entire
        else:
            mayavi_routtex = rtex
            mayavi_louttex = ltex


#--------------------------------------------
#--------------------------------------------
#----- OTHERS LEVELS ------------------------
#--------------------------------------------
#--------------------------------------------
level = 1
while len(Blob2D.nodes) != 0 or len(Blob3D.nodes) != 0:
    level += 1
    # Merge all leaves in their parent structure
    # (keep lonely leaves as they are)
    for l in Blob2D.leaves.values():
        if l.parent is not None:
            l.parent.merge_child(l)
        elif l.parent is None and l.associated_3D_blob is not None and \
             l.associated_3D_blob.id == 0:
            del Blob2D.leaves[l.id]
    
    for l in Blob3D.leaves.values():
        if l.parent is not None:
            l.parent.merge_child(l)
    
    for l in Blob2D.nodes.values():
        l.already_displayed = False
    for l in Blob3D.nodes.values():
        l.already_displayed = False
    for l in Blob2D.leaves.values():
        l.already_displayed = False
    for l in Blob3D.leaves.values():
        l.already_displayed = False
    
    # Algorithm initialization
    blobs2D_list = Blob2D.leaves.values()
    blobs3D_vertices = []
    blobs3D_list = []
    for b in Blob3D.leaves.values():
        blobs3D_vertices.append(b.vertices)
        blobs3D_list.append(b)
    nb_3D_blobs = blobs3D_vertices.__len__()
    
    #--------------------
    #- FIRST ASSOCIATION
        
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
                                             gamma, sigma, './results/ver0/2nd_res.txt',
                                             blobs2D_list, blobs3D_list,
                                             explode=False)
    
    #-----------------------------------------
    #- MERGING SOME 2D BLOBS INTO THEIR PARENT
    
    ### Replace severals 2D blobs linked to the same 3D one by their hierarchichal
    ### parent
    old_proba = []
    while np.any(old_proba != proba):
        old_proba = proba
        for leaf in Blob2D.leaves.values():
            if not isinstance(leaf.parent, None.__class__):
                brothers = leaf.parent.children
            else:
                brothers = []
            all_linked_to_the_same = True
            brothers_id = []
            for j in brothers:
                brothers_id.append(j.id)
                # brother is associated to the same 3D blob
                if ((j.associated_3D_blob == leaf.associated_3D_blob) and \
                    (not isinstance(j.associated_3D_blob, None.__class__))):
                    all_linked_to_the_same &= True
                # brother has only one potentially associated blob (and it is
                # the which "leaf" is associated to)
                #elif ((np.shape(j.potentially_associated)[0] == 1) and \
                #     (not isinstance(leaf.associated_3D_blob, None.__class__)) and \
                #      (leaf.associated_3D_blob in j.potentially_associated)):
                #    all_linked_to_the_same &= True
                # maybe current leaf is not associated with a blob but has
                # a potentially associated one
                #elif ((np.shape(leaf.potentially_associated)[0] == 1) and \
                #     (not isinstance(j.associated_3D_blob, None.__class__)) and \
                #      (j.associated_3D_blob in leaf.potentially_associated)):
                #    all_linked_to_the_same &= True
                # brother can be linked to another 3D blob
                else:
                    all_linked_to_the_same &= False
            if (all_linked_to_the_same and \
                (not isinstance(leaf.parent, None.__class__))):
                linked_3D_blob = leaf.associated_3D_blob
                parent_blob = leaf.parent
                parent_blob.associate_3Dblob(linked_3D_blob)
                for i in brothers_id:
                    parent_blob.merge_child(Blob2D.leaves[i])
      
        ### Compute distances between each pair of (3D blobs)-(new 2D blobs centers)
        dist, dist_arg = compute_distances(blobs3D_vertices, Blob2D.leaves.values())
        dist_display = np.zeros((len(Blob2D.leaves.values()), nb_3D_blobs+1))
        dist_display[:,0:-1] = dist.copy()
        dist_display = np.sqrt(dist_display)
        dist_display[:,nb_3D_blobs] = -1.
        
        ### Match each new 2D blobs with one or several 3D blob(s)
        proba, gamma = compute_association_proba(Blob2D.leaves.values(),
                                                 nb_3D_blobs, gamma_prime, sigma,
                                                 dist_display[:,0:-1],
                                                 exclusion=False)
    
    ### Post-processing the results
    new_blobs2D_list = plot_matching_results(proba, dist_display, gamma_prime,
                                             gamma, sigma,
                                             './results/ver0/2nd_new_res.txt',
                                             Blob2D.leaves.values(), blobs3D_list,
                                             explode=False)
    
    # construct a list of associated 2D blobs for each 3D blobs
    # and sort it by their link probability value
    nested_association_lists = []
    if blobs3D_list:
        for i in np.arange(nb_3D_blobs):
            association_list = []
            for blob2D in Blob2D.leaves.values():
                if (blobs3D_list[i] in blob2D.potentialy_associated):
                    association_list.append(blob2D)
                elif (not isinstance(blob2D.associated_3D_blob,None.__class__) and \
                      blob2D.associated_3D_blob.id == blobs3D_list[i].id):
                    association_list.insert(0,blob2D)
            association_list = sort_list_by_link(association_list,
                                                 blobs3D_list[i].id)
            if association_list:
                association_list.insert(0, blobs3D_list[i].id)
                nested_association_lists.append(association_list)
        
    # set matplotlib figure basis
    fig = plt.figure(level)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    #ax.text((Blob3D.default_xpos-Blob2D.default_xpos)/2, 15,
    #        "Subject %s, Contrast %s, gamma=%g\n2nd level" %(SUBJECT, CONTRAST, gamma),
    #        horizontalalignment='center')
    ax.set_xlim(Blob2D.default_xpos-15, Blob3D.default_xpos+15)
    #ax.set_ylim(-Blob2DDisplay.spacing*np.amax([nb_linked,
    #                                            nb_3D_blobs-1]),25)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    lines_colors = ['b','g','c','m','k','y', (1.,0.5,0.), (0.5,0.5,0.), (0.,0.5,0.)]
    
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
                ax.plot([b.get_xpos()+Blob2D.radius/2.,
                         Blob3D.leaves[blob3D_id].get_xpos()-Blob3D.radius/2.],
                        [b.get_ypos(),Blob3D.leaves[blob3D_id].get_ypos()],
                        color=lines_colors[blob3D_id%lines_colors.__len__()])
            elif link > 100*threshold_maybe:
                ax.plot([b.get_xpos()+Blob2D.radius/2.,
                         Blob3D.leaves[blob3D_id].get_xpos()-Blob3D.radius/2.],
                        [b.get_ypos(),Blob3D.leaves[blob3D_id].get_ypos()],
                        '--', color=lines_colors[blob3D_id%lines_colors.__len__()])
    
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
            ax.plot([child.get_xpos()+Blob3D.radius/2.,
                     blob3D.get_xpos()-Blob3D.radius/2.],
                    [child.get_ypos(),blob3D.get_ypos()],
                    color='black')
    
    # display 2D blobs hierarchy
    for blob2D in Blob2D.nodes.values():
        blob2D.display(ax)
        for child in blob2D.children:
            if child.is_sub_blob:
                ax.plot([child.get_xpos()-Blob2D.radius/2.,
                         blob2D.get_xpos()+Blob2D.radius/2.],
                        [child.get_ypos(),blob2D.get_ypos()],
                        color=lines_colors[blob2D.id%lines_colors.__len__()])
            else:
                ax.plot([child.get_xpos()-Blob2D.radius/2.,
                         blob2D.get_xpos()+Blob2D.radius/2.],
                        [child.get_ypos(),blob2D.get_ypos()],
                        color='black')

    # Update Blob2D.all_blobs
    # /!\ fixme : very dirty !
    for leaf in Blob2D.leaves.values():
        Blob2D.all_blobs[leaf.id] = leaf
    for node in Blob2D.nodes.values():
        Blob2D.all_blobs[node.id] = node
    if blobs2D_to_show_bckup[0] == -3.:
        # choose textures
        ltex = blobs2D_ltex.copy()
        rtex = blobs2D_rtex.copy()
        if blobs2D_to_show_bckup[0] != -2.:
            if blobs2D_to_show_bckup[0] == -3.:
                blobs2D_to_show = []
                for b in Blob2D.leaves.values():
                    blobs2D_to_show.append(b.id)
            ltex[:] = -1.
            rtex[:] = -1.
            for i in blobs2D_to_show:
                blob = Blob2D.leaves[i]
                if (not isinstance(blob.associated_3D_blob, None.__class__)):
                    value = blob.associated_3D_blob.id
                else:
                    value = -0.7
                if blob.hemisphere == "left":
                    ltex[blob.vertices_id] = value
                else:
                    rtex[blob.vertices_id] = value

        ### Finally write output (right and left) textures
        out_dir = "%s_level%03d" %(OUTPUT_DIR, level)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        output_rtex = tio.Texture("%s/%s" %(out_dir,rresults_output), data=rtex)
        output_rtex.write()
        output_ltex = tio.Texture("%s/%s" %(out_dir,lresults_output), data=ltex)
        output_ltex.write()
        
        ### Output textures with entire domain
        # fill the entire blob domain
        ltex_entire = ltex.copy()
        rtex_entire = rtex.copy()
        for b in Blob2D.nodes.values():
            if b.hemisphere == "left":
                the_tex = ltex_entire
            else:
                the_tex = rtex_entire
            for i in b.vertices_id:
                if the_tex[i] == -1:
                    the_tex[i] = -0.7
        # write results
        out_dir = "%s_level%03d" %(OUTPUT_ENTIRE_DOMAIN_DIR, level)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        output_entire_domain_rtex = tio.Texture("%s/%s" %(out_dir,rresults_entire_domain_output), data=rtex_entire)
        output_entire_domain_rtex.write()
        output_entire_domain_ltex = tio.Texture("%s/%s" %(out_dir,lresults_entire_domain_output), data=ltex_entire)
        output_entire_domain_ltex.write()
        
        ### Auxiliary results large domain
        all_rvertices = np.array([[],[],[]], ndmin=2).T
        all_lvertices = np.array([[],[],[]], ndmin=2).T
        all_rvertices_id = np.array([], dtype=int)
        all_lvertices_id = np.array([], dtype=int)
        for b in Blob2D.all_blobs.values():
            if b.hemisphere == "right":
                all_rvertices = np.concatenate((all_rvertices, b.vertices))
                all_rvertices_id = np.concatenate((all_rvertices_id, b.vertices_id))
            else:
                all_lvertices = np.concatenate((all_lvertices, b.vertices))
                all_lvertices_id = np.concatenate((all_lvertices_id, b.vertices_id))
        # right hemisphere cluster
        rassignment = cl.voronoi(all_rvertices, max_pos[rindex])
        rtex_aux_large = -np.ones(rtex.shape[0])
        rtex_aux_large[all_rvertices_id] = rassignment
        # left hemisphere cluster
        lassignment = cl.voronoi(all_lvertices, max_pos[lindex])
        ltex_aux_large = -np.ones(ltex.shape[0])
        ltex_aux_large[all_lvertices_id] = lassignment
        # write results
        out_dir = "%s_level%03d" %(OUTPUT_LARGE_AUX_DIR, level)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        output_aux_large_rtex = tio.Texture("%s/%s" %(out_dir,rresults_aux_large_output), data=rtex_aux_large)
        output_aux_large_rtex.write()
        output_aux_large_ltex = tio.Texture("%s/%s" %(out_dir,lresults_aux_large_output), data=ltex_aux_large)
        output_aux_large_ltex.write()
        
        ### Auxiliary results restricted domain
        # right hemisphere cluster
        all_rblobs_vertices = rvertices[rtex != -1]
        rassignment = cl.voronoi(all_rblobs_vertices, max_pos[rindex])
        rtex_aux = -np.ones(rtex.shape[0])
        rtex_aux[rtex != -1] = rassignment
        # left hemisphere cluster
        all_lblobs_vertices = lvertices[ltex != -1]
        lassignment = cl.voronoi(all_lblobs_vertices, max_pos[lindex])
        ltex_aux = -np.ones(ltex.shape[0])
        ltex_aux[ltex != -1] = lassignment
        # write results
        out_dir = "%s_level%03d" %(OUTPUT_AUX_DIR, level)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        output_aux_rtex = tio.Texture("%s/%s" %(out_dir,rresults_aux_output), data=rtex_aux)
        output_aux_rtex.write()
        output_aux_ltex = tio.Texture("%s/%s" %(out_dir,lresults_aux_output), data=ltex_aux)
        output_aux_ltex.write()

        ### Coordinates results
        rtex_coord = -np.ones(rtex.size)
        ltex_coord = -np.ones(ltex.size)
        max_region = {}
        max_region_location = {}
        max_region_hemisphere = {}
        for b in Blob2D.leaves.values():
            if b.associated_3D_blob is not None and \
               b.associated_3D_blob.id != 0:
                if b.associated_3D_blob.id in max_region.keys():
                    if max_region[b.associated_3D_blob.id] < b.get_argmax_activation():
                        max_region[b.associated_3D_blob.id] = \
                                    b.get_argmax_activation()
                        max_region_location[b.associated_3D_blob.id] = \
                                    b.vertices_id[b.get_argmax_activation()]
                        max_region_hemisphere[b.associated_3D_blob.id] = \
                                    b.hemisphere
                else:
                    max_region[b.associated_3D_blob.id] = \
                                    b.get_argmax_activation()
                    max_region_location[b.associated_3D_blob.id] = \
                                    b.vertices_id[b.get_argmax_activation()]
                    max_region_hemisphere[b.associated_3D_blob.id] = \
                                    b.hemisphere
        for r in max_region.keys():
            if max_region_hemisphere[r] == "right":
                rtex_coord[max_region_location[r]] = 10.
            else:
                ltex_coord[max_region_location[r]] = 10.
        # write results
        out_dir = "%s_level%03d" %(OUTPUT_COORD_DIR, level)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        output_coord_rtex = tio.Texture("%s/%s" %(out_dir,rresults_coord_output), data=rtex_coord)
        output_coord_rtex.write()
        output_coord_ltex = tio.Texture("%s/%s" %(out_dir,lresults_coord_output), data=ltex_coord)
        output_coord_ltex.write()
        
        ### Coordinates former results
        rtex_fcoord = -np.ones(rtex.size)
        ltex_fcoord = -np.ones(ltex.size)
        for b in Blob2D.leaves.values():
            if b.hemisphere == "right":
                rtex_fcoord[b.vertices_id[b.get_argmax_activation()]] = 10.
            else:
                ltex_fcoord[b.vertices_id[b.get_argmax_activation()]] = 10.
        # write results
        out_dir = "%s_level%03d" %(OUTPUT_FCOORD_DIR, level)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        output_fcoord_rtex = tio.Texture("%s/%s" %(out_dir,rresults_fcoord_output), data=rtex_fcoord)
        output_fcoord_rtex.write()
        output_fcoord_ltex = tio.Texture("%s/%s" %(out_dir,lresults_fcoord_output), data=ltex_fcoord)
        output_fcoord_ltex.write()

        if mayavi_outtex_level == level:
            if mayavi_outtex_type == "aux":
                mayavi_routtex = rtex_aux_large
                mayavi_louttex = ltex_aux
            elif mayavi_outtex_type == "aux_large":
                mayavi_routtex = rtex_aux_large
                mayavi_louttex = ltex_aux_large
            elif mayavi_outtex_type == "coord":
                mayavi_routtex = rtex_coord
                mayavi_louttex = ltex_coord
            elif mayavi_outtex_type == "fcoord":
                mayavi_routtex = rtex_fcoord
                mayavi_louttex = ltex_fcoord
            elif mayavi_outtex_type == "entire":
                mayavi_routtex = rtex_entire
                mayavi_louttex = ltex_entire
            else:
                mayavi_routtex = rtex
                mayavi_louttex = ltex

if mayavi_outtex_level == -1 or mayavi_outtex_level > level:
    if mayavi_outtex_type == "aux":
        mayavi_routtex = rtex_aux_large
        mayavi_louttex = ltex_aux
    elif mayavi_outtex_type == "aux_large":
        mayavi_routtex = rtex_aux_large
        mayavi_louttex = ltex_aux_large
    elif mayavi_outtex_type == "coord":
        mayavi_routtex = rtex_coord
        mayavi_louttex = ltex_coord
    elif mayavi_outtex_type == "fcoord":
        mayavi_routtex = rtex_fcoord
        mayavi_louttex = ltex_fcoord
    elif mayavi_outtex_type == "entire":
        mayavi_routtex = rtex_entire
        mayavi_louttex = ltex_entire
    else:
        mayavi_routtex = rtex
        mayavi_louttex = ltex


### Mayavi Plot
# colors
lut_colors = np.array([[64,0,64,255], [128,0,64,255], [192,0,64,255],
                       [64,0,128,255], [128,0,128,255], [192,0,128,255],
                       [64,0,192,255], [128,0,192,255], [192,0,192,255],
                       [0,64,64,255], [0,128,64,255], [0,192,64,255],
                       [0,64,128,255], [0,128,128,255], [0,192,128,255],
                       [0,64,192,255], [0,128,192,255], [0,192,192,255],
                       [64,64,64,255], [128,64,64,255], [192,64,64,255],
                       [64,64,128,255], [128,64,128,255], [192,64,128,255],
                       [64,64,192,255], [128,64,192,255], [192,64,192,255],
                                        [64,128,64,255], [64,192,64,255],
                       [64,64,128,255], [64,128,128,255], [64,192,128,255],
                       [64,64,192,255], [64,128,192,255], [64,192,192,255],
                       [192,64,64,255], [192,128,64,255], [192,192,64,255],
                       [192,64,128,255], [192,128,128,255], [192,192,128,255],
                       [192,64,192,255], [192,128,192,255],                 
                       [64,64,64,255], [128,64,64,255], [192,64,64,255],
                       [64,64,128,255], [128,64,128,255], [192,64,128,255],
                       [64,64,192,255], [128,64,192,255], [192,64,192,255],
                       [64,128,64,255], [128,128,64,255], [192,128,64,255],
                       [64,128,128,255],                  [192,128,128,255],
                       [64,128,192,255], [128,128,192,255], [192,128,192,255],
                       [128,64,64,255], [128,128,64,255], [128,192,64,255],
                       [128,64,128,255], [128,128,128,255], [128,192,128,255],
                       [128,64,192,255], [128,128,192,255], [128,192,192,255],
                       [64,0,0,255], [128,0,0,255], [192,0,0,255],
                       [0,64,0,255], [0,128,0,255], [0,128,0,255],
                       [0,0,64,255], [0,0,128,255], [0,0,192,255],
                       [64,64,0,255], [128,64,0,255], [192,64,0,255],
                       [64,128,0,255], [128,128,0,255], [192,128,0,255],
                       [64,192,0,255], [128,192,0,255], [192,192,0,255]],
                      dtype=int)
                       
# plot left hemisphere
mayavi_lmesh = mayavi.triangular_mesh(lvertices[:,0], lvertices[:,1],
                                      lvertices[:,2], ltriangles,
                                      scalars=mayavi_louttex,
                                      transparent=False, opacity=1.)
lhlut = mayavi_lmesh.module_manager.scalar_lut_manager.lut.table.to_array()
if np.amax(mayavi_louttex)+1 == 0.:
    factor = 1
else:
    factor = int(np.amax(mayavi_louttex)+1)
ratio = 256./factor
# brain background color
lhlut[:np.floor(0.3*ratio),:] = np.array([127,127,127,255])
# 2D blobs with linkage doubt color
lhlut[np.floor(0.3*ratio):ratio,:] = np.array([255,255,255,255])
# 2D blobs associated to no 3D blob color
lhlut[ratio:2*ratio,:] = np.array([0,0,0,255])
mayavi_lmesh.module_manager.scalar_lut_manager.lut.table = lhlut
for i in range(2,factor):
    lhlut[i*int(ratio):(i+1)*int(ratio),:] = lut_colors[6+i-2]

# plot right hemisphere
mayavi_rmesh = mayavi.triangular_mesh(rvertices[:,0], rvertices[:,1],
                                      rvertices[:,2], rtriangles,
                                      scalars=mayavi_routtex,
                                      transparent=False, opacity=1.)
rhlut = mayavi_rmesh.module_manager.scalar_lut_manager.lut.table.to_array()
if np.amax(mayavi_routtex)+1 == 0.:
    factor = 1
else:
    factor = int(np.amax(mayavi_routtex)+1)
ratio = 256./factor
# brain background color
rhlut[:np.floor(0.3*ratio),:] = np.array([127,127,127,255])
# 2D blobs with linkage doubt color
rhlut[np.floor(0.3*ratio):ratio,:] = np.array([255,255,255,255])
# 2D blobs associated to no 3D blob color
rhlut[ratio:2*ratio,:] = np.array([0,0,0,255])
mayavi_rmesh.module_manager.scalar_lut_manager.lut.table = rhlut
for i in range(2,factor):
    rhlut[i*int(ratio):(i+1)*int(ratio),:] = lut_colors[6+i-2]

# enable mayavi rendering (because we have disabled it)
blobs3D_mayavi_src.scene.disable_render = False 
# show matplotlib graphics
plt.show()
