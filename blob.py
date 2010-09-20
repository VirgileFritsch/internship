import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.text as mtxt

class Blob:
    """Represents a blob in a context of a hierarchical model.

    Attributes
    ----------
    vertices : array, shape = [3, n_vertices]
        Blob vertices coordinates.

    vertices_id : array, type = int, shape = [1, n_vertices]
        Blobs vertices ids.

    activation : array, type = float, shape = [1, n_vertices]
        Vertices intensities.

    children : list, type = Blob, shape = [1, n_children]
        Children of the blob in the hierarchy.

    parent : Blob
        Parent of the blob in the hierarchy.

    #----------------------------------------------------------------
    #----- This can be ignored when not looking at the code in detail.
    y_pos_value : float
        Default y position for matching plotting.

    y_pos_coeffs : float
    #----------------------------------------------------------------

    Methods
    -------
    add_child(child_to_add) :
        Adds a child to the current blob.
        
    remove_child(child_to_remove) :
        Removes a child of the current blob.

    # deprecated ?
    get_max_activation_location() :
        Gets the position of the blob as the one of its most intensive voxel.
    
    # deprecated !
    set_parent(parent)
    get_max_activation()
    get_argmax_activation()
    
    """

    def __init__(self, vertices_list, vertices_id_list,
                 activation, parent=None):
        """Constructs a new blob.
        
        """
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
        """Adds a child to the current blob.

        In the case where the blob already has a child, it just adds
        the new one the list of children. If the current blob is a
        leaf, the method tags it as a node after having added the
        child. Indeed, the current blob is no more an end-bob in the
        blobs hierarchy.
        
        """
        if (child_to_add not in self.children):
            self.children.append(child_to_add)
        self.change_to_node()

    def remove_child(self, child_to_remove):
        """Removes a current blob's child.

        The method removes the given child from te current blob's
        children list. If there is no more child is that list, the
        current blob is tagged as a leaf, because it has becomed an
        end-blob in the blobs hierarchy.
        """
        self.children.remove(child_to_remove)
        if (not self.children):
            self.change_to_leaf()

    def set_parent(self, parent):
        """
        """
        parent.add_child(self)
        self.parent = parent

    def get_max_activation(self):
        """
        """
        return np.amax(self.activation)

    def get_argmax_activation(self):
        """
        """
        return np.argmax(self.activation)

    def get_max_activation_location(self):
        """
        """
        return self.vertices[np.argmax(self.activation)]


class Blob3D(Blob):
    """Represents a 3D blob in a context of a hierarchical model.

    Class variables
    ---------------
    nb_blobs : int
        Number of instantiated 3D blobs. Used to compute new blobs' id.

    all_blobs : dict, type = int -> Blob3D
        All the instantiated blobs, indexed by their ids.
        We are sure to find any 3D blob in that dictionary.

    leaves : dict, type = int -> Blob3D
        All the instantiated blobs that are leaves, indexed by their ids.
        This dictionary is ought to change over the steps of the algorithm.
        It is refered to only to loop over the leaves.
        /!\ Maybe could we get rid of it...

    nodes : dict, type = int -> Blob3D
        All the instantiated blobs that are nodes, indexed by their ids.
        This dictionary is ought to change over the steps of the algorithm.
        It is refered to only to loop over the nodes.
        /!\ Maybe could we get rid of it...
        
    """
    nb_blobs = -1
    all_blobs = {}
    leaves = {}
    nodes = {}

    # Plotting parameters
    spacing = 8.
    default_xpos = 35.
    radius = 8.
    
    def __init__(self, vertices_list, vertices_id_list,
                 activation, parent=None):
        """Constructs a new 3D blob.
        
        """
        # create a new blob
        Blob.__init__(self, vertices_list, vertices_id_list,
                      activation, parent)

        # set blob id and increase total number of 3D blobs
        self.id = Blob3D.nb_blobs + 1
        Blob3D.nb_blobs += 1
        Blob3D.all_blobs[self.id] = self
        if self.id != 0:
            Blob3D.leaves[self.id] = self

        # tag to avoid displaying the same blob several times
        self.already_displayed = False

    def compute_center(self):
        """Computes current blob center.

        This is available as a method and not as an attribute because
        the current blob's vertices are likely to change through the
        steps of the algorithm.
        
        """
        if not isinstance(self.vertices, None.__class__):
            tmp_vertices = self.vertices.copy()
            for child in self.children:
                tmp_vertices = np.vstack((tmp_vertices, child.vertices.copy()))
            res = np.mean(tmp_vertices, 0)
        else:
            res = np.nan
        
        return res

    def is_leaf(self):
        """Tests whether or not the current blob is a leaf.

        I don't like the way it's done.

        """
        if self.id in Blob3D.leaves.keys():
            res = True
        else:
            res = False

        return res

    def is_node(self):
        """Tests whether or not the current blob is a node.

        I don't like the way it's done.

        """
        if self.id in Blob3D.nodes.keys():
            res = True
        else:
            res = False

        return res

    def change_to_leaf(self):
        """Turns the current blob to a leaf.

        A blob which was a node can become a leaf.
        This can happen when we remove the last of its children.
        
        """
        if self.is_node():
            del  Blob3D.nodes[self.id]
        if not self.is_leaf():
            Blob3D.leaves[self.id] = self

    def change_to_node(self):
        """Turns the current blob to a node.

        A blob which was a leaf can become a node.
        This can happen when we add it a child.
        
        """
        if self.is_leaf():
            del Blob3D.leaves[self.id]
        if not self.is_node():
            Blob3D.nodes[self.id] = self

    ### Methods for matching plotting (will be explained later...)
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
                    if (linked_blob.associated_3D_blob == self) or \
                       (self in linked_blob.potentialy_associated):
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
    """Represents a 2D blob in a context of a hierarchical model.

    Class variables
    ---------------
    nb_blobs : int
        Number of instantiated 2D blobs. Used to compute new blobs' id.

    all_blobs : dict, type = int -> Blob2D
        All the instantiated blobs, indexed by their ids.
        We are sure to find any 2D blob in that dictionary.

    leaves : dict, type = int -> Blob2D
        All the instantiated blobs that are leaves, indexed by their ids.
        This dictionary is ought to change over the steps of the algorithm.
        It is refered to only to loop over the leaves.
        /!\ Maybe could we get rid of it...

    nodes : dict, type = int -> Blob2D
        All the instantiated blobs that are nodes, indexed by their ids.
        This dictionary is ought to change over the steps of the algorithm.
        It is refered to only to loop over the nodes.
        /!\ Maybe could we get rid of it...
        
    """
    nb_blobs = 0
    all_blobs = {}
    leaves = {}
    nodes = {}
    
    # Plotting parameters
    display_id = 0.
    default_xpos = 0.
    radius = 8.
    spacing = 30.
    
    def __init__(self, vertices_list, vertices_id_list, activation,
                 hemisphere, parent=None, sub_blob=False, meta_blob=False):
        """Construct a new 2D blob.
        
        """
        # create a new blob
        Blob.__init__(self, vertices_list, vertices_id_list,
                      activation, parent)
        self.hemisphere = hemisphere
        
        # set blob id and increase total number of 2D blobs
        self.id = Blob2D.nb_blobs + 1
        Blob2D.nb_blobs += 1
        Blob2D.all_blobs[self.id] = self
        Blob2D.leaves[self.id] = self
        
        # compute and set blob center
        self.center = self.compute_center()
        
        # set default associated 3D blob (-1: associated to nothing,
        # O is "we don't know about the association")
        self.associated_3D_blob = None
        self.potentialy_associated = []
        self.association_probas = None
        self.regions_probas = None
        
        # tag to avoid displaying the same blob several times
        self.already_displayed = False
        
        #
        self.is_sub_blob = sub_blob
        self.is_meta_blob = meta_blob
    
    def compute_center(self):
        """Computes current blob center.

        This is available as a method and not as an attribute because
        the current blob's vertices are likely to change through the
        steps of the algorithm.
        
        """
        return np.mean(self.vertices, 0)
    
    def change_to_leaf(self):
        """Turns the current blob to a leaf.

        A blob which was a node can become a leaf.
        This can happen when we remove the last of its children.
        
        """
        if self.is_node():
            del  Blob2D.nodes[self.id]
        if not self.is_leaf():
            Blob2D.leaves[self.id] = self
    
    def change_to_node(self):
        """Turns the current blob to a node.

        A blob which was a leaf can become a node.
        This can happen when we add it a child.
        
        """
        if self.is_leaf():
            del  Blob2D.leaves[self.id]
        if not self.is_node():
            Blob2D.nodes[self.id] = self
    
    def associate_3Dblob(self, blob3D):
        """Associates the current blob to a 3D blob.

        2D blobs are matched to 3D blobs with a certain probability.
        When the link probability given two blobs is high enough
        (i.e. above a given threshold), we don't have any doubt that
        the two blobs are related, and that they have the same
        functional meaning (note that 2D blobs are thought to be more
        accurate since they suit the subject's anatomy).  So, when we
        don't have any doubt about the link, we can associate the two
        blobs for sure. That is what this method does. The certainty
        threshold is assumed to be more that 50% (if not, how could we
        be sure of the relationship between the two blobs ?), so that
        a 2D blob can only be associated to a unique 3D blob.

        """
        self.associated_3D_blob = blob3D
        
    def associate_potential(self, blob3D):
        """Adds a potentialy associated 3D blob the the current 2D one.

        2D blobs are matched to 3D blobs with a certain probability.
        When the link probability given two blobs is not high enough
        (i.e. under a given threshold), we have a doubt that the two
        blobs are related, and that they have the same functional
        meaning. But it can be useful to retain which 3D blobs were
        thought to be related to the current 2D blobs in case we can
        get more information to complete our matching.

        """
        if (blob3D not in self.potentialy_associated):
            self.potentialy_associated.append(blob3D)
            
    def set_association_probas(self, probas):
        '''
        '''
        self.association_probas = probas

    def is_leaf(self):
        """Tests whether or not the current blob is a leaf.

        I don't like the way it's done.

        """
        if self.id in Blob2D.leaves.keys():
            res = True
        else:
            res = False
            
        return res
    
    def is_node(self):
        """Tests whether or not the current blob is a node.

        I don't like the way it's done.

        """
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
        """Merges one of its children into the current blob.

        When two end-blobs are part of the same functional region, and
        when they furthermore are brothers (i.e. they have the same
        parent blob in the hierarchy), they should be merged in their
        parent blob which better correspond to the functional reality.
        The consequences of a merging is that the merged blob is no
        more held as a leaf and that the parent blob can become one.
        Thanks to the implementation of remove_child function, we
        don't have to carry about what happen to the parent blob.
 
        """
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
            ax.text(self.xpos, self.ypos,
                    '%d(%d-%g)'%(self.id,
                              self.vertices_id[self.get_argmax_activation()],
                              self.activation.mean()),
                    horizontalalignment='center', verticalalignment='center',
                    color=display_color, fontsize=11)
            ax.add_patch(p)
        elif ((not self.already_displayed) and self.is_node()):
            self.xpos = self.get_xpos()
            self.ypos = self.get_ypos()
            Blob2D.display_id += 1.
            p = mpatches.Circle((self.xpos,self.ypos),
                                Blob2D.radius, color=circle_color)
            ax.text(self.xpos, self.ypos,
                    '%d(%d-%g)'%(self.id,
                              self.vertices_id[self.get_argmax_activation()],
                              self.activation.mean()),
                    horizontalalignment='center', verticalalignment='center',
                    color=text_color, fontsize=11)
            ax.add_patch(p)
        self.already_displayed = True

