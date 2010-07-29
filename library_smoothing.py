"""
This is a corpus of functions used in brain surface smoothing context.

Author: Virgile Fritsch, 2010

"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import mesh_processing as mep

# -----------------------------------------------------------
# --------- Functions for validating ------------------------
# -----------------------------------------------------------

def compare_theorical_practical(input_mesh_graph, activation_data,
                                smoothed_activation_data, vertex, FWHM,
                                figure_id=1):
    """
    This function compares the results of the diffusion smoothing to a
    gaussian smoothing that would theorically be obtained with the same
    FWHM.
    The comparison procedure is only suitable for a map with only one
    activated vertex.
    
    Parameters
    ----------
    input_mesh_graph: ?
                      the mesh the activation data map corresponds to,
                      converted to a graph. Useful to effectively getting
                      the edges of the mesh.
    activation_data: array (shape=(time_samples, nb_vertices))
                     original activation data map, before smoothing.
    smoothed_activation_data: array (shape=(time_samples, nb_vertices))
                              activation data map, after diffusion smoothing.
    vertex: int,
            the only activated vertex of the data map.
    FWHM: float,
          FWHM of the gaussian smoothing corresponding to the diffusion
          smoothing that has been applied to original data.
    figure_id: int, optional,
               id of the first plot windows that will be displayed on screen.
               As the function releases three plot windows, their ids will
               vary from figure_id to figure_id+2.
    
    Procedure
    ---------
    The function computes the distance form the only activated vertex
    to each other vertex on the mesh. The distances are used to compute
    the activation value of each vertex of the mesh that would be
    obtained with a gaussian smoothing.
    Theorical and practical activation rates are ploted on a graph (after
    being sorted by distance). Note that only the vertices with an
    activation rate greater than a given threshold are ploted.
    The difference between theorical and practical activaiton rates is
    also ploted.
    
    """    
    DEBUG_MODE = False
    
    edges = input_mesh_graph.get_edges()
    # compute all distances from our vertex to another
    distances = input_mesh_graph.dijkstra(vertex)
    if DEBUG_MODE:
        print "distances : ", distances[distances < 10]

    # get vertex neighbors, sorted by growing distance from it
    vertex_neighbors = \
        np.sort(np.concatenate(([vertex], edges[edges[:,0]==vertex, 1]), 1))
    
    # compute theorical activation rates after smoothing
    sigma  = FWHM/(2 * np.sqrt(2 * np.log(2)))
    activation_rate = activation_data[0,vertex] * \
        np.exp(-distances**2/(2*sigma**2)) / ((sigma**2)*(2*np.pi))
    activation_rate /= activation_rate.sum()
    theorical_smoothing = activation_rate[vertex_neighbors]
    argsort_distances = np.argsort(distances[vertex_neighbors])
    theorical_smoothing = theorical_smoothing[argsort_distances]
    
    # get practical smoothing
    practical_smoothing = smoothed_activation_data[0,vertex_neighbors]
    practical_smoothing = practical_smoothing[argsort_distances]
    
    # graphically compare theorical and practical diffusion
    # for immediate neighbors
    x = distances[vertex_neighbors][argsort_distances]
    plt.figure(figure_id)
    plt.plot(x, theorical_smoothing, linewidth=1.0)
    # corrected distances (from Fischl)
    plt.plot(((1. + np.sqrt(2.))/2.) * x, practical_smoothing, linewidth=1.0)
    plt.legend(('theorical smoothing', 'practical smoothing'),
               'upper center', shadow=True)
    leg = plt.gca().get_legend()
    leg.draw_frame(False)
    plt.title('Theorical vs practical smoothing around vertex %(vertex)d' \
               % {'vertex': vertex})
    
    # graphically compare theorical and practical diffusion (thresholded)
    threshold_mask = activation_rate > 0.001
    theorical_smoothing_thresholded = activation_rate[threshold_mask]
    argsort_distances = np.argsort(distances[threshold_mask])
    theorical_smoothing_thresholded = \
                            theorical_smoothing_thresholded[argsort_distances]
    practical_smoothing_thresholded = smoothed_activation_data[0,
                                                               threshold_mask]
    practical_smoothing_thresholded = \
                            practical_smoothing_thresholded[argsort_distances]
    
    x = distances[threshold_mask][argsort_distances]
    plt.figure(figure_id + 1)
    plt.plot(x, theorical_smoothing_thresholded, linewidth=1.0)
    plt.plot(((1. + np.sqrt(2.))/2.) * x, practical_smoothing_thresholded,
             linewidth=1.0)
    plt.legend(('theorical smoothing', 'practical smoothing'),
               'upper center', shadow=True)
    leg = plt.gca().get_legend()
    leg.draw_frame(False)
    plt.title('Theorical vs practical smoothing around vertex %(vertex)d ' \
              'with threshold' % {'vertex': vertex})

    # difference between theorical and practical diffusion
    plt.figure(figure_id + 2)
    diff = practical_smoothing_thresholded - theorical_smoothing_thresholded
    #plt.plot(x, diff, linewidth=1.0, color='red')
    plt.title('Difference between theorical and practical smoothing around ' \
              'vertex %(vertex)d' % {'vertex': vertex})

    mean_error = diff.sum() / diff.size
    var_error = ((diff - mean_error)**2).sum() / diff.size
    rmse = np.sqrt((diff**2).sum())
    
    return mean_error, var_error, rmse


# -----------------------------------------------------------
# --------- Functions for debuging --------------------------
# -----------------------------------------------------------

def generate_texture_unique_activation(nb_vertices, vertex, intensity):
    """
    This function creates a cortical activation data map (texture) where
    only one vertex is activated.
    The generated texture has only one time sample.

    Parameters
    ----------
    nb_vertices: int,
                 number of vertices per time sample of the new
                 activation data map
    vertex: int,
            the only activated vertex
    intensity: intensity of the only activated vertex
    
    """
    # create a new empty texture
    new_activation_data = np.zeros((1, nb_vertices))
    # create an activation for one vertex
    new_activation_data[0,vertex] = intensity

    return new_activation_data

def generate_random_texture(time_samples, nb_vertices, max_intensity):
    """
    This function creates a random cortical activation data map (texture),
    based on the shape of a preexisting one.
    Shape of the returned activation map: (time_samples, nb_vertices)

    Parameters
    ----------
    time_samples: int,
                  number of time samples for the new activation data map
    nb_vertices: int,
                 number of vertices per time sample of the new
                 activation data map
    max_intensity: maximum abs(intensity) of a vertex
    
    """
    # create a new random texture
    new_activation_data = max_intensity * \
            (2 * np.random.rand(time_samples, nb_vertices) - 1)

    return new_activation_data


def display_local_mesh(edges, vertices, vertex, neighbor1=-1, neighbor2=-1):
    """
    This function display a chosen vertex and its close neighbors,
    thus reconstructing a local mesh.

    Parameters
    ----------
    edges: array (shape=(nb_edges, 3), type=int),
           contains all the edges forming the mesh as an array of
           two ends (two vertex ids).
           Note that the edges have to be sorted on their first end.
    vertices: array (shape=(nb_vertices, 3), type=float),
              contains all the vertices forming the mesh grid as
              arrays of three coordinates in the R3 space
    vertex: int,
            the vertex at the which the local mesh is centered
    neighbor1: int, optional
               one of the vertex neighbors we want to localize
    neighbor2: int, optional
               one of the vertex neighbors we want to localize

    """
    # initializations first...
    fig = plt.figure()
    ax = Axes3D(fig)
    colors_cycle = np.tile(["blue", "green", "red",
                            "cyan", "yellow", "black"], 2)
    # get vertex neighbors
    vertex_neighbors = edges[edges[:,0]==vertex, 1]
    vertices_to_plot = vertices[np.concatenate(([vertex], vertex_neighbors), 1)]
    # only print the edges involving the chosen vertex
    for pi in range(1, vertices_to_plot.shape[0]):
        pi_neighbors = edges[edges[:,0]==vertex_neighbors[pi-1], 1]
        pi_neighbors_plot = vertices[np.concatenate(([vertex_neighbors[pi-1]],
                                                     pi_neighbors), 1)]
        # plot all edges involving the current neighbor too
        # (for a better view of the local mesh)
        for pj in range(1, pi_neighbors_plot.shape[0]):
            if pi_neighbors[pj-1] != vertex:
                ax.plot3D(pi_neighbors_plot[[0,pj], 0],
                          pi_neighbors_plot[[0,pj], 1],
                          pi_neighbors_plot[[0,pj], 2],
                          color=colors_cycle[pi-1], linestyle="solid")
        if vertex_neighbors[pi-1] == neighbor1:  
            ax.plot3D(vertices_to_plot[[0,pi], 0],
                      vertices_to_plot[[0,pi], 1],
                      vertices_to_plot[[0,pi], 2],
                      color="magenta", linestyle="dashed")
        elif vertex_neighbors[pi-1] == neighbor2:  
            ax.plot3D(vertices_to_plot[[0,pi], 0],
                      vertices_to_plot[[0,pi], 1],
                      vertices_to_plot[[0,pi], 2],
                      color="magenta", linestyle="dashdot")
        else:
            ax.plot3D(vertices_to_plot[[0,pi], 0],
                      vertices_to_plot[[0,pi], 1],
                      vertices_to_plot[[0,pi], 2],
                      color="magenta", linestyle="solid")

    return



# -----------------------------------------------------------
# --------- Main functions ----------------------------------
# -----------------------------------------------------------

def compute_areas_and_cotangentes(polygons, vertices):
    """
    This function computes areas and angles cotangente values of
    the triangles (refered as "polygons") drawing a triangular mesh
    
    Parameters
    ----------
    polygons: array (shape=(nb_polygons, 3), type=int),
              contains all the triangles forming the mesh as
              arrays of three points (referenced by their indices)
    vertices: array (shape=(nb_vertices, 3), type=float),
              contains all the vertices forming the mesh grid as
              arrays of three coordinates in the R3 space

    """
    DEBUG_MODE = False
    assert (polygons.shape[0] > 0 and polygons.shape[1] == 3),\
           'The shape of the <polygons> parameter must be (nb_polygons, 3)'
    assert (vertices.shape[0] > 0 and vertices.shape[1] == 3),\
           'The shape of the <vertices> parameter must be (nb_vertices, 3)'
        
    # some initializations...
    nb_vertices = vertices.shape[0]
    nb_polygons = polygons.shape[0]
    # get the polygons edges as vectors
    permutation_aux = np.array([(0,0,1), (1,0,0), (0,1,0)], dtype=int)
    v_from_polygons = vertices[polygons] - \
                      vertices[np.dot(polygons, permutation_aux)]
    vAB_from_polygons = v_from_polygons[:,0] * -1.
    vAC_from_polygons = v_from_polygons[:,2]
    vCB_from_polygons = v_from_polygons[:,1]

    ### Compute polygons areas
    # compute the polygons areas using cross product (A = 0.5*||AB /\ AC||)
    vAB_vAC_cross_product = np.cross(vAB_from_polygons, vAC_from_polygons)
    polygons_double_areas = np.sqrt(np.sum(vAB_vAC_cross_product**2, 1))
    polygons_areas = polygons_double_areas * 0.5

    ### Compute polygons angles cotangente
    # compute the dot product of each polygons couple of edges
    edges_dot_product_of_polygons = np.zeros((3, nb_polygons))
    edges_dot_product_of_polygons[0] = \
                            np.sum(vAB_from_polygons * vAC_from_polygons, 1)
    edges_dot_product_of_polygons[1] = \
                            np.sum(vCB_from_polygons * vAB_from_polygons, 1)
    edges_dot_product_of_polygons[2] = \
                      -1. * np.sum(vAC_from_polygons * vCB_from_polygons, 1)
    if DEBUG_MODE:
        # compute edges lenghts :
        norm_AB = np.sqrt(np.sum(vAB_from_polygons**2, 1))
        norm_AC = np.sqrt(np.sum(vAC_from_polygons**2, 1))
        norm_CB = np.sqrt(np.sum(vCB_from_polygons**2, 1))
        # compute angles cosinus values
        cosinus = np.zeros((3, nb_focus_polygons))
        cosinus[0] = edges_dot_product_of_polygons[0] / (norm_AB * norm_AC)
        cosinus[1] = edges_dot_product_of_polygons[1] / (norm_CB * norm_AB)
        cosinus[2] = edges_dot_product_of_polygons[2] / (norm_AC * norm_CB)
        # compute angles sinus values
        sinus = np.zeros((3, nb_focus_polygons))
        sinus[0] = polygons_double_areas / (norm_AB * norm_AC)
        sinus[1] = polygons_double_areas / (norm_CB * norm_AB)
        sinus[2] = polygons_double_areas / (norm_AC * norm_CB)
    # compute angles cotangente values as cos/sin
    # (note that sinus is never equal to zero in a triangular mesh)
    polygons_cotan = edges_dot_product_of_polygons / polygons_double_areas

    return polygons_areas, polygons_cotan


def compute_weights_matrix(polygons, vertices, edges):
    """
    This function computes the weights matrix from the structure of
    the mesh (polygons) and the coordinates of the vertices forming
    the mesh's grid (vertices)

    Parameters
    ----------
    polygons: array (shape=(nb_polyons, 3), type=int),
              contains all the triangles forming the mesh as arrays of
              three points (referenced by their indices)
    vertices: array (shape=(nb_vertices, 3), type=float),
              contains all the vertices forming the mesh grid as
              arrays of three coordinates in the R3 space
    edges: array (shape=(nb_edges, 2), type=int),
           contains all the edges forming the mesh as an array of
           two ends (two vertex ids).
           Note that the edges have to be sorted on their first end.

    """
    DEBUG_MODE = False
    assert (polygons.shape[0] > 0 and polygons.shape[1] == 3), \
           'The shape of the <polygons> parameter must be (nb_polygons, 3)'
    assert (vertices.shape[0] > 0 and vertices.shape[1] == 3), \
           'The shape of the <vertices> parameter must be (nb_vertices, 3)'
    assert (edges.shape[0] > 0 and edges.shape[1] == 2), \
           'The shape of the <edges> parameter must be (nb_edges, 2)'
    assert (edges.shape[0] > polygons.shape[0]), \
           'There can\'t be more polygons than edges'
    # some initializations...
    nb_vertices = vertices.shape[0]
    nb_polygons = polygons.shape[0]
    
    ### Compute polygons ares and angles cotangente values
    polygons_areas, polygons_cotan = \
                    compute_areas_and_cotangentes(polygons, vertices)
    if DEBUG_MODE:
        polygons_areas[polygons_areas < 1e-6] = 1e-6
        print polygons_areas[polygons_areas < 0.01]
        print polygons[np.where(polygons_areas < 0.01)], \
          vertices[polygons[np.where(polygons_areas < 0.01)]]

    #-----------------------------------------------------------
    # Now that we have the polygons areas and the cotangente 
    # values of their angles, we need to use it in order to
    # compute the weights associated with each vertex of the mesh.
    # Let us define some useful indices first...
    #-----------------------------------------------------------

    ### Compute useful indices to make us able to find the right 
    ### theta/phi angles associated to a given edge

    # compute where to find the cotangente value of the theta angle
    # associated with triangle edges
    # --> store it using sparse matrix structures (COO)
    # first ends of edges
    cotan_theta_location_row = np.ravel(polygons)
    # corresponding second ends of edges
    permutation_aux = np.array([(0,0,1),(1,0,0),(0,1,0)], dtype=int)
    cotan_theta_location_col = np.ravel(np.dot(polygons, permutation_aux))
    # associate angle numbers in the triangle containing the theta
    # corresponding to the edges (see definition of cotan_triangle_id_data)
    cotan_theta_angle_id_data = np.tile([2,0,1], nb_polygons)

    # compute where to find the cotangente value of the phi angle
    # associated with triangle edges
    # --> store it using sparse matrix structures (COO)
    # first ends of edges
    permutation_aux = np.array([(1,0,0),(0,0,1),(0,1,0)], dtype=int)
    cotan_phi_location_row = np.ravel(np.dot(polygons, permutation_aux))
    # corresponding second ends of edges
    permutation_aux = np.array([(0,0,1),(0,1,0),(1,0,0)], dtype=int)
    cotan_phi_location_col = np.ravel(np.dot(polygons, permutation_aux))
    # associate angle numbers in the triangle containing the phi
    # corresponding to the edges (see definition of cotan_triangle_id_data)
    # /!\ todo : we may not need to compute this because we guess the 
    #            matrix storing the phi indices can be obtained from
    #            the matrix storing the theta indices
    cotan_phi_angle_id_data = np.tile([1,0,2], nb_polygons)

    # associate a triangle number to each pair of vertices forming
    # an edge depending on where to find the corresponding theta/phi
    # angle
    cotan_triangle_id_data = np.repeat(np.arange(nb_polygons), 3)

    # store the previously computed locations in sparse matrices
    cotan_theta_triangle_id = \
        sp.sparse.coo_matrix((cotan_triangle_id_data,
                        (cotan_theta_location_row,cotan_theta_location_col)),
                        shape=(nb_vertices, nb_vertices)).tolil()
    cotan_phi_triangle_id = \
        sp.sparse.coo_matrix((cotan_triangle_id_data,
                             (cotan_phi_location_row,cotan_phi_location_col)),
                             shape=(nb_vertices, nb_vertices)).tolil()
    cotan_theta_angle_id = \
        sp.sparse.coo_matrix((cotan_theta_angle_id_data,
                        (cotan_theta_location_row,cotan_theta_location_col)),
                        shape=(nb_vertices, nb_vertices)).tolil()
    cotan_phi_angle_id = \
        sp.sparse.coo_matrix((cotan_phi_angle_id_data,
                             (cotan_phi_location_row,cotan_phi_location_col)),
                             shape=(nb_vertices, nb_vertices)).tolil()

    #-----------------------------------------------------------
    # We are now able to efficiently compute the weights 
    # associated with each vertex of the mesh :
    #-----------------------------------------------------------

    ### Compute the weight associated with each vertex and then 
    ### construct a sparse matrix in COO format
    
    # Lets' deal with neighboring triangles areas first
    # (that's the difficult part...).
    # associate a triangle with each edge and get his area
    pi_ends = edges[:,0]
    pj_ends = edges[:,1]
    neighboring_triangles = cotan_theta_triangle_id[pi_ends, pj_ends]
    neighboring_triangles_area = \
            polygons_areas[np.ravel(neighboring_triangles.toarray())]
    # store the results in a sparse matrix
    areas_matrix = \
        sp.sparse.coo_matrix((neighboring_triangles_area, (pi_ends, pj_ends)),
                             shape=(nb_vertices,nb_vertices)).tocsr()
    # construct an array used to later divide the cotangentes sum
    aux_repeat = np.ravel((areas_matrix / areas_matrix).sum(1)).astype(int)
    areas_sums = np.ravel(areas_matrix.sum(1))
    areas_sums = np.repeat(areas_sums, aux_repeat)
    
    # Now deal with cotangentes
    # use the previously constructed indices to find out the right values
    neighbors_theta_angles_index = \
            np.ravel(cotan_theta_angle_id[pi_ends, pj_ends].toarray())
    neighbors_theta_triangles_index = \
            np.ravel(cotan_theta_triangle_id[pi_ends, pj_ends].toarray())
    neighbors_theta_angles = polygons_cotan[neighbors_theta_angles_index,
                                            neighbors_theta_triangles_index]
    neighbors_phi_angles_index = \
            np.ravel(cotan_phi_angle_id[pi_ends, pj_ends].toarray())
    neighbors_phi_triangles_index = \
            np.ravel(cotan_phi_triangle_id[pi_ends, pj_ends].toarray())
    neighbors_phi_angles = polygons_cotan[neighbors_phi_angles_index,
                                          neighbors_phi_triangles_index]
    # cotan(theta) + cotan(phi)
    data = (neighbors_theta_angles + neighbors_phi_angles) / areas_sums

    # Finally construct the weights matrix
    weights_matrix = \
        sp.sparse.coo_matrix((data, (pi_ends, pj_ends)),
                             shape=(nb_vertices, nb_vertices)).tocsr()
    
    return weights_matrix


def compute_weights_matrix_biased(polygons, vertices, edges):
    """
    /!\ Warning: this function is for debug purpose only /!\
    
    This function computes the weights matrix from the structure of
    the mesh (polygons) and the coordinates of the vertices forming
    the mesh's grid (vertices)

    Parameters
    ----------
    polygons: array (shape=(nb_polyons, 3), type=int),
              contains all the triangles forming the mesh as arrays of
              three points (referenced by their indices)
    vertices: array (shape=(nb_vertices, 3), type=float),
              contains all the vertices forming the mesh grid as
              arrays of three coordinates in the R3 space
    edges: array (shape=(nb_edges, 2), type=int),
           contains all the edges forming the mesh as an array of
           two ends (two vertex ids).
           Note that the edges have to be sorted on their first end.

    """
    DEBUG_MODE = False
    assert (polygons.shape[0] > 0 and polygons.shape[1] == 3), \
           'The shape of the <polygons> parameter must be (nb_polygons, 3)'
    assert (vertices.shape[0] > 0 and vertices.shape[1] == 3), \
           'The shape of the <vertices> parameter must be (nb_vertices, 3)'
    assert (edges.shape[0] > 0 and edges.shape[1] == 2), \
           'The shape of the <edges> parameter must be (nb_edges, 2)'
    assert (edges.shape[0] > polygons.shape[0]), \
           'There can\'t be more polygons than edges'
    
    # some initializations...
    nb_vertices = vertices.shape[0]
    nb_polygons = polygons.shape[0]
    
    ### Compute polygons ares and angles cotangente values
    polygons_areas, polygons_cotan = \
                    compute_areas_and_cotangentes(polygons, vertices)
    polygons_areas[polygons_areas < 1e-6] = 1e-6
    if DEBUG_MODE:
        print polygons_areas[polygons_areas < 0.01]
        print polygons[np.where(polygons_areas < 0.01)], \
          vertices[polygons[np.where(polygons_areas < 0.01)]]

    #-----------------------------------------------------------
    # Now that we have the polygons areas and the cotangente 
    # values of their angles, we need to use it in order to
    # compute the weights associated with each vertex of the mesh.
    # Let us define some useful indices first...
    #-----------------------------------------------------------

    ### Compute useful indices to make us able to find the right 
    ### theta/phi angles associated to a given edge

    # compute where to find the cotangente value of the theta angle
    # associated with triangle edges
    # --> store it using sparse matrix structures (COO)
    # first ends of edges
    cotan_theta_location_row = np.ravel(polygons)
    # corresponding second ends of edges
    permutation_aux = np.array([(0,0,1),(1,0,0),(0,1,0)], dtype=int)
    cotan_theta_location_col = np.ravel(np.dot(polygons, permutation_aux))
    # associate angle numbers in the triangle containing the theta
    # corresponding to the edges (see definition of cotan_triangle_id_data)
    cotan_theta_angle_id_data = np.tile([2,0,1], nb_polygons)

    # compute where to find the cotangente value of the phi angle
    # associated with triangle edges
    # --> store it using sparse matrix structures (COO)
    # first ends of edges
    permutation_aux = np.array([(1,0,0),(0,0,1),(0,1,0)], dtype=int)
    cotan_phi_location_row = np.ravel(np.dot(polygons, permutation_aux))
    # corresponding second ends of edges
    permutation_aux = np.array([(0,0,1),(0,1,0),(1,0,0)], dtype=int)
    cotan_phi_location_col = np.ravel(np.dot(polygons, permutation_aux))
    # associate angle numbers in the triangle containing the phi
    # corresponding to the edges (see definition of cotan_triangle_id_data)
    # /!\ todo : we may not need to compute this because we guess the 
    #            matrix storing the phi indices can be obtained from
    #            the matrix storing the theta indices
    cotan_phi_angle_id_data = np.tile([1,0,2], nb_polygons)

    # associate a triangle number to each pair of vertices forming
    # an edge depending on where to find the corresponding theta/phi
    # angle
    cotan_triangle_id_data = np.repeat(range(0, nb_polygons),3)

    # store the previously computed locations in sparse matrices
    cotan_theta_triangle_id = \
        sp.sparse.coo_matrix((cotan_triangle_id_data,
                        (cotan_theta_location_row,cotan_theta_location_col)),
                        shape=(nb_vertices, nb_vertices)).tolil()
    cotan_phi_triangle_id = \
        sp.sparse.coo_matrix((cotan_triangle_id_data,
                             (cotan_phi_location_row,cotan_phi_location_col)),
                             shape=(nb_vertices, nb_vertices)).tolil()
    cotan_theta_angle_id = \
        sp.sparse.coo_matrix((cotan_theta_angle_id_data,
                        (cotan_theta_location_row,cotan_theta_location_col)),
                        shape=(nb_vertices, nb_vertices)).tolil()
    cotan_phi_angle_id = \
        sp.sparse.coo_matrix((cotan_phi_angle_id_data,
                             (cotan_phi_location_row,cotan_phi_location_col)),
                             shape=(nb_vertices, nb_vertices)).tolil()

    #-----------------------------------------------------------
    # We are now able to efficiently compute the weights 
    # associated with each vertex of the mesh :
    #-----------------------------------------------------------

    ### Compute the weight associated with each vertex and then 
    ### construct a sparse matrix in COO format
    
    # Lets' deal with neighboring triangles areas first
    # (that's the difficult part...).
    # associate a triangle with each edge and get his area
    pi_ends = edges[:,0]
    pj_ends = edges[:,1]
    neighboring_triangles = cotan_theta_triangle_id[pi_ends, pj_ends]
    neighboring_triangles_area = \
            polygons_areas[np.ravel(neighboring_triangles.toarray())]
    # store the results in a sparse matrix
    areas_matrix = \
        sp.sparse.coo_matrix((neighboring_triangles_area, (pi_ends, pj_ends)),
                             shape=(nb_vertices,nb_vertices)).tocsr()
    # construct an array used to later divide the cotangentes sum
    aux_repeat = np.ravel((areas_matrix / areas_matrix).sum(1)).astype(int)
    areas_sums = np.ravel(areas_matrix.sum(1))
    areas_sums = np.repeat(areas_sums, aux_repeat)
    
    # Now deal with cotangentes
    # use the previously constructed indices to find out the right values
    neighbors_theta_angles_index = \
            np.ravel(cotan_theta_angle_id[pi_ends, pj_ends].toarray())
    neighbors_theta_triangles_index = \
            np.ravel(cotan_theta_triangle_id[pi_ends, pj_ends].toarray())
    neighbors_theta_angles = polygons_cotan[neighbors_theta_angles_index,
                                            neighbors_theta_triangles_index]
    neighbors_phi_angles_index = \
            np.ravel(cotan_phi_angle_id[pi_ends, pj_ends].toarray())
    neighbors_phi_triangles_index = \
            np.ravel(cotan_phi_triangle_id[pi_ends, pj_ends].toarray())
    neighbors_phi_angles = polygons_cotan[neighbors_phi_angles_index,
                                          neighbors_phi_triangles_index]
    # cotan(theta) + cotan(phi)
    data = (neighbors_theta_angles + neighbors_phi_angles) / areas_sums

    # Finally construct the weights matrix
    # /!\ warning : this is where the matrix is biased
    data[data < 0] = 0
    weights_matrix_biased = \
        sp.sparse.coo_matrix((data, (pi_ends, pj_ends)),
                             shape=(nb_vertices, nb_vertices)).tocsr()
    
    return weights_matrix_biased


def define_LB_operator(weights_matrix):
    """
    This function defines the Laplace-Beltrami operator,
    once given the weights associated with the edges of the mesh

    Parameters
    ----------
    weights_matrix: sparse matrix in CSR format
                    (shape=(nb_vertices, nb_vertices)),
                    associating a weight to each edge of the input mesh

    """
    assert (weights_matrix.shape[0] == weights_matrix.shape[1]), \
           'The weights matrix must be a square one'
    
    nb_vertices = weights_matrix.shape[0]
    # construct the Laplace-Beltrami operator
    diag_weights = sp.sparse.coo_matrix((np.ravel(weights_matrix.sum(1)),
                        (np.arange(nb_vertices),np.arange(nb_vertices))),
                        shape=(nb_vertices, nb_vertices)).tocsr()
    LB_operator = weights_matrix - diag_weights

    return LB_operator


def diffusion_smoothing(activation_data, LB_operator, N, dt):
    """
    This function applies the Laplace-Beltrami operator (LB_operator)
    across the cortical surface texture contained in activation_data

    Parameters
    ----------
    activation_data: array (shape=(time_samples, nb_vertices))
                     activation data to be smoothed across the brain
                     surface
    LB_operator: sparse matrix in CSR or CRS format
                 (shape=(nb_vertices, nb_vertices)),
                 the Laplace-Beltrami operator as a matrix
    N: int,
       the number of iterations needed
    dt: float,
        time step of the diffusion process

    """
    nb_time_samples = activation_data.shape[0]
    smoothed_activation_data = activation_data.copy()
    for ts in np.arange(0, nb_time_samples):
        for i in np.arange(N):       
            smoothed_activation_data[ts,:] += \
                        dt * LB_operator * smoothed_activation_data[ts,:]

    return smoothed_activation_data

def diffusion_smoothing_step_by_step(activation_data, LB_operator, N, dt):
    """
    This function applies the Laplace-Beltrami operator (LB_operator)
    across the cortical surface texture contained in activation_data,
    where t-th time sample corresponds to the t-th iteration

    Parameters
    ----------
    activation_data: array (shape=(1, nb_vertices))
                     activation data to be smoothed across the brain
                     surface
    LB_operator: sparse matrix in CSR or CRS format
                 (shape=(nb_vertices, nb_vertices)),
                 the Laplace-Beltrami operator as a matrix
    N: int,
       the number of iterations needed
    dt: float,
        time step of the diffusion process

    """
    nb_time_samples = 1
    smoothed_activation_data = np.zeros((N+1,activation_data.shape[1]))
    smoothed_activation_data[0,:] = activation_data[0,:]
    for i in np.arange(N):
        smoothed_activation_data[i+1,:] = smoothed_activation_data[i,:] + \
                    dt * LB_operator * smoothed_activation_data[i,:]
    
    return smoothed_activation_data


def compute_smoothing_parameters(weights_matrix, FWHM, dt_max=0.):
    """
    This function computes the number of iterations and
    the accurate time step needed to smooth out activation data
    on the brain surface as if we were using a gaussian kernel 
    with a given FWHM.
    
    Parameters
    ----------
    weights_matrix: sparse matrix in CSR format
                    (shape=(nb_vertices, nb_vertices)),
                    associating a weight to each edge of the input mesh.
    FWHM: float,
          FHWM > 0,
          parameter of the gaussian kernel equivalent to the smoothing
          we want to perform.
    dt_max: float, optional,
            the maximum time step. Only use this if the automatically
            computed dt_max leads to divergence.
    
    Algorithm
    ---------
    A maximum time step (dt_max) is computed, such as we have the condition
    min_i F(pi,t_n) < F(p,t_n+1) < max_i F(pi,t_n) (see Chung & Taylor's paper)
    verified.
    Then, the corresponding number of iterations (N) is chosen to be an integer.
    Finally, dt is adjusted for the smoothing to be equivalent to a gaussian
    smoothing with the chosen FWHM (and dt <= dt_max).
    
    Note
    ----
    dt_max = min_i (1/sum(w_i)) is supposed to be a good criterion.
    Nevertheless, in theory, there can still be divergence with such
    a dt_max, depending on the activation data and the structure of
    the mesh.
    In the case where divergence is obvious, it is possible to force
    dt_max to be smaller than min_i (1/sum(w_i)), using dt_max
    parameter. One should try to reduce dt_max dividing it by two,
    then by four, and so on. If there still is divergence and that
    the computation time is too long, a refinement of the mesh is anavoidable
    in order to use Chung & Taylor's approach.
    
    See
    ---
    compute_weights_matrix
    
    """
    # compute maximal dt
    if dt_max == 0.:
        dt_max = 1/np.amax(np.ravel(weights_matrix.sum(1)))
    # compute the number of iterations N
    N = int(np.ceil((FWHM**2)/(16 * np.log(2) * dt_max)))
    # adjust dt now that we have N
    dt = (FWHM**2)/(16 * np.log(2) * N)

    return N, dt


