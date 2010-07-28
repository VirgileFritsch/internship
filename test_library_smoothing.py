"""
Collection of tests for the diffusion_smoothing library.

Author: Virgile Fritsch, 2010

"""

import numpy as np
import library_smoothing as smooth

# -----------------------------------------------------------
# --------- Test compute_areas_and_cotangentes --------------
# -----------------------------------------------------------

def test1_1():
    """
    This test function builds a triangle with the points
        - (0, 0, 0)
        - (0, 1, 0)
        - (1, 0, 0)
    and then has its area and cotangentes values computed by
    the smoothing library.
    Results should be :
        - area = 0.5
        - cotan(A) = 0, cotan(B) = 1, cotan(C) = 1
    
    """
    vertices = np.array([[0.,0.,0.], [0., 1., 0.], [1., 0., 0.]])
    polygons = np.array([[0, 1, 2]])
    areas, cotangentes = smooth.compute_areas_and_cotangentes(polygons, vertices)

    assert (areas.shape == (1,) and areas[0] == 0.5)
    assert (cotangentes.shape == (3,1) and \
            np.all(np.equal(cotangentes, np.array([[0.], [1.], [1.]]))))

def test1_2():
    """
    This test function builds two triangles, respectively with the points
        - (0, 0, 0), (0, 1, 0), (1, 0, 0)
        - (0, 1, 0), (1, 0, 0), (0, 0, 1)
    and then has their areas and cotangentes values computed by
    the smoothing library.
    Results should be :
        - areas = (0.5, sqrt(3)/2)
        - cotan(A) = (0, 1/sqrt(3)),
          cotan(B) = (1, 1/sqrt(3)),
          cotan(C) = (1, 1/sqrt(3))
    
    """
    vertices = np.array([[0.,0.,0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    polygons = np.array([[0, 1, 2], [1, 2, 3]])
    areas, cotangentes = smooth.compute_areas_and_cotangentes(polygons, vertices)

    assert (areas.shape == (2,))
    assert (np.all(np.equal(areas, np.array([0.5, np.sqrt(3)/2]))))
    assert (cotangentes.shape == (3,2))
    assert (np.all(np.equal(cotangentes, np.array([[0., 1/np.sqrt(3)],
                                                   [1., 1/np.sqrt(3)],
                                                   [1., 1/np.sqrt(3)]]))))



# -----------------------------------------------------------
# --------- Test compute_weights_matrix ---------------------
# -----------------------------------------------------------

def test2_1():
    """
    This functions builds a planar mesh and checks that the weights of the
    middle vertex are correct. Here is the mesh (with the non-zero weights):
        ________________
        |\      |      /|
        |  \   0.5   /  |
        |    \  |  /    |
        |_0.5__\|/__0.5_|
        |      /|\      |
        |    /  |  \    |
        |  /   0.5   \  |
        |/______|______\|

    
    """
    # create vertices
    x = np.array([0., 1., 2.])
    x = np.repeat(x, 3)
    y = np.array([0., 1., 2.])
    y = np.tile(y ,3)
    z = np.zeros(9)
    vertices = np.vstack((x, y, z)).T
    
    # create polygons
    polygons = np.array([[0, 3, 4], [0, 4, 1], [1, 4, 2], [2, 4, 5],
                         [3, 6, 4], [4, 6, 7], [4, 7, 8], [4, 8, 5]])
    
    # get edges
    nb_edges = 3 * polygons.shape[0]
    edges = np.zeros((nb_edges, 2))
    # get the polygons edges as tuples
    permutator = np.array([(0,0,1),(1,0,0),(0,1,0)], dtype=int)
    edges[:,0] = np.ravel(polygons)
    edges[:,1] = np.ravel(np.dot(polygons, permutator))
    ind = np.lexsort((edges[:,1], edges[:,0]))
    edges = edges[ind]

    weights_matrix = smooth.compute_weights_matrix(polygons, vertices, edges)

    assert (weights_matrix[4,0] == 0. and weights_matrix[4,2] == 0. and
            weights_matrix[4,6] == 0. and weights_matrix[4,8] == 0.)
    assert (weights_matrix[4,1] == 0.5 and weights_matrix[4,3] == 0.5 and
            weights_matrix[4,5] == 0.5 and weights_matrix[4,7] == 0.5)

def test2_2():
    """
    This functions builds a planar mesh and checks that the weights of the
    middle vertex are correct. Here is the mesh:
    
        \\
        |\ \ 
        | \  \
        |  \   \_________
        | -0.22 |      /|
        |    \ 0.88  /  |
        |     \ |  /    |
        |_0.66_\|/__0.44|
        |      /|\      |
        |    /  |  \    |
        |  /   0.44  \  |
        |/______|______\|

    
    """
    # create vertices
    x = np.array([0., 1., 2.])
    x = np.repeat(x, 3)
    y = np.array([0., 1., 2.])
    y = np.tile(y ,3)
    y[2] = 3.
    z = np.zeros(9)
    vertices = np.vstack((x, y, z)).T
    
    # create polygons
    polygons = np.array([[0, 3, 4], [0, 4, 1], [1, 4, 2], [2, 4, 5],
                         [3, 6, 4], [4, 6, 7], [4, 7, 8], [4, 8, 5]])
    
    # get edges
    nb_edges = 3 * polygons.shape[0]
    edges = np.zeros((nb_edges, 2))
    # get the polygons edges as tuples
    permutator = np.array([(0,0,1),(1,0,0),(0,1,0)], dtype=int)
    edges[:,0] = np.ravel(polygons)
    edges[:,1] = np.ravel(np.dot(polygons, permutator))
    ind = np.lexsort((edges[:,1], edges[:,0]))
    edges = edges[ind]

    weights_matrix = smooth.compute_weights_matrix(polygons, vertices, edges)

    assert (weights_matrix[4,0] == 0. and
            weights_matrix[4,2] == -0.22222222222222221 and
            weights_matrix[4,6] == 0. and weights_matrix[4,8] == 0.)
    assert (weights_matrix[4,1] == 0.66666666666666663 and
            weights_matrix[4,3] == 4./9. and
            weights_matrix[4,5] == 0.88888888888888884 and
            weights_matrix[4,7] == 4./9.)


# -----------------------------------------------------------
# --------- Test compute_weights_matrix_biased --------------
# -----------------------------------------------------------
# /!\ Warning: This function is for debug purpose only /!\

def test3_1():
    """
    This functions builds a planar mesh and checks that the weights of the
    middle vertex are correct. Here is the mesh (with the non-zero weights):
        ________________
        |\      |      /|
        |  \   0.5   /  |
        |    \  |  /    |
        |_0.5__\|/__0.5_|
        |      /|\      |
        |    /  |  \    |
        |  /   0.5   \  |
        |/______|______\|

    
    """
    # create vertices
    x = np.array([0., 1., 2.])
    x = np.repeat(x, 3)
    y = np.array([0., 1., 2.])
    y = np.tile(y ,3)
    z = np.zeros(9)
    vertices = np.vstack((x, y, z)).T
    
    # create polygons
    polygons = np.array([[0, 3, 4], [0, 4, 1], [1, 4, 2], [2, 4, 5],
                         [3, 6, 4], [4, 6, 7], [4, 7, 8], [4, 8, 5]])
    
    # get edges
    nb_edges = 3 * polygons.shape[0]
    edges = np.zeros((nb_edges, 2))
    # get the polygons edges as tuples
    permutator = np.array([(0,0,1),(1,0,0),(0,1,0)], dtype=int)
    edges[:,0] = np.ravel(polygons)
    edges[:,1] = np.ravel(np.dot(polygons, permutator))
    ind = np.lexsort((edges[:,1], edges[:,0]))
    edges = edges[ind]

    weights_matrix = smooth.compute_weights_matrix_biased(polygons,
                                                          vertices, edges)

    assert (weights_matrix[4,0] == 0. and weights_matrix[4,2] == 0. and
            weights_matrix[4,6] == 0. and weights_matrix[4,8] == 0.)
    assert (weights_matrix[4,1] == 0.5 and weights_matrix[4,3] == 0.5 and
            weights_matrix[4,5] == 0.5 and weights_matrix[4,7] == 0.5)


def test3_2():
    """
    This functions builds a planar mesh and checks that the weights of the
    middle vertex are correct. Here is the mesh:
    
        \\
        |\ \ 
        | \  \
        |  \   \_________
        |   0   |      /|
        |    \ 0.88  /  |
        |     \ |  /    |
        |_0.66_\|/_0.44_|
        |      /|\      |
        |    /  |  \    |
        |  /   0.44  \  |
        |/______|______\|

    
    """
    # create vertices
    x = np.array([0., 1., 2.])
    x = np.repeat(x, 3)
    y = np.array([0., 1., 2.])
    y = np.tile(y ,3)
    y[2] = 3.
    z = np.zeros(9)
    vertices = np.vstack((x, y, z)).T
    
    # create polygons
    polygons = np.array([[0, 3, 4], [0, 4, 1], [1, 4, 2], [2, 4, 5],
                         [3, 6, 4], [4, 6, 7], [4, 7, 8], [4, 8, 5]])
    
    # get edges
    nb_edges = 3 * polygons.shape[0]
    edges = np.zeros((nb_edges, 2))
    # get the polygons edges as tuples
    permutator = np.array([(0,0,1),(1,0,0),(0,1,0)], dtype=int)
    edges[:,0] = np.ravel(polygons)
    edges[:,1] = np.ravel(np.dot(polygons, permutator))
    ind = np.lexsort((edges[:,1], edges[:,0]))
    edges = edges[ind]

    weights_matrix = smooth.compute_weights_matrix_biased(polygons,
                                                          vertices, edges)

    assert (weights_matrix[4,0] == 0. and
            weights_matrix[4,2] == 0. and
            weights_matrix[4,6] == 0. and weights_matrix[4,8] == 0.)
    assert (weights_matrix[4,1] == 0.66666666666666663 and
            weights_matrix[4,3] == 4./9. and
            weights_matrix[4,5] == 0.88888888888888884 and
            weights_matrix[4,7] == 4./9.)


# -----------------------------------------------------------
# --------- Test define_LB_operator -------------------------
# -----------------------------------------------------------

def test4_1():
    """
    This functions builds a planar mesh and checks that the weights of the
    middle vertex is correct. Here is the mesh (with the non-zero weights):
        ________________
        |\      |      /|
        |  \   0.5   /  |
        |    \  |  /    |
        |_0.5__\|/__0.5_|   sum(wi) = 2
        |      /|\      |
        |    /  |  \    |
        |  /   0.5   \  |
        |/______|______\|
    
    
    It also checks that the Laplace-Beltrami operator is correctly defined
    """
    # create vertices
    x = np.array([0., 1., 2.])
    x = np.repeat(x, 3)
    y = np.array([0., 1., 2.])
    y = np.tile(y ,3)
    z = np.zeros(9)
    vertices = np.vstack((x, y, z)).T
    
    # create polygons
    polygons = np.array([[0, 3, 4], [0, 4, 1], [1, 4, 2], [2, 4, 5],
                         [3, 6, 4], [4, 6, 7], [4, 7, 8], [4, 8, 5]])
    
    # get edges
    nb_edges = 3 * polygons.shape[0]
    edges = np.zeros((nb_edges, 2))
    # get the polygons edges as tuples
    permutator = np.array([(0,0,1),(1,0,0),(0,1,0)], dtype=int)
    edges[:,0] = np.ravel(polygons)
    edges[:,1] = np.ravel(np.dot(polygons, permutator))
    ind = np.lexsort((edges[:,1], edges[:,0]))
    edges = edges[ind]

    weights_matrix = smooth.compute_weights_matrix(polygons, vertices, edges)
    LB_operator = smooth.define_LB_operator(weights_matrix)

    assert (LB_operator[4,0] == 0. and LB_operator[4,2] == 0. and
            LB_operator[4,6] == 0. and LB_operator[4,8] == 0.)
    assert (LB_operator[4,1] == 0.5 and LB_operator[4,3] == 0.5 and
            LB_operator[4,5] == 0.5 and LB_operator[4,7] == 0.5)
    assert (LB_operator[4,4] == -2.)



# -----------------------------------------------------------
# --------- Test compute_smoothing_parameters ---------------
# -----------------------------------------------------------

def test5_1():
    """
    This functions builds a planar mesh and compute the associated
    weights matrix:
        ________________
        |\      |      /|
        |  \   0.5   /  |
        |    \  |  /    |
        |_0.5__\|/__0.5_|
        |      /|\      |
        |    /  |  \    |
        |  /   0.5   \  |
        |/______|______\|
    
    
    Then, the smoothing parameters are computed according to the given FWHM.
    The function checks that the computed parameters are correct.
    
    """
    # create vertices
    x = np.array([0., 1., 2.])
    x = np.repeat(x, 3)
    y = np.array([0., 1., 2.])
    y = np.tile(y ,3)
    z = np.zeros(9)
    vertices = np.vstack((x, y, z)).T
    
    # create polygons
    polygons = np.array([[0, 3, 4], [0, 4, 1], [1, 4, 2], [2, 4, 5],
                         [3, 6, 4], [4, 6, 7], [4, 7, 8], [4, 8, 5]])
    
    # get edges
    nb_edges = 3 * polygons.shape[0]
    edges = np.zeros((nb_edges, 2))
    # get the polygons edges as tuples
    permutator = np.array([(0,0,1),(1,0,0),(0,1,0)], dtype=int)
    edges[:,0] = np.ravel(polygons)
    edges[:,1] = np.ravel(np.dot(polygons, permutator))
    ind = np.lexsort((edges[:,1], edges[:,0]))
    edges = edges[ind]

    weights_matrix = smooth.compute_weights_matrix(polygons, vertices, edges)
    dt_max = 1/np.amax(np.ravel(weights_matrix.sum(1)))
    dt_max /= 2
    for FWHM in np.arange(1, 20):
        N, dt = smooth.compute_smoothing_parameters(weights_matrix, FWHM)
        assert (FWHM == 4 * np.sqrt(np.log(2) * N * dt))
        N, dt = smooth.compute_smoothing_parameters(weights_matrix,
                                                    FWHM, dt_max)
        assert (FWHM == 4 * np.sqrt(np.log(2) * N * dt))



if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])

