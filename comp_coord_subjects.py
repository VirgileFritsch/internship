import numpy as np
import sys

from nipy.neurospin.glm_files_layout import tio
from gifti import loadImage
import nipy.neurospin.graph.graph as fg

from database_archi import *

#--------------------------------------------------------
#subj = ['s12069', 's12300', 's12401', 's12431', 's12508', 's12532', 's12539', 's12562','s12590', 's12635', 's12636', 's12898', 's12081', 's12165', 's12207', 's12344', 's12352', 's12370', 's12381', 's12405', 's12414', 's12432']
subj = ['s12069', 's12081', 's12300']
subj2 = subj[:]
#--------------------------------------------------------
gamma = 10.
coord_type = "coord"

#--------------------------------------------------------
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

import pdb
def find_score(distances, current_val=0, local_min=np.infty, level=1, gamma=10):
    if distances.shape[0] > 1:
        current_min = local_min
        res = np.infty
        res_arg = []
        for i, d in enumerate(distances[0,:]):
            new_val = current_val + d
            if new_val < current_min:
                if i != distances.shape[1]-1:
                    sub_dist = np.hstack((distances[1:,:i], distances[1:,i+1:]))
                else:
                    sub_dist = distances[1:,:]
                tmp_res, tmp_arg = find_score(sub_dist, new_val, current_min, level+1)
                if tmp_res < current_min:
                    current_min = tmp_res
                    res = current_min
                    res_arg = np.concatenate(([i],tmp_arg))
            else:
                continue
    else:
        min_arg = np.argmin(distances[0,:])
        res = current_val + distances[0,min_arg]
        res_arg = [min_arg]

    return res, res_arg

def format_score(score):
    no_association = score.size
    numbers = range(score.size)
    for i in range(score.size):
        if score[i] >= len(numbers):
            score[i] = no_association
        else:
            swap = numbers[score[i]]
            del numbers[score[i]]
            score[i] = swap
    
    return score

#--------------------------------------------------------
print "\t"
for s in subj:
    print "\t%s" %s,
print
removed = 1
for s1 in subj:
    print "%s\t" %s1, "-\t"*removed,
    ### Read subject s1 meshes
    # left hemisphere
    left_mesh = loadImage("%s/%s/surf/%s" %(ROOT_PATH, s1, LMESH_GII))
    c, n, t = left_mesh.getArrays()
    s1_ltriangles = t.getData()
    s1_lvertices = c.getData()
    # right hemisphere
    right_mesh = loadImage("%s/%s/surf/%s" %(ROOT_PATH, s1, RMESH_GII))
    c, n, t = right_mesh.getArrays()
    s1_rtriangles = t.getData()
    s1_rvertices = c.getData()
    ### Read subject s1 coordinate textures (level 1)
    s1_lcoord_tex = "%s/%s/experiments/smoothed_FWHM%g/%s/results_%s_level001/left_%s_FWHM2D%g_smin2D%i_FWHM3D%g_smin3D%i.tex" %(ROOT_PATH, s1, FWHM, CONTRAST, coord_type, CONTRAST, FWHM, SMIN, FWHM3D, SMIN3D)
    s1_lcoord = tio.Texture(s1_lcoord_tex).read(s1_lcoord_tex).data
    s1_rcoord_tex = "%s/%s/experiments/smoothed_FWHM%g/%s/results_%s_level001/right_%s_FWHM2D%g_smin2D%i_FWHM3D%g_smin3D%i.tex" %(ROOT_PATH, s1, FWHM, CONTRAST, coord_type, CONTRAST, FWHM, SMIN, FWHM3D, SMIN3D)
    s1_rcoord = tio.Texture(s1_rcoord_tex).read(s1_rcoord_tex).data

    subj2.remove(s1)
    removed += 1
    for s2 in subj2:
        
        ### Read subject s2 meshes
        # left hemisphere
        left_mesh = loadImage("%s/%s/surf/%s" %(ROOT_PATH, s2, LMESH_GII))
        c, n, t = left_mesh.getArrays()
        s2_lvertices = c.getData()
        # right hemisphere
        right_mesh = loadImage("%s/%s/surf/%s" %(ROOT_PATH, s2, RMESH_GII))
        c, n, t = right_mesh.getArrays()
        s2_rvertices = c.getData()
        ### Read subject s1 coordinate textures (level 1)
        s2_lcoord_tex = "%s/%s/experiments/smoothed_FWHM%g/%s/results_%s_level001/left_%s_FWHM2D%g_smin2D%i_FWHM3D%g_smin3D%i.tex" %(ROOT_PATH, s2, FWHM, CONTRAST, coord_type, CONTRAST, FWHM, SMIN, FWHM3D, SMIN3D)
        s2_lcoord = tio.Texture(s2_lcoord_tex).read(s2_lcoord_tex).data
        s2_rcoord_tex = "%s/%s/experiments/smoothed_FWHM%g/%s/results_%s_level001/right_%s_FWHM2D%g_smin2D%i_FWHM3D%g_smin3D%i.tex" %(ROOT_PATH, s2, FWHM, CONTRAST, coord_type, CONTRAST, FWHM, SMIN, FWHM3D, SMIN3D)
        s2_rcoord = tio.Texture(s2_rcoord_tex).read(s2_rcoord_tex).data

        ### Compute mean meshes
        # left hemisphere
        mean_lvertices = np.hstack(
            (s1_lvertices, s2_lvertices)).reshape(
            (s1_lvertices.shape[0],2,3)).mean(1)
        mean_lmesh_graph = mesh_to_graph(mean_lvertices, s1_ltriangles)
        # right hemisphere
        mean_rvertices = np.hstack(
            (s1_rvertices, s2_rvertices)).reshape(
            (s1_rvertices.shape[0],2,3)).mean(1)
        mean_rmesh_graph = mesh_to_graph(mean_rvertices, s1_rtriangles)
        
        ### Compute distances
        s1_lpeaks = np.where(s1_lcoord != -1)[0]
        s2_lpeaks = np.where(s2_lcoord != -1)[0]
        if s1_lpeaks.size < s2_lpeaks.size:
            less_peaked = s1_lpeaks
            most_peaked = s2_lpeaks
        else:
            less_peaked = s2_lpeaks
            most_peaked = s1_lpeaks
        n = less_peaked.size
        p = most_peaked.size
        ldistances = np.zeros((n,p))
        for i, vertex_id in enumerate(less_peaked):
            ldistances[i,:] = mean_lmesh_graph.dijkstra(vertex_id)[most_peaked]
        # reorder matrix for a faster algorithm
        reorder_aux = np.zeros((n,n))
        reorder_aux[np.arange(0,n),np.argsort(np.amin(ldistances, 1))] = 1
        ldistances = np.dot(reorder_aux, ldistances)
        ldistances = np.hstack((ldistances, gamma*np.ones((n,1))))
        ldistances = np.vstack((ldistances, gamma*np.ones((p-n,p+1))))

        comp_score, trace_score = find_score(ldistances, 0, gamma*p)
        # take into account the fact we took submatrices to recover
        # right indices
        trace_score_formated = format_score(trace_score.copy())
        print "%.2f\t" %comp_score,
        #print comp_score, trace_score, trace_score_formated
        #print ldistances[np.arange(ldistances.shape[0]),trace_score_formated]
    print
    sys.stdout.flush()
        
