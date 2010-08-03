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
def find_score(distances, current_val=0, local_min=np.infty, gamma=10.):
    if distances.shape[0] > 1:
        current_min = local_min
        res = np.infty
        res_arg = np.array([])
        for i, d in enumerate(distances[0,:]):
            if d > gamma:
                continue
            new_val = current_val + d
            if new_val < current_min:
                if i != distances.shape[1]-1:
                    sub_dist = np.hstack((distances[1:,:i], distances[1:,i+1:]))
                else:
                    sub_dist = distances[1:,:]
                tmp_res, tmp_arg = find_score(sub_dist, new_val,
                                              current_min, gamma)
                if tmp_res < current_min:
                    current_min = tmp_res
                    res = current_min
                    res_arg = np.concatenate(([i],tmp_arg))
            else:
                continue
    else:
        min_arg = np.argmin(distances[0,:])
        res = current_val + distances[0,min_arg]
        res_arg = np.array([min_arg])

    return res, res_arg

def format_score(score):
    no_association = score.size-1
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
all_lalone = {}
all_ralone = {}
all_llinked = {}
all_rlinked = {}
for s in subj:
    all_lalone[s] = []
    all_ralone[s] = []
    all_llinked[s] = []
    all_rlinked[s] = []
removed = 1
print 'Comparison %s -- gamma=%g' %(coord_type, gamma)
for s1_id, s1 in enumerate(subj):
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

    s1_lpeaks = np.where(s1_lcoord != -1)[0]
    s1_rpeaks = np.where(s1_rcoord != -1)[0]

    subj2.remove(s1)
    removed += 1
    for s2_id, s2 in enumerate(subj2):
        print "-- Subject %s vs subject %s" %(s1, s2)
        
        ### Read subject s2 meshes
        # left hemisphere
        left_mesh = loadImage("%s/%s/surf/%s" %(ROOT_PATH, s2, LMESH_GII))
        c, n, t = left_mesh.getArrays()
        s2_lvertices = c.getData()
        # right hemisphere
        right_mesh = loadImage("%s/%s/surf/%s" %(ROOT_PATH, s2, RMESH_GII))
        c, n, t = right_mesh.getArrays()
        s2_rvertices = c.getData()
        ### Read subject s2 coordinate textures (level 1)
        s2_lcoord_tex = "%s/%s/experiments/smoothed_FWHM%g/%s/results_%s_level001/left_%s_FWHM2D%g_smin2D%i_FWHM3D%g_smin3D%i.tex" %(ROOT_PATH, s2, FWHM, CONTRAST, coord_type, CONTRAST, FWHM, SMIN, FWHM3D, SMIN3D)
        s2_lcoord = tio.Texture(s2_lcoord_tex).read(s2_lcoord_tex).data
        s2_rcoord_tex = "%s/%s/experiments/smoothed_FWHM%g/%s/results_%s_level001/right_%s_FWHM2D%g_smin2D%i_FWHM3D%g_smin3D%i.tex" %(ROOT_PATH, s2, FWHM, CONTRAST, coord_type, CONTRAST, FWHM, SMIN, FWHM3D, SMIN3D)
        s2_rcoord = tio.Texture(s2_rcoord_tex).read(s2_rcoord_tex).data

        ### ---------------
        ### Process left hemisphere
        ### ---------------
        no_association = False
        ### Compute mean meshes
        mean_lvertices = np.hstack(
            (s1_lvertices, s2_lvertices)).reshape(
            (s1_lvertices.shape[0],2,3)).mean(1)
        mean_lmesh_graph = mesh_to_graph(mean_lvertices, s1_ltriangles)

        ### Compute distances
        s2_lpeaks = np.where(s2_lcoord != -1)[0]
        if s1_lpeaks.size < s2_lpeaks.size:
            less_peaked = s1_lpeaks
            less_peaked_id = s1
            most_peaked = s2_lpeaks
            most_peaked_id = s2
        else:
            less_peaked = s2_lpeaks
            less_peaked_id = s2
            most_peaked = s1_lpeaks
            most_peaked_id = s1
        n = less_peaked.size
        p = most_peaked.size
        ldistances = np.zeros((n,p))
        for i, vertex_id in enumerate(less_peaked):
            ldistances[i,:] = mean_lmesh_graph.dijkstra(vertex_id)[most_peaked]
        # scale distances
        ldistances = ((1. + np.sqrt(2.))/2.) * ldistances
        if np.amin(ldistances) >= gamma:
            comp_score_l = (gamma / 2.) * (n + p)
            all_lalone[less_peaked_id].append(less_peaked)
            all_lalone[most_peaked_id].append(most_peaked)
            no_association = True
        else:
            ### Reorder matrix for a faster algorithm
            # remove rows >= gamma
            n_lalone = less_peaked[np.where(np.amin(ldistances, 1) >= gamma)[0]]
            n_lmaybe_linked = less_peaked[np.where(np.amin(ldistances,1)<gamma)[0]]
            ldistances = ldistances[np.amin(ldistances, 1) < gamma,:]
            # remove columns >= gamma
            p_lalone = most_peaked[np.where(np.amin(ldistances, 0) >= gamma)[0]]
            p_lmaybe_linked = most_peaked[np.where(np.amin(ldistances,0)<gamma)[0]]
            ldistances = ldistances[:,np.amin(ldistances, 0) < gamma]
            # reorder the matrix
            m = ldistances.shape[0]
            q = ldistances.shape[1]
            reorder = np.zeros((m,m))
            reorder_aux = np.argsort(np.amin(ldistances, 1))
            reorder[np.arange(0,m),reorder_aux] = 1
            ldistances = np.dot(reorder, ldistances)
            ldistances = np.hstack((ldistances, gamma*np.ones((m,1))))
            
            ### Finally find best matches
            comp_score_l, trace_score = find_score(ldistances, 0, gamma*p, gamma)
            comp_score_l += (gamma/2.)*(n_lalone.size + p_lalone.size)
            # take into account the fact we took submatrices to recover
            # right indices
            trace_score_formated = format_score(trace_score.copy())
            match_distances = \
                    ldistances[np.arange(ldistances.shape[0]),trace_score_formated]
            # update s1 dict
            new_n_lalone = n_lmaybe_linked[match_distances == gamma]
            n_lalone = np.concatenate((n_lalone, new_n_lalone))
            n_llinked = n_lmaybe_linked[match_distances != gamma]
            all_lalone[less_peaked_id].append(n_lalone)
            all_llinked[less_peaked_id].append(n_llinked)
            # update s2 dict
            mask_p_llinked = trace_score_formated[match_distances == gamma]
            new_p_lalone = \
                    p_lmaybe_linked[mask_p_llinked != ldistances.shape[1]]
            p_lalone = np.concatenate((p_lalone, new_p_lalone))
            p_llinked = \
                    p_lmaybe_linked[trace_score_formated[[match_distances != gamma]]]
            all_lalone[most_peaked_id].append(p_lalone)
            all_llinked[most_peaked_id].append(p_llinked)
        print "Left score: %.2f" %comp_score_l,
        if no_association:
            print "(no association found)"
        else:
            print
                
        ### ---------------
        ### Process right hemisphere
        ### ---------------
        no_association = False
        ### Compute mean meshes
        mean_rvertices = np.hstack(
            (s1_rvertices, s2_rvertices)).reshape(
            (s1_rvertices.shape[0],2,3)).mean(1)
        mean_rmesh_graph = mesh_to_graph(mean_rvertices, s1_rtriangles)        
        ### Compute distances
        s2_rpeaks = np.where(s2_rcoord != -1)[0]
        if s1_rpeaks.size < s2_rpeaks.size:
            less_peaked = s1_rpeaks
            less_peaked_id = s1
            most_peaked = s2_rpeaks
            most_peaked_id = s2
        else:
            less_peaked = s2_rpeaks
            less_peaked_id = s2
            most_peaked = s1_rpeaks
            most_peaked_id = s1
        n = less_peaked.size
        p = most_peaked.size
        rdistances = np.zeros((n,p))
        for i, vertex_id in enumerate(less_peaked):
            rdistances[i,:] = mean_rmesh_graph.dijkstra(vertex_id)[most_peaked]
        # scale distances
        rdistances = ((1. + np.sqrt(2.))/2.) * rdistances
        if np.amin(rdistances) >= gamma:
            comp_score_r = (gamma / 2.) * (less_peaked.size + most_peaked.size)
            all_ralone[less_peaked_id].append(less_peaked)
            all_ralone[most_peaked_id].append(most_peaked)
            no_association = True
        else:
            ### Reorder matrix for a faster algorithm
            # remove rows >= gamma
            n_ralone = less_peaked[np.where(np.amin(rdistances, 1) >= gamma)[0]]
            n_rmaybe_linked = less_peaked[np.where(np.amin(rdistances,1)<gamma)[0]]
            rdistances = rdistances[np.amin(rdistances, 1) < gamma,:]
            # remove columns >= gamma
            p_ralone = most_peaked[np.where(np.amin(rdistances, 0) >= gamma)[0]]
            p_rmaybe_linked = most_peaked[np.where(np.amin(rdistances,0)<gamma)[0]]
            rdistances = rdistances[:,np.amin(rdistances, 0) < gamma]
            # reorder the matrix
            m = rdistances.shape[0]
            q = rdistances.shape[1]
            reorder = np.zeros((m,m))
            reorder_aux = np.argsort(np.amin(rdistances, 1))
            reorder[np.arange(0,m),reorder_aux] = 1
            rdistances = np.dot(reorder, rdistances)
            rdistances = np.hstack((rdistances, gamma*np.ones((m,1))))
            
            ### Finally find best matches
            comp_score_r, trace_score = find_score(rdistances, 0, gamma*p, gamma)
            comp_score_r += (gamma/2.)*(n_ralone.size + p_ralone.size)
            # take into account the fact we took submatrices to recover
            # right indices
            trace_score_formated = format_score(trace_score.copy())
            match_distances = \
                    rdistances[np.arange(rdistances.shape[0]),trace_score_formated]
            # update s1 dict
            new_n_ralone = n_rmaybe_linked[match_distances == gamma]
            n_ralone = np.concatenate((n_ralone, new_n_ralone))
            n_rlinked = n_rmaybe_linked[match_distances != gamma]
            all_ralone[less_peaked_id].append(n_ralone)
            all_rlinked[less_peaked_id].append(n_rlinked)
            # update s2 dict
            mask_p_rlinked = trace_score_formated[match_distances == gamma]
            new_p_ralone = \
                    p_rmaybe_linked[mask_p_rlinked != rdistances.shape[1]]
            p_ralone = np.concatenate((p_ralone, new_p_ralone))
            p_rlinked = \
                p_rmaybe_linked[trace_score_formated[[match_distances != gamma]]]
            all_ralone[most_peaked_id].append(p_ralone)
            all_rlinked[most_peaked_id].append(p_rlinked)
        print "Right score: %.2f" %comp_score_r,
        if no_association:
            print "(no association found)"
        else:
            print
            
        print "Total score: %.2f" %(comp_score_l + comp_score_r)

    sys.stdout.flush()
