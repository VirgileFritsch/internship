import numpy as np
import sys

from nipy.neurospin.glm_files_layout import tio
from gifti import loadImage
from kuhnMunkres import maxWeightMatching
import nipy.neurospin.graph.graph as fg

from database_archi import *

#--------------------------------------------------------
all_subj = ['s12069', 's12300', 's12401', 's12431', 's12508', 's12532', 's12539', 's12562','s12590', 's12635', 's12636', 's12898', 's12081', 's12165', 's12207', 's12344', 's12352', 's12370', 's12381', 's12405', 's12414', 's12432']
#subj = ['s12069', 's12081', 's12300']
#subj = ['s12539', 's12539']
subj2 = all_subj[:]
subj1 = [SUBJECT]
#--------------------------------------------------------
bound = 24;
gamma = 35./16.
coord_type = "coord_test2"

#--------------------------------------------------------
def phi(dik, bound):
    """ Triweight kernel
    """
    dik[dik > bound] = bound
    return 2. * (35./32.)*(1-((dik-bound)/bound)**2)**3

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

#--------------------------------------------------------

all_lalone = {}
all_ralone = {}
all_llinked = {}
all_rlinked = {}
for s in all_subj:
    all_lalone[s] = []
    all_ralone[s] = []
    all_llinked[s] = []
    all_rlinked[s] = []

all_scores = np.zeros((len(subj1),len(subj2)))
#removed = 1
print 'Comparison %s -- gamma=%g' %(coord_type, gamma)
for s1_id, s1 in enumerate(subj1):
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

    #subj2.remove(s1)
    #removed += 1
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
        if n != 0:
            ldistances = np.zeros((n,p))
            for i, vertex_id in enumerate(less_peaked):
                ldistances[i,:] = \
                            mean_lmesh_graph.dijkstra(vertex_id)[most_peaked]
            # scale distances
            ldistances = ((1. + np.sqrt(2.))/2.) * ldistances
            ldistances = phi(ldistances, bound)
            #ldistances = np.hstack((ldistances, gamma*np.ones((n,n))))
            if (p-n) > 0:
                ldistances = np.vstack((ldistances, gamma*np.ones((p-n,p))))
            (Mu,Mv,val) = maxWeightMatching(-ldistances)
            matching_res = ldistances[Mu.keys(), Mu.values()]
            val = np.abs((val)/min(n,p))
            print val
            mask = np.where(matching_res < gamma)[0]
            mask = mask[mask < less_peaked.size]
            all_llinked[less_peaked_id].append(less_peaked[mask])
            mask = np.where(matching_res >= gamma)[0]
            mask = mask[mask < less_peaked.size]
            all_lalone[less_peaked_id].append(less_peaked[mask])
            mask = ldistances[Mv.values(), Mv.keys()]
            
            matching_res = ldistances[Mv.values(), Mv.keys()]
            mask = np.where(matching_res < gamma)[0]
            mask = mask[mask < most_peaked.size]
            all_llinked[most_peaked_id].append(most_peaked[mask])
            mask = np.where(matching_res >= gamma)[0]
            mask = mask[mask < most_peaked.size]
            all_lalone[most_peaked_id].append(most_peaked[mask])
            
            all_scores[s1_id,s2_id] = val
        else:
            val = p*gamma / 2.
            print val
            all_lalone[most_peaked_id].append(most_peaked)
            all_scores[s1_id,s2_id] = val
                
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
        if n != 0:
            rdistances = np.zeros((n,p))
            for i, vertex_id in enumerate(less_peaked):
                rdistances[i,:] = \
                            mean_rmesh_graph.dijkstra(vertex_id)[most_peaked]
            # scale distances
            rdistances = ((1. + np.sqrt(2.))/2.) * rdistances
            rdistances = phi(rdistances, bound)
            #rdistances =  np.hstack((rdistances, gamma*np.ones((n,n))))
            if (p - n) > 0:
                rdistances =  np.vstack((rdistances, gamma*np.ones((p-n,p))))
            (Mu, Mv, val) = maxWeightMatching(-rdistances)
            val = np.abs((val)/min(n,p))
            matching_res = rdistances[Mu.keys(), Mu.values()]
            print val
            # update dictionaries
            mask = np.where(matching_res < gamma)[0]
            mask = mask[mask < less_peaked.size]
            all_rlinked[less_peaked_id].append(less_peaked[mask])
            mask = np.where(matching_res >= gamma)[0]
            mask = mask[mask < less_peaked.size]
            all_ralone[less_peaked_id].append(less_peaked[mask])
            mask = rdistances[Mv.values(), Mv.keys()]
            
            matching_res = rdistances[Mv.values(), Mv.keys()]
            mask = np.where(matching_res < gamma)[0]
            mask = mask[mask < most_peaked.size]
            all_rlinked[most_peaked_id].append(most_peaked[mask])
            mask = np.where(matching_res >= gamma)[0]
            mask = mask[mask < most_peaked.size]
            all_ralone[most_peaked_id].append(most_peaked[mask])
            
            all_scores[s1_id,s2_id] += val
        else:
            val = p*gamma / 2.
            print val
            all_ralone[most_peaked_id].append(most_peaked)
            all_scores[s1_id,s2_id] += val
            

if s1 == 's12069':
    np.savez('/volatile/comp_%s/comp_%s.npz' %(CONTRAST,coord_type), all_scores)
else:
    toto = np.load('/volatile/comp_%s/comp_%s.npz' %(CONTRAST,coord_type))
    titi = np.vstack((toto['arr_0'], all_scores))
    np.savez('/volatile/comp_%s/comp_%s.npz' %(CONTRAST,coord_type),titi)


def find_freq(results, vertex):
    count = 0
    for l in results:
        if vertex in l:
            count += 1
    return count

tmp_results = np.zeros((2,s1_rpeaks.size))
for i, v in enumerate(s1_rpeaks):
    freq = find_freq(all_rlinked[SUBJECT], v)
    freq2 = find_freq(all_ralone[SUBJECT], v)
    print v, "\t", ((freq-2.)/21.) * 100.
    tmp_results[0,i] = v
    tmp_results[1,i] = ((freq-2.)/21.) * 100.
np.savez('/volatile/comp_%s/comp_%s_%s_right.npz' %(CONTRAST, coord_type, s1), tmp_results)

print
tmp_results = np.zeros((2,s1_lpeaks.size))
for i, v in enumerate(s1_lpeaks):
    freq = find_freq(all_llinked[SUBJECT], v)
    freq2 = find_freq(all_lalone[SUBJECT], v)
    print v, "\t", ((freq-2.)/21.) * 100., "\t"
    tmp_results[0,i] = v
    tmp_results[1,i] = ((freq-2.)/21.) * 100.
np.savez('/volatile/comp_%s/comp_%s_%s_left.npz' %(CONTRAST, coord_type, s1), tmp_results)
