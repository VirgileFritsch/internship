"""
Script comparing pairwise similarity between subjects, given surface
coordinates of their activation peaks.
The similarity distance is the one from B. Thirion and al. 2010.

Author: Virgile Fritsch, 2010

"""
import sys, os
import numpy as np

from nipy.neurospin.glm_files_layout import tio
from gifti import loadImage

# -----------------------------------------------------------
# --------- Paths -------------------------------------------
# -----------------------------------------------------------
from database_archi import *
if CONTRAST == "simulation":
    middle_path = "matching"
else:
    middle_path = "experiments/smoothed_FWHM5/%s" %CONTRAST

#----- Path to the subjects database
db_path = '/volatile/subjects_database'

#----- Path to the simulated subjects database
simul_root_path = '/volatile/virgile/internship/simulation'
simul_id = 'sim1'
simul_path = '%s/%s' %(simul_root_path, simul_id)
if CONTRAST != "simulation":
    simul_path = db_path

# -----------------------------------------------------------
# --------- Parameters --------------------------------------
# -----------------------------------------------------------
#all_subj = ['f%d'%s for s in range(1,23)]
all_subj = ['s12069', 's12300', 's12401', 's12431', 's12508', 's12532', 's12539', 's12562','s12590', 's12635', 's12636', 's12898', 's12081', 's12165', 's12207', 's12344', 's12352', 's12370', 's12381', 's12405', 's12414', 's12432']

DELTA = .15
coord_types = ["fcoord", "coord"]

# -----------------------------------------------------------
# --------- Start pairwise comparison -----------------------
# -----------------------------------------------------------
nb_subjects = len(all_subj)
subj1 = all_subj[:]
subj2 = all_subj[:]
all_lvertices = {}
all_rvertices = {}
all_llinked = {}
all_rlinked = {}
for s in all_subj:
    all_llinked[s] = []
    all_rlinked[s] = []

all_scores = np.zeros((len(subj1),len(subj2)))

# Read longitude and latitude textures
longitude_tex_path = "%s/ico100_7_lon.tex" %(db_path)
longitude = tio.Texture(longitude_tex_path).read(longitude_tex_path).data
latitude_tex_path = "%s/ico100_7_lat.tex" %(db_path)
latitude = tio.Texture(latitude_tex_path).read(latitude_tex_path).data

scores_by_coord_type = []
lvertices_by_coord_type = []
rvertices_by_coord_type = []
for coord in coord_types:
    print 'Comparison %s (DELTA=%g)' %(coord, DELTA)
    for s1_id, s1 in enumerate(subj1):
        ### Read subject s1 coordinate textures (level 1)
        s1_lcoord_tex = "%s/%s/%s/results_%s_level001/left.tex" \
                        %(simul_path, s1, middle_path, coord)
        s1_lcoord = tio.Texture(s1_lcoord_tex).read(s1_lcoord_tex).data
        s1_rcoord_tex = "%s/%s/%s/results_%s_level001/right.tex" \
                        %(simul_path, s1, middle_path, coord)
        s1_rcoord = tio.Texture(s1_rcoord_tex).read(s1_rcoord_tex).data
    
        s1_lpeaks = np.where(s1_lcoord != -1)[0]
        all_lvertices[s1] = s1_lpeaks
        s1_ltheta = longitude[s1_lpeaks]
        s1_lphi = latitude[s1_lpeaks]
    
        s1_rpeaks = np.where(s1_rcoord != -1)[0]
        all_rvertices[s1] = s1_rpeaks
        s1_rtheta = longitude[s1_rpeaks]
        s1_rphi = latitude[s1_rpeaks]

        for s2_id, s2 in enumerate(subj2):
            #print "-- Subject %s vs subject %s" %(s1, s2)
            ### Read subject s2 coordinate textures (level 1)
            s2_lcoord_tex = "%s/%s/%s/results_%s_level001/left.tex" \
                            %(simul_path, s2, middle_path, coord)
            s2_lcoord = tio.Texture(s2_lcoord_tex).read(s2_lcoord_tex).data
            s2_rcoord_tex = "%s/%s/%s/results_%s_level001/right.tex" \
                            %(simul_path, s2, middle_path, coord)
            s2_rcoord = tio.Texture(s2_rcoord_tex).read(s2_rcoord_tex).data
    
            ### ---------------
            ### Process left hemisphere
            ### ---------------
            ### Compute distances
            s2_lpeaks = np.where(s2_lcoord != -1)[0]
            s2_ltheta = longitude[s2_lpeaks]
            s2_lphi = latitude[s2_lpeaks]
            n = s1_lpeaks.size
            p = s2_lpeaks.size
            
            if n != 0 and p != 0:
                long_diff = np.tile(s1_ltheta, (p,1)).T - \
                            np.tile(s2_ltheta, (n,1))
                lat_diff = np.tile(s1_lphi, (p,1)).T - np.tile(s2_lphi, (n,1))
                term1 = np.cos(long_diff)
                term2 = (np.cos(lat_diff)-1) * \
                        np.tile(np.sin(s1_ltheta), (p,1)).T * \
                        np.tile(np.sin(s2_ltheta), (n,1))
                ldistances = np.arccos(term1 + term2)
                ldistances_aux = np.exp(-((ldistances)**2)/(2.*DELTA**2))
                lsensitivity_argmax = np.argmax(ldistances_aux, 1)
                lsensitivity = 1. - (1./n) * np.sum(
                    ldistances_aux[np.arange(n), lsensitivity_argmax])
    
                # update s2 linked peaks
                all_llinked[s2].append(
                    s2_lpeaks[np.unique(lsensitivity_argmax)])
                # update scores dictionary
                all_scores[s1_id,s2_id] = lsensitivity
            else:
                lsensitivity = 1.
                # update scores dictionary
                all_scores[s1_id,s2_id] = lsensitivity
                
            ### ---------------
            ### Process right hemisphere
            ### ---------------
            ### Compute distances
            s2_rpeaks = np.where(s2_rcoord != -1)[0]
            s2_rtheta = longitude[s2_rpeaks]
            s2_rphi = latitude[s2_rpeaks]
            n = s1_rpeaks.size
            p = s2_rpeaks.size
            
            if n != 0 and p != 0:
                long_diff = np.tile(s1_rtheta, (p,1)).T - \
                            np.tile(s2_rtheta, (n,1))
                lat_diff = np.tile(s1_rphi, (p,1)).T - np.tile(s2_rphi, (n,1))
                term1 = np.cos(long_diff)
                term2 = (np.cos(lat_diff)-1) * \
                        np.tile(np.sin(s1_rtheta), (p,1)).T * \
                        np.tile(np.sin(s2_rtheta), (n,1))
                rdistances = np.arccos(term1 + term2)
                rdistances_aux = np.exp(-((rdistances)**2)/(2.*DELTA**2))
                rsensitivity_argmax = np.argmax(rdistances_aux, 1)
                rsensitivity = 1. - (1./n) * np.sum(
                    rdistances_aux[np.arange(n), rsensitivity_argmax])
    
                # update s2 linked peaks
                all_rlinked[s2].append(
                    s2_rpeaks[np.unique(rsensitivity_argmax)])
                # update scores dictionary
                all_scores[s1_id,s2_id] += rsensitivity
            else:
                rsensitivity = 1.
                # update scores dictionary
                all_scores[s1_id,s2_id] += rsensitivity

    scores_by_coord_type.append(all_scores.copy())
    lvertices_by_coord_type.append(all_lvertices.copy())
    rvertices_by_coord_type.append(all_rvertices.copy())

res_coord_1 = scores_by_coord_type[0]
lvertices_1 = lvertices_by_coord_type[0]
rvertices_1 = rvertices_by_coord_type[0]
res_coord_2 = scores_by_coord_type[1]
lvertices_2 = lvertices_by_coord_type[1]
rvertices_2 = rvertices_by_coord_type[1]

"""
#-----------------------------------------------------------------------
# Analysis of artifact recognition amongst subjects
subjects_with_artifact = \
                np.load('%s/subjects_with_artifact.npz' %simul_path)['arr_0']
for coord_index, coord_type in enumerate(coord_types):
    res_coord = scores_by_coord_type[coord_index]
    lvertices = lvertices_by_coord_type[coord_index]
    rvertices = rvertices_by_coord_type[coord_index]

    subjects_good = 0
    subjects_bad = 0
    subjects_same = 0
    subjects_error = 0
    for s_id, s in enumerate(all_subj):
        if s_id+1 in subjects_with_artifact:
            selected = rvertices['f%d'%(s_id+1)]
            blob_composition = np.load(
                '%s/f%d/blob_composition.npz' %(simul_path, s_id+1))['arr_0']
            artifact_composition = np.load(
                '%s/f%d/artifact_composition.npz'%(simul_path, s_id+1))['arr_0']
            count_blob = 0
            count_artifact = 0
            # count how many selected peaks comes from the blob/the artifact
            for v in selected:
                if v in blob_composition:
                    count_blob += 1
                if v in artifact_composition:
                    count_artifact += 1
            # analyse results
            if count_blob != 0 and count_artifact != 0:
                subjects_same += 1
                #print "%s: artifact not detected" %s
            elif count_blob == 0 and count_artifact == 0:
                #print "%s: empty texture" %s
                subjects_error += 1
            elif count_artifact == 0:
                #print "%s: artifact removed" %s
                subjects_good += 1
            elif count_artifact > 0:
                #print "%s: artifact kept (/!\)" %s
                subjects_bad += 1

    if subjects_error > 0:
        raise Exception("There was an error on %d subjects" %subjects_error)
    print "Artifact removed: %d/%d" %(subjects_good, len(all_subj))
    print "Artifact not detected: %d/%d" %(subjects_same, len(all_subj))
    print "Artifact kept: %d/%d" %(subjects_bad, len(all_subj))

    # ...
    if os.path.isfile('%s/results/valid_%s.npz' %(simul_path, coord_type)):
        previous_results = np.load(
            '%s/results/valid_%s.npz' %(simul_path, coord_type))['res']
        previous_results[0] += subjects_good
        previous_results[1] += subjects_same
        previous_results[2] += subjects_bad
        previous_results[3] += len(subjects_with_artifact)
        previous_results[4] += 1
        np.savez('%s/results/valid_%s.npz' %(simul_path, coord_type),
                 res=previous_results)
    else:
        if not os.path.isdir('%s/results' %simul_path):
            os.mkdir('%s/results' %simul_path)
        previous_results = np.zeros(5)
        previous_results[0] = subjects_good
        previous_results[1] = subjects_same
        previous_results[2] = subjects_bad
        previous_results[3] = len(subjects_with_artifact)
        previous_results[4] = 1
        np.savez('%s/results/valid_%s.npz' %(simul_path, coord_type),
                 res=previous_results)
            
"""


#-----------------------------------------------------------------------
# Find significant changes
th = 2.960
for coord_index_1, coord_type_1 in enumerate(coord_types[:-1]):
    res_coord_1 = scores_by_coord_type[coord_index_1]
    lvertices_1 = lvertices_by_coord_type[coord_index_1]
    rvertices_1 = rvertices_by_coord_type[coord_index_1]
    for offset_index, coord_type_2 in enumerate(coord_types[coord_index_1+1:]):
        coord_index_2 = coord_index_1 + offset_index + 1
        res_coord_2 = scores_by_coord_type[coord_index_2]
        lvertices_2 = lvertices_by_coord_type[coord_index_2]
        rvertices_2 = rvertices_by_coord_type[coord_index_2]

        print "Comparison %s vs %s" %(coord_type_1, coord_type_2)
        diff = res_coord_1 - res_coord_2
        diff_test = diff.copy()
        diff_test[np.arange(nb_subjects), np.arange(nb_subjects)] = 100.
        diff_test = np.ravel(diff_test[diff_test != 100.])
        comp_mean = diff_test.mean()
        comp_std = diff_test.std()
        t_stat = (comp_mean/comp_std) * np.sqrt(nb_subjects * (nb_subjects -1))
        if t_stat > th:
            print "Significant improvement: %f - %f" %(comp_mean, comp_std)
        elif t_stat < -th:
            print "Significant worsening: %f - %f" %(comp_mean, comp_std)
        else:
            print "No significant change"

        if os.path.isfile('%s/results/comp_%s-%s.npz' \
                          %(simul_path, coord_type_1, coord_type_2)):
            previous_results = np.load('%s/results/comp_%s-%s.npz' \
                            %(simul_path, coord_type_1, coord_type_2))
            previous_significant_means = previous_results['mean']
            previous_significant_means = np.concatenate(
                (previous_significant_means, [comp_mean]))
            previous_significant_std = previous_results['std']
            previous_significant_std = np.concatenate(
                (previous_significant_std, [comp_std]))
            np.savez('%s/results/comp_%s-%s.npz' \
                     %(simul_path, coord_type_1, coord_type_2),
                     mean=previous_significant_means,
                     std=previous_significant_std)
        else:
            if not os.path.isdir('%s/results' %simul_path):
                os.mkdir('%s/results' %simul_path)
            significant_means = np.array([comp_mean])
            significant_std = np.array([comp_std])
            np.savez('%s/results/comp_%s-%s.npz' \
                     %(simul_path, coord_type_1, coord_type_2),
                     mean=significant_means, std=significant_std)



"""
def find_freq(results, vertex):
    count = 0
    for l in results:
        if vertex in l:
            count += 1
    return count


tmp_results = np.zeros((2,all_rvertices["s12069"].size))
for i, v in enumerate(all_rvertices["s12069"]):
    freq = find_freq(all_rlinked["s12069"], v)
    print v, "\t", ((freq-1.)/(len(all_subj)-1.)) * 100.
    tmp_results[0,i] = v
    tmp_results[1,i] = ((freq-1.)/(len(all_subj)-1.)) * 100.


print
tmp_results = np.zeros((2,all_lvertices["s12069"].size))
for i, v in enumerate(all_lvertices["s12069"]):
    freq = find_freq(all_llinked["s12069"], v)
    print v, "\t", ((freq-1.)/(len(all_subj)-1.)) * 100.
    tmp_results[0,i] = v
    tmp_results[1,i] = ((freq-1.)/(len(all_subj)-1.)) * 100.

"""
