"""
Script comparing pairwise similarity between subjects, given surface
coordinates of their activation peaks.
The similarity distance is the one from B. Thirion and al. 2010.

Author: Virgile Fritsch, 2010

"""
import sys
import numpy as np

from nipy.neurospin.glm_files_layout import tio
from gifti import loadImage

# -----------------------------------------------------------
# --------- Paths -------------------------------------------
# -----------------------------------------------------------
from database_archi import *

#----- Path to the subjects database
db_path = '/volatile/subjects_database'

#----- Path to the simulated subjects database
simul_root_path = '/volatile/virgile/internship/simulation'
simul_id = 'sim1'
simul_path = '%s/%s' %(simul_root_path, simul_id)

# -----------------------------------------------------------
# --------- Parameters --------------------------------------
# -----------------------------------------------------------
all_subj = ['f%d'%s for s in range(1,23)]
subj2 = all_subj[:]
subj1 = all_subj[:]

DELTA = .15
coord = "coord"

# -----------------------------------------------------------
# --------- Start pairwise comparison -----------------------
# -----------------------------------------------------------
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

print 'Comparison %s vs ... (DELTA=%g)' %(coord, DELTA)
for s1_id, s1 in enumerate(subj1):
    ### Read subject s1 coordinate textures (level 1)
    s1_lcoord_tex = "%s/%s/matching/results_%s_level001/left.tex" \
                    %(simul_path, s1, coord)
    s1_lcoord = tio.Texture(s1_lcoord_tex).read(s1_lcoord_tex).data
    s1_rcoord_tex = "%s/%s/matching/results_%s_level001/right.tex" \
                    %(simul_path, s1, coord)
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
        s2_lcoord_tex = "%s/%s/matching/results_%s_level001/left.tex" \
                        %(simul_path, s2, coord)
        s2_lcoord = tio.Texture(s2_lcoord_tex).read(s2_lcoord_tex).data
        s2_rcoord_tex = "%s/%s/matching/results_%s_level001/right.tex" \
                        %(simul_path, s2, coord)
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
            long_diff = np.tile(s1_ltheta, (p,1)).T - np.tile(s2_ltheta, (n,1))
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
            all_llinked[s2].append(s2_lpeaks[np.unique(lsensitivity_argmax)])
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
            long_diff = np.tile(s1_rtheta, (p,1)).T - np.tile(s2_rtheta, (n,1))
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
            all_rlinked[s2].append(s2_rpeaks[np.unique(rsensitivity_argmax)])
            # update scores dictionary
            all_scores[s1_id,s2_id] += rsensitivity
        else:
            rsensitivity = 1.
            # update scores dictionary
            all_scores[s1_id,s2_id] += rsensitivity


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
