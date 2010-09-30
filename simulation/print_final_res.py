import os
import numpy as np

#----- Path to the simulated subjects database
simul_root_path = '/volatile/virgile/internship/simulation'
simul_id = 'sim1'
simul_path = '%s/%s' %(simul_root_path, simul_id)

coord_types = ["fcoord", "coord"]

for coord_type in coord_types:
    res = np.load('%s/results/valid_%s.npz' %(simul_path, coord_type))['res']
    
    subjects_good = res[0]
    subjects_same = res[1]
    subjects_bad = res[2]
    nb_subjects_with_artifact = res[3]
    nb_simul = res[4]

    print "--> Results %s:" %coord_type
    print "Artifact removed: %d/%d" %(subjects_good, nb_subjects_with_artifact)
    print "Artifact not detected: %d/%d" \
          %(subjects_same, nb_subjects_with_artifact)
    print "Artifact kept: %d/%d" %(subjects_bad, nb_subjects_with_artifact)
    print

for coord_index_1, coord_type_1 in enumerate(coord_types[:-1]):
    for coord_type_2 in coord_types[coord_index_1+1:]:
        if os.path.isfile('%s/results/comp_%s-%s.npz' \
                          %(simul_path, coord_type_1, coord_type_2)):
            res_comp = np.load('%s//results/comp_%s-%s.npz' \
                               %(simul_path, coord_type_1, coord_type_2))
            res_comp_mean = res_comp['mean']
            res_comp_std = res_comp['std']
            res_comp_improved = res_comp_mean[res_comp_mean > 0.]
            res_comp_worsened = res_comp_mean[res_comp_mean < 0.]

            print "--> Comparison %s - %s" %(coord_type_1, coord_type_2)
            print "Significant improvement: %d/%d (mean=%f)" \
                  %(res_comp_improved.size, nb_simul, res_comp_improved.mean())
            print "Significant worsening: %d/%d (mean=%f)" \
                  %(res_comp_worsened.size, nb_simul, res_comp_worsened.mean())
            print
