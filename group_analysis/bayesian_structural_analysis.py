# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Example of a script that uses the BSA (Bayesian Structural Analysis)
i.e. nipy.neurospin.spatial_models.bayesian_structural_analysis
module

Author : Bertrand Thirion, 2008-2010
"""
print __doc__

#autoindent
import numpy as np
import scipy.stats as st
import os.path as op
import tempfile

from nipy.neurospin.spatial_models.bsa_io import make_bsa_image


# Get the data
#get_data_light.getIt()
subj = ['s12069', 's12300', 's12401', 's12431', 's12508', 's12532', 's12539', 's12562','s12590', 's12635', 's12636', 's12898', 's12081', 's12165', 's12207', 's12344', 's12352', 's12370', 's12381', 's12405', 's12414', 's12432']
nbsubj = len(subj)
db_path = '/volatile/subjects_database'
mask_images = [op.join(db_path,"%s/fMRI/default_acquisition/Minf/mask.nii") % s for s in subj]
betas = [op.join(db_path,"%s/fMRI/default_acquisition/glm/smoothed_FWHM5/Contrast/audio-video_z_map.nii") % s for s in subj]

nbeta = 25


#data_dir = op.expanduser(op.join('~', '.nipy', 'tests', 'data',
#                                 'group_t_images'))
#betas =[ op.join(data_dir,'spmT_%04d_subj_%02d.nii'%(nbeta,n))
#                 for n in range(nbsubj)]

# set various parameters
subj_id = ['%04d' %i for i in range(nbsubj)]
theta = float(st.t.isf(0.01, 100))
dmax = 4.
ths = 0 #2# or nbsubj/4
thq = 0.95
verbose = 1
smin = 5
swd = tempfile.mkdtemp()
method = 'quick'
print 'method used:', method

# call the function
AF, BF = make_bsa_image(mask_images, betas, theta, dmax, ths, thq, smin, swd,
                        method, subj_id, '%04d'%nbeta, reshuffle=False)

# Write the result. OK, this is only a temporary solution
import pickle
picname = op.join(swd,"AF_%04d.pic" %nbeta)
pickle.dump(AF, open(picname, 'w'), 2)
picname = op.join(swd,"BF_%04d.pic" %nbeta)
pickle.dump(BF, open(picname, 'w'), 2)

print "Wrote all the results in directory %s"%swd
