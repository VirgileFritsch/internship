import numpy as np
from nipy.io.imageformats import load, Nifti1Image, save

all_subj = ['s12069', 's12300', 's12401', 's12431', 's12508', 's12532', 's12539', 's12562','s12590', 's12635', 's12636', 's12898', 's12081', 's12165', 's12207', 's12344', 's12352', 's12370', 's12381', 's12405', 's12414', 's12432']
nb_subj = len(all_subj)

for s in all_subj:
    template = load('/volatile/subjects_database/%s/fMRI/default_acquisition/glm/smoothed_FWHM5/Contrast/audio-video_z_map.nii'%s).get_data().copy()
    affine = load('/volatile/subjects_database/%s/fMRI/default_acquisition/glm/smoothed_FWHM5/Contrast/audio-video_z_map.nii'%s).get_affine()
    shape = load('/volatile/subjects_database/%s/fMRI/default_acquisition/glm/smoothed_FWHM5/Contrast/audio-video_z_map.nii'%s).get_shape()
    beta = np.load('/volatile/subjects_database/%s/fMRI/default_acquisition/glm/smoothed_FWHM5/loc2/vba.npz'%s)['beta']
    
    new_data = np.asarray(template)
    new_data[new_data != 0] = beta
    beta = beta.reshape(shape)

    save(Nifti1Image(new_data, affine), '/volatile/subjects_database/%s/fMRI/default_acquisition/glm/smoothed_FWHM5/loc2/beta.nii'%s)

