"""
Script to convert a mesh in the aims coordinates system to the same mesh in the freesurfer coordinates system.

"""
import sys
import numpy as np
from soma import aims

subjects = ['s12069', 's12300', 's12401', 's12431', 's12508', 's12532', 's12539', 's12562','s12590', 's12635', 's12636', 's12898', 's12081', 's12165', 's12207', 's12344', 's12352', 's12370', 's12381', 's12405', 's12414', 's12432']
db_path = '/volatile/subjects_database'

for s in subjects:
    # Left hemisphere
    mesh = aims.read('%s/%s/surf/lh.r.aims.white.normalized.mesh' %(db_path, s))
    #ref = aims.read('%s/s12069/mri/orig/001.nii' %db_path)
    ref = aims.read('%s/s12069/experiments/smoothed_FWHM5/audio-video_z_map_smin5_theta3.3/leaves.nii' %db_path)
    
    z = np.array(ref.header()['transformations'][0]).reshape(4, 4)
    
    for i in range(len(mesh.vertex())):
        mesh.vertex()[i] = np.dot(z, np.hstack((mesh.vertex()[i], [1])))[:3]
    for p in mesh.polygon():
        p[0], p[2] = p[2], p[0]
    mesh.updateNormals()
    
    aims.write(mesh, '%s/%s/surf/lh.r.white.normalized.gii' %(bd_path, s))

    # Right hemisphere
    mesh = aims.read('%s/%s/surf/rh.r.aims.white.normalized.mesh' %(db_path, s))
    #ref = aims.read('%s/s12069/mri/orig/001.nii' %db_path)
    ref = aims.read('%s/s12069/experiments/smoothed_FWHM5/audio-video_z_map_smin5_theta3.3/leaves.nii' %db_path)
    
    z = np.array(ref.header()['transformations'][0]).reshape(4, 4)
    
    for i in range(len(mesh.vertex())):
        mesh.vertex()[i] = np.dot(z, np.hstack((mesh.vertex()[i], [1])))[:3]
    for p in mesh.polygon():
        p[0], p[2] = p[2], p[0]
    mesh.updateNormals()
    
    aims.write(mesh, '%s/%s/surf/rh.r.white.normalized.gii' %(bd_path, s))
