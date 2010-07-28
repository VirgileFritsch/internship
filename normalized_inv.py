from numpy import array, dot, hstack, reshape
from soma import aims
import sys

subjects = ['s12069', 's12300', 's12401', 's12431', 's12508', 's12532', 's12539', 's12562','s12590', 's12635', 's12636', 's12898', 's12081', 's12165', 's12207', 's12344', 's12352', 's12370', 's12381', 's12405', 's12414', 's12432']
#subjects = ['s12207']
for s in subjects:
    mesh = aims.read('/data/home/virgile/virgile_internship/%s/surf/lh.r.aims.white.normalized.mesh'%s)
    #anat = aims.read('/data/home/virgile/virgile_internship/s12069/mri/orig/001.nii')
    anat = aims.read('/data/home/virgile/virgile_internship/s12069/experiments/smoothed_FWHM5/audio-video_z_map_smin5_theta3.3/leaves.nii')
    
    z = array(anat.header()['transformations'][0]).reshape(4, 4)
    
    for i in range(len(mesh.vertex())):
        mesh.vertex()[i] = dot(z,hstack((mesh.vertex()[i], [1])))[:3]
    
    for p in mesh.polygon():
        p[0],p[2] = p[2],p[0]
    
    mesh.updateNormals()
    aims.write(mesh, '/data/home/virgile/virgile_internship/%s/surf/lh.r.white.normalized.gii'%s)
