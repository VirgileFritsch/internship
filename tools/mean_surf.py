"""
Script to generate a mean mesh from a set of subjects meshes.

"""
import numpy as np
from gifti import loadImage, saveImage, GiftiImage_fromTriangles

# list of subjects
subj = ['s12069', 's12300', 's12401', 's12431', 's12508', 's12532', 's12539', 's12562','s12590', 's12635', 's12636', 's12898', 's12081', 's12165', 's12207', 's12344', 's12352', 's12370', 's12381', 's12405', 's12414', 's12432']
nsubj = len(subj)
database_path = '/volatile/subjects_database'

lvertices = 0.
rvertices = 0.
for s in subj:
    lmesh_path = '%s/%s/surf/lh.r.white.normalized.gii' %(database_path, s)
    rmesh_path = '%s/%s/surf/rh.r.white.normalized.gii' %(database_path, s)
    
    # read left hemisphere data
    lmesh = loadImage(lmesh_path)
    if len(lmesh.getArrays()) == 2:
        lc, lt = lmesh.getArrays()
    elif len(lmesh.getArrays()) == 3:
        lc, ln, lt = lmesh.getArrays()
    else:
        raise Exception("Unable to read gifti data")
    lvertices += lc.getData()

    # read right hemisphere data
    rmesh = loadImage(rmesh_path)
    if len(rmesh.getArrays()) == 2:
        rc, rt = rmesh.getArrays()
    elif len(rmesh.getArrays()) == 3:
        rc, rn, rt = rmesh.getArrays()
    else:
        raise Exception("Unable to read gifti data")
    rvertices += rc.getData()

# compute the mean of all meshes
lvertices /= float(nsubj)
rvertices /= float(nsubj)

# construct and write output meshes
mean_lsurf = GiftiImage_fromTriangles(lvertices, lt.getData())
mean_rsurf = GiftiImage_fromTriangles(rvertices, rt.getData())
saveImage(mean_lsurf,
          '%s/group_analysis/surf/lh.r.white.normalized.gii' %database_path)
saveImage(mean_rsurf,
          '%s/group_analysis/surf/rh.r.white.normalized.gii' %database_path)
