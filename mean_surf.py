import numpy as np
from gifti import loadImage, saveImage, GiftiImage_fromTriangles

# list of subjects
subj = ['s12069', 's12300', 's12401', 's12431', 's12508', 's12532', 's12539', 's12562','s12590', 's12635', 's12636', 's12898', 's12081', 's12165', 's12207', 's12344', 's12352', 's12370', 's12381', 's12405', 's12414', 's12432']
nsubj = len(subj)

lvertices = 0.
rvertices = 0.
for s in subj:
    lmesh_path = '/data/home/virgile/virgile_internship/%s/surf/lh.r.white.normalized.gii' %s
    rmesh_path = '/data/home/virgile/virgile_internship/%s/surf/rh.r.white.normalized.gii' %s
    lmesh = loadImage(lmesh_path)
    rmesh = loadImage(rmesh_path)

    lc, ln, lt = lmesh.getArrays()
    lvertices += lc.getData()

    rc, rn, rt = rmesh.getArrays()
    rvertices += rc.getData()

lvertices /= float(nsubj)
rvertices /= float(nsubj)

mean_lsurf = GiftiImage_fromTriangles(lvertices, lt.getData())
mean_rsurf = GiftiImage_fromTriangles(rvertices, rt.getData())

saveImage(mean_lsurf, '/data/home/virgile/virgile_internship/group_analysis/surf/lh.r.white.normalized.gii')
saveImage(mean_rsurf, '/data/home/virgile/virgile_internship/group_analysis/surf/rh.r.white.normalized.gii')
