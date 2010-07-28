import numpy as np
from gifti import loadImage, saveImage, GiftiImage_fromTriangles
from soma import aims

# list of subjects
subj = ['s12069', 's12300', 's12401', 's12431', 's12508', 's12532', 's12539', 's12562','s12590', 's12635', 's12636', 's12898', 's12081', 's12165', 's12207', 's12344', 's12352', 's12370', 's12381', 's12405', 's12414', 's12432']
nsubj = len(subj)

lvertices = 0.
rvertices = 0.
for s in subj:
    lmesh_path = '/data/home/virgile/virgile_internship/%s/surf/lh.r.aims.white.mesh' %s
    rmesh_path = '/data/home/virgile/virgile_internship/%s/surf/rh.r.aims.white.mesh' %s
    R = aims.Reader()
    
    lmesh = R.read(lmesh_path)
    lvertices += np.asarray(lmesh.vertex())
    ltriangles = np.asarray(lmesh.polygon())
    
    rmesh = R.read(rmesh_path)
    rvertices += np.asarray(rmesh.vertex())
    rtriangles = np.asarray(rmesh.polygon())

lvertices /= float(nsubj)
rvertices /= float(nsubj)

W = aims.Writer()
new_rmesh = aims.AimsTimeSurface_3()
rvert = new_rmesh.vertex()
rpoly = new_rmesh.polygon()
rvert.assign([aims.Point3df(x) for x in rvertices])
rpoly.assign([aims.AimsVector_U32_3(x) for x in rtriangles])
aims.write(new_rmesh, '/data/home/virgile/virgile_internship/group/surf/rh.r.aims.white.mesh')

new_lmesh = aims.AimsTimeSurface_3()
lvert = new_lmesh.vertex()
lpoly = new_lmesh.polygon()
lvert.assign([aims.Point3df(x) for x in lvertices])
lpoly.assign([aims.AimsVector_U32_3(x) for x in ltriangles])
aims.write(new_lmesh, '/data/home/virgile/virgile_internship/group/surf/lh.r.aims.white.mesh')
