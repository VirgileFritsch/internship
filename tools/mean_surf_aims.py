"""
Script to generate a mean mesh from a set of subjects meshes.
AIMS version

"""
import numpy as np
from gifti import loadImage, saveImage, GiftiImage_fromTriangles
from soma import aims

# list of subjects
subj = ['s12069', 's12300', 's12401', 's12431', 's12508', 's12532', 's12539', 's12562','s12590', 's12635', 's12636', 's12898', 's12081', 's12165', 's12207', 's12344', 's12352', 's12370', 's12381', 's12405', 's12414', 's12432']
nsubj = len(subj)
db_path = '/volatile/subjects_database'

lvertices = 0.
rvertices = 0.
for s in subj:
    lmesh_path = '%s/%s/surf/lh.r.aims.white.normalized.mesh' %(db_path, s)
    rmesh_path = '%s/%s/surf/rh.r.aims.white.normalized.mesh' %(db_path, s)
    R = aims.Reader()

    # read left hemisphere data
    lmesh = R.read(lmesh_path)
    lvertices += np.asarray(lmesh.vertex())
    ltriangles = np.asarray(lmesh.polygon())

    # read right hemisphere data
    rmesh = R.read(rmesh_path)
    rvertices += np.asarray(rmesh.vertex())
    rtriangles = np.asarray(rmesh.polygon())

# compute the mean of all meshes
lvertices /= float(nsubj)
rvertices /= float(nsubj)

# construct and write output meshes
W = aims.Writer()

new_rmesh = aims.AimsTimeSurface_3()
rvert = new_rmesh.vertex()
rpoly = new_rmesh.polygon()
rvert.assign([aims.Point3df(x) for x in rvertices])
rpoly.assign([aims.AimsVector_U32_3(x) for x in rtriangles])
aims.write(new_rmesh, '%s/group/surf/rh.r.aims.white.normalized.mesh' %db_path)

new_lmesh = aims.AimsTimeSurface_3()
lvert = new_lmesh.vertex()
lpoly = new_lmesh.polygon()
lvert.assign([aims.Point3df(x) for x in lvertices])
lpoly.assign([aims.AimsVector_U32_3(x) for x in ltriangles])
aims.write(new_lmesh, '%s/group/surf/lh.r.aims.white.normalized.mesh' %db_path)
