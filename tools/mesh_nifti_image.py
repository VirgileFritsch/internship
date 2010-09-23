"""
Script to generate meshes from thresholded activation images
(for visualization in anatomist)

"""

from database_archi import *
import commands

# File to generate meshes from
what = "leaves"

# Output directory
swd = BLOBS3D_DIR

# Mesh it
print commands.getoutput("AimsMesh -i '%s'/%s.nii" %(swd, what))
print commands.getoutput("AimsMeshes2Graph -i '%s'/%s*.mesh -o '%s'/%s_.arg" %(swd,what,swd,what))
