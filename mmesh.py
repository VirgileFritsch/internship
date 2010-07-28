"""
Script to generate meshes from thresholded activation images
(for visualization in anatomist)
"""

from database_archi import *
import commands

what = "leaves"
#swd='/data/home/virgile/virgile_internship/group_analysis/smoothed_FWHM5/blobs3D_audio-video'
swd=BLOBS3D_DIR
#print commands.getoutput("AimsThreshold -i '%s'/leaves.nii -o '%s'/leaves.nii -t -1.0"%(swd,swd))
print commands.getoutput("AimsMesh -i '%s'/%s.nii" %(swd, what))
print commands.getoutput("AimsMeshes2Graph -i '%s'/%s*.mesh -o '%s'/%s_.arg" %(swd,what,swd,what))
