#from tmp_subject import *
import numpy as np

### Paths
ROOT_PATH = "/volatile/subjects_database"
SUBJECT = "s12069"
CONTRAST = "audio-video_z_map"
#------------------------------------------------------
MAIN_PATH = "%s/%s" %(ROOT_PATH, SUBJECT)
#-----------------------------------------------------#

### Cortical meshes
LMESH = "lh.r.white.normalized"
RMESH = "rh.r.white.normalized"
brain_mask_path = "%s/fMRI/default_acquisition/Minf/mask.nii" %MAIN_PATH
#------------------------------------------------------
LMESH_AIMS = "%s.mesh" %LMESH
RMESH_AIMS = "%s.mesh" %RMESH
lmesh_path_aims = '%s/surf/%s' %(MAIN_PATH, LMESH_AIMS)
rmesh_path_aims = '%s/surf/%s' %(MAIN_PATH, RMESH_AIMS)
LMESH_GII = "%s.gii" %LMESH
RMESH_GII = "%s.gii" %RMESH
lmesh_path_gii = '%s/surf/%s' %(MAIN_PATH, LMESH_GII)
rmesh_path_gii = '%s/surf/%s' %(MAIN_PATH, RMESH_GII)
#-----------------------------------------------------#

### fMRI data
FWHM3D = 5.
#------------------------------------------------------
# initial activation map
orig_data_path = "%s/fMRI/default_acquisition/loc2/waloc_corr4D.nii" %MAIN_PATH
# smoothed activation map
SIGMA3D  = FWHM3D/(2 * np.sqrt(2 * np.log(2)))
smoothed_data_path = "%s/fMRI/default_acquisition/loc2/swaloc_corr4D_FWHM%g.nii" \
                     %(MAIN_PATH, FWHM3D)
# glm processed maps
glm_data_path = '%s/fMRI/default_acquisition/glm/smoothed_FWHM%g/Contrast/%s.nii' \
                %(MAIN_PATH, FWHM3D, CONTRAST)
#-----------------------------------------------------#

### Textures
FWHM = 5.
#------------------------------------------------------
# initial activation map
orig_ltex_path = "%s/fct/loc1/lh.aloc1.tex" %MAIN_PATH
orig_rtex_path = "%s/fct/loc1/rh.aloc1.tex" %MAIN_PATH
# smoothed activation maps
smoothed_ltex_path = "%s/fct/loc1/lh.saloc1_FWHM%g.tex" %(MAIN_PATH, FWHM)
smoothed_rtex_path = "%s/fct/loc1/rh.saloc1_FWHM%g.tex" %(MAIN_PATH, FWHM)
# glm processed maps
glm_ltex_path = '%s/fct/glm/smoothed_FWHM%g/Contrast/left_%s.tex' \
                %(MAIN_PATH, FWHM, CONTRAST)
glm_rtex_path = '%s/fct/glm/smoothed_FWHM%g/Contrast/right_%s.tex' \
                %(MAIN_PATH, FWHM, CONTRAST)
#-----------------------------------------------------#

### 2D blobs
SMIN = 5
THETA = 3.3 #threshold
BLOB2D_TYPE = "blobs"
#------------------------------------------------------
BLOBS2D_TEX = "%s_%s_FWHM%g_smin%i_theta%g.tex" %(CONTRAST, BLOB2D_TYPE, FWHM,
                                                  SMIN, THETA)
BLOBS2D_LTEX = "left_" + BLOBS2D_TEX
BLOBS2D_RTEX = "right_" + BLOBS2D_TEX
BLOBS2D_SUBDIR = "experiments/surf_blobs"
blobs2D_ltex_path = '%s/%s/%s' %(MAIN_PATH, BLOBS2D_SUBDIR, BLOBS2D_LTEX)
blobs2D_rtex_path = '%s/%s/%s' %(MAIN_PATH, BLOBS2D_SUBDIR, BLOBS2D_RTEX)
#-----------------------------------------------------#

### 3D blobs
SMIN3D = 5
THETA3D = 3.3 #threshold
BLOB3D_TYPE = BLOB2D_TYPE
#------------------------------------------------------
BLOBS3D_SUBDIR = "smoothed_FWHM%g/%s_smin%i_theta%g" %(FWHM3D, CONTRAST, SMIN3D, THETA3D)
BLOBS3D_DIR = '%s/experiments/%s' %(MAIN_PATH, BLOBS3D_SUBDIR)
blobs3D_path = '%s/experiments/%s/leaves.nii' %(MAIN_PATH, BLOBS3D_SUBDIR)
#-----------------------------------------------------#

### Blobs matching output
OUTPUT_DIR = '%s/experiments/smoothed_FWHM%g/%s/results' %(MAIN_PATH, FWHM3D, CONTRAST)
lresults_output = 'left_%s_FWHM2D%g_smin2D%i_FWHM3D%g_smin3D%i.tex' %(CONTRAST, FWHM, SMIN, FWHM3D, SMIN3D)
rresults_output = 'right_%s_FWHM2D%g_smin2D%i_FWHM3D%g_smin3D%i.tex' %(CONTRAST, FWHM, SMIN, FWHM3D, SMIN3D)

### Blobs matching output (entire domain)
OUTPUT_ENTIRE_DOMAIN_DIR = '%s/experiments/smoothed_FWHM%g/%s/results_entire_domain' %(MAIN_PATH, FWHM3D, CONTRAST)
lresults_entire_domain_output = 'left_%s_FWHM2D%g_smin2D%i_FWHM3D%g_smin3D%i.tex' %(CONTRAST, FWHM, SMIN, FWHM3D, SMIN3D)
rresults_entire_domain_output = 'right_%s_FWHM2D%g_smin2D%i_FWHM3D%g_smin3D%i.tex' %(CONTRAST, FWHM, SMIN, FWHM3D, SMIN3D)

### Auxiliary results output
OUTPUT_AUX_DIR = '%s/experiments/smoothed_FWHM%g/%s/results_aux' %(MAIN_PATH, FWHM3D, CONTRAST)
lresults_aux_output = 'left_%s_FWHM2D%g_smin2D%i_FWHM3D%g_smin3D%i.tex' %(CONTRAST, FWHM, SMIN, FWHM3D, SMIN3D)
rresults_aux_output = 'right_%s_FWHM2D%g_smin2D%i_FWHM3D%g_smin3D%i.tex' %(CONTRAST, FWHM, SMIN, FWHM3D, SMIN3D)

### Auxiliary large results output
OUTPUT_LARGE_AUX_DIR = '%s/experiments/smoothed_FWHM%g/%s/results_aux_large' %(MAIN_PATH, FWHM3D, CONTRAST)
lresults_aux_large_output = 'left_%s_FWHM2D%g_smin2D%i_FWHM3D%g_smin3D%i.tex' %(CONTRAST, FWHM, SMIN, FWHM3D, SMIN3D)
rresults_aux_large_output = 'right_%s_FWHM2D%g_smin2D%i_FWHM3D%g_smin3D%i.tex' %(CONTRAST, FWHM, SMIN, FWHM3D, SMIN3D)

### Coordinates results output
OUTPUT_COORD_DIR = '%s/experiments/smoothed_FWHM%g/%s/results_coord' %(MAIN_PATH, FWHM3D, CONTRAST)
lresults_coord_output = 'left_%s_FWHM2D%g_smin2D%i_FWHM3D%g_smin3D%i.tex' %(CONTRAST, FWHM, SMIN, FWHM3D, SMIN3D)
rresults_coord_output = 'right_%s_FWHM2D%g_smin2D%i_FWHM3D%g_smin3D%i.tex' %(CONTRAST, FWHM, SMIN, FWHM3D, SMIN3D)
