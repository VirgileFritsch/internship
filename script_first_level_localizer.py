"""
Script that perform the first-level analysis on a subject 4D fMRI acquisition.

Author : Lise Favre, Bertrand Thirion, 2008-2010
"""
import os
from configobj import ConfigObj
from numpy import arange

from nipy.neurospin.utils.mask import compute_mask_files
from nipy.neurospin.glm_files_layout import glm_tools, contrast_tools

# -----------------------------------------------------------
# --------- Paths ----------------------------
# -----------------------------------------------------------
from database_archi import *

# Path to the subjects database
DBPath = ROOT_PATH

# Subjects list (to run the analysis over multiple subjects)
Subjects = [SUBJECT]

# Acquisition(s) and session(s) to run the script on
Acquisitions = ["default_acquisition"]
Sessions = ["loc2"]

# Model's id (e.g. models with different amount of smoothing)
model_id = "smoothed_FWHM%g" %FWHM3D

# Name of the input nifti files to process
fmri_wc = "swaloc_corr4D_FWHM%g.nii" %FWHM3D

# ---------------------------------------------------------
# -------- General Information and parameters -------------
# ---------------------------------------------------------

tr = 2.4
nb_frames = 128
frametimes = tr * arange(nb_frames)

Conditions = [ 'damier_H', 'damier_V', 'clicDaudio', 'clicGaudio', 
'clicDvideo', 'clicGvideo', 'calculaudio', 'calculvideo', 'phrasevideo', 
'phraseaudio' ]

#---------- Masking parameters 
infTh = 0.4
supTh = 0.9

#---------- Design Matrix

# Possible choices for hrf_model : "Canonical", \
# "Canonical With Derivative" or "FIR"
hrf_model = "Canonical With Derivative"

# Possible choices for drift_model : "Blank", "Cosine", "Polynomial"
drift_model = "Cosine"
hfcut = 128

#-------------- GLM options
# Possible choices : "Kalman_AR1", "Kalman", "Ordinary Least Squares"
fit_algo = "Kalman_AR1"


# ---------------------------------------------------------
# ------ Routines definition ------------------------------
# ---------------------------------------------------------

def generate_localizer_contrasts(contrast):
    """
    This utility appends standard localizer contrasts
    to the input contrast structure

    Parameters
    ----------
    contrast: configObj
        that contains the automatically generated contarsts

    Caveat
    ------
    contrast is changed in place
    
    """
    d = contrast.dic
    d["audio"] = d["clicDaudio"] + d["clicGaudio"] +\
                 d["calculaudio"] + d["phraseaudio"]
    d["video"] = d["clicDvideo"] + d["clicGvideo"] + \
                 d["calculvideo"] + d["phrasevideo"]
    d["left"] = d["clicGaudio"] + d["clicGvideo"]
    d["right"] = d["clicDaudio"] + d["clicDvideo"] 
    d["computation"] = d["calculaudio"] +d["calculvideo"]
    d["sentences"] = d["phraseaudio"] + d["phrasevideo"]
    d["H-V"] = d["damier_H"] - d["damier_V"]
    d["V-H"] =d["damier_V"] - d["damier_H"]
    d["left-right"] = d["left"] - d["right"]
    d["right-left"] = d["right"] - d["left"]
    d["audio-video"] = d["audio"] - d["video"]
    d["video-audio"] = d["video"] - d["audio"]
    d["computation-sentences"] = d["computation"] - d["sentences"]
    d["reading-visual"] = d["sentences"]*2 - d["damier_H"] - d["damier_V"]


# -----------------------------------------------------------
# --------- Launching Pipeline on all subjects, -------------
# --------- all acquisitions, all sessions      -------------
# -----------------------------------------------------------

# Treat sequentially all subjects & acquisitions
for s in Subjects:
    print "Subject : %s" % s
    
    for a in Acquisitions:
        # step 1. set all the paths
        basePath = os.sep.join((DBPath, s, "fMRI", a))
        paths = glm_tools.generate_all_brainvisa_paths(basePath, Sessions, 
                                                       fmri_wc, model_id)  
          
        misc = ConfigObj(paths['misc'])
        misc["sessions"] = Sessions
        misc["tasks"] = Conditions
        misc["mask_url"] = paths['mask']
        misc.write()

        # step 2. Create one design matrix for each session
        design_matrices = {}
        for sess in Sessions:
            design_matrices[sess] = glm_tools.design_matrix(
                paths['misc'], paths['dmtx'][sess], sess, paths['paradigm'],
                frametimes, hrf_model=hrf_model, drift_model=drift_model,
                hfcut=hfcut, model=model_id)
            
        # step 3. Compute the Mask
        # fixme : it should be possible to provide a pre-computed mask
        print "Computing the Mask"
        mask_array = compute_mask_files(paths['fmri'].values()[0][0], 
                                        paths['mask'], True, infTh, supTh)
        
        # step 4. Creating functional contrasts
        print "Creating Contrasts"
        clist = contrast_tools.ContrastList(misc=ConfigObj(paths['misc']),
                                            model=model_id)
        generate_localizer_contrasts(clist)
        contrast = clist.save_dic(paths['contrast_file'])
        CompletePaths = glm_tools.generate_brainvisa_ouput_paths( 
                        paths["contrasts"], contrast)

        # step 5. Fit the  glm for each session
        glms = {}
        for sess in Sessions:
            print "Fitting GLM for session : %s" % sess
            glms[sess] = glm_tools.glm_fit(
                paths['fmri'][sess], design_matrices[sess],
                paths['glm_dump'][sess], paths['glm_config'][sess],
                fit_algo, paths['mask'])
            
        #step 6. Compute Contrasts
        print "Computing contrasts"
        glm_tools.compute_contrasts(contrast, misc, CompletePaths,
                                    glms, model=model_id)

        
