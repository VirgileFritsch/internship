import numpy as np
import nibabel as nb

import tio

from nipype.utils.misc import package_check
package_check('nipy')
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, \
    TraitedSpec, File, OutputMultiPath
from nipype.utils.filemanip import split_filename
from nipy.labs.utils.design_matrix import DesignMatrix, \
    load_protocol_from_csv_file
from nipy.labs.glm import glm


class GLM2DInputSpec(BaseInterfaceInputSpec):
    input_texture_left = File(
        exists=True, desc="texture to process (left hemisphere)",
        mandatory=True)
    input_texture_right = File(
        exists=True, desc="texture to process (right hemisphere)",
        mandatory=True)
    paradigm = File(exists=True, desc="csv file corresponding to the paradigm",
                    mandatory=True)


class GLM2DOutputSpec(TraitedSpec):
    z_maps_left = OutputMultiPath(File(exists=True))
    z_maps_right = OutputMultiPath(File(exists=True))


class GLM2D(BaseInterface):
    """Performs the first-level analysis on a subject cortical surface.
    
    """
    input_spec = GLM2DInputSpec
    output_spec = GLM2DOutputSpec
    
    tr = 2.4
    nb_frames = 128
    frametimes = tr * np.arange(nb_frames)
    # Masking parameters 
    infTh = 0.4
    supTh = 0.9
    # Possible choices for hrf_model : "Canonical",
    # "Canonical With Derivative" or "FIR"
    hrf_model = "Canonical With Derivative"
    # Possible choices for drift_model : "Blank", "Cosine", "Polynomial"
    drift_model = "Cosine"
    hfcut = 128
    # GLM options
    # Possible choices : "Kalman_AR1", "Kalman", "Ordinary Least Squares"
    fit_algo = "ar1"
    fit_method = "kalman"
    
    def _run_interface(self, runtime):
        # Create functional contrasts
        Conditions = ['damier_H', 'damier_V', 'clicDaudio', 'clicGaudio', 
                      'clicDvideo', 'clicGvideo', 'calculaudio',
                      'calculvideo', 'phrasevideo', 'phraseaudio']
        contrasts = {}
        for i, cond in enumerate(Conditions):
            contrasts[cond] = np.zeros(20 + 5, dtype=np.float)
            contrasts[cond][2 * i] = 1.
        generate_localizer_contrasts(contrasts)
        self._contrasts = contrasts
        
        # Process hemisphere-wise
        for hemisphere in ['left', 'right']:
            print "%s hemisphere" % hemisphere
            if hemisphere == 'left':
                fname = self.inputs.input_texture_left
            else:
                fname = self.inputs.input_texture_right
            input_data = tio.Texture(fname).read(fname).data
            
            # Create the design matrix
            print "\tCreating the design matrix"
            paradigm = load_protocol_from_csv_file(self.inputs.paradigm)[0]
            design_matrix = DesignMatrix(
                self.frametimes, paradigm=paradigm, hrf_model=self.hrf_model,
                drift_model=self.drift_model, hfcut=self.hfcut)
            design_matrix.estimate()
            
            # Fit the glm
            print "\tFitting GLM"
            actual_glm = glm(model=self.fit_algo, method=self.fit_method)
            actual_glm.fit(input_data, design_matrix.matrix, axis=0)
            
            # Compute Contrasts
            for c in contrasts.keys():
                con_data = np.ravel(actual_glm.contrast(contrasts[c]).effect)
                # save output image
                _, base, _ = split_filename(fname)
                output_tex = tio.Texture(base + '_%s_tmap.nii' % c,
                                         data=con_data)
                output_tex.write()
    
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        # left hemisphere
        _, base, _ = split_filename(self.inputs.input_texture_left)
        outputs["z_maps_left"] = [base + '_%s_tmap.nii' % c \
                                 for c in self._contrasts.keys()]
        # right hemisphere
        _, base, _ = split_filename(self.inputs.input_texture_left)
        outputs["z_maps_right"] = [base + '_%s_tmap.nii' % c \
                                 for c in self._contrasts.keys()]
        
        return outputs


# -----------------------------------------------------------
# --------- Define some useful functions --------------------
# -----------------------------------------------------------
def generate_localizer_contrasts(contrasts):
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
    d = contrasts
    d["audio"] = d["clicDaudio"] + d["clicGaudio"] + \
                 d["calculaudio"] + d["phraseaudio"]
    d["video"] = d["clicDvideo"] + d["clicGvideo"] + \
                 d["calculvideo"] + d["phrasevideo"]
    d["left"] = d["clicGaudio"] + d["clicGvideo"]
    d["right"] = d["clicDaudio"] + d["clicDvideo"] 
    d["computation"] = d["calculaudio"] + d["calculvideo"]
    d["sentences"] = d["phraseaudio"] + d["phrasevideo"]
    d["H-V"] = d["damier_H"] - d["damier_V"]
    d["V-H"] = d["damier_V"] - d["damier_H"]
    d["left-right"] = d["left"] - d["right"]
    d["right-left"] = d["right"] - d["left"]
    d["audio-video"] = d["audio"] - d["video"]
    d["video-audio"] = d["video"] - d["audio"]
    d["computation-sentences"] = d["computation"] - d["sentences"]
    d["reading-visual"] = d["sentences"] * 2 - d["damier_H"] - d["damier_V"]
