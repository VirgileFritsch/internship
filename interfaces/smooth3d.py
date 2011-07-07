import os
import nibabel as nb
import numpy as np

import scipy.ndimage as sn
from nipy.labs import as_volume_img

from nipype.utils.misc import package_check
package_check('nipy')
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, \
    TraitedSpec, traits, File
from nipype.utils.filemanip import split_filename


class Smooth3DInputSpec(BaseInterfaceInputSpec):
    raw_image = File(exists=True, desc="volume image to be smoothed",
                     mandatory=True)
    fwhm = traits.Float(desc="amount of (gaussian) smoothing", mandatory=True)


class Smooth3DOutputSpec(TraitedSpec):
    smoothed_image = File(exists=True, desc="smoothed volume image")


class Smooth3D(BaseInterface):
    """Performs a gaussian smoothing on a 4D fMRI volume image (nifti format)
    
    """
    input_spec = Smooth3DInputSpec
    output_spec = Smooth3DOutputSpec
    
    def _run_interface(self, runtime):        
        # Read input image
        fname = self.inputs.raw_image
        fwhm = self.inputs.fwhm
        input_image = as_volume_img(fname)
        input_data = input_image.get_data()
        
        # Perform smoothing
        kernel_param = np.asarray([fwhm, fwhm, fwhm, 0.]) / \
            (2.35482 * np.diag(input_image.xyz_ordered(input_image).affine))
        output_data = sn.gaussian_filter(input_data, kernel_param)
        output_image = nb.Nifti1Image(output_data, input_image.affine)

        # Save output image
        _, base, _ = split_filename(fname)
        nb.save(output_image, 's' + base + '_fwhm' + str(fwhm) + '.nii')
        
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.raw_image
        fwhm = self.inputs.fwhm
        _, base, _ = split_filename(fname)
        outputs["smoothed_image"] = os.path.abspath(
            's' + base + '_fwhm' + str(fwhm) + '.nii')
        return outputs
