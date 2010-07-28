import scipy.ndimage as sn
import numpy as np
from nipy.io.imageformats import load, Nifti1Image, save

from database_archi import *

input_image = load(orig_data_path)
input_data = input_image.get_data()
shape = input_image.get_shape()
affine = input_image.get_affine()

output_data = input_data.copy()
output_data = sn.gaussian_filter(output_data,
                                 [SIGMA3D/3., SIGMA3D/3., SIGMA3D/3., 0])

save(Nifti1Image(output_data, affine), smoothed_data_path)

