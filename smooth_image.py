"""
This script performs a gaussian smoothing on a
4D IRMf data image (nifti format).

Author: Bertrand Thirion
"""
import numpy as np
import scipy.ndimage as sn
from nipy.io.imageformats import load, Nifti1Image, save

# -----------------------------------------------------------
# --------- Parameters --------------------------------------
# -----------------------------------------------------------
from database_archi import *

# Amount of (gaussian) smoothing
#SIGMA3D = ?

# Path to the original image
#orig_data_path = ?

# Output path
#smoothed_data_path = ?

# -----------------------------------------------------------
# --------- Script starts -----------------------------------
# -----------------------------------------------------------
# Read input image
input_image = load(orig_data_path)
input_data = input_image.get_data()
shape = input_image.get_shape()
affine = input_image.get_affine()

# Init output data
output_data = input_data.copy()
# Perform smoothing
output_data = sn.gaussian_filter(output_data,
                                 [SIGMA3D/3., SIGMA3D/3., SIGMA3D/3., 0])
output_image = Nifti1Image(output_data, affine)

# Save output image
save(output_image, smoothed_data_path)

