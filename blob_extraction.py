"""
This scipt extracts the blobs from an fRMI image.

Author : Virgile Fritsch, 2010,
	 adapted from original Bertrand Thirion's script, 2009
"""
#autoindent
print __doc__

import numpy as np
import pylab as pl
import matplotlib
import scipy.ndimage as sn
import numpy as np
from nipy.io.imageformats import load, Nifti1Image, save

import nipy.neurospin.graph.field as ff
import nipy.neurospin.utils.simul_multisubject_fmri_dataset as simul
import nipy.neurospin.spatial_models.hroi as hroi

# get the image to extract the blobs from
input_path = '/data/home/virgile/virgile_internship/s12069/fMRI/acquisition/glm/not_normalized/Contrast/right-left_z_map.nii'
input_image = load(input_path)
input_data = input_image.get_data()
header = input_image.get_header()

# extract information about the image
dimx = header["dim"][1]
dimy = header["dim"][2]
dimz = header["dim"][3]

nbvox = dimx*dimy*dimz
x = np.reshape(input_data, (dimx, dimy, dimz))
beta = np.reshape(x, (nbvox, 1))

xyz = np.array(np.where(x)).T

# build the field instance
F = ff.Field(nbvox)
F.from_3d_grid(xyz, 18)
F.set_field(beta)

# compute the blobs
th = 2.36
smin = 5
affine = input_image.get_affine()
shape = input_image.get_shape()
nroi = hroi.NROI_from_field(F, affine, shape, xyz, refdim=0, th=th, smin = smin)

bmap = np.zeros(nbvox)
label = -np.ones(nbvox)

if nroi!=None:
    # compute the average signal within each blob
    nroi.set_discrete_feature_from_index('activation',beta)
    bfm = nroi.discrete_to_roi_features('activation')

    # plot the input image 
    idx = nroi.discrete_features['index']
    for k in range(nroi.k):
        bmap[idx[k]] = bfm[k]
        label[idx[k]] = k
        
label = np.reshape(label,(dimx,dimy))
bmap = np.reshape(bmap,(dimx,dimy))

aux1 = (0-x.min())/(x.max()-x.min())
aux2 = (bmap.max()-x.min())/(x.max()-x.min())
cdict = {'red': ((0.0, 0.0, 0.7), 
                 (aux1, 0.7, 0.7),
                 (aux2, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.7),
                   (aux1, 0.7, 0.0),
                   (aux2, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),
         'blue': ((0.0, 0.0, 0.7),
                  (aux1, 0.7, 0.0),
                  (aux2, 0.5, 0.5),
                  (1.0, 1.0, 1.0))}
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)

pl.figure(figsize=(12, 3))
pl.subplot(1, 3, 1)
pl.imshow(np.squeeze(x), interpolation='nearest', cmap=my_cmap)
cb = pl.colorbar()
for t in cb.ax.get_yticklabels():
    t.set_fontsize(16)
    
pl.axis('off')
pl.title('Thresholded data')

# plot the blob label image
pl.subplot(1, 3, 2)
pl.imshow(label, interpolation='nearest')
pl.colorbar()
pl.title('Blob labels')

# plot the blob-averaged signal image
aux = 0.01#(th-bmap.min())/(bmap.max()-bmap.min())
cdict = {'red': ((0.0, 0.0, 0.7), (aux, 0.7, 0.7), (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.7), (aux, 0.7, 0.0), (1.0, 1.0, 1.0)),
         'blue': ((0.0, 0.0, 0.7), (aux, 0.7, 0.0), (1.0, 0.5, 1.0))}
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict, 256)

pl.subplot(1, 3, 3)
pl.imshow(bmap, interpolation='nearest', cmap=my_cmap)
cb = pl.colorbar()
for t in cb.ax.get_yticklabels():
    t.set_fontsize(16)
pl.axis('off')
pl.title('Blob average')
pl.show()
