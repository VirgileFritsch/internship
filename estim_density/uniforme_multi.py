import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from nipy.io.imageformats import load
from gifti import loadImage

from database_archi import *

# -----------------------------------------------------------
# --------- Parameters --------------------------------------
# -----------------------------------------------------------

brain_size = 150.
step = 0.1
nb_points = 30

# -----------------------------------------------------------
# --------- Script starts -----------------------------------
# -----------------------------------------------------------

### Load cortical surface for the subject under study and randomly select
### some points over it
# load left hemisphere data
lmesh = loadImage(lmesh_path_gii)
c, n, t  = lmesh.getArrays()
lvertices = c.getData()
# load right hemisphere data
rmesh = loadImage(rmesh_path_gii)
c, n, t  = rmesh.getArrays()
rvertices = c.getData()
# points random selection
points = np.random.permutation(
    np.concatenate((lvertices, rvertices)))[0:nb_points]

### Estimate a density function for each point
fid = 0
x = np.arange(0., brain_size, step)
fh = np.zeros((points.shape[0], x.size))
for sphere_center in points:
	### Load the mask and its points coordinates
	brain_mask = load(brain_mask_path)
	affine = brain_mask.get_affine()
	data = brain_mask.get_data() != 0.
	ijk = np.array(np.where(data)).T
	nvox = ijk.shape[0]
	rcoord = np.dot(np.hstack((ijk,np.ones((nvox,1)))), affine.T)
	
	### Estimate the repartition function by finding
	### how many points are both in the brain and in a sphere centered at
	### the observation point.
	nb_voxels = rcoord.shape[0]
	dist = np.sqrt(((rcoord[:,0:3] - sphere_center)**2).sum(1))
	r = np.arange(0., brain_size, step)
	F = -np.ones(r.size)
	for i in np.arange(r.size):
	    nb_inside = np.where(dist < r[i])[0].shape[0]
	    F[i] = float(nb_inside) / float(nb_voxels)
	
	plt.figure(1)
	plt.title('Repartition function')
	plt.plot(r, F)
	
	
	### Estimate the density with a gaussian kernel
	n = dist.size
	sigma = 1.05 * np.std(dist) * n**(-0.2)
	kernel_arg = (np.tile(x, (n,1)).T - dist) / sigma
        fh[fid,:] = ((1/np.sqrt(2*np.pi)) * \
                     np.exp(-0.5*kernel_arg**2)).sum(1) / \
                    (n*sigma)
	
	plt.figure(4)
        plt.plot(x, fh[fid,:])
        fid += 1

plt.figure(5)
plt.title("Density (mean)")
fh_mean = np.mean(fh, 0)
plt.plot(x, fh_mean)

### Quartic (biweight) kernel least-square approximation
kernel_mean = np.amax(x) / 2.
u = (x.copy() - kernel_mean)
u /= np.amax(np.abs(u))
# quartic kernel
Ku = 15./16. * (1.-u**2)**2
Ku /= (Ku.sum() * step)
# tricube kernel
Ku2 = 35./32. * (1.-u**2)**3
Ku2 /= (Ku2.sum() * step)

plt.plot(x, Ku, color='red')

plt.show()
