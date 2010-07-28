import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from nipy.io.imageformats import load

from database_archi import *

# -----------------------------------------------------------
# --------- Script starts -----------------------------------
# -----------------------------------------------------------

points = np.array([[ 53.36233902, -12.36352539, -10.821661  ],
                   [ 48.66433716,  -9.39040279,  -5.0609808 ],
                   [ 58.8147583 ,  -7.41517019,  -7.60639238],
                   [ 54.48002625,  -5.92973137,  -8.36483383],
                   [ 52.96157074, -20.89279366,  -1.66167939],
                   [ 63.88621902, -28.02826691, -11.92555618],
                   [ 58.55256653, -17.23157501,  -6.51361465],
                   [ 55.60184479,   1.53732455,  -5.11147547],
                   [ 57.47002411, -28.75539207,  -7.428267  ],
                   [ 60.03553391, -31.07783318, -16.27619362],
                   [ 56.28676605,  18.48613167,  21.51802254],
                   [ 57.16527939,   7.23125887,  23.21502113],
                   [ 58.08297348,  -3.2823174 , -17.19317436],
                   [ 58.28199768, -46.93468857,  -6.94407892],
                   [ 51.55797958, -41.14827728,  -9.77885914],
                   [ 52.26821518,   4.67607498,  -1.40015888],
                   [ 58.39663315, -16.34000015,   7.62653446],
                   [ 46.5516777 , -26.21550179,  -7.27934074],
                   [-46.61600876, -20.56976318,   4.43893051],
                   [-51.86567688, -23.73220062,   1.82313955],
                   [-45.50168228, -27.93459511,  -1.00070846],
                   [-44.67000961, -12.41861629,   4.72936916],
                   [-35.65114212, -16.53756332,   3.40436125],
                   [-54.76918793, -30.05534363,   1.59474063],
                   [-30.89898491, -33.929039  ,   4.38110447],
                   [-48.5066185 , -10.33981323,  -6.58376169]])
sphere_center = points[13]

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
r = np.arange(0, 150, 0.1)
F = -np.ones(r.size)
for i in np.arange(r.size):
    nb_inside = np.where(dist < r[i])[0].shape[0]
    F[i] = float(nb_inside) / float(nb_voxels)

plt.figure(1)
plt.title('Repartition function')
plt.plot(r, F)

"""
### Estimate the density with Parzen-Rosenblatt
# => Cross validation to choose h
ratio = 0.2
nb_obs_per_sample = np.floor(ratio * nb_voxels)
nb_samples = int(np.floor(nb_voxels / nb_obs_per_sample))
n = nb_samples * nb_obs_per_sample
potential_h = np.arange(1.2, 2., 0.01)
# try with severals permutations of the data
nb_part = 1
criterion = 1000*np.ones((nb_part, potential_h.size))
for p in np.arange(nb_part):
    print "Part", p
    sys.stdout.flush()
    obs_tot = np.random.permutation(dist)[0:n]
    # test all the potential h
    for i in np.arange(potential_h.size):
        h = potential_h[i]
        print " h = %g" %h
        sys.stdout.flush()
        r = np.arange(0, 150, h)
        f = np.zeros((nb_samples, r.size+1))
        expectation = 0.
        # learn a density for each sample
        for j in np.arange(nb_samples):
            obs = obs_tot.copy()
            nobs = obs[j*nb_obs_per_sample:(j+1)*nb_obs_per_sample]
            obs = np.concatenate((obs[0:int(j*nb_obs_per_sample)-1],
                                  obs[int((j+1)*nb_obs_per_sample):obs.size]))
            for k in np.arange(r.size-1):
                mask = np.where(obs > r[k])
                nb_inside = np.where(obs[mask] < r[k+1])[0].shape[0]
                f[j,k] = float(nb_inside) / (h*(n - nb_obs_per_sample))
            # find closest r for each x in nobs
            r_aux = np.tile(r, (nb_obs_per_sample,1))
            nobs_aux = np.tile(nobs, (r.size,1))
            closest_aux = nobs_aux.T - r_aux
            closest = np.argmin(np.abs(closest_aux), 1)
            # Expectation is the mean of density value taken
            # for observations which have not been used for
            # the density computation
            expectation += f[int(j),closest].sum()
        expectation /= float(n)
        f_tot = np.zeros(r.size)
        for k in np.arange(r.size-1):
            mask = np.where(obs_tot > r[k])
            nb_inside = np.where(obs_tot[mask] < r[k+1])[0].shape[0]
            f_tot[k] = float(nb_inside) / (h*n)
        criterion[p,i] = h*(f_tot**2).sum() - 2 * expectation
criterion_mean = np.mean(criterion, 0)
best_h = potential_h[np.argmin(criterion_mean)]
#plt.figure(2)
#plt.title("Cross validation criterion")
#plt.plot(potential_h, criterion_mean)

# Finally plot the density function with the best h
h = best_h
nb_voxels = rcoord.shape[0]
dist = np.sqrt(((rcoord[:,0:3] - sphere_center)**2).sum(1))
r = np.arange(0, 150, h)
f = np.zeros(r.size)
for i in np.arange(r.size-1):
    mask = np.where(dist > r[i])
    nb_inside = np.where(dist[mask] < r[i+1])[0].shape[0]
    f[i] = float(nb_inside) / (h*float(nb_voxels))

#plt.figure(3)
#plt.title("Density estimation")
#plt.plot(r, f)
"""

"""
# Fit a beta law
def my_beta_distribution(x, a, b):
    return (1/sp.special.beta(a,b)) * (x**(a-1)) * ((1-x)**(b-1))

popt, pcov = sp.optimize.curve_fit(my_beta_ditribution, r, f)
plt.plot(r, my_beta_distribution(r, popt[0], popt[1]))
"""

### Estimate the density with a guassian kernel
n = dist.size
sigma = 1.05 * np.std(dist) * n**(-0.2)
x = np.arange(0., 150., 0.1)
kernel_arg = (np.tile(x, (n,1)).T - dist) / sigma
fh = ((1/np.sqrt(2*np.pi)) * np.exp(-0.5*kernel_arg**2)).sum(1) / (n*sigma)

plt.figure(4)
#plt.title('Gaussian Kernel estimated density (sigma = %g)'%sigma)
plt.plot(x, fh)
#plt.subplots_adjust(bottom=0.2)
#plt.legend(('PR (h=%g)' %h, 'Gaussian Kernel (sigma = %g)' %sigma),
#            loc='lower center', bbox_to_anchor = (0.5, -0.25), shadow=True)

plt.show()
