"""
Script generating a simulated set of subjects having the same 3D blob.
Every subject has a corresponding 2D blob and some of them have a
projection artifact.

Author: Virgile Fritsch, 2010

"""

SHOW_SIMUL = False

import sys, os
import numpy as np

from nipy.neurospin.glm_files_layout import tio
from nipy.io.imageformats import load, Nifti1Image, save
import nipy.neurospin.graph.field as ff
import nipy.neurospin.graph.graph as fg
from nipy.neurospin.viz_tools.maps_3d import affine_img_src
from scikits.learn import ball_tree
from gifti import loadImage

if SHOW_SIMUL:
    import enthought.mayavi.mlab as mlab

# -----------------------------------------------------------
# --------- Paths -------------------------------------------
# (Reference is subject s12069)
# -----------------------------------------------------------
#from database_archi_simul import *

simul_root_path = '/volatile/virgile/internship/simulation'
simul_id = 'sim1'
simul_path = '%s/%s' %(simul_root_path, simul_id)
#----- Path to the reference nii file
ref_nii_path = '%s/ref/ref.nii' %(simul_path)

#----- Path to the reference hemisphere mesh
rmesh_path_gii = '%s/ref/surf/rh.r.white.normalized.gii' %(simul_path)

#----- Output path for simulated 3D blob
simul_3D_path = '%s/simul_3D.nii' %(simul_path)

#----- Output path for simulated 2D blobs
simul_2D_rtex_path = '%s/simul_2D_right.tex' %(simul_path)
simul_2D_ltex_path = '%s/simul_2D_left.tex' %(simul_path)

# -----------------------------------------------------------
# --------- Parameters --------------------------------------
# -----------------------------------------------------------

#----- Vertices id corresponding to the artifacts reference locations
artifact_id = 7458
artifact2_id = 36817

#----- Variability on main activations locations
sigma = 5.
#----- Variability on artifacts locations
sigma_artifact = 5.

#----- Create wide blobs (i.e. not a unique point)
LARGE_BLOBS = True

if LARGE_BLOBS:
    #----- Mean and std deviation of main activations sizes
    mu_blob_wideness = 12.
    sigma_blob_wideness = 4.
    #----- Mean and std deviation of artifacts sizes
    mu_artifact_wideness = mu_blob_wideness / 5.
    sigma_artifact_wideness = mu_artifact_wideness / 3.

if SHOW_SIMUL:
#----- Texture for mayavi visualization, can be:
# - "ref", to see main activation and artifacts locations
# - "f1", "f2", ... , "fn", to see one particular simulated subject map
# - "global" (or everything else), to see a superposition of all maps
    mayavi_tex_type = "global"
    mayavi_rtex = None  #don't change this

#------------------------------------------------------------------
def mesh_to_graph(vertices, poly):
    """
    This function builds an fff graph from a mesh
    (Taken from nipy mesh_processing.py but removed the aims dependancy)
    """
    V = len(vertices)
    E = poly.shape[0]
    edges = np.zeros((3*E,2))
    weights = np.zeros(3*E)

    for i in range(E):
        sa = poly[i,0]
        sb = poly[i,1]
        sc = poly[i,2]
        
        edges[3*i] = np.array([sa,sb])
        edges[3*i+1] = np.array([sa,sc])
        edges[3*i+2] = np.array([sb,sc])    
            
    G = fg.WeightedGraph(V, edges, weights)

    # symmeterize the graph
    G.symmeterize()

    # remove redundant edges
    G.cut_redundancies()

    # make it a metric graph
    G.set_euclidian(vertices)

    return G
#------------------------------------------------------------------

### Simulate 3D blob
# Read input ref nifti image
input_image = load(ref_nii_path)
input_data = input_image.get_data()
affine = input_image.get_affine()
# New data (hard defined)
new_data = np.asarray(input_data).copy()
new_data[:,:,:] = -1.
new_data[1:4,28:31,19:22] = 5.
new_data[1:4,31:35,17:20] = 5.
blob3D_center = np.dot([2., 31., 19.], affine[:3,:3]) + affine[:3,3]
# Build and write simulated 3D blob nifti image
output_image = Nifti1Image(new_data, affine)
save(output_image, simul_3D_path)

# Display simulated 3D blob with Mayavi
if SHOW_SIMUL:
    mlab.figure(1, bgcolor=(0.,0.,0.))
    blobs3D_mayavi_src = affine_img_src(new_data, affine)
    contour = mlab.pipeline.contour(blobs3D_mayavi_src)
    contour = mlab.pipeline.poly_data_normals(contour)
    contour.filter.splitting = False
    contour.update_data()
    surface = mlab.pipeline.surface(contour, color=(1., 1., 1.))
    surface.actor.property.specular = 0.3405
    surface.actor.property.specular_power = 21.3
    mlab.points3d(blob3D_center[0], blob3D_center[1], blob3D_center[2],
                  scale_factor=1, color=(1.,1.,1.))
    
### Find nearest neighbor on the surface
# load right hemisphere data
rmesh = loadImage(rmesh_path_gii)
if len(rmesh.getArrays()) == 2:
    c, t  = rmesh.getArrays()
elif len(rmesh.getArrays()) == 3:
    c, n, t  = rmesh.getArrays()
else:
    raise Exception("Could not read gifti data")
rvertices = c.getData()
rtriangles = t.getData()
rgraph = mesh_to_graph(rvertices, rtriangles)
# find nearest
search_aux = ball_tree.BallTree(rvertices)
nearest = search_aux.query(blob3D_center)
nearest_id = nearest[1].astype(int)
# write ref texture for simulated 2D blobs
ref_rtex = -np.ones(rvertices.shape[0])
ref_rtex[nearest_id] = 1.
output_ref_rtex = tio.Texture(simul_2D_rtex_path, data=ref_rtex)
output_ref_rtex.write()
if SHOW_SIMUL and mayavi_tex_type == "ref":
    mayavi_rtex = ref_rtex.copy()

### Build 22 textures with an activation located around the nearest
### neighbor we just found
dist = rgraph.dijkstra(nearest_id)
dist_artifact1 = rgraph.dijkstra(artifact_id)
dist_artifact2 = rgraph.dijkstra(artifact2_id)
global_rtex = -np.ones(rvertices.shape[0])  #summary texture
subjects_with_artifact = []
for i in range(1,23):
    print "Generating texture for f%d" %i
    if not os.path.exists('%s/f%d' %(simul_path, i)):
        os.makedirs('%s/f%d' %(simul_path, i))
    # build new texture
    new_rtex = -np.ones(rvertices.shape[0])
    # one blob located close to the reference peak
    noise = sigma * np.random.randn()
    blob_location = np.argmin(np.abs(dist-noise))
    if LARGE_BLOBS:
        neighborhood = rgraph.dijkstra(blob_location)
        # choose blob's wideness
        blob = []
        while len(blob) <= 2:
            blob_wideness = np.abs(
                sigma_blob_wideness * np.random.randn() + \
                mu_blob_wideness)
            blob = np.where(neighborhood <= blob_wideness)[0]
        new_rtex[blob] = i
        global_rtex[blob] = i
        np.savez('%s/f%d/blob_composition.npz' %(simul_path, i), blob)
    new_rtex[blob_location] = i
    global_rtex[blob_location] = i
    #global_rtex[blob_location] += 0.1
    
    # in 50% cases, an artifact is created
    if np.random.randn() > 0.:
        subjects_with_artifact.append(i)
        if np.random.randn() > 0.:
            dist_artifact = dist_artifact1
        else:
            dist_artifact = dist_artifact2
        print "  generating an artifact"
        noise = sigma_artifact * np.random.randn()
        artifact_location = np.argmin(np.abs(dist_artifact-noise))
        if LARGE_BLOBS:
            neighborhood = rgraph.dijkstra(artifact_location)
            #choose artifact's wideness
            artifact = []
            while len(artifact) <= 2:
                artifact_wideness = np.abs(
                    sigma_artifact_wideness * np.random.randn() + \
                    mu_artifact_wideness)
                artifact = np.where(neighborhood <= artifact_wideness)[0]
            new_rtex[artifact] = i
            global_rtex[artifact] = i
            np.savez('%s/f%d/artifact_composition.npz'%(simul_path,i), artifact)
        new_rtex[artifact_location] = i
        global_rtex[artifact_location] = i
        #global_rtex[artifact_location] += 0.1
    # save new texture
    new_rtex_output = tio.Texture(
        '%s/f%d/right_simul_2D.tex' %(simul_path, i), data=new_rtex)
    new_rtex_output.write()
    # save empty texture for left hemisphere
    new_ltex = -np.ones(rvertices.shape[0])
    new_ltex_output = tio.Texture(
        '%s/f%d/left_simul_2D.tex' %(simul_path, i), data=new_ltex)
    new_ltex_output.write()
    if SHOW_SIMUL and mayavi_tex_type == "f%d"%i:
        mayavi_rtex = new_rtex.copy()

if SHOW_SIMUL and not mayavi_rtex:
    mayavi_rtex = global_rtex.copy()
    
# Plot chosen texture with mayavi
if SHOW_SIMUL:
    mayavi_rmesh = mlab.triangular_mesh(
        rvertices[:,0], rvertices[:,1], rvertices[:,2], rtriangles,
        scalars=mayavi_rtex, transparent=False, opacity=1.)
    mayavi_rmesh.parent.parent.filter.feature_angle = 180.
    rhlut = mayavi_rmesh.module_manager.scalar_lut_manager.lut.table.to_array()
    rhlut[0,:] = [127, 127, 127, 255]
    mayavi_rmesh.module_manager.scalar_lut_manager.lut.table = rhlut


### Write texture output
global_rtex_output = tio.Texture(
    '%s/global_simul_2D.tex' %(simul_path), data=global_rtex)
global_rtex_output.write()

subjects_with_artifact = np.asarray(subjects_with_artifact)
np.savez('%s/subjects_with_artifact.npz' %simul_path, subjects_with_artifact)
#np.savez('%s/artifacts_location.npz' %ROOT_PATH, artifacts_location)
#np.savez('%s/blobs_location.npz' %ROOT_PATH, blobs_location)
