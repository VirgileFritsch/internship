"""
Script to match 2D and 3D blobs.

Assuming that surface- and volume-based analyses have been made
independantly over the same subject brain, and that hierarchical blobs
extraction have been performed in both case, this script computes a
probability that a given 2D blob corresponds to the same functionnal
region that a given 3D blob. Thus, (non-connex) functional regions are
reconstructed on the cortical surface. This is done for every 2D-3D
couple of blobs, regarding on how distant are the blobs. For more
detail about the performed algorithm, see ?

Authors: Virgile Fritsch and Bertrand Thirion, 2010

"""

SHOW_MATCHING = True
SHOW_HIERARCHY = False
SHOW_OUTPUTS = True

import sys, copy, os
import numpy as np
import scipy as sp
from scipy import ndimage
from nipy.neurospin.glm_files_layout import tio
from nipy.io.imageformats import load
from nipy.neurospin.spatial_models.roi import DiscreteROI, MultipleROI
from nipy.neurospin.viz_tools.maps_3d import affine_img_src
import nipy.neurospin.graph.field as ff
import nipy.neurospin.graph.graph as fg
import nipy.neurospin.clustering.clustering as cl
from nipy.neurospin.spatial_models import hroi
from gifti import loadImage
from scikits.learn import ball_tree
from nipy.neurospin.spatial_models.discrete_domain import domain_from_mesh
from nipy.neurospin.spatial_models.discrete_domain import domain_from_image

if SHOW_MATCHING:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.text as mtxt

if SHOW_OUTPUTS:
    import enthought.mayavi.mlab as mayavi
    from enthought.tvtk.api import tvtk

from blob import Blob2D, Blob3D

# -----------------------------------------------------------
# --------- Paths and Parameters ----------------------------
# -----------------------------------------------------------
from database_archi import *



#----- Choose 3D blobs to show (from 1 to ..., -3 to show everything)
blobs3D_to_show = [-3]

#----- Choose 2D blobs to show
#----- (from 0 to ..., -3 to show everything, -2 for original blobs)
blobs2D_to_show = [-3]
blobs2D_to_show_bckup = np.array(blobs2D_to_show).copy()

#----- Choose kind of texture to plot with mayavi
mayavi_outtex_type = "default"

#----- Model parameters
GAMMA = 0.9  #a priori probability
SIGMA = 5.  #bandwith

#----- Thresholds for blobs matching
threshold_sure = 7.0e-1
threshold_maybe = 2.5e-1


#TEMP----------------------------------------------------------
# debug mode ?
DEBUG = True

# write textures ?
WRITE = False

# special case when matching blobs of group analysis results
if SUBJECT == "group":
    GA_TYPE = "vrfx"
    r_path = "/data/home/virgile/virgile_internship"
    m_path = "%s/group_analysis/smoothed_FWHM0" % r_path
    lmesh_path_gii = "%s/group_analysis/surf/lh.r.white.normalized.gii" % r_path
    rmesh_path_gii = "%s/group_analysis/surf/rh.r.white.normalized.gii" % r_path
    glm_ltex_path = "%s/left_%s_%s.tex" % (m_path, GA_TYPE, CONTRAST)
    glm_rtex_path = "%s/right_%s_%s.tex" % (m_path, GA_TYPE, CONTRAST)
    glm_data_path = "%s/%s_%s.nii" % (m_path, GA_TYPE, CONTRAST)
    OUTPUT_DIR = "%s/%s/results" %(m_path, CONTRAST)
    lresults_output = "left_%s_%s_results.tex" % (GA_TYPE, CONTRAST)
    rresults_output = "right_%s_%s_results.tex" %(GA_TYPE, CONTRAST)
    OUTPUT_ENTIRE_DOMAIN_DIR = "%s/%s/results_entire_domain" %(m_path, CONTRAST)
    lresults_entire_domain_output = "left_%s_%s_results_entire_domain.tex" % (GA_TYPE, CONTRAST)
    rresults_entire_domain_output = \
        "right_%s_%s_results_entire_domain.tex" % (GA_TYPE, CONTRAST)
    OUTPUT_AUX_DIR = "%s/%s/results_aux" %(m_path, CONTRAST)
    lresults_aux_output = "left_%s_%s_results_aux.tex" \
                          % (GA_TYPE, CONTRAST)
    rresults_aux_output = "right_%s_%s_results_aux.tex" \
                          % (GA_TYPE, CONTRAST)
    OUTPUT_LARGE_AUX_DIR = "%s/%s/results_aux_large" %(m_path, CONTRAST)
    lresults_aux_large_output = "left_%s_%s_results_aux_large.tex" \
                          % (GA_TYPE, CONTRAST)
    rresults_aux_large_output = "right_%s_%s_results_aux_large.tex" \
                          % (GA_TYPE, CONTRAST)
    OUTPUT_COORD_DIR = "%s/%s/results_coord" %(m_path, CONTRAST)
    lresults_coord_output = "left_%s_%s_results_coord.tex" \
                            % (GA_TYPE, CONTRAST)
    rresults_coord_output = "right_%s_%s_results_coord.tex" \
                            % (GA_TYPE, CONTRAST)
    blobs3D_path = "%s/blobs3D_%s/leaves.nii" % (m_path, CONTRAST)
#TEMP------------------------------------------------------------

# colors
lut_colors = np.array(
    [[64,0,64,255], [128,0,64,255], [192,0,64,255], [64,0,128,255],
     [128,0,128,255], [192,0,128,255], [64,0,192,255], [128,0,192,255],
     [192,0,192,255], [0,64,64,255], [0,128,64,255], [0,192,64,255],
     [0,64,128,255], [0,128,128,255], [0,192,128,255], [0,64,192,255],
     [0,128,192,255], [0,192,192,255], [64,64,64,255], [128,64,64,255],
     [192,64,64,255], [64,64,128,255], [128,64,128,255], [192,64,128,255],
     [64,64,192,255], [128,64,192,255], [192,64,192,255], [64,128,64,255],
     [64,192,64,255], [64,64,128,255], [64,128,128,255], [64,192,128,255],
     [64,64,192,255], [64,128,192,255], [64,192,192,255],
     [192,64,64,255], [192,128,64,255], [192,192,64,255],
     [192,64,128,255], [192,128,128,255], [192,192,128,255],
     [192,64,192,255], [192,128,192,255],                 
     [64,64,64,255], [128,64,64,255], [192,64,64,255],
     [64,64,128,255], [128,64,128,255], [192,64,128,255],
     [64,64,192,255], [128,64,192,255], [192,64,192,255],
     [64,128,64,255], [128,128,64,255], [192,128,64,255],
     [64,128,128,255],                  [192,128,128,255],
     [64,128,192,255], [128,128,192,255], [192,128,192,255],
     [128,64,64,255], [128,128,64,255], [128,192,64,255],
     [128,64,128,255], [128,128,128,255], [128,192,128,255],
     [128,64,192,255], [128,128,192,255], [128,192,192,255],
     [64,0,0,255], [128,0,0,255], [192,0,0,255],
     [0,64,0,255], [0,128,0,255], [0,128,0,255],
     [0,0,64,255], [0,0,128,255], [0,0,192,255],
     [64,64,0,255], [128,64,0,255], [192,64,0,255],
     [64,128,0,255], [128,128,0,255], [192,128,0,255],
     [64,192,0,255], [128,192,0,255], [192,192,0,255]],
    dtype=int)
lut_colors = np.tile(lut_colors, (2,1))

    


# -----------------------------------------------------------
# --------- Routines definition -----------------------------
# -----------------------------------------------------------

def compute_distances(blobs3D_list, blobs2D_list):
    """Computes every distances between 3D blobs and 2D blobs.

    Each 2D blob is reduced to an estimate of its center (as the mean
    of all its vertices). We then look for the min distance between
    this center and every vertices of a given 3D blob as the distance
    between the two blobs.

    """
    # use every point in each 3D blob
    blobs3D_vertices = []
    for blob in blobs3D_list:
        blobs3D_vertices.append(blob.vertices)
    # use 2D blobs centers
    blobs2D_centers = np.zeros((len(blobs2D_list), 3))
    i = 0
    for blob in blobs2D_list:
        blobs2D_centers[i,:] = blob.center
        i += 1
    return compute_distances_aux(blobs3D_vertices, blobs2D_centers)


def compute_distances_aux(blobs3D_vertices, blobs2D_centers):
    """Auxiliary function to compute the distances between 2D and 3D blobs.

    dist : the distances

    """
    nb_2Dblobs = len(blobs2D_centers)
    nb_3Dblobs = len(blobs3D_vertices)
    dist = np.zeros((nb_2Dblobs, nb_3Dblobs))
    for k in np.arange(nb_3Dblobs):
        vertices = blobs3D_vertices[k]
        blob3D_vertices_aux = np.tile(vertices.reshape(-1,1,3),
                                  (1,nb_2Dblobs,1))
        blobs2D_centers_aux = np.tile(blobs2D_centers,
                                  (vertices.shape[0],1,1))
        dist_all = ((blob3D_vertices_aux - blobs2D_centers_aux)**2).sum(2)
        dist[:,k] = np.amin(dist_all, 0)
    
    return dist

def compute_association_proba(blobs2D_list, nb_3Dblobs, gamma, sigma, dist):
    nb_2Dblobs = len(blobs2D_list)
    
    dist2 = dist.copy()
    
    gamma = float(gamma)
    ## H0 (no link)
    phi_all = phi(dist2)
    probaH0 = np.tile(phi_all.prod(1),(phi_all.shape[1],1)).T
    ## H1 (link)
    # "exponential" part
    dist2_exp = np.exp(-0.5 * (dist2 / sigma)**2)
    Zi = 2. / (sigma * np.sqrt(2. * np.pi))
    proba_exp = dist2_exp * Zi
    # "volumic repartition" part
    phi_all[phi_all == 0.] = 1.
    proba_rep = probaH0 / phi_all
    # combine the two parts
    probaH1 = proba_exp * proba_rep
    
    ## Final proba (bayesian inference)
    proba = np.zeros((nb_2Dblobs, nb_3Dblobs+1))
    proba[:,0:-1] = (probaH1 * gamma/nb_3Dblobs) / \
                    ((1.-gamma)*probaH0 + \
                     (gamma/nb_3Dblobs)* \
                     np.tile(probaH1.sum(1), (nb_3Dblobs,1)).T)
    proba[:,nb_3Dblobs] = (probaH0[:,0]*(1-gamma)) / \
                           ((1.-gamma)*probaH0[:,0] + \
                           (gamma/nb_3Dblobs)*probaH1.sum(1))
    
    return proba


def phi(dik):
    dik[dik > 150.] = 150.
    return (15./16.)*(1-((dik-75.)/75.)**2)**2


def plot_matching_results(proba, dist, gamma, sigma, file, blobs2D_list,
                          blobs3D_list):
    global SUBJECT, CONTRAST, threshold_sure, threshold_maybe

    nb_2Dblobs = dist.shape[0]
    nb_3Dblobs = dist.shape[1] - 1
    res_file = open(file, 'w')
    sys.stdout = res_file
    print "### SUBJECT: %s" %SUBJECT
    print "# Contrast: %s" %CONTRAST
    print "# gamma: %g\n" %gamma
    print "nb_2D_blobs = %d" %nb_2Dblobs
    print "nb_3D_blobs = %d\n" %nb_3Dblobs

    for i in np.arange(nb_2Dblobs):
        b2d = blobs2D_list[i]
        if b2d.parent:
            parent_id = b2d.parent.id
        else:
            parent_id = -1

        # Check p(ai = j | Di) for all j
        for j in np.arange(nb_3Dblobs):
            b3d = blobs3D_list[j]
            if proba[i,j] > threshold_sure:
                # i and j are linked for sure
                print "2D blob %d (%d) is thought to be" %(b2d.id, parent_id),\
                      "related to 3D blob %d at %f%%" %(b3d.id, 100*proba[i, j])
                b2d.associate_3Dblob(b3d)
            elif proba[i,j] > threshold_maybe:
                # i and j may be linked
                b2d.associate_potential(b3d)
                
        # Check p(ai = 0 | Di)
        if proba[i, nb_3Dblobs] > threshold_sure:
            # i is not linked to any 3D blob
            print "2D blob %d (%d) is not thought to be" %(b2d.id, parent_id),\
                  "related to any 3D blob at %f%%" %(100*proba[i, nb_3Dblobs])
            b2d.associate_3Dblob(Blob3D.all_blobs[0])
        elif proba[i, nb_3Dblobs] > threshold_maybe:
            # i is thought not to be linked to any 3D blob
            b2d.associate_potential(Blob3D.all_blobs[0])
        
        # Print results of checking in a file
        significant_neighbors = np.where(proba[i,:] > 5.e-2)[0]
        print "Blob 2D %d (%d):" %(b2d.id, parent_id)
        significant_neighbors_id = significant_neighbors.copy()
        for k in np.arange(len(significant_neighbors_id)):
            if significant_neighbors[k] != nb_3Dblobs:
                significant_neighbors_id[k] = \
                           blobs3D_list[significant_neighbors[k]].id
            else:
                significant_neighbors_id[k] = -1
        b2d.set_association_probas(
            np.vstack((significant_neighbors_id,
                       100.*proba[i,significant_neighbors],
                       dist[i,significant_neighbors])).T)
        print b2d.association_probas
        print ""
    
    res_file.close()
    sys.stdout = sys.__stdout__
    
    return


def load_hemisphere_data(mesh_path_gii, tex_path, hemisphere):
    """Loads an hemisphere data (mesh, texture).

    Fixme
    -----
    Not very well done because its dependant from THETA, SMIN, and
    because it uses the number of 2D blobs already instantiated.

    """
    mesh = loadImage(mesh_path_gii)
    c = mesh.getArrays()[0]
    t = mesh.getArrays()[-1]
    vertices = c.getData()
    triangles = t.getData()
    tex = tio.Texture(tex_path).read(tex_path).data

    blobs2D_tex = -np.ones(tex.size)
    old_nb_blobs = Blob2D.nb_blobs
    domain = domain_from_mesh(mesh_path_gii)
    nroi = hroi.HROI_as_discrete_domain_blobs(
        domain, tex, threshold=THETA, smin=SMIN)
    if nroi:
        for i in np.arange(nroi.k):
            vertices_id = np.where(nroi.label == i)[0]
            new_blob = Blob2D(vertices[vertices_id], vertices_id,
                              tex[vertices_id], hemisphere)
        # update associations between blobs
        parents = nroi.get_parents()
        for c, p in enumerate(parents):
            b = Blob2D.all_blobs[c+1+old_nb_blobs]
            parent = Blob2D.all_blobs[p+1+old_nb_blobs]
            if parent.id != b.id:  # avoid recursive parenthood
                b.set_parent(parent)
        
        leaves = nroi.reduce_to_leaves()
        for k in range(leaves.k):
            blobs2D_tex[leaves.label == k] =  k + old_nb_blobs

    return vertices, triangles, blobs2D_tex


# -----------------------------------------------------------
# --------- Script (part 1): IO and data handling -----------
# -----------------------------------------------------------
    
### Load left hemisphere data and construct 2D blobs hierarchy
lvertices, ltriangles, blobs2D_ltex = load_hemisphere_data(
    lmesh_path_gii, glm_ltex_path, "left")

### Load right hemisphere data and construct 2D blobs hierarchy
rvertices, rtriangles, blobs2D_rtex = load_hemisphere_data(
    rmesh_path_gii, glm_rtex_path, "right")

### Get 3D blobs hierarchy
# get data domain (~ data mask)
mask = load(brain_mask_path)
mask_data = mask.get_data()
domain3D = domain_from_image(mask)
# get data
nim = load(glm_data_path)
volume_image_shape = nim.get_data().shape
glm_data = nim.get_data()[mask_data != 0]
# construct the blobs hierarchy
nroi3D = hroi.HROI_as_discrete_domain_blobs(
    domain3D, glm_data.ravel(), threshold=THETA3D, smin=SMIN3D)
# create the right number of blobs
Blob3D(None, None, None)  # 3D blob with id=0
if nroi3D:
    blobs3D_pos = nroi3D.domain.get_coord()
    for i in np.arange(nroi3D.k):
        vertices_id = np.where(nroi3D.label == i)[0]
        vertices = blobs3D_pos[vertices_id]
        Blob3D(vertices, vertices_id, nroi3D.get_feature('signal')[i])
    # update associations between blobs
    parents = nroi3D.get_parents()
    for c, p in enumerate(parents):
        b = Blob3D.all_blobs[c+1]
        parent = Blob3D.all_blobs[p+1]
        if parent.id != b.id:  # avoid recursive parenthood
            b.set_parent(parent)

### Plot the 3D blobs
if blobs3D_to_show[0] == -3:
    blobs3D_to_show = []
    for b in Blob3D.leaves.values():
        blobs3D_to_show.append(b.id)
if SHOW_OUTPUTS:
    mayavi.figure(1, bgcolor=(0.,0.,0.))
    label_image_data = np.zeros(domain3D.topology.shape[0])
    for k in blobs3D_to_show:
        blob_center = Blob3D.all_blobs[k].compute_center()
        mayavi.points3d(blob_center[0], blob_center[1], blob_center[2],
                        scale_factor=1)
        label_image_data[nroi3D.label == k-1] = 2*k
    # define data used for texturing
    label_image = np.zeros(volume_image_shape)
    label_image[mask_data != 0] = label_image_data
    # define data used for contouring
    ref_image = np.zeros(volume_image_shape)
    ref_image[label_image != 0] = 1.
    # plot 3D blobs' contours
    src = affine_img_src(ref_image, nim.get_affine())
    # add label information for texturing (= blobs colors)
    array_id = src.image_data.point_data.add_array(
        label_image.T.ravel().astype(ref_image.dtype))
    src.image_data.point_data.get_array(array_id).name = 'labels'
    src.image_data.update()
    # plot 3D blobs' contours
    src2 = mayavi.pipeline.set_active_attribute(src, point_scalars='scalar')
    contour = mayavi.pipeline.contour(src2)
    # add a texture to contours
    contour2 = mayavi.pipeline.set_active_attribute(contour, point_scalars='labels')
    contour2 = mayavi.pipeline.poly_data_normals(contour2)
    contour2.filter.splitting = False
    contour2.update_data()
    # disable rendering for script acceleration purpose
    src.scene.disable_render = True
    surface = mayavi.pipeline.surface(contour2)
    surface.actor.property.specular = 0.3405
    surface.actor.property.specular_power = 21.3

    contour_labels = mayavi.pipeline.set_active_attribute(contour, point_scalars='labels')
    contour_labels = mayavi.pipeline.poly_data_normals(contour_labels)
    contour_labels.filter.splitting = False
    contour_labels.update_data()
    surface_labels = mayavi.pipeline.surface(contour_labels)
    surface_labels.visible = False

# -----------------------------------------------------------
# --------- Script (part 2): blobs matching -----------------
# -----------------------------------------------------------
# get the 2D blobs that are leaves
blobs2D_list = Blob2D.leaves.values()
# ? (start)
max_pos = np.zeros((len(blobs2D_list),3))
rindex = []
lindex = []
for i in np.arange(len(blobs2D_list)):
    max_pos[i,:] = blobs2D_list[i].get_max_activation_location()
    if blobs2D_list[i].hemisphere == "right":
        rindex.append(i)
    else:
        lindex.append(i)
# ? (end)

# get the 3D blobs that are leaves
blobs3D_list = []
for b in Blob3D.leaves.values():
    blobs3D_list.append(b)
nb_3D_blobs = len(blobs3D_list)
if nb_3D_blobs == 0:
    nb_3D_blobs = 1

#--------------------
#- FIRST ASSOCIATION
### Compute distances between each pair of (3D blobs)-(2D blobs centers)
dist = compute_distances(blobs3D_list, blobs2D_list)
dist_display = np.zeros((len(blobs2D_list), nb_3D_blobs+1))
dist_display[:,0:-1] = dist.copy()
dist_display = np.sqrt(dist_display)
dist_display[:,nb_3D_blobs] = -1.

dist_bckup = dist.copy()
dist_display_bckup = dist_display.copy()

### Match each 2D blob with one or several 3D blob(s)
proba = compute_association_proba(
    blobs2D_list, nb_3D_blobs, GAMMA, SIGMA, dist_display[:,0:-1])
proba[np.isnan(proba)] = 0.

proba_bckup = proba.copy()

### Post-processing the results
plot_matching_results(proba, dist_display, GAMMA, SIGMA,
                      './results/ver0/res.txt', blobs2D_list, blobs3D_list)

blobs2D_list_bckup = copy.deepcopy(blobs2D_list)

#-----------------------------------------
#- MERGING SOME 2D BLOBS INTO THEIR PARENT
### Merge 2D blobs linked to the same 3D blob in their common parent
old_proba = []
while np.any(old_proba != proba):
    old_proba = proba
    for leaf in Blob2D.leaves.values():
        if not isinstance(leaf.parent, None.__class__):
            brothers = leaf.parent.children
        else:
            brothers = []
        all_linked_to_the_same = True
        brothers_id = []
        for j in brothers:
            brothers_id.append(j.id)
            # brother is associated to the same 3D blob
            if ((j.associated_3D_blob == leaf.associated_3D_blob) and \
                (not isinstance(j.associated_3D_blob, None.__class__))):
                all_linked_to_the_same &= True
            # brother has only one potentialy associated blob (and it is
            # the which "leaf" is associated to)
            #elif ((np.shape(j.potentialy_associated)[0] == 1) and \
            #     (not isinstance(leaf.associated_3D_blob, None.__class__)) and \
            #      (leaf.associated_3D_blob in j.potentialy_associated)):
            #    all_linked_to_the_same &= True
            # maybe current leaf is not associated with a blob but has
            # a potentialy associated one
            #elif ((np.shape(leaf.potentialy_associated)[0] == 1) and \
            #     (not isinstance(j.associated_3D_blob, None.__class__)) and \
            #      (j.associated_3D_blob in leaf.potentialy_associated)):
            #    all_linked_to_the_same &= True
            # brother can be linked to another 3D blob
            else:
                all_linked_to_the_same &= False
        if (all_linked_to_the_same and \
            (not isinstance(leaf.parent, None.__class__))):
            linked_3D_blob = leaf.associated_3D_blob
            parent_blob = leaf.parent
            parent_blob.associate_3Dblob(linked_3D_blob)
            for i in brothers_id:
                parent_blob.merge_child(Blob2D.leaves[i])
  
    ### Compute distances between each pair of (3D blobs)-(new 2D blobs centers)
    dist = compute_distances(blobs3D_list, Blob2D.leaves.values())
    dist_display = np.zeros((len(Blob2D.leaves.values()), nb_3D_blobs+1))
    dist_display[:,0:-1] = dist.copy()
    dist_display = np.sqrt(dist_display)
    dist_display[:,nb_3D_blobs] = -1.
    
    ### Match each new 2D blobs with one or several 3D blob(s)
    proba = compute_association_proba(
        Blob2D.leaves.values(), nb_3D_blobs, GAMMA, SIGMA, dist_display[:,0:-1])
    proba[np.isnan(proba)] = 0.

### Post-processing the results
plot_matching_results(
    proba, dist_display, GAMMA, SIGMA, './results/ver0/new_res.txt',
    Blob2D.leaves.values(), blobs3D_list)

#-----------------------------------------
#- TRY TO FIND SOME ARTEFACTS
# retrieve all regions' composition
region = {}
region_size = {}
for b in Blob2D.leaves.values():
    if b.associated_3D_blob is not None and b.associated_3D_blob.id != 0:
        region_id = b.associated_3D_blob.id
        if region_id in region.keys():
            region[region_id].append(b.id)
            region_size[region_id] += b.vertices_id.size
        else:
            region[region_id] = [b.id]
            region_size[region_id] = b.vertices_id.size
    for l in b.potentialy_associated:
        if l.id != 0:
            region_id = l.id
            if region_id in region.keys():
                region[region_id].append(b.id)
                region_size[region_id] += b.vertices_id.size
            else:
                region[region_id] = [b.id]
                region_size[region_id] = b.vertices_id.size

# consolidate regions composition by finding possible artefacts
for r in region.keys():
    print "--- Analysis of region %d" %r
    sum_probas_region = 0.
    for blob in region[r]:
        b = Blob2D.leaves[blob]
        row = np.where(b.association_probas[:,0] == r)[0]
        b.regions_probas = b.association_probas.copy()
        b.regions_probas[row,1] *= (float(b.vertices_id.size) / region_size[r])
        sum_probas_region += b.regions_probas[row,1]
    for blob in region[r]:
        b = Blob2D.leaves[blob]
        row = np.where(b.regions_probas[:,0] == r)[0]
        b.regions_probas[row,1] /= sum_probas_region
        print "Blob %d, %g/%g, %g -> %g" %(b.id, b.vertices_id.size, \
                                           region_size[r], \
                                           b.association_probas[row,1], \
                                           b.regions_probas[row,1]*100)

#-----------------------------------------
#- DISPLAY THE FIRST RESULTS

if SHOW_MATCHING:
    def sort_list_by_link(my_list, blob3D_id):
        for i in np.arange(my_list.__len__()):
            links = my_list[i].association_probas
            max_blob = links[links[:,0] == blob3D_id,:][0,1]
            max_index = i
            for j in np.arange(i+1, my_list.__len__()):
                links = my_list[j].association_probas
                link = links[links[:,0] == blob3D_id,:][0,1]
                if (link > max_blob):
                    max_blob = link
                    max_index = j
            tmp_swap = my_list[i]
            my_list[i] = my_list[max_index]
            my_list[max_index] = tmp_swap
    
        return my_list
    
    # construct a list of associated 2D blobs for each 3D blobs
    # and sort it by their link probability value
    nested_association_lists = []
    if blobs3D_list:
        for i in np.arange(nb_3D_blobs):
            association_list = []
            for blob2D in Blob2D.leaves.values():
                if (blobs3D_list[i] in blob2D.potentialy_associated):
                    association_list.append(blob2D)
                elif (not isinstance(blob2D.associated_3D_blob,None.__class__) and \
                      blob2D.associated_3D_blob.id == blobs3D_list[i].id):
                    association_list.insert(0,blob2D)
            association_list = sort_list_by_link(association_list,
                                                 blobs3D_list[i].id)
            if association_list:
                association_list.insert(0,blobs3D_list[i].id)
                nested_association_lists.append(association_list)
     
    # set matplotlib figure basis
    fig = plt.figure(1)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.text((Blob3D.default_xpos-Blob2D.default_xpos)/2, 15,
            "Subject %s, Contrast %s, gamma=%g" %(SUBJECT, CONTRAST, GAMMA),
            horizontalalignment='center')
    ax.set_xlim(Blob2D.default_xpos-15, Blob3D.default_xpos+15)
    #ax.set_ylim(-Blob2DDisplay.spacing*np.amax([nb_linked,
    #                                            nb_3D_blobs-1]),25)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    lines_colors = ['b','g','c','m','k','y', (1.,0.5,0.), (0.5,0.5,0.), (0.,0.5,0.)]
    
    # display the associated blobs
    for l in nested_association_lists:
        blob3D_id = l[0]
        for b in l[1:]:
            b.display(ax)
        Blob3D.leaves[blob3D_id].display(ax)
        for b in l[1:]:
            probas = b.association_probas
            link = probas[probas[:,0]==blob3D_id,:][0,1]
            if link > 100*threshold_sure:
                ax.plot([b.get_xpos()+Blob2D.radius/2.,
                         Blob3D.leaves[blob3D_id].get_xpos()-Blob3D.radius/2.],
                        [b.get_ypos(),Blob3D.leaves[blob3D_id].get_ypos()],
                        color=tuple(lut_colors[blob3D_id][0:3]/255.))
            elif link > 100*threshold_maybe:
                ax.plot([b.get_xpos()+Blob2D.radius/2.,
                         Blob3D.leaves[blob3D_id].get_xpos()-Blob3D.radius/2.],
                        [b.get_ypos(),Blob3D.leaves[blob3D_id].get_ypos()],
                        '--', color=tuple(lut_colors[blob3D_id][0:3]/255.))
    
    # display 2D blobs that have no children and no association
    for blob2D in Blob2D.leaves.values():
        if (blob2D.associated_3D_blob == Blob3D.all_blobs[0]):
            blob2D.display(ax, circle_color='red')
        else:
            blob2D.display(ax, circle_color='green')
    
    if SHOW_HIERARCHY:
        # display 3D blobs hierarchy
        for blob3D in Blob3D.nodes.values():
            blob3D.display(ax)
            for child in blob3D.children:
                if child.already_displayed:
                    ax.plot([child.get_xpos()+Blob3D.radius/2.,
                            blob3D.get_xpos()-Blob3D.radius/2.],
                            [child.get_ypos(),blob3D.get_ypos()],
                            color='black')
        
        # display 2D blobs hierarchy
        for blob2D in Blob2D.nodes.values():
            blob2D.display(ax)
            for child in blob2D.children:
                if child.is_sub_blob:
                    ax.plot([child.get_xpos()-Blob2D.radius/2.,
                             blob2D.get_xpos()+Blob2D.radius/2.],
                            [child.get_ypos(),blob2D.get_ypos()],
                            color=lines_colors[blob2D.id%len(lines_colors)])
                else:
                    ax.plot([child.get_xpos()-Blob2D.radius/2.,
                             blob2D.get_xpos()+Blob2D.radius/2.],
                            [child.get_ypos(),blob2D.get_ypos()],
                            color='black')

# choose textures
ltex = blobs2D_ltex.copy()
rtex = blobs2D_rtex.copy()
if blobs2D_to_show_bckup[0] != -2.:
    if blobs2D_to_show_bckup[0] == -3.:
        blobs2D_to_show = []
        for b in Blob2D.leaves.values():
            blobs2D_to_show.append(b.id)
    ltex[:] = -1.
    rtex[:] = -1.
    for i in blobs2D_to_show:
        blob = Blob2D.all_blobs[i]
        if (not isinstance(blob.associated_3D_blob, None.__class__)):
            value = blob.associated_3D_blob.id
        else:
            value = -0.3
        if blob.hemisphere == "left":
            ltex[blob.vertices_id] = value
        else:
            rtex[blob.vertices_id] = value
mayavi_routtex = rtex
mayavi_louttex = ltex

if blobs2D_to_show_bckup[0] == -3.:
    ### Finally write output (right and left) textures
    out_dir = "%s_level%03d" %(OUTPUT_DIR, 1)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_rtex = tio.Texture("%s/%s" %(out_dir,rresults_output), data=rtex)
    if WRITE:
        output_rtex.write()
    output_ltex = tio.Texture("%s/%s" %(out_dir,lresults_output), data=ltex)
    if WRITE:
        output_ltex.write()
    
    ### Output textures with entire domain
    # fill the entire blob domain
    ltex_entire = ltex.copy()
    rtex_entire = rtex.copy()
    for b in Blob2D.nodes.values():
        if b.hemisphere == "left":
            the_tex = ltex_entire
        else:
            the_tex = rtex_entire
        for i in b.vertices_id:
            if the_tex[i] == -1:
                the_tex[i] = -0.7
    # write results
    out_dir = "%s_level%03d" %(OUTPUT_ENTIRE_DOMAIN_DIR, 1)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_entire_domain_rtex = tio.Texture("%s/%s" %(out_dir,rresults_entire_domain_output), data=rtex_entire)
    if WRITE:
        output_entire_domain_rtex.write()
    output_entire_domain_ltex = tio.Texture("%s/%s" %(out_dir,lresults_entire_domain_output), data=ltex_entire)
    if WRITE:
        output_entire_domain_ltex.write()
    
    ### Auxiliary results large domain
    all_rvertices = np.array([[],[],[]], ndmin=2).T
    all_lvertices = np.array([[],[],[]], ndmin=2).T
    all_rvertices_id = np.array([], dtype=int)
    all_lvertices_id = np.array([], dtype=int)
    for b in Blob2D.all_blobs.values():
        if b.hemisphere == "right":
            all_rvertices = np.concatenate((all_rvertices, b.vertices))
            all_rvertices_id = np.concatenate((all_rvertices_id, b.vertices_id))
        else:
            all_lvertices = np.concatenate((all_lvertices, b.vertices))
            all_lvertices_id = np.concatenate((all_lvertices_id, b.vertices_id))
    # right hemisphere cluster
    rtex_aux_large = -np.ones(rtex.shape[0])
    if len(rindex) != 0:
        rassignment = cl.voronoi(all_rvertices, max_pos[rindex])
        rtex_aux_large[all_rvertices_id] = rassignment
    # left hemisphere cluster
    ltex_aux_large = -np.ones(ltex.shape[0])
    if len(lindex) != 0:
        lassignment = cl.voronoi(all_lvertices, max_pos[lindex])
        ltex_aux_large[all_lvertices_id] = lassignment
    # write results
    out_dir = "%s_level%03d" %(OUTPUT_LARGE_AUX_DIR, 1)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_aux_large_rtex = tio.Texture("%s/%s" %(out_dir,rresults_aux_large_output), data=rtex_aux_large)
    if WRITE:
        output_aux_large_rtex.write()
    output_aux_large_ltex = tio.Texture("%s/%s" %(out_dir,lresults_aux_large_output), data=ltex_aux_large)
    if WRITE:
        output_aux_large_ltex.write()
    
    ### Auxiliary results restricted domain
    # right hemisphere cluster
    all_rblobs_vertices = rvertices[rtex != -1]
    rtex_aux = -np.ones(rtex.shape[0])
    if len(rindex) != 0:
        rassignment = cl.voronoi(all_rblobs_vertices, max_pos[rindex])
        rtex_aux[rtex != -1] = rassignment
    # left hemisphere cluster
    all_lblobs_vertices = lvertices[ltex != -1]
    ltex_aux = -np.ones(ltex.shape[0])
    if len(lindex) != 0:
        lassignment = cl.voronoi(all_lblobs_vertices, max_pos[lindex])
        ltex_aux[ltex != -1] = lassignment
    # write results
    out_dir = "%s_level%03d" %(OUTPUT_AUX_DIR, 1)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_aux_rtex = tio.Texture("%s/%s" %(out_dir,rresults_aux_output), data=rtex_aux)
    if WRITE:
        output_aux_rtex.write()
    output_aux_ltex = tio.Texture("%s/%s" %(out_dir,lresults_aux_output), data=ltex_aux)
    if WRITE:
        output_aux_ltex.write()

    ### Coordinates results
    rtex_coord = -np.ones(rtex.size)
    ltex_coord = -np.ones(ltex.size)
    max_region = {}
    max_region_location = {}
    max_region_hemisphere = {}
    for b in Blob2D.leaves.values():
        if b.associated_3D_blob is not None and \
           b.associated_3D_blob.id != 0:
            if b.associated_3D_blob.id in max_region.keys():
                if max_region[b.associated_3D_blob.id] < b.get_argmax_activation():
                    max_region[b.associated_3D_blob.id] = \
                                b.get_argmax_activation()
                    max_region_location[b.associated_3D_blob.id] = \
                                b.vertices_id[b.get_argmax_activation()]
                    max_region_hemisphere[b.associated_3D_blob.id] = \
                                b.hemisphere
            else:
                max_region[b.associated_3D_blob.id] = \
                                b.get_argmax_activation()
                max_region_location[b.associated_3D_blob.id] = \
                                b.vertices_id[b.get_argmax_activation()]
                max_region_hemisphere[b.associated_3D_blob.id] = \
                                b.hemisphere
        else:
            if b.hemisphere == "right":
                rtex_coord[b.vertices_id[b.get_argmax_activation()]] = 10.
                #rtex_coord[b.vertices_id[b.get_argmax_activation()]] = \
                #                    b.vertices_id[b.get_argmax_activation()]
            else:
                ltex_coord[b.vertices_id[b.get_argmax_activation()]] = 10.
                #ltex_coord[b.vertices_id[b.get_argmax_activation()]] = \
                #                    b.vertices_id[b.get_argmax_activation()]
    for r in max_region.keys():
        if max_region_hemisphere[r] == "right":
            rtex_coord[max_region_location[r]] = 10.
            #rtex_coord[max_region_location[r]] = max_region_location[r]
        else:
            ltex_coord[max_region_location[r]] = 10.
            #ltex_coord[max_region_location[r]] = max_region_location[r]
    # write results
    out_dir = "%s_level%03d" %(OUTPUT_COORD_DIR, 1)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_coord_rtex = tio.Texture("%s/%s" %(out_dir,rresults_coord_output), data=rtex_coord)
    if WRITE:
        output_coord_rtex.write()
    output_coord_ltex = tio.Texture("%s/%s" %(out_dir,lresults_coord_output), data=ltex_coord)
    if WRITE:
        output_coord_ltex.write()

    ### Coordinates former results
    rtex_fcoord = -np.ones(rtex.size)
    ltex_fcoord = -np.ones(ltex.size)
    for b in Blob2D.leaves.values():
        if b.hemisphere == "right":
            rtex_fcoord[b.vertices_id[b.get_argmax_activation()]] = 10.
            #rtex_fcoord[b.vertices_id[b.get_argmax_activation()]] = \
            #                        b.vertices_id[b.get_argmax_activation()]
        else:
            ltex_fcoord[b.vertices_id[b.get_argmax_activation()]] = 10.
            #ltex_fcoord[b.vertices_id[b.get_argmax_activation()]] = \
            #                        b.vertices_id[b.get_argmax_activation()]
    # write results
    out_dir = "%s_level%03d" %(OUTPUT_FCOORD_DIR, 1)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_fcoord_rtex = tio.Texture("%s/%s" %(out_dir,rresults_fcoord_output), data=rtex_fcoord)
    if WRITE:
        output_fcoord_rtex.write()
    output_fcoord_ltex = tio.Texture("%s/%s" %(out_dir,lresults_fcoord_output), data=ltex_fcoord)
    if WRITE:
        output_fcoord_ltex.write()

    if mayavi_outtex_type == "aux":
        mayavi_routtex = rtex_aux_large
        mayavi_louttex = ltex_aux
    elif mayavi_outtex_type == "aux_large":
        mayavi_routtex = rtex_aux_large
        mayavi_louttex = ltex_aux_large
    elif mayavi_outtex_type == "coord":
        mayavi_routtex = rtex_coord
        mayavi_louttex = ltex_coord
    elif mayavi_outtex_type == "fcoord":
        mayavi_routtex = rtex_fcoord
        mayavi_louttex = ltex_fcoord
    elif mayavi_outtex_type == "entire":
        mayavi_routtex = rtex_entire
        mayavi_louttex = ltex_entire
    else:
        mayavi_routtex = rtex
        mayavi_louttex = ltex


if SHOW_OUTPUTS:
    ### Mayavi Plot
    # LUT definition
    nb_colors = 2*len(Blob3D.leaves.keys()) + 4
    surface.module_manager.scalar_lut_manager.number_of_colors = nb_colors
    lut = np.asarray(surface.module_manager.scalar_lut_manager.lut.table)
    
    lut[0,:] = np.array([127,127,127,255])
    lut[1,:] = np.array([255,255,255,255])
    lut[2:4,:] = np.array([0,0,0,255])
    lut[4::2,:] = lut_colors[0:(nb_colors-4.)/2.,:]
    lut[5::2,:] = lut_colors[0:(nb_colors-4.)/2.,:]

    # tune labels display
    nb_labels = len(Blob3D.leaves.keys())
    scalar_lut_manager = surface_labels.module_manager.scalar_lut_manager
    scalar_lut_manager.lut.table = lut
    scalar_lut_manager.show_legend = True
    scalar_lut_manager.number_of_labels = nb_labels + 1
    scalar_lut_manager.use_default_range = False
    scalar_lut_manager.data_range = np.array([0.,nb_labels])
    new_width = 0.8 * nb_labels/(nb_labels + 2.)
    scalar_bar = scalar_lut_manager.scalar_bar
    scalar_bar.width = new_width
    scalar_bar.position = np.array([0.1 + 3.*(0.8-new_width)/4.,0.01])
    scalar_bar.position2 = np.array([new_width, 0.17])
    scalar_bar.label_format = '%g'
    # texture 3D blobs
    surface.module_manager.scalar_lut_manager.lut.table = lut
    surface.module_manager.scalar_lut_manager.show_legend = True
    surface.module_manager.scalar_lut_manager.use_default_range = False
    surface.module_manager.scalar_lut_manager.data_range = \
                                        np.array([-1.5,0.5+(nb_colors-4.)/2.])
    surface.module_manager.scalar_lut_manager.number_of_labels = 0
    surface.module_manager.scalar_lut_manager.use_default_name = False
    surface.module_manager.scalar_lut_manager.data_name = u''
    
    # plot left hemisphere
    mayavi_lmesh = mayavi.triangular_mesh(
        lvertices[:,0], lvertices[:,1], lvertices[:,2], ltriangles,
        scalars=mayavi_louttex, transparent=False, opacity=1.,
        name="LeftHemisphere")
    mayavi_lmesh.parent.parent.filter.feature_angle = 180.
    mayavi_lmesh.module_manager.scalar_lut_manager.lut.table = lut
    mayavi_lmesh.module_manager.scalar_lut_manager.use_default_range = False
    mayavi_lmesh.module_manager.scalar_lut_manager.data_range = \
                                        np.array([-1.,(nb_colors-4.)/2.])
    # plot right hemisphere
    mayavi_rmesh = mayavi.triangular_mesh(
        rvertices[:,0], rvertices[:,1], rvertices[:,2], rtriangles,
        scalars=mayavi_routtex, transparent=False, opacity=1.,
        name="RightHemisphere")
    mayavi_rmesh.parent.parent.filter.feature_angle = 180.
    mayavi_rmesh.module_manager.scalar_lut_manager.lut.table = lut
    mayavi_rmesh.module_manager.scalar_lut_manager.use_default_range = False
    mayavi_rmesh.module_manager.scalar_lut_manager.data_range = \
                                        np.array([-1.,(nb_colors-4.)/2.])

if SHOW_OUTPUTS:
    # enable mayavi rendering (because we have disabled it)
    src.scene.disable_render = False
if SHOW_MATCHING:
    # show matplotlib graphics
    plt.show()

