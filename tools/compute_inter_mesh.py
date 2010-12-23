import numpy as np
#import nibabel.gifti.gifti as nbg
from gifti import loadImage, saveImage, GiftiImage_fromTriangles
import enthought.mayavi.mlab as mlab

# -----------------------------------------------------------
# --------- Paths and Parameters ----------------------------
# -----------------------------------------------------------
subjects = ['s12069', 's12300', 's12401', 's12431', 's12508', 's12532', 's12539', 's12562','s12590', 's12635', 's12636', 's12898', 's12081', 's12165', 's12207', 's12344', 's12352', 's12370', 's12381', 's12405', 's12414', 's12432']
nsubj = len(subjects)
db_path = '/volatile/subjects_database'

# -----------------------------------------------------------
# --------- Routines definition -----------------------------
# -----------------------------------------------------------
def compute_intermediate_mesh(white_mesh_path, gray_mesh_path,
                              normalized_white_mesh_path, output_mesh_path):
    """

    """
    # compute the cortical thickness for each vertex
    white_mesh = loadImage(white_mesh_path)
    gray_mesh = loadImage(gray_mesh_path)
    white_mesh_coord = white_mesh.getArrays()[0].getData()
    gray_mesh_coord = gray_mesh.getArrays()[0].getData()
    thickness_coord = gray_mesh_coord - white_mesh_coord
    thickness = np.sqrt((thickness_coord**2).sum(1))
    
    normalized_white_mesh = loadImage(normalized_white_mesh_path)
    vertices = normalized_white_mesh.getArrays()[0].getData()
    x, y, z = vertices.T
    triangles = normalized_white_mesh.getArrays()[-1].getData()
    # recompute normals with Mayavi
    mayavi_mesh = mlab.pipeline.triangular_mesh_source(
        x, y, z, triangles, transparent=False, opacity=1., figure=False)
    mayavi_normals = mlab.pipeline.poly_data_normals(mayavi_mesh)
    mayavi_normals.filter.splitting = False
    mayavi_normals.update_data()
    # get the normals
    point_data = mayavi_normals.outputs[0]
    recomputed_normals = point_data.point_data.normals.to_array()

    # compte and create output mesh
    normals_norms = np.sqrt((recomputed_normals**2).sum(1))
    white_mesh_normalized_coord_delta = \
        (recomputed_normals * np.tile(thickness/2., 3).reshape((3,-1)).T) \
        / np.tile(normals_norms, 3).reshape((3,-1)).T
    # write output mesh

    output_mesh = GiftiImage_fromTriangles(
        vertices + white_mesh_normalized_coord_delta, triangles)
    saveImage(output_mesh, output_mesh_path)

    return

# -----------------------------------------------------------
# --------- Script ------------------------------------------
# -----------------------------------------------------------
for s in subjects:
    print s
    toto = compute_intermediate_mesh(
        '%s/%s/surf/lh.r.white.gii' %(db_path, s),
        '%s/%s/surf/lh.r.pial.gii' %(db_path, s),
        '%s/%s/surf/lh.r.white.normalized.gii' %(db_path, s),
        '%s/%s/surf/lh.r.mean.normalized.gii' %(db_path, s)
        )
    titi = compute_intermediate_mesh(
        '%s/%s/surf/rh.r.white.gii' %(db_path, s),
        '%s/%s/surf/rh.r.pial.gii' %(db_path, s),
        '%s/%s/surf/rh.r.white.normalized.gii' %(db_path, s),
        '%s/%s/surf/rh.r.mean.normalized.gii' %(db_path, s)
        )
