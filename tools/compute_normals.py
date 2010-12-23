"""
Example script to get normals from a cortical mesh

Author: Virgile Fritsch, September 2010

"""
import enthought.mayavi.mlab as mlab
from enthought.mayavi.tools.pipeline import poly_data_normals
from gifti import loadImage

# Load the mesh data (left hemisphere)
lmesh_path_gii = '/volatile/subjects_database/s12069/surf/lh.r.white.normalized.gii'
lmesh = loadImage(lmesh_path_gii)

# Extract data from it
lmesh_arrays = lmesh.getArrays()
if len(lmesh_arrays) == 2:
    c, t = lmesh_arrays
elif len(lmesh_arrays) == 3:
    c, n, t  = lmesh_arrays
    normals = n.getData()  # we might already have normals here...
else:
    raise ValueError("Error during gifti data extraction")
vertices = c.getData()
triangles = t.getData()

# ... but let's recompute it with mayavi
# load mesh in mayavi pipeline
mayavi_lmesh = mlab.pipeline.triangular_mesh_source(
    vertices[:,0], vertices[:,1], vertices[:,2],
    triangles, transparent=False, opacity=1.,
    figure=False)
# add a filter for normals extraction
mayavi_normals = mlab.pipeline.poly_data_normals(mayavi_lmesh)
mayavi_normals.filter.splitting = False  #prevent the mesh from being smoothed
mayavi_normals.update_data()
# get the normals
point_data = mayavi_normals.outputs[0]
recomputed_normals = point_data.point_data.normals.to_array()

# Recomputed normals are close to the gifti ones but there still are some
# differences which I can't explain yet.
