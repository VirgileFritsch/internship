"""
Viewing a texture on a given mesh

"""
import numpy as np
import nibabel.gifti.gifti as nbg
from nipy.neurospin.glm_files_layout import tio

import enthought.mayavi.mlab as mlab

lmesh_path = "/volatile/subjects_database/s12069/surf/lh.r.white.gii"
rmesh_path = "/volatile/subjects_database/s12069/surf/rh.r.white.gii"
ltex_path = "/volatile/virgile/internship/simulation/sim1/f1/matching/results_fcoord_level001/left.tex"
rtex_path = "/volatile/virgile/internship/simulation/sim1/f1/matching/results_fcoord_level001/right.tex"

def plot_mesh(mesh_path, mesh_name, tex_data):
    """Plots a mesh (with a given texture) with Mayavi
    
    """
    # read mesh
    mesh = nbg.loadImage(mesh_path)
    c = mesh.darrays[0]
    t = mesh.darrays[-1]
    x, y, z = c.data.T
    triangles = t.data
    # plot mesh
    mayavi_mesh = mlab.triangular_mesh(
        x, y, z, triangles, transparent=False, opacity=1.,
        name=mesh_name, scalars=tex_data)
    # make it nice
    mayavi_mesh.parent.parent.filter.feature_angle = 180.
    # change colormap (-1 values => gray)
    lut = np.asarray(mayavi_mesh.module_manager.scalar_lut_manager.lut.table)
    lut[0,:] = np.array([127,127,127,255])
    mayavi_mesh.module_manager.scalar_lut_manager.lut.table = lut
    return

if __name__ == "__main__":
    rtex = tio.Texture(rtex_path).read(rtex_path).data
    plot_mesh(rmesh_path, "RightHemishpere", rtex)
    ltex = tio.Texture(ltex_path).read(ltex_path).data
    plot_mesh(lmesh_path, "LeftHemishpere", ltex)
