# /!\ fixme : compatibility with gii texture instead of tex
import os
import sys
import nibabel.gifti as nbg
import numpy as np

import library_smoothing as smooth
import tio

from nipype.utils.misc import package_check
package_check('nipy')
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, \
    TraitedSpec, traits, File
from nipype.utils.filemanip import split_filename


class Smooth2DInputSpec(BaseInterfaceInputSpec):
    # left hemisphere
    raw_texture_left = File(
        exists=True, desc="texture image to be smoothed (left hemisphere)",
        mandatory=True)
    mesh_left = File(
        exists=True, desc="mesh of the subject's left hemisphere",
        mandatory=True)
    # right hemisphere
    raw_texture_right = File(
        exists=True, desc="texture image to be smoothed (right hemisphere)",
        mandatory=True)
    mesh_right = File(
        exists=True, desc="mesh of the subject's right hemisphere",
        mandatory=True)
    # operation parameters
    fwhm = traits.Float(desc="amount of (gaussian) smoothing", mandatory=True)


class Smooth2DOutputSpec(TraitedSpec):
    smoothed_texture_left = File(
        exists=True, desc="smoothed texture image (left hemisphere)")
    smoothed_texture_right = File(
        exists=True, desc="smoothed texture image (right hemisphere)")


class Smooth2D(BaseInterface):
    """Performs a diffusion smoothing on brain surface using heat equation.

    It implements the method described in the Chung & Taylor's paper :
    Diffusion Smoothing on Brain Surface via Finite Element Method.

    """
    input_spec = Smooth2DInputSpec
    output_spec = Smooth2DOutputSpec
    
    def _run_interface(self, runtime):
        # Process hemispheres separately
        fwhm = self.inputs.fwhm
        for hemisphere in ['left', 'right']:
            print "Smoothing: processing %s hemisphere:" % hemisphere
            sys.stdout.flush()
            if hemisphere == "right":
                mesh_path = self.inputs.mesh_right
                orig_tex_path = self.inputs.raw_texture_right
            else:
                mesh_path = self.inputs.mesh_left
                orig_tex_path = self.inputs.raw_texture_left
            
            ### Get information from input mesh
            # /!\ fixme : check that input_mesh is a triangular mesh
            print "  * Getting information from input mesh"
            sys.stdout.flush()
            input_mesh = nbg.giftiio.read(mesh_path)
            input_mesh_arrays = input_mesh.darrays
            if len(input_mesh_arrays) == 2:
                c, t = input_mesh_arrays
            elif len(input_mesh_arrays) == 3:
                c, n, t = input_mesh_arrays
            else:
                raise ValueError("Error during gifti data extraction")
            vertices = c.data
            polygons = t.data
            edges = get_edges_from_polygons(polygons, vertices)
            
            ### Get information from input texture
            # /!\ fixme : check that input_tex corresponds to the mesh
            print "  * Getting information from input texture"
            sys.stdout.flush()
            input_tex = tio.Texture(orig_tex_path).read(orig_tex_path)
            activation_data = input_tex.data
            activation_data[np.isnan(activation_data)] = 0
                
            ### Construct the weights matrix
            print "  * Computing the weights matrix"
            sys.stdout.flush()
            weights_matrix = smooth.compute_weights_matrix(
                polygons, vertices, edges)
                
            ### Define the Laplace-Beltrami operator
            LB_operator = smooth.define_LB_operator(weights_matrix)
                
            ### Compute the number of iterations needed
            N, dt = smooth.compute_smoothing_parameters(weights_matrix, fwhm)
                
            ### Apply smoothing
            print "  * Smoothing...(FWHM = %g)" % fwhm
            sys.stdout.flush()
            smoothed_activation_data = smooth.diffusion_smoothing(
                activation_data, LB_operator, N, dt)
            
            ### Write smoothed data into a new texture file
            print "  * Writing output texture"
            sys.stdout.flush()
            _, base, _ = split_filename(orig_tex_path)
            output_tex = tio.Texture(
                's' + base + '_fwhm' + str(fwhm) + '.tex',
                data=smoothed_activation_data)
            output_tex.write()
            
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        fwhm = self.inputs.fwhm
        for hemisphere in ['left', 'right']:
            if hemisphere == "right":
                orig_tex_path = self.inputs.raw_texture_right
            else:
                orig_tex_path = self.inputs.raw_texture_left
            _, base, _ = split_filename(orig_tex_path)
            outputs["smoothed_texture_%s" % hemisphere] = os.path.abspath(
                's' + base + '_fwhm' + str(fwhm) + '.tex')
        return outputs


# -----------------------------------------------------------
# --------- Define some useful functions --------------------
# -----------------------------------------------------------
def get_edges_from_polygons(polygons, vertices):
    """Builds a mesh edges list from its polygons and vertices.

    """
    nb_edges = 3 * polygons.shape[0]
    edges = np.zeros((nb_edges, 2))
    # get the polygons edges as tuples
    permut = np.array([(0, 0, 1), (1, 0, 0), (0, 1, 0)], dtype=int)
    edges[:, 0] = np.ravel(polygons)
    edges[:, 1] = np.ravel(np.dot(polygons, permut))
    ind = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[ind]

    return edges
