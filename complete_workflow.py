from interfaces import Smooth2D, Smooth3D, GLM3D, GLM2D
import nipype.pipeline.engine as pe

###########
### 3D part
# 3D smoothing
smoother_3d = pe.Node(interface=Smooth3D(fwhm=5), name="smooth3d")
smoother_3d.inputs.raw_image = '/home/virgile/subjects_database/s12069/fMRI/default_acquisition/loc2/waloc_corr4D.nii'

# Volume GLM
paradigm_file = '/home/virgile/subjects_database/s12069/fMRI/default_acquisition/Minf/paradigm.csv'
glm_3d = pe.Node(interface=GLM3D(paradigm=paradigm_file), name="glm3d")

# Create workflow
workflow3D = pe.Workflow(name='workflow_test_3d')
workflow3D.base_dir = '.'
workflow3D.add_nodes([smoother_3d, glm_3d])
workflow3D.connect(smoother_3d, 'smoothed_image', glm_3d, 'input_volume')
#workflow3D.run()

###########
### 2D part
# 2D smoothing
smoother_2d = pe.Node(interface=Smooth2D(fwhm=5), name="smooth2d")
smoother_2d.inputs.raw_texture_left = '/home/virgile/subjects_database/s12069/fct/loc1/lh.aloc1.tex'
smoother_2d.inputs.mesh_left = '/home/virgile/subjects_database/s12069/surf/lh.r.white.gii'
smoother_2d.inputs.raw_texture_right = '/home/virgile/subjects_database/s12069/fct/loc1/rh.aloc1.tex'
smoother_2d.inputs.mesh_right = '/home/virgile/subjects_database/s12069/surf/rh.r.white.gii'

# Cortical GLM
paradigm_file = '/home/virgile/subjects_database/s12069/fMRI/default_acquisition/Minf/paradigm.csv'
glm_2d = pe.Node(interface=GLM2D(paradigm=paradigm_file), name="glm2d")

# Create workflow
workflow2D = pe.Workflow(name='workflow_test_2d')
workflow2D.base_dir = '.'
workflow2D.add_nodes([smoother_2d])
workflow2D.connect(smoother_2d, 'smoothed_texture_left',
                   glm_2d, 'input_texture_left')
workflow2D.connect(smoother_2d, 'smoothed_texture_right',
                   glm_2d, 'input_texture_right')
workflow2D.run()

"""
# Run on multiple subjects
raw_images = ['/home/virgile/subjects_database/%s/fMRI/default_acquisition/loc2/waloc_corr4D.nii' % s for s in ['s12069', 's12207']]
smoother_3d.iterables = ('raw_image', raw_images)
workflow.run()
"""
