import os
import logging

# * Configure the folders --------------------------------------------

# The main directory of the pipeline
pipeline_dir = "/work/users/y/u/yuukias/Heart_Pipeline"  
# The directory for slurm scrips
code_dir = os.path.join(pipeline_dir, 'code')

# All instance2(visit1) and instance3(visit2) data should be placed in this folder
data_raw_dir = '/work/users/y/u/yuukias/database/UKBiobank/data_field' 
# (Optional) Directory for ground truth contours
contour_gt_dir = '/work/users/y/u/yuukias/database/UKBiobank/return/contour_cvi42'  
# Directory to store all files
data_dir = '/work/users/y/u/yuukias/Heart_Pipeline/data' # intermedite files will also be generated here
data_visit1_dir = os.path.join(data_dir, 'visit1', 'nii')
data_visit2_dir = os.path.join(data_dir, 'visit2', 'nii')
data_failed_dir = os.path.join(os.path.dirname(data_dir), 'data_failed')
data_failed_visit1_dir = os.path.join(data_failed_dir, 'visit1', 'nii')
data_failed_visit2_dir = os.path.join(data_failed_dir, 'visit2', 'nii')

# The directory to store all features
features_dir = '/work/users/y/u/yuukias/Heart_Pipeline/doc/Pipeline_Result/'
features_visit1_dir = os.path.join(features_dir, 'visit1')
features_visit2_dir = os.path.join(features_dir, 'visit2')

# * Configure the parameters for dataset -----------------------------------------

# If set to None, then the pipeline will only makes use of the visit1 data
retest_suffix = "retest"   # e.g. This will recognize 20208_retest in data_raw_dir

# Modify here to select the modality you want to use
modality = ["la"]
# modality = ["la", 'sa']
# modality = ['la', 'sa', 'aor', 'tag', 'lvot', 'blood', 't1']


# * Configure the parameters for the logging mechanism ---------------------------

logging_config = '/work/users/y/u/yuukias/Heart_Pipeline/logging.conf'
logging_level = logging.INFO
