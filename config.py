import os
import logging

# Configure the folders --------------------------------------------
# modify: the main directory of the pipeline
pipeline_dir = "/work/users/y/u/yuukias/Heart_Pipeline"  
code_dir = os.path.join(pipeline_dir, 'code')


# modify: All visit2 and visit3 data should be placed in this folder
data_raw_dir = '/work/users/y/u/yuukias/database/UKBiobank/data_field' 
# modify: optional for ground truth
contour_gt_dir = '/work/users/y/u/yuukias/database/UKBiobank/return/contour_cvi42'  
# modify: the intermedite files will also be generated here
data_dir = '/work/users/y/u/yuukias/Heart_Pipeline/data'
data_visit1_dir = os.path.join(data_dir, 'visit1', 'nii')
data_visit2_dir = os.path.join(data_dir, 'visit2', 'nii')
data_failed_dir = os.path.join(os.path.dirname(data_dir), 'data_failed')
data_failed_visit1_dir = os.path.join(data_failed_dir, 'visit1', 'nii')
data_failed_visit2_dir = os.path.join(data_failed_dir, 'visit2', 'nii')
# modify: the final features will be saved here
features_out_dir = '/work/users/y/u/yuukias/Heart_Pipeline/doc/Pipeline_Result/'

# Configure the parameters for dataset -----------------------------------------
# modify: If set to None, then the pipeline will only makes use of the visit2 data
retest_suffix = "retest"   # e.g. This will recognize 20208_retest in data_raw_dir

# modify here to select the subjects you want to use
modality = ["la"]
# modality = ["la", 'sa']
# modality = ['la', 'sa', 'aor', 'tag', 'lvot', 'blood', 't1']


# Configure the parameters for the pipeline -----------------------------------------
logging_config = '/work/users/y/u/yuukias/Heart_Pipeline/logging.conf'
logging_level = logging.INFO
