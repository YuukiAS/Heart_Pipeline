import os
import logging

# * Configure essential paths --------------------------------------------

# The main directory of the pipeline
pipeline_dir = "/work/users/y/u/yuukias/Heart_Pipeline"  
temp_dir = os.path.join(pipeline_dir, 'temp')
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
data_visit1_nnunet_dir = os.path.join(data_dir, 'visit1', 'nii_nnunet')
data_visit2_nnunet_dir = os.path.join(data_dir, 'visit2', 'nii_nnunet')
data_failed_dir = os.path.join(os.path.dirname(data_dir), 'data_failed')
data_failed_visit1_dir = os.path.join(data_failed_dir, 'visit1', 'nii')
data_failed_visit2_dir = os.path.join(data_failed_dir, 'visit2', 'nii')

# The directory to store all features
features_dir = '/work/users/y/u/yuukias/Heart_Pipeline/doc/Pipeline_Result/'
# The final results will be stored in "comprehensive" subfolder
features_visit1_dir = os.path.join(features_dir, 'visit1')
features_visit2_dir = os.path.join(features_dir, 'visit2')

# The file to store BSA information
# Define We use Du Bois formula: BSA = 0.007184 * weight^0.425 * height^0.725
BSA_file = "/work/users/y/u/yuukias/database/UKBiobank/data_field/Personal_Features/BSA.csv"
# The column name for BSA in the BSA_file, should contain column "eid" as identifier
BSA_col_name = "BSA [m^2]"

# * Configure the parameters for dataset -----------------------------------------

# If set to None, then the pipeline will only makes use of the visit1 data
retest_suffix = "retest"   # e.g. This will recognize 20208_retest in data_raw_dir

# Modify here to select the modality you want to use
modality = ["la", 'sa']
# modality = ['la', 'sa', 'aor', 'tag', 'lvot', 'blood', 't1']

# Please also run step1_preaprare_data_ecg.py so that ECG-related features can be generated
useECG = True

# * Configure the parameters for the logging mechanism ---------------------------

logging_config = '/work/users/y/u/yuukias/Heart_Pipeline/logging.conf'
logging_level = logging.INFO
