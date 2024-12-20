import os
import logging

# * Configure paths to bulk data --------------------------------------------

pipeline_dir = "/work/users/y/u/yuukias/Heart_Pipeline"  # The main directory of the pipeline
temp_dir = os.path.join(pipeline_dir, "temp")  # The directory for temporary files
code_dir = os.path.join(pipeline_dir, "code")  # The directory for slurm scrips

# All instance2 (visit1) and instance3 (visit2) data should be placed in this folder
data_raw_dir = "/work/users/y/u/yuukias/database/UKBiobank/data_field"

# (Optional) Directory for ground truth contours
contour_gt_dir = "/work/users/y/u/yuukias/database/UKBiobank/return/contour_cvi42"

# Directory to store all files
data_dir = "/work/users/y/u/yuukias/Heart_Pipeline/data"  # intermedite files will also be generated here
data_visit1_dir = os.path.join(data_dir, "visit1", "nii")
data_visit2_dir = os.path.join(data_dir, "visit2", "nii")

# The directory to store all external codes
lib_dir = "/work/users/y/u/yuukias/Heart_Pipeline/lib"

# The directory to store all trained weights of DL models
model_dir = "/work/users/y/u/yuukias/Heart_Pipeline/model"

# The directory to store all features
features_dir = "/work/users/y/u/yuukias/Heart_Pipeline/data/features/Pipeline_Result"
# * Note: The final results will be stored in 'aggregated' subfolder
features_visit1_dir = os.path.join(features_dir, "visit1")
features_visit2_dir = os.path.join(features_dir, "visit2")

# * Configure paths to tabular data --------------------------------------------
# * Note: All csv files should contain column 'eid' as identifier for subjects

# The file to store BSA information
# Define We use Du Bois formula: BSA = 0.007184 * weight^0.425 * height^0.725
BSA_file = "/work/users/y/u/yuukias/database/UKBiobank/data_field/Personal_Features/BSA.csv"
BSA_col_name = "BSA [m^2]"

# Central pulse pressure: difference between systolic and diastolic blood pressure
pressure_file = "/work/users/y/u/yuukias/database/UKBiobank/data_field/Cardiac_Features/12678_central_pulse.csv"
pressure_col_name = "12678-2.0"

# * Configure the parameters for data processing -----------------------------------------

# If set to None, then the pipeline will only makes use of the visit1 data
retest_suffix = "retest"  # e.g. This will recognize 20208_retest in data_raw_dir

# Modify here to select the modality you want to use
# These are all the CMR modalities provided by UKBiobank
modality = ["aortic_scout", "la", "sa", "aortic_dist", "tag", "lvot", "aortic_blood_flow", "shmolli"]

# If set to True, please also run step1_preaprare_data_ecg.py so that ECG-related features can be generated
useECG = True

# * Configure the parameters for segmentation -----------------------------------------
partition = "htzhulab"  # define the name of partition used for slurm job
# The virtual environment to be activated before prediction
model_envs = {
    "nnUNet": "/work/users/y/u/yuukias/Annotation/nnUNet/env_nnUNet",
    "UMamba": "/work/users/y/u/yuukias/Annotation/UMamba/env_UMamba",
}
# The model used for segmentation of each modality
model_used = {"lvot": "nnUNet", "aortic_blood_flow": "nnUNet", "shmolli": "nnUNet"}

# * Configure the parameters for the logging mechanism ---------------------------

logging_config = "/work/users/y/u/yuukias/Heart_Pipeline/logging.conf"
logging_level = logging.INFO

# * Configure the path and parameters for third-paty tools -----------------------------------------

par_config_dir = "/work/users/y/u/yuukias/Heart_Pipeline/third_party/config"

# Put the folder containing the binaries here, not the path to binaries
average_3d_ffd_path = "/work/users/y/u/yuukias/Heart_Pipeline/third_party/average_3d_ffd"
# The strain requires MIRTK to have transform-points executable!
# Please make sure VTK is installed and install MIRTK by setting following arguments:
# export VTK_DIR=/path/VTK/build
# cmake .. -DVTK_DIR=/path/VTK/build -DWITH_VTK=ON -DMODULE_PointSet=ON
MIRTK_path = "/work/users/y/u/yuukias/Heart_Pipeline/third_party/MIRTK/build/bin"
