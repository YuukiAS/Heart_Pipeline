import os
import logging

logging_level = logging.INFO

# Note: Please modify all lines following "Modify" for configuration

# * Step1: Configure the pipeline folders --------------------------------------------

# Modify: The main directory of the pipeline
pipeline_dir = "/work/users/y/u/yuukias/Heart_Pipeline"  # The main directory of the pipeline
temp_dir = os.path.join(pipeline_dir, "temp")  # The directory for temporary files
code_dir = os.path.join(pipeline_dir, "code")  # The directory for slurm scrips
lib_dir = os.path.join(pipeline_dir, "lib")  # The directory for all external codes
model_dir = os.path.join(pipeline_dir, "model")  # The directory for weights of all trained models
data_dir = os.path.join(pipeline_dir, "data")  # * The directory for Nifti data and all other intermediate results
data_visit1_dir = os.path.join(data_dir, "visit1", "nii")
data_visit2_dir = os.path.join(data_dir, "visit2", "nii")
features_dir = os.path.join(data_dir, "features")  # * The directory to store all extracted features
# Note: The final results will be stored in 'aggregated' subfolder
features_visit1_dir = os.path.join(features_dir, "visit1")
features_visit2_dir = os.path.join(features_dir, "visit2")


# * Step2: Configure the raw data folders and files -----------------------------------

# Modify: The directory that contains all the raw data
# Note: For the detailed structure, please refer to README.md
data_raw_dir = "/work/users/y/u/yuukias/database/UKBiobank/data_field"

# Modify: Set retest_suffix to None if you only want to use the first visit data
retest_suffix = "retest"  # e.g. This will recognize 20208_retest in data_raw_dir

# Note: Both files need to be created. Formula for BSA is given. You should use data field 50, 21002 and 12678 in UKBiobank
# Note: All csv files should contain column 'eid' as identifier for subjects

# Modify: The file to store the BSA information
# define We use Du Bois formula: BSA = 0.007184 * weight^0.425 * height^0.725. Weight is in field 21002, height in field 50
BSA_file = "/work/users/y/u/yuukias/database/UKBiobank/data_field/Personal_Features/BSA.csv"
BSA_col_name = "BSA [m^2]"  # The BSA_col_name is the name of column that stores BSA information

# Modify: The file to store the central pulse pressure information
# define Central pulse pressure: difference between systolic and diastolic blood pressure (stored in field 12678)
pressure_file = "/work/users/y/u/yuukias/database/UKBiobank/data_field/Cardiac_Features/12678_central_pulse.csv"
pressure_col_name = "12678-2.0"  # The pressure_col_name is the name of column that stores pulse pressure information (20210)

# * Step3: Configure the modalities to use for pipeline -----------------------------------------

# Modify: Select the modality you want to use
# Note: To use all the modalities, modality should be set as follows. You can remove any of them when needed.
# modality = ["aortic_scout", "la", "sa", "aortic_dist", "tag", "lvot", "aortic_blood_flow", "shmolli"]
modality = ["aortic_scout", "la", "sa", "aortic_dist", "tag", "lvot", "aortic_blood_flow", "shmolli"]

# Modify: Whether or not to use ECG data
# Note: If set to True, you should run step1_prepare_data_ecg.py to prepare the ECG data
useECG = True  # (6025, 20205)

# * Step4: Configure the details for segmentation -----------------------------------------

# Modify: Set to the name of partition in Slurm which has access to GPU resources
partition = "volta-gpu"  

# Modify: To make use of nnUNet and UMamba, we recommend creating virtual environment for each model to avoid potential conflicts
model_envs = {
    "nnUNet": "/work/users/y/u/yuukias/Annotation/nnUNet/env_nnUNet",
    "UMamba": "/work/users/y/u/yuukias/Annotation/UMamba/env_UMamba",
}
# Modify: The model to use for segmentation of each modality
model_used = {"lvot": "nnUNet", "aortic_blood_flow": "nnUNet", "shmolli": "nnUNet"}  # (20212,20213,20214)


# * Step5: Configure the third-party tools -----------------------------------------

# Modify: The folder that contains the binary file `average_3d_ffd`
average_3d_ffd_path = "/work/users/y/u/yuukias/Heart_Pipeline/third_party/average_3d_ffd"  # (20208,20209)

# Modify: The folder that contains the binary file `mirtk`
# Note: Please make sure VTK is installed and install MIRTK by setting following arguments:
# export VTK_DIR=/path/VTK/build
# cmake .. -DVTK_DIR=/path/VTK/build -DWITH_VTK=ON -DMODULE_PointSet=ON
MIRTK_path = "/work/users/y/u/yuukias/Heart_Pipeline/third_party/MIRTK/build/bin"  # (20208,20209)

# Modify: The folder that contains the configuration files for MIRTK
par_config_dir = "/work/users/y/u/yuukias/Heart_Pipeline/third_party/config"