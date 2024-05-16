import os

# Configure the folders --------------------------------------------
# modify: the main directory of the pipeline
pipeline_dir = "/work/users/y/u/yuukias/Heart_pipeline"  
code_dir = os.path.join(pipeline_dir, 'code')


# modify: All visit2 and visit3 data should be placed in this folder
data_raw_dir = '/work/users/y/u/yuukias/database/UKBiobank/data_field' 
# modify: optional for ground truth
contour_gt_dir = '/work/users/y/u/yuukias/database/UKBiobank/return/contour_cvi42'  
# modify: the intermedite files will also be generated here
out_dir = '/work/users/y/u/yuukias/Heart_pipeline/data/'  


logging_config = '/work/users/y/u/yuukias/Heart_pipeline/logging.conf'

# Configure the parameters -----------------------------------------
# modify: If set to None, then the pipeline will only makes use of the visit2 data
retest_suffix = "retest"   # e.g. This will recognize 20208_retest in data_raw_dir

# modify here to select the subjects you want to use
modality = ["la"]
# modality = ['la', 'sa', 'aor', 'tag', 'lvot', 'blood', 't1']
