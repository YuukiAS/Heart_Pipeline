import os

pipeline_dir = "/work/users/y/u/yuukias/Heart_pipeline"  # modify: the main directory of the pipeline

data_raw_dir = '/work/users/y/u/yuukias/database/UKBiobank/data_field'  # modify
contour_gt_dir = '/work/users/y/u/yuukias/database/UKBiobank/return/contour_cvi42'  # modify: optional for ground truth
out_dir = '/work/users/y/u/yuukias/Heart_pipeline/data/'  # modify: the intermedite files will also be generated here

code_dir = os.path.join(pipeline_dir, 'code')