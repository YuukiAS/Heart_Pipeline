import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config
from utils.log_utils import setup_logging
from utils.os_utils import (
    setup_segmentation_folders, 
    prepare_files_to_segment,
    run_segment_code,
    obtain_files_segmented
)

logger = setup_logging("segment_native_t1")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type = str, help="Folder for one subject that contains Nifti files", required=True)
parser.add_argument("--model", type = str, help="Trained model to be used for segmentation", required=True, choices=["nnUNet", "UMamba"])

if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = args.data_dir
    temp_dir = config.temp_dir
    subject = os.path.basename(data_dir)

    model_dir = config.model_dir

    ShMOLLI_name = os.path.join(data_dir, "shmolli_t1map.nii.gz")
    if not os.path.exists(ShMOLLI_name):
        logger.error(f"Native T1 mapping file for {subject} does not exist")
        sys.exit(1)

    if args.model == "nnUNet":
        trained_model_path = os.path.join(config.model_dir, "Native_T1_20214", "Dataset20214_ShMOLLI", "nnUNetTrainer__nnUNetPlans__2d")        
    elif args.model == "UMamba":
        trained_model_path = os.path.join(config.model_dir, "Native_T1_20214", "Dataset20214_ShMOLLI", "nnUNetTrainerUMambaBot__nnUNetPlans__2d")
    else:
        raise ValueError("Model should be either nnUNet or UMamba")

    logger.info(f"{subject}: Setting up folders for segmentation")
    setup_segmentation_folders(20214, "ShMOLLI", trained_model_path)

    # * We will split the file and put them in temp forlder for training
    logger.info(f"{subject}: Preparing files for segmentation")
    prepare_files_to_segment(ShMOLLI_name, subject, 20214, "ShMOLLI")

    logger.info(f"{subject}: Start predicting segmentations using model {args.model}")
    run_segment_code(subject, 20214, "ShMOLLI", args.model)

    logger.info(f"{subject}: Retrieve segmented images")
    obtain_files_segmented(subject, 20214, "ShMOLLI", os.path.join(data_dir, f"seg_shmolli_t1map.nii.gz"))

    logger.info(f"{subject}: Finish segmentation of native T1 mapping")