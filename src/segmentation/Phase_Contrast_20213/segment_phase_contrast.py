import os
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config
from utils.log_utils import setup_logging
from utils.os_utils import setup_segmentation_folders, prepare_files_to_segment, run_segment_code, obtain_files_segmented

logger = setup_logging("segment_phase_contrast")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Folder for one subject that contains Nifti files", required=True)
parser.add_argument(
    "--model", type=str, help="Trained model to be used for segmentation", required=True, choices=["nnUNet", "UMamba"]
)

if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = args.data_dir
    temp_dir = config.temp_dir
    subject = os.path.basename(data_dir)

    model_dir = config.model_dir

    img_morphology_name = os.path.join(data_dir, "aortic_flow_processed.nii.gz")
    if not os.path.exists(img_morphology_name):
        logger.error(f"{subject}: Processed phase Contrast Cine MRI file does not exist")
        sys.exit(1)

    if args.model == "nnUNet":
        trained_model_path = os.path.join(
            config.model_dir, "Phase_Contrast_20213", "Dataset20213_Phase_Contrast", "nnUNetTrainer__nnUNetPlans__2d"
        )
    elif args.model == "UMamba":
        trained_model_path = os.path.join(
            config.model_dir, "Phase_Contrast_20213", "Dataset20213_Phase_Contrast", "nnUNetTrainerUMambaBot__nnUNetPlans__2d"
        )
    else:
        raise ValueError("Model should be either nnUNet or UMamba")

    logger.info(f"{subject}: Setting up folders for segmentation")
    setup_segmentation_folders(20213, "Phase_Contrast", trained_model_path)

    # * We will split the file and put them in temp forlder for training
    logger.info(f"{subject}: Preparing files for segmentation")
    prepare_files_to_segment(img_morphology_name, subject, 20213, "Phase_Contrast")

    logger.info(f"{subject}: Start predicting segmentations using model {args.model}")
    run_segment_code(subject, 20213, "Phase_Contrast", args.model)

    logger.info(f"{subject}: Retrieve segmented images")
    obtain_files_segmented(subject, 20213, "Phase_Contrast", os.path.join(data_dir, "seg_aortic_flow.nii.gz"))

    logger.info(f"{subject}: Finish segmentation of Phase_Contrast Cine")
