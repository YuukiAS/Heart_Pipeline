import os
import numpy as np
import nibabel as nib
import cv2
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config
from utils.log_utils import setup_logging
from utils.os_utils import setup_segmentation_folders, prepare_files_to_segment, run_segment_code, obtain_files_segmented

logger = setup_logging("segment_native_t1")

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

    ShMOLLI_name = os.path.join(data_dir, "shmolli_t1map.nii.gz")
    if not os.path.exists(ShMOLLI_name):
        logger.error(f"Native T1 mapping file for {subject} does not exist")
        sys.exit(1)

    if args.model == "nnUNet":
        trained_model_path = os.path.join(
            config.model_dir, "Native_T1_20214", "Dataset20214_ShMOLLI", "nnUNetTrainer__nnUNetPlans__2d"
        )
    elif args.model == "UMamba":
        trained_model_path = os.path.join(
            config.model_dir, "Native_T1_20214", "Dataset20214_ShMOLLI", "nnUNetTrainerUMambaBot__nnUNetPlans__2d"
        )
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
    obtain_files_segmented(subject, 20214, "ShMOLLI", os.path.join(data_dir, "seg_shmolli_t1map.nii.gz"))

    # * For about 1/9 cases, the segmentation of LV and myocardium are swapped, we need to correct it.

    nii_label_original = nib.load(os.path.join(data_dir, "seg_shmolli_t1map.nii.gz"))
    label = nii_label_original.get_fdata()[:, :, 0, 0]

    mask1 = label == 1
    mask1 = mask1.astype(np.uint8)
    mask2 = label == 2
    mask2 = mask2.astype(np.uint8)

    contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    try:
        contour1 = max(contours1, key=lambda x: cv2.arcLength(x, False))
        contour2 = max(contours2, key=lambda x: cv2.arcLength(x, False))
    except ValueError:
        logger.error(f"{subject}: No contours found in the segmentation, the segmentation is invalid.")
        os.remove(os.path.join(data_dir, "seg_shmolli_t1map.nii.gz"))
        sys.exit(1)

    # >= 0 means the point is inside the contour
    contour1_nest = all(
        cv2.pointPolygonTest(contour2, (int(round(point[0][0])), int(round(point[0][1]))), False) >= 0 for point in contour1
    )
    contour2_nest = all(
        cv2.pointPolygonTest(contour1, (int(round(point[0][0])), int(round(point[0][1]))), False) >= 0 for point in contour2
    )

    print(contour1_nest, contour2_nest)
    # By default, contour2_nest should be True, we should correct it if contour1_nest is True
    if contour1_nest:
        logger.info(f"{subject}: Correcting the segmentation of LV and myocardium")
        label_corrected = np.where(label == 1, 2, np.where(label == 2, 1, label))
        # expand dimensions back to 4 and cast to integter
        label_corrected = np.expand_dims(label_corrected, axis=-1)
        label_corrected = np.expand_dims(label_corrected, axis=-1)
        label_corrected = np.round(label_corrected).astype(np.uint8)
        nii_label_corrected = nib.Nifti1Image(label_corrected, nii_label_original.affine, nii_label_original.header)
        os.remove(os.path.join(data_dir, "seg_shmolli_t1map.nii.gz"))
        nib.save(nii_label_corrected, os.path.join(data_dir, "seg_shmolli_t1map.nii.gz"))
    elif contour2_nest:
        logger.info(f"{subject}: Segmentation of LV and myocardium is correct")
    else:
        # for visit2, there are around 284 such subjects
        logger.error(f"{subject}: Segmentation of LV and myocardium is invalid")
        os.remove(os.path.join(data_dir, "seg_shmolli_t1map.nii.gz"))
        sys.exit(1)

    logger.info(f"{subject}: Finish segmentation of native T1 mapping")
