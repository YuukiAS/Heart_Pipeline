"""
This file prepares the folder structure of the dataset that can be utilized for nnUNet.
Please make sure `prepare_data.py` has already been executed before running this script.
Since not all nii files have corresponding ground truth, the function will only make use of those who have.
"""

import os
import shutil
import numpy as np
import argparse
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import logging
import logging.config
import sys

sys.path.append("..")
import config

logging.config.fileConfig(config.logging_config)
logger = logging.getLogger("prepare_data_nnunet")
logging.basicConfig(level=config.logging_level)


def make_out_dirs(dataset_id: int, task_name="UKBiobank"):
    dataset_name = f"Dataset{dataset_id:03d}_{task_name}"

    nnUNet_raw = os.environ["nnUNet_raw"]
    out_dir = Path(nnUNet_raw.replace('"', "")) / dataset_name
    out_train_dir = out_dir / "imagesTr"
    out_labels_dir = out_dir / "labelsTr"
    out_test_dir = out_dir / "imagesTs"

    logger.info("Removing existing files.")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)

    return out_dir, out_train_dir, out_labels_dir, out_test_dir


def copy_files(
    src_data_folder: Path, modality: str, train_dir: Path, labels_dir: Path, test_dir: Path, copyTest=False, seed=1234
):
    """Copy files from the nii folder to the nii_nnunet folder. Returns the number of training cases."""

    subjects_all = os.listdir(src_data_folder)

    subjects_train = []
    subjects_test = []

    logger.info("Determining subjects with and without ground truth.")
    for subject in tqdm(subjects_all):
        subject_dir = os.path.join(src_data_folder, subject)
        if not os.path.exists(os.path.join(subject_dir, f"{modality}.nii.gz")):
            logger.warning(f"Subject {subject} does not have {modality}.nii.gz, skip")
            continue

        if os.path.exists(os.path.join(subject_dir, f"label_{modality}.nii.gz")):
            subjects_train.append(subject_dir)
        else:
            subjects_test.append(subject_dir)

    with open(os.path.join(config.temp_dir, "prepare_nnunet.pkl"), "wb") as f:
        pickle.dump([subjects_train, subjects_test], f)

    logger.info(f"Number of subjects with ground truth: {len(subjects_train)}")
    logger.info(f"Number of subjects without ground truth: {len(subjects_test)}")

    def _copy_file(img_file, img_dir, gt_file=None, label_dir=None):
        subject = img_file.split("/")[-2]

        cnt = 0  # Number of frames with ground truth

        if gt_file is not None and label_dir is not None:
            logger.info(f"Start processing {subject} with ground truth")

            img_4D = nib.load(img_file)
            T = img_4D.shape[3]
            header = img_4D.header
            affine = img_4D.affine
            img_4D = img_4D.get_fdata()
            gt_4D = nib.load(gt_file).get_fdata()

            for t in range(T):
                img = img_4D[:, :, :, t]
                gt = gt_4D[:, :, :, t]
                if np.sum(gt) == 0:
                    # skip frame that is not annotated
                    continue
                cnt += 1
                if img.shape[2] == 1:
                    # change to 2D image if needed
                    header["dim"][0] = 2
                    img = np.squeeze(img, axis=2)
                    gt = np.squeeze(gt, axis=2)
                img = nib.Nifti1Image(img, affine, header)
                gt = nib.Nifti1Image(gt, affine, header)
                # ref https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md
                nib.save(img, img_dir / f"{subject}_{t:04d}_0000.nii.gz")
                nib.save(gt, label_dir / f"{subject}_{t:04d}.nii.gz")

        else:
            logger.info(f"Start processing {subject} without ground truth")
            img_4D = nib.load(img_file)
            T = img_4D.shape[3]
            header = img_4D.header
            affine = img_4D.affine
            img_4D = img_4D.get_fdata()

            for t in range(T):
                img = img_4D[:, :, :, t]
                if img.shape[2] == 1:
                    # change to 2D image if needed
                    header["dim"][0] = 2
                    img = np.squeeze(img, axis=2)
                img = nib.Nifti1Image(img, affine, header)
                nib.save(img, img_dir / f"{subject}_{t:04d}_0000.nii.gz")

        logger.info(f"Finished processing {subject}")

        return cnt

    num_training_cases = 0
    logger.info("Copying training files and corresponding labels.")
    pbar = tqdm(total=len(subjects_train))
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for subject in subjects_train:
            while len(futures) >= 8:
                for future in as_completed(futures):
                    num_training_cases += future.result()
                    pbar.update(1)
                    futures.remove(future)
            future = executor.submit(
                _copy_file,
                os.path.join(subject, f"{modality}.nii.gz"),
                train_dir,
                os.path.join(subject, f"label_{modality}.nii.gz"),
                labels_dir,
            )
            futures.append(future)

        for future in as_completed(futures):
            future.result()
            pbar.update(1)
    pbar.close()

    logger.info("Copying test files.")
    pbar = tqdm(total=len(subjects_test))
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for subject in subjects_test:
            while len(futures) >= 8:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)
                    futures.remove(future)
            future = executor.submit(_copy_file, os.path.join(subject, f"{modality}.nii.gz"), test_dir)
            futures.append(future)

        for future in as_completed(futures):
            future.result()
            pbar.update(1)
    pbar.close()

    return num_training_cases


def convert_ukbiobank(src_data_folder: str, dataset_id, modality: str, copyTest=False):
    out_dir, train_dir, labels_dir, test_dir = make_out_dirs(dataset_id=dataset_id, task_name=f"UKBiobank_{modality}")
    num_training_cases = copy_files(Path(src_data_folder), modality, train_dir, labels_dir, test_dir, copyTest=copyTest)

    if modality == "sa":
        generate_dataset_json(
            str(out_dir),
            channel_names={
                0: "cineMRI",
            },
            labels={
                "background": 0,
                "LV": 1,
                "LVM": 2,
                "RV": 3,
            },
            file_ending=".nii.gz",
            num_training_cases=num_training_cases,
        )
    elif modality == "la_2ch":
        generate_dataset_json(
            str(out_dir),
            channel_names={
                0: "cineMRI",
            },
            labels={
                "background": 0,
                "LA": 1,
            },
            file_ending=".nii.gz",
            num_training_cases=num_training_cases,
        )
    elif modality == "la_4ch":
        generate_dataset_json(
            str(out_dir),
            channel_names={
                0: "cineMRI",
            },
            labels={
                "background": 0,
                "LA": 1,
                "RA": 2,
            },
            file_ending=".nii.gz",
            num_training_cases=num_training_cases,
        )
    else:
        raise ValueError(f"Modality {modality} is not supported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # e.g. For visit1 & sa, the dataset id is 100
    parser.add_argument("-d", "--dataset_id", required=True, type=int, help="nnU-Net Dataset ID, default: 100")
    parser.add_argument(
        "-m",
        "--modality",
        type=str,
        choices=["sa", "la_2ch", "la_4ch"],
        help="The modality of UKBiobank data to be prepared",
    )
    parser.add_argument("--retest", action="store_true")
    args = parser.parse_args()

    src_data_folder = config.data_visit1_dir if not args.retest else config.data_visit2_dir
    if not args.retest:
        os.makedirs(config.data_visit1_nnunet_dir, exist_ok=True)
        os.environ["nnUNet_raw"] = os.path.join(config.data_visit1_nnunet_dir, "nnUNet_raw")
        os.environ["nnUNet_preprocessed"] = os.path.join(config.data_visit1_nnunet_dir, "nnUNet_preprocessed")
        os.environ["nnUNet_results"] = os.path.join(config.data_visit1_nnunet_dir, "nnUNet_results")
    else:
        os.makedirs(config.data_visit2_nnunet_dir, exist_ok=True)
        os.environ["nnUNet_raw"] = os.path.join(config.data_visit2_nnunet_dir, "nnUNet_raw")
        os.environ["nnUNet_preprocessed"] = os.path.join(config.data_visit2_nnunet_dir, "nnUNet_preprocessed")
        os.environ["nnUNet_results"] = os.path.join(config.data_visit2_nnunet_dir, "nnUNet_results")

    logger.info("Start Conversion")
    # Note that there are too many files in visit1, we only make copy of visit2 to save space
    convert_ukbiobank(src_data_folder, args.dataset_id, args.modality, copyTest=args.retest)
    logger.info("Done!")
