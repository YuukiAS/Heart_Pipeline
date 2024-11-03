import os
import numpy as np
import nibabel as nib
import pandas as pd
import argparse
from tqdm import tqdm
import pyvista as pv
import SimpleITK as sitk
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from lib.Aortic_Scout_20207.train_operations.inference_2d_seg import single_image_inference

import config
from utils.log_utils import setup_logging


logger = setup_logging("eval_aorta_structure")

parser = argparse.ArgumentParser()
parser.add_argument("--retest", action="store_true")
parser.add_argument("--data_list", help="List of subjects to be processed", nargs="*", type=int, required=True)
parser.add_argument("--file_name", help="Name of the csv file to save the features")


if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = config.data_visit2_dir if args.retest else config.data_visit1_dir

    df = pd.DataFrame()

    for subject in tqdm(args.data_list):
        subject = str(subject)
        logger.info(f"Calculating aortic structure features for subject {subject}")
        sub_dir = os.path.join(data_dir, subject)

        aorta_name = os.path.join(sub_dir, "aortic_scout.nii.gz")
        seg_aorta_name = os.path.join(sub_dir, "seg_aortic_scout.nii.gz")

        if not os.path.exists(aorta_name):
            logger.error(f"Aorta structure file for {subject} does not exist")
            continue

        if not os.path.exists(seg_aorta_name):
            logger.error(f"Segmentation of aorta structure for {subject} does not exist")
            continue

        feature_dict = {
            "eid": subject,
        }

        df_row = pd.DataFrame([feature_dict])
        # todo: Add quality control
        df_phenotype = single_image_inference(image_path=aorta_name, 
                        label_path=seg_aorta_name,
                        data_output_path=os.path.join(data_dir, "mesh"))
        df_phenotype = df_phenotype.drop(columns=["pid"])
        df_row = pd.concat([df_row, df_phenotype.reset_index(drop=True)], axis=1)
    
        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "aortic_structure")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df = df[[col for col in df.columns if col != 'eid'] + ['eid']]  # move 'eid' to the last column
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), index=False)
