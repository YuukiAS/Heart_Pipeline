# Copyright 2019, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import argparse
import shutil
import pandas as pd
from tqdm import tqdm
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config

from utils.log_utils import setup_logging

from utils.quality_control_utils import sa_pass_quality_control
from utils.cardiac_utils import cine_2d_sa_motion_and_strain_analysis

logger = setup_logging("eval_strain_sax")

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
        logger.info(f"Calculating circumferential and radial strain for subject {subject}")
        sub_dir = os.path.join(data_dir, subject)
        seg_sa_name = os.path.join(sub_dir, "seg_sa.nii.gz")
        seg_sa_ED_name = os.path.join(sub_dir, "seg_sa_ED.nii.gz")

        if not os.path.exists(seg_sa_ED_name) or not os.path.exists(seg_sa_name):
            logger.error(f"Segmentation of short axis file for {subject} does not exist")
            continue

        if not sa_pass_quality_control(seg_sa_name):
            logger.error(f"{subject}: seg_sa does not pass quality control, skipped.")
            continue

        feature_dict = {
            "eid": subject,
        }

        # * We will make use of MIRTK and average_3d_ffd here
        # Note that we should not use export here, as it will exit after subprocess terminates
        os.environ["PATH"] = config.MIRTK_path + os.pathsep + os.environ["PATH"]
        os.environ["PATH"] = config.average_3d_ffd_path + os.pathsep + os.environ["PATH"]
        # os.system(f"export PATH={config.MIRTK_path}:$PATH")
        # os.system(f"export PATH={config.average_3d_ffd_path}:$PATH")

        # Define This file contains registration parameters for each resolution level
        par_config_name = os.path.join(config.par_config_dir, "ffd_cine_sa_2d_motion.cfg")

        # Directory to store intermediate motion tracking results
        temp_dir = os.path.join(sub_dir, "temp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        # Directory to store final motion tracking results
        ft_dir = os.path.join(sub_dir, "feature_tracking")
        if not os.path.exists(ft_dir):
            os.makedirs(ft_dir)

        # Perform motion tracking on short-axis images and calculate the strain
        radial_strain, circum_strain = cine_2d_sa_motion_and_strain_analysis(sub_dir, par_config_name, temp_dir, ft_dir)
        logger.info(f"{subject}: Radial and circumferential strain calculated, remove intermediate files.")

        # Remove intermediate files
        shutil.rmtree(temp_dir)

        for i in range(16):
            feature_dict.update({
                f"LV: Radial strain (AHA_{i + 1}) [%]": radial_strain[i, :].max(),
                f"LV: Circumferential strain( AHA_{i + 1}) [%]": circum_strain[i, :].min(),
            })

        feature_dict.update({
            "LV: Radial strain (Global) [%]": radial_strain[16, :].max(),
            "LV: Circumferential strain (Global) [%]": circum_strain[16, :].min(),
        })

        # todo: Make a time series

        # * Feature 1: Strain rate

        radial_strain_lowess = None

        # * We use the maximum absolute value: minimum for circumferential (negative) and maximum for radial (positive)
        target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
        target_dir = os.path.join(target_dir, "strain")
        os.makedirs(target_dir, exist_ok=True)
        df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
        df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), ignore_index=True)