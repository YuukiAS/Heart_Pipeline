# Copyright 2018, Wenjia Bai. All Rights Reserved.
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
# ==============================================================================
import os
import numpy as np
import nibabel as nib
import pandas as pd
import argparse
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config

from utils.log_utils import setup_logging
from utils.quality_control_utils import aorta_pass_quality_control

logger = setup_logging("eval_aorta_distensibility")

parser = argparse.ArgumentParser()
parser.add_argument("--retest", action="store_true")
parser.add_argument("--data_list", help="List of subjects to be processed", nargs="*", type=int, required=True)
parser.add_argument("--file_name", help="Name of the csv file to save the features")


if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = config.data_visit2_dir if args.retest else config.data_visit1_dir

    df = pd.DataFrame()

    # Read the spreadsheet for blood pressure information
    # Use central blood pressure provided by the Vicorder software
    # [1] Steffen E. Petersen et al. UK Biobank’s cardiovascular magneticresonance protocol. JCMR, 2016.
    # Aortic distensibility represents the relative change in area of the aorta per unit pressure,
    # taken here as the "central pulse pressure".
    #
    # The Vicorder software calculates values for central blood pressure by applying a previously described
    # brachial-to-aortic transfer function. What I observed from the data and Figure 5 in the SOP pdf
    # (https://biobank.ctsu.ox.ac.uk/crystal/docs/vicorder_in_cmri.pdf) is that after the transfer, the DBP
    # keeps the same as the brachial DBP, but the SBP is different.
    df_info = pd.read_csv(args.pressure_csv, header=[0, 1], index_col=0)
    central_pp = df_info["Central pulse pressure during PWA"][["12678-2.0", "12678-2.1"]].mean(axis=1)

    # Discard central blood pressure < 10 mmHg
    central_pp[central_pp < 10] = np.nan

    data_path = args.data_dir

    for subject in tqdm(args.data_list):
        subject = str(subject)
        logger.info(f"Calculating aortic area and distensibility for subject {subject}")
        sub_dir = os.path.join(data_dir, subject)

        aorta_name = os.path.join(data_dir, "aorta.nii.gz")
        # 1 is ascending aorta, 2 is descending aorta
        seg_aorta_name = os.path.join(data_dir, "seg_aorta.nii.gz")
        nim_aorta = nib.load(aorta_name)
        nim_seg_aorta = nib.load(seg_aorta_name)
        aorta = nim_aorta.get_fdata()
        seg_aorta = nim_seg_aorta.get_fdata()

        if not os.path.exists(aorta_name) or not os.path.exists(seg_aorta_name):
            logger.error(f"Original image or segmentation of aorta file for {subject} does not exist")
            continue

        if not aorta_pass_quality_control(aorta, seg_aorta):  # 4 criterions
            logger.error(f"{subject}: seg_aorta does not pass quality control, skipped.")
            continue

        feature_dict = {
            "eid": subject,
        }

        dx, dy = nim_aorta.header["pixdim"][1:3]
        area_per_pixel = dx * dy

        logger.info(f"{subject}: Measuring area for the ascending and descending aorta")
        areas = {}
        for label, value in [("AAo", 1), ("DAo", 2)]:
            areas[label] = {}
            A_label = np.sum(seg_aorta == value, axis=(0, 1, 2)) * area_per_pixel
            areas[label]["max"] = A_label.max()
            areas[label]["min"] = A_label.min()

        feature_dict.update(
            {
                "Ascending Aorta: Maxium Area [mm^2]": areas["AAo"]["max"],
                "Ascending Aorta: Minimum Area [mm^2]": areas["AAo"]["min"],
                "Descending Aorta: Maxium Area [mm^2]": areas["DAo"]["max"],
                "Descending Aorta: Minimum Area [mm^2]": areas["DAo"]["min"],
            }
        )

        try:
            # define: Central pulse pressure, which is the difference between systolic and diastolic blood pressure
            pressure_info = pd.read_csv(config.pressure_file)[["eid", config.pressure_col_name]]
            pressure_subject = pressure_info[pressure_info["eid"] == int(subject)][config.pressure_col_name].values[0]
            if pressure_subject < 10:
                raise ValueError
            logger.info(f"{subject}: Measuring distensibility for the ascending and descending aorta")

            for label in ["AAo", "DAo"]:
                areas[label]["distensibility"] = (
                    (areas[label]["max"] - areas[label]["min"]) / (areas[label]["min"] * pressure_subject * 1e3)
                )

            feature_dict.update(
                {
                    "Ascending Aorta: Distensibility [10^-3/mmHg]": areas["AAo"]["distensibility"],
                    "Descending Aorta: Distensibility [10^-3/mmHg]": areas["DAo"]["distensibility"],
                }
            )
        except (FileNotFoundError, IndexError):
            logger.error(f"{subject}: Central pulse pressure information not found, no distensibility calculated.")
            continue
        except ValueError:
            logger.warning(f"{subject}: Central pulse pressure < 10 mmHg, no distensibility calculated.")
            continue

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "aorta_distensibility")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), index=False)
