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
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import pickle
import vtk

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config

from utils.log_utils import setup_logging
from utils.quality_control_utils import sa_pass_quality_control
from utils.cardiac_utils import evaluate_wall_thickness, evaluate_radius_thickness_disparity, fractal_dimension


logger = setup_logging("eval_wall_thickness")

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
        logger.info(f"Calculating myocardial wall thickness for subject {subject}")
        sub_dir = os.path.join(data_dir, subject)

        # Quality control for segmentation at ED
        sa_ED_name = f"{sub_dir}/sa_ED.nii.gz"
        seg_sa_name = f"{sub_dir}/seg_sa.nii.gz"
        seg_sa_ED_name = f"{sub_dir}/seg_sa_ED.nii.gz"
        if not os.path.exists(seg_sa_name):
            logger.error(f"Segmentation of short axis file for {subject} does not exist")
            continue

        if not sa_pass_quality_control(seg_sa_name):  # default t = 0
            # If the segmentation quality is low, evaluation of wall thickness may fail.
            logger.error(f"{subject}: seg_sa does not pass sa_pass_quality_control, skipped.")
            continue

        nim_sa = nib.load(sa_ED_name)
        nim_sa_ED = nib.load(sa_ED_name)
        nim_seg_sa_ED = nib.load(seg_sa_ED_name)
        seg_sa = nim_sa.get_fdata()
        seg_sa_ED = nim_seg_sa_ED.get_fdata()

        feature_dict = {
            "eid": subject,
        }

        # * Basic Feature: Evaluate myocardial wall thickness at ED (17 segment)
        # Ref https://media.springernature.com/lw685/springer-static/image/chp%3A10.1007%2F978-3-030-63560-2_16/MediaObjects/496124_1_En_16_Fig13_HTML.png
        # Ref https://www.pmod.com/files/download/v34/doc/pcardp/3618.gif for illustration
        # Ref https://jcmr-online.biomedcentral.com/articles/10.1186/s12968-020-00683-3 for reference range

        # Note that measurements on long axis at basal and mid-cavity level have been shown to be significantly
        # greater compared to short axis measurements; while apical level demonstrates the opposite trend
        # Save wall thickness for all the subjects
        endo_poly, epi_poly, index, table_thickness, table_thickness_max = evaluate_wall_thickness(
            seg_sa_ED, nim_sa_ED, save_epi_contour=True
        )

        endo_writer = vtk.vtkPolyDataWriter()
        endo_output_name = f"{sub_dir}/landmark/endo.vtk"
        endo_writer.SetFileName(endo_output_name)
        endo_writer.SetInputData(endo_poly)
        endo_writer.Write()

        epi_writer = vtk.vtkPolyDataWriter()
        epi_output_name = f"{sub_dir}/landmark/epi.vtk"
        epi_writer.SetFileName(epi_output_name)
        epi_writer.SetInputData(epi_poly)
        epi_writer.Write()
        logger.info(f"{subject}: Wall-thickness evaluation completed, data saved")

        # 16 segment + 1 global
        for i in range(16):
            feature_dict.update(
                {
                    f"Myo: Thickness (AHA_{i + 1} [mm])": table_thickness[i],
                    f"Myo: Thickness (AHA_{i + 1}_max [mm])": table_thickness_max[i],
                }
            )
        feature_dict.update(
            {
                "Myo: Thickness (Global_mean) [mm]": table_thickness[16],
                "Myo: Thickness (Global_max) [mm]": table_thickness_max[16],
            }
        )

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

        # * Feature 2: Radius and thickness that incorporate distance to barycenter of LV cavity (with modification)
        # Refer https://www.sciencedirect.com/science/article/pii/S1361841519300519
        # * This features utilizes the entire time series instead of just ED

        # BSA is required in calculation
        try:
            BSA_info = pd.read_csv(config.BSA_file)[["eid", config.BSA_col_name]]
            BSA_subject = BSA_info[BSA_info["eid"] == int(subject)][config.BSA_col_name].values[0]
            logger.info(f"{subject}: Implement disparity features")
            radius_motion_disparity, thickness_motion_disparity, radius, thickness = (
                evaluate_radius_thickness_disparity(seg_sa, nim_sa, BSA_subject)
            )
            feature_dict.update({
                "Myo: Radius motion disparity": radius_motion_disparity,
                "Myo: Thickness motion disparity": thickness_motion_disparity,
            })
            logger.info(f"{subject}: Motion disparity features calculation calculated, saving raw data to pickle files")
            os.makedirs(f"{sub_dir}/raw", exist_ok=True)
            with open(f"{sub_dir}/raw/myo_radius_thickness.pkl", "wb") as raw_file:
                pickle.dump({
                    "Myo: Radius [cm]": radius,
                    "Myo: Thickness [cm]": thickness,
                }, raw_file)
        except IndexError:
            logger.error(f"{subject}: BSA information not found, skip disparity features.")
            continue

        # * Feature 3: Fractal dimension for trabeculation
        # Refer  https://jcmr-online.biomedcentral.com/articles/10.1186/1532-429X-15-36

        fd = fractal_dimension(nim_sa_ED, seg_sa_ED)
        feature_dict.update({"Myo: Fractal dimension": fd})
        logger.info(f"{subject}: Fractal dimension calculation completed")

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "wall_thickness")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), ignore_index=True)
