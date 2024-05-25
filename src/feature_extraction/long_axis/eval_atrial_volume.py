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
# =========================c===================================================
import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import vtk
import math
import logging
import logging.config

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config
from utils.quality_control_utils import atrium_pass_quality_control
from utils.cardiac_utils import evaluate_atrial_area_length

logging.config.fileConfig(config.logging_config)
logger = logging.getLogger("eval_atrial_volume")
logging.basicConfig(level=config.logging_level)


parser = argparse.ArgumentParser()
parser.add_argument("--retest", action="store_true")
parser.add_argument("--data_list", help="List of subjects to be processed", nargs="*", type=int, required=True)
parser.add_argument("--file_name", help="Name of the csv file to save the features")

if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = config.data_visit2_dir if args.retest else config.data_visit1_dir

    df = pd.DataFrame()

    for subject in args.data_list:
        subject = str(subject)
        logger.info(f"Calculating atrial volume features for {subject}")
        sub_dir = os.path.join(data_dir, subject)
        sa_name = f"{sub_dir}/sa.nii.gz"
        seg_la_2ch_name = f"{sub_dir}/seg_la_2ch.nii.gz"
        seg_la_4ch_name = f"{sub_dir}/seg_la_4ch.nii.gz"

        if not os.path.exists(sa_name):
            logger.error(f"Short axis file for {subject} does not exist.")
            continue

        # Measurements
        # A: area
        # L: length
        # V: volume
        # lm: landmark, which is the top and bottom of the atrium determined using long_axis
        A = {}
        L = {}
        V = {}
        lm = {}

        # Determine the long-axis from short-axis image
        nim_sa = nib.load(sa_name)
        # * Refer to biobank_utils.py for more information, used to determine top and bottom of atrium
        long_axis = nim_sa.affine[:3, 2] / np.linalg.norm(nim_sa.affine[:3, 2])
        if long_axis[2] < 0:
            long_axis *= -1  # make sure distance is positive

        if os.path.exists(seg_la_2ch_name):
            # Analyse 2 chamber view image
            nim_2ch = nib.load(seg_la_2ch_name)
            seg_la_2ch = nim_2ch.get_fdata()
            T = nim_2ch.header["dim"][4]  # number of time frames

            # Perform quality control for the segmentation
            if not atrium_pass_quality_control(seg_la_2ch, {"LA": 1}):
                logger.warning(f"{subject} seg_la_2ch does not pass atrium_pass_quality_control, skipped.")
                continue

            A["LA_2ch"] = np.zeros(T)
            L["LA_2ch"] = np.zeros(T)
            V["LA_2ch"] = np.zeros(T)
            lm["2ch"] = {}
            for t in range(T):
                area, length, landmarks = evaluate_atrial_area_length(seg_la_2ch[:, :, 0, t], nim_2ch, long_axis)
                if isinstance(area, int) and area < 0:
                    # todo: Is there any chance that area will be negative?
                    logger.warning("Negative area detected, skipping this frame.")
                    continue

                # record features for each time frame
                A["LA_2ch"][t] = area[0]  # we use list return here as 4ch view will have both atrium
                L["LA_2ch"][t] = length[0]
                V["LA_2ch"][t] = 8 / (3 * math.pi) * area[0] * area[0] / length[0]
                lm["2ch"][t] = landmarks

                if t == 0:
                    # Write the landmarks to a vtk file
                    points = vtk.vtkPoints()
                    for p in landmarks:
                        points.InsertNextPoint(p[0], p[1], p[2])
                    poly = vtk.vtkPolyData()
                    poly.SetPoints(points)
                    writer = vtk.vtkPolyDataWriter()
                    writer.SetInputData(poly)
                    os.makedirs(f"{sub_dir}/landmark", exist_ok=True)
                    writer.SetFileName(f"{sub_dir}/landmark/lm_la_2ch_{t:02d}.vtk")
                    writer.Write()
        else:
            logger.error(f"Segmentation of 2-chamber long axis file for {subject} does not exist.")
            continue

        if os.path.exists(seg_la_4ch_name):
            # Analyse 4 chamber view image
            nim_4ch = nib.load(seg_la_4ch_name)
            seg_la_4ch = nim_4ch.get_fdata()

            # Perform quality control for the segmentation
            if not atrium_pass_quality_control(seg_la_4ch, {"LA": 1, "RA": 2}):
                print(f"{subject} seg_la_4ch does not pass atrium_pass_quality_control, skipped.")
                continue

            A["LA_4ch"] = np.zeros(T)
            L["LA_4ch"] = np.zeros(T)
            V["LA_4ch"] = np.zeros(T)
            V["LA_bip"] = np.zeros(T)
            A["RA_4ch"] = np.zeros(T)
            L["RA_4ch"] = np.zeros(T)
            V["RA_4ch"] = np.zeros(T)
            lm["4ch"] = {}
            for t in range(T):
                area, length, landmarks = evaluate_atrial_area_length(seg_la_4ch[:, :, 0, t], nim_4ch, long_axis)
                if isinstance(area, int) and area < 0:
                    logger.warning("Negative area detected, skipping this frame.")
                    continue

                A["LA_4ch"][t] = area[0]
                L["LA_4ch"][t] = length[0]
                V["LA_4ch"][t] = 8 / (3 * math.pi) * area[0] * area[0] / length[0]
                # * We only report the LA volume calculated using the biplane area-length formula (using both modality)
                # * Check https://doi.org/10.1038/s41591-020-1009-y
                V["LA_bip"][t] = 8 / (3 * math.pi) * area[0] * A["LA_2ch"][t] / (0.5 * (length[0] + L["LA_2ch"][t]))

                A["RA_4ch"][t] = area[1]
                L["RA_4ch"][t] = length[1]
                V["RA_4ch"][t] = 8 / (3 * math.pi) * area[1] * area[1] / length[1]
                lm["4ch"][t] = landmarks

                if t == 0:
                    # Write the landmarks to a vtk file
                    points = vtk.vtkPoints()
                    for p in landmarks:
                        points.InsertNextPoint(p[0], p[1], p[2])
                    poly = vtk.vtkPolyData()
                    poly.SetPoints(points)
                    writer = vtk.vtkPolyDataWriter()
                    writer.SetInputData(poly)
                    os.makedirs(f"{sub_dir}/landmark", exist_ok=True)
                    writer.SetFileName(f"{sub_dir}/landmark/lm_la_4ch_{t:02d}.vtk")
                    writer.Write()
        else:
            logger.error(f"Segmentation of 4-chamber long axis file for {subject} does not exist.")
            continue

        # -------------------------
        # * Start to record features
        # Left atrial volume: bi-plane estimation
        # Right atrial volume: single plane estimation
        feature_dict = {
            "eid": subject,
            "LA_max(bip) [mL]": np.max(V["LA_bip"]),
            "LA_min(bip) [mL]": np.min(V["LA_bip"]),
            "LA_SV(bip) [mL]": np.max(V["LA_bip"]) - np.min(V["LA_bip"]),
            "LA_EF(bip) [%]": (np.max(V["LA_bip"]) - np.min(V["LA_bip"])) / np.max(V["LA_bip"]) * 100,
            # All RA are only determined using 4ch view
            "RA_max [mL]": np.max(V["RA_4ch"]),
            "RA_min [mL]": np.min(V["RA_4ch"]),
            "RA_SV [mL]": np.max(V["RA_4ch"]) - np.min(V["RA_4ch"]),
            "RA_EF [%]": (np.max(V["RA_4ch"]) - np.min(V["RA_4ch"])) / np.max(V["RA_4ch"]) * 100,
        }
        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    # Save the features to a csv file
    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "atrium")
    os.makedirs(target_dir, exist_ok=True)
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"))
