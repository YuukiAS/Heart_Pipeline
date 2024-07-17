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
import numpy as np
import pandas as pd
import nibabel as nib
import pickle
import vtk
import math
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config

from utils.log_utils import setup_logging
from utils.cardiac_utils import evaluate_ventricular_length_sax, evaluate_ventricular_length_lax
from utils.quality_control_utils import sa_pass_quality_control

logger = setup_logging("eval_ventricular_volume")

parser = argparse.ArgumentParser()
parser.add_argument("--retest", action="store_true")
parser.add_argument("--data_list", help="List of subjects to be processed", nargs="*", type=int, required=True)
parser.add_argument("--file_name", help="Name of the csv file to save the features")

if __name__ == '__main__':
    args = parser.parse_args()

    data_dir = config.data_visit2_dir if args.retest else config.data_visit1_dir

    df = pd.DataFrame()

    for subject in tqdm(args.data_list):
        subject = str(subject)
        logger.info(f"Calculating ventricular volume for subject {subject}")
        sub_dir = os.path.join(data_dir, subject)
        sa_name = f"{sub_dir}/sa.nii.gz"
        la_4ch_name = f"{sub_dir}/la_4ch.nii.gz"
        seg_sa_name = f"{sub_dir}/seg_sa.nii.gz"
        seg4_la_name = f"{sub_dir}/seg4_la_4ch.nii.gz"

        if not os.path.exists(sa_name):
            logger.error(f"Short axis file for {subject} does not exist")
            continue

        if not os.path.exists(la_4ch_name):
            logger.error(f"4-chamber long axis file for {subject} does not exist")
            continue

        if not os.path.exists(seg_sa_name):
            logger.error(f"Segmentation of short axis file for {subject} does not exist")
            continue

        if not sa_pass_quality_control(seg_sa_name):  # default t = 0
            logger.error(f"{subject}: seg_sa does not pass sa_pass_quality_control, skipped.")
            continue

        nim_sa = nib.load(sa_name)
        nim_la_4ch = nib.load(la_4ch_name)
        pixdim = nim_sa.header['pixdim'][1:4]
        volume_per_pix = pixdim[0] * pixdim[1] * pixdim[2] * 1e-3
        density = 1.05  # myocardium density is approximately 1.05 g/mL

        # Heart rate
        duration_per_cycle = nim_sa.header['dim'][4] * nim_sa.header['pixdim'][4]  # e.g. 50 time frame, 15.5 ms per cycle
        heart_rate = 60.0 / duration_per_cycle

        # Segmentation
        seg_sa = nib.load(seg_sa_name).get_fdata()
        seg4_la = nib.load(seg4_la_name).get_fdata()
        L_sax = {}
        L_4ch = {}
        V = {}
        V["LV"] = np.sum(seg_sa == 1, axis=(0, 1, 2)) * volume_per_pix
        V["RV"] = np.sum(seg_sa == 3, axis=(0, 1, 2)) * volume_per_pix
        lm_sax = {}
        lm_4ch = {}

        # Save time series of volume and display the time series of ventricular volume
        os.makedirs(f"{sub_dir}/timeseries", exist_ok=True)
        logger.info(f"Saving time series of ventricular volume for {subject}")
        with open(f"{sub_dir}/timeseries/ventricle.pkl", "wb") as time_series_file:
            pickle.dump({
                "LV: Volume [mL]": V["LV"],
                "RV: Volume [mL]": V["RV"]
            }, time_series_file)

        logger.info(f"{subject}: Record basic features")
        feature_dict = {
            "eid": subject,
        }

        T_ED = 0  # bSSFP uses retrospective gating
        T_ES = np.argmin(V["LV"])

        feature_dict.update({
            "LV: V_ED [mL]": np.sum(seg_sa[:, :, :, T_ED] == 1) * volume_per_pix,
            "LVM: Mass_ED [g]": np.sum(seg_sa[:, :, :, T_ED] == 2) * volume_per_pix * density,
            "RV: V_ED [mL]": np.sum(seg_sa[:, :, :, T_ED] == 3) * volume_per_pix,
            "LV: V_ES [mL]": np.sum(seg_sa[:, :, :, T_ES] == 1) * volume_per_pix,
            "LVM: Mass_ES [g]": np.sum(seg_sa[:, :, :, T_ES] == 2) * volume_per_pix * density,
            "RV: V_ES [mL]": np.sum(seg_sa[:, :, :, T_ES] == 3) * volume_per_pix,
        })

        feature_dict.update({
            "LV: SV [mL]": feature_dict["LV: V_ED [mL]"] - feature_dict["LV: V_ES [mL]"],

            "RV: SV [mL]": feature_dict["RV: V_ED [mL]"] - feature_dict["RV: V_ES [mL]"],
        })

        feature_dict.update({
            "LV: CO [L/min]": feature_dict["LV: SV [mL]"] * heart_rate * 1e-3,
            "LV: EF [%]": feature_dict["LV: SV [mL]"] / feature_dict["LV: V_ED [mL]"] * 100,

            "RV: CO [L/min]": feature_dict["RV: SV [mL]"] * heart_rate * 1e-3,
            "RV: EF [%]": feature_dict["RV: SV [mL]"] / feature_dict["RV: V_ED [mL]"] * 100,
        })

        # * Implement more advanced features -----------------------------------

        # * Feature1: Indexed volume
        logger.info(f"{subject}: Implement indexed volume features")
        try:
            BSA_info = pd.read_csv(config.BSA_file)[["eid", config.BSA_col_name]]
            BSA_subject = BSA_info[BSA_info["eid"] == int(subject)][config.BSA_col_name].values[0]
        except IndexError:
            logger.error(f"{subject}: BSA information not found, skipped.")
            # As BSA is a crucial feature, we skip the subject if BSA is not found
            continue

        feature_dict.update({
            "LV: V_ED/BSA [mL/m^2]": feature_dict["LV: V_ED [mL]"] / BSA_subject,
            "LVM: Mass_ED/BSA [g/m^2]": feature_dict["LVM: Mass_ED [g]"] / BSA_subject,
            "RV: V_ED/BSA [mL/m^2]": feature_dict["RV: V_ED [mL]"] / BSA_subject,
            "LV: V_ES/BSA [mL/m^2]": feature_dict["LV: V_ES [mL]"] / BSA_subject,
            "LVM: Mass_ES/BSA [g/m^2]": feature_dict["LVM: Mass_ES [g]"] / BSA_subject,
            "RV: V_ES/BSA [mL/m^2]": feature_dict["RV: V_ES [mL]"] / BSA_subject,
            # define Cardiac Index, check https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4830061/pdf/12968_2016_Article_236.pdf for reference range
            "LV: CI [L/min/m^2]": feature_dict["LV: CO [L/min]"] / BSA_subject,
            "RV: CI [L/min/m^2]": feature_dict["RV: CO [L/min]"] / BSA_subject,
        })

        # * Feature2: LV Diameter and Spherical Index
        logger.info(f"{subject}: Implement LV diameter and spherical index")
        long_axis = nim_sa.affine[:3, 2] / np.linalg.norm(nim_sa.affine[:3, 2])
        short_axis = nim_la_4ch.affine[:3, 2] / np.linalg.norm(nim_la_4ch.affine[:3, 2])

        try:  
            L_sax["ED"], lm_sax["ED"] = evaluate_ventricular_length_sax(seg_sa[:,:,:,T_ED], nim_sa, long_axis, short_axis)
            points = vtk.vtkPoints()
            for p in lm_sax["ED"]:
                points.InsertNextPoint(p[0], p[1], 0)
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(poly)
            os.makedirs(f"{sub_dir}/landmark", exist_ok=True)
            writer.SetFileName(f"{sub_dir}/landmark/ventricle_sa_ED.vtk")
            writer.Write()

            L_4ch["ED"], lm_4ch["ED"] = evaluate_ventricular_length_lax(seg4_la[:,:,0,T_ED], nim_la_4ch, long_axis, short_axis)
            points = vtk.vtkPoints()
            for p in lm_4ch["ED"]:
                points.InsertNextPoint(p[0], p[1], 0)
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(poly)
            os.makedirs(f"{sub_dir}/landmark", exist_ok=True)
            writer.SetFileName(f"{sub_dir}/landmark/ventricle_4ch_ED.vtk")
            writer.Write()           

            # * We use the diameter measured in long-axis to calculate spherical index, see https://pubmed.ncbi.nlm.nih.gov/27571232/
            V_sphere_ED = 4 / 3 * math.pi * (L_4ch["ED"] / 2) ** 3

            feature_dict.update({
                "LV: D_ED(sax) [cm]": L_sax["ED"],
                "LV: D_ED(4ch) [cm]": L_4ch["ED"],
                "LV: Spherical Index": feature_dict["LV: V_ED [mL]"] / V_sphere_ED,
            })


        except ValueError:
            logger.warning(f"{subject}: Failed to determine the diameter of left ventricle at ED")

        try:
            L_sax["ES"], lm_sax["ES"] = evaluate_ventricular_length_sax(seg_sa[:,:,:,T_ES], nim_sa, long_axis, short_axis)
            points = vtk.vtkPoints()
            for p in lm_sax["ES"]:
                points.InsertNextPoint(p[0], p[1], 0)
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(poly)
            os.makedirs(f"{sub_dir}/landmark", exist_ok=True)
            writer.SetFileName(f"{sub_dir}/landmark/ventricle_sa_ES.vtk")
            writer.Write()

            L_4ch["ES"], lm_4ch["ES"] = evaluate_ventricular_length_lax(seg4_la[:,:,0,T_ES], nim_la_4ch, long_axis, short_axis)
            points = vtk.vtkPoints()
            for p in lm_4ch["ES"]:
                points.InsertNextPoint(p[0], p[1], 0)
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(poly)
            os.makedirs(f"{sub_dir}/landmark", exist_ok=True)
            writer.SetFileName(f"{sub_dir}/landmark/ventricle_4ch_ES.vtk")
            writer.Write()   

            feature_dict.update({
                "LV: D_ES(sax) [cm]": L_sax["ES"],
                "LV: D_ES(4ch) [cm]": L_4ch["ES"],
            })
        except ValueError:
            logger.warning(f"{subject}: Failed to determine the diameter of left ventricle at ES")

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "ventricle")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"))
