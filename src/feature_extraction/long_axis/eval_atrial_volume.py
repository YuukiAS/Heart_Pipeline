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
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import vtk
import math
import pickle
from tqdm import tqdm
from scipy.signal import find_peaks
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ecg.ecg_processor import ECG_Processor

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config

from utils.quality_control_utils import atrium_pass_quality_control
from utils.cardiac_utils import evaluate_atrial_area_length
from utils.analyze_utils import plot_time_series_double_x, plot_time_series_double_x_y
from utils.log_utils import setup_logging

logger = setup_logging("eval_atrial_volume")


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
        # L_L: length
        # V: volume
        # lm: landmark, which is the top and bottom of the atrium determined using long_axis
        A = {}
        L_L = {}  # longitudinal diameter
        L_T = {}  # transverse diameter
        V = {}
        lm = {}

        # * Generate basic features from segmentation ---------------------------
        logger.info(f"{subject}: Process information from segmentation")
        # Determine the long-axis from short-axis image
        nim_sa = nib.load(sa_name)
        # Refer to biobank_utils.py for more information, used to determine top and bottom of atrium
        long_axis = nim_sa.affine[:3, 2] / np.linalg.norm(nim_sa.affine[:3, 2])
        if long_axis[2] < 0:
            long_axis *= -1  # make sure distance is positive

        if os.path.exists(seg_la_2ch_name):
            # Analyse 2 chamber view image
            nim_2ch = nib.load(seg_la_2ch_name)
            seg_la_2ch = nim_2ch.get_fdata()
            T = nim_2ch.header["dim"][4]  # number of time frames
            temporal_resolution = nim_2ch.header["pixdim"][4] * 1000 # unit: ms

            # Perform quality control for the segmentation
            if not atrium_pass_quality_control(seg_la_2ch, {"LA": 1}):
                logger.error(f"{subject} seg_la_2ch does not pass atrium_pass_quality_control, skipped.")
                continue

            A["LA_2ch"] = np.zeros(T)
            L_L["LA_2ch"] = np.zeros(T)
            L_T["LA_2ch"] = np.zeros(T)
            V["LA_2ch"] = np.zeros(T)
            lm["2ch"] = {}
            for t in range(T):
                try:
                    area, length, landmarks = evaluate_atrial_area_length(seg_la_2ch[:, :, 0, t], nim_2ch, long_axis)
                except ValueError as e:
                    logger.error(f"Error in evaluating atrial area and length for {subject} at time frame {t}: {e}")
                    continue

                if isinstance(area, int) and area < 0:
                    raise ValueError("Negative area detected.")

                # record features for each time frame
                A["LA_2ch"][t] = area[0]  # we use list return here as 4ch view will have both atrium
                L_L["LA_2ch"][t] = length[0]
                L_T["LA_2ch"][t] = length[1]
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
                logger.error(f"{subject} seg_la_4ch does not pass atrium_pass_quality_control, skipped.")
                continue

            A["LA_4ch"] = np.zeros(T)
            L_L["LA_4ch"] = np.zeros(T)
            L_T["LA_4ch"] = np.zeros(T)
            V["LA_4ch"] = np.zeros(T)
            V["LA_bip"] = np.zeros(T)
            A["RA_4ch"] = np.zeros(T)
            L_L["RA_4ch"] = np.zeros(T)
            L_T["RA_4ch"] = np.zeros(T)
            V["RA_4ch"] = np.zeros(T)
            lm["4ch"] = {}
            for t in range(T):
                try:
                    area, length, landmarks = evaluate_atrial_area_length(seg_la_4ch[:, :, 0, t], nim_4ch, long_axis)
                except ValueError as e:
                    logger.error(f"Error in evaluating atrial area and length for {subject} at time frame {t}: {e}")
                    continue
                if isinstance(area, int) and area < 0:
                    raise ValueError("Negative area detected.")

                A["LA_4ch"][t] = area[0]
                L_L["LA_4ch"][t] = length[0]
                L_T["LA_4ch"][t] = length[1]
                V["LA_4ch"][t] = 8 / (3 * math.pi) * area[0] * area[0] / length[0]

                # We only report the LA volume calculated using the biplane area-length formula (using both modality)
                # Ref https://doi.org/10.1038/s41591-020-1009-y
                V["LA_bip"][t] = 8 / (3 * math.pi) * area[0] * A["LA_2ch"][t] / (0.5 * (length[0] + L_L["LA_2ch"][t]))

                A["RA_4ch"][t] = area[1]
                L_L["RA_4ch"][t] = length[2]  # LA and RA are processed sequentially
                L_T["RA_4ch"][t] = length[3]
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
 
        # * Record basic features -----------------------------------------------
        # Left atrial volume: bi-plane estimation
        # Right atrial volume: single plane estimation
        logger.info(f"{subject}: Record basic features")
        feature_dict = {
            "eid": subject,
        }

        LAV_max = np.max(V["LA_bip"])
        LAV_min = np.min(V["LA_bip"])
        feature_dict.update({
            # LA are determined using both 2ch and 4ch view
            "LA: D_longitudinal(2ch) [cm]": np.max(L_L["LA_2ch"]),
            "LA: D_longitudinal(4ch) [cm]": np.max(L_L["LA_4ch"]),
            "LA: A_max(2ch) [mm^2]": np.max(A["LA_2ch"]),
            "LA: A_min(2ch) [mm^2]": np.min(A["LA_2ch"]),
            "LA: A_max(4ch) [mm^2]": np.max(A["LA_4ch"]),
            "LA: A_min(4ch) [mm^2]": np.min(A["LA_4ch"]),
            "LA: V_max(bip) [mL]": LAV_max,
            "LA: V_min(bip) [mL]": LAV_min,
            "LA: Total SV(bip) [mL]": LAV_max - LAV_min,
            "LA: EF_total [%]": (LAV_max - LAV_min) / LAV_max * 100,
            "LA: EI [%]": (LAV_max - LAV_min) / LAV_min * 100,  # expansion index

            # All RA are only determined using 4ch view
            "RA: D_longitudinal [cm]": np.max(L_L["RA_4ch"]),
            "RA: A_max [mm^2]": np.max(A["RA_4ch"]),
            "RA: A_min [mm^2]": np.min(A["RA_4ch"]),
            "RA: V_max [mL]": np.max(V["RA_4ch"]),
            "RA: V_min [mL]": np.min(V["RA_4ch"]),
            "RA: Total SV [mL]": np.max(V["RA_4ch"]) - np.min(V["RA_4ch"]),
            "RA: EF_total [%]": (np.max(V["RA_4ch"]) - np.min(V["RA_4ch"])) / np.max(V["RA_4ch"]) * 100,
            "RA: EI [%]": (np.max(V["RA_4ch"]) - np.min(V["RA_4ch"])) / np.min(V["RA_4ch"]) * 100
        })

        # Save time series of volume and display the time series of atrial volume
        os.makedirs(f"{sub_dir}/timeseries", exist_ok=True)
        logger.info(f"Saving time series of atrial volume for {subject}")
        with open(f"{sub_dir}/timeseries/atrium.pkl", "wb") as time_series_file:
            pickle.dump({
                "LA: Volume(bip) [mL]": V["LA_bip"],
                "RA: Volume(bip) [mL]": V["RA_4ch"]
            }, time_series_file)

        time_grid_point = np.arange(T)
        time_grid_real = time_grid_point * temporal_resolution
        
        fig, ax1, ax2 = plot_time_series_double_x(time_grid_point, time_grid_real, V["LA_bip"], 
                                                  "Time [frame]", "Time [ms]", "Volume [mL]",
                                                  lambda x: x * temporal_resolution,
                                                  lambda x: x / temporal_resolution)
        fig.savefig(f"{sub_dir}/timeseries/atrium_volume_raw.png")
        plt.close(fig)


        # * Implement more advanced features -----------------------------------

        # * Feature1: Indexed volume
        logger.info(f"{subject}: Implement indexed volume features")
        try:
            BSA_info = pd.read_csv(config.BSA_file)[["eid", config.BSA_col_name]]
            BSA_subject = BSA_info[BSA_info["eid"] == int(subject)][config.BSA_col_name].values[0]
        except IndexError:
            logger.error(f"{subject}: BSA information not found, skipped.")
            continue
        feature_dict.update({
            "LA: D_longitudinal(2ch)/BSA [cm/m^2]": np.max(L_L["LA_2ch"]) / BSA_subject,
            "LA: D_longitudinal(4ch)/BSA [cm/m^2]": np.max(L_L["LA_4ch"]) / BSA_subject,
            "LA: A_max(2ch)/BSA [mm^2/m^2]": np.max(A["LA_2ch"]) / BSA_subject,
            "LA: A_min(2ch)/BSA [mm^2/m^2]": np.min(A["LA_2ch"]) / BSA_subject,
            "LA: A_max(4ch)/BSA [mm^2/m^2]": np.max(A["LA_4ch"]) / BSA_subject,
            "LA: A_min(4ch)/BSA [mm^2/m^2]": np.min(A["LA_4ch"]) / BSA_subject,
            "LA: V_max(bip)/BSA [mL/m^2]": LAV_max / BSA_subject,
            "LA: V_min(bip)/BSA [mL/m^2]": LAV_min / BSA_subject,

            "RA: V_max(bip)/BSA [mL/m^2]": np.max(V["RA_4ch"]) / BSA_subject,
            "RA: V_min(bip)/BSA [mL/m^2]": np.min(V["RA_4ch"]) / BSA_subject
        })

        # * Feature2: Transverse diameter
        # Ref https://jcmr-online.biomedcentral.com/articles/10.1186/s12968-020-00683-3
        # Ref https://www.sciencedirect.com/science/article/pii/S1097664723013455
        logger.info(f"{subject}: Implement transverse diameter features")
        feature_dict.update({
            "LA: D_transverse(2ch) [cm]": np.max(L_T["LA_2ch"]),
            "LA: D_transverse(2ch)/BSA [cm/m^2]": np.max(L_T["LA_2ch"]) / BSA_subject,
            "LA: D_transverse(4ch) [cm]": np.max(L_T["LA_4ch"]),
            "LA: D_transverse(4ch)/BSA [cm/m^2]": np.max(L_T["LA_4ch"]) / BSA_subject,

            "RA: D_transverse [cm]": np.max(L_T["RA_4ch"])
        })

        # * Feature3: LA Spherical Index
        logger.info(f"{subject}: Implement LA spherical index features")
        # Ref https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6404907/pdf/JAH3-7-e009793.pdf
        # Define The maximum LA length is chosen among 2 chamber and 4 chamber view
        LAD_L_max = max(np.max(L_L["LA_2ch"]), np.max(L_L["LA_4ch"]))
        LAD_T_max = max(np.max(L_T["LA_2ch"]), np.max(L_T["LA_4ch"]))
        LAD_max = max(LAD_L_max, LAD_T_max)

        V_sphere_max = 4 / 3 * math.pi * (LAD_max / 2) ** 3

        LA_spherical_index = LAV_max / V_sphere_max

        feature_dict.update({
            "LA: Spherical Index [%]": LA_spherical_index
        })

        # * Feature4: Early/Late Peak Emptying Rate
        # Ref https://pubmed.ncbi.nlm.nih.gov/30128617/
        V_LA = V["LA_bip"]
        # V_LA_lowess = sm.nonparametric.lowess(V_LA, time_grid_real, frac=0.1)
        # V_LA_lowess_x = V_LA_lowess[:, 0]
        # V_LA_lowess_y = V_LA_lowess[:, 1]
        fANCOVA = importr('fANCOVA')
        x_r = FloatVector(time_grid_real)
        y_r = FloatVector(V_LA)
        # * We use the lowess provided by R that has Generalized Cross Validation as its criterion
        loess_fit = fANCOVA.loess_as(x_r, y_r,degree = 2,criterion = "gcv")
        V_LA_lowess_x = np.array(loess_fit.rx2('x')).reshape(T,)
        V_LA_lowess_y = np.array(loess_fit.rx2('fitted'))

        V_LA_diff_y = np.diff(V_LA_lowess_y) / np.diff(V_LA_lowess_x) * 1000 # unit: mL/s
        V_LA_diff_x = (V_LA_lowess_x[:-1] + V_LA_lowess_x[1:]) / 2

        T_max = np.argmax(V["LA_bip"])

        try:
            L1 = T_max
            T_peak_pos_1 = find_peaks(V_LA_diff_y[:T_max], distance = math.ceil(L1 / 3))[0]
            T_peak_pos_1 = T_peak_pos_1[np.argmax(V_LA_diff_y[T_peak_pos_1])]

            L2 = len(V_LA_diff_y) - T_max
            T_peak_pos_2 = find_peaks(V_LA_diff_y[T_max:], distance = math.ceil(L2 / 3))[0]
            T_peak_pos_2 = [peak + T_max for peak in T_peak_pos_2]
            T_peak_pos_2 = T_peak_pos_2[np.argmax(V_LA_diff_y[T_peak_pos_2])]

            L3 = T_peak_pos_2 - T_max
            T_peak_neg_1 = find_peaks(-V_LA_diff_y[T_max:T_peak_pos_2], distance = math.ceil(L3 / 3))[0]
            T_peak_neg_1 = [peak + T_max for peak in T_peak_neg_1]
            T_peak_neg_1 = T_peak_neg_1[np.argmax(-V_LA_diff_y[T_peak_neg_1])]

            L4 = len(V_LA_diff_y) - T_peak_pos_2
            T_peak_neg_2 = np.argmax(-V_LA_diff_y[T_peak_pos_2:])
            T_peak_neg_2 = T_peak_neg_2 + T_peak_pos_2
        except Exception as e:
            logger.error(f"{subject}: Error {e} in determining PER, skipped.")
            continue

        colors = ['blue'] * len(V_LA_diff_x)
        colors[T_max] = 'red'
        colors[T_peak_pos_1] = 'orange'
        colors[T_peak_pos_2] = 'orange'
        colors[T_peak_neg_1] = 'yellow'
        colors[T_peak_neg_2] = 'yellow'


        plt.subplot(1,2,1)
        plt.plot(V_LA_lowess_x, V_LA_lowess_y, color='blue')
        plt.scatter(time_grid_real, V_LA, color='blue')
        plt.xlabel("Time [ms]")
        plt.ylabel("Volume [mL]")
        plt.subplot(1,2,2)
        plt.plot(V_LA_diff_x, V_LA_diff_y, color='blue')
        plt.scatter(V_LA_diff_x, V_LA_diff_y, color=colors)
        plt.xlabel("Time [ms]")
        plt.ylabel("dV/dt [mL/s]")
        plt.savefig(f"{sub_dir}/timeseries/atrium_volume_diff.png")
        plt.close()

        feature_dict.update({
            "LA: PER-E [mL/s]": -V_LA_diff_y[T_peak_neg_1],  # absolute value
            "LA: PER-A [mL/s]": -V_LA_diff_y[T_peak_neg_2],
        })

        # * Feature5: Volume before atrial contraction and emptying fraction
        logger.info(f"{subject}: Implement volume before atrial contraction features")
        
        ecg_processor = ECG_Processor(subject, args.retest)

        if config.useECG and not ecg_processor.check_data_rest():
            logger.warning(f"{subject}: No ECG rest data, pre-contraction volume will not be extracted.")
        else:
            logger.info(f"{subject}: ECG rest data exists, extracting pre-contraction volume.")

            try:
                time_points_dict = ecg_processor.determine_timepoint_LA()
                t_max_ecg = time_points_dict["t_max"]
                T_max_ecg = round(t_max_ecg / temporal_resolution)
                t_pre_a_ecg = time_points_dict["t_pre_a"]
                T_pre_a_ecg = round(t_pre_a_ecg / temporal_resolution)
            except ValueError as e:
                logger.error(f"{subject}: Error {e} in determining time points using ECG, skipped.")
                continue

            if T_pre_a_ecg >= T:
                logger.error(f"{subject}: Pre-contraction time frame is out of range, skipped.")
                continue
            elif T_pre_a_ecg < T_peak_pos_2 or T_pre_a_ecg > T_peak_neg_2:
                # The pre-contraction time frame should be in the middle of the 
                # second positive and the second negative peak
                logger.warning(f"{subject}: Quality control for pre-contratcion time fails, " \
                                "no relevant feature will be extracted.")
            else:
                colors = ['blue'] * len(V["LA_bip"])
                colors[T_max] = 'red'
                colors[T_max_ecg] = 'orange'
                colors[T_pre_a_ecg] = 'yellow'

                fig, ax1, ax2 = plot_time_series_double_x_y(time_grid_point, time_grid_real, V_LA_lowess_y, V["LA_bip"],
                                        "Time [frame]", "Time [ms]", "Volume [mL]",
                                        lambda x: x * temporal_resolution,
                                        lambda x: x / temporal_resolution,
                                        colors = colors)
                fig.savefig(f"{sub_dir}/timeseries/atrium_volume.png")
                plt.close(fig)

                # Since the volumes around pre-contraction stage is highly variant, 
                # we use previously-obtained smoothed values
                LAV_pre_a = V_LA_lowess_y[np.argmin(abs(V_LA_lowess_x - t_pre_a_ecg))]

                # closest time to the pre-contraction time in LOWESS
                feature_dict.update({
                    "LA: V_pre_a [mL]": LAV_pre_a,
                    "LA: V_pre_a/BSA [mL/m^2]": LAV_pre_a / BSA_subject
                })
                logger.info(f"{subject}: Pre-contraction volume extracted successfully.")

                # Ref https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4508385/pdf/BMRI2015-765921.pdf
                # Ref https://pubmed.ncbi.nlm.nih.gov/24291276/
                feature_dict.update({
                    "LA: V_pre_a [mL]": LAV_pre_a,
                    "LA: V_pre_a/BSA [mL/m^2]": LAV_pre_a / BSA_subject,
                    "LA: EF_booster [%]": (LAV_pre_a - LAV_min) / LAV_pre_a * 100,  # also called EF_active
                    "LA: EF_conduit [%]": (LAV_max - LAV_pre_a) / LAV_max * 100,  # also called EF_passive
                })

        # todo: IPVT, IPVT Ratio, LA conduit volume, LA functional index, Average atrial contribution

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    # Save the features to a csv file
    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "atrium")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"))
