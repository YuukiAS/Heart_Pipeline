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
from matplotlib.patches import FancyArrowPatch
import pandas as pd
import nibabel as nib
import vtk
import math
from tqdm import tqdm
from scipy.signal import find_peaks
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ECG_6025_20205.ecg_processor import ECG_Processor

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config

from utils.log_utils import setup_logging
from utils.quality_control_utils import atrium_pass_quality_control
from utils.cardiac_utils import evaluate_atrial_area_length
from utils.analyze_utils import plot_time_series_dual_axes, plot_time_series_dual_axes_double_y, analyze_time_series_derivative
from utils.biobank_utils import query_BSA

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
        la_2ch_name = f"{sub_dir}/la_2ch.nii.gz"
        la_4ch_name = f"{sub_dir}/la_4ch.nii.gz"
        seg_la_2ch_name = f"{sub_dir}/seg_la_2ch.nii.gz"
        seg_la_4ch_name = f"{sub_dir}/seg_la_4ch.nii.gz"

        if not os.path.exists(sa_name):
            logger.error(f"Short axis file for {subject} does not exist.")
            continue

        if not os.path.exists(seg_la_2ch_name) and not os.path.exists(seg_la_4ch_name):
            logger.error(f"Segmentation of long axis file for {subject} does not exist.")
            continue

        if not os.path.exists(seg_la_2ch_name):
            logger.warning(f"Segmentation of 2-chamber long axis file for {subject} does not exist.")

        if not os.path.exists(seg_la_4ch_name):
            logger.warning(f"Segmentation of 4-chamber long axis file for {subject} does not exist.")

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
        nim_la_2ch = nib.load(la_2ch_name)
        nim_la_4ch = nib.load(la_4ch_name)
        la_2ch = nim_la_2ch.get_fdata()
        la_4ch = nim_la_4ch.get_fdata()
        seg_la_2ch = nib.load(seg_la_2ch_name).get_fdata()
        seg_la_2ch_nan = np.where(seg_la_2ch == 0, np.nan, seg_la_2ch)
        seg_la_4ch = nib.load(seg_la_4ch_name).get_fdata()
        seg_la_4ch_nan = np.where(seg_la_4ch == 0, np.nan, seg_la_4ch)
        long_axis = nim_sa.affine[:3, 2] / np.linalg.norm(nim_sa.affine[:3, 2])  # used to determine top and bottom of atrium
        if long_axis[2] < 0:
            long_axis *= -1  # make sure distance is positive

        if os.path.exists(seg_la_2ch_name):
            # Analyse 2 chamber view image
            nim_2ch = nib.load(seg_la_2ch_name)
            short_axis = nim_2ch.affine[:3, 2] / np.linalg.norm(nim_2ch.affine[:3, 2])
            if short_axis[2] < 0:
                short_axis *= -1
            seg_la_2ch = nim_2ch.get_fdata()
            T = nim_2ch.header["dim"][4]  # number of time frames
            temporal_resolution = nim_2ch.header["pixdim"][4] * 1000  # unit: ms

            # Perform quality control for the segmentation
            if not atrium_pass_quality_control(seg_la_2ch, {"LA": 1}):
                logger.error(f"{subject}: seg_la_2ch does not pass quality control, skipped.")
                continue

            A["LA_2ch"] = np.zeros(T)
            L_L["LA_2ch"] = np.zeros(T)
            L_T["LA_2ch"] = np.zeros(T)
            V["LA_2ch"] = np.zeros(T)
            lm["2ch"] = {}
            for t in range(T):
                try:
                    area, length, landmarks = evaluate_atrial_area_length(seg_la_2ch[:, :, 0, t], nim_2ch, long_axis, short_axis)
                except ValueError as e:
                    logger.error(f"Error in evaluating atrial area and length for {subject} at time frame {t}: {e}")
                    continue

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
                        points.InsertNextPoint(p[0], p[1], 0)
                    poly = vtk.vtkPolyData()
                    poly.SetPoints(points)
                    writer = vtk.vtkPolyDataWriter()
                    writer.SetInputData(poly)
                    os.makedirs(f"{sub_dir}/landmark", exist_ok=True)
                    writer.SetFileName(f"{sub_dir}/landmark/atrium_la_2ch_{t:02d}.vtk")
                    writer.Write()
        else:
            logger.error(f"Segmentation of 2-chamber long axis file for {subject} does not exist.")
            continue

        if os.path.exists(seg_la_4ch_name):
            # Analyse 4 chamber view image
            nim_4ch = nib.load(seg_la_4ch_name)
            short_axis = nim_4ch.affine[:3, 2] / np.linalg.norm(nim_4ch.affine[:3, 2])
            if short_axis[2] < 0:
                short_axis *= -1
            seg_la_4ch = nim_4ch.get_fdata()

            # Perform quality control for the segmentation
            if not atrium_pass_quality_control(seg_la_4ch, {"LA": 1, "RA": 2}):
                logger.error(f"{subject} seg_la_4ch does not pass quality control, skipped.")
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
                    area, length, landmarks = evaluate_atrial_area_length(seg_la_4ch[:, :, 0, t], nim_4ch, long_axis, short_axis)
                except ValueError as e:
                    logger.error(f"Error in evaluating atrial area and length for {subject} at time frame {t}: {e}")
                    continue

                A["LA_4ch"][t] = area[0]
                L_L["LA_4ch"][t] = length[0]
                L_T["LA_4ch"][t] = length[1]
                V["LA_4ch"][t] = 8 / (3 * math.pi) * area[0] * area[0] / length[0]

                # * We only report the LA volume calculated using the biplane area-length formula (using both modality)
                V["LA_bip"][t] = 8 / (3 * math.pi) * area[0] * A["LA_2ch"][t] / (0.5 * (length[0] + L_L["LA_2ch"][t]))

                A["RA_4ch"][t] = area[1]
                L_L["RA_4ch"][t] = length[2]  # LA and RA are processed sequentially
                L_T["RA_4ch"][t] = length[3]
                V["RA_4ch"][t] = 8 / (3 * math.pi) * area[1] * area[1] / length[2]
                lm["4ch"][t] = landmarks

                if t == 0:
                    # Write the landmarks to a vtk file
                    points = vtk.vtkPoints()
                    for p in landmarks:
                        points.InsertNextPoint(p[0], p[1], 0)
                    poly = vtk.vtkPolyData()
                    poly.SetPoints(points)
                    writer = vtk.vtkPolyDataWriter()
                    writer.SetInputData(poly)
                    os.makedirs(f"{sub_dir}/landmark", exist_ok=True)
                    writer.SetFileName(f"{sub_dir}/landmark/atrium_la_4ch_{t:02d}.vtk")
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
        T_max = np.argmax(V["LA_bip"])
        T_min = np.argmin(V["LA_bip"])

        feature_dict.update(
            {
                # LA are determined using both 2ch and 4ch view
                # define The distance between the posterior wall of atrium and center of valve plane
                "LA: D_longitudinal (2ch) [cm]": np.max(L_L["LA_2ch"]),
                "LA: D_longitudinal (4ch) [cm]": np.max(L_L["LA_4ch"]),
                "LA: A_max (2ch) [cm^2]": np.max(A["LA_2ch"]),
                "LA: A_min (2ch) [cm^2]": np.min(A["LA_2ch"]),
                "LA: A_max (4ch) [cm^2]": np.max(A["LA_4ch"]),
                "LA: A_min (4ch) [cm^2]": np.min(A["LA_4ch"]),
                "LA: V_max (bip) [mL]": LAV_max,
                "LA: V_min (bip) [mL]": LAV_min,
                "LA: Total SV (bip) [mL]": LAV_max - LAV_min,
                "LA: EF_total [%]": (LAV_max - LAV_min) / LAV_max * 100,
                # All RA are only determined using 4ch view
                "RA: D_longitudinal [cm]": np.max(L_L["RA_4ch"]),
                "RA: A_max [cm^2]": np.max(A["RA_4ch"]),
                "RA: A_min [cm^2]": np.min(A["RA_4ch"]),
                "RA: V_max [mL]": np.max(V["RA_4ch"]),
                "RA: V_min [mL]": np.min(V["RA_4ch"]),
                # "RA: Total SV [mL]": np.max(V["RA_4ch"]) - np.min(V["RA_4ch"]),
                "RA: EF_total [%]": (np.max(V["RA_4ch"]) - np.min(V["RA_4ch"])) / np.max(V["RA_4ch"]) * 100,
                # "RA: EI [%]": (np.max(V["RA_4ch"]) - np.min(V["RA_4ch"])) / np.min(V["RA_4ch"]) * 100,
            }
        )

        LA_EI = (LAV_max - LAV_min) / LAV_min * 100
        RA_EI = (np.max(V["RA_4ch"]) - np.min(V["RA_4ch"])) / np.min(V["RA_4ch"]) * 100

        if LA_EI < 300:
            feature_dict.update({"LA: EI [%]": LA_EI})
        else:
            logger.warning(f"{subject}: Extremely high LA Expansion Index detected, skipped.")
        if RA_EI < 300:
            feature_dict.update({"RA: EI [%]": RA_EI})
        else:
            logger.warning(f"{subject}: Extremely high RA Expansion Index detected, skipped.")

        # Save time series of volume and display the time series of atrial volume
        os.makedirs(f"{sub_dir}/timeseries", exist_ok=True)
        logger.info(f"{subject}: Saving atrial volume and time data")
        os.makedirs(f"{sub_dir}/timeseries", exist_ok=True)
        data_time = {
            "LA: Volume (bip) [mL]": V["LA_bip"],
            "RA: Volume (bip) [mL]": V["RA_4ch"],
            "LA: T_max": T_max,
            "LA: T_min": T_min,
        }
        np.savez(f"{sub_dir}/timeseries/atrium.npz", **data_time)

        # Visualize the segmentations
        logger.info(f"{subject}: Visualizing atrial segmentation on long-axis images")
        os.makedirs(f"{sub_dir}/visualization/atrium", exist_ok=True)
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        ax[0, 0].imshow(la_2ch[:, :, 0, T_max], cmap="gray")
        ax[0, 0].imshow(seg_la_2ch_nan[:, :, 0, T_max], cmap="jet", alpha=0.5)
        ax[0, 0].title.set_text("2-chamber view: Maximal Volume")
        ax[0, 1].imshow(la_4ch[:, :, 0, T_max], cmap="gray")
        ax[0, 1].imshow(seg_la_4ch_nan[:, :, 0, T_max], cmap="jet", alpha=0.5)
        ax[0, 1].title.set_text("4-chamber view: Maximal Volume")
        ax[1, 0].imshow(la_2ch[:, :, 0, T_min], cmap="gray")
        ax[1, 0].imshow(seg_la_2ch_nan[:, :, 0, T_min], cmap="jet", alpha=0.5)
        ax[1, 0].title.set_text("2-chamber view: Minimal Volume")
        ax[1, 1].imshow(la_4ch[:, :, 0, T_min], cmap="gray")
        ax[1, 1].imshow(seg_la_4ch_nan[:, :, 0, T_min], cmap="jet", alpha=0.5)
        ax[1, 1].title.set_text("4-chamber view: Minimal Volume")
        plt.tight_layout()
        plt.savefig(f"{sub_dir}/visualization/atrium/seg_la.png")
        plt.close(fig)

        time_grid_point = np.arange(T)
        time_grid_real = time_grid_point * temporal_resolution  # unit: ms

        fig, ax1, ax2 = plot_time_series_dual_axes(
            time_grid_point,
            V["LA_bip"],
            "Time [frame]",
            "Time [ms]",
            "Volume [mL]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Atrial Volume Time Series",
        )
        fig.savefig(f"{sub_dir}/timeseries/atrium_volume_raw.png")
        plt.close(fig)

        # * Implement more advanced features -----------------------------------

        # * Feature1: Indexed volume
        logger.info(f"{subject}: Implement indexed volume features")
        try:
            BSA_subject = query_BSA(subject)
        except (FileNotFoundError, IndexError):
            # As BSA is a crucial feature for subsequent features, we skip the subject if BSA is not found
            logger.error(f"{subject}: BSA information not found, skipped.")
            continue
        feature_dict.update(
            {
                "LA: D_longitudinal/BSA (2ch) [cm/m^2]": np.max(L_L["LA_2ch"]) / BSA_subject,
                "LA: D_longitudinal/BSA (4ch) [cm/m^2]": np.max(L_L["LA_4ch"]) / BSA_subject,
                "LA: A_max/BSA (2ch) [mm^2/m^2]": np.max(A["LA_2ch"]) / BSA_subject,
                "LA: A_min/BSA (2ch) [mm^2/m^2]": np.min(A["LA_2ch"]) / BSA_subject,
                "LA: A_max/BSA (4ch) [mm^2/m^2]": np.max(A["LA_4ch"]) / BSA_subject,
                "LA: A_min/BSA (4ch) [mm^2/m^2]": np.min(A["LA_4ch"]) / BSA_subject,
                "LA: V_max/BSA (bip) [mL/m^2]": LAV_max / BSA_subject,
                "LA: V_min/BSA (bip) [mL/m^2]": LAV_min / BSA_subject,
                "RA: D_longitudinal/BSA [cm/m^2]": np.max(L_L["RA_4ch"]) / BSA_subject,
                "RA: A_max/BSA [mm^2/m^2]": np.max(A["RA_4ch"]) / BSA_subject,
                "RA: A_min/BSA [mm^2/m^2]": np.min(A["RA_4ch"]) / BSA_subject,
                "RA: V_max/BSA (bip) [mL/m^2]": np.max(V["RA_4ch"]) / BSA_subject,
                "RA: V_min/BSA (bip) [mL/m^2]": np.min(V["RA_4ch"]) / BSA_subject,
            }
        )

        # * Feature2: Transverse diameter
        # Ref Reference left atrial dimensions and volumes by steady state free precession cardiovascular magnetic resonance https://www.sciencedirect.com/science/article/pii/S1097664723013455
        # Ref Reference right atrial dimensions and volume estimation  https://doi.org/10.1186/1532-429X-15-29
        logger.info(f"{subject}: Implement transverse diameter features")
        feature_dict.update(
            {
                # define Obtained perpendicular to longitudinal diameter, at the mid-level of the atrium
                "LA: D_transverse (2ch) [cm]": np.max(L_T["LA_2ch"]),
                "LA: D_transverse/BSA (2ch) [cm/m^2]": np.max(L_T["LA_2ch"]) / BSA_subject,
                "LA: D_transverse (4ch) [cm]": np.max(L_T["LA_4ch"]),
                "LA: D_transverse/BSA (4ch) [cm/m^2]": np.max(L_T["LA_4ch"]) / BSA_subject,
                "RA: D_transverse [cm]": np.max(L_T["RA_4ch"]),
                "RA: D_transverse/BSA [cm/m^2]": np.max(L_T["RA_4ch"]) / BSA_subject,
            }
        )

        # Visualize the landmarks for longitudinal and transverse diameters
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(la_2ch[:, :, 0, np.argmax(L_L["LA_2ch"])], cmap="gray")
        x_coords = [p[1] for p in lm["2ch"][np.argmax(L_L["LA_2ch"])][:2]]
        y_coords = [p[0] for p in lm["2ch"][np.argmax(L_L["LA_2ch"])][:2]]
        plt.scatter(x_coords, y_coords, c="r", label="Longitudinal", s=8)
        plt.plot(x_coords, y_coords, c="r", linestyle='--')
        plt.title("Maximum LA Longitudinal Diameter (2ch)")
        plt.text(
            0.5,
            -0.1,
            f"Longitudinal Diameter is {feature_dict['LA: D_longitudinal (2ch) [cm]']:.2f} cm",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.legend(loc="lower right")
        plt.subplot(1, 2, 2)
        plt.imshow(la_2ch[:, :, 0, np.argmax(L_T["LA_2ch"])], cmap="gray")
        x_coords = [p[1] for p in lm["2ch"][np.argmax(L_T["LA_2ch"])][2:]]
        y_coords = [p[0] for p in lm["2ch"][np.argmax(L_T["LA_2ch"])][2:]]
        plt.scatter(x_coords, y_coords, c="b", label="Transverse", s=8)
        plt.plot(x_coords, y_coords, c="b", linestyle='--')
        plt.title("Maximum LA Transverse Diameter (2ch)")
        plt.text(
            0.5,
            -0.1,
            f"Transverse Diameter is {feature_dict['LA: D_transverse (2ch) [cm]']:.2f} cm",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.legend(loc="lower right")
        plt.savefig(f"{sub_dir}/visualization/atrium/la_2ch_diameter.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(la_4ch[:, :, 0, np.argmax(L_L["LA_4ch"])], cmap="gray")
        x_coords = [p[1] for p in lm["4ch"][np.argmax(L_L["LA_4ch"])][:2]]
        y_coords = [p[0] for p in lm["4ch"][np.argmax(L_L["LA_4ch"])][:2]]
        plt.scatter(x_coords, y_coords, c="r", label="Longitudinal", s=8)
        plt.plot(x_coords, y_coords, c="r", linestyle='--')
        plt.title("Maximum LA Longitudinal Diameter (4ch)")
        plt.text(
            0.5,
            -0.1,
            f"Longitudinal Diameter is {feature_dict['LA: D_longitudinal (4ch) [cm]']:.2f} cm",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.legend(loc="lower right")
        plt.subplot(1, 2, 2)
        plt.imshow(la_4ch[:, :, 0, np.argmax(L_T["LA_4ch"])], cmap="gray")
        x_coords = [p[1] for p in lm["4ch"][np.argmax(L_T["LA_4ch"])][2:4]]
        y_coords = [p[0] for p in lm["4ch"][np.argmax(L_T["LA_4ch"])][2:4]]
        plt.scatter(x_coords, y_coords, c="b", label="Transverse", s=8)
        plt.plot(x_coords, y_coords, c="b", linestyle='--')
        plt.title("Maximum LA Transverse Diameter (4ch)")
        plt.text(
            0.5,
            -0.1,
            f"Transverse Diameter is {feature_dict['LA: D_transverse (4ch) [cm]']:.2f} cm",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.legend(loc="lower right")
        plt.savefig(f"{sub_dir}/visualization/atrium/la_4ch_diameter.png")
        plt.close()

        # * Feature3: LA Spherical Index
        logger.info(f"{subject}: Implement LA spherical index features")
        # Ref Incremental Value of Left Atrial Geometric Remodeling in Predicting Late Atrial Fibrillation Recurrence https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6404907/pdf/JAH3-7-e009793.pdf
        # define The maximum LA length is chosen among 2 chamber and 4 chamber view
        LAD_L_max = max(np.max(L_L["LA_2ch"]), np.max(L_L["LA_4ch"]))
        LAD_T_max = max(np.max(L_T["LA_2ch"]), np.max(L_T["LA_4ch"]))
        LAD_max = max(LAD_L_max, LAD_T_max)

        V_sphere_max = 4 / 3 * math.pi * (LAD_max / 2) ** 3

        LA_spherical_index = LAV_max / V_sphere_max
        if LA_spherical_index > 5:
            logger.warning(f"{subject}: Extremely high LA spherical index detected, skipped.")
        else:
            feature_dict.update({"LA: Sphericity_Index": LA_spherical_index})

        # * Feature4: Early/Late Peak Emptying Rate
        # Ref Diastolic dysfunction evaluated by cardiac magnetic resonance https://pubmed.ncbi.nlm.nih.gov/30128617/
        V_LA = V["LA_bip"]

        fANCOVA = importr("fANCOVA")
        x_r = FloatVector(time_grid_real)
        y_r = FloatVector(V_LA)
        # We use the loess provided by R that has Generalized Cross Validation as its criterion
        loess_fit = fANCOVA.loess_as(x_r, y_r, degree=2, criterion="gcv")
        V_LA_loess_x = np.array(loess_fit.rx2("x")).reshape(
            T,
        )
        V_LA_loess_y = np.array(loess_fit.rx2("fitted"))

        # * Instead of using np.diff, we use np.gradient to ensure the same length
        V_LA_diff_y = np.gradient(V_LA_loess_y, V_LA_loess_x) * 1000  # unit: mL/s

        try:
            # These time points are used for quality control of ECG timepoint and visualization, not for calculation of PER
            # Generally, it should be positive peak -> negative peak -> positive peak -> negative peak

            L1 = T_max
            T_peak_pos_1 = find_peaks(V_LA_diff_y[:T_max], distance=math.ceil(L1 / 3))[0]
            T_peak_pos_1 = T_peak_pos_1[np.argmax(V_LA_diff_y[T_peak_pos_1])]

            L2 = len(V_LA_diff_y) - T_max
            T_peak_pos_2 = find_peaks(V_LA_diff_y[T_max:], distance=math.ceil(L2 / 3))[0]
            T_peak_pos_2 = [peak + T_max for peak in T_peak_pos_2]
            T_peak_pos_2 = T_peak_pos_2[np.argmax(V_LA_diff_y[T_peak_pos_2])]

            L3 = T_peak_pos_2 - T_max
            T_peak_neg_1 = find_peaks(-V_LA_diff_y[T_max:T_peak_pos_2], distance=math.ceil(L3 / 3))[0]
            T_peak_neg_1 = [peak + T_max for peak in T_peak_neg_1]
            T_peak_neg_1 = T_peak_neg_1[np.argmax(-V_LA_diff_y[T_peak_neg_1])]

            # No need to find peaks for the last segment of cardiac cycle
            L4 = len(V_LA_diff_y) - T_peak_pos_2
            T_peak_neg_2 = np.argmax(-V_LA_diff_y[T_peak_pos_2:])
            T_peak_neg_2 = T_peak_neg_2 + T_peak_pos_2
        except Exception as e:
            logger.error(f"{subject}: Error {e} in determining PER, skipped.")
            continue

        # * We calculate these rates through the slope of smoothed volumes of a few adjacent frames to avoid overestimation
        try:
            _, T_PER, _, PER = analyze_time_series_derivative(time_grid_real, V_LA, n_pos=0, n_neg=2)
            logger.info(f"{subject}: Implementing peak emptying rate features")
            # define PER_E: Early atrial peak emptying rate, PER_A: Late atrial peak emptying rate
            PER_E = PER[np.argmin(T_PER)] * 1000  # unit: mL/s
            PER_A = PER[np.argmax(T_PER)] * 1000  # unit: mL/s
            if PER_E >= 0:
                raise ValueError("PER_E should not be positive.")
            if PER_A >= 0:
                raise ValueError("PER_A should not be positive.")
            if abs(PER_E) < 10 or abs(PER_A) < 10:
                raise ValueError("Extremely small PER values detected, skipped.")
            if abs(PER_E / PER_A) > 5:
                logger.warning(f"{subject}: Extremely high PER-E/PER-A detected, skipped.")
            else:
                feature_dict.update(
                    {
                        "LA: PER-E [mL/s]": abs(PER_E),  # positive value should be reported; convert ms to s
                        "LA: PER-A [mL/s]": abs(PER_A),
                        "LA: PER-E/BSA [mL/s/m^2]": abs(PER_E) / BSA_subject,
                        "LA: PER-A/BSA [mL/s/m^2]": abs(PER_A) / BSA_subject,
                        "LA: PER-E/PER-A": abs(PER_E / PER_A),
                    }
                )
        except (IndexError, ValueError) as e:
            logger.warning(f"{subject}: PER calculation failed: {e}")

        # We will postpone the visualization to the end, after the time point of atrial contribution is determined.

        # * Feature5: Volume before atrial contraction and emptying fraction
        logger.info(f"{subject}: Implement volume before atrial contraction features")

        ecg_processor = ECG_Processor(subject, args.retest)

        if config.useECG and not ecg_processor.check_data_rest():
            logger.warning(f"{subject}: No ECG rest data, pre-contraction volume will not be extracted.")
        else:
            logger.info(f"{subject}: ECG rest data exists, extracting pre-contraction volume.")

            try:
                # The atrial contraction time point is determined using 12-lead rest ECG
                time_points_dict = ecg_processor.determine_timepoint_LA()
                t_max_ecg = time_points_dict["t_max"]
                T_max_ecg = round(t_max_ecg / temporal_resolution)
                t_pre_a_ecg = time_points_dict["t_pre_a"]
                T_pre_a_ecg = round(t_pre_a_ecg / temporal_resolution)
            except ValueError as e:
                logger.error(f"{subject}: Error {e} in determining atrial contraction time point using ECG, skipped.")
                continue

            if T_pre_a_ecg >= T:
                logger.error(f"{subject}: Pre-contraction time frame is out of range, skipped.")
                continue
            elif T_pre_a_ecg < T_peak_pos_2 or T_pre_a_ecg > T_peak_neg_2:
                # The pre-contraction time frame should be in the middle of the
                # second positive and the second negative peak
                logger.warning(
                    f"{subject}: Quality control for pre-contraction time fails, no relevant feature will be extracted."
                )
            else:
                LAV_pre_a = V["LA_bip"][T_pre_a_ecg]

                colors = ["blue"] * len(V["LA_bip"])
                colors[T_max] = "red"
                colors[T_pre_a_ecg] = "orange"
                fig, ax1, ax2 = plot_time_series_dual_axes_double_y(
                    time_grid_point,
                    V_LA_loess_y,
                    V["LA_bip"],
                    "Time [frame]",
                    "Time [ms]",
                    "Volume [mL]",
                    lambda x: x * temporal_resolution,
                    lambda x: x / temporal_resolution,
                    title=f"Subject {subject}: Atrial Volume Time Series (Smoothed)",
                    colors=colors,
                )
                ax1.axvline(x=T_max, color="red", linestyle="--", alpha=0.7)
                ax1.text(
                    T_max,
                    ax1.get_ylim()[1],
                    "Maximum LA Volume",
                    verticalalignment="bottom",
                    horizontalalignment="center",
                    fontsize=6,
                    color="black",
                )
                ax1.axvline(x=T_pre_a_ecg, color="orange", linestyle="--", alpha=0.7)
                (
                    ax1.text(
                        T_pre_a_ecg,
                        ax1.get_ylim()[1],
                        "Atrial Contraction",
                        verticalalignment="bottom",
                        horizontalalignment="center",
                        fontsize=6,
                        color="black",
                    ),
                )
                box_text = (
                    "LA Volume\n"
                    f"Maximum: {LAV_max:.2f} mL\n"
                    f"Minimum: {LAV_min:.2f} mL\n"
                    f"Prior to Atrial Contraction: {LAV_pre_a:.2f} mL"
                )
                box_props = dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8)
                ax1.text(
                    0.98,
                    0.95,
                    box_text,
                    transform=ax1.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=box_props,
                )
                # Explicitly annotate three phases
                arrow_reservoir = FancyArrowPatch(
                    (0, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                    (T_max, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                    arrowstyle="<->",
                    linestyle="--",
                    color="black",
                    alpha=0.7,
                    mutation_scale=15,
                )
                ax1.add_patch(arrow_reservoir)
                ax1.text(
                    T_max / 2,
                    ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
                    "Reservoir",
                    fontsize=6,
                    color="black",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                arrow_conduit = FancyArrowPatch(
                    (T_max, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                    (T_pre_a_ecg, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                    arrowstyle="<->",
                    linestyle="--",
                    color="black",
                    alpha=0.7,
                    mutation_scale=15,
                )
                ax1.add_patch(arrow_conduit)
                ax1.text(
                    (T_max + T_pre_a_ecg) / 2,
                    ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
                    "Conduit",
                    fontsize=6,
                    color="black",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                arrow_booster = FancyArrowPatch(
                    (T_pre_a_ecg, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                    (ax1.get_xlim()[1], ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                    arrowstyle="<->",
                    linestyle="--",
                    color="black",
                    alpha=0.7,
                    mutation_scale=15,
                )
                ax1.add_patch(arrow_booster)
                ax1.text(
                    (T_pre_a_ecg + ax1.get_xlim()[1]) / 2,
                    ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
                    "Booster",
                    fontsize=6,
                    color="black",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                fig.savefig(f"{sub_dir}/timeseries/atrium_volume.png")
                plt.close(fig)

                # update npz
                data_time.update({"LA: T_pre_a": T_pre_a_ecg})
                np.savez(f"{sub_dir}/timeseries/atrium.npz", **data_time)
                logger.info(f"{subject}: Pre-contraction volume extracted successfully.")

                # Ref Left Atrial Size and Function: role in prognosis https://pubmed.ncbi.nlm.nih.gov/24291276/
                feature_dict.update(
                    {
                        "LA: V_pre_a [mL]": LAV_pre_a,
                        "LA: V_pre_a/BSA [mL/m^2]": LAV_pre_a / BSA_subject,
                        "LA: EF_booster [%]": (LAV_pre_a - LAV_min) / LAV_pre_a * 100,  # also called EF_active
                        "LA: EF_conduit [%]": (LAV_max - LAV_pre_a) / LAV_max * 100,  # also called EF_passive
                    }
                )

            # * We now visualize the curve of derivative of volume
            # * As PER are derived using multiple adjacenet frames, the PER will be slower than the values in figure.
            if "LA: PER-E [mL/s]" in feature_dict:
                colors = ["blue"] * len(V_LA_diff_y)
                colors[T_max] = "red"
                colors[T_pre_a_ecg] = "orange"
                colors[T_peak_neg_1] = "aqua"
                colors[T_peak_neg_2] = "lime"
                fig, ax1, ax2 = plot_time_series_dual_axes(
                    time_grid_point,
                    V_LA_diff_y,
                    "Time [frame]",
                    "Time [ms]",
                    "dV/dt [mL/s]",
                    lambda x: x * temporal_resolution,
                    lambda x: x / temporal_resolution,
                    title=f"Subject {subject}: Derivative of Atrial Volume Time Series",
                    colors=colors,
                )
                ax1.axvline(x=T_max, color="red", linestyle="--", alpha=0.7)
                ax1.text(
                    T_max,
                    ax1.get_ylim()[1],
                    "Maximum LA Volume",
                    verticalalignment="bottom",
                    horizontalalignment="center",
                    fontsize=6,
                    color="black",
                )
                ax1.axvline(x=T_pre_a_ecg, color="orange", linestyle="--", alpha=0.7)
                (
                    ax1.text(
                        T_pre_a_ecg,
                        ax1.get_ylim()[1],
                        "Atrial Contraction",
                        verticalalignment="bottom",
                        horizontalalignment="center",
                        fontsize=6,
                        color="black",
                    ),
                )
                ax1.axvline(x=T_peak_neg_1, color="aqua", linestyle="--", alpha=0.7)
                ax1.text(
                    T_peak_neg_1,
                    ax1.get_ylim()[0],
                    "PER-E",
                    verticalalignment="bottom",
                    horizontalalignment="center",
                    fontsize=6,
                    color="black",
                )
                ax1.axvline(x=T_peak_neg_2, color="lime", linestyle="--", alpha=0.7)
                (
                    ax1.text(
                        T_peak_neg_2,
                        ax1.get_ylim()[0],
                        "PER-A",
                        verticalalignment="bottom",
                        horizontalalignment="center",
                        fontsize=6,
                        color="black",
                    ),
                )
                box_text = (
                    "Peak Emptying Rate\n"
                    "(Obtained through multiple frames)\n"
                    f"PER-E: {abs(PER_E):.2f} mL/s\n"
                    f"PER-A: {abs(PER_A):.2f} mL/s"
                )
                box_props = dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8)
                ax1.text(
                    0.98,
                    0.95,
                    box_text,
                    transform=ax1.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=box_props,
                )
                arrow_reservoir = FancyArrowPatch(
                    (0, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                    (T_max, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                    arrowstyle="<->",
                    linestyle="--",
                    color="black",
                    alpha=0.7,
                    mutation_scale=15,
                )
                ax1.add_patch(arrow_reservoir)
                ax1.text(
                    T_max / 2,
                    ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
                    "Reservoir",
                    fontsize=6,
                    color="black",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                arrow_conduit = FancyArrowPatch(
                    (T_max, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                    (T_pre_a_ecg, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                    arrowstyle="<->",
                    linestyle="--",
                    color="black",
                    alpha=0.7,
                    mutation_scale=15,
                )
                ax1.add_patch(arrow_conduit)
                ax1.text(
                    (T_max + T_pre_a_ecg) / 2,
                    ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
                    "Conduit",
                    fontsize=6,
                    color="black",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                arrow_booster = FancyArrowPatch(
                    (T_pre_a_ecg, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                    (ax1.get_xlim()[1], ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                    arrowstyle="<->",
                    linestyle="--",
                    color="black",
                    alpha=0.7,
                    mutation_scale=15,
                )
                ax1.add_patch(arrow_booster)
                ax1.text(
                    (T_pre_a_ecg + ax1.get_xlim()[1]) / 2,
                    ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
                    "Booster",
                    fontsize=6,
                    color="black",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                fig.savefig(f"{sub_dir}/timeseries/atrium_volume_rate.png")
                plt.close(fig)

        # * Plot volume curve together with ECG

        if config.useECG and ecg_processor.check_data_rest():
            logger.info(f"{subject}: Visualize atrial volume alongside with ECG")

            ecg_info = ecg_processor.visualize_ecg_info()
            time = ecg_info["time"]
            signal = ecg_info["signal"]
            P_index = ecg_info["P_index"]
            Q_index = ecg_info["Q_index"]
            R_index = ecg_info["R_index"]
            S_index = ecg_info["S_index"]
            T_index = ecg_info["T_index"]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax1.plot(time_grid_real, V["LA_bip"], color="r")
            ax1.set_xlabel("Time [ms]")
            ax1.set_ylabel("Volume [mL]")
            ax2.plot(time, signal, color="b")
            ax2.set_ylabel("ECG Signal")
            ax2.set_yticks([])

            # Similar annotations as before
            ax1.axvline(x=T_max * temporal_resolution, color="red", linestyle="--", alpha=0.7)
            ax1.text(
                T_max * temporal_resolution,
                ax1.get_ylim()[1],
                "Maximum LA Volume",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=6,
                color="black",
            )
            ax1.axvline(x=T_pre_a_ecg * temporal_resolution, color="orange", linestyle="--", alpha=0.7)
            (
                ax1.text(
                    T_pre_a_ecg * temporal_resolution,
                    ax1.get_ylim()[1],
                    "Atrial Contraction",
                    verticalalignment="bottom",
                    horizontalalignment="center",
                    fontsize=6,
                    color="black",
                ),
            )
            arrow_reservoir = FancyArrowPatch(
                (0, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                (T_max * temporal_resolution, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                arrowstyle="<->",
                linestyle="--",
                color="black",
                alpha=0.7,
                mutation_scale=15,
            )
            ax1.add_patch(arrow_reservoir)
            ax1.text(
                T_max * temporal_resolution / 2,
                ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
                "Reservoir",
                fontsize=6,
                color="black",
                horizontalalignment="center",
                verticalalignment="center",
            )
            arrow_conduit = FancyArrowPatch(
                (T_max * temporal_resolution, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                (T_pre_a_ecg * temporal_resolution, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                arrowstyle="<->",
                linestyle="--",
                color="black",
                alpha=0.7,
                mutation_scale=15,
            )
            ax1.add_patch(arrow_conduit)
            ax1.text(
                (T_max + T_pre_a_ecg) * temporal_resolution / 2,
                ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
                "Conduit",
                fontsize=6,
                color="black",
                horizontalalignment="center",
                verticalalignment="center",
            )
            arrow_booster = FancyArrowPatch(
                (T_pre_a_ecg * temporal_resolution, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                (ax1.get_xlim()[1], ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                arrowstyle="<->",
                linestyle="--",
                color="black",
                alpha=0.7,
                mutation_scale=15,
            )
            ax1.add_patch(arrow_booster)
            ax1.text(
                (T_pre_a_ecg * temporal_resolution + ax1.get_xlim()[1]) / 2,
                ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
                "Booster",
                fontsize=6,
                color="black",
                horizontalalignment="center",
                verticalalignment="center",
            )

            # Annotate ECG timepoints
            ax2.annotate(
                "R",
                xy=(time[R_index], signal[R_index]),
                xytext=(time[R_index], signal[R_index] - 20),
                arrowprops=dict(facecolor="black", arrowstyle="->"),
                fontsize=8,
                ha="center",
                color="black",
            )
            ax2.annotate(
                "Q",
                xy=(time[Q_index], signal[Q_index]),
                xytext=(time[Q_index], signal[Q_index] + 20),
                arrowprops=dict(facecolor="black", arrowstyle="->"),
                fontsize=8,
                ha="center",
                color="black",
            )
            ax2.annotate(
                "S",
                xy=(time[S_index], signal[S_index]),
                xytext=(time[S_index], signal[S_index] - 20),
                arrowprops=dict(facecolor="black", arrowstyle="->"),
                fontsize=8,
                ha="center",
                color="black",
            )
            ax2.annotate(
                "P",
                xy=(time[P_index], signal[P_index]),
                xytext=(time[P_index], signal[P_index] + 20),
                arrowprops=dict(facecolor="black", arrowstyle="->"),
                fontsize=8,
                ha="center",
                color="black",
            )
            ax2.annotate(
                "T",
                xy=(time[T_index], signal[T_index]),
                xytext=(time[T_index], signal[T_index] + 20),
                arrowprops=dict(facecolor="black", arrowstyle="->"),
                fontsize=8,
                ha="center",
                color="black",
            )
            plt.suptitle(f"Subject {subject}: Atrial Volume Time Series and ECG Signal")
            plt.tight_layout()
            fig.savefig(f"{sub_dir}/timeseries/atrium_volume_ecg.png")
            plt.close(fig)

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    # Save the features to a csv file
    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "atrium")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), index=False)
