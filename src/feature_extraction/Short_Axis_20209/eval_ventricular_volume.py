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
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import nibabel as nib
import vtk
import math
from tqdm import tqdm
from scipy.signal import find_peaks
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
import warnings

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ECG_6025_20205.ecg_processor import ECG_Processor

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config

from utils.log_utils import setup_logging
from utils.quality_control_utils import sa_pass_quality_control
from utils.cardiac_utils import evaluate_ventricular_length_sax, evaluate_ventricular_length_lax
from utils.analyze_utils import plot_time_series_dual_axes, plot_time_series_dual_axes_double_y, analyze_time_series_derivative
from utils.biobank_utils import query_BSA

warnings.filterwarnings("ignore", category=UserWarning)
logger = setup_logging("eval_ventricular_volume")

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
            logger.error(f"{subject}: seg_sa does not pass quality control, skipped.")
            continue

        nim_sa = nib.load(sa_name)
        nim_la_4ch = nib.load(la_4ch_name)
        T = nim_sa.header["dim"][4]  # number of time frames
        temporal_resolution = nim_sa.header["pixdim"][4] * 1000  # unit: ms
        pixdim = nim_sa.header["pixdim"][1:4]
        volume_per_pix = pixdim[0] * pixdim[1] * pixdim[2] * 1e-3
        density = 1.05  # myocardium density is approximately 1.05 g/mL

        # Heart rate
        duration_per_cycle = nim_sa.header["dim"][4] * nim_sa.header["pixdim"][4]  # e.g. 50 time frame, 15.5 ms per cycle
        heart_rate = 60.0 / duration_per_cycle

        # Segmentation
        sa = nib.load(sa_name).get_fdata()
        la_4ch = nib.load(la_4ch_name).get_fdata()
        seg_sa = nib.load(seg_sa_name).get_fdata()
        seg_sa_nan = np.where(seg_sa == 0, np.nan, seg_sa)
        seg4_la = nib.load(seg4_la_name).get_fdata()
        L_sax = {}  # define transverse diameter
        L_4ch_long = {}  # define longitudinal diameter
        L_4ch_trans = {}
        lm_sax = {}
        lm_4ch_long = {}
        lm_4ch_trans = {}
        V = {}
        V["LV"] = np.sum(seg_sa == 1, axis=(0, 1, 2)) * volume_per_pix
        V["RV"] = np.sum(seg_sa == 3, axis=(0, 1, 2)) * volume_per_pix

        T_ED = 0  # bSSFP uses retrospective gating
        T_ES = np.argmin(V["LV"])
        logger.info(f"{subject}: ED frame: {T_ED}, ES frame: {T_ES}")

        # Save time series of volume and display the time series of ventricular volume
        os.makedirs(f"{sub_dir}/timeseries", exist_ok=True)
        logger.info(f"{subject}: Saving ventricular volume and time data")
        data_time = {
            "LV: Volume [mL]": V["LV"],
            "RV: Volume [mL]": V["RV"],
            "LV: T_ED [frame]": T_ED,
            "LV: T_ES [frame]": T_ES,
            "LV: T_ED [ms]": T_ED * temporal_resolution,
            "LV: T_ES [ms]": T_ES * temporal_resolution,
        }
        np.savez(f"{sub_dir}/timeseries/ventricle.npz", **data_time)

        logger.info(f"{subject}: Visualizing ventricular segmentation on short-axis images")
        os.makedirs(f"{sub_dir}/visualization/ventricle", exist_ok=True)
        N_slice = sa.shape[2]

        fig_ED, ax_ED = plt.subplots(2, N_slice // 2, figsize=(20, 10))
        for s, ax in enumerate(ax_ED.flat):
            ax.imshow(sa[:, :, s, T_ED], cmap="gray")
            ax.imshow(seg_sa_nan[:, :, s, T_ED], cmap="jet", alpha=0.5)
            ax.set_title(f"ED: Slice {s + 1}")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"{sub_dir}/visualization/ventricle/sa_ED.png")
        plt.close(fig_ED)

        fig_ES, ax_ES = plt.subplots(2, N_slice // 2, figsize=(20, 10))
        for s, ax in enumerate(ax_ES.flat):
            ax.imshow(sa[:, :, s, T_ES], cmap="gray")
            ax.imshow(seg_sa_nan[:, :, s, T_ES], cmap="jet", alpha=0.5)
            ax.set_title(f"ES: Slice {s + 1}")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"{sub_dir}/visualization/ventricle/sa_ES.png")
        plt.close(fig_ES)

        logger.info(f"{subject}: Record basic features")
        feature_dict = {
            "eid": subject,
        }

        LVEDV = np.sum(seg_sa[:, :, :, T_ED] == 1) * volume_per_pix
        RVEDV = np.sum(seg_sa[:, :, :, T_ED] == 3) * volume_per_pix
        LVESV = np.sum(seg_sa[:, :, :, T_ES] == 1) * volume_per_pix
        RVESV = np.sum(seg_sa[:, :, :, T_ES] == 3) * volume_per_pix
        Myo_EDV = np.sum(seg_sa[:, :, :, T_ED] == 2) * volume_per_pix
        Myo_ESV = np.sum(seg_sa[:, :, :, T_ES] == 2) * volume_per_pix
        feature_dict.update(
            {
                "LV: V_ED [mL]": LVEDV,
                "RV: V_ED [mL]": RVEDV,
                "LV: V_ES [mL]": LVESV,
                "RV: V_ES [mL]": RVESV,
                # Since mass should be insensitive to temporal order, we take the average here.
                "Myo: Mass [g]": (Myo_EDV + Myo_ESV) / 2 * density,
            }
        )

        feature_dict.update(
            {
                "LV: SV [mL]": feature_dict["LV: V_ED [mL]"] - feature_dict["LV: V_ES [mL]"],
                "RV: SV [mL]": feature_dict["RV: V_ED [mL]"] - feature_dict["RV: V_ES [mL]"],
            }
        )

        feature_dict.update(
            {
                "LV: CO [L/min]": feature_dict["LV: SV [mL]"] * heart_rate * 1e-3,
                "LV: EF [%]": feature_dict["LV: SV [mL]"] / feature_dict["LV: V_ED [mL]"] * 100,
                "RV: CO [L/min]": feature_dict["RV: SV [mL]"] * heart_rate * 1e-3,
                "RV: EF [%]": feature_dict["RV: SV [mL]"] / feature_dict["RV: V_ED [mL]"] * 100,
            }
        )

        time_grid_point = np.arange(T)
        time_grid_real = time_grid_point * temporal_resolution  # unit: ms

        fig, ax1, ax2 = plot_time_series_dual_axes(
            time_grid_point,
            V["LV"],
            "Time [frame]",
            "Time [ms]",
            "Volume [mL]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Ventricular Volume Time Series",
        )
        fig.savefig(f"{sub_dir}/timeseries/ventricle_volume_raw.png")
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
                "LV: V_ED/BSA [mL/m^2]": feature_dict["LV: V_ED [mL]"] / BSA_subject,
                "Myo: Mass/BSA [g/m^2]": feature_dict["Myo: Mass [g]"] / BSA_subject,
                "RV: V_ED/BSA [mL/m^2]": feature_dict["RV: V_ED [mL]"] / BSA_subject,
                "LV: V_ES/BSA [mL/m^2]": feature_dict["LV: V_ES [mL]"] / BSA_subject,
                "RV: V_ES/BSA [mL/m^2]": feature_dict["RV: V_ES [mL]"] / BSA_subject,
                # define Cardiac Index
                # Ref Cardiovascular magnetic resonance reference ranges for the heart and aorta in Chinese at 3T https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4830061/pdf/12968_2016_Article_236.pdf
                "LV: CI [L/min/m^2]": feature_dict["LV: CO [L/min]"] / BSA_subject,
                "RV: CI [L/min/m^2]": feature_dict["RV: CO [L/min]"] / BSA_subject,
            }
        )

        # * Feature2: LV Diameter and Spherical Index
        logger.info(f"{subject}: Implement LV diameter and spherical index")
        long_axis = nim_sa.affine[:3, 2] / np.linalg.norm(nim_sa.affine[:3, 2])
        short_axis = nim_la_4ch.affine[:3, 2] / np.linalg.norm(nim_la_4ch.affine[:3, 2])

        try:
            L_sax["ED"], lm_sax["ED"], slice_ED = evaluate_ventricular_length_sax(
                seg_sa[:, :, :, T_ED], nim_sa, long_axis, short_axis
            )
            points = vtk.vtkPoints()
            for p in lm_sax["ED"]:
                points.InsertNextPoint(p[0], p[1], 0)
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(poly)
            os.makedirs(f"{sub_dir}/landmark", exist_ok=True)
            writer.SetFileName(f"{sub_dir}/landmark/ventricle_transverse_diameter_sa_ED.vtk")
            writer.Write()

            L_4ch_long["ED"], lm_4ch_long["ED"], L_4ch_trans["ED"], lm_4ch_trans["ED"] = evaluate_ventricular_length_lax(
                seg4_la[:, :, 0, T_ED], nim_la_4ch, long_axis, short_axis
            )
            points = vtk.vtkPoints()
            for p in lm_4ch_long["ED"]:
                points.InsertNextPoint(p[0], p[1], 0)
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(poly)
            os.makedirs(f"{sub_dir}/landmark", exist_ok=True)
            writer.SetFileName(f"{sub_dir}/landmark/ventricle_longitudinal_diameter_4ch_ED.vtk")
            writer.Write()

            points = vtk.vtkPoints()
            for p in lm_4ch_trans["ED"]:
                points.InsertNextPoint(p[0], p[1], 0)
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(poly)
            os.makedirs(f"{sub_dir}/landmark", exist_ok=True)
            writer.SetFileName(f"{sub_dir}/landmark/ventricle_transverse_diameter_4ch_ED.vtk")
            writer.Write()

            # * We use the diameter measured in long-axis to calculate spherical index, see https://pubmed.ncbi.nlm.nih.gov/27571232/
            V_sphere_ED = 4 / 3 * math.pi * (L_4ch_long["ED"] / 2) ** 3

            feature_dict.update(
                {
                    "LV: D_transverse_ED (sax) [cm]": L_sax["ED"],
                    # define: The distance between the LV apex and interventricular septum
                    # Therefore, it should be slightly smaller than that the distance between LV apex to mitral valve
                    "LV: D_longitudinal_ED (4ch) [cm]": L_4ch_long["ED"],
                    # define The distance between basal septum and lateral wall
                    "LV: D_transverse_ED (4ch) [cm]": L_4ch_trans["ED"],
                }
            )
            if feature_dict["LV: V_ED [mL]"] / V_sphere_ED > 10:
                logger.warning(f"{subject}: Extremely high sphericity index at ED detected, skipped.")
            else:
                feature_dict.update({"LV: Sphericity_Index_ED": feature_dict["LV: V_ED [mL]"] / V_sphere_ED})

            plt.imshow(sa[:, :, slice_ED, 0], cmap="gray")
            x_coords = [p[1] for p in lm_sax["ED"]]
            y_coords = [p[0] for p in lm_sax["ED"]]
            plt.scatter(x_coords, y_coords, c="r", label="Transverse", s=8)
            plt.title(f"ED: Transverse Diameter (sax-slice{slice_ED + 1})")
            plt.legend(loc="lower right")
            plt.savefig(f"{sub_dir}/visualization/ventricle/sa_ED_diameter.png")
            plt.close()

            plt.imshow(la_4ch[:, :, 0, 0], cmap="gray")
            x_coords = [p[1] for p in lm_4ch_long["ED"]]
            y_coords = [p[0] for p in lm_4ch_long["ED"]]
            plt.scatter(x_coords, y_coords, c="r", label="Longitudinal", s=8)
            x_coords = [p[1] for p in lm_4ch_trans["ED"]]
            y_coords = [p[0] for p in lm_4ch_trans["ED"]]
            plt.scatter(x_coords, y_coords, c="b", label="Transverse", s=8)
            plt.title("ED: Longitudinal and Transverse Diameter (4ch)")
            plt.legend(loc="lower right")
            plt.savefig(f"{sub_dir}/visualization/ventricle/la_ED_diameter.png")
            plt.close()

        except ValueError:
            logger.warning(f"{subject}: Failed to determine the diameter of left ventricle at ED")

        try:
            L_sax["ES"], lm_sax["ES"], slice_ES = evaluate_ventricular_length_sax(
                seg_sa[:, :, :, T_ES], nim_sa, long_axis, short_axis
            )
            points = vtk.vtkPoints()
            for p in lm_sax["ES"]:
                points.InsertNextPoint(p[0], p[1], 0)
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(poly)
            os.makedirs(f"{sub_dir}/landmark", exist_ok=True)
            writer.SetFileName(f"{sub_dir}/landmark/ventricle_longitudinal_diameter_sa_ES.vtk")
            writer.Write()

            L_4ch_long["ES"], lm_4ch_long["ES"], L_4ch_trans["ES"], lm_4ch_trans["ES"] = evaluate_ventricular_length_lax(
                seg4_la[:, :, 0, T_ES], nim_la_4ch, long_axis, short_axis
            )
            points = vtk.vtkPoints()
            for p in lm_4ch_long["ES"]:
                points.InsertNextPoint(p[0], p[1], 0)
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(poly)
            os.makedirs(f"{sub_dir}/landmark", exist_ok=True)
            writer.SetFileName(f"{sub_dir}/landmark/ventricle_longitudinal_diameter_4ch_ES.vtk")
            writer.Write()

            points = vtk.vtkPoints()
            for p in lm_4ch_trans["ES"]:
                points.InsertNextPoint(p[0], p[1], 0)
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(poly)
            os.makedirs(f"{sub_dir}/landmark", exist_ok=True)
            writer.SetFileName(f"{sub_dir}/landmark/ventricle_transverse_diameter_4ch_ES.vtk")
            writer.Write()

            V_sphere_ES = 4 / 3 * math.pi * (L_4ch_long["ES"] / 2) ** 3

            feature_dict.update(
                {
                    "LV: D_transverse_ES (sax) [cm]": L_sax["ES"],
                    "LV: D_longitudinal_ES (4ch) [cm]": L_4ch_long["ES"],
                    "LV: D_transverse_ES (4ch) [cm]": L_4ch_trans["ES"],
                }
            )
            if feature_dict["LV: V_ES [mL]"] / V_sphere_ES > 10:
                logger.warning(f"{subject}: Extremely high sphericity index at ES detected, skipped.")
            else:
                feature_dict.update({"LV: Sphericity_Index_ES": feature_dict["LV: V_ES [mL]"] / V_sphere_ES})

            plt.imshow(sa[:, :, slice_ES, T_ES], cmap="gray")
            x_coords = [p[1] for p in lm_sax["ES"]]
            y_coords = [p[0] for p in lm_sax["ES"]]
            plt.scatter(x_coords, y_coords, c="r", label="Transverse", s=8)
            plt.title(f"ES: Transverse Diameter (sax-slice{slice_ES + 1})")
            plt.legend(loc="lower right")
            plt.savefig(f"{sub_dir}/visualization/ventricle/sa_ES_diameter.png")
            plt.close()

            plt.imshow(la_4ch[:, :, 0, T_ES], cmap="gray")
            x_coords = [p[1] for p in lm_4ch_long["ES"]]
            y_coords = [p[0] for p in lm_4ch_long["ES"]]
            plt.scatter(x_coords, y_coords, c="r", label="Longitudinal", s=8)
            x_coords = [p[1] for p in lm_4ch_trans["ES"]]
            y_coords = [p[0] for p in lm_4ch_trans["ES"]]
            plt.scatter(x_coords, y_coords, c="b", label="Transverse", s=8)
            plt.title("ES: Longitudinal and Transverse Diameter (4ch)")
            plt.legend(loc="lower right")
            plt.savefig(f"{sub_dir}/visualization/ventricle/la_ES_diameter.png")
            plt.close()
        except ValueError:
            logger.warning(f"{subject}: Failed to determine the diameter of left ventricle at ES")

        # * Feature3: Early/Late Peak Filling Rate
        # Ref Diastolic dysfunction evaluated by cardiac magnetic resonance https://pubmed.ncbi.nlm.nih.gov/30128617/
        V_LV = V["LV"]

        fANCOVA = importr("fANCOVA")
        x_r = FloatVector(time_grid_real)
        y_r = FloatVector(V_LV)
        # We use the loess provided by R that has Generalized Cross Validation as its criterion
        loess_fit = fANCOVA.loess_as(x_r, y_r, degree=2, criterion="gcv")
        V_LV_loess_x = np.array(loess_fit.rx2("x")).reshape(
            T,
        )
        V_LV_loess_y = np.array(loess_fit.rx2("fitted"))

        colors = ["blue"] * len(V["LV"])
        colors[T_ES] = "purple"
        fig, ax1, ax2 = plot_time_series_dual_axes_double_y(
            time_grid_point,
            V_LV_loess_y,
            V["LV"],
            "Time [frame]",
            "Time [ms]",
            "Volume [mL]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Ventricular Volume Time Series (Smoothed)",
            colors=colors,
        )
        ax1.axvline(x=T_ES, color="purple", linestyle="--", alpha=0.7)
        ax1.text(
            T_ES,
            ax1.get_ylim()[1],
            "End Systole",
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=6,
            color="black",
        )
        # Explicitly annotate two phases
        arrow_systole = FancyArrowPatch(
            (0, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
            (T_ES, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
            arrowstyle="<->",
            linestyle="--",
            color="black",
            alpha=0.7,
            mutation_scale=15,
        )
        ax1.add_patch(arrow_systole)
        ax1.text(
            T_ES / 2,
            ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
            "Systole",
            fontsize=6,
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )
        arrow_diastole = FancyArrowPatch(
            (T_ES, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
            (ax1.get_xlim()[1], ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
            arrowstyle="<->",
            linestyle="--",
            color="black",
            alpha=0.7,
            mutation_scale=15,
        )
        ax1.add_patch(arrow_diastole)
        ax1.text(
            (T_ES + ax1.get_xlim()[1]) / 2,
            ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
            "Diastole",
            fontsize=6,
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )
        box_text = "LV Volume\n" f"End-Diastolic Volume: {LVEDV:.2f} mL\n" f"End-Systolic Volume: {LVESV:.2f} mL"
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
        fig.savefig(f"{sub_dir}/timeseries/ventricle_volume.png")
        plt.close(fig)

        # * Instead of using np.diff, we use np.gradient to ensure the same length
        V_LV_diff_y = np.gradient(V_LV_loess_y, V_LV_loess_x) * 1000  # unit: mL/s

        try:
            # These time points are purely used for visualization, not for calculation of PFR
            # Generally, it should be negative peak -> positive peak -> negative peak / 0 -> positive peak

            L1 = T_ES
            T_peak_neg_1 = find_peaks(-V_LV_diff_y[:T_ES], distance=math.ceil(L1 / 3))[0]
            T_peak_neg_1 = T_peak_neg_1[np.argmax(-V_LV_diff_y[T_peak_neg_1])]

            L2 = len(V_LV_diff_y) - T_ES
            T_peak_neg_2 = find_peaks(-V_LV_diff_y[T_ES:], distance=math.ceil(L2 / 3))[0]  # this may not lead to negative value
            T_peak_neg_2 = [peak + T_ES for peak in T_peak_neg_2]
            T_peak_neg_2 = T_peak_neg_2[np.argmax(-V_LV_diff_y[T_peak_neg_2])]

            L3 = T_peak_neg_2 - T_ES
            T_peak_pos_1 = find_peaks(V_LV_diff_y[T_ES:T_peak_neg_2], distance=math.ceil(L3 / 3))[0]
            T_peak_pos_1 = [peak + T_ES for peak in T_peak_pos_1]
            T_peak_pos_1 = T_peak_pos_1[np.argmax(V_LV_diff_y[T_peak_pos_1])]

            # No need to find peaks for the last segment of cardiac cycle
            L4 = len(V_LV_diff_y) - T_peak_neg_2
            T_peak_pos_2 = np.argmax(V_LV_diff_y[T_peak_neg_2:])
            T_peak_pos_2 = T_peak_pos_2 + T_peak_neg_2
        except Exception as e:
            logger.error(f"{subject}: Error {e} in determining PFR, skipped.")
            continue

        try:
            T_PFR, _, PFR, _ = analyze_time_series_derivative(time_grid_real / 1000, V["LV"], n_pos=2, n_neg=0)  # unit: ml/s
            logger.info(f"{subject}: Implementing peak filling rate (PFR) features")
            # define PER_E: Early atrial peak emptying rate, PER_A: Late atrial peak emptying rate
            PFR_E = PFR[np.argmin(T_PFR)]
            PFR_A = PFR[np.argmax(T_PFR)]
            if PFR_E <= 0:
                raise ValueError("PFR_E should be positive")
            if PFR_A <= 0:
                raise ValueError("PFR_A should be positive")
            if PFR_E < 10 or PFR_A < 10:
                raise ValueError("Extremely small PFR values detected, skipped.")
            feature_dict.update(
                {
                    "LV: PFR-E [mL/s]": PFR_E,
                    "LV: PFR-A [mL/s]": PFR_A,
                    "LV: PFR-E/BSA [mL/s/m^2]": PFR_E / BSA_subject,
                    "LV: PFR-A/BSA [mL/s/m^2]": PFR_A / BSA_subject,
                }
            )
            if PFR_E / PFR_A > 10:
                logger.warning(f"{subject}: Extremely high PFR-E/PFR-A detected, skipped.")
            else:
                feature_dict.update({"LV: PFR-E/PFR-A": PFR_E / PFR_A})

                colors = ["blue"] * len(V_LV_diff_y)
                colors[T_ES] = "purple"
                colors[T_peak_pos_1] = "aqua"
                colors[T_peak_pos_2] = "lime"
                fig, ax1, ax2 = plot_time_series_dual_axes(
                    time_grid_point,
                    V_LV_diff_y,
                    "Time [frame]",
                    "Time [ms]",
                    "dV/dt [mL/s]",
                    lambda x: x * temporal_resolution,
                    lambda x: x / temporal_resolution,
                    title=f"Subject {subject}: Derivative of Ventricular Volume Time Series",
                    colors=colors,
                )
                ax1.axvline(x=T_ES, color="purple", linestyle="--", alpha=0.7)
                ax1.text(
                    T_ES,
                    ax1.get_ylim()[1],
                    "End Systole",
                    verticalalignment="bottom",
                    horizontalalignment="center",
                    fontsize=6,
                    color="black",
                )
                ax1.axvline(x=T_peak_pos_1, color="aqua", linestyle="--", alpha=0.7)
                ax1.text(
                    T_peak_pos_1,
                    ax1.get_ylim()[0],
                    "PFR-E",
                    verticalalignment="bottom",
                    horizontalalignment="center",
                    fontsize=6,
                    color="black",
                )
                ax1.axvline(x=T_peak_pos_2, color="lime", linestyle="--", alpha=0.7)
                (
                    ax1.text(
                        T_peak_neg_2,
                        ax1.get_ylim()[0],
                        "PFR-A",
                        verticalalignment="bottom",
                        horizontalalignment="center",
                        fontsize=6,
                        color="black",
                    ),
                )
                box_text = (
                    "Peak Filling Rate\n"
                    "(Obtained through multiple frames)\n"
                    f"PFR-E: {abs(PFR_E):.2f} mL/s\n"
                    f"PFR-A: {abs(PFR_A):.2f} mL/s"
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
                arrow_systole = FancyArrowPatch(
                    (0, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                    (T_ES, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                    arrowstyle="<->",
                    linestyle="--",
                    color="black",
                    alpha=0.7,
                    mutation_scale=15,
                )
                ax1.add_patch(arrow_systole)
                ax1.text(
                    T_ES / 2,
                    ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
                    "Systole",
                    fontsize=6,
                    color="black",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                arrow_diastole = FancyArrowPatch(
                    (T_ES, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                    (ax1.get_xlim()[1], ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                    arrowstyle="<->",
                    linestyle="--",
                    color="black",
                    alpha=0.7,
                    mutation_scale=15,
                )
                ax1.add_patch(arrow_diastole)
                ax1.text(
                    (T_ES + ax1.get_xlim()[1]) / 2,
                    ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
                    "Diastole",
                    fontsize=6,
                    color="black",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                fig.savefig(f"{sub_dir}/timeseries/ventricle_volume_rate.png")
                plt.close(fig)
        except (IndexError, ValueError) as e:
            logger.warning(f"{subject}: PFR calculation failed: {e}")

        # * Plot volume curve together with ECG

        ecg_processor = ECG_Processor(subject, args.retest)
        if config.useECG and ecg_processor.check_data_rest():
            # Ref Expanding application of the Wiggers diagram to teach cardiovascular physiology. https://doi.org/10.1152/advan.00123.2013
            # Note we don't have any pressure features
            logger.info(f"{subject}: Create Wiggers diagram that combines ECG and ventricular volume")

            ecg_info = ecg_processor.visualize_ecg_info()
            time = ecg_info["time"]
            signal = ecg_info["signal"]
            P_index = ecg_info["P_index"]
            Q_index = ecg_info["Q_index"]
            R_index = ecg_info["R_index"]
            S_index = ecg_info["S_index"]
            T_index = ecg_info["T_index"]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax1.plot(time_grid_real, V["LV"], color="r")
            ax1.set_xlabel("Time [ms]")
            ax1.set_ylabel("Volume [mL]")
            ax2.plot(time, signal, color="b")
            ax2.set_ylabel("ECG Signal")
            ax2.set_yticks([])

            # Similar annotations as before
            ax1.axvline(x=T_ES * temporal_resolution, color="purple", linestyle="--", alpha=0.7)
            ax1.text(
                T_ES * temporal_resolution,
                ax1.get_ylim()[1],
                "End Systole",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=6,
                color="black",
            )
            arrow_systole = FancyArrowPatch(
                (0, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                (T_ES * temporal_resolution, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                arrowstyle="<->",
                linestyle="--",
                color="black",
                alpha=0.7,
                mutation_scale=15,
            )
            ax1.add_patch(arrow_systole)
            ax1.text(
                T_ES * temporal_resolution / 2,
                ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
                "Systole",
                fontsize=6,
                color="black",
                horizontalalignment="center",
                verticalalignment="center",
            )
            arrow_diastole = FancyArrowPatch(
                (T_ES * temporal_resolution, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                (ax1.get_xlim()[1], ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
                arrowstyle="<->",
                linestyle="--",
                color="black",
                alpha=0.7,
                mutation_scale=15,
            )
            ax1.add_patch(arrow_diastole)
            ax1.text(
                (T_ES * temporal_resolution + ax1.get_xlim()[1]) / 2,
                ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
                "Diastole",
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
            plt.suptitle(f"Subject {subject}: Ventricular Volume Time Series and ECG Signal")
            plt.tight_layout()
            fig.savefig(f"{sub_dir}/timeseries/ventricle_ecg.png")

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "ventricle")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), index=False)
