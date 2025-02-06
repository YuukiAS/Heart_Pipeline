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
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import nibabel as nib
import argparse
import shutil
from tqdm import tqdm
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ECG_6025_20205.ecg_processor import ECG_Processor

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config

from utils.log_utils import setup_logging
from utils.quality_control_utils import sa_pass_quality_control
from utils.analyze_utils import (
    plot_time_series_dual_axes,
    plot_time_series_dual_axes_double_y,
    analyze_time_series_derivative,
    plot_bulls_eye,
)
from utils.cardiac_utils import cine_2d_sa_motion_and_strain_analysis, evaluate_strain_by_length_sa, evaluate_torsion

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

        nim_sa = nib.load(seg_sa_name)
        seg_sa = nim_sa.get_fdata()
        T = nim_sa.header["dim"][4]
        temporal_resolution = nim_sa.header["pixdim"][4] * 1000
        time_grid_point = np.arange(T)
        time_grid_real = time_grid_point * temporal_resolution  # unit: ms

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

        logger.info(f"{subject}: Perform motion tracking on short-axis images and calculate the strain.")
        # Perform motion tracking on short-axis images and calculate the strain
        cine_2d_sa_motion_and_strain_analysis(sub_dir, par_config_name, temp_dir, ft_dir)
        radial_strain, circum_strain = evaluate_strain_by_length_sa(f"{ft_dir}/myo_contour_sa/myo_contour_fr", T, ft_dir)
        logger.info(f"{subject}: Radial and circumferential strain calculated, remove intermediate files.")

        # Remove intermediate files
        shutil.rmtree(temp_dir)

        # * We use the maximum absolute value: minimum for circumferential (negative) and maximum for radial (positive)
        for i in range(16):
            feature_dict.update(
                {
                    f"Strain-SAX: Circumferential strain (AHA_{i + 1}) [%]": circum_strain[i, :].min(),
                    f"Strain-SAX: Radial strain (AHA_{i + 1}) [%]": radial_strain[i, :].max(),
                }
            )

        feature_dict.update(
            {
                "Strain-SAX: Circumferential strain (Global) [%]": circum_strain[16, :].min(),
                "Strain-SAX: Radial strain (Global) [%]": radial_strain[16, :].max(),
            }
        )

        # Store the time series of global strains
        logger.info(f"{subject}: Store time series of global strains.")
        os.makedirs(f"{sub_dir}/timeseries", exist_ok=True)
        data_strain = {
            "Strain-SAX: Circumferential Strain [%]": circum_strain,
            "Strain-SAX: Radial Strain [%]": radial_strain,
            "Strain-SAX: Global Circumferential Strain [%]": circum_strain[16, :],
            "Strain-SAX: Global Radial Strain [%]": radial_strain[16, :],
        }
        np.savez(f"{sub_dir}/timeseries/strain_sax.npz", **data_strain)

        # Add Bull's eye plot
        fig, ax = plot_bulls_eye(np.min(circum_strain, axis=1)[:16], title="Circumferential Strain", label="Strain [%]")
        fig.savefig(f"{sub_dir}/visualization/myocardium/strain_circum.png")
        plt.close(fig)
        fig, ax = plot_bulls_eye(np.max(radial_strain, axis=1)[:16], title="Radial Strain", label="Strain [%]")
        fig.savefig(f"{sub_dir}/visualization/myocardium/strain_radial.png")
        plt.close(fig)

        # Make a time series plot and store the time series of global strains
        logger.info(f"{subject}: Plot time series of global strains.")
        fig, ax1, ax2 = plot_time_series_dual_axes(
            time_grid_point,
            circum_strain[16, :],
            "Time [frame]",
            "Time [ms]",
            "Strain [%]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Global Circumferential Strain (GCS) Time Series",
        )
        fig.savefig(f"{sub_dir}/timeseries/gcs_raw.png")
        plt.close(fig)

        fig, ax1, ax2 = plot_time_series_dual_axes(
            time_grid_point,
            radial_strain[16, :],
            "Time [frame]",
            "Time [ms]",
            "Strain [%]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Global Radial Strain (GRS) Time Series",
        )
        fig.savefig(f"{sub_dir}/timeseries/grs_raw.png")
        plt.close(fig)

        # Read in important time points
        try:
            ventricle = np.load(f"{sub_dir}/timeseries/ventricle.npz")
        except FileNotFoundError:
            logger.error(f"{subject}: No ventricle time series information, skipped.")
            continue

        T_ES = ventricle["LV: T_ES [frame]"]
        T_1_3_DD = T_ES + math.ceil((50 - T_ES) / 3)

        # * Feature 1: Peak-systolic strain, end-systolic strain, post-systolic strain
        # We will use smoothed strain for all advanced features
        fANCOVA = importr("fANCOVA")
        x_r = FloatVector(time_grid_real)
        y_r_gcs = FloatVector(circum_strain[16, :])
        loess_fit_gcs = fANCOVA.loess_as(x_r, y_r_gcs, degree=2, criterion="gcv")
        GCS_loess_x = np.array(loess_fit_gcs.rx2("x")).reshape(
            T,
        )
        GCS_loess_y = np.array(loess_fit_gcs.rx2("fitted"))

        y_r_grs = FloatVector(radial_strain[16, :])
        loess_fit_grs = fANCOVA.loess_as(x_r, y_r_grs, degree=2, criterion="gcv")
        GRS_loess_x = np.array(loess_fit_grs.rx2("x")).reshape(
            T,
        )
        GRS_loess_y = np.array(loess_fit_grs.rx2("fitted"))

        # All following values are in absolute values!
        circum_strain_ES = abs(GCS_loess_y[T_ES])

        T_circum_strain_peak = np.argmax(abs(GCS_loess_y))
        circum_strain_peak = np.max(abs(GCS_loess_y))

        if circum_strain_ES < 0.1 or circum_strain_peak < 0.1:
            logger.error(f"{subject}: Extremely small circumferential strain values detected skipped.")
            continue

        feature_dict.update(
            {
                "Strain-SAX: End Systolic Circumferential Strain (Absolute Value) [%]": circum_strain_ES,
                "Strain-SAX: Peak Systolic Circumferential Strain (Absolute Value) [%]": circum_strain_peak,
            }
        )
        logger.info(f"{subject}: End-systolic, peak-systolic circumferential strain calculated.")

        T_circum_strain_post = None
        circum_strain_post = None
        if T_circum_strain_peak > T_ES:
            # define Post systolic strain is calculated as the difference between the maximum strain and systolic strain
            # Ref Postsystolic Shortening by Speckle Tracking Echocardiography Is an Independent Predictor. https://doi.org/10.1161/JAHA.117.008367
            T_circum_strain_post = T_circum_strain_peak
            circum_strain_post = circum_strain_peak - circum_strain_ES

            feature_dict.update(
                {
                    "Strain-SAX: Post Systolic Circumferential Strain (Absolute Value) [%]": circum_strain_post,
                }
            )
            logger.info(f"{subject}: Post-systolic circumferential strain calculated.")

        radial_strain_ES = abs(GRS_loess_y[T_ES])

        T_radial_strain_peak = np.argmax(abs(GRS_loess_y))
        radial_strain_peak = np.max(abs(GRS_loess_y))

        if radial_strain_ES < 0.1 or radial_strain_peak < 0.1:
            logger.error(f"{subject}: Extremely small radial strain values detected skipped.")
            continue

        feature_dict.update(
            {
                "Strain-SAX: End Systolic Radial Strain (Absolute Value) [%]": radial_strain_ES,
                "Strain-SAX: Peak Systolic Radial Strain (Absolute Value) [%]": radial_strain_peak,
            }
        )
        logger.info(f"{subject}: End-systolic, peak-systolic radial strain calculated.")

        T_radial_strain_post = None
        radial_strain_post = None
        if T_radial_strain_peak > T_ES:
            # Post systolic shortening exists
            T_radial_strain_post = T_radial_strain_peak
            radial_strain_post = radial_strain_peak - radial_strain_ES

            feature_dict.update(
                {
                    "Strain-SAX: Post Systolic Radial Strain (Absolute Value) [%]": radial_strain_post,
                }
            )
            logger.info(f"{subject}: End-systolic, post-systolic radial strain calculated.")

        # Add more information to the time series of strain
        colors = ["blue"] * len(GCS_loess_x)
        colors[T_ES] = "purple"
        colors[T_1_3_DD] = "green"
        if T_circum_strain_post is None:
            colors[T_circum_strain_peak] = "red"
        else:
            colors[T_circum_strain_post] = "orange"
        fig, ax1, ax2 = plot_time_series_dual_axes_double_y(
            time_grid_point,
            circum_strain[16, :],
            GCS_loess_y,
            "Time [frame]",
            "Time [ms]",
            "Strain [%]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Global Circumferential Strain (GCS) Time Series (Smoothed)",
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
        ax1.axvline(x=T_1_3_DD, color="green", linestyle="--", alpha=0.7)
        ax1.text(
            T_1_3_DD,
            ax1.get_ylim()[1],
            "First 1/3 Diastolic Duration",
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=6,
            color="black",
        )
        if T_circum_strain_post is None:
            ax1.axvline(x=T_circum_strain_peak, color="red", linestyle="--", alpha=0.7)
            ax1.text(
                T_circum_strain_peak,
                ax1.get_ylim()[0],
                "Peak Systolic Strain",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=6,
                color="black",
            )
        else:
            ax1.axvline(x=T_circum_strain_post, color="orange", linestyle="--", alpha=0.7)
            ax1.text(
                T_circum_strain_post,
                ax1.get_ylim()[0],
                "Post Systolic Strain",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=6,
                color="black",
            )
        if T_circum_strain_post is None:
            box_text = (
                "Global Circumferential Strain\n"
                f"End Systolic: Strain: {circum_strain_ES:.2f} %\n"
                f"Peak Systolic Strain: {circum_strain_peak:.2f} %"
            )
        else:
            box_text = (
                "Global Circumferential Strain\n"
                f"End Systolic: Strain: {circum_strain_ES:.2f} %\n"
                f"Peak Systolic Strain: {circum_strain_peak:.2f} %\n"
                f"Post Systolic Strain: {circum_strain_post:.2f} %"
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
        fig.savefig(f"{sub_dir}/timeseries/gcs.png")
        plt.close(fig)

        # Visualize in the same way for global radial strain
        colors = ["blue"] * len(GRS_loess_x)
        colors[T_ES] = "purple"
        colors[T_1_3_DD] = "green"
        if T_radial_strain_post is None:
            colors[T_radial_strain_peak] = "red"
        else:
            colors[T_radial_strain_post] = "orange"
        fig, ax1, ax2 = plot_time_series_dual_axes_double_y(
            time_grid_point,
            radial_strain[16, :],
            GRS_loess_y,
            "Time [frame]",
            "Time [ms]",
            "Strain [%]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Global Radial Strain (GRS) Time Series (Smoothed)",
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
        ax1.axvline(x=T_1_3_DD, color="green", linestyle="--", alpha=0.7)
        ax1.text(
            T_1_3_DD,
            ax1.get_ylim()[1],
            "First 1/3 Diastolic Duration",
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=6,
            color="black",
        )
        if T_radial_strain_post is None:
            ax1.axvline(x=T_radial_strain_peak, color="red", linestyle="--", alpha=0.7)
            ax1.text(
                T_radial_strain_peak,
                ax1.get_ylim()[0],
                "Peak Systolic Strain",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=6,
                color="black",
            )
        else:
            ax1.axvline(x=T_radial_strain_post, color="orange", linestyle="--", alpha=0.7)
            ax1.text(
                T_radial_strain_post,
                ax1.get_ylim()[0],
                "Post Systolic Strain",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=6,
                color="black",
            )
        if T_radial_strain_post is None:
            box_text = (
                "Global Radial Strain\n"
                f"End Systolic: Strain: {radial_strain_ES:.2f} %\n"
                f"Peak Systolic Strain: {radial_strain_peak:.2f} %"
            )
        else:
            box_text = (
                "Global Radial Strain\n"
                f"End Systolic: Strain: {radial_strain_ES:.2f} %\n"
                f"Peak Systolic Strain: {radial_strain_peak:.2f} %\n"
                f"Post Systolic Strain: {radial_strain_post:.2f} %"
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
        fig.savefig(f"{sub_dir}/timeseries/grs.png")
        plt.close(fig)

        # * Visualize strains alongside ECG, also determining the RR interval if available
        ecg_processor = ECG_Processor(subject, args.retest)
        if not config.useECG or not ecg_processor.check_data_rest():
            RR_interval = None
            logger.warning(f"{subject}: No ECG rest data, time to peak strain index will not be calculated.")
        else:
            RR_interval = ecg_processor.determine_RR_interval()  # should be close to MeanNN in neurokit2
            logger.info(f"{subject}: RR interval is {RR_interval:.2f} ms.")

            logger.info(f"{subject}: Visualize strains alongside with ECG")

            ecg_info = ecg_processor.visualize_ecg_info()
            time = ecg_info["time"]
            signal = ecg_info["signal"]
            P_index = ecg_info["P_index"]
            Q_index = ecg_info["Q_index"]
            R_index = ecg_info["R_index"]
            S_index = ecg_info["S_index"]
            T_index = ecg_info["T_index"]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax1.plot(time_grid_real, circum_strain[16, :], color="r")
            ax1.set_xlabel("Time [ms]")
            ax1.set_ylabel("Circumferential Strain [%]")
            ax2.plot(time, signal, color="b")
            ax2.set_ylabel("ECG Signal")
            ax2.set_yticks([])
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
            plt.suptitle(f"Subject {subject}: Global Circumferential Strain (GCS) Time Series and ECG Signal")
            plt.tight_layout()
            fig.savefig(f"{sub_dir}/timeseries/gcs_ecg.png")
            plt.close(fig)

            # Visualize in the same way for global radial strain
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax1.plot(time_grid_real, radial_strain[16, :], color="r")
            ax1.set_xlabel("Time [ms]")
            ax1.set_ylabel("Radial Strain [%]")
            ax2.plot(time, signal, color="b")
            ax2.set_ylabel("ECG Signal")
            ax2.set_yticks([])
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
            plt.suptitle(f"Subject {subject}: Global Radial Strain (GRS) Time Series and ECG Signal")
            plt.tight_layout()
            fig.savefig(f"{sub_dir}/timeseries/grs_ecg.png")
            plt.close(fig)

        # * Feature 2: Strain rate

        # Instead of using np.diff, we use np.gradient to ensure the same length
        GCS_diff_y = np.gradient(GCS_loess_y, GCS_loess_x) * 1000   # unit: %/s
        GRS_diff_y = np.gradient(GRS_loess_y, GRS_loess_x) * 1000

        try:
            T_GCSR_pos, T_GCSR_neg, GCSR_pos, GCSR_neg = analyze_time_series_derivative(
                time_grid_real / 1000,
                circum_strain[16, :] / 100,  # Since strain is in %
                n_pos=2,
                n_neg=1,
            )
            GCSR_S = GCSR_neg[0]
            GCSR_E = GCSR_pos[np.argmin(T_GCSR_pos)]
            GCSR_A = GCSR_pos[np.argmax(T_GCSR_pos)]
            if np.min(T_GCSR_pos) < T_ES:
                raise ValueError("Time for peak early diastolic circumferential strain rate should be after ES.")
            feature_dict.update(
                {
                    "Strain-SAX: Peak Systolic Circumferential Strain Rate [1/s]": GCSR_S,
                    "Strain-SAX: Early Diastolic Circumferential Strain Rate [1/s]": GCSR_E,
                    "Strain-SAX: Late Diastolic Circumferential Strain Rate [1/s]": GCSR_A,
                }
            )
            logger.info(f"{subject}: Global circumferential strain rate calculated.")

            # Plot strain rate
            T_GCSR_S = T_GCSR_neg[0]
            T_GCSR_E = np.min(T_GCSR_pos)
            T_GCSR_A = np.max(T_GCSR_pos)
            colors = ["blue"] * len(GCS_loess_x)
            colors[T_ES] = "purple"
            colors[T_GCSR_S] = "deepskyblue"
            colors[T_GCSR_E] = "aqua"
            colors[T_GCSR_A] = "lime"
            fig, ax1, ax2 = plot_time_series_dual_axes(
                time_grid_point,
                GCS_diff_y / 100,
                "Time [frame]",
                "Time [ms]",
                "Circumferential Strain Rate [1/s]",
                lambda x: x * temporal_resolution,
                lambda x: x / temporal_resolution,
                title=f"Subject {subject}: Global Circumferential Strain Rate Time Series",
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
            ax1.axvline(x=T_GCSR_S, color="deepskyblue", linestyle="--", alpha=0.7)
            ax1.text(
                T_GCSR_S,
                ax1.get_ylim()[1],
                "Peak Systolic SR",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=6,
                color="black",
            )
            ax1.axvline(x=T_GCSR_E, color="aqua", linestyle="--", alpha=0.7)
            ax1.text(
                T_GCSR_E,
                ax1.get_ylim()[1],
                "Early Diastolic SR",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=6,
                color="black",
            )
            ax1.axvline(x=T_GCSR_A, color="lime", linestyle="--", alpha=0.7)
            ax1.text(
                T_GCSR_A,
                ax1.get_ylim()[1],
                "Late Diastolic SR",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=6,
                color="black",
            )
            box_text = (
                "Global Circumferential Strain Rate\n"
                f"Peak Systolic Strain Rate: {GCSR_S:.2f} 1/s\n"
                f"Early Diastolic Strain Rate: {GCSR_E:.2f} 1/s\n"
                f"Late Diastolic Strain Rate: {GCSR_A:.2f} 1/s"
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
            fig.savefig(f"{sub_dir}/timeseries/gcsr.png")
            plt.close(fig)
        except ValueError as e:
            logger.warning(f"{subject}: {e}  No global circumferential strain rate calculated.")

        try:
            T_GRSR_pos, T_GRSR_neg, GRSR_pos, GRSR_neg = analyze_time_series_derivative(
                time_grid_real / 1000,
                radial_strain[16, :] / 100,  # Since strain is in %
                n_pos=1,
                n_neg=2,
            )
            GRSR_S = GRSR_pos[0]
            GRSR_E = GRSR_neg[np.argmin(T_GRSR_neg)]
            GRSR_A = GRSR_neg[np.argmax(T_GRSR_neg)]
            if np.min(T_GRSR_neg) < T_ES:
                raise ValueError("Time for peak early diastolic radial strain rate should be after ES.")
            feature_dict.update(
                {
                    "Strain-SAX: Peak Systolic Radial Strain Rate [1/s]": GRSR_S,
                    "Strain-SAX: Early Diastolic Radial Strain Rate [1/s]": GRSR_E,
                    "Strain-SAX: Late Diastolic Radial Strain Rate [1/s]": GRSR_A,
                }
            )
            logger.info(f"{subject}: Global radial strain rate calculated.")

            T_GRSR_S = T_GRSR_pos[0]
            T_GRSR_E = np.min(T_GRSR_neg)
            T_GRSR_A = np.max(T_GRSR_neg)
            colors = ["blue"] * len(GRS_loess_x)
            colors[T_ES] = "purple"
            colors[T_GRSR_S] = "deepskyblue"
            colors[T_GRSR_E] = "aqua"
            colors[T_GRSR_A] = "lime"
            fig, ax1, ax2 = plot_time_series_dual_axes(
                time_grid_point,
                GRS_diff_y / 100,
                "Time [frame]",
                "Time [ms]",
                "Radial Strain Rate [1/s]",
                lambda x: x * temporal_resolution,
                lambda x: x / temporal_resolution,
                title=f"Subject {subject}: Global Radial Strain Rate Time Series",
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
            ax1.axvline(x=T_GRSR_S, color="deepskyblue", linestyle="--", alpha=0.7)
            ax1.text(
                T_GRSR_S,
                ax1.get_ylim()[1],
                "Peak Systolic SR",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=6,
                color="black",
            )
            ax1.axvline(x=T_GRSR_E, color="aqua", linestyle="--", alpha=0.7)
            ax1.text(
                T_GRSR_E,
                ax1.get_ylim()[1],
                "Early Diastolic SR",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=6,
                color="black",
            )
            ax1.axvline(x=T_GRSR_A, color="lime", linestyle="--", alpha=0.7)
            ax1.text(
                T_GRSR_A,
                ax1.get_ylim()[1],
                "Late Diastolic SR",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=6,
                color="black",
            )
            box_text = (
                "Global Radial Strain Rate\n"
                f"Peak Systolic Strain Rate: {GRSR_S:.2f} 1/s\n"
                f"Early Diastolic Strain Rate: {GRSR_E:.2f} 1/s\n"
                f"Late Diastolic Strain Rate: {GRSR_A:.2f} 1/s"
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
            fig.savefig(f"{sub_dir}/timeseries/grsr.png")
            plt.close(fig)
        except ValueError as e:
            logger.warning(f"{subject}: {e} No global radial strain rate calculated.")

        # * Feature 3 Time to peak strain

        # We use global value here. But For we will consider each segment for longitudinal strain.
        t_circum_strain_peak = T_circum_strain_peak * temporal_resolution  # unit: ms
        t_radial_strain_peak = T_radial_strain_peak * temporal_resolution

        feature_dict.update(
            {
                "Strain-SAX: Time to Peak Circumferential Strain [ms]": t_circum_strain_peak,
                "Strain-SAX: Time to Peak Radial Strain [ms]": t_radial_strain_peak,
            }
        )

        if RR_interval:
            t_circum_strain_peak_index = t_circum_strain_peak / RR_interval
            t_radial_strain_peak_index = t_radial_strain_peak / RR_interval

            feature_dict.update(
                {
                    "Strain-SAX: Time to Peak Circumferential Strain Index": t_circum_strain_peak_index,
                    "Strain-SAX: Time to Peak Radial Strain Index": t_radial_strain_peak_index,
                }
            )

        # * Feature 4 Strain imaging diastolic index (SI-DI)
        # Ref Prediction of coronary artery stenosis using strain imaging diastolic index at rest in patients https://www.sciencedirect.com/science/article/pii/S0914508711000116

        # define SI-DI can be calculated through (Strain at ES - Strain at 1/3 DD) / Strain at ES
        circum_strain_1_3_DD = abs(GCS_loess_y[T_1_3_DD])
        circum_SI_DI = (circum_strain_ES - circum_strain_1_3_DD) / circum_strain_ES

        radial_strain_1_3_DD = abs(GRS_loess_y[T_1_3_DD])
        radial_SI_DI = (radial_strain_ES - radial_strain_1_3_DD) / radial_strain_ES

        if circum_SI_DI > 2:
            logger.warning(f"{subject}: Extremely high circumferential strain imaging diastolic index (SI-DI) detected, skipped.")
        else:
            logger.info(f"{subject}: Circumferential Strain imaging diastolic index (SI-DI) calculated.")
            feature_dict.update(
                {
                    "Strain-SAX: Circumferential Strain Imaging Diastolic Index [%]": circum_SI_DI * 100,
                }
            )
        if radial_SI_DI > 2:
            logger.warning(f"{subject}: Extremely high radial strain imaging diastolic index (SI-DI) detected, skipped.")
        else:
            logger.info(f"{subject}: Radial Strain imaging diastolic index (SI-DI) calculated.")
            feature_dict.update(
                {
                    "Strain-SAX: Radial Strain Imaging Diastolic Index [%]": radial_SI_DI * 100,
                }
            )

        # * Feature 5 Torsion and recoil rate

        try:
            # Ref Quantification of Left Ventricular Torsion and Diastolic Recoil Using Myocardial Feature Tracking. https://doi.org/10.1371/journal.pone.0109164
            # * Be aware that we do not use the top basal and apical slices.
            # * Therefore, the obtained rotations will be smaller than ones using top slices.
            # * However, this should not affect the calculation of torsion as torsion is normalized by slice distance
            torsion_endo, torsion_epi, torsion_global, basal_slice, apical_slice = evaluate_torsion(
                seg_sa, nim_sa, contour_name_stem=f"{ft_dir}/myo_contour_sa/myo_contour_fr"
            )
            logger.info(f"{subject}: Torsion calculated using {basal_slice} to {apical_slice}, save time series plots.")

            feature_dict.update(
                {
                    "Strain-SAX: Endocardial Torsion [°/cm]": np.max(torsion_endo["torsion"]),
                    "Strain-SAX: Epicardial Torsion [°/cm]": np.max(torsion_epi["torsion"]),
                    "Strain-SAX: Global Torsion [°/cm]": np.max(torsion_global["torsion"]),
                }
            )

            T_endo_torsion_peak = np.argmax(torsion_endo["torsion"])
            T_epi_torsion_peak = np.argmax(torsion_epi["torsion"])
            T_global_torsion_peak = np.argmax(torsion_global["torsion"])

            t_endo_torsion_peak = T_endo_torsion_peak * temporal_resolution  # unit: ms
            t_epi_torsion_peak = T_epi_torsion_peak * temporal_resolution
            t_global_torsion_peak = T_global_torsion_peak * temporal_resolution

            feature_dict.update(
                {
                    "Strain-SAX: Time to Peak Endocardial Torsion [ms]": t_endo_torsion_peak,
                    "Strain-SAX: Time to Peak Epicardial Torsion [ms]": t_epi_torsion_peak,
                    "Strain-SAX: Time to Peak Global Torsion [ms]": t_global_torsion_peak,
                }
            )

            # Define Peak recoil rate is the maximum negative slope of torsion-time curve
            try:
                _, T_recoil_rate_endo, _, recoil_rate_endo = analyze_time_series_derivative(
                    time_grid_real, torsion_endo["torsion"], n_pos=0, n_neg=1, method="difference"
                )

                _, T_recoil_rate_epi, _, recoil_rate_epi = analyze_time_series_derivative(
                    time_grid_real, torsion_epi["torsion"], n_pos=0, n_neg=1, method="difference"
                )

                _, T_recoil_rate_global, _, recoil_rate_global = analyze_time_series_derivative(
                    time_grid_real, torsion_global["torsion"], n_pos=0, n_neg=1, method="difference"
                )

                logger.info(f"{subject}: Recoil rate calculated.")
                feature_dict.update(
                    {
                        "Strain-SAX: Endocardial Recoil Rate [°/cm/s]": recoil_rate_endo[0],
                        "Strain-SAX: Epicardial Recoil Rate [°/cm/s]": recoil_rate_epi[0],
                        "Strain-SAX: Global Recoil Rate [°/cm/s]": recoil_rate_global[0],
                    }
                )

                fig, ax1 = plt.subplots(figsize=(12, 8), sharex=True)
                ax1.plot(np.arange(T), torsion_endo["base"], label="basal rotation", color="black")
                ax1.plot(np.arange(T), torsion_endo["apex"], label="apical rotation", color="black", linestyle="--")
                ax1.set_xlabel("Time [frame]")
                ax1.set_ylabel("Rotation [°]")
                ax1_ylim = ax1.get_ylim()
                ax1_ylim_max = max(abs(ax1_ylim[0]), abs(ax1_ylim[1]))
                ax1_new_ylim = (-ax1_ylim_max, ax1_ylim_max)
                ax1.set_ylim(ax1_new_ylim)
                ax1.axhline(y=0, color="gray", linestyle="--")
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
                box_text = (
                    "Endocardial\n"
                    f"Torsion: {np.max(torsion_endo['torsion']):.2f} °/cm\n"
                    f"Peak Recoil Rate: {recoil_rate_endo[0]:.2f} °/cm/s"
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
                ax2 = ax1.twinx()
                ax2.plot(np.arange(T), torsion_endo["torsion"], label="Torsion", color="blue")
                ax2.scatter(T_recoil_rate_endo, torsion_endo["torsion"][T_recoil_rate_endo], color="red", marker="x")
                ax2.set_ylabel("Torsion [°/cm]")
                # Align the zero line of the second y-axis
                ax2_ylim = ax2.get_ylim()
                ax2_new_ylim = (-ax2_ylim[1], ax2_ylim[1])
                ax2.set_ylim(ax2_new_ylim)
                fig.suptitle(f"Subject {subject}: Endocardial Rotation and Torsion Time Series")
                fig.legend(loc="lower right")
                plt.savefig(f"{sub_dir}/timeseries/torsion_endo.png")
                plt.close(fig)

                fig, ax1 = plt.subplots(figsize=(12, 8), sharex=True)
                ax1.plot(np.arange(T), torsion_epi["base"], label="basal rotation", color="black")
                ax1.plot(np.arange(T), torsion_epi["apex"], label="apical rotation", color="black", linestyle="--")
                ax1.set_xlabel("Time [frame]")
                ax1.set_ylabel("Rotation [°]")
                ax1_ylim = ax1.get_ylim()
                ax1_ylim_max = max(abs(ax1_ylim[0]), abs(ax1_ylim[1]))
                ax1_new_ylim = (-ax1_ylim_max, ax1_ylim_max)
                ax1.set_ylim(ax1_new_ylim)
                ax1.axhline(y=0, color="gray", linestyle="--")
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
                box_text = (
                    "Epicardial\n"
                    f"Torsion: {np.max(torsion_epi['torsion']):.2f} °/cm\n"
                    f"Peak Recoil Rate: {recoil_rate_epi[0]:.2f} °/cm/s"
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
                ax2 = ax1.twinx()
                ax2.plot(np.arange(T), torsion_epi["torsion"], label="Torsion", color="blue")
                ax2.scatter(T_recoil_rate_epi, torsion_epi["torsion"][T_recoil_rate_epi], color="red", marker="x")
                ax2.set_ylabel("Torsion [°/cm]")
                ax2_ylim = ax2.get_ylim()
                ax2_new_ylim = (-ax2_ylim[1], ax2_ylim[1])
                ax2.set_ylim(ax2_new_ylim)
                fig.suptitle(f"Subject {subject}: Epicardial Rotation and Torsion Time Series")
                fig.legend(loc="lower right")
                plt.savefig(f"{sub_dir}/timeseries/torsion_epi.png")
                plt.close(fig)

                fig, ax1 = plt.subplots(figsize=(12, 8), sharex=True)
                ax1.plot(np.arange(T), torsion_global["base"], label="basal rotation", color="black")
                ax1.plot(np.arange(T), torsion_global["apex"], label="apical rotation", color="black", linestyle="--")
                ax1.set_xlabel("Time [frame]")
                ax1.set_ylabel("Rotation [°]")
                ax1_ylim = ax1.get_ylim()
                ax1_ylim_max = max(abs(ax1_ylim[0]), abs(ax1_ylim[1]))
                ax1_new_ylim = (-ax1_ylim_max, ax1_ylim_max)
                ax1.set_ylim(ax1_new_ylim)
                ax1.axhline(y=0, color="gray", linestyle="--")
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
                box_text = (
                    "Global\n"
                    f"Torsion: {np.max(torsion_global['torsion']):.2f} °/cm\n"
                    f"Peak Recoil Rate: {recoil_rate_global[0]:.2f} °/cm/s"
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
                ax2 = ax1.twinx()
                ax2.plot(np.arange(T), torsion_global["torsion"], label="Torsion", color="blue")
                ax2.scatter(T_recoil_rate_global, torsion_global["torsion"][T_recoil_rate_global], color="red", marker="x")
                ax2.set_ylabel("Torsion [°/cm]")
                ax2_ylim = ax2.get_ylim()
                ax2_ylim_max = max(abs(ax2_ylim[0]), abs(ax2_ylim[1]))
                ax2_new_ylim = (-ax2_ylim_max, ax2_ylim_max)
                ax2.set_ylim(ax2_new_ylim)
                fig.suptitle(f"Subject {subject}: Global Rotation and Torsion Time Series")
                fig.legend(loc="lower right")
                plt.savefig(f"{sub_dir}/timeseries/torsion_global.png")
                plt.close(fig)

            except ValueError as e:
                logger.warning(f"{subject}: {e}  No recoil rate calculated.")

        except ValueError as e:
            logger.warning(f"{subject}: {e} No torsion calculated.")
            continue

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "strain_sax")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), index=False)
