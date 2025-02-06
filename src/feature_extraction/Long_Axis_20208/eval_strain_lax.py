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
from utils.quality_control_utils import la_pass_quality_control
from utils.analyze_utils import plot_time_series_dual_axes, plot_time_series_dual_axes_double_y, analyze_time_series_derivative
from utils.cardiac_utils import cine_2d_la_motion_and_strain_analysis, evaluate_strain_by_length_la

logger = setup_logging("eval_strain_lax")

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
        logger.info(f"Calculating longitudinal strain for subject {subject}")
        sub_dir = os.path.join(data_dir, subject)
        seg_la_name = os.path.join(sub_dir, "seg4_la_4ch.nii.gz")

        nim_la = nib.load(seg_la_name)
        T = nim_la.header["dim"][4]
        temporal_resolution = nim_la.header["pixdim"][4] * 1000
        time_grid_point = np.arange(T)
        time_grid_real = time_grid_point * temporal_resolution  # unit: ms

        if not os.path.exists(seg_la_name):
            logger.error(f"Segmentation of long axis file for {subject} does not exist")
            continue

        if not la_pass_quality_control(seg_la_name):
            logger.error(f"{subject}: seg_la does not pass quality control, skipped.")
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

        logger.info(f"{subject}: Perform motion tracking on long-axis images and calculate the strain.")
        # Perform motion tracking on long-axis images and calculate the strain
        cine_2d_la_motion_and_strain_analysis(sub_dir, par_config_name, temp_dir, ft_dir)
        longit_strain = evaluate_strain_by_length_la(f"{ft_dir}/myo_contour_la/la_4ch_myo_contour_fr", T, ft_dir)
        logger.info(f"{subject}: Longitudinal strain calculated, remove intermediate files.")

        # Remove intermediate files
        shutil.rmtree(temp_dir)

        # * We use the maximum absolute value: minimum for longitudinal (negative)

        for i in range(6):
            feature_dict.update(
                {
                    f"Strain-LAX: Longitudinal Strain (Segment_{i + 1}) [%]": longit_strain[i, :].min(),  # no need to plus one
                }
            )

        feature_dict.update(
            {
                "Strain-LAX: Longitudinal Strain (Global) [%]": longit_strain[6, :].min(),
            }
        )

        os.makedirs(f"{sub_dir}/timeseries", exist_ok=True)

        # Store the time series of global strains
        logger.info(f"{subject}: Store time series of global strains.")
        data_strain = {
            "Strain-LAX: Longitudinal Strain [%]": longit_strain,
            "Strain-LAX: Global Longitudinal Strain [%]": longit_strain[6, :],
        }
        np.savez(f"{sub_dir}/timeseries/strain_lax.npz", **data_strain)

        # * Make a time series plot and store the time series of global strain
        logger.info(f"{subject}: Plot time series of global strain.")
        fig, ax1, ax2 = plot_time_series_dual_axes(
            time_grid_point,
            longit_strain[6, :],
            "Time [frame]",
            "Time [ms]",
            "Strain [%]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Global Longitudinal Strain (GLS) Time Series",
        )
        fig.savefig(f"{sub_dir}/timeseries/gls_raw.png")
        plt.close(fig)

        # Read in important time points
        try:
            ventricle = np.load(f"{sub_dir}/timeseries/ventricle.npz")
        except FileNotFoundError:
            logger.error(f"{subject}: No ventricle time series information, skipped.")
            continue

        T_ES = ventricle["LV: T_ES [frame]"]
        T_1_3_DD = T_ES + math.ceil((50 - T_ES) / 3)  # used to calculate SI-DI

        # * Feature 1: Peak-systolic strain, end-systolic strain, post-systolic strain
        # Ref Post-systolic shortening index by echocardiography evaluation of dyssynchrony https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9409501/pdf/pone.0273419.pdf

        # We will use smoothed strain for all advanced features
        fANCOVA = importr("fANCOVA")
        x_r = FloatVector(time_grid_real)
        y_r_gls = FloatVector(longit_strain[6, :])
        loess_fit_gls = fANCOVA.loess_as(x_r, y_r_gls, degree=2, criterion="gcv")
        GLS_loess_x = np.array(loess_fit_gls.rx2("x")).reshape(
            T,
        )
        GLS_loess_y = np.array(loess_fit_gls.rx2("fitted"))

        # In addition to global longitudinal strain, we will need time to peak strain for each segment to calculate MDI
        T_longit_strain_peak_list = []
        for i in range(6):
            y_r_ls_i = FloatVector(longit_strain[i, :])
            loess_fit_ls_i = fANCOVA.loess_as(x_r, y_r_ls_i, degree=2, criterion="gcv")
            LS_loess_y_i = np.array(loess_fit_ls_i.rx2("fitted"))
            T_longit_strain_peak_list.append(np.argmax(abs(LS_loess_y_i)))

        # All following values are in absolute values!
        longit_strain_ES = abs(GLS_loess_y[T_ES])

        T_longit_strain_peak = np.argmax(abs(GLS_loess_y))
        longit_strain_peak = np.max(abs(GLS_loess_y))

        feature_dict.update(
            {
                "Strain-LAX: End Systolic Longitudinal Strain (Absolute Value) [%]": longit_strain_ES,
                "Strain-LAX: Peak Systolic Longitudinal Strain (Absolute Value) [%]": longit_strain_peak,
            }
        )
        logger.info(f"{subject}: End-systolic, peak-systolic longitudinal global strain calculated.")

        if longit_strain_ES < 0.1 or longit_strain_peak < 0.1:
            logger.error(f"{subject}: Extremely small longitudinal strain values detected skipped.")
            continue

        T_longit_strain_post = None
        longit_strain_post = None

        if T_longit_strain_peak > T_ES:
            # define Post systolic strain is calculated as the difference between the maximum strain and systolic strain
            # Ref Postsystolic Shortening by Speckle Tracking Echocardiography Is an Independent Predictor. https://doi.org/10.1161/JAHA.117.008367
            T_longit_strain_post = T_longit_strain_peak
            longit_strain_post = longit_strain_peak - longit_strain_ES

            feature_dict.update(
                {
                    "Strain-LAX: Post Systolic Longitudinal Strain (Absolute Value) [%]": longit_strain_post,
                }
            )
            logger.info(f"{subject}: Post-systolic longitudinal global strain calculated.")

        # Add more information to the time series of strain
        colors = ["blue"] * len(GLS_loess_x)
        colors[T_ES] = "purple"
        colors[T_1_3_DD] = "green"
        if T_longit_strain_post is None:
            colors[T_longit_strain_peak] = "red"
        else:
            colors[T_longit_strain_post] = "orange"
        fig, ax1, ax2 = plot_time_series_dual_axes_double_y(
            time_grid_point,
            longit_strain[6, :],
            GLS_loess_y,
            "Time [frame]",
            "Time [ms]",
            "Strain [%]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Global Longitudinal Strain (GLS) Time Series (Smoothed)",
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
        if T_longit_strain_post is None:
            ax1.axvline(x=T_longit_strain_peak, color="red", linestyle="--", alpha=0.7)
            ax1.text(
                T_longit_strain_peak,
                ax1.get_ylim()[0],
                "Peak Systolic Strain",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=6,
                color="black",
            )
        else:
            ax1.axvline(x=T_longit_strain_post, color="orange", linestyle="--", alpha=0.7)
            ax1.text(
                T_longit_strain_post,
                ax1.get_ylim()[0],
                "Post Systolic Strain",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=6,
                color="black",
            )
        if T_longit_strain_post is None:
            box_text = (
                "Global Longitudinal Strain\n"
                f"End Systolic: Strain: {longit_strain_ES:.2f} %\n"
                f"Peak Systolic Strain: {longit_strain_peak:.2f} %"
            )
        else:
            box_text = (
                "Global Longitudinal Strain\n"
                f"End Systolic: Strain: {longit_strain_ES:.2f} %\n"
                f"Peak Systolic Strain: {longit_strain_peak:.2f} %\n"
                f"Post Systolic Strain: {longit_strain_post:.2f} %"
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
        fig.savefig(f"{sub_dir}/timeseries/gls.png")
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
            ax1.plot(time_grid_real, longit_strain[6, :], color="r")
            ax1.set_xlabel("Time [ms]")
            ax1.set_ylabel("Longitudinal Strain [%]")
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
            plt.suptitle(f"Subject {subject}: Global Longitudinal Strain (GLS) Time Series and ECG Signal")
            plt.tight_layout()
            fig.savefig(f"{sub_dir}/timeseries/gls_ecg.png")
            plt.close(fig)

        # * Feature 2: Strain Rate
        # Instead of using np.diff, we use np.gradient to ensure the same length
        GLS_diff_y = np.gradient(GLS_loess_x, GLS_loess_y) * 1000    # unit: %/s

        try:
            T_GLSR_pos, T_GLSR_neg, GLSR_pos, GLSR_neg = analyze_time_series_derivative(
                time_grid_real / 1000,
                longit_strain[6, :] / 100,  # Since strain is in %
                n_pos=2,
                n_neg=1,
            )
            GLSR_S = GLSR_neg[0]
            GLSR_E = GLSR_pos[np.argmin(T_GLSR_pos)]
            GLSR_A = GLSR_pos[np.argmax(T_GLSR_pos)]
            if np.min(T_GLSR_pos) < T_ES:
                raise ValueError("Time for peak early diastolic longitudinal strain rate should be after ES.")
            feature_dict.update(
                {
                    "Strain-LAX: Peak Systolic Longitudinal Strain Rate [1/s]": GLSR_S,
                    "Strain-LAX: Early Diastolic Longitudinal Strain Rate [1/s]": GLSR_E,
                    "Strain-LAX: Late Diastolic Longitudinal Strain Rate [1/s]": GLSR_A,
                }
            )

            logger.info(f"{subject}: Global longitudinal strain rate calculated.")

            # Plot strain rate
            T_GLSR_S = T_GLSR_neg[0]
            T_GLSR_E = np.min(T_GLSR_pos)
            T_GLSR_A = np.max(T_GLSR_pos)
            colors = ["blue"] * len(GLS_loess_x)
            colors[T_ES] = "purple"
            colors[T_GLSR_S] = "deepskyblue"
            colors[T_GLSR_E] = "aqua"
            colors[T_GLSR_A] = "lime"
            fig, ax1, ax2 = plot_time_series_dual_axes(
                time_grid_point,
                GLS_diff_y / 100,
                "Time [frame]",
                "Time [ms]",
                "Longitudinal Strain Rate [1/s]",
                lambda x: x * temporal_resolution,
                lambda x: x / temporal_resolution,
                title=f"Subject {subject}: Global Longitudinal Strain Rate Time Series",
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
            ax1.axvline(x=T_GLSR_S, color="deepskyblue", linestyle="--", alpha=0.7)
            ax1.text(
                T_GLSR_S,
                ax1.get_ylim()[1],
                "Peak Systolic SR",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=6,
                color="black",
            )
            ax1.axvline(x=T_GLSR_E, color="aqua", linestyle="--", alpha=0.7)
            ax1.text(
                T_GLSR_E,
                ax1.get_ylim()[1],
                "Early Diastolic SR",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=6,
                color="black",
            )
            ax1.axvline(x=T_GLSR_A, color="lime", linestyle="--", alpha=0.7)
            ax1.text(
                T_GLSR_A,
                ax1.get_ylim()[1],
                "Late Diastolic SR",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=6,
                color="black",
            )
            box_text = (
                "Global Circumferential Strain Rate\n"
                f"Peak Systolic Strain Rate: {GLSR_S:.2f} 1/s\n"
                f"Early Diastolic Strain Rate: {GLSR_E:.2f} 1/s\n"
                f"Late Diastolic Strain Rate: {GLSR_A:.2f} 1/s"
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
            fig.savefig(f"{sub_dir}/timeseries/glsr.png")
            plt.close(fig)
        except ValueError as e:
            logger.warning(f"{subject}: {e}  No global longitudinal strain rate calculated.")

        # * Feature 3: Time to peak strain, Post systolic strain index (PSI), Mechanical dispersion/dys-synchrony index (MDI)

        t_longit_strain_peak = T_longit_strain_peak * temporal_resolution  # unit: ms
        feature_dict.update(
            {
                "Strain-LAX: Time to Peak Longitudinal Strain [ms]": t_longit_strain_peak,
            }
        )

        if RR_interval:
            t_longit_strain_peak_index = t_longit_strain_peak / RR_interval
            logger.info(f"{subject}: ECG rest data exists, calculate global time to peak strain index.")
            feature_dict.update(
                {
                    "Strain-LAX: Time to Peak Longitudinal Strain Index": t_longit_strain_peak_index,
                }
            )

        # Ref Postsystolic Shortening Is an Independent Predictor of Cardiovascular Events and Mortality. https://doi.org/10.1161/JAHA.117.008367
        # Ref Post-systolic shortening index by echocardiography evaluation of dyssynchrony https://doi.org/10.1371/journal.pone.0273419
        # Ref Post-systolic shortening predicts heart failure following acute coronary syndrome https://doi.org/10.1016/j.ijcard.2018.11.106
        # If no regional post systolic strain present, it PSI is 0.
        # If a single wall segment exhibits PSI>=20%, the wall is categorized as having Post systolic shortening (PSS)
        if T_longit_strain_post is not None:
            longit_PSI = longit_strain_post / (longit_strain_peak) * 100
            feature_dict.update(
                {
                    "Strain-LAX: Longitudinal Strain Post Systolic Index [%]": longit_PSI,
                }
            )
            logger.info(f"{subject}: Global Post-systolic index calculated.")
        else:
            longit_PSI = 0

        # Ref Post-systolic shortening index by echocardiography evaluation of dyssynchrony https://doi.org/10.1371/journal.pone.0273419
        # Ref Mechanical dispersion and global longitudinal strain by speckle tracking echocardiography https://doi.org/10.1111/echo.13547
        # Ref Prognostic importance of left ventricular mechanical dyssynchrony in heart failure with preserved ejection fraction https://doi.org/10.1002/ejhf.789
        t_longit_strain_peak_list = []  # define used to calculate Mechanical dispersion index
        for i in range(6):
            t_longit_strain_peak_i = T_longit_strain_peak_list[i] * temporal_resolution
            t_longit_strain_peak_list.append(t_longit_strain_peak_i)

        # define Mechanical Dispersion: Standard deviation of all LV segmental time-to-peak intervals
        # Since analysis is based on frames, MSI should only be reported if segments have distinct time to peak strain
        if np.unique(t_longit_strain_peak_list).shape[0] > 1:
            MS = np.std(t_longit_strain_peak_list)  # Therefore it measures the dys-synchrony
            feature_dict.update(
                {
                    "Strain-LAX: Longitudinal Strain Mechanical Dispersion [ms]": MS,
                }
            )
            logger.info(f"{subject}: Mechanical dispersion calculated.")

        # * Feature 4 Strain imaging diastolic index (SI-DI)
        # Ref Prediction of coronary artery stenosis using strain imaging diastolic index at rest https://doi.org/10.1016/j.jjcc.2011.01.008

        # define SI-DI can be calculated through (Strain at ES - Strain at 1/3 DD) / Strain at ES
        longit_strain_1_3_DD = abs(GLS_loess_y[T_1_3_DD])
        longit_SI_DI = (longit_strain_ES - longit_strain_1_3_DD) / longit_strain_ES

        if longit_SI_DI > 2:
            logger.warning(f"{subject}: Extremely high longitudinal strain imaging diastolic index (SI-DI) detected, skipped.")
        else:
            logger.info(f"{subject}: Longitudinal strain imaging diastolic index (SI-DI) calculated.")
            feature_dict.update(
                {
                    "Strain-LAX: Longitudinal Strain Imaging Diastolic Index [%]": longit_SI_DI * 100,
                }
            )

        # * Feature 5 Pre-Systolic stretch (PSS, SPS)
        # Ref Multi-Parametric Speckle Tracking Analyses to Characterize Cardiac Amyloidosis https://doi.org/10.1007/s00380-022-02047-6.
        GLS_loess_y_systole = GLS_loess_y[:T_ES]
        if np.max(GLS_loess_y_systole) > 0:
            longit_strain_systole_positive = np.max(GLS_loess_y_systole)
            # define Systolic stretch: Peak extension strain during systole / maximum strain change during systole
            # note that longit_strain_ES is absolute value, so we add instead of subtract
            pre_systolic_stretch = longit_strain_systole_positive / (longit_strain_systole_positive + longit_strain_ES) * 100
            feature_dict.update(
                {
                    "Strain-LAX: Longitudinal Strain Pre Systolic Stretch [%]": pre_systolic_stretch,
                }
            )
            logger.info(f"{subject}: Systolic stretch calculated.")

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "strain_lax")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), index=False)
