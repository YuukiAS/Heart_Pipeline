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
import pickle
from tqdm import tqdm
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ecg.ecg_processor import ECG_Processor

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config

from utils.log_utils import setup_logging
from utils.quality_control_utils import sa_pass_quality_control
from utils.analyze_utils import plot_time_series_double_x, plot_time_series_double_x_y, analyze_time_series_derivative
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
        temporal_resolution = nim_sa.header["pixdim"][4]
        time_grid_point = np.arange(T)
        time_grid_real = time_grid_point * temporal_resolution  # unit: s

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
        radial_strain, circum_strain = evaluate_strain_by_length_sa(
            f"{ft_dir}/myo_contour_sa/myo_contour_fr", T, ft_dir
        )
        logger.info(f"{subject}: Radial and circumferential strain calculated, remove intermediate files.")

        # Remove intermediate files
        shutil.rmtree(temp_dir)

        # * We use the maximum absolute value: minimum for circumferential (negative) and maximum for radial (positive)

        for i in range(16):
            feature_dict.update(
                {
                    f"LV: Circumferential strain (AHA_{i + 1}) [%]": circum_strain[i, :].min(),
                    f"LV: Radial strain (AHA_{i + 1}) [%]": radial_strain[i, :].max(),
                }
            )

        feature_dict.update(
            {
                "LV: Circumferential strain (Global) [%]": circum_strain[16, :].min(),
                "LV: Radial strain (Global) [%]": radial_strain[16, :].max(),
            }
        )

        # * Make a time series plot and store the time series of global strain
        logger.info(f"{subject}: Plot and store time series of global strain.")

        with open(f"{sub_dir}/timeseries/strain_sa.pkl", "wb") as time_series_file:
            pickle.dump(
                {
                    "LV: Global Circumferential Strain (GCS) [%]": circum_strain[16, :],
                    "LV: Global Radial Strain (GRS) [%]": radial_strain[16, :],
                },
                time_series_file,
            )

        fig, ax1, ax2 = plot_time_series_double_x(
            time_grid_point,
            time_grid_real * 1000,
            circum_strain[16, :],
            "Time [frame]",
            "Time [ms]",
            "Strain [%]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Global Circumferential Strain (GCS) Time Series (Raw)",
        )
        fig.savefig(f"{sub_dir}/timeseries/gcs_raw.png")
        plt.close(fig)

        fig, ax1, ax2 = plot_time_series_double_x(
            time_grid_point,
            time_grid_real * 1000,
            radial_strain[16, :],
            "Time [frame]",
            "Time [ms]",
            "Strain [%]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Global Radial Strain (GRS) Time Series (Raw)",
        )
        fig.savefig(f"{sub_dir}/timeseries/grs_raw.png")
        plt.close(fig)

        # Read in important time points
        with open(f"{sub_dir}/timeseries/ventricle.pkl", "rb") as time_series_file:
            ventricle = pickle.load(time_series_file)

        with open(f"{sub_dir}/timeseries/atrium.pkl", "rb") as time_series_file:
            atrium = pickle.load(time_series_file)

        T_ES = ventricle["LV: T_ES"]
        T_1_3_DD = T_ES + math.ceil((50 - T_ES) / 3)
        T_pre_a = None

        try:
            T_pre_a = atrium["LA: T_pre_a"]
        except KeyError:
            logger.warning(f"{subject}: No atrial contraction time information for strain calculation.")

        # * Feature 1: Strain rate

        try:
            T_GCSR_pos, T_GCSR_neg, GCSR_pos, GCSR_neg = analyze_time_series_derivative(
                time_grid_real,
                circum_strain[16, :] / 100,  # Since strain is in %
                n_pos=2,
                n_neg=1,
                method="loess",  # for strain rate, we don't use the moving average method
            )

            GCSR_S = GCSR_neg[0]
            GCSR_E = GCSR_pos[np.argmin(T_GCSR_pos)]
            GCSR_A = GCSR_pos[np.argmax(T_GCSR_pos)]
            if np.min(T_GCSR_pos) < T_ES:
                raise ValueError("Time for peak early diastolic circumferential strain rate should be after ES.")
            feature_dict.update(
                {
                    "LV: Circumferential Strain Rate (Peak-systolic) [1/s]": GCSR_S,
                    "LV: Circumferential Strain Rate (Early-diastole) [1/s]": GCSR_E,
                    "LV: Circumferential Strain Rate (Late-diastole) [1/s]": GCSR_A,
                }
            )
            logger.info(f"{subject}: Global circumferential strain rate calculated.")
        except ValueError as e:
            logger.warning(f"{subject}: {e}  No global circumferential strain rate calculated.")

        try:
            T_GRSR_pos, T_GRSR_neg, GRSR_pos, GRSR_neg = analyze_time_series_derivative(
                time_grid_real,
                radial_strain[16, :] / 100,  # Since strain is in %
                n_pos=1,
                n_neg=2,
                method="loess",
            )
            GRSR_S = GRSR_pos[0]
            GRSR_E = GRSR_neg[np.argmin(T_GRSR_neg)]
            GRSR_A = GRSR_neg[np.argmax(T_GRSR_neg)]
            if np.min(T_GRSR_neg) < T_ES:
                raise ValueError("Time for peak early diastolic radial strain rate should be after ES.")
            feature_dict.update(
                {
                    "LV: Radial Strain Rate (Peak-systolic) [1/s]": GRSR_S,
                    "LV: Radial Strain Rate (Early-diastole) [1/s]": GRSR_E,
                    "LV: Radial Strain Rate (Late-diastole) [1/s]": GRSR_A,
                }
            )
            logger.info(f"{subject}: Global radial strain rate calculated.")
        except ValueError as e:
            logger.warning(f"{subject}: {e} No global radial strain rate calculated.")

        # * Feature 2  Peak-systolic strain, end-systolic strain, post-systolic strain

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

        colors = ["blue"] * T
        colors[T_ES] = "red"
        colors[T_1_3_DD] = "green"
        if T_pre_a is not None:
            colors[T_pre_a] = "orange"

        fig, ax1, ax2 = plot_time_series_double_x_y(
            time_grid_point,
            time_grid_real * 1000,
            circum_strain[16, :],
            GCS_loess_y,
            "Time [frame]",
            "Time [ms]",
            "Strain [%]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Global Circumferential Strain (GCS) Time Series",
            colors=colors,
        )
        fig.savefig(f"{sub_dir}/timeseries/gcs.png")
        plt.close(fig)

        fig, ax1, ax2 = plot_time_series_double_x_y(
            time_grid_point,
            time_grid_real * 1000,
            radial_strain[16, :],
            GRS_loess_y,
            "Time [frame]",
            "Time [ms]",
            "Strain [%]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Global Radial Strain (GRS) Time Series",
            colors=colors,
        )
        fig.savefig(f"{sub_dir}/timeseries/grs.png")
        plt.close(fig)

        # All following values are in absolute values!

        circum_strain_ES = abs(GCS_loess_y[T_ES])

        T_circum_strain_peak = np.argmax(abs(GCS_loess_y))
        cicum_strain_peak = np.max(abs(GCS_loess_y))

        T_circum_strain_post = None
        circum_strain_post = None
        if T_circum_strain_peak > T_ES:
            T_circum_strain_post = T_circum_strain_peak
            circum_strain_post = cicum_strain_peak

            feature_dict.update(
                {
                    "LV: Circumferential Strain (End-systolic, absolute value) [%]": circum_strain_ES,
                    "LV: Circumferential Strain (Post-systolic, absolute value) [%]": circum_strain_post,
                }
            )
            logger.info(f"{subject}: End-systolic, post-systolic circumferential strain calculated.")
        else:
            feature_dict.update(
                {
                    "LV: Circumferential Strain (End-systolic, absolute value) [%]": circum_strain_ES,
                    "LV: Circumferential Strain (Peak-systolic, absolute value) [%]": cicum_strain_peak,
                }
            )
            logger.info(f"{subject}: End-systolic, peak-systolic circumferential strain calculated.")

        radial_strain_ES = abs(GRS_loess_y[T_ES])

        T_radial_strain_peak = np.argmax(abs(GRS_loess_y))
        radial_strain_peak = np.max(abs(GRS_loess_y))

        T_radial_strain_post = None
        radial_strain_post = None
        if T_radial_strain_peak > T_ES:
            # Post systolic shortening exists
            T_radial_strain_post = T_radial_strain_peak
            radial_strain_post = radial_strain_peak

            feature_dict.update(
                {
                    "LV: Radial Strain (End-systolic, absolute value) [%]": radial_strain_ES,
                    "LV: Radial Strain (Post-systolic, absolute value) [%]": radial_strain_post,
                }
            )
            logger.info(f"{subject}: End-systolic, post-systolic radial strain calculated.")
        else:
            feature_dict.update(
                {
                    "LV: Radial Strain (End-systolic, absolute value) [%]": radial_strain_ES,
                    "LV: Radial Strain (Peak-systolic, absolute value) [%]": radial_strain_peak,
                }
            )
            logger.info(f"{subject}: End-systolic, peak-systolic radial strain calculated.")

        # * Feature 3 Time to peak interval

        # We use global value here, while for longitudinal strain, we consider each segment

        t_circum_strain_peak = T_circum_strain_peak * temporal_resolution * 1000  # unit: ms
        t_radial_strain_peak = T_radial_strain_peak * temporal_resolution * 1000

        feature_dict.update(
            {
                "LV: Circumferential Strain: Time to Peak [ms]": t_circum_strain_peak,
                "LV: Radial Strain: Time to Peak [ms]": t_radial_strain_peak,
            }
        )

        ecg_processor = ECG_Processor(subject, args.retest)

        if config.useECG and not ecg_processor.check_data_rest():
            logger.warning(f"{subject}: No ECG rest data, time to peak strain index will not be calculated.")
        else:
            logger.info(f"{subject}: ECG rest data exists, calculate time to peak strain index.")
            RR_interval = ecg_processor.determine_RR_interval()  # should be close to MeanNN in neurokit2

            t_circum_strain_peak_index = t_circum_strain_peak / RR_interval
            t_radial_strain_peak_index = t_radial_strain_peak / RR_interval

            feature_dict.update(
                {
                    "LV: Circumferential Strain: Time to Peak Index": t_circum_strain_peak_index,
                    "LV: Radial Strain: Time to Peak Index": t_radial_strain_peak_index,
                }
            )

        # * Feature 4 Strain imaging diastolic index (SI-DI)

        # Ref https://www.sciencedirect.com/science/article/pii/S0914508711000116

        circum_strain_1_3_DD = abs(GCS_loess_y[T_1_3_DD])

        circum_SI_DI = (circum_strain_ES - circum_strain_1_3_DD) / circum_strain_ES

        radial_strain_1_3_DD = abs(GRS_loess_y[T_1_3_DD])

        radial_SI_DI = (radial_strain_ES - radial_strain_1_3_DD) / radial_strain_ES

        feature_dict.update(
            {
                "LV: Circumferential Strain: Strain Imaging Diastolic Index [%]": circum_SI_DI * 100,
                "LV: Radial Strain: Strain Imaging Diastolic Index [%]": radial_SI_DI * 100,
            }
        )

        logger.info(f"{subject}: Strain imaging diastolic index (SI-DI) calculated.")

        # * Feature 5 Torsion and recoil rate, Recoil rate

        try:
            torsion_endo, torsion_epi, torsion_global, basal_slice, apical_slice = evaluate_torsion(
                seg_sa, nim_sa, contour_name_stem=f"{ft_dir}/myo_contour_sa/myo_contour_fr"
            )
            logger.info(f"{subject}: Torsion calculated, save time series plots.")

            plt.plot(np.arange(T), torsion_endo["base"], label="base")
            plt.plot(np.arange(T), torsion_endo["apex"], label="apex")
            plt.plot(np.arange(T), torsion_endo["twist"], label="twist")
            plt.axhline(0, color='black', linestyle='--')
            plt.ylabel("°")
            plt.vlines(T_ES, -10, 10, color="black")
            plt.legend()
            plt.title(f"Subject {subject}: Endocardial Twist Time Series using Slice {basal_slice} to {apical_slice}")
            plt.savefig(f"{sub_dir}/timeseries/twist_endo.png")
            plt.close()

            plt.plot(np.arange(T), torsion_epi["base"], label="base")
            plt.plot(np.arange(T), torsion_epi["apex"], label="apex")
            plt.plot(np.arange(T), torsion_epi["twist"], label="twist")
            plt.axhline(0, color='black', linestyle='--')
            plt.ylabel("°")
            plt.vlines(T_ES, -10, 10, color="black")
            plt.legend()
            plt.title(f"Subject {subject}: Epicardial Twist Time Series using Slice {basal_slice} to {apical_slice}")
            plt.savefig(f"{sub_dir}/timeseries/twist_epi.png")
            plt.close()

            plt.plot(np.arange(T), torsion_global["base"], label="base")
            plt.plot(np.arange(T), torsion_global["apex"], label="apex")
            plt.plot(np.arange(T), torsion_global["twist"], label="twist")
            plt.axhline(0, color='black', linestyle='--')
            plt.ylabel("°")
            plt.vlines(T_ES, -10, 10, color="black")
            plt.legend()
            plt.title(f"Subject {subject}: Global Twist Time Series using Slice {basal_slice} to {apical_slice}")
            plt.savefig(f"{sub_dir}/timeseries/twist_global.png")
            plt.close()

            feature_dict.update(
                {
                    "LV: Endocardial Torsion [°/cm]": np.max(torsion_endo["torsion"]),
                    "LV: Epicardial Torsion [°/cm]": np.max(torsion_epi["torsion"]),
                    "LV: Global Torsion [°/cm]": np.max(torsion_global["torsion"]),
                }
            )

            plt.plot(np.arange(T), torsion_endo["torsion"], label="endo")
            plt.plot(np.arange(T), torsion_epi["torsion"], label="epi")
            plt.plot(np.arange(T), torsion_global["torsion"], label="global")
            plt.axhline(0, color='black', linestyle='--')
            plt.ylabel("°/cm")
            plt.vlines(T_ES, -3, 3, color="black")
            plt.legend()
            plt.title(f"Subject {subject}: Torsion Time Series using Slice {basal_slice} to {apical_slice}")
            plt.savefig(f"{sub_dir}/timeseries/torsion.png")
            plt.close()

            T_endo_torsion_peak = np.argmax(torsion_endo["torsion"])
            T_epi_torsion_peak = np.argmax(torsion_epi["torsion"])
            T_global_torsion_peak = np.argmax(torsion_global["torsion"])

            t_endo_torsion_peak = T_endo_torsion_peak * temporal_resolution * 1000  # unit: ms
            t_epi_torsion_peak = T_epi_torsion_peak * temporal_resolution * 1000
            t_global_torsion_peak = T_global_torsion_peak * temporal_resolution * 1000

            feature_dict.update(
                {
                    "LV: Endocardial Torsion: Time to Peak [ms]": t_endo_torsion_peak,
                    "LV: Epicardial Torsion: Time to Peak [ms]": t_epi_torsion_peak,
                    "LV: Global Torsion: Time to Peak [ms]": t_global_torsion_peak,
                }
            )

            # Define Peak recoil rate is the maximum negative slope of torsion-time curve

            try:
                _, _, _, recoil_rate_endo = analyze_time_series_derivative(
                    time_grid_real,
                    torsion_endo["torsion"],
                    n_pos=0,
                    n_neg=1,
                    method="loess",
                )

                _, _, _, recoil_rate_epi = analyze_time_series_derivative(
                    time_grid_real,
                    torsion_epi["torsion"],
                    n_pos=0,
                    n_neg=1,
                    method="loess",
                )

                _, _, _, recoil_rate_global = analyze_time_series_derivative(
                    time_grid_real,
                    torsion_global["torsion"],
                    n_pos=0,
                    n_neg=1,
                    method="loess",
                )

                logger.info(f"{subject}: Recoil rate calculated.")
                feature_dict.update(
                    {
                        "LV: Endocardial Recoil Rate [°/cm/s]": recoil_rate_endo[0],
                        "LV: Epicardial Recoil Rate [°/cm/s]": recoil_rate_epi[0],
                        "LV: Global Recoil Rate [°/cm/s]": recoil_rate_global[0],
                    }
                )
            except ValueError as e:
                logger.warning(f"{subject}: {e}  No recoil rate calculated.")

        except ValueError as e:
            logger.warning(f"{subject}: {e} No torsion calculated.")
            continue

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "strain")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), index=False)
