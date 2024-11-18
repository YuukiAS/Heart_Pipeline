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
from utils.analyze_utils import plot_time_series_double_x, plot_time_series_double_x_y, analyze_time_series_derivative
from utils.cardiac_utils import cine_2d_la_motion_and_strain_analysis, evaluate_strain_by_length_la

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
        logger.info(f"Calculating longitudinal strain for subject {subject}")
        sub_dir = os.path.join(data_dir, subject)
        seg_la_name = os.path.join(sub_dir, "seg4_la_4ch.nii.gz")

        nim = nib.load(seg_la_name)
        T = nim.header["dim"][4]
        temporal_resolution = nim.header["pixdim"][4]
        time_grid_point = np.arange(T)
        time_grid_real = time_grid_point * temporal_resolution  # unit: s

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
                    f"LV: Longitudinal (Segment_{i + 1}) [%]": longit_strain[i, :].min(),  # no need to plus one
                }
            )

        feature_dict.update(
            {
                "LV: Longitudinal (Global) [%]": longit_strain[6, :].min(),
            }
        )

        # * Make a time series plot and store the time series of global strain
        logger.info(f"{subject}: Plot time series of global strain.")

        fig, ax1, ax2 = plot_time_series_double_x(
            time_grid_point,
            time_grid_real * 1000,
            longit_strain[6, :],
            "Time [frame]",
            "Time [ms]",
            "Strain [%]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Global Longitudinal Strain (GLS) Time Series (Raw)",
        )
        fig.savefig(f"{sub_dir}/timeseries/gls_raw.png")
        plt.close(fig)

        # Read in important time points
        ventricle = np.load(f"{sub_dir}/timeseries/ventricle.npz")
        atrium = np.load(f"{sub_dir}/timeseries/atrium.npz")

        T_ES = ventricle["LV: T_ES"]
        T_1_3_DD = T_ES + math.ceil((50 - T_ES) / 3)
        T_pre_a = None

        try:
            T_pre_a = atrium["LA: T_pre_a"]
        except KeyError:
            logger.warning(f"{subject}: No atrial contraction time information for strain calculation.")

        # * Feature 1: Strain Rate

        try:
            T_GLSR_pos, T_GLSR_neg, GLSR_pos, GLSR_neg = analyze_time_series_derivative(
                time_grid_real,
                longit_strain[6, :] / 100,  # Since strain is in %
                n_pos=2,
                n_neg=1,
                method="loess",  # for strain rate, we don't use the moving average method
            )

            GLSR_S = GLSR_neg[0]
            GLSR_E = GLSR_pos[np.argmin(T_GLSR_pos)]
            GLSR_A = GLSR_pos[np.argmax(T_GLSR_pos)]
            if np.min(T_GLSR_pos) < T_ES:
                raise ValueError("Time for peak early diastolic longitudinal strain rate should be after ES.")
            feature_dict.update(
                {
                    "LV: Longitudinal Strain Rate (Peak-systolic) [1/s]": GLSR_S,
                    "LV: Longitudinal Strain Rate (Early-diastole) [1/s]": GLSR_E,
                    "LV: Longitudinal Strain Rate (Late-diastole) [1/s]": GLSR_A,
                }
            )
            logger.info(f"{subject}: Global longitudinal strain rate calculated.")
        except ValueError as e:
            logger.warning(f"{subject}: {e}  No global longitudinal strain rate calculated.")

        # * Feature 2: Peak-systolic strain, end-systolic strain, post-systolic strain
        # * Feature 3: Post systolic index (PSI), Time to peak interval, Mechanical dispersion index (MDI)
        # * We record both global ones and ones for each segment

        # Ref https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9409501/pdf/pone.0273419.pdf
        PSS = False  # Post-systolic shortening
        t_longit_strain_peak_list = []  # define used to calculate Mechanical dispersion index

        # Determine the RR interval
        ecg_processor = ECG_Processor(subject, args.retest)

        if config.useECG and not ecg_processor.check_data_rest():
            RR_interval = None
            logger.warning(f"{subject}: No ECG rest data, time to peak strain index will not be calculated.")
        else:
            RR_interval = ecg_processor.determine_RR_interval()  # should be close to MeanNN in neurokit2

        for i in range(7):
            # We will use smoothed strain for all advanced features
            fANCOVA = importr("fANCOVA")
            x_r = FloatVector(time_grid_real)
            y_r_i = FloatVector(longit_strain[6, :])
            loess_fit_i = fANCOVA.loess_as(x_r, y_r_i, degree=2, criterion="gcv")
            loess_x = np.array(loess_fit_i.rx2("x")).reshape(
                T,
            )
            loess_y_i = np.array(loess_fit_i.rx2("fitted"))

            if i == 6:
                GLS_loess_y = loess_y_i

            if i == 6:  # We make a time series plot for global longitudinal strain
                colors = ["blue"] * T
                colors[T_ES] = "red"
                colors[T_1_3_DD] = "green"
                if T_pre_a is not None:
                    colors[T_pre_a] = "orange"

                fig, ax1, ax2 = plot_time_series_double_x_y(
                    time_grid_point,
                    time_grid_real * 1000,
                    longit_strain[6, :],
                    loess_y_i,
                    "Time [frame]",
                    "Time [ms]",
                    "Strain [%]",
                    lambda x: x * temporal_resolution,
                    lambda x: x / temporal_resolution,
                    title=f"Subject {subject}: Global Longitulongit_strain_ESdinal Strain (GLS) Time Series",
                    colors=colors,
                )
                fig.savefig(f"{sub_dir}/timeseries/gls.png")
                plt.close(fig)

            longit_strain_ES = abs(loess_y_i[T_ES])

            T_longit_strain_peak = np.argmax(abs(loess_y_i))
            longit_strain_peak = np.max(abs(loess_y_i))

            T_longit_strain_post = None
            longit_strain_post = None
            if T_longit_strain_peak > T_ES:
                # Post systolic shortening exists
                T_longit_strain_post = T_longit_strain_peak
                longit_strain_post = longit_strain_peak

                if i == 6:
                    feature_dict.update(
                        {
                            "LV: Longitudinal Strain (Global End-systolic, absolute value) [%]": longit_strain_ES,
                            "LV: Longitudinal Strain (Global Post-systolic, absolute value) [%]": longit_strain_post,
                        }
                    )
                    logger.info(f"{subject}: End-systolic, post-systolic longitudinal global strain calculated.")
                else:
                    feature_dict.update(
                        {
                            f"LV: Longitudinal Strain (Segment_{i + 1} " f"End-systolic, absolute value) [%]": longit_strain_ES,
                            f"LV: Longitudinal Strain (Segment_{i + 1} "
                            f"Post-systolic, absolute value) [%]": longit_strain_post,
                        }
                    )
            else:
                if i == 6:
                    feature_dict.update(
                        {
                            "LV: Longitudinal Strain (Global End-systolic, absolute value) [%]": longit_strain_ES,
                            "LV: Longitudinal Strain (Global Peak-systolic, absolute value) [%]": longit_strain_peak,
                        }
                    )
                    logger.info(f"{subject}: End-systolic, peak-systolic longitudinal global strain calculated.")
                else:
                    feature_dict.update(
                        {
                            f"LV: Longitudinal Strain (Segment_{i + 1} " f"End-systolic, absolute value) [%]": longit_strain_ES,
                            f"LV: Longitudinal Strain (Segment_{i + 1} "
                            f"Peak-systolic, absolute value) [%]": longit_strain_peak,
                        }
                    )

            # Ref https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9409501/pdf/pone.0273419.pdf

            if longit_strain_post is not None:
                longit_PSI = (longit_strain_post - longit_strain_ES) / (longit_strain_post)
            else:
                longit_PSI = 0

            if i == 6:
                feature_dict.update(
                    {
                        "LV: Longitudinal Strain: Post-systolic Index (Global) [%]": longit_PSI * 100,
                    }
                )
                logger.info(f"{subject}: Global Post-systolic index calculated.")
            else:
                feature_dict.update(
                    {
                        f"LV: Longitudinal Strain: Post-systolic Index (Segment_{i + 1}) [%]": longit_PSI * 100,
                    }
                )

                if longit_PSI > 0.2:
                    PSS = True

            t_longit_strain_peak = T_longit_strain_peak * temporal_resolution * 1000  # unit: ms

            feature_dict.update(
                {
                    "LV: Longitudinal Strain: Time to Peak [ms]": t_longit_strain_peak,
                }
            )

            if i == 6:
                logger.info(f"{subject}: ECG rest data exists, calculate global time to peak strain index.")
            else:
                t_longit_strain_peak_list.append(t_longit_strain_peak)

            if RR_interval is not None:
                t_longit_strain_peak_index = t_longit_strain_peak / RR_interval

                feature_dict.update(
                    {
                        "LV: Longitudinal Strain: Time to Peak Index": t_longit_strain_peak_index,
                    }
                )

        # Ref https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9409501/pdf/pone.0273419.pdf
        # Since analysis is based on frames, MSI should only be reported if segments have distinct time to peak strain
        if np.unique(t_longit_strain_peak_list).shape[0] > 1:
            MSI = np.std(t_longit_strain_peak_list)
            feature_dict.update(
                {
                    "LV: Longitudinal Strain: Mechanical Dispersion Index [ms]": MSI,
                }
            )
            logger.info(f"{subject}: Mechanical dispersion index calculated.")

        # * Feature 4 Systolic stretch, Strain imaging diastolic index (SI-DI)

        # Ref https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9349311/pdf/380_2022_Article_2047.pdf

        GLS_loess_y_systole = GLS_loess_y[:T_ES]
        try:
            if np.max(GLS_loess_y_systole) < 0:
                raise ValueError("No positive strain in early systole.")
            longit_strain_systole_positive = np.max(GLS_loess_y_systole)
            longit_strain_systole_negative = np.min(GLS_loess_y_systole)
            systolic_stretch = longit_strain_systole_positive / (longit_strain_systole_positive - longit_strain_systole_negative)
            feature_dict.update(
                {
                    "LV: Longitudinal Strain: Systolic Stretch [%]": systolic_stretch * 100,
                }
            )
            logger.info(f"{subject}: Systolic stretch calculated.")
        except ValueError:
            logger.warning(f"{subject}: No positive strain in early systole, systolic stretch will not be calculated.")

        # Ref https://www.sciencedirect.com/science/article/pii/S0914508711000116

        longit_strain_1_3_DD = abs(GLS_loess_y[T_1_3_DD])

        longit_SI_DI = (longit_strain_ES - longit_strain_1_3_DD) / longit_strain_ES

        feature_dict.update(
            {
                "LV: Longitudinal Strain: Strain Imaging Diastolic Index [%]": longit_SI_DI * 100,
            }
        )

        logger.info(f"{subject}: Strain imaging diastolic index (SI-DI) calculated.")

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "strain")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), index=False)
