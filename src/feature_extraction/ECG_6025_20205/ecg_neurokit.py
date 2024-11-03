"""
Extract ECG features using Neurokit2 package in python.
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm
import sys
import neurokit2 as nk

from ecg_processor import ECG_Processor

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config
from utils.log_utils import setup_logging

logger = setup_logging("ecg_neurokit")

parser = argparse.ArgumentParser()
parser.add_argument("--retest", action="store_true")

SAMPLING_RATE = 500


if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = config.data_visit2_dir if args.retest else config.data_visit1_dir
    if not args.retest:
        # Name of the csv file to save the features
        file_rest_name = f"{config.features_visit1_dir}/aggregated/ecg_rest_neurokit.csv"
        file_exercise_name = f"{config.features_visit1_dir}/aggregated/ecg_exercise_neurokit.csv"
    else:
        file_rest_name = f"{config.features_visit2_dir}/aggregated/ecg_rest_neurokit.csv"
        file_exercise_name = f"{config.features_visit2_dir}/aggregated/ecg_exercise_neurokit.csv"

    subjects = os.listdir(data_dir)

    df_rest = pd.DataFrame()
    df_exercise = pd.DataFrame()
    for subject in tqdm(subjects):
        ecg_processor = ECG_Processor(subject, args.retest)

        df_rest_row = pd.DataFrame()
        try:
            voltages_rest = ecg_processor.get_voltages_rest()

            for lead_name, lead_voltages in voltages_rest.items():
                data_rest, info_rest = nk.ecg_process(lead_voltages, sampling_rate=SAMPLING_RATE)
                features_rest_time = nk.hrv_time(data_rest, sampling_rate=SAMPLING_RATE)
                features_rest_time = features_rest_time.add_prefix(f"lead{lead_name}_")

                # concate each one to one row
                df_rest_row = pd.concat([df_rest_row, features_rest_time], axis=1)

            logger.info(f"{subject}: Generate features for ECG rest data")
            df_rest_row["eid"] = subject
            df_rest = pd.concat([df_rest, df_rest_row], ignore_index=True)

        except (ValueError, TypeError, IndexError, ZeroDivisionError) as e:
            logger.warning(f"{subject}: Fail to extract rest features: {e}")

        df_exercise_row = pd.DataFrame()
        try:
            voltages_divided = ecg_processor.get_voltages_exercise()
            phases = ["pretest", "exercise_constant", "exercise_ramping", "rest"]

            for phase, voltages in zip(phases, voltages_divided):
                for lead_name, lead_voltages in voltages.items():
                    data_exercise, info = nk.ecg_process(lead_voltages, sampling_rate=SAMPLING_RATE)
                    features_time = nk.hrv_time(data_exercise, sampling_rate=SAMPLING_RATE)
                    features_time = features_time.add_prefix(f"{phase}_lead{lead_name}_")
                    df_exercise_row = pd.concat([df_exercise_row, features_time], axis=1)

                    features_frequency = nk.hrv_frequency(data_exercise, sampling_rate=SAMPLING_RATE)
                    features_frequency = features_frequency.add_prefix(f"{phase}_lead{lead_name}_")
                    df_exercise_row = pd.concat([df_exercise_row, features_frequency], axis=1)

                    if phase not in ["pretest", "rest"]:
                        features_nonlinear = nk.hrv_nonlinear(data_exercise, sampling_rate=SAMPLING_RATE)
                        features_nonlinear = features_nonlinear.add_prefix(f"{phase}_lead{lead_name}_")
                        df_exercise_row = pd.concat([df_exercise_row, features_nonlinear], axis=1)

            logger.info(f"{subject}: Generate features for ECG exercise data")
            df_exercise_row["eid"] = subject
            df_exercise = pd.concat([df_exercise, df_exercise_row], ignore_index=True)

        except (ValueError, TypeError, IndexError, ZeroDivisionError) as e:
            logger.warning(f"{subject}: Fail to extract exercise features: {e}")

    # Clean data
    df_rest.dropna(axis=1, how="all", inplace=True)
    df_rest = df_rest.loc[:, df_rest.nunique() > 1]
    df_rest.sort_index(axis=1, inplace=True)
    df_exercise.dropna(axis=1, how="all", inplace=True)
    df_exercise = df_exercise.loc[:, df_exercise.nunique() > 1]
    df_exercise.sort_index(axis=1, inplace=True)

    os.makedirs(os.path.dirname(file_rest_name), exist_ok=True)
    df_rest.to_csv(file_rest_name, index=False)
    df_exercise.to_csv(file_exercise_name, index=False)
