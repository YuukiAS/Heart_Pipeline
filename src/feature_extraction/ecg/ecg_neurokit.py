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
        file_rest_name = f"{config.features_visit1_dir}/comprehensive/ecg_rest_neurokit.csv"
        file_exercise_name = f"{config.features_visit1_dir}/comprehensive/ecg_exercise_neurokit.csv"
    else:
        file_rest_name = f"{config.features_visit2_dir}/comprehensive/ecg_rest_neurokit.csv"
        file_exercise_name = f"{config.features_visit2_dir}/comprehensive/ecg_exercise_neurokit.csv"

    subjects = os.listdir(data_dir)

    df_rest = pd.DataFrame()
    df_exercise = pd.DataFrame()
    for subject in tqdm(subjects):
        ecg_processor = ECG_Processor(subject, args.retest)

        df_rest_row = pd.DataFrame()
        try:
            voltages_rest = ecg_processor.get_voltages_rest()
            logger.info(f"{subject}: Generate features for ECG rest data")

            for lead_name, lead_voltages in voltages_rest.keys():
                df_rest, info_rest = nk.ecg_process(lead_voltages, sampling_rate=SAMPLING_RATE)
                features_rest_time = nk.hrv_time(df_rest, sampling_rate=SAMPLING_RATE)
                features_rest_time.add_prefix(f"{lead_name}_")

                # concate each one to one row
                df_rest_row = pd.concat([df_rest_row, features_rest_time], axis=1)

            df_rest_row["subject"] = subject
            df_rest = pd.concat([df_rest, df_rest_row], ignore_index=True)

        except ValueError:
            logger.warning(f"ECG rest data does not exist for the subject {subject}")

        df_exercise_row = pd.DataFrame()
        try:
            voltages_rest = ecg_processor.get_voltages_exercise()
            logger.info("{subject}: Generate features for ECG exercise data")

            for lead_name, lead_voltages in voltages_rest.keys():
                df_exercise, info_exercise = nk.ecg_process(lead_voltages, sampling_rate=SAMPLING_RATE)
                features_exercise_time = nk.hrv_time(df_exercise, sampling_rate=SAMPLING_RATE)
                features_exercise_time.add_prefix(f"{lead_name}_")

                # concate each one to one row
                df_exercise_row = pd.concat([df_exercise_row, features_exercise_time], axis=1)

            df_exercise_row["subject"] = subject
            df_exercise = pd.concat([df_exercise, df_exercise_row], ignore_index=True)

        except ValueError:
            logger.warning(f"ECG exercise data does not exist for the subject {subject}")

    df_rest.sort_index(axis=1, inplace=True)
    df_exercise.sort_index(axis=1, inplace=True)
    os.makedirs(os.path.dirname(file_rest_name), exist_ok=True)
    df_rest.to_csv(file_rest_name)
    df_exercise.to_csv(file_exercise_name)