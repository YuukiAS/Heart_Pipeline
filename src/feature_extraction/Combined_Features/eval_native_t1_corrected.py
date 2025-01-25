"""
Normal myocardium blood averages approximately 6% of the myocardium, and the T1 value of blood is higher than that of myocardium.
Thus the T1 value of the myocardium is expected to be influenced by both the T1 value, and the amount of blood in the myocardium.
Please refer to paper "Blood correction reduces variability and gender differences in native myocardial T1 values" https://doi.org/10.1186/s12968-017-0353-7
"""

import os
import argparse
import pandas as pd
from scipy.stats import linregress
from tqdm import tqdm

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config

from utils.log_utils import setup_logging


logger = setup_logging("eval_native_t1_corrected")

parser = argparse.ArgumentParser()
parser.add_argument("--retest", action="store_true")
parser.add_argument("--data_list", help="List of subjects to be processed", nargs="*", type=int, required=True)
parser.add_argument("--file_name", help="Name of the csv file to save the features")

if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = config.data_visit2_dir if args.retest else config.data_visit1_dir
    features_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir

    # To calculate the corrected native T1 value, we need to obtain the mean of uncorrected ones in first round

    t1_features_csv = os.path.join(features_dir, "aggregated", "native_T1.csv")
    t1_features_csv = pd.read_csv(t1_features_csv)
    # exclude nan values (failed subjects)
    t1_features_csv = t1_features_csv.dropna(
        subset=[
            "Native T1: Myocardium-Global [ms]",
            "Native T1: Myocardium-IVS [ms]",
            "Native T1: Myocardium-FW [ms]",
            "Native T1: LV Blood Pool [ms]",
            "Native T1: RV Blood Pool [ms]",
        ]
    )
    t1_features_csv["Native T1: Blood Pool [ms]"] = (
        t1_features_csv["Native T1: LV Blood Pool [ms]"] + t1_features_csv["Native T1: RV Blood Pool [ms]"]
    ) / 2
    t1_blood_mean = t1_features_csv["Native T1: Blood Pool [ms]"].mean()
    r1_blood_mean = 1 / t1_blood_mean

    df = pd.DataFrame()

    for subject in tqdm(args.data_list):
        subject = str(subject)
        subject_int = int(subject)
        sub_dir = os.path.join(data_dir, subject)

        if subject_int not in t1_features_csv["eid"].values:
            logger.error(f"Native T1 features not found for subject {subject}")
            continue

        feature_dict = {
            "eid": subject,
        }
        try:
            t1_features = t1_features_csv[t1_features_csv["eid"] == subject_int]
            t1_global_uncorrected = t1_features["Native T1: Myocardium-Global [ms]"].values[0]
            t1_IVS_uncorrected = t1_features["Native T1: Myocardium-IVS [ms]"].values[0]
            t1_FW_uncorrected = t1_features["Native T1: Myocardium-FW [ms]"].values[0]
            t1_blood_subject = t1_features["Native T1: Blood Pool [ms]"].values[0]
            r1_blood_subject = 1 / t1_blood_subject

            logger.info("Calculating alpha to be used for correction")
            # Ref https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5381013/pdf/12968_2017_Article_353.pdf

            alpha_global, _, _, _, _ = linregress(
                t1_features_csv["Native T1: Blood Pool [ms]"],
                t1_features_csv["Native T1: Myocardium-Global [ms]"],
            )
            alpha_IVS, _, _, _, _ = linregress(
                t1_features_csv["Native T1: Blood Pool [ms]"],
                t1_features_csv["Native T1: Myocardium-IVS [ms]"],
            )
            alpha_FW, _, _, _, _ = linregress(
                t1_features_csv["Native T1: Blood Pool [ms]"],
                t1_features_csv["Native T1: Myocardium-FW [ms]"],
            )
            logger.info(f"Alpha values: Global {alpha_global}, IVS {alpha_IVS}, FW {alpha_FW}")

            logger.info(f"Calculating corrected native T1 for subject {subject}")
            t1_global_corrected = t1_global_uncorrected + alpha_global * (r1_blood_mean - r1_blood_subject) * 1000**2
            logger.info(f"{subject}: T1 value for all myocardium is {t1_global_uncorrected:.2f} -> {t1_global_corrected:.2f} ms")
            t1_IVS_corrected = t1_IVS_uncorrected + alpha_IVS * (r1_blood_mean - r1_blood_subject) * 1000**2
            logger.info(f"{subject}: T1 value for IVS myocardium is {t1_IVS_uncorrected:.2f} -> {t1_IVS_corrected:.2f} ms")
            t1_FW_corrected = t1_FW_uncorrected + alpha_FW * (r1_blood_mean - r1_blood_subject) * 1000**2
            logger.info(f"{subject}: T1 value for FW myocardium is {t1_FW_uncorrected:.2f} -> {t1_FW_corrected:.2f} ms")

            feature_dict.update(
                {
                    "Native T1-Corrected: Myocardium-Global-corrected [ms]": f"{t1_global_corrected:.2f}",
                    "Native T1-Corrected: Myocardium-IVS-corrected [ms]": f"{t1_IVS_corrected:.2f}",
                    "Native T1-Corrected: Myocardium-FW-corrected [ms]": f"{t1_FW_corrected:.2f}",
                }
            )
        except KeyError as e:
            logger.error(f"KeyError: {e} for subject {subject}")
            continue

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "native_T1_corrected")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), index=False)
