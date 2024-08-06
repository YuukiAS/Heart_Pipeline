import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import pickle
from tqdm import tqdm

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config

from utils.log_utils import setup_logging
from utils.cardiac_utils import evaluate_AVPD

logger = setup_logging("eval_ventricular_atrium_feature")

parser = argparse.ArgumentParser()
parser.add_argument("--retest", action="store_true")
parser.add_argument("--data_list", help="List of subjects to be processed", nargs="*", type=int, required=True)
parser.add_argument("--file_name", help="Name of the csv file to save the features")


if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = config.data_visit2_dir if args.retest else config.data_visit1_dir
    features_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir

    ventricular_features_csv = os.path.join(features_dir, "comprehensive", "ventricular_volume.csv")
    ventricular_features_csv = pd.read_csv(ventricular_features_csv)
    atrial_features_csv = os.path.join(features_dir, "comprehensive", "atrial_volume.csv")
    atrial_features_csv = pd.read_csv(atrial_features_csv)

    df = pd.DataFrame()

    for subject in tqdm(args.data_list):
        subject = str(subject)
        subject_int = int(subject)
        sub_dir = os.path.join(data_dir, subject)
        sa_name = f"{sub_dir}/sa.nii.gz"
        la_4ch_name = f"{sub_dir}/la_4ch.nii.gz"
        nim_sa = nib.load(sa_name)
        nim_la_4ch = nib.load(la_4ch_name)
        seg_sa_name = f"{sub_dir}/seg_sa.nii.gz"
        seg4_la_4ch_name = f"{sub_dir}/seg4_la_4ch.nii.gz"
        seg4_la_4ch = nib.load(seg4_la_4ch_name).get_fdata()

        # * To get combined features, we need to make sure basic ventricle/atrium featrues have been extracted

        if subject_int not in ventricular_features_csv["eid"].values:
            logger.error(f"Ventricular features not found for subject {subject}")
            continue

        if subject_int not in atrial_features_csv["eid"].values:
            logger.error(f"Atrial features not found for subject {subject}")
            continue

        logger.info(f"Calculating combined (ventricular+atrial) features for subject {subject}")
        feature_dict = {
            "eid": subject,
        }

        ventricular_features = ventricular_features_csv[ventricular_features_csv["eid"] == subject_int]
        ventricular_time_series_file = os.path.join(sub_dir, "timeseries", "ventricle.pkl")
        try:
            with open(ventricular_time_series_file, "rb") as f:
                ventricular_time_series = pickle.load(f)
        except FileNotFoundError:
            logger.error(f"Ventricular time series not found for subject {subject}")
            continue
        atrial_features = atrial_features_csv[atrial_features_csv["eid"] == subject_int]
        atrial_time_series_file = os.path.join(sub_dir, "timeseries", "atrium.pkl")
        try:
            with open(atrial_time_series_file, "rb") as f:
                atrial_time_series = pickle.load(f)
        except FileNotFoundError:
            logger.error(f"Atrial time series not found for subject {subject}")
            continue

        # * Feature1: IPVT

        # Ref https://link.springer.com/article/10.1007/s00330-018-5571-3#Fig1
        try:
            IPVT = ventricular_features["LV: SV [mL]"].values[0] - atrial_features["LA: Total SV(bip) [mL]"].values[0]
            if IPVT < 0:
                raise ValueError()
            IPVT_ratio = IPVT / atrial_features["LA: Total SV(bip) [mL]"].values[0]
            logger.info(f"{subject}: Extract IPVT features")
            feature_dict.update({
                "IPVT": IPVT,
                "IPVT Ratio": IPVT_ratio,
            })
        except ValueError:
            logger.error(f"{subject}: IPVT is negative, skipped.")

        # * Feature2: Average Atrial Contribution to LVEDV88

        # Ref https://www.sciencedirect.com/science/article/pii/0002934375902296
        LVEDV = ventricular_features["LV: V_ED [mL]"].values[0]
        # Since T_pre_a is determined using ECG, it may not be available for all subjects
        try:
            T_pre_a = atrial_time_series["LA: T_pre_a"]
            LV_pre_a = ventricular_time_series["LV: Volume [mL]"][T_pre_a]
            LVEDV = ventricular_features["LV: V_ED [mL]"].values[0]
            LVESV = ventricular_features["LV: V_ES [mL]"].values[0]
            if LVEDV < LV_pre_a:
                raise ValueError()
            Contribution_AC_LVSV = (LVEDV - LV_pre_a) / (LVEDV - LVESV)
            Contribution_AC_LVEDV = (LVEDV - LV_pre_a) / LVEDV
            logger.info(f"{subject}: Extract % AC features")
            feature_dict.update({
                "LV: V_pre_a [mL]": LV_pre_a,
                "% AC to LVSV [%]": Contribution_AC_LVSV * 100,
                "% AC to LVEDV [%]": Contribution_AC_LVEDV * 100,
            })
        except KeyError:
            logger.warning(f"{subject}: Pre-contraction time point not found, % AC will not be extracted")
        except ValueError:
            logger.error(f"{subject}: Pre-contraction volume is greater than ED volume, skipped.")

        # * Feature3: AVPD

        # Ref https://pubmed.ncbi.nlm.nih.gov/17098822/
        T_ED = ventricular_time_series["LV: T_ED"]
        T_ES = ventricular_time_series["LV: T_ES"]

        long_axis = nim_sa.affine[:3, 2] / np.linalg.norm(nim_sa.affine[:3, 2])
        # define  AV plane descent
        try:
            AVPD = evaluate_AVPD(seg4_la_4ch.astype(np.uint8), nim_la_4ch, long_axis, T_ED, T_ES)
            logger.info(f"{subject}: Extract AVPD features")
            feature_dict.update({
                "AVPD [mm]": AVPD,
        })
        except ValueError as e:
            logger.error(f"{subject}: AVPD extraction failed due to reason: {e}")

        logger.info("All combined feature of ventricle and atrium are extracted successfully!")
        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "combined")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"))
