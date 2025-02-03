import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config

from utils.log_utils import setup_logging
from utils.cardiac_utils import evaluate_valve_length, evaluate_AVPD

logger = setup_logging("eval_ventricular_atrium_feature")

parser = argparse.ArgumentParser()
parser.add_argument("--retest", action="store_true")
parser.add_argument("--data_list", help="List of subjects to be processed", nargs="*", type=int, required=True)
parser.add_argument("--file_name", help="Name of the csv file to save the features")


if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = config.data_visit2_dir if args.retest else config.data_visit1_dir
    features_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir

    ventricular_features_csv = os.path.join(features_dir, "aggregated", "ventricular_volume.csv")
    ventricular_features_csv = pd.read_csv(ventricular_features_csv)
    atrial_features_csv = os.path.join(features_dir, "aggregated", "atrial_volume.csv")
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

        logger.info(f"Calculating combined features using ventricle and atrium for subject {subject}")
        feature_dict = {
            "eid": subject,
        }

        ventricular_features = ventricular_features_csv[ventricular_features_csv["eid"] == subject_int]
        try:
            ventricular_time_series = np.load(os.path.join(sub_dir, "timeseries", "ventricle.npz"))
        except FileNotFoundError:
            logger.error(f"Ventricular features or time series not found for subject {subject}")
            continue
        atrial_features = atrial_features_csv[atrial_features_csv["eid"] == subject_int]
        try:
            atrial_time_series = np.load(os.path.join(sub_dir, "timeseries", "atrium.npz"))
        except FileNotFoundError:
            logger.error(f"Atrial features or time series not found for subject {subject}")
            continue

        T_ED = ventricular_time_series["LV: T_ED [frame]"]
        T_ES = ventricular_time_series["LV: T_ES [frame]"]
        os.makedirs(os.path.join(sub_dir, "visualization", "combined"), exist_ok=True)

        # * Feature1: IPVT

        # Ref Diastolic dysfunction evaluated by cardiac magnetic resonance https://link.springer.com/article/10.1007/s00330-018-5571-3#Fig1
        try:
            IPVT = ventricular_features["LV: SV [mL]"].values[0] - atrial_features["LA: Total SV (bip) [mL]"].values[0]
            if IPVT < 0:
                raise ValueError()
            IPVT_ratio = IPVT / atrial_features["LA: Total SV (bip) [mL]"].values[0]
            logger.info(f"{subject}: Extract IPVT features")
            feature_dict.update(
                {
                    "LVLA: IPVT [mL]": IPVT,
                    "LVLA: IPVT Ratio": IPVT_ratio,
                }
            )
        except ValueError:
            logger.error(f"{subject}: IPVT is negative, skipped.")

        # * Feature2: Average Atrial Contribution to LVEDV

        # Ref Left atrial transport function in myocardial infarction https://www.sciencedirect.com/science/article/pii/0002934375902296
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
            feature_dict.update(
                {
                    "LVLA: % AC to LVSV [%]": Contribution_AC_LVSV * 100,
                    "LVLA: % AC to LVEDV [%]": Contribution_AC_LVEDV * 100,
                }
            )
        except KeyError:
            logger.warning(f"{subject}: Pre-contraction time point not found, % AC will not be extracted")
        except ValueError:
            logger.error(f"{subject}: Pre-contraction volume is greater than ED volume, skipped.")

        # * Feature3: Valve Diameters at ED and ES

        try:
            TA_diameters_ED, MA_diameters_ED, fig_valve_ED = evaluate_valve_length(
                seg4_la_4ch.astype(np.uint8), nim_la_4ch, T_ED, visualize=True
            )
            logger.info(f"{subject}: Extract Tricuspid and Mitral valve diameters")
            feature_dict.update(
                {
                    "Valve: Tricuspid_diameter_ED [cm]": TA_diameters_ED["min"],
                    "Valve: Mitral_diameter_ED [cm]": MA_diameters_ED["min"],
                }
            )
            if TA_diameters_ED["min"] / MA_diameters_ED["min"] > 5:
                logger.warning(f"{subject}: Extremely high TA/MA Diameter Ratio at ED, skipped.")
            else:
                feature_dict.update(
                    {
                        "Valve: TA/MA Ratio_ED": TA_diameters_ED["min"] / MA_diameters_ED["min"],
                    }
                )
            fig_valve_ED.savefig(os.path.join(sub_dir, "visualization", "combined", "valve_ED.png"))
        except ValueError as e:
            logger.error(f"{subject}: Valve diameters at ED failed due to reason: {e}")

        try:
            TA_diameters_ES, MA_diameters_ES, fig_valve_ES = evaluate_valve_length(
                seg4_la_4ch.astype(np.uint8), nim_la_4ch, T_ES, visualize=True
            )
            logger.info(f"{subject}: Extract Tricuspid and Mitral valve diameters")
            feature_dict.update(
                {
                    "Valve: Tricuspid_diameter_ES [cm]": TA_diameters_ES["min"],
                    "Valve: Mitral_diameter_ES [cm]": MA_diameters_ES["min"],
                }
            )
            # no ratio calculated at ES
            fig_valve_ES.savefig(os.path.join(sub_dir, "visualization", "combined", "valve_ES.png"))
        except ValueError as e:
            logger.error(f"{subject}: Valve diameters at ES failed due to reason: {e}")

        # * Feature4: AVPD (AV plane descent)
        # Ref Atrioventricular plane displacement is the major contributor to left ventricular pumping https://pubmed.ncbi.nlm.nih.gov/17098822/

        try:
            AVPD, fig_AVPD = evaluate_AVPD(seg4_la_4ch.astype(np.uint8), nim_la_4ch, T_ED, T_ES, visualize=True)
            logger.info(f"{subject}: Extract AVPD features")
            feature_dict.update(
                {
                    "LVLA: AVPD [cm]": AVPD,
                }
            )
            fig_AVPD.savefig(os.path.join(sub_dir, "visualization", "combined", "AV_plane.png"))
        except ValueError as e:
            logger.error(f"{subject}: AVPD extraction failed due to reason: {e}")

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "ventricular_atrial_feature")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), index=False)
