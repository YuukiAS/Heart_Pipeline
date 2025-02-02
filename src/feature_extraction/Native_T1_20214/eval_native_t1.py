import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import argparse
import pandas as pd
import nibabel as nib
from tqdm import tqdm

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config

from utils.log_utils import setup_logging
from utils.quality_control_utils import shmolli_pass_quality_control
from utils.cardiac_utils import evaluate_t1_uncorrected


logger = setup_logging("eval_native_t1_uncorrected")

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
        sub_dir = os.path.join(data_dir, subject)
        ShMOLLI_name = f"{sub_dir}/shmolli_t1map.nii.gz"
        seg_ShMOLLI_name = f"{sub_dir}/seg_shmolli_t1map.nii.gz"

        if not os.path.exists(ShMOLLI_name):
            logger.error(f"Native T1 mapping file for {subject} does not exist")
            continue

        if not os.path.exists(seg_ShMOLLI_name):
            logger.error(f"Segmentation of native T1 mapping file for {subject} does not exist")
            continue

        labels = {"Myo": 1, "LV": 2, "RV": 3}

        if not shmolli_pass_quality_control(seg_ShMOLLI_name, labels):
            logger.error(f"{subject}: seg_ShMOLLI does not pass quality control, skipped.")
            continue

        nim_ShMOLLI = nib.load(ShMOLLI_name)
        ShMOOLI = nim_ShMOLLI.get_fdata()
        nim_seg_ShMOLLI = nib.load(seg_ShMOLLI_name)
        seg_ShMOLLI = nim_seg_ShMOLLI.get_fdata()
        seg_ShMOLLI_nan = np.where(seg_ShMOLLI == 0, np.nan, seg_ShMOLLI)

        feature_dict = {
            "eid": subject,
        }

        logger.info(f"Make visualization of raw native T1 for subject {subject}")
        os.makedirs(f"{sub_dir}/visualization/ventricle", exist_ok=True)
        plt.title("Native T1")
        plt.imshow(ShMOOLI[:, :, 0, 0], cmap="gray")
        plt.imshow(seg_ShMOLLI_nan[:, :, 0, 0], cmap="jet", alpha=0.5)
        plt.colorbar()
        legend_labels = ["Myocardium", "LV", "RV"]
        colors = ["blue", "green", "red"]
        patches = [Patch(color=colors[i], label=legend_labels[i]) for i in range(len(legend_labels))]
        plt.legend(handles=patches, loc="lower right")
        plt.savefig(f"{sub_dir}/visualization/ventricle/native_t1.png")
        plt.close()

        logger.info(f"Calculating uncorrected native T1 for subject {subject}")
        # Ref https://jcmr-online.biomedcentral.com/articles/10.1186/s12968-020-00650-y
        try:
            t1_global_uncorrected, t1_IVS_uncorrected, t1_FW_uncorrected, t1_blood_left, t1_blood_right, figure = (
                evaluate_t1_uncorrected(ShMOOLI[:, :, 0, 0], seg_ShMOLLI[:, :, 0, 0], labels)
            )
            figure.savefig(f"{sub_dir}/visualization/ventricle/native_t1_ivs_fw_blood.png")

            feature_dict.update(
                {
                    "Native T1: Myocardium-Global [ms]": f"{t1_global_uncorrected:.2f}",
                    "Native T1: Myocardium-IVS [ms]": f"{t1_IVS_uncorrected:.2f}",
                    "Native T1: Myocardium-FW [ms]": f"{t1_FW_uncorrected:.2f}",
                    "Native T1: LV Blood Pool [ms]": f"{t1_blood_left:.2f}",
                    "Native T1: RV Blood Pool [ms]": f"{t1_blood_right:.2f}",
                }
            )
        except (ValueError, IndexError) as e:
            logger.error(f"Error in calculating uncorrected native T1 for subject {subject}: {e}")
            continue

        # Note that the corrected version requires a linear regression and need to be obtained after first round

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "native_T1")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), index=False)
