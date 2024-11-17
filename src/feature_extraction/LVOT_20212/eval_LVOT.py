import os
import shutil
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import cv2
from collections import OrderedDict
from tqdm import tqdm
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import config
from utils.log_utils import setup_logging
from utils.quality_control_utils import lvot_pass_quality_control

logger = setup_logging("eval_lvot")

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
        ID = int(subject)
        logger.info(f"Calculating aortic annulus and root diameters for subject {subject}")
        sub_dir = os.path.join(data_dir, subject)

        # * Use the rotated and bias corrected images instead of raw ones
        lvot_name = os.path.join(sub_dir, "lvot_processed.nii.gz")
        seg_lvot_name = os.path.join(sub_dir, "seg_lvot.nii.gz")

        if not os.path.exists(lvot_name) or not os.path.exists(seg_lvot_name):
            logger.error(f"LVOT MRI file for {subject} or its segmentation does not exist")
            continue

        lvot = nib.load(lvot_name).get_fdata()
        affine = nib.load(lvot_name).affine
        seg_lvot = nib.load(seg_lvot_name).get_fdata()
        T = seg_lvot.shape[3]
        pixdim = nib.load(lvot_name).header["pixdim"][1:4]

        feature_dict = {
            "eid": subject,
        }

        L = {"aortic_valve_annulus": [], "aortic_sinuses": [], "sinotubular_junction": []}

        shutil.rmtree(os.path.join(sub_dir, "visualization", "aorta"), ignore_errors=True)
        os.makedirs(os.path.join(sub_dir, "visualization", "aorta"), exist_ok=True)
        logger.info(f"{subject}: Start calculating aortic annulus and root diameters")
        for t in range(T):
            if not lvot_pass_quality_control(seg_lvot_name, t):
                logger.warning(f"{subject}: Quality control for LVOT failed at time frame {t}")
                continue
            seg_lvot_t = seg_lvot[:, :, 0, t]

            # * Calculate annulus and sinotubular junction first

            mask1 = seg_lvot_t == 1  # mask for aortic root
            mask1 = mask1.astype(np.uint8)
            mask1 = np.ascontiguousarray(mask1)

            mask1_contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask1_contour = mask1_contours[0]
            mask1_border = np.zeros_like(mask1)
            cv2.drawContours(mask1_border, [mask1_contour], -1, 1, -1)

            epsilon = 0.01
            approx_corners = cv2.approxPolyDP(mask1_contour, epsilon * cv2.arcLength(mask1_contour, True), True)
            while len(approx_corners) > 4:
                epsilon += 0.01
                approx_corners = cv2.approxPolyDP(mask1_contour, epsilon * cv2.arcLength(mask1_contour, True), True)
            if epsilon > 0.1:
                logger.warning(f"{subject}-Timeframe {t}: Too many corners detected")
                continue

            corner_points = [(point[0][0], point[0][1]) for point in approx_corners]

            mask2 = seg_lvot_t == 2  # mask for LV outflow tract
            mask1_2_nan = np.where((seg_lvot_t == 0) | (seg_lvot_t == 3), np.nan, seg_lvot_t)
            mask2_coords = np.column_stack(np.where(mask2))
            mask2_coords = np.column_stack((mask2_coords[:, 1], mask2_coords[:, 0]))  # exchange x and y

            distances = []
            for point in mask2_coords:
                x, y = point
                for corner in corner_points:
                    distance = np.sqrt((x - corner[0]) ** 2 + (y - corner[1]) ** 2)
                    distances.append((distance, corner))
            distances.sort(key=lambda x: x[0])
            sorted_corners = [corner for _, corner in distances]
            unique_corners = list(OrderedDict.fromkeys(sorted_corners))
            bottom_two = unique_corners[:2]  # define valve aortic annulus
            top_two = unique_corners[2:]  # define sinotubular junction

            top_two_real = [np.dot(affine, np.array([*point, 0, 1]))[:3] for point in top_two]
            bottom_two_real = [np.dot(affine, np.array([*point, 0, 1]))[:3] for point in bottom_two]

            # unit: mm
            diameter_sinotubular = np.linalg.norm(np.array(top_two_real[0]) - np.array(top_two_real[1]))
            L["sinotubular_junction"].append(diameter_sinotubular)
            diameter_annulus = np.linalg.norm(np.array(bottom_two_real[0]) - np.array(bottom_two_real[1]))
            L["aortic_valve_annulus"].append(diameter_annulus)

            # * Calculate sinuses of valsalva, which is the largest diameter of the aortic root

            top_vector = np.array(top_two[1]) - np.array(top_two[0])
            top_vector_normalized = top_vector / np.linalg.norm(top_vector)
            bottom_vector = np.array(bottom_two[1]) - np.array(bottom_two[0])
            bottom_vector_normalized = bottom_vector / np.linalg.norm(bottom_vector)
            if top_vector_normalized[0] * bottom_vector_normalized[0] < 0:
                bottom_vector_normalized = -bottom_vector_normalized

            mask1_contour_points = mask1_contour[:, 0, :]

            max_length = 0
            middle_two = None
            for i, point1 in enumerate(mask1_contour_points):
                for j, point2 in enumerate(mask1_contour_points):
                    if i == j:
                        continue
                    vector = point2 - point1
                    vector_normalized = vector / np.linalg.norm(vector)
                    angle_cos_1 = np.dot(vector_normalized, top_vector_normalized)
                    angle_cos_2 = np.dot(vector_normalized, bottom_vector_normalized)
                    # if angle_cos close to 1, then lines are approximately parallel
                    if abs(angle_cos_1) > 0.995 or abs(angle_cos_2) > 0.995:
                        length = np.linalg.norm(point2 - point1)
                        if length > max_length:
                            max_length = length
                            middle_two = (point1, point2)

            middle_two_real = [np.dot(affine, np.array([*point, 0, 1]))[:3] for point in middle_two]
            diameter_aortic_sinus = np.linalg.norm(np.array(middle_two_real[0]) - np.array(middle_two_real[1]))

            if diameter_aortic_sinus <= max(diameter_sinotubular, diameter_annulus):
                logger.warning(
                    f"{subject}-Timeframe {t}: Aortic sinus should be larger than sinotubular junction and aortic annulus"
                )
                continue

            L["aortic_sinuses"].append(diameter_aortic_sinus)

            # Make some visualizations
            plt.imshow(lvot[:, :, 0, t], cmap="gray")
            plt.imshow(mask1_2_nan, cmap="jet", alpha=0.5)
            plt.scatter(*zip(*top_two), c="blue", s=4, label="Sinotubular Junction")
            plt.scatter(*zip(*bottom_two), c="red", s=4, label="Aortic Valve Annulus")
            plt.scatter(*zip(*middle_two), c="green", s=4, label="Aortic Sinuses")
            plt.plot([top_two[0][0], top_two[1][0]], [top_two[0][1], top_two[1][1]], color="blue", linestyle="--", linewidth=1.5)
            plt.plot(
                [bottom_two[0][0], bottom_two[1][0]],
                [bottom_two[0][1], bottom_two[1][1]],
                color="red",
                linestyle="--",
                linewidth=1.5,
            )
            plt.plot(
                [middle_two[0][0], middle_two[1][0]],
                [middle_two[0][1], middle_two[1][1]],
                color="green",
                linestyle="--",
                linewidth=1.5,
            )
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(sub_dir, "visualization", "aorta", f"LVOT_{t}.png"))
            plt.close()

        # todo Indexed version
        logger.info(f"{subject}: Aortic annulus and root diameters calculated for {len(L['aortic_sinuses'])} time frames")

        feature_dict.update(
            {
                "LVOT: Aortic Valve Annulus Diameter (mm)": np.median(L["aortic_valve_annulus"]),
                "LVOT: Aortic Sinuses Diameter (mm)": np.median(L["aortic_sinuses"]),
                "LVOT: Sinotubular Junction Diameter (mm)": np.median(L["sinotubular_junction"]),
            }
        )

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "aorta_diameter")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df = df[[col for col in df.columns if col != "eid"] + ["eid"]]  # move 'eid' to the last column
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), index=False)
