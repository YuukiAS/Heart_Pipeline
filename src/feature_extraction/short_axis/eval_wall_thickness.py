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
import argparse
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import vtk

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config

from utils.log_utils import setup_logging
from utils.quality_control_utils import sa_pass_quality_control
from utils.cardiac_utils import evaluate_wall_thickness


logger = setup_logging("eval_wall_thickness")

parser = argparse.ArgumentParser()
parser.add_argument("--retest", action="store_true")
parser.add_argument("--data_list", help="List of subjects to be processed", nargs="*", type=int, required=True)
parser.add_argument("--file_name", help="Name of the csv file to save the features")

# todo: Move to cardiac utils

# def fractal_dimension(nim_sa_ED: nib.nifti1.Nifti1Image, seg_sa_ED: Tuple[float, float, float]):
#     fds = np.array([])

#     img_sa = nim_sa_ED.get_fdata()
#     for i in range(0, seg_sa_ED.shape[2]):
#         seg_endo = seg_sa_ED[:, :, i] == 1
#         img_endo = img_sa[:, :, i] * seg_endo
#         if img_endo.sum() == 0:
#             continue
#         img_endo = (255 * (img_endo - np.min(img_endo)) / (np.max(img_endo) - np.min(img_endo))).astype(np.uint8)
#         # plt.imshow(img_endo, cmap="gray")
#         seg_backgroud = (img_endo > 0).astype(np.uint8)
#         adaptive_thresh = cv2.adaptiveThreshold(
#             img_endo, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#         seg_endo_trabeculation = cv2.bitwise_and(adaptive_thresh, seg_backgroud)
#         # plt.imshow(seg_endo_trabeculation, cmap="gray")

#         # Find a bounding box that contain the endocardium and trabeculation
#         coords = cv2.findNonZero(seg_endo_trabeculation)
#         x, y, w, h = cv2.boundingRect(coords)
#         # print(f"{i}: {w}, {h}")
#         if w < 20 or h < 20:
#             continue
#         x -= 10
#         y -= 10
#         w += 20
#         h += 20
#         seg_endo_trabeculation_cropped = seg_endo_trabeculation[y:y + h, x:x + w]

#         contours, _ = cv2.findContours(seg_endo_trabeculation_cropped, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#         seg_endo_trabeculation_contour = np.zeros_like(seg_endo_trabeculation_cropped)
#         cv2.drawContours(seg_endo_trabeculation_contour, contours, -1, 255, 1)

#         scale = np.arange(0.05, 0.5, 0.01)
#         bins = scale * seg_endo_trabeculation_cropped.shape[0]
#         ps.metrics.boxcount(seg_endo_trabeculation_contour, bins=bins)
#         boxcount_data = ps.metrics.boxcount(seg_endo_trabeculation_contour, bins=bins)
#         slope, _, _, _, _ = linregress(np.log(scale), np.log(boxcount_data.count))
#         slope = abs(slope)
#         if slope < 1 or slope > 2:
#             raise ValueError("Fractal dimension should lie between 1 and 2.")
#         fds.append(slope)

#     return fds.mean()


if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = config.data_visit2_dir if args.retest else config.data_visit1_dir

    df = pd.DataFrame()

    for subject in tqdm(args.data_list):
        subject = str(subject)
        logger.info(f"Calculating myocardial wall thickness for subject {subject}")
        sub_dir = os.path.join(data_dir, subject)

        # Quality control for segmentation at ED
        seg_sa_name = f"{sub_dir}/seg_sa.nii.gz"
        if not os.path.exists(seg_sa_name):
            logger.error(f"Segmentation of short axis file for {subject} does not exist")
            continue

        if not sa_pass_quality_control(seg_sa_name):  # default t = 0
            # If the segmentation quality is low, evaluation of wall thickness may fail.
            logger.error(f"{subject}: seg_sa does not pass sa_pass_quality_control, skipped.")
            continue

        nim_sa = nib.load(seg_sa_name)
        seg_sa = nim_sa.get_fdata()

        feature_dict = {
            "eid": subject,
        }

        # Evaluate myocardial wall thickness
        # Note that measurements on long axis at basal and mid-cavity level have been shown to be significantly
        # greater compared to short axis measurements; while apical level demonstrates the opposite trend
        # Ref https://jcmr-online.biomedcentral.com/articles/10.1186/s12968-020-00683-3 for reference range
        # todo: At ED?
        endo_poly, epi_poly, index, table_thickness, table_thickness_max = evaluate_wall_thickness(
            seg_sa, 
            nim_sa,
            save_epi_contour=True
        )
        logger.info("Wall thickness evaluation completed, start saving data")

        endo_writer = vtk.vtkPolyDataWriter()
        endo_output_name = f"{sub_dir}/landmark/endo.vtk"
        endo_writer.SetFileName(endo_output_name)
        endo_writer.SetInputData(endo_poly)
        endo_writer.Write()

        epi_writer = vtk.vtkPolyDataWriter()
        epi_output_name = f"{sub_dir}/landmark/epi.vtk"
        epi_writer.SetFileName(epi_output_name)
        epi_writer.SetInputData(epi_poly)
        epi_writer.Write()

        # 16 segment + 1 global
        for i in range(16):
            feature_dict.update(
                {
                    f"WT: AHA_{i + 1} (mm)": table_thickness[i],
                    f"WT: AHA_{i + 1}_max (mm)": table_thickness_max[i],
                }
            )
        feature_dict.update(
            {
                "WT: Global (mm)": table_thickness[16],
                "WT: Global_max (mm)": table_thickness_max[16],
            }
        )

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    # Save wall thickness for all the subjects
    # Check https://www.pmod.com/files/download/v34/doc/pcardp/3618.gif for illustration
    # Basal Segments: 1 Basal Anterior, 2 Basal Anteroseptal, 3 Basal Inferoseptal, 4 Basal Inferior, 5 Basal Inferolateral, 6 Basal Anterolateral
    # Mid-cavity Segments: 7 Mid Anterior, 8 Mid Anteroseptal, 9 Mid Inferoseptal, 10 Mid Inferior, 11 Mid Inferolateral, 12 Mid Anterolateral
    # Apical Segments: 13 Apical Anterior, 14 Apical Septal, 15 Apical Inferior, 16 Apical Lateral
    # Illustration: https://media.springernature.com/lw685/springer-static/image/chp%3A10.1007%2F978-3-030-63560-2_16/MediaObjects/496124_1_En_16_Fig13_HTML.png

    # todo: Explainable paper features

    # * Feature 2: Fractal dimension for trabeculation
    # Refer  https://jcmr-online.biomedcentral.com/articles/10.1186/1532-429X-15-36

    fractal_dimension(nim_sa, seg_sa)



    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "wall_thickness")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"))
