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
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
import vtk

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config

from utils.log_utils import setup_logging
from utils.quality_control_utils import sa_pass_quality_control
from utils.analyze_utils import plot_bulls_eye
from utils.cardiac_utils import evaluate_wall_thickness, evaluate_radius_thickness_disparity, fractal_dimension
from utils.biobank_utils import query_BSA


logger = setup_logging("eval_wall_thickness")

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
        logger.info(f"Calculating myocardial wall thickness for subject {subject}")
        sub_dir = os.path.join(data_dir, subject)
        sa_name = f"{sub_dir}/sa.nii.gz"
        sa_ED_name = f"{sub_dir}/sa_ED.nii.gz"
        seg_sa_name = f"{sub_dir}/seg_sa.nii.gz"
        seg_sa_ED_name = f"{sub_dir}/seg_sa_ED.nii.gz"
        seg_sa_ES_name = f"{sub_dir}/seg_sa_ES.nii.gz"

        nim_sa = nib.load(sa_name)
        nim_sa_ED = nib.load(sa_ED_name)

        # Quality control for segmentation at ED

        if not os.path.exists(sa_name):
            logger.error(f"Short axis file for {subject} does not exist")
            continue

        if not os.path.exists(seg_sa_name):
            logger.error(f"Segmentation of short axis file for {subject} does not exist")
            continue

        if not sa_pass_quality_control(seg_sa_name):  # default t = 0
            # If the segmentation quality is low, evaluation of wall thickness may fail.
            logger.error(f"{subject}: seg_sa does not pass quality control, skipped.")
            continue

        nim_seg_sa = nib.load(seg_sa_name)
        nim_seg_sa_ED = nib.load(seg_sa_ED_name)
        seg_sa = nim_seg_sa.get_fdata()
        seg_sa_ED = nim_seg_sa_ED.get_fdata()
        seg_sa_ES = nib.load(seg_sa_ES_name).get_fdata()

        feature_dict = {
            "eid": subject,
        }

        # * Basic Feature: Evaluate myocardial wall thickness at ED (17 segment). Note segment 17 is usually excluded
        # Note that measurements on long axis at basal and mid-cavity level have been shown to be greater compared to short axis.
        # While apical level demonstrates the opposite trend such phenomenon.

        endo_poly_ED, epi_poly_ED, wall_thickness_ED, _, _ = evaluate_wall_thickness(seg_sa_ED, nim_sa_ED)

        endo_writer = vtk.vtkPolyDataWriter()
        endo_output_name_ED = f"{sub_dir}/landmark/myocardium_endo_ED.vtk"
        endo_writer.SetFileName(endo_output_name_ED)
        endo_writer.SetInputData(endo_poly_ED)
        endo_writer.Write()

        epi_writer = vtk.vtkPolyDataWriter()
        epi_output_name_ED = f"{sub_dir}/landmark/myocardium_epi_ED.vtk"
        epi_writer.SetFileName(epi_output_name_ED)
        epi_writer.SetInputData(epi_poly_ED)
        epi_writer.Write()
        logger.info(f"{subject}: Wall-thickness evaluation at ED completed, myocardial contours are saved")

        # 16 segment + 1 global
        # * The mean is the major feature, as more reference range will report such feature instead of the max.
        for i in range(16):
            feature_dict.update(
                {
                    f"Myo: Thickness (AHA_{i + 1}) [mm]": wall_thickness_ED[i],  # mean
                }
            )
        feature_dict.update(
            {
                "Myo: Thickness (Global) [mm]": wall_thickness_ED[16],
            }
        )

        # * Feature 1: Calculate the wall thickening ratio
        # Ref Reference parameters for left ventricular wall thickness, thickening, and motion in stress myocardial perfusion CT https://doi.org/10.1016/j.clinimag.2019.04.002

        # We only record the mean wall thickness, not the maximal or minimal ones
        endo_poly_ES, epi_poly_ES, wall_thickness_ES, _, _ = evaluate_wall_thickness(seg_sa_ES, nim_sa_ED)

        thickening = np.zeros(17)
        thickening = (wall_thickness_ES - wall_thickness_ED) / wall_thickness_ED * 100

        for i in range(16):
            feature_dict.update(
                {
                    f"Myo: Thickening (AHA_{i + 1}) [%]": thickening[i],
                }
            )
        feature_dict.update(
            {
                "Myo: Thickening (Global) [%]": thickening[16],
            }
        )

        endo_writer = vtk.vtkPolyDataWriter()
        endo_output_name_ES = f"{sub_dir}/landmark/myocardium_endo_ES.vtk"
        endo_writer.SetFileName(endo_output_name_ES)
        endo_writer.SetInputData(endo_poly_ES)
        endo_writer.Write()

        epi_writer = vtk.vtkPolyDataWriter()
        epi_output_name_ES = f"{sub_dir}/landmark/myocardium_epi_ES.vtk"
        epi_writer.SetFileName(epi_output_name_ES)
        epi_writer.SetInputData(epi_poly_ES)
        epi_writer.Write()
        logger.info(f"{subject}: Wall-thickness evaluation at ES completed, myocardial contours are saved")

        # Add Bull's eye plot
        os.makedirs(f"{sub_dir}/visualization/myocardium", exist_ok=True)
        fig, ax = plot_bulls_eye(wall_thickness_ED[:16], title="Myocardium Thickness at ED", label="Thickness [mm]")
        fig.savefig(f"{sub_dir}/visualization/myocardium/thickness_ED.png")
        plt.close(fig)
        fig, ax = plot_bulls_eye(wall_thickness_ES[:16], title="Myocardium Thickness at ES", label="Thickness [mm]")
        fig.savefig(f"{sub_dir}/visualization/myocardium/thickness_ES.png")
        plt.close(fig)
        try:
            fig, ax = plot_bulls_eye(thickening[:16], title="Myocardium Thickening", label="Thickening [%]")
            fig.savefig(f"{sub_dir}/visualization/myocardium/thickening.png")
            plt.close(fig)
        except ValueError as e:
            logger.warning(f"{subject}: Error {e} when plotting bull's eye plot for thickening")

        # * Feature 2: Radius and thickness that incorporate distance to barycenter of LV cavity (with modification)
        # Refer Explainable cardiac pathology classification on cine MRI with motion characterization https://www.sciencedirect.com/science/article/pii/S1361841519300519
        # * This features utilizes the entire time series instead of just feature at ED, and can indicate abnormal contraction

        # BSA is required in calculation
        try:
            BSA_subject = query_BSA(subject)
            logger.info(f"{subject}: Implement disparity features")
            # we temporarily do not use radius and thickness
            radius_motion_disparity, thickness_motion_disparity, radius, thickness = evaluate_radius_thickness_disparity(
                seg_sa, nim_sa, BSA_subject
            )
            if radius_motion_disparity > 5:
                logger.warning(f"{subject}: Extremely high radius motion disparity detected, skipped")
            else:
                feature_dict.update({"Myo: Radius motion disparity": radius_motion_disparity})
            if thickness_motion_disparity > 5:
                logger.warning(f"{subject}: Extremely high thickness motion disparity detected, skipped")
            else:
                feature_dict.update({"Myo: Thickness motion disparity": thickness_motion_disparity})
        except (FileNotFoundError, IndexError):
            logger.warning(f"{subject}: BSA information not found, skip disparity features.")
        except ValueError as e:
            logger.warning(f"{subject}: Error {e} when determining disparity features.")

        # * Feature 3: Global fractal dimension (FD) for trabeculation
        # Ref Quantification of left ventricular trabeculae using fractal analysis. https://doi.org/10.1186/1532-429X-15-36
        # From the reference, there is no significant difference between global FD and maximal apical FD.
        global_fd, _, fds, img_roi, seg_roi = fractal_dimension(seg_sa_ED, nim_sa_ED, visualize_info=True)
        logger.info(f"{subject}: Fractal dimensions are {fds}")

        for i, fd in enumerate(fds):
            if np.isnan(fd):
                continue
            else:
                plt.figure(figsize=(12, 10))
                plt.subplot(1, 3, 1)
                plt.imshow(seg_sa_ED[:, :, i], cmap="gray")
                plt.title(f"Full Segmentation: Slice{i}")
                plt.subplot(1, 3, 2)
                plt.imshow(img_roi[:, :, i], cmap="gray")
                plt.title(f"Image for Thresholding: Slice{i}")
                plt.subplot(1, 3, 3)
                plt.imshow(seg_roi[:, :, i], cmap="gray")
                plt.title(f"Segmentation for Analysis: Slice{i}")
                plt.text(0.5, -0.15, f"Fractal Dimension is {fd:.2f}", ha="center", va="center", transform=plt.gca().transAxes)
                plt.savefig(f"{sub_dir}/visualization/myocardium/fd_{i}.png")
                plt.close()

        feature_dict.update({"Myo: Fractal dimension": global_fd})
        logger.info(f"{subject}: Fractal dimension calculation completed")

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "wall_thickness")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), index=False)
