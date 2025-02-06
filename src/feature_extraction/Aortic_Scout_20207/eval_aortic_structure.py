"""
We directly make use of the code provided by the following paper:
Three-dimensional aortic geometry: Clinical correlates, prognostic value and genetic architecture https://doi.org/10.1101/2024.05.09.593413
The code utilizes the Vascular Modeling Toolkit (VMTK) http://www.vmtk.org/index.html to calculate aortic features.
"""

import os
import pandas as pd
import argparse
from tqdm import tqdm
import pyvista as pv
import SimpleITK as sitk
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
# from lib.Aortic_Scout_20207.data_operations.image_utils import voxel_to_mesh
from lib.Aortic_Scout_20207.train_operations.inference_2d_seg import single_image_inference
from lib.Aortic_Scout_20207.data_operations.phenotypes import (
    volume_to_smooth_mesh,
    Aorta,
    extract_lower_upper_aorta,
    split_aorta_arch,
    extract_descending_aorta,
)

import config
from utils.log_utils import setup_logging


logger = setup_logging("eval_aorta_structure")

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
        logger.info(f"Calculating aortic structure features for subject {subject}")
        sub_dir = os.path.join(data_dir, subject)

        aorta_name = os.path.join(sub_dir, "aortic_scout.nii.gz")
        seg_aorta_name = os.path.join(sub_dir, "seg_aortic_scout.nii.gz")

        if not os.path.exists(aorta_name):
            logger.error(f"Aorta structure file for {subject} does not exist")
            continue

        if not os.path.exists(seg_aorta_name):
            logger.error(f"Segmentation of aorta structure for {subject} does not exist")
            continue

        feature_dict = {
            "eid": subject,
        }

        # Ref Features definition: A Framework for Geometric Analysis of Vascular Structures: Application to Cerebral Aneurysms https://doi.org/10.1109/TMI.2009.2021652
        # Ref Region definition:  Age-Related Changes in Aortic Arch Geometry. https://doi.org/10.1016/j.jacc.2011.06.012

        logger.info(f"{subject}: Calculating aortic structure features")
        df_raw = single_image_inference(
            image_path=aorta_name, label_path=seg_aorta_name, data_output_path=os.path.join(data_dir, "aortic_mesh")
        )
        df_raw = df_raw.drop(columns=["pid"])

        # df_row = pd.concat([df_row, df_raw.reset_index(drop=True)], axis=1)

        # * Feature1: Features extracted using centerline, which measures the global aortic structure

        feature_dict.update(
            {
                "Aortic Structure-Centerline: Maxium Diameter [mm]": df_raw["max_aorta_diameter"].values[0] * 10,
                "Aortic Structure-Centerline: Mean Diameter [mm]": df_raw["mean_aorta_diameter"].values[0] * 10,
                "Aortic Structure-Centerline: Min Diameter [mm]": df_raw["min_aorta_diameter"].values[0] * 10,
                "Aortic Structure-Centerline: Length [mm]": df_raw["overall_length"].values[0] * 10,
                "Aortic Structure-Centerline: Linear Distance [mm]": df_raw["overall_linear_length"].values[0] * 10,
                # define Tortuosity: The ratio of the length of and linear distance - 1. Straight line has zero tortuosity
                # We already subtract 1 in df_raw
                "Aortic Structure-Centerline: Tortuosity Index [%]": df_raw["overall_tortuosity"].values[0] * 100,
                # define Curvature: The inverse of the radius of the local osculating circle (circle that approximates the curve)
                "Aortic Structure-Centerline: Curvature [1/mm]": df_raw["overall_curvature"].values[0] / 10,
                # define Torsion: The amount by which osculating plane (identified by osculating circle) rotates along the line
                "Aortic Structure-Centerline: Torsion [1/mm]": df_raw["overall_torsion"].values[0] / 10,
            }
        )

        # * Feature2: Features extracted using ascending arch

        feature_dict.update(
            {
                "Aortic Structure-Ascending Arch: Maxium Diameter [mm]": df_raw["max_ascending_arch"].values[0] * 10,
                "Aortic Structure-Ascending Arch: Mean Diameter [mm]": df_raw["mean_ascending_arch"].values[0] * 10,
                "Aortic Structure-Ascending Arch: Min Diameter [mm]": df_raw["min_ascending_arch"].values[0] * 10,
                "Aortic Structure-Ascending Arch: Length [mm]": df_raw["ascending_arch_length"].values[0] * 10,
                "Aortic Structure-Ascending Arch: Linear Distance [mm]": df_raw["linear_ascending_arch_length"].values[0] * 10,
                "Aortic Structure-Ascending Arch: Tortuosity Index [%]": df_raw["ascending_arch_tortuosity"].values[0] * 100,
                "Aortic Structure-Ascending Arch: Curvature [1/mm]": df_raw["ascending_arch_curvature"].values[0] / 10,
                "Aortic Structure-Ascending Arch: Torsion [1/mm]": df_raw["ascending_arch_torsion"].values[0] / 10,
            }
        )

        # * Feature3: Features extracted using descending arch

        feature_dict.update(
            {
                "Aortic Structure-Descending Arch: Maxium Diameter [mm]": df_raw["max_descending_arch"].values[0] * 10,
                "Aortic Structure-Descending Arch: Mean Diameter [mm]": df_raw["mean_descending_arch"].values[0] * 10,
                "Aortic Structure-Descending Arch: Min Diameter [mm]": df_raw["min_descending_arch"].values[0] * 10,
                "Aortic Structure-Descending Arch: Length [mm]": df_raw["descending_arch_length"].values[0] * 10,
                "Aortic Structure-Descending Arch: Linear Distance [mm]": df_raw["linear_descending_arch_length"].values[0] * 10,
                "Aortic Structure-Descending Arch: Tortuosity Index [%]": df_raw["descending_arch_tortuosity"].values[0] * 100,
                "Aortic Structure-Descending Arch: Curvature [1/mm]": df_raw["descending_arch_curvature"].values[0] / 10,
                "Aortic Structure-Descending Arch: Torsion [1/mm]": df_raw["descending_arch_torsion"].values[0] / 10,
            }
        )

        # * Feature4: Features extracted using descending aorta (descending arch + lower aorta)

        feature_dict.update(
            {
                "Aortic Structure-Descending Aorta: Maximum Diameter [mm]": df_raw["max_descending_aorta"].values[0] * 10,
                "Aortic Structure-Descending Aorta: Mean Diameter [mm]": df_raw["mean_descending_aorta"].values[0] * 10,
                "Aortic Structure-Descending Aorta: Min Diameter [mm]": df_raw["min_descending_aorta"].values[0] * 10,
                "Aortic Structure-Descending Aorta: Length [mm]": df_raw["descending_aorta_length"].values[0] * 10,
                "Aortic Structure-Descending Aorta: Linear Distance [mm]": df_raw["linear_descending_aorta_length"].values[0]
                * 10,
                "Aortic Structure-Descending Aorta: Tortuosity Index [%]": df_raw["descending_aorta_tortuosity"].values[0] * 100,
                "Aortic Structure-Descending Aorta: Curvature [1/mm]": df_raw["descending_aorta_curvature"].values[0] / 10,
                "Aortic Structure-Descending Aorta: Torsion [1/mm]": df_raw["descending_aorta_torsion"].values[0] / 10,
            }
        )

        # * Feature5: Features extracted using arch/upper aorta (ascending arch + descending arch)

        feature_dict.update(
            {
                "Aortic Structure-Arch: Maximum Diameter [mm]": df_raw["max_arch_diameter"].values[0] * 10,
                "Aortic Structure-Arch: Mean Diameter [mm]": df_raw["mean_arch_diameter"].values[0] * 10,
                "Aortic Structure-Arch: Min Diameter [mm]": df_raw["min_arch_diameter"].values[0] * 10,
                "Aortic Structure-Arch: Length [mm]": df_raw["arch_length"].values[0] * 10,
                "Aortic Structure-Arch: Arch Width [mm]": df_raw["arch_width"].values[0]
                * 10,  # can also obtained from linear_arch_length
                "Aortic Structure-Arch: Arch Height [mm]": df_raw["arch_height"].values[0] * 10,
                "Aortic Structure-Arch: Tortuosity Index [%]": df_raw["arch_tortuosity"].values[0] * 100,
                "Aortic Structure-Arch: Curvature [1/mm]": df_raw["arch_curvature"].values[0] / 10,
                "Aortic Structure-Arch: Torsion [1/mm]": df_raw["arch_torsion"].values[0] / 10,
            }
        )

        # * Feature6: Features extracted using lower aorta (descending aorta - descending arch)
        # Between the proximal and distal descending aortic acquisition planes

        feature_dict.update(
            {
                "Aortic Structure-Lower Aorta: Maximum Diameter [mm]": df_raw["max_lower_descending_diameter"].values[0] * 10,
                "Aortic Structure-Lower Aorta: Mean Diameter [mm]": df_raw["mean_lower_descending_diameter"].values[0] * 10,
                "Aortic Structure-Lower Aorta: Min Diameter [mm]": df_raw["min_lower_descending_diameter"].values[0] * 10,
                "Aortic Structure-Lower Aorta: Length [mm]": df_raw["lower_descending_length"].values[0] * 10,
                "Aortic Structure-Lower Aorta: Linear Distance [mm]": df_raw["linear_lower_descending_length"].values[0] * 10,
                "Aortic Structure-Lower Aorta: Tortuosity Index [%]": df_raw["lower_descending_tortuosity"].values[0] * 100,
                "Aortic Structure-Lower Aorta: Curvature [1/mm]": df_raw["lower_descending_curvature"].values[0] / 10,
                "Aortic Structure-Lower Aorta: Torsion [1/mm]": df_raw["lower_descending_torsion"].values[0] / 10,
            }
        )

        # * Add visualizations for the inference above
        logger.info(f"{subject}: Generating visualizations for aortic structure")
        pv.start_xvfb()
        seg_aorta = sitk.ReadImage(seg_aorta_name)

        os.makedirs(os.path.join(sub_dir, "visualization", "aorta"), exist_ok=True)
        # morphological operations through blood and fill, extract largest continual region
        vtk_mesh = volume_to_smooth_mesh(itk_image=seg_aorta, dilate_num=1, erode_num=1)
        plotter1 = pv.Plotter(window_size=(870, 870), off_screen=True)
        plotter1.add_mesh(vtk_mesh, color="red", line_width=5)
        plotter1.show_bounds(
            grid="front", location="outer", all_edges=True, ticks="both", xtitle="X Axis", ytitle="Y Axis", ztitle="Z Axis"
        )
        plotter1.view_yz()
        plotter1.screenshot(os.path.join(sub_dir, "visualization", "aorta", "aorta_structure.png"))

        aorta = Aorta(itk_image=seg_aorta, path=aorta_name, output_dir=os.path.join(data_dir, "aortic_mesh"))
        aorta.process_itk_data()

        centerline_point_arr = []
        centerline_points = aorta.centerline.GetPoints()
        for idx in range(centerline_points.GetNumberOfPoints()):
            curr_point = centerline_points.GetPoint(idx)
            # iterate over x, y, z: i will be x, y, z in the loop
            for i in range(len(curr_point)):
                point = curr_point[i]
                if point <= 0:
                    # define proximal end of the ascending aorta
                    analysis_point = curr_point  # this will be the first element of returned point list array
                    break
        # define distal end of the descending aorta
        end_point = aorta.centerline_point_arr[-1]

        plotter2 = pv.Plotter(window_size=(870, 870), off_screen=True)
        plotter2.add_mesh(vtk_mesh, color="red")
        sphere_radius = 12
        blue_marker = pv.Sphere(radius=sphere_radius, center=list(analysis_point))
        purple_marker = pv.Sphere(radius=sphere_radius, center=end_point)
        plotter2.add_mesh(blue_marker, color="blue")
        plotter2.add_mesh(purple_marker, color="purple")
        plotter2.show_bounds(
            grid="front", location="outer", all_edges=True, ticks="both", xtitle="X Axis", ytitle="Y Axis", ztitle="Z Axis"
        )
        plotter2.view_yz()
        plotter2.screenshot(os.path.join(sub_dir, "visualization", "aorta", "aorta_structure_landmark.png"))

        lower_aorta, upper_aorta = extract_lower_upper_aorta(
            aorta.centerline, point_list=aorta.centerline_point_arr, analysis_idx=aorta.analysis_idx
        )
        plotter3 = pv.Plotter(window_size=(870, 870), off_screen=True)
        plotter3.add_mesh(lower_aorta, color="blue", label="Lower Aorta")
        plotter3.add_mesh(upper_aorta, color="green", label="Upper Aorta")
        plotter3.show_bounds(
            grid="front", location="outer", all_edges=True, ticks="both", xtitle="X Axis", ytitle="Y Axis", ztitle="Z Axis"
        )
        plotter3.view_yz()
        plotter3.add_legend()
        plotter3.screenshot(os.path.join(sub_dir, "visualization", "aorta", "aorta_structure_lower_upper.png"))

        ascending_arch, descending_arch = split_aorta_arch(upper_aorta, analysis_idx=aorta.analysis_idx)
        plotter4 = pv.Plotter(window_size=(870, 870), off_screen=True)
        plotter4.add_mesh(ascending_arch, color="red", label="Ascending Arch")
        plotter4.add_mesh(descending_arch, color="orange", label="Descending Arch")
        plotter4.show_bounds(
            grid="front", location="outer", all_edges=True, ticks="both", xtitle="X Axis", ytitle="Y Axis", ztitle="Z Axis"
        )
        plotter4.add_legend()
        plotter4.view_yz()
        plotter4.screenshot(os.path.join(sub_dir, "visualization", "aorta", "aorta_structure_arch.png"))

        descending_aorta = extract_descending_aorta(
            aorta.centerline, point_list=aorta.centerline_point_arr, analysis_idx=aorta.analysis_idx
        )
        plotter5 = pv.Plotter(window_size=(870, 870), off_screen=True)
        plotter5.add_mesh(descending_aorta, color="blue", label="Descending Aorta")
        plotter5.show_bounds(
            grid="front", location="outer", all_edges=True, ticks="both", xtitle="X Axis", ytitle="Y Axis", ztitle="Z Axis"
        )
        plotter5.add_legend()
        plotter5.view_yz()
        plotter5.screenshot(os.path.join(sub_dir, "visualization", "aorta", "aorta_structure_descending.png"))

        df_row = pd.DataFrame([feature_dict])
        df_row = df_row.round(5)
        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "aortic_structure")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df = df[[col for col in df.columns if col != "eid"] + ["eid"]]  # move 'eid' to the last column
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), index=False)
