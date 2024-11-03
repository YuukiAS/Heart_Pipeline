import os
import numpy as np
import nibabel as nib
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

        df_row = pd.DataFrame([feature_dict])
        logger.info(f"{subject}: Calculating aortic structure features")
        df_phenotype = single_image_inference(
            image_path=aorta_name, label_path=seg_aorta_name, data_output_path=os.path.join(data_dir, "aortic_mesh")
        )
        df_phenotype = df_phenotype.drop(columns=["pid"])
        df_row = pd.concat([df_row, df_phenotype.reset_index(drop=True)], axis=1)

        # * Add visualizations for above inference
        logger.info(f"{subject}: Generating visualizations for aortic structure")
        pv.start_xvfb()
        seg_aorta = sitk.ReadImage(seg_aorta_name)

        # morphological operations through blood and fill, extract largest continual region
        vtk_mesh = volume_to_smooth_mesh(itk_image=seg_aorta, dilate_num=1, erode_num=1)
        plotter1 = pv.Plotter(window_size=(870, 870), off_screen=True)
        plotter1.add_mesh(vtk_mesh, color="red", line_width=5)
        plotter1.show_bounds(
            grid="front", location="outer", all_edges=True, ticks="both", xtitle="X Axis", ytitle="Y Axis", ztitle="Z Axis"
        )
        plotter1.view_yz()
        plotter1.screenshot(os.path.join(sub_dir, "visualization", "aortic_structure", "aorta.png"))

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
        plotter2.screenshot(os.path.join(sub_dir, "visualization", "aortic_structure", "aorta_landmark.png"))

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
        plotter3.screenshot(os.path.join(sub_dir, "visualization", "aortic_structure", "aorta_lower_upper.png"))

        # Upper aorta: ascending aorta (arch) and descending arch in the descending aorta
        ascending_arch, descending_arch = split_aorta_arch(upper_aorta, analysis_idx=aorta.analysis_idx)
        plotter4 = pv.Plotter(window_size=(870, 870), off_screen=True)
        plotter4.add_mesh(ascending_arch, color="red", label="Ascending Arch")
        plotter4.add_mesh(descending_arch, color="orange", label="Descending Arch")
        plotter4.show_bounds(
            grid="front", location="outer", all_edges=True, ticks="both", xtitle="X Axis", ytitle="Y Axis", ztitle="Z Axis"
        )
        plotter4.add_legend()
        plotter4.view_yz()
        plotter4.screenshot(os.path.join(sub_dir, "visualization", "aortic_structure", "aorta_arch.png"))

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
        plotter5.screenshot(os.path.join(sub_dir, "visualization", "aortic_structure", "aorta_descending.png"))

        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "aortic_structure")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df = df[[col for col in df.columns if col != "eid"] + ["eid"]]  # move 'eid' to the last column
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), index=False)
