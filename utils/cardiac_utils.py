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
# ==============================================================================
import os
import math
import numpy as np
import nibabel as nib
import cv2
from collections import OrderedDict
import vtk
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple
from vtk.util import numpy_support
import porespy as ps
from scipy import interpolate
from scipy.stats import linregress
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_dilation, binary_erosion

# import skimage.measure
from .image_utils import (
    get_largest_cc,
    remove_small_cc,
    padding,
    split_volume,
    split_sequence,
    np_categorical_dice,
    make_sequence,
    auto_crop_image,
)

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from utils.log_utils import setup_logging

logger = setup_logging("cardiac-utils")


def approximate_contour(contour, factor=4, smooth=0.05, periodic=False):
    """Approximate a contour through upsampling and smoothing.

    contour: input contour
    factor: upsampling factor for the contour
    smooth: smoothing factor for controling the number of spline knots.
            Number of knots will be increased until the smoothing
            condition is satisfied:
            sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s
            which means the larger s is, the fewer knots will be used,
            thus the contour will be smoother but also deviating more
            from the input contour.
    periodic: set to True if this is a closed contour, otherwise False.

    return the upsampled and smoothed contour
    """
    # The input contour
    N = len(contour)
    dt = 1.0 / N
    t = np.arange(N) * dt
    x = contour[:, 0]
    y = contour[:, 1]

    # Pad the contour before approximation to avoid underestimating
    # the values at the end points
    r = int(0.5 * N)
    t_pad = np.concatenate((np.arange(-r, 0) * dt, t, 1 + np.arange(0, r) * dt))
    if periodic:
        x_pad = np.concatenate((x[-r:], x, x[:r]))
        y_pad = np.concatenate((y[-r:], y, y[:r]))
    else:
        x_pad = np.concatenate((np.repeat(x[0], repeats=r), x, np.repeat(x[-1], repeats=r)))
        y_pad = np.concatenate((np.repeat(y[0], repeats=r), y, np.repeat(y[-1], repeats=r)))

    # Fit the contour with splines with a smoothness constraint
    fx = interpolate.UnivariateSpline(t_pad, x_pad, s=smooth * len(t_pad))
    fy = interpolate.UnivariateSpline(t_pad, y_pad, s=smooth * len(t_pad))

    # Evaluate the new contour
    N2 = N * factor
    dt2 = 1.0 / N2
    t2 = np.arange(N2) * dt2
    x2, y2 = fx(t2), fy(t2)
    contour2 = np.stack((x2, y2), axis=1)
    return contour2


def determine_aha_coordinate_system(seg_sa, affine_sa):
    """
    Determine the AHA coordinate system using the mid-cavity slice
    of the short-axis image segmentation.
    """

    if seg_sa.ndim != 3:
        raise ValueError("The input segmentation should be 3D.")

    # Label class in the segmentation
    labels = {"BG": 0, "LV": 1, "Myo": 2, "RV": 3}

    # Find the mid-cavity slice
    _, _, cz = [np.mean(x) for x in np.nonzero(seg_sa == labels["LV"])]  # only uses the z coordinate
    z = int(round(cz))
    seg_z = seg_sa[:, :, z]

    endo = (seg_z == labels["LV"]).astype(np.uint8)
    endo = get_largest_cc(endo).astype(np.uint8)
    myo = (seg_z == labels["Myo"]).astype(np.uint8)
    myo = remove_small_cc(myo).astype(np.uint8)
    epi = (endo | myo).astype(np.uint8)
    epi = get_largest_cc(epi).astype(np.uint8)
    rv = (seg_z == labels["RV"]).astype(np.uint8)
    rv = get_largest_cc(rv).astype(np.uint8)

    # Extract epicardial contour
    contours, _ = cv2.findContours(cv2.inRange(epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    epi_contour = contours[0][:, 0, :]

    # define Septum is the intersection between LV and RV
    septum = []
    dilate_iter = 1
    while len(septum) == 0:
        # Dilate the RV till it intersects with LV epicardium.
        # Normally, this is fulfilled after just one iteration.
        rv_dilate = cv2.dilate(rv, np.ones((3, 3), dtype=np.uint8), iterations=dilate_iter)
        if dilate_iter > 10:
            raise ValueError("Dilate_iter reaches ceiling, cannot find the septum!")
        dilate_iter += 1
        for y, x in epi_contour:
            if rv_dilate[x, y] == 1:
                septum += [[x, y]]

    # The middle of the septum
    mx, my = septum[int(round(0.5 * len(septum)))]
    point_septum = np.dot(affine_sa, np.array([mx, my, z, 1]))[:3]

    # Find the centre of the LV cavity
    cx, cy = [np.mean(x) for x in np.nonzero(endo)]
    point_cavity = np.dot(affine_sa, np.array([cx, cy, z, 1]))[:3]

    # Determine the AHA coordinate system
    axis = {}
    axis["lv_to_sep"] = point_septum - point_cavity  # distance from the cavity centre to the septum
    axis["lv_to_sep"] /= np.linalg.norm(axis["lv_to_sep"])
    axis["apex_to_base"] = np.copy(affine_sa[:3, 2])  # distance from the apex to the base
    axis["apex_to_base"] /= np.linalg.norm(axis["apex_to_base"])
    if axis["apex_to_base"][2] < 0:  # make sure z-axis is positive
        axis["apex_to_base"] *= -1
    axis["inf_to_ant"] = np.cross(axis["apex_to_base"], axis["lv_to_sep"])  # from inferior wall to anterior wall
    return axis


def determine_aha_part(seg_sa, affine_sa, three_slices=False):
    """Determine the AHA part for each slice."""
    # Label class in the segmentation
    labels = {"BG": 0, "LV": 1, "Myo": 2, "RV": 3}

    # Sort the z-axis positions of the slices with both endo and epicardium
    # segmentations
    X, Y, Z = seg_sa.shape[:3]
    z_pos = []
    for z in range(Z):
        seg_z = seg_sa[:, :, z]
        endo = (seg_z == labels["LV"]).astype(np.uint8)  # doesn't include myocardium
        myo = (seg_z == labels["Myo"]).astype(np.uint8)
        pixel_thres = 10
        if (np.sum(endo) < pixel_thres) or (np.sum(myo) < pixel_thres):
            continue
        z_pos += [(z, np.dot(affine_sa, np.array([X / 2.0, Y / 2.0, z, 1]))[2])]
    z_pos = sorted(z_pos, key=lambda x: -x[1])

    # Divide the slices into three parts: basal, mid-cavity and apical
    n_slice = len(z_pos)
    part_z = {}
    if three_slices:
        # Select three slices (basal, mid and apical) for strain analysis, inspired by:
        #
        # [1] Robin J. Taylor, et al. Myocardial strain measurement with
        # feature-tracking cardiovascular magnetic resonance: normal values.
        # European Heart Journal - Cardiovascular Imaging, (2015) 16, 871-881.
        #
        # [2] A. Schuster, et al. Cardiovascular magnetic resonance feature-
        # tracking assessment of myocardial mechanics: Intervendor agreement
        # and considerations regarding reproducibility. Clinical Radiology
        # 70 (2015), 989-998.

        # Use the slice at 25% location from base to apex.
        # Avoid using the first one or two basal slices, as the myocardium
        # will move out of plane at ES due to longitudinal motion, which will
        # be a problem for 2D in-plane motion tracking.
        z = int(round((n_slice - 1) * 0.25))
        part_z[z_pos[z][0]] = "basal"

        # Use the central slice.
        z = int(round((n_slice - 1) * 0.5))
        part_z[z_pos[z][0]] = "mid"

        # Use the slice at 75% location from base to apex.
        # In the most apical slices, the myocardium looks blurry and
        # may not be suitable for motion tracking.
        z = int(round((n_slice - 1) * 0.75))
        part_z[z_pos[z][0]] = "apical"
    else:
        # Use all the slices
        i1 = int(math.ceil(n_slice / 3.0))
        i2 = int(math.ceil(2 * n_slice / 3.0))
        i3 = n_slice

        for i in range(0, i1):
            part_z[z_pos[i][0]] = "basal"

        for i in range(i1, i2):
            part_z[z_pos[i][0]] = "mid"

        for i in range(i2, i3):
            part_z[z_pos[i][0]] = "apical"
    return part_z


def determine_aha_segment_id(point, lv_centre, aha_axis, part):
    """Determine the AHA segment ID given a point,
    the LV cavity center and the coordinate system.
    """
    d = point - lv_centre
    x = np.dot(d, aha_axis["inf_to_ant"])
    y = np.dot(d, aha_axis["lv_to_sep"])
    deg = math.degrees(math.atan2(y, x))
    seg_id = 0

    if part == "basal":
        if (deg >= -30) and (deg < 30):
            seg_id = 1
        elif (deg >= 30) and (deg < 90):
            seg_id = 2
        elif (deg >= 90) and (deg < 150):
            seg_id = 3
        elif (deg >= 150) or (deg < -150):
            seg_id = 4
        elif (deg >= -150) and (deg < -90):
            seg_id = 5
        elif (deg >= -90) and (deg < -30):
            seg_id = 6
        else:
            logger.error("Error: wrong degree {0}!".format(deg))
            exit(1)
    elif part == "mid":
        if (deg >= -30) and (deg < 30):
            seg_id = 7
        elif (deg >= 30) and (deg < 90):
            seg_id = 8
        elif (deg >= 90) and (deg < 150):
            seg_id = 9
        elif (deg >= 150) or (deg < -150):
            seg_id = 10
        elif (deg >= -150) and (deg < -90):
            seg_id = 11
        elif (deg >= -90) and (deg < -30):
            seg_id = 12
        else:
            logger.error("Error: wrong degree {0}!".format(deg))
            exit(1)
    elif part == "apical":
        if (deg >= -45) and (deg < 45):
            seg_id = 13
        elif (deg >= 45) and (deg < 135):
            seg_id = 14
        elif (deg >= 135) or (deg < -135):
            seg_id = 15
        elif (deg >= -135) and (deg < -45):
            seg_id = 16
        else:
            logger.error("Error: wrong degree {0}!".format(deg))
            exit(1)
    elif part == "apex":
        seg_id = 17
    else:
        logger.error("Error: unknown part {0}!".format(part))
        exit(1)
    return seg_id


def evaluate_wall_thickness(seg, nim_sa, part=None, save_epi_contour=False):
    """Evaluate myocardial wall thickness."""

    if seg.ndim != 3:
        raise ValueError("The input segmentation should be 3D.")

    affine = nim_sa.affine
    Z = nim_sa.header["dim"][3]

    # Label class in the segmentation
    labels = {"BG": 0, "LV": 1, "Myo": 2, "RV": 3}

    # Determine the AHA coordinate system using the mid-cavity slice
    aha_axis = determine_aha_coordinate_system(seg, affine)

    # Determine the AHA part (basal, mid, apical) of each slice if part is not provided
    part_z = {}
    if not part:
        part_z = determine_aha_part(seg, affine)
    else:
        part_z = {z: part for z in range(Z)}

    # Construct the points set to represent the endocardial contours
    endo_points = vtk.vtkPoints()
    thickness = vtk.vtkDoubleArray()
    thickness.SetName("Thickness")
    points_aha = vtk.vtkIntArray()
    points_aha.SetName("Segment ID")
    point_id = 0
    lines = vtk.vtkCellArray()

    # Save epicardial contour for debug and demonstration purposes
    if save_epi_contour:
        epi_points = vtk.vtkPoints()
        points_epi_aha = vtk.vtkIntArray()
        points_epi_aha.SetName("Segment ID")
        point_epi_id = 0
        lines_epi = vtk.vtkCellArray()

    # For each slice
    for z in range(Z):
        # Check whether there is endocardial segmentation and it is not too small,
        # e.g. a single pixel, which either means the structure is missing or
        # causes problem in contour interpolation.
        seg_z = seg[:, :, z]
        endo = (seg_z == labels["LV"]).astype(np.uint8)
        endo = get_largest_cc(endo).astype(np.uint8)
        myo = (seg_z == labels["Myo"]).astype(np.uint8)
        myo = remove_small_cc(myo).astype(np.uint8)
        epi = (endo | myo).astype(np.uint8)
        epi = get_largest_cc(epi).astype(np.uint8)
        pixel_thres = 10
        if (np.sum(endo) < pixel_thres) or (np.sum(myo) < pixel_thres):
            continue

        # Calculate the centre of the LV cavity
        # Get the largest component in case we have a bad segmentation
        cx, cy = [np.mean(x) for x in np.nonzero(endo)]
        lv_centre = np.dot(affine, np.array([cx, cy, z, 1]))[:3]

        # Extract endocardial contour
        # Note: cv2 considers an input image as a Y x X array, which is different
        # from nibabel which assumes a X x Y array.
        contours, _ = cv2.findContours(cv2.inRange(endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        endo_contour = contours[0][:, 0, :]

        # Extract epicardial contour
        contours, _ = cv2.findContours(cv2.inRange(epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        epi_contour = contours[0][:, 0, :]

        # Smooth the contours
        endo_contour = approximate_contour(endo_contour, periodic=True)
        epi_contour = approximate_contour(epi_contour, periodic=True)

        # A polydata representation of the epicardial contour
        epi_points_z = vtk.vtkPoints()
        for y, x in epi_contour:
            p = np.dot(affine, np.array([x, y, z, 1]))[:3]
            epi_points_z.InsertNextPoint(p)
        epi_poly_z = vtk.vtkPolyData()
        epi_poly_z.SetPoints(epi_points_z)

        # Point locator for the epicardial contour
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(epi_poly_z)
        locator.BuildLocator()

        # * For each point on endocardium, find the closest point on epicardium
        N = endo_contour.shape[0]
        for i in range(N):
            y, x = endo_contour[i]

            # The world coordinate of this point
            p = np.dot(affine, np.array([x, y, z, 1]))[:3]
            endo_points.InsertNextPoint(p)

            # The closest epicardial point
            q = np.array(epi_points_z.GetPoint(locator.FindClosestPoint(p)))

            # The distance from endo to epi
            dist_pq = np.linalg.norm(q - p)

            # Add the point data
            thickness.InsertNextTuple1(dist_pq)
            # * Determine the AHA segment ID based on part (basal, mid, apical)
            seg_id = determine_aha_segment_id(p, lv_centre, aha_axis, part_z[z])
            points_aha.InsertNextTuple1(seg_id)

            # Record the first point of the current contour
            if i == 0:
                contour_start_id = point_id

            # Add the line
            if i == (N - 1):
                lines.InsertNextCell(2, [point_id, contour_start_id])
            else:
                lines.InsertNextCell(2, [point_id, point_id + 1])

            # Increment the point index
            point_id += 1

        if save_epi_contour:
            # For each point on epicardium
            N = epi_contour.shape[0]
            for i in range(N):
                y, x = epi_contour[i]

                # The world coordinate of this point
                p = np.dot(affine, np.array([x, y, z, 1]))[:3]
                epi_points.InsertNextPoint(p)
                seg_id = determine_aha_segment_id(p, lv_centre, aha_axis, part_z[z])
                points_epi_aha.InsertNextTuple1(seg_id)

                # Record the first point of the current contour
                if i == 0:
                    contour_start_id = point_epi_id

                # Add the line
                if i == (N - 1):
                    lines_epi.InsertNextCell(2, [point_epi_id, contour_start_id])
                else:
                    lines_epi.InsertNextCell(2, [point_epi_id, point_epi_id + 1])

                # Increment the point index
                point_epi_id += 1

    # Save to a vtk file
    endo_poly = vtk.vtkPolyData()
    endo_poly.SetPoints(endo_points)
    endo_poly.GetPointData().AddArray(thickness)
    endo_poly.GetPointData().AddArray(points_aha)
    endo_poly.SetLines(lines)

    # writer = vtk.vtkPolyDataWriter()
    # output_name = "{0}.vtk".format(output_name_stem)
    # writer.SetFileName(output_name)
    # writer.SetInputData(endo_poly)
    # writer.Write()

    if save_epi_contour:
        epi_poly = vtk.vtkPolyData()
        epi_poly.SetPoints(epi_points)
        epi_poly.GetPointData().AddArray(points_epi_aha)
        epi_poly.SetLines(lines_epi)

        # writer = vtk.vtkPolyDataWriter()
        # output_name = "{0}_epi.vtk".format(output_name_stem)
        # writer.SetFileName(output_name)
        # writer.SetInputData(epi_poly)
        # writer.Write()
    else:
        epi_poly = None

    # Evaluate the wall thickness per AHA segment and save to a csv file
    table_thickness = np.zeros(17)
    table_thickness_max = np.zeros(17)
    np_thickness = numpy_support.vtk_to_numpy(thickness).astype(np.float32)
    np_points_aha = numpy_support.vtk_to_numpy(points_aha).astype(np.int8)

    for i in range(16):
        table_thickness[i] = np.mean(np_thickness[np_points_aha == (i + 1)])
        table_thickness_max[i] = np.max(np_thickness[np_points_aha == (i + 1)])
    table_thickness[-1] = np.mean(np_thickness)
    table_thickness_max[-1] = np.max(np_thickness)

    index = [str(x) for x in np.arange(1, 17)] + ["Global"]
    # df = pd.DataFrame(table_thickness, index=index, columns=["Thickness"])
    # df.to_csv("{0}.csv".format(output_name_stem))

    # df = pd.DataFrame(table_thickness_max, index=index, columns=["Thickness_Max"])
    # df.to_csv("{0}_max.csv".format(output_name_stem))

    return endo_poly, epi_poly, index, table_thickness, table_thickness_max


def extract_sa_myocardial_contour(seg_name, contour_name_stem1, contour_name_stem2, part=None, three_slices=False):
    """
    Extract the myocardial contours, including both endo and epicardial contours.
    Determine the AHA segment ID for all the contour points.

    By default, part is None. This function will automatically determine the part
    for each slice (basal, mid or apical).
    If part is given, this function will use the given part for the image slice.
    """
    # Read the segmentation image
    nim = nib.load(seg_name)
    _, _, Z = nim.header["dim"][1:4]
    affine = nim.affine
    seg = nim.get_fdata()

    # Label class in the segmentation
    labels = {"BG": 0, "LV": 1, "Myo": 2, "RV": 3}

    # Determine the AHA coordinate system using the mid-cavity slice
    aha_axis = determine_aha_coordinate_system(seg, affine)

    # Determine the AHA part of each slice (basal, mid, apical)
    part_z = {}
    if not part:
        part_z = determine_aha_part(seg, affine, three_slices=three_slices)
    else:
        part_z = {z: part for z in range(Z)}

    if three_slices:
        logger.info(f"We will analyze three slices: {part_z}")

    # For each slice
    for z in range(Z):
        # Check whether there is the endocardial segmentation
        seg_z = seg[:, :, z]
        endo = (seg_z == labels["LV"]).astype(np.uint8)
        endo = get_largest_cc(endo).astype(np.uint8)
        myo = (seg_z == labels["Myo"]).astype(np.uint8)
        myo = remove_small_cc(myo).astype(np.uint8)
        epi = (endo | myo).astype(np.uint8)
        epi = get_largest_cc(epi).astype(np.uint8)
        pixel_thres = 10
        if (np.sum(endo) < pixel_thres) or (np.sum(myo) < pixel_thres):
            continue

        # Check whether this slice is going to be analysed
        if z not in part_z.keys():
            continue

        # * Construct the points set and data arrays to represent both endo and epicardial contours
        points = vtk.vtkPoints()  # define The real world coordinates of the points

        points_radial = vtk.vtkFloatArray()  # define Radial direction from cavity center to the point (normalized)
        points_radial.SetName("Direction_Radial")
        points_radial.SetNumberOfComponents(3)

        points_label = vtk.vtkIntArray()  # define Label of the point (1 = endo, 2 = epi)
        points_label.SetName("Label")

        points_aha = vtk.vtkIntArray()  # define AHA segment ID 1~16
        points_aha.SetName("Segment ID")

        point_id = 0

        lines = vtk.vtkCellArray()  # define Two consecutive points

        lines_aha = vtk.vtkIntArray()
        lines_aha.SetName("Segment ID")

        lines_dir = vtk.vtkIntArray()
        lines_dir.SetName("Direction ID")

        # Calculate the centre of the LV cavity
        # Get the largest component in case we have a bad segmentation
        cx, cy = [np.mean(x) for x in np.nonzero(endo)]
        lv_centre = np.dot(affine, np.array([cx, cy, z, 1]))[:3]

        # * Extract epicardial contour -------------------------------------------
        contours, _ = cv2.findContours(cv2.inRange(epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        epi_contour = contours[0][:, 0, :]
        epi_contour = approximate_contour(epi_contour, periodic=True)

        N = epi_contour.shape[0]
        for i in range(N):
            y, x = epi_contour[i]

            # The world coordinate of this point
            p = np.dot(affine, np.array([x, y, z, 1]))[:3]
            points.InsertNextPoint(p[0], p[1], p[2])

            # The radial direction from the cavity centre to this point
            d_rad = p - lv_centre
            d_rad = d_rad / np.linalg.norm(d_rad)
            points_radial.InsertNextTuple3(d_rad[0], d_rad[1], d_rad[2])

            # Record the type of the point (1 = endo, 2 = epi)
            points_label.InsertNextTuple1(2)

            # Record the AHA segment ID
            seg_id = determine_aha_segment_id(p, lv_centre, aha_axis, part_z[z])
            points_aha.InsertNextTuple1(seg_id)

            # Record the first point of the current contour
            if i == 0:
                contour_start_id = point_id

            # * Add the circumferential line for epicardial contour
            if i == (N - 1):
                lines.InsertNextCell(2, [point_id, contour_start_id])
            else:
                lines.InsertNextCell(2, [point_id, point_id + 1])

            lines_aha.InsertNextTuple1(seg_id)
            # Line direction (1 = radial, 2 = circumferential, 3 = longitudinal)
            lines_dir.InsertNextTuple1(2)

            # Increment the point index
            point_id += 1

        # Point locator
        epi_points = vtk.vtkPoints()
        epi_points.DeepCopy(points)
        epi_poly = vtk.vtkPolyData()
        epi_poly.SetPoints(epi_points)
        locator = vtk.vtkPointLocator()  # define locator can be used to find the closest point
        locator.SetDataSet(epi_poly)
        locator.BuildLocator()

        # * Extract endocardial contour  -------------------------------------------
        # Note: cv2 considers an input image as a Y x X array, which is different
        # from nibabel which assumes a X x Y array.
        contours, _ = cv2.findContours(cv2.inRange(endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        endo_contour = contours[0][:, 0, :]
        endo_contour = approximate_contour(endo_contour, periodic=True)

        N = endo_contour.shape[0]
        for i in range(N):
            y, x = endo_contour[i]

            # The world coordinate of this point
            p = np.dot(affine, np.array([x, y, z, 1]))[:3]
            points.InsertNextPoint(p[0], p[1], p[2])

            # The radial direction from the cavity centre to this point
            d_rad = p - lv_centre
            d_rad = d_rad / np.linalg.norm(d_rad)
            points_radial.InsertNextTuple3(d_rad[0], d_rad[1], d_rad[2])

            # Record the type of the point (1 = endo, 2 = epi)
            points_label.InsertNextTuple1(1)

            # Record the AHA segment ID
            seg_id = determine_aha_segment_id(p, lv_centre, aha_axis, part_z[z])
            points_aha.InsertNextTuple1(seg_id)

            # Record the first point of the current contour
            if i == 0:
                contour_start_id = point_id

            # * Add the circumferential line for endocardial contour
            if i == (N - 1):
                lines.InsertNextCell(2, [point_id, contour_start_id])
            else:
                lines.InsertNextCell(2, [point_id, point_id + 1])

            lines_aha.InsertNextTuple1(seg_id)
            # Line direction (1 = radial, 2 = circumferential, 3 = longitudinal)
            lines_dir.InsertNextTuple1(2)

            # Add the radial line for every few points
            n_radial = 36
            M = int(round(N / float(n_radial)))
            if i % M == 0:
                # The closest epicardial points
                ids = vtk.vtkIdList()
                n_ids = 10
                # * Find the n_ids closest points on the epicardial contour to p on endocardial contour
                locator.FindClosestNPoints(n_ids, p, ids)

                # The point that aligns with the radial direction
                val = []
                for j in range(n_ids):
                    q = epi_points.GetPoint(ids.GetId(j))
                    d = (q - lv_centre) / np.linalg.norm(q - lv_centre)
                    val += [np.dot(d, d_rad)]
                val = np.array(val)
                epi_point_id = ids.GetId(np.argmax(val))

                # Add the radial line (endocardial and epicardial)
                lines.InsertNextCell(2, [point_id, epi_point_id])
                lines_aha.InsertNextTuple1(seg_id)
                # Line direction (1 = radial, 2 = circumferential, 3 = longitudinal)
                lines_dir.InsertNextTuple1(1)

            # Increment the point index
            point_id += 1

        # Save the contour for each slice
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.GetPointData().AddArray(points_label)
        poly.GetPointData().AddArray(points_aha)
        poly.GetPointData().AddArray(points_radial)
        poly.SetLines(lines)
        poly.GetCellData().AddArray(lines_aha)
        poly.GetCellData().AddArray(lines_dir)

        writer = vtk.vtkPolyDataWriter()
        contour_name = f"{contour_name_stem1}{z:02d}{contour_name_stem2}.vtk"
        writer.SetFileName(contour_name)
        writer.SetInputData(poly)
        writer.Write()
        os.system('sed -i "1s/4.1/4.0/" {0}'.format(contour_name))


def evaluate_strain_by_length_sa(contour_name_stem, T, result_dir):
    """Calculate the radial and circumferential strain based on the line length"""
    os.makedirs(f"{result_dir}/strain_sa", exist_ok=True)

    # Read the polydata at the first time frame (ED frame)
    fr = 0
    # read poly from vtk file that represents transformed myocardial contours
    # e.g. myo_contour_fr0.vtk
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName("{0}{1:02d}.vtk".format(contour_name_stem, fr))
    reader.Update()
    poly = reader.GetOutput()
    points = poly.GetPoints()

    # * Calculate the length of each line
    lines = poly.GetLines()
    lines_aha = poly.GetCellData().GetArray("Segment ID")
    lines_dir = poly.GetCellData().GetArray("Direction ID")
    n_lines = lines.GetNumberOfCells()
    length_ED = np.zeros(n_lines)
    seg_id = np.zeros(n_lines)
    dir_id = np.zeros(n_lines)

    lines.InitTraversal()
    for i in range(n_lines):
        ids = vtk.vtkIdList()
        lines.GetNextCell(ids)
        # Calculate the length of this line
        p1 = np.array(points.GetPoint(ids.GetId(0)))
        p2 = np.array(points.GetPoint(ids.GetId(1)))
        d = np.linalg.norm(p1 - p2)
        seg_id[i] = lines_aha.GetValue(i)
        # 1 = radial, 2 = circumferential, 3 = longitudinal (la)
        dir_id[i] = lines_dir.GetValue(i)
        length_ED[i] = d

    # * For each time frame later, calculate the strain, i.e. change of length
    table_strain = {}
    table_strain["radial"] = np.zeros((17, T))
    table_strain["circum"] = np.zeros((17, T))

    for fr in range(0, T):
        # Read the polydata
        reader = vtk.vtkPolyDataReader()
        # * We will use all three slices to calculate the strain
        filename_myo = "{0}{1:02d}.vtk".format(contour_name_stem, fr)
        reader.SetFileName(filename_myo)
        reader.Update()
        poly = reader.GetOutput()
        points = poly.GetPoints()

        # Calculate the strain for each line
        lines = poly.GetLines()
        n_lines = lines.GetNumberOfCells()
        strain = np.zeros(n_lines)
        vtk_strain = vtk.vtkFloatArray()
        vtk_strain.SetName("Strain")
        lines.InitTraversal()
        for i in range(n_lines):
            ids = vtk.vtkIdList()
            lines.GetNextCell(ids)
            p1 = np.array(points.GetPoint(ids.GetId(0)))
            p2 = np.array(points.GetPoint(ids.GetId(1)))
            d = np.linalg.norm(p1 - p2)  # use norm as distance

            # * Strain of this line (unit: %)
            strain[i] = (d - length_ED[i]) / length_ED[i] * 100
            vtk_strain.InsertNextTuple1(strain[i])

        # Save the strain array to the vtk file
        poly.GetCellData().AddArray(vtk_strain)
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(poly)
        filename_strain = f"{result_dir}/strain_sa/strain_fr{fr:02d}.vtk"
        writer.SetFileName(filename_strain)
        writer.Write()
        os.system('sed -i "1s/4.1/4.0/" {0}'.format(filename_strain))  # change the version number from 4.1 to 4.0

        # Calculate the segmental and global strains
        for i in range(0, 16):
            table_strain["radial"][i, fr] = np.mean(strain[(seg_id == (i + 1)) & (dir_id == 1)])
            table_strain["circum"][i, fr] = np.mean(strain[(seg_id == (i + 1)) & (dir_id == 2)])
        table_strain["radial"][-1, fr] = np.mean(strain[dir_id == 1])  # global
        table_strain["circum"][-1, fr] = np.mean(strain[dir_id == 2])

    return table_strain["radial"], table_strain["circum"]


def cine_2d_sa_motion_and_strain_analysis(data_dir, par_config_name, temp_dir, result_dir, eval_dice=False):
    """
    Perform motion tracking and strain analysis for short-axis cine MR images.
    """
    # Make folders to arrange files
    os.makedirs(f"{temp_dir}/sa", exist_ok=True)
    os.makedirs(f"{temp_dir}/seg_sa", exist_ok=True)
    os.makedirs(f"{temp_dir}/myo_contour", exist_ok=True)
    os.makedirs(f"{temp_dir}/pair", exist_ok=True)
    os.makedirs(f"{temp_dir}/forward", exist_ok=True)
    os.makedirs(f"{temp_dir}/backward", exist_ok=True)
    os.makedirs(f"{temp_dir}/combined", exist_ok=True)
    os.makedirs(f"{result_dir}/myo_contour_sa", exist_ok=True)
    os.makedirs(f"{result_dir}/log", exist_ok=True)
    os.makedirs(f"{result_dir}/doc", exist_ok=True)
    mirtk_log_file = f"{result_dir}/log/mirtk_sa.log"

    # Focus on the left ventricle so that motion tracking is less affected by
    # the movement of RV and LV outflow tract
    padding(
        f"{data_dir}/seg_sa_ED.nii.gz",  # input A
        f"{data_dir}/seg_sa_ED.nii.gz",  # input B
        f"{temp_dir}/seg_sa/seg_sa_LV_ED.nii.gz",  # output
        3,  # set all pixels with value 3 (RV) to 0 (BG), focusing on LV only
        0,
    )

    # Crop the image to save computation for image registration
    auto_crop_image(f"{temp_dir}/seg_sa/seg_sa_LV_ED.nii.gz", f"{temp_dir}/seg_sa/seg_sa_LV_crop_ED.nii.gz", 20)

    # Transform sa/seg_sa.nii.gz to sa/seg_sa_crop.nii.gz using seg_sa_LV_crop_ED.nii.gz
    os.system(
        f"mirtk transform-image {data_dir}/sa.nii.gz {temp_dir}/sa/sa_crop.nii.gz "
        f"-target {temp_dir}/seg_sa/seg_sa_LV_crop_ED.nii.gz"
    )
    os.system(
        f"mirtk transform-image {data_dir}/seg_sa.nii.gz {temp_dir}/seg_sa/seg_sa_crop.nii.gz "
        f"-target {temp_dir}/seg_sa/seg_sa_LV_crop_ED.nii.gz"
    )

    nim = nib.load(f"{temp_dir}/sa/sa_crop.nii.gz")
    Z = nim.header["dim"][3]
    T = nim.header["dim"][4]
    dt = nim.header["pixdim"][4]
    slice_used = []

    # Split a volume into slices
    split_volume(f"{temp_dir}/sa/sa_crop.nii.gz", f"{temp_dir}/sa/sa_crop_z")
    split_volume(f"{temp_dir}/seg_sa/seg_sa_crop.nii.gz", f"{temp_dir}/seg_sa/seg_sa_crop_z")

    # Extract the myocardial contours for three slices, respectively basal, mid-cavity and apical
    extract_sa_myocardial_contour(
        f"{data_dir}/seg_sa_ED.nii.gz", f"{temp_dir}/myo_contour/myo_contour_z", "_ED", three_slices=True
    )
    logger.info("Myocardial contour at ED extracted")

    # * Inter-frame motion estimation through registration
    if eval_dice:
        dice_lv_myo = []
    for z in range(Z):
        # Since we set three_slices=True, only three slices will be analysed
        if not os.path.exists(f"{temp_dir}/myo_contour/myo_contour_z{z:02d}_ED.vtk"):
            continue
        logger.info(f"Analyze slice {z}")
        slice_used.append(z)
        # Split the cine sequence for this slice (split according to time instead of slice)
        # e.g. sa_crop_z00.nii.gz -> sa_crop_z00_fr00.nii.gz, sa_crop_z00_fr01.nii.gz, ...
        split_sequence(f"{temp_dir}/sa/sa_crop_z{z:02d}.nii.gz", f"{temp_dir}/sa/sa_crop_z{z:02d}_fr")

        logger.info(f"Slice {z}: Forward registration")
        # Ref https://doi.org/10.1038/s41591-020-1009-y
        # * Image registration between successive time frames for forward image registration
        # * Note1: We can use mirtk info to get information about generated .dof.gz file
        # * Note2: For mirtk register <img1> <img2> -dofout <dof>, the transformation is from img2 to img1
        # results are named as ffd_z{z:02d}_pair_00_to_01.dof.gz, ffd_z{z:02d}_pair_01_to_02.dof.gz, ...
        for fr in range(1, T):
            target_fr = fr - 1
            source_fr = fr
            target = f"{temp_dir}/sa/sa_crop_z{z:02d}_fr{target_fr:02d}.nii.gz"
            source = f"{temp_dir}/sa/sa_crop_z{z:02d}_fr{source_fr:02d}.nii.gz"
            # output transformation to be used
            dof = f"{temp_dir}/pair/ffd_z{z:02d}_pair_{target_fr:02d}_to_{source_fr:02d}.dof.gz"
            os.system(f"mirtk register {target} {source} -parin {par_config_name} -dofout {dof} >> {mirtk_log_file} 2>&1")

        # * Start forward image registration
        # results are named as ffd_z{z:02d}_forward_00_to_01.dof.gz, ffd_z{z:02d}_forward_00_to_02.dof.gz, ...

        # For the first and second time frame, directly copy the transformation
        os.system(
            f"cp {temp_dir}/pair/ffd_z{z:02d}_pair_00_to_01.dof.gz " f"{temp_dir}/forward/ffd_z{z:02d}_forward_00_to_01.dof.gz"
        )
        # For the rest time frames, compose the transformation fields
        for fr in range(2, T):
            dofs = ""
            for k in range(1, fr + 1):
                dof = f"{temp_dir}/pair/ffd_z{z:02d}_pair_{k - 1:02d}_to_{k:02d}.dof.gz"
                dofs += dof + " "
            dof_out = f"{temp_dir}/forward/ffd_z{z:02d}_forward_00_to_{fr:02d}.dof.gz"
            # todo Use -approximate will cause strain computation failed
            os.system(f"mirtk compose-dofs {dofs} {dof_out}")

        logger.info(f"Slice {z}: Backward registration")
        # * Image registration reversely between successive time frames for backward image registration
        # results are named as ffd_z{z:02d}_pair_01_to_00.dof.gz, ffd_z{z:02d}_pair_02_to_01.dof.gz, ...
        for fr in range(T - 1, 0, -1):
            target_fr = (fr + 1) % T
            source_fr = fr
            target = f"{temp_dir}/sa/sa_crop_z{z:02d}_fr{target_fr:02d}.nii.gz"
            source = f"{temp_dir}/sa/sa_crop_z{z:02d}_fr{source_fr:02d}.nii.gz"
            dof = f"{temp_dir}/pair/ffd_z{z:02d}_pair_{target_fr:02d}_to_{source_fr:02d}.dof.gz"
            os.system(f"mirtk register {target} {source} -parin {par_config_name} -dofout {dof} >> {mirtk_log_file} 2>&1")

        # * Start backward image registration
        # results are named as ffd_z{z:02d}_backward_00_to_01.dof.gz, ffd_z{z:02d}_backward_00_to_02.dof.gz, ...

        # For the first and last time frame, directly copy the transformation
        os.system(
            f"cp {temp_dir}/pair/ffd_z{z:02d}_pair_00_to_{(T - 1):02d}.dof.gz "
            f"{temp_dir}/backward/ffd_z{z:02d}_backward_00_to_{(T - 1):02d}.dof.gz"
        )
        # For the rest time frames, compose the transformation fields
        for fr in range(T - 2, 0, -1):
            dofs = ""
            for k in range(T - 1, fr - 1, -1):
                dof = f"{temp_dir}/pair/ffd_z{z:02d}_pair_{((k + 1) % T):02d}_to_{k:02d}.dof.gz"
                dofs += dof + " "
            dof_out = f"{temp_dir}/backward/ffd_z{z:02d}_backward_00_to_{fr:02d}.dof.gz"
            os.system(f"mirtk compose-dofs {dofs} {dof_out}")

        logger.info(f"Slice {z}: Combine the forward and backward transformations")
        # * Average the forward and backward transformations
        # * For a frame at early stage of a cardiac cycle, the forward displacement field will have a higher weight
        os.system(f"mirtk init-dof {temp_dir}/forward/ffd_z{z:02d}_forward_00_to_00.dof.gz")  # initialize a dof file
        os.system(f"mirtk init-dof {temp_dir}/backward/ffd_z{z:02d}_backward_00_to_00.dof.gz")
        os.system(f"mirtk init-dof {temp_dir}/combined/ffd_z{z:02d}_00_to_00.dof.gz")

        # todo: Currently we directly copy the transformations.
        # todo: Current Dice is very bad (0.3) for Myo
        # todo: average_3d_ffd will crush for the first frame; and cannot be used when compose-dofs -approximate
        # os.system(
        #     f"cp {temp_dir}/forward/ffd_z{z:02d}_forward_00_to_01.dof.gz "
        #     f"{temp_dir}/combined/ffd_z{z:02d}_00_to_01.dof.gz"
        # )
        # for fr in range(2, T):
        #     dof_forward = f"{temp_dir}/forward/ffd_z{z:02d}_forward_00_to_{fr:02d}.dof.gz"
        #     weight_forward = float(T - fr) / T
        #     dof_backward = f"{temp_dir}/backward/ffd_z{z:02d}_backward_00_to_{fr:02d}.dof.gz"
        #     weight_backward = float(fr) / T
        #     # combined transformation to be created
        #     dof_combine = f"{temp_dir}/combined/ffd_z{z:02d}_00_to_{fr:02d}.dof.gz"
        #     # 2 means there are two input files
        #     os.system(
        #          f"average_3d_ffd 2 {dof_forward} {weight_forward} {dof_backward} {weight_backward} {dof_combine}"
        #     )

        for fr in range(1, T):
            dof_forward = f"{temp_dir}/forward/ffd_z{z:02d}_forward_00_to_{fr:02d}.dof.gz"
            dof_backward = f"{temp_dir}/backward/ffd_z{z:02d}_backward_00_to_{fr:02d}.dof.gz"
            # combined transformation to be created
            dof_combine = f"{temp_dir}/combined/ffd_z{z:02d}_00_to_{fr:02d}.dof.gz"

            if fr > T // 2:
                os.system(f"cp {dof_forward} {dof_combine}")
            else:
                os.system(f"cp {dof_backward} {dof_combine}")

        # * Transform the contours using combined transformation
        # results are named as myo_contour_z{z:02d}_fr{fr:02d}.vtk
        for fr in range(0, T):
            os.system(
                f"mirtk transform-points "
                f"{temp_dir}/myo_contour/myo_contour_z{z:02d}_ED.vtk "
                f"{temp_dir}/myo_contour/myo_contour_z{z:02d}_fr{fr:02d}.vtk "
                f"-dofin {temp_dir}/combined/ffd_z{z:02d}_00_to_{fr:02d}.dof.gz"
            )

        # Evaluate the Dice metric to ensure the accuracy of strain calculation
        if eval_dice:
            logger.info(f"Evaluate the Dice metric for slice {z}")

            split_sequence(
                f"{temp_dir}/seg_sa/seg_sa_crop_z{z:02d}.nii.gz",
                f"{temp_dir}/seg_sa/seg_sa_crop_z{z:02d}_fr",
            )

            image_names = []
            for fr in range(0, T):
                # Use the obtained dof to warp image of future frames and compare it to the first one
                os.system(
                    f"mirtk transform-image {temp_dir}/seg_sa/seg_sa_crop_z{z:02d}_fr{fr:02d}.nii.gz "
                    f"{temp_dir}/seg_sa/seg_sa_crop_warp_ffd_z{z:02d}_fr{fr:02d}.nii.gz "
                    f"-dofin {temp_dir}/combined/ffd_z{z:02d}_00_to_{fr:02d}.dof.gz "
                    f"-target {temp_dir}/seg_sa/seg_sa_crop_z{z:02d}_fr00.nii.gz"
                )
                image_A = nib.load(f"{temp_dir}/seg_sa/seg_sa_crop_z{z:02d}_fr00.nii.gz").get_fdata()  # target image
                image_B = nib.load(
                    f"{temp_dir}/seg_sa/seg_sa_crop_warp_ffd_z{z:02d}_fr{fr:02d}.nii.gz"
                ).get_fdata()  # warped image
                # reduce the extra dimension for warped image
                image_B = image_B[:, :, :, 0]
                nim_B = nib.load(f"{temp_dir}/seg_sa/seg_sa_crop_warp_ffd_z{z:02d}_fr{fr:02d}.nii.gz")
                nim_B_new = nib.Nifti1Image(image_B, nim_B.affine, nim_B.header)
                nib.save(nim_B_new, f"{temp_dir}/seg_sa/seg_sa_crop_warp_ffd_z{z:02d}_fr{fr:02d}.nii.gz")
                # evaluate dice metric over LV and Myo on the warped segmentation and the target segmentation
                dice_lv_myo.append([np_categorical_dice(image_A, image_B, 1), np_categorical_dice(image_A, image_B, 2)])
                image_names.append(f"{temp_dir}/seg_sa/seg_sa_crop_warp_ffd_z{z:02d}_fr{fr:02d}.nii.gz")

            sa_warp_combined_name = f"{temp_dir}/seg_sa/seg_sa_crop_warp_ffd_z{z:02d}.nii.gz"  # a sequence to be made
            make_sequence(image_names, dt, sa_warp_combined_name)  # opposite to split_sequence, dt is the time interval

    # output the dice result for all slices to csv
    if eval_dice:
        df_dice = pd.DataFrame(dice_lv_myo)
        df_dice.columns = ["LV", "Myo"]
        df_dice.index = [f"Slice{s}:{t}" for s in slice_used for t in range(T)]
        # append mean result at the end
        df_dice.loc["mean"] = df_dice.mean()
        logger.info(f"Mean Dice for LV: {df_dice.loc['mean'].values[0]}")
        logger.info(f"Mean Dice for Myo: {df_dice.loc['mean'].values[1]}")
        df_dice.to_csv(f"{result_dir}/doc/dice_sa_warp_ffd.csv", index=True, header=True)

    # * Merge the 2D tracked contours from all the slice into one vtk
    logger.info("Merge the 2D tracked contours from all the slice into one vtk")
    for fr in range(0, T):
        contours = vtk.vtkAppendPolyData()
        reader = {}
        # recall that if we set three_slices=True, only three slices will be analysed
        for z in range(Z):
            # result from contours tracked using combined transformation from average_3d_ffd
            if not os.path.exists(f"{temp_dir}/myo_contour/myo_contour_z{z:02d}_fr{fr:02d}.vtk"):
                continue
            reader[z] = vtk.vtkPolyDataReader()
            reader[z].SetFileName(f"{temp_dir}/myo_contour/myo_contour_z{z:02d}_fr{fr:02d}.vtk")
            reader[z].Update()
            contours.AddInputData(reader[z].GetOutput())
        contours.Update()
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(f"{result_dir}/myo_contour_sa/myo_contour_fr{fr:02d}.vtk")
        writer.SetInputData(contours.GetOutput())
        writer.Write()

    return


def remove_mitral_valve_points(endo_contour, epi_contour, mitral_plane):
    """Remove the mitral valve points from the contours and
    start the contours from the point next to the mitral valve plane.
    So connecting the lines will be easier in the next step.
    """
    N = endo_contour.shape[0]
    start_i = 0
    for i in range(N):
        y, x = endo_contour[i]
        prev_y, prev_x = endo_contour[(i - 1) % N]
        if not mitral_plane[x, y] and mitral_plane[prev_x, prev_y]:
            start_i = i
            break
    endo_contour = np.concatenate((endo_contour[start_i:], endo_contour[:start_i]))

    N = endo_contour.shape[0]
    end_i = N
    for i in range(N):
        y, x = endo_contour[i]
        if mitral_plane[x, y]:
            end_i = i
            break
    endo_contour = endo_contour[:end_i]

    N = epi_contour.shape[0]
    start_i = 0
    for i in range(N):
        y, x = epi_contour[i]
        y2, x2 = epi_contour[(i - 1) % N]
        if not mitral_plane[x, y] and mitral_plane[x2, y2]:
            start_i = i
            break
    epi_contour = np.concatenate((epi_contour[start_i:], epi_contour[:start_i]))

    N = epi_contour.shape[0]
    end_i = N
    for i in range(N):
        y, x = epi_contour[i]
        if mitral_plane[x, y]:
            end_i = i
            break
    epi_contour = epi_contour[:end_i]
    return endo_contour, epi_contour


def _determine_sa_qualified_slice(seg_sa: Tuple[float, float]):
    """
    Determine whether certain slice of the short-axis is qualified at a given timepoint.
    """
    if seg_sa.ndim != 2:
        raise ValueError("The input should be a 2D image.")

    # Label class in the segmentation
    labels = {"BG": 0, "LV": 1, "Myo": 2, "RV": 3}

    # Criterion 1: The area of LV and RV should be above a threshold
    pixel_thres = 10
    if np.sum(seg_sa == labels["LV"]) < pixel_thres:
        return False
    if np.sum(seg_sa == labels["RV"]) < pixel_thres:
        return False

    # Criterion 2: If the myocardium can surround LV perfectly, then we can determine the basal slice.
    LV_mask = (seg_sa == labels["LV"]).astype(np.uint8)
    myo_mask = (seg_sa == labels["Myo"]).astype(np.uint8)
    contours, _ = cv2.findContours(myo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(LV_mask.shape[0]):
        for j in range(LV_mask.shape[1]):
            if LV_mask[i, j] == 1:
                if all(cv2.pointPolygonTest(contour, (j, i), False) < 0 for contour in contours):
                    return False
    return True


def determine_sa_basal_slice(seg_sa: Tuple[float, float, float]):
    """
    Determine the basal slice of the short-axis image for a given timepoint.
    """
    if seg_sa.ndim != 3:
        raise ValueError("The input should be a 3D image.")

    _, _, Z = seg_sa.shape

    for z in range(Z):
        if z > Z / 3:
            # In this case, the slice will be close to apex and should not be considered
            raise ValueError("The basal slice is not found.")

        seg_z = seg_sa[:, :, z]

        if not _determine_sa_qualified_slice(seg_z):
            continue
        return z


def determine_sa_apical_slice(seg_sa: Tuple[float, float, float]):
    """
    Determine the apical slice of the short-axis image for a given timepoint.
    """
    if seg_sa.ndim != 3:
        raise ValueError("The input should be a 3D image.")

    _, _, Z = seg_sa.shape

    for z in range(Z - 1, -1, -1):
        if z < Z / 3:
            # In this case, the slice will be close to base and should not be considered
            raise ValueError("The apical slice is not found.")

        seg_z = seg_sa[:, :, z]

        if not _determine_sa_qualified_slice(seg_z):
            continue
        return z


def determine_axes(label_i, nim, long_axis):
    """
    Determine the major axis and minor axis using the binary segmentation and
    return the axes as well as lines that represents the major/minor axis that entirely covered by the segmentation.
    """
    if label_i.ndim != 2:
        raise ValueError("The input should be a 2D image.")

    points_label = np.nonzero(label_i)
    points = []
    for j in range(len(points_label[0])):  # the number of points
        x = points_label[0][j]
        y = points_label[1][j]
        # np.dot(nim.affine, np.array([x, y, 0, 1]))[:3] is equivalent to apply_affine(nim.affine, [x, y, 0])
        # define (0,1): 0 means it is 2D image, 1 means homogeneous coordinate
        points += [[x, y, np.dot(np.dot(nim.affine, np.array([x, y, 0, 1]))[:3], long_axis)]]
    points = np.array(points)
    points = points[points[:, 2].argsort()]

    n_points = len(points)
    top_points = points[int(2 * n_points / 3) :]
    cx, cy, _ = np.mean(top_points, axis=0)

    # The centre at the bottom part (bottom third)
    bottom_points = points[: int(n_points / 3)]
    bx, by, _ = np.mean(bottom_points, axis=0)

    # Determine the major axis by connecting the geometric centre and the bottom centre
    major_axis = np.array([cx - bx, cy - by])  # major_axis is a vector
    major_axis = major_axis / np.linalg.norm(major_axis)  # normalization

    px = cx + major_axis[0] * 100
    py = cy + major_axis[1] * 100
    qx = cx - major_axis[0] * 100
    qy = cy - major_axis[1] * 100

    image_line_major = np.zeros(label_i.shape)
    cv2.line(image_line_major, (int(qy), int(qx)), (int(py), int(px)), (1, 0, 0))
    image_line_major = label_i & (image_line_major > 0)

    minor_axis = np.array([-major_axis[1], major_axis[0]])  # minor_axis is a vector
    minor_axis = minor_axis / np.linalg.norm(minor_axis)  # normalization

    # Mid level, to be used when determining tranverse diameter
    mx, my, _ = np.mean(points, axis=0)

    rx = mx + minor_axis[0] * 100
    ry = my + minor_axis[1] * 100
    sx = mx - minor_axis[0] * 100
    sy = my - minor_axis[1] * 100
    if np.isnan(rx) or np.isnan(ry) or np.isnan(sx) or np.isnan(sy):
        raise ValueError("Minor axis can not determined.")

    image_line_minor = np.zeros(label_i.shape)
    cv2.line(image_line_minor, (int(sy), int(sx)), (int(ry), int(rx)), (1, 0, 0))
    image_line_minor = label_i & (image_line_minor > 0)

    return major_axis, minor_axis, image_line_major, image_line_minor


def determine_valve_landmark(seg4: Tuple[float, float]):
    """
    Determine the landmark for valves, so that valve diameter as well as AVPD can be measured.

    Parameters
    ----------
    seg4 : Tuple[float, float]
        The segmentation of all four chambers and myocardium in the long-axis image.
    """
    seg_LV = seg4 == 1
    seg_LV = seg_LV.astype(np.uint8)
    seg_LV = np.ascontiguousarray(seg_LV)
    seg_LV_contours, _ = cv2.findContours(seg_LV, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seg_LV_contour = seg_LV_contours[0]
    seg_LV_border = np.zeros_like(seg_LV)
    cv2.drawContours(seg_LV_border, seg_LV_contours, -1, 1, 1)
    seg_LV_coords = np.column_stack(np.where(seg_LV))
    seg_LV_coords = np.column_stack([seg_LV_coords[:, 1], seg_LV_coords[:, 0]])

    seg_RV = seg4 == 3
    seg_RV = seg_RV.astype(np.uint8)
    seg_RV = np.ascontiguousarray(seg_RV)
    seg_RV_contours, _ = cv2.findContours(seg_RV, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seg_RV_contour = seg_RV_contours[0]
    seg_RV_border = np.zeros_like(seg_RV)
    cv2.drawContours(seg_RV_border, seg_RV_contours, -1, 1, 1)
    seg_RV_coords = np.column_stack(np.where(seg_RV))
    seg_RV_coords = np.column_stack([seg_RV_coords[:, 1], seg_RV_coords[:, 0]])

    seg_LA = seg4 == 4
    seg_LA = seg_LA.astype(np.uint8)
    seg_LA = np.ascontiguousarray(seg_LA)
    seg_LA_contours, _ = cv2.findContours(seg_LA, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seg_LA_contour = seg_LA_contours[0]
    seg_LA_border = np.zeros_like(seg_LA)
    cv2.drawContours(seg_LA_border, seg_LA_contours, -1, 1, 1)
    seg_LA_coords = np.column_stack(np.where(seg_LA))
    seg_LA_coords = np.column_stack([seg_LA_coords[:, 1], seg_LA_coords[:, 0]])

    seg_RA = seg4 == 5
    seg_RA = seg_RA.astype(np.uint8)
    seg_RA = np.ascontiguousarray(seg_RA)
    seg_RA_contours, _ = cv2.findContours(seg_RA, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seg_RA_contour = seg_RA_contours[0]
    seg_RA_border = np.zeros_like(seg_RA)
    cv2.drawContours(seg_RA_border, seg_RA_contours, -1, 1, 1)
    seg_RA_coords = np.column_stack(np.where(seg_RA))
    seg_RA_coords = np.column_stack([seg_RA_coords[:, 1], seg_RA_coords[:, 0]])

    # * From cloest point in LA to LV
    seg_LA_epsilon = 0.01
    seg_LA_corners = cv2.approxPolyDP(seg_LA_contour, seg_LA_epsilon * cv2.arcLength(seg_LA_contour, True), True)
    while len(seg_LA_corners) > 4:
        seg_LA_epsilon += 0.01
        seg_LA_corners = cv2.approxPolyDP(seg_LA_contour, seg_LA_epsilon * cv2.arcLength(seg_LA_contour, True), True)
    if seg_LA_epsilon > 0.1:
        raise ValueError("Too many corners for LA detected")

    seg_LA_corner_points = [(point[0][0], point[0][1]) for point in seg_LA_corners]

    LA_LV_distances = []
    for point in seg_LV_coords:
        x, y = point
        for corner in seg_LA_corner_points:
            distance = np.sqrt((x - corner[0]) ** 2 + (y - corner[1]) ** 2)
            LA_LV_distances.append((distance, corner))
    LA_LV_distances.sort(key=lambda x: x[0])
    LA_LV_sorted_corners = [corner for _, corner in LA_LV_distances]
    LA_LV_unique_corners = list(OrderedDict.fromkeys(LA_LV_sorted_corners))
    LA_lm = LA_LV_unique_corners[:2]

    # * From closest point in LV to LA
    seg_LV_epsilon = 0.01
    seg_LV_corners = cv2.approxPolyDP(seg_LV_contour, seg_LV_epsilon * cv2.arcLength(seg_LV_contour, True), True)
    while len(seg_LV_corners) > 4:
        seg_LV_epsilon += 0.01
        seg_LV_corners = cv2.approxPolyDP(seg_LV_contour, seg_LV_epsilon * cv2.arcLength(seg_LV_contour, True), True)
    if seg_LV_epsilon > 0.1:
        raise ValueError("Too many corners for LV detected")

    seg_LV_corner_points = [(point[0][0], point[0][1]) for point in seg_LV_corners]

    LV_LA_distances = []
    for point in seg_LA_coords:
        x, y = point
        for corner in seg_LV_corner_points:
            distance = np.sqrt((x - corner[0]) ** 2 + (y - corner[1]) ** 2)
            LV_LA_distances.append((distance, corner))
    LV_LA_distances.sort(key=lambda x: x[0])
    LV_LA_sorted_corners = [corner for _, corner in LV_LA_distances]
    LV_LA_unique_corners = list(OrderedDict.fromkeys(LV_LA_sorted_corners))
    LV_lm = LV_LA_unique_corners[:2]

    # * From closest point in RA to RV
    seg_RA_epsilon = 0.01
    seg_RA_corners = cv2.approxPolyDP(seg_RA_contour, seg_RA_epsilon * cv2.arcLength(seg_RA_contour, True), True)
    while len(seg_RA_corners) > 4:
        seg_RA_epsilon += 0.01
        seg_RA_corners = cv2.approxPolyDP(seg_RA_contour, seg_RA_epsilon * cv2.arcLength(seg_RA_contour, True), True)
    if seg_RA_epsilon > 0.1:
        raise ValueError("Too many corners for RA detected")

    seg_RA_corner_points = [(point[0][0], point[0][1]) for point in seg_RA_corners]

    RA_RV_distances = []
    for point in seg_RV_coords:
        x, y = point
        for corner in seg_RA_corner_points:
            distance = np.sqrt((x - corner[0]) ** 2 + (y - corner[1]) ** 2)
            RA_RV_distances.append((distance, corner))
    RA_RV_distances.sort(key=lambda x: x[0])
    RA_RV_sorted_corners = [corner for _, corner in RA_RV_distances]
    RA_RV_unique_corners = list(OrderedDict.fromkeys(RA_RV_sorted_corners))
    RA_lm = RA_RV_unique_corners[:2]

    # * From closest point in RV to RA
    seg_RV_epsilon = 0.01
    seg_RV_corners = cv2.approxPolyDP(seg_RV_contour, seg_RV_epsilon * cv2.arcLength(seg_RV_contour, True), True)
    while len(seg_RV_corners) > 4:
        seg_RV_epsilon += 0.01
        seg_RV_corners = cv2.approxPolyDP(seg_RV_contour, seg_RV_epsilon * cv2.arcLength(seg_RV_contour, True), True)
    if seg_RV_epsilon > 0.1:
        raise ValueError("Too many corners for RV detected")

    seg_RV_corner_points = [(point[0][0], point[0][1]) for point in seg_RV_corners]

    RV_RA_distances = []
    for point in seg_RA_coords:
        x, y = point
        for corner in seg_RV_corner_points:
            distance = np.sqrt((x - corner[0]) ** 2 + (y - corner[1]) ** 2)
            RV_RA_distances.append((distance, corner))
    RV_RA_distances.sort(key=lambda x: x[0])
    RV_RA_sorted_corners = [corner for _, corner in RV_RA_distances]
    RV_RA_unique_corners = list(OrderedDict.fromkeys(RV_RA_sorted_corners))
    RV_lm = RV_RA_unique_corners[:2]

    return LV_lm, LA_lm, RV_lm, RA_lm


def determine_la_aha_part(seg_la, affine_la, affine_sa):
    """Extract the mid-line of the left ventricle, record its index
    along the long-axis and determine the part for each index.
    """
    # Label class in the segmentation
    labels = {"BG": 0, "LV": 1, "Myo": 2, "RV": 3, "LA": 4, "RA": 5}

    # Sort the left ventricle and myocardium points according to their long-axis locations
    lv_myo_points = []
    X, Y = seg_la.shape[:2]
    z = 0
    for y in range(Y):
        for x in range(X):
            if seg_la[x, y] == labels["LV"] or seg_la[x, y] == labels["Myo"]:
                z_sa = np.dot(np.linalg.inv(affine_sa), np.dot(affine_la, np.array([x, y, z, 1])))[2]
                la_idx = int(round(z_sa * 2))
                lv_myo_points += [[x, y, la_idx]]
    lv_myo_points = np.array(lv_myo_points)
    lv_myo_idx_min = np.min(lv_myo_points[:, 2])
    lv_myo_idx_max = np.max(lv_myo_points[:, 2])

    # Determine the AHA part according to the slice location along the long-axis
    if affine_sa[2, 2] > 0:
        la_idx = np.arange(lv_myo_idx_max, lv_myo_idx_min, -1)
    else:
        la_idx = np.arange(lv_myo_idx_min, lv_myo_idx_max + 1, 1)

    n_la_idx = len(la_idx)
    i1 = int(math.ceil(n_la_idx / 3.0))
    i2 = int(math.ceil(2 * n_la_idx / 3.0))
    i3 = n_la_idx

    part_z = {}
    for i in range(0, i1):
        part_z[la_idx[i]] = "basal"

    for i in range(i1, i2):
        part_z[la_idx[i]] = "mid"

    for i in range(i2, i3):
        part_z[la_idx[i]] = "apical"

    # Extract the mid-line of left ventricle endocardium.
    # Only use the endocardium points so that it would not be affected by
    # the myocardium points at the most basal slices.
    lv_points = []
    X, Y = seg_la.shape[:2]
    z = 0
    for y in range(Y):
        for x in range(X):
            if seg_la[x, y] == labels["LV"]:
                z_sa = np.dot(np.linalg.inv(affine_sa), np.dot(affine_la, np.array([x, y, z, 1])))[2]
                la_idx = int(round(z_sa * 2))
                lv_points += [[x, y, la_idx]]
    lv_points = np.array(lv_points)
    lv_idx_min = np.min(lv_points[:, 2])
    lv_idx_max = np.max(lv_points[:, 2])

    mid_line = {}
    for la_idx in range(lv_idx_min, lv_idx_max + 1):
        mx, my = np.mean(lv_points[lv_points[:, 2] == la_idx, :2], axis=0)
        mid_line[la_idx] = np.dot(affine_la, np.array([mx, my, z, 1]))[:3]

    for la_idx in range(lv_myo_idx_min, lv_idx_min):
        mid_line[la_idx] = mid_line[lv_idx_min]

    for la_idx in range(lv_idx_max, lv_myo_idx_max + 1):
        mid_line[la_idx] = mid_line[lv_idx_max]
    return part_z, mid_line


def determine_la_aha_segment_id(point, la_idx, axis, mid_line, part_z):
    """Determine the AHA segment ID given a point on long-axis images."""
    # The mid-point at this position
    mid_point = mid_line[la_idx]

    # The line from the mid-point to the contour point
    vec = point - mid_point
    if np.dot(vec, axis["lv_to_sep"]) > 0:
        # This is spetum
        if part_z[la_idx] == "basal":
            # basal septal
            seg_id = 1
        elif part_z[la_idx] == "mid":
            # mid septal
            seg_id = 3
        elif part_z[la_idx] == "apical":
            # apical septal
            seg_id = 5
    else:
        # This is lateral
        if part_z[la_idx] == "basal":
            # basal lateral
            seg_id = 2
        elif part_z[la_idx] == "mid":
            # mid lateral
            seg_id = 4
        elif part_z[la_idx] == "apical":
            # apical lateral
            seg_id = 6
    return seg_id


def extract_la_myocardial_contour(seg_la_name, seg_sa_name, contour_name):
    """
    Extract the myocardial contours on long-axis images.
    Also, determine the AHA segment ID for all the contour points.
    """
    # Read the segmentation image
    nim = nib.load(seg_la_name)
    # X, Y, Z = nim.header["dim"][1:4]
    affine = nim.affine
    seg = nim.get_fdata()

    # Label class in the segmentation
    labels = {"BG": 0, "LV": 1, "Myo": 2, "RV": 3, "LA": 4, "RA": 5}

    # Determine the AHA coordinate system using the mid-cavity slice of short-axis images
    nim_sa = nib.load(seg_sa_name)
    affine_sa = nim_sa.affine
    seg_sa = nim_sa.get_fdata()
    aha_axis = determine_aha_coordinate_system(seg_sa, affine_sa)

    # Construct the points set and data arrays to represent both endo and epicardial contours
    points = vtk.vtkPoints()
    points_radial = vtk.vtkFloatArray()
    points_radial.SetName("Direction_Radial")
    points_radial.SetNumberOfComponents(3)
    points_label = vtk.vtkIntArray()
    points_label.SetName("Label")
    points_aha = vtk.vtkIntArray()
    points_aha.SetName("Segment ID")
    point_id = 0
    lines = vtk.vtkCellArray()
    lines_aha = vtk.vtkIntArray()
    lines_aha.SetName("Segment ID")
    lines_dir = vtk.vtkIntArray()
    lines_dir.SetName("Direction ID")  # to be used when calculating the strain, need to be saved in vtk file

    # Check whether there is the endocardial segmentation
    # Only keep the largest connected component
    z = 0
    seg_z = seg[:, :, z]
    endo = (seg_z == labels["LV"]).astype(np.uint8)
    endo = get_largest_cc(endo).astype(np.uint8)
    # The myocardium may be split to two parts due to the very thin apex.
    # So we do not apply get_largest_cc() to it. However, we remove small pieces, which
    # may cause problems in determining the contours.
    myo = (seg_z == labels["Myo"]).astype(np.uint8)
    myo = remove_small_cc(myo).astype(np.uint8)
    epi = (endo | myo).astype(np.uint8)
    epi = get_largest_cc(epi).astype(np.uint8)

    # Extract endocardial contour
    # Note: cv2 considers an input image as a Y x X array, which is different
    # from nibabel which assumes a X x Y array.
    contours, _ = cv2.findContours(cv2.inRange(endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    endo_contour = contours[0][:, 0, :]

    # Extract epicardial contour
    contours, _ = cv2.findContours(cv2.inRange(epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    epi_contour = contours[0][:, 0, :]

    # Record the points located on the mitral valve plane.
    mitral_plane = np.zeros(seg_z.shape)
    N = epi_contour.shape[0]
    for i in range(N):
        y, x = epi_contour[i]
        if endo[x, y]:
            mitral_plane[x, y] = 1

    # Remove the mitral valve points from the contours and
    # start the contours from the point next to the mitral valve plane.
    # So connecting the lines will be easier in the next step.
    if np.sum(mitral_plane) >= 1:
        endo_contour, epi_contour = remove_mitral_valve_points(endo_contour, epi_contour, mitral_plane)

    # Note that remove_mitral_valve_points may fail if the endo or epi has more
    # than one connected components. As a result, the endo_contour or epi_contour
    # may only have zero or one points left, which cause problems for approximate_contour.

    # Smooth the contours
    if len(endo_contour) >= 2:
        endo_contour = approximate_contour(endo_contour)
    if len(epi_contour) >= 2:
        epi_contour = approximate_contour(epi_contour)

    # Determine the aha part and extract the mid-line of the left ventricle
    part_z, mid_line = determine_la_aha_part(seg_z, affine, affine_sa)
    la_idx_min = np.array([x for x in part_z.keys()]).min()
    la_idx_max = np.array([x for x in part_z.keys()]).max()

    # Go through the endo contour points
    N = endo_contour.shape[0]
    for i in range(N):
        y, x = endo_contour[i]

        # The world coordinate of this point
        p = np.dot(affine, np.array([x, y, z, 1]))[:3]
        points.InsertNextPoint(p[0], p[1], p[2])

        # The index along the long axis
        z_sa = np.dot(np.linalg.inv(affine_sa), np.hstack([p, 1]))[2]
        la_idx = int(round(z_sa * 2))
        la_idx = max(la_idx, la_idx_min)
        la_idx = min(la_idx, la_idx_max)

        # The radial direction
        mid_point = mid_line[la_idx]
        d = p - mid_point
        d = d / np.linalg.norm(d)
        points_radial.InsertNextTuple3(d[0], d[1], d[2])

        # Record the type of the point (1 = endo, 2 = epi)
        points_label.InsertNextTuple1(1)

        # Record the segment ID
        seg_id = determine_la_aha_segment_id(p, la_idx, aha_axis, mid_line, part_z)
        points_aha.InsertNextTuple1(seg_id)

        # Add the line
        if i < (N - 1):
            lines.InsertNextCell(2, [point_id, point_id + 1])
            lines_aha.InsertNextTuple1(seg_id)

            # Line direction (1 = radial, 2 = circumferential, 3 = longitudinal)
            lines_dir.InsertNextTuple1(3)

        # Increment the point index
        point_id += 1

    # Go through the epi contour points
    N = epi_contour.shape[0]
    for i in range(N):
        y, x = epi_contour[i]

        # The world coordinate of this point
        p = np.dot(affine, np.array([x, y, z, 1]))[:3]
        points.InsertNextPoint(p[0], p[1], p[2])

        # The index along the long axis
        z_sa = np.dot(np.linalg.inv(affine_sa), np.hstack([p, 1]))[2]
        la_idx = int(round(z_sa * 2))
        la_idx = max(la_idx, la_idx_min)
        la_idx = min(la_idx, la_idx_max)

        # The radial direction
        mid_point = mid_line[la_idx]
        d = p - mid_point
        d = d / np.linalg.norm(d)
        points_radial.InsertNextTuple3(d[0], d[1], d[2])

        # Record the type of the point (1 = endo, 2 = epi)
        points_label.InsertNextTuple1(2)

        # Record the segment ID
        seg_id = determine_la_aha_segment_id(p, la_idx, aha_axis, mid_line, part_z)
        points_aha.InsertNextTuple1(seg_id)

        # Add the line
        if i < (N - 1):
            lines.InsertNextCell(2, [point_id, point_id + 1])
            lines_aha.InsertNextTuple1(seg_id)

            # Line direction (1 = radial, 2 = circumferential, 3 = longitudinal)
            lines_dir.InsertNextTuple1(3)

        # Increment the point index
        point_id += 1

    # Save to a vtk file
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.GetPointData().AddArray(points_label)
    poly.GetPointData().AddArray(points_aha)
    poly.GetPointData().AddArray(points_radial)
    poly.SetLines(lines)
    poly.GetCellData().AddArray(lines_aha)
    poly.GetCellData().AddArray(lines_dir)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(contour_name)
    writer.SetInputData(poly)
    writer.Write()

    # Change vtk file version to 4.0 to avoid the warning by MIRTK, which is
    # developed using VTK 6.3, which does not know file version 4.1.
    os.system('sed -i "1s/4.1/4.0/" {0}'.format(contour_name))


def evaluate_strain_by_length_la(contour_name_stem, T, result_dir):
    """Calculate the longitudinal strain based on the line length"""
    os.makedirs(f"{result_dir}/strain_la", exist_ok=True)

    # Read the polydata at the first time frame (ED frame)
    fr = 0
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName("{0}{1:02d}.vtk".format(contour_name_stem, fr))
    reader.Update()
    poly = reader.GetOutput()
    points = poly.GetPoints()

    # Calculate the length of each line
    lines = poly.GetLines()
    lines_aha = poly.GetCellData().GetArray("Segment ID")
    lines_dir = poly.GetCellData().GetArray("Direction ID")
    n_lines = lines.GetNumberOfCells()
    length_ED = np.zeros(n_lines)
    seg_id = np.zeros(n_lines)
    dir_id = np.zeros(n_lines)

    lines.InitTraversal()
    for i in range(n_lines):
        ids = vtk.vtkIdList()
        lines.GetNextCell(ids)
        p1 = np.array(points.GetPoint(ids.GetId(0)))
        p2 = np.array(points.GetPoint(ids.GetId(1)))
        d = np.linalg.norm(p1 - p2)
        seg_id[i] = lines_aha.GetValue(i)
        dir_id[i] = lines_dir.GetValue(i)
        length_ED[i] = d

    # For each time frame, calculate the strain, i.e. change of length
    table_strain = {}
    table_strain["longit"] = np.zeros((7, T))

    for fr in range(0, T):
        # Read the polydata
        reader = vtk.vtkPolyDataReader()
        filename = "{0}{1:02d}.vtk".format(contour_name_stem, fr)
        reader.SetFileName(filename)
        reader.Update()
        poly = reader.GetOutput()
        points = poly.GetPoints()

        # Calculate the strain for each line
        lines = poly.GetLines()
        n_lines = lines.GetNumberOfCells()
        strain = np.zeros(n_lines)
        vtk_strain = vtk.vtkFloatArray()
        vtk_strain.SetName("Strain")
        lines.InitTraversal()
        for i in range(n_lines):
            ids = vtk.vtkIdList()
            lines.GetNextCell(ids)
            p1 = np.array(points.GetPoint(ids.GetId(0)))
            p2 = np.array(points.GetPoint(ids.GetId(1)))
            d = np.linalg.norm(p1 - p2)

            # Strain of this line (unit: %)
            strain[i] = (d - length_ED[i]) / length_ED[i] * 100
            vtk_strain.InsertNextTuple1(strain[i])

        # Save the strain array to the vtk file
        poly.GetCellData().AddArray(vtk_strain)
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(poly)
        filename_strain = f"{result_dir}/strain_la/strain_fr{fr:02d}.vtk"
        writer.SetFileName(filename_strain)
        writer.Write()
        os.system('sed -i "1s/4.1/4.0/" {0}'.format(filename_strain))

        # Calculate the segmental and global strains
        for i in range(6):
            table_strain["longit"][i, fr] = np.mean(strain[(seg_id == (i + 1)) & (dir_id == 3)])
        table_strain["longit"][-1, fr] = np.mean(strain[dir_id == 3])  # global

    return table_strain["longit"]


def cine_2d_la_motion_and_strain_analysis(data_dir, par_config_name, temp_dir, result_dir, eval_dice=False):
    """
    Perform motion tracking and strain analysis for long-axis cine MR images.
    """
    # Make folders to arrange files
    os.makedirs(f"{temp_dir}/la", exist_ok=True)
    os.makedirs(f"{temp_dir}/seg_la", exist_ok=True)
    os.makedirs(f"{temp_dir}/myo_contour", exist_ok=True)
    os.makedirs(f"{temp_dir}/pair", exist_ok=True)
    os.makedirs(f"{temp_dir}/forward", exist_ok=True)
    os.makedirs(f"{temp_dir}/backward", exist_ok=True)
    os.makedirs(f"{temp_dir}/combined", exist_ok=True)
    os.makedirs(f"{result_dir}/myo_contour_la", exist_ok=True)
    os.makedirs(f"{result_dir}/doc", exist_ok=True)
    mirtk_log_file = f"{result_dir}/log/mirtk_la.log"

    # Crop the image to save computation for image registration
    # Focus on the left ventricle so that motion tracking is less affected by
    # the movement of RV and LV outflow tract
    padding(
        f"{data_dir}/seg4_la_4ch_ED.nii.gz",
        f"{data_dir}/seg4_la_4ch_ED.nii.gz",
        f"{temp_dir}/seg_la/seg4_la_4ch_LV_ED.nii.gz",
        2,  # set all pixels with value 2 (Myo) to 1, it will not influence evaluation of Dice.
        1,
    )
    padding(
        f"{temp_dir}/seg_la/seg4_la_4ch_LV_ED.nii.gz",
        f"{temp_dir}/seg_la/seg4_la_4ch_LV_ED.nii.gz",
        f"{temp_dir}/seg_la/seg4_la_4ch_LV_ED.nii.gz",
        3,  # set all pixels with value 3 (RV) to 0 (BG)
        0,
    )
    padding(
        f"{temp_dir}/seg_la/seg4_la_4ch_LV_ED.nii.gz",
        f"{temp_dir}/seg_la/seg4_la_4ch_LV_ED.nii.gz",
        f"{temp_dir}/seg_la/seg4_la_4ch_LV_ED.nii.gz",
        4,  # set all pixels with value 4 (LA) to 0 (BG)
        0,
    )
    padding(
        f"{temp_dir}/seg_la/seg4_la_4ch_LV_ED.nii.gz",
        f"{temp_dir}/seg_la/seg4_la_4ch_LV_ED.nii.gz",
        f"{temp_dir}/seg_la/seg4_la_4ch_LV_ED.nii.gz",
        5,  # set all pixels with value 5 (RV) to 0 (BG)
        0,
    )

    # Crop the image to save computation for image registration
    auto_crop_image(f"{temp_dir}/seg_la/seg4_la_4ch_LV_ED.nii.gz", f"{temp_dir}/seg_la/seg4_la_4ch_LV_crop_ED.nii.gz", 20)

    # Transform la/seg_la.nii.gz to la/seg_la_crop.nii.gz using seg_la_LV_crop_ED.nii.gz
    os.system(
        f"mirtk transform-image {data_dir}/la_4ch.nii.gz {temp_dir}/la/la_4ch_crop.nii.gz "
        f"-target {temp_dir}/seg_la/seg4_la_4ch_LV_crop_ED.nii.gz"
    )
    os.system(
        f"mirtk transform-image {data_dir}/seg4_la_4ch.nii.gz {temp_dir}/seg_la/seg4_la_4ch_crop.nii.gz "
        f"-target {temp_dir}/seg_la/seg4_la_4ch_LV_crop_ED.nii.gz"
    )

    nim = nib.load(f"{temp_dir}/la/la_4ch_crop.nii.gz")
    # For long axis, we don't need to care about slice
    T = nim.header["dim"][4]
    dt = nim.header["pixdim"][4]

    split_sequence(f"{temp_dir}/la/la_4ch_crop.nii.gz", f"{temp_dir}/la/la_4ch_crop_fr")

    # Extract the myocardial contours for the slices
    extract_la_myocardial_contour(
        f"{data_dir}/seg4_la_4ch_ED.nii.gz",
        f"{data_dir}/seg_sa_ED.nii.gz",  # requires SA image to determine coordinate system
        f"{temp_dir}/myo_contour/la_4ch_myo_contour_ED.vtk",
    )
    logger.info("Myocardial contour at ED extracted")

    logger.info("Forward registration")
    # * Image registration between successive time frames for forward image registration
    for fr in range(1, T):
        target_fr = fr - 1
        source_fr = fr
        target = f"{temp_dir}/la/la_4ch_crop_fr{target_fr:02d}.nii.gz"
        source = f"{temp_dir}/la/la_4ch_crop_fr{source_fr:02d}.nii.gz"
        dof = f"{temp_dir}/pair/ffd_la_4ch_pair_{target_fr:02d}_to_{source_fr:02d}.dof.gz"
        os.system(f"mirtk register {target} {source} -parin {par_config_name} -dofout {dof} >> {mirtk_log_file} 2>&1")

    # * Start forward image registration
    os.system(f"cp {temp_dir}/pair/ffd_la_4ch_pair_00_to_01.dof.gz " f"{temp_dir}/forward/ffd_la_4ch_forward_00_to_01.dof.gz")
    for fr in range(2, T):
        dofs = ""
        for k in range(1, fr + 1):
            dof = f"{temp_dir}/pair/ffd_la_4ch_pair_{k - 1:02d}_to_{k:02d}.dof.gz"
            dofs += dof + " "
        dof_out = f"{temp_dir}/forward/ffd_la_4ch_forward_00_to_{fr:02d}.dof.gz"
        os.system(f"mirtk compose-dofs {dofs} {dof_out}")

    logger.info("Backward registration")
    # * Image registration reversely between successive time frames for backward image registration
    for fr in range(T - 1, 0, -1):
        target_fr = (fr + 1) % T
        source_fr = fr
        target = f"{temp_dir}/la/la_4ch_crop_fr{target_fr:02d}.nii.gz"
        source = f"{temp_dir}/la/la_4ch_crop_fr{source_fr:02d}.nii.gz"
        dof = f"{temp_dir}/pair/ffd_la_4ch_pair_{target_fr:02d}_to_{source_fr:02d}.dof.gz"
        os.system(f"mirtk register {target} {source} -parin {par_config_name} -dofout {dof} >> {mirtk_log_file} 2>&1")

    # For the first and last time frame, directly copy the transformation
    os.system(
        f"cp {temp_dir}/pair/ffd_la_4ch_pair_00_to_{(T - 1):02d}.dof.gz "
        f"{temp_dir}/backward/ffd_la_4ch_backward_00_to_{(T - 1):02d}.dof.gz"
    )
    # For the rest time frames, compose the transformation fields
    for fr in range(T - 2, 0, -1):
        dofs = ""
        for k in range(T - 1, fr - 1, -1):
            dof = f"{temp_dir}/pair/ffd_la_4ch_pair_{(k + 1) % T:02d}_to_{k:02d}.dof.gz"
            dofs += dof + " "
        dof_out = f"{temp_dir}/backward/ffd_la_4ch_backward_00_to_{fr:02d}.dof.gz"
        os.system(f"mirtk compose-dofs {dofs} {dof_out}")

    logger.info("Combine the forward and backward transformations")
    # Average the forward and backward transformations
    os.system(f"mirtk init-dof {temp_dir}/forward/ffd_la_4ch_forward_00_to_00.dof.gz")
    os.system(f"mirtk init-dof {temp_dir}/backward/ffd_la_4ch_backward_00_to_00.dof.gz")
    os.system(f"mirtk init-dof {temp_dir}/combined/ffd_la_4ch_00_to_00.dof.gz")

    # todo: Similar to short axis, currently we directly copy the transformations.
    # for fr in range(1, T):
    #     dof_forward = "{0}/ffd_la_4ch_forward_00_to_{1:02d}.dof.gz".format(output_dir, fr)
    #     weight_forward = float(T - fr) / T
    #     dof_backward = "{0}/ffd_la_4ch_backward_00_to_{1:02d}.dof.gz".format(output_dir, fr)
    #     weight_backward = float(fr) / T
    #     dof_combine = "{0}/ffd_la_4ch_00_to_{1:02d}.dof.gz".format(output_dir, fr)
    #     os.system(
    #         "average_3d_ffd 2 {0} {1} {2} {3} {4}".format(
    #             dof_forward, weight_forward, dof_backward, weight_backward, dof_combine
    #         )
    #     )

    for fr in range(1, T):
        dof_forward = f"{temp_dir}/forward/ffd_la_4ch_forward_00_to_{fr:02d}.dof.gz"
        dof_backward = f"{temp_dir}/backward/ffd_la_4ch_backward_00_to_{fr:02d}.dof.gz"
        # combined transformation to be created
        dof_combine = f"{temp_dir}/combined/ffd_la_4ch_00_to_{fr:02d}.dof.gz"

        if fr > T // 2:
            os.system(f"cp {dof_forward} {dof_combine}")
        else:
            os.system(f"cp {dof_backward} {dof_combine}")

    # Transform the contours and calculate the strain
    for fr in range(0, T):
        os.system(
            f"mirtk transform-points {temp_dir}/myo_contour/la_4ch_myo_contour_ED.vtk "
            f"{result_dir}/myo_contour_la/la_4ch_myo_contour_fr{fr:02d}.vtk "
            f"-dofin {temp_dir}/combined/ffd_la_4ch_00_to_{fr:02d}.dof.gz"
        )

    if eval_dice:
        dice_lv_myo = []
        image_names = []

        split_sequence(f"{temp_dir}/seg_la/seg4_la_4ch_crop.nii.gz", f"{temp_dir}/seg_la/seg4_la_4ch_crop_fr")

        for fr in range(0, T):
            os.system(
                f"mirtk transform-image {temp_dir}/seg_la/seg4_la_4ch_crop_fr{fr:02d}.nii.gz "
                f"{temp_dir}/seg_la/seg4_la_4ch_crop_warp_ffd_fr{fr:02d}.nii.gz "
                f"-dofin {temp_dir}/combined/ffd_la_4ch_00_to_{fr:02d}.dof.gz "
                f"-target {temp_dir}/seg_la/seg4_la_4ch_crop_fr00.nii.gz"
            )
            image_A = nib.load(f"{temp_dir}/seg_la/seg4_la_4ch_crop_fr00.nii.gz").get_fdata()
            image_B = nib.load(f"{temp_dir}/seg_la/seg4_la_4ch_crop_warp_ffd_fr{fr:02d}.nii.gz").get_fdata()
            image_B = image_B[:, :, :, 0]
            nim_B = nib.load(f"{temp_dir}/seg_la/seg4_la_4ch_crop_warp_ffd_fr{fr:02d}.nii.gz")
            nim_B_new = nib.Nifti1Image(image_B, nim_B.affine, nim_B.header)
            nib.save(nim_B_new, f"{temp_dir}/seg_la/seg4_la_4ch_crop_warp_ffd_fr{fr:02d}.nii.gz")
            dice_lv_myo.append([np_categorical_dice(image_A, image_B, 1), np_categorical_dice(image_A, image_B, 2)])
            image_names.append(f"{temp_dir}/seg_la/seg4_la_4ch_crop_warp_ffd_fr{fr:02d}.nii.gz")

        la_warp_combined_name = f"{temp_dir}/seg_la/seg4_la_4ch_crop_warp_ffd.nii.gz"
        make_sequence(image_names, dt, la_warp_combined_name)

        df_dice = pd.DataFrame(dice_lv_myo)
        df_dice.columns = ["LV", "Myo"]
        df_dice.index = [f"Frame {i}" for i in range(T)]
        # append mean result at the end
        df_dice.loc["mean"] = df_dice.mean()
        logger.info(f"Mean Dice for LV: {df_dice.loc['mean'].values[0]}")
        logger.info(f"Mean Dice for Myo: {df_dice.loc['mean'].values[1]}")
        df_dice.to_csv(f"{result_dir}/doc/dice_la_4ch_warp_ffd.csv", index=True, header=True)

    return


def plot_bulls_eye(data, vmin, vmax, cmap="Reds", color_line="black"):
    """
    Plot the bull's eye plot.
    For an example of Bull's eye plot, refer to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4862218/pdf/40001_2016_Article_216.pdf.
    data: values for 16 segments
    """
    if len(data) != 16:
        logger.error("Error: len(data) != 16!")
        exit(1)

    # The cartesian coordinate and the polar coordinate
    x = np.linspace(-1, 1, 201)
    y = np.linspace(-1, 1, 201)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx * xx + yy * yy)
    theta = np.degrees(np.arctan2(yy, xx))

    # The radius and degree for each segment
    R1, R2, R3, R4 = 1, 0.65, 0.3, 0.0
    rad_deg = {
        1: (R1, R2, 60, 120),
        2: (R1, R2, 120, 180),
        3: (R1, R2, -180, -120),
        4: (R1, R2, -120, -60),
        5: (R1, R2, -60, 0),
        6: (R1, R2, 0, 60),
        7: (R2, R3, 60, 120),
        8: (R2, R3, 120, 180),
        9: (R2, R3, -180, -120),
        10: (R2, R3, -120, -60),
        11: (R2, R3, -60, 0),
        12: (R2, R3, 0, 60),
        13: (R3, R4, 45, 135),
        14: (R3, R4, 135, -135),
        15: (R3, R4, -135, -45),
        16: (R3, R4, -45, 45),
    }

    # Plot the segments
    canvas = np.zeros(xx.shape)
    cx, cy = (np.array(xx.shape) - 1) / 2
    sz = cx

    for i in range(1, 17):
        val = data[i - 1]
        r1, r2, theta1, theta2 = rad_deg[i]
        if theta2 > theta1:
            mask = ((r < r1) & (r >= r2)) & ((theta >= theta1) & (theta < theta2))
        else:
            mask = ((r < r1) & (r >= r2)) & ((theta >= theta1) | (theta < theta2))
        canvas[mask] = val
    plt.imshow(canvas, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.axis("off")
    plt.gca().invert_yaxis()  # gca(): get current axes

    # Plot the circles
    for r in [R1, R2, R3]:
        deg = np.linspace(0, 2 * np.pi, 201)
        circle_x = cx + sz * r * np.cos(deg)
        circle_y = cy + sz * r * np.sin(deg)
        plt.plot(circle_x, circle_y, color=color_line)

    # Plot the lines between segments
    for i in range(1, 17):
        r1, r2, theta1, theta2 = rad_deg[i]
        line_x = cx + sz * np.array([r1, r2]) * np.cos(np.radians(theta1))
        line_y = cy + sz * np.array([r1, r2]) * np.sin(np.radians(theta1))
        plt.plot(line_x, line_y, color=color_line)

    # Plot the indicator for RV insertion points
    for i in [2, 4]:
        r1, r2, theta1, theta2 = rad_deg[i]
        x = cx + sz * r1 * np.cos(np.radians(theta1))
        y = cy + sz * r1 * np.sin(np.radians(theta1))
        plt.plot([x, x - sz * 0.2], [y, y], color=color_line)


def evaluate_ventricular_length_sax(label_sa: Tuple[float, float, float], nim_sa: nib.nifti1.Nifti1Image, long_axis, short_axis):
    """
    Evaluate the ventricular length from short-axis view images.
    """

    if label_sa.ndim != 3:
        raise ValueError("The label_sa should be a 3D image.")

    # End-diastolic diameter should be measured at the plane which contained the basal papillary muscle
    basal_slice = determine_sa_basal_slice(label_sa)

    label_sa = label_sa[:, :, basal_slice]

    # Go through the label class
    L = []
    landmarks = []

    lab = 1  # We are only interested in left ventricle
    # The binary label map
    label_i = label_sa == lab

    # Get the largest component in case we have a bad segmentation
    label_i = get_largest_cc(label_i)

    _, _, image_line, image_line_minor = determine_axes(label_i, nim_sa, long_axis)

    points_line = np.nonzero(image_line)

    points = []
    points_image = []  # for easier visualization
    for j in range(len(points_line[0])):
        x = points_line[0][j]
        y = points_line[1][j]
        # World coordinate
        point = np.dot(nim_sa.affine, np.array([x, y, 0, 1]))[:3]
        # Distance along the long-axis
        points += [np.append(point, np.dot(point, long_axis))]  # (x,y,distance)
        points_image += [np.append((x, y), np.dot(point, long_axis))]
    points = np.array(points)
    points_image = np.array(points_image)
    if len(points) == 0:
        raise ValueError("No intersection points found in the ventricle.")
    points = points[points[:, 3].argsort(), :3]  # sort by the distance along the long-axis
    points_image = points_image[points_image[:, 2].argsort(), :2]

    # Landmarks of the intersection points are the top and bottom points along points_line
    landmarks += [points_image[0]]
    landmarks += [points_image[-1]]
    # Longitudinal diameter; Unit: cm
    L += [np.linalg.norm(points[-1] - points[0]) * 1e-1]

    points_line_minor = np.nonzero(image_line_minor)
    points_minor = []
    points_minor_image = []
    for j in range(len(points_line_minor[0])):
        x = points_line_minor[0][j]
        y = points_line_minor[1][j]
        # World coordinate
        point = np.dot(nim_sa.affine, np.array([x, y, 0, 1]))[:3]
        # Distance along the short axis
        points_minor += [np.append(point, np.dot(point, short_axis))]
        points_minor_image += [np.append((x, y), np.dot(point, short_axis))]
    points_minor = np.array(points_minor)
    points_minor_image = np.array(points_minor_image)
    points_minor = points_minor[points_minor[:, 3].argsort(), :3]  # sort by the distance along the short-axis
    points_minor_image = points_minor_image[points_minor_image[:, 2].argsort(), :2]

    landmarks += [points_minor_image[0]]
    landmarks += [points_minor_image[-1]]
    # Transverse diameter; Unit: cm
    L += [np.linalg.norm(points_minor[-1] - points_minor[0]) * 1e-1]

    return np.max(L), landmarks


def evaluate_ventricular_length_lax(label_la_seg4: Tuple[float, float], nim_la: nib.nifti1.Nifti1Image, long_axis, short_axis):
    """
    Evaluate the ventricle length from 4 chamber view image.
    """

    if label_la_seg4.ndim != 2:
        raise ValueError("The label_la should be a 2D image.")
    if len(np.unique(label_la_seg4)) != 6:
        raise ValueError("The label_la should have segmentation for all four chambers.")

    # Go through the label class
    landmarks = []

    labels = {"BG": 0, "LV": 1, "Myo": 2, "RV": 3, "LA": 4, "RA": 5}
    label_LV = label_la_seg4 == labels["LV"]

    # Get the largest component in case we have a bad segmentation
    label_LV = get_largest_cc(label_LV)

    _, _, image_line, _ = determine_axes(label_LV, nim_la, long_axis)

    points_line = np.nonzero(image_line)
    points = []
    points_image = []  # for easier visualization
    for j in range(len(points_line[0])):
        x = points_line[0][j]
        y = points_line[1][j]
        # World coordinate
        point = np.dot(nim_la.affine, np.array([x, y, 0, 1]))[:3]
        # Distance along the long-axis
        points += [np.append(point, np.dot(point, long_axis))]
        points_image += [np.append((x, y), np.dot(point, long_axis))]
    points = np.array(points)
    points_image = np.array(points_image)
    if len(points) == 0:
        raise ValueError("No intersection points found in the ventricle.")
    points = points[points[:, 3].argsort(), :3]  # sort by the distance along the long-axis
    points_image = points_image[points_image[:, 2].argsort(), :2]

    # Landmarks of the intersection points are the top and bottom points along points_line
    landmarks += [points_image[0]]
    landmarks += [points_image[-1]]
    # Longitudinal diameter; Unit: cm
    L = np.linalg.norm(points[-1] - points[0]) * 1e-1  # Here we have already applied nim.affine, no need to multiply

    return np.max(L), landmarks


def evaluate_atrial_area_length(label_la: Tuple[float, float], nim_la: nib.nifti1.Nifti1Image, long_axis, short_axis):
    """
    Evaluate the atrial area and length from 2 chamber or 4 chamber view images.
    """

    if label_la.ndim != 2:
        raise ValueError("The label_la should be a 2D image.")

    # Area per pixel
    pixdim = nim_la.header["pixdim"][1:4]
    area_per_pix = pixdim[0] * pixdim[1] * 1e-2  # Unit: cm^2

    # Go through the label class
    L = []
    A = []
    landmarks = []
    labels = np.sort(list(set(np.unique(label_la)) - set([0])))
    for i in labels:
        # The binary label map
        label_i = label_la == i

        # Get the largest component in case we have a bad segmentation
        label_i = get_largest_cc(label_i)

        # Calculate the area
        A += [np.sum(label_i) * area_per_pix]

        _, _, image_line, image_line_minor = determine_axes(label_i, nim_la, long_axis)

        points_line = np.nonzero(image_line)
        points = []
        points_image = []  # for easier visualization
        for j in range(len(points_line[0])):
            x = points_line[0][j]
            y = points_line[1][j]
            # World coordinate
            point = np.dot(nim_la.affine, np.array([x, y, 0, 1]))[:3]
            # Distance along the long-axis
            points += [np.append(point, np.dot(point, long_axis))]
            points_image += [np.append((x, y), np.dot(point, long_axis))]
        points = np.array(points)
        points_image = np.array(points_image)
        if len(points) == 0:
            raise ValueError("No intersection points found in the atrium.")
        points = points[points[:, 3].argsort(), :3]  # sort by the distance along the long-axis
        points_image = points_image[points_image[:, 2].argsort(), :2]

        # Landmarks of the intersection points are the top and bottom points along points_line
        landmarks += [points_image[0]]
        landmarks += [points_image[-1]]
        # Longitudinal diameter; Unit: cm
        L += [np.linalg.norm(points[-1] - points[0]) * 1e-1]  # Here we have already applied nim.affine, no need to multiply

        # Define Transverse diameter is obtained perpendicular to longitudinal diameter, at the mid level of atrium
        # Ref https://jcmr-online.biomedcentral.com/articles/10.1186/1532-429X-15-29 for example in right atrium

        points_line_minor = np.nonzero(image_line_minor)
        points_minor = []
        points_minor_image = []  # for easier visualization
        for j in range(len(points_line_minor[0])):
            x = points_line_minor[0][j]
            y = points_line_minor[1][j]
            # World coordinate
            point = np.dot(nim_la.affine, np.array([x, y, 0, 1]))[:3]
            # Distance along the short-axis
            points_minor += [np.append(point, np.dot(point, short_axis))]
            points_minor_image += [np.append((x, y), np.dot(point, short_axis))]
        points_minor = np.array(points_minor)
        points_minor_image = np.array(points_minor_image)
        points_minor = points_minor[points_minor[:, 3].argsort(), :3]  # sort by the distance along the short-axis
        points_minor_image = points_minor_image[points_minor_image[:, 2].argsort(), :2]
        landmarks += [points_minor_image[0]]
        landmarks += [points_minor_image[-1]]
        # Transverse diameter; Unit: cm
        L += [np.linalg.norm(points_minor[-1] - points_minor[0]) * 1e-1]

    return A, L, landmarks


def evaluate_valve_diameter(
    label_la_seg4: Tuple[float, float, float, float],
    nim_la: nib.nifti1.Nifti1Image,
    t,
    display=False,
):
    """
    Calculate the mitral valve and tricuspid valve diameters at end-diastole and end-systole using 4 chamber long axis images.
    """
    # Similar setting as evaluate_AVPD()
    if label_la_seg4.ndim != 4:
        raise ValueError("The label_la should be a 4D image.")
    if len(np.unique(label_la_seg4)) != 6:
        raise ValueError("The label_la_seg4 should have segmentation for all four chambers.")

    label_t = label_la_seg4[:, :, 0, t]

    # * We use the average of ventricles and atriums
    LV_lm, LA_lm, RV_lm, RA_lm = determine_valve_landmark(label_t)

    LV_lm_real = list(map(lambda x: np.dot(nim_la.affine, np.array([x[0], x[1], 0, 1]))[:3], LV_lm))
    LA_lm_real = list(map(lambda x: np.dot(nim_la.affine, np.array([x[0], x[1], 0, 1]))[:3], LA_lm))
    RV_lm_real = list(map(lambda x: np.dot(nim_la.affine, np.array([x[0], x[1], 0, 1]))[:3], RV_lm))
    RA_lm_real = list(map(lambda x: np.dot(nim_la.affine, np.array([x[0], x[1], 0, 1]))[:3], RA_lm))

    TA_diameters = {"ventricle": None, "atrium": None, "average": None, "min": None}
    MA_diameters = {"ventricle": None, "atrium": None, "average": None, "min": None}

    # unit: cm
    TA_diameters["ventricle"] = np.linalg.norm(LV_lm_real[0] - LV_lm_real[1]) * 1e-1
    TA_diameters["atrium"] = np.linalg.norm(LA_lm_real[0] - LA_lm_real[1]) * 1e-1
    TA_diameters["average"] = (TA_diameters["ventricle"] + TA_diameters["atrium"]) / 2
    TA_diameters["min"] = min(TA_diameters["ventricle"], TA_diameters["atrium"])
    MA_diameters["ventricle"] = np.linalg.norm(RV_lm_real[0] - RV_lm_real[1]) * 1e-1
    MA_diameters["atrium"] = np.linalg.norm(RA_lm_real[0] - RA_lm_real[1]) * 1e-1
    MA_diameters["average"] = (MA_diameters["ventricle"] + MA_diameters["atrium"]) / 2
    MA_diameters["min"] = min(MA_diameters["ventricle"], MA_diameters["atrium"])

    if display is True:
        fig = plt.figure(figsize=(9, 6))
        plt.imshow(label_t, cmap="gray")
        plt.title(f"Valve Landmarks (Timeframe = {t})")
        plt.scatter(*zip(*LV_lm), c="red", s=8, label="Landmark 1 (Mitral)")
        plt.scatter(*zip(*LA_lm), c="orange", s=8, label="Landmark 2 (Mitral)")
        plt.scatter(*zip(*RV_lm), c="blue", s=8, label="Landmark 3 (Tricuspid)")
        plt.scatter(*zip(*RA_lm), c="cyan", s=8, label="Landmark 4 (Tricuspid)")
        plt.plot([LV_lm[0][0], LV_lm[1][0]], [LV_lm[0][1], LV_lm[1][1]], c="gray", linewidth=1.5)
        plt.plot([LA_lm[0][0], LA_lm[1][0]], [LA_lm[0][1], LA_lm[1][1]], c="gray", linestyle="--", linewidth=1.5)
        plt.plot([RV_lm[0][0], RV_lm[1][0]], [RV_lm[0][1], RV_lm[1][1]], c="gold", linewidth=1.5)
        plt.plot([RA_lm[0][0], RA_lm[1][0]], [RA_lm[0][1], RA_lm[1][1]], c="gold", linestyle="--", linewidth=1.5)
        plt.legend(loc="lower right")

        return TA_diameters, MA_diameters, fig
    else:
        return TA_diameters, MA_diameters


def evaluate_AVPD(
    label_la_seg4: Tuple[float, float, float, float],
    nim_la: nib.nifti1.Nifti1Image,
    t_ED,
    t_ES,
    display=False,
):
    """
    Determine the atrioventricular plane displacement (AVPD).
    Since the ventricle and atrium are segmented separately in `seg_la_2ch` and `seg_la_4ch` files,
    only `seg4_la_4ch` files should be used. The result is then the AVPD of anterolateral wall
    (https://journals.physiology.org/doi/epdf/10.1152/ajpheart.01148.2006).
    """

    # * Currently, we follow https://jcmr-online.biomedcentral.com/articles/10.1186/s12968-020-00683-3
    # * only consider AVPD of LV at ED and ES. For time series of AVPD, feature tracking would be better

    if label_la_seg4.ndim != 4:
        raise ValueError("The label_la should be a 4D image.")
    if len(np.unique(label_la_seg4)) != 6:
        raise ValueError("The label_la_seg4 should have segmentation for all four chambers.")

    label_ED = label_la_seg4[:, :, 0, t_ED]
    label_ES = label_la_seg4[:, :, 0, t_ES]

    AV = {"ventricle": {}, "atrium": {}}
    AV_displacement = np.zeros(4)

    LV_lm_ED, LA_lm_ED, RV_lm_ED, RA_lm_ED = determine_valve_landmark(label_ED)
    LV_lm_ES, LA_lm_ES, RV_lm_ES, RA_lm_ES = determine_valve_landmark(label_ES)

    lm_all = [LV_lm_ED, LA_lm_ED, RV_lm_ED, RA_lm_ED, LV_lm_ES, LA_lm_ES, RV_lm_ES, RA_lm_ES]
    # For some cases, we will fail to detemine landmark
    if any(lm is None for lm in lm_all):
        raise ValueError("Some landmarks are failed to be determined and thus missing.")

    # * We choose the pair that has the largest distance for ventricle and atrium respectively
    ventricle_lm_ED = np.vstack([LV_lm_ED, RV_lm_ED])
    ventricle_lm_ED_dist = cdist(ventricle_lm_ED, ventricle_lm_ED)
    i, j = np.unravel_index(np.argmax(ventricle_lm_ED_dist), ventricle_lm_ED_dist.shape)
    ventricle_lm_ED = [ventricle_lm_ED[i], ventricle_lm_ED[j]]
    ventricle_lm_ED_real = list(map(lambda x: np.dot(nim_la.affine, np.array([x[0], x[1], 0, 1]))[:3], ventricle_lm_ED))

    ventricle_lm_ES = np.vstack([LV_lm_ES, RV_lm_ES])
    ventricle_lm_ES_dist = cdist(ventricle_lm_ES, ventricle_lm_ES)
    i, j = np.unravel_index(np.argmax(ventricle_lm_ES_dist), ventricle_lm_ES_dist.shape)
    ventricle_lm_ES = [ventricle_lm_ES[i], ventricle_lm_ES[j]]
    # sort ventricle_lm_ES according to ventricle_lm_ED
    dist1 = np.linalg.norm(ventricle_lm_ES[0] - ventricle_lm_ED[0])
    dist2 = np.linalg.norm(ventricle_lm_ES[1] - ventricle_lm_ED[0])
    if dist1 > dist2:
        ventricle_lm_ES = [ventricle_lm_ES[1], ventricle_lm_ES[0]]
    ventricle_lm_ES_real = list(map(lambda x: np.dot(nim_la.affine, np.array([x[0], x[1], 0, 1]))[:3], ventricle_lm_ES))

    AV["ventricle"]["ED"] = ventricle_lm_ED_real
    AV["ventricle"]["ES"] = ventricle_lm_ES_real

    atrium_lm_ED = np.vstack([LA_lm_ED, RA_lm_ED])
    atrium_lm_ED_dist = cdist(atrium_lm_ED, atrium_lm_ED)
    i, j = np.unravel_index(np.argmax(atrium_lm_ED_dist), atrium_lm_ED_dist.shape)
    atrium_lm_ED = [atrium_lm_ED[i], atrium_lm_ED[j]]
    atrium_lm_ED_real = list(map(lambda x: np.dot(nim_la.affine, np.array([x[0], x[1], 0, 1]))[:3], atrium_lm_ED))

    atrium_lm_ES = np.vstack([LA_lm_ES, RA_lm_ES])
    atrium_lm_ES_dist = cdist(atrium_lm_ES, atrium_lm_ES)
    i, j = np.unravel_index(np.argmax(atrium_lm_ES_dist), atrium_lm_ES_dist.shape)
    atrium_lm_ES = [atrium_lm_ES[i], atrium_lm_ES[j]]
    # sort atrium_lm_ES according to atrium_lm_ED
    dist1 = np.linalg.norm(atrium_lm_ES[0] - atrium_lm_ED[0])
    dist2 = np.linalg.norm(atrium_lm_ES[1] - atrium_lm_ED[0])
    if dist1 > dist2:
        atrium_lm_ES = [atrium_lm_ES[1], atrium_lm_ES[0]]
    atrium_lm_ES_real = list(map(lambda x: np.dot(nim_la.affine, np.array([x[0], x[1], 0, 1]))[:3], atrium_lm_ES))

    AV["atrium"]["ED"] = atrium_lm_ED_real
    AV["atrium"]["ES"] = atrium_lm_ES_real

    AV_displacement[0] = np.linalg.norm(np.array(AV["ventricle"]["ED"][0]) - np.array(AV["ventricle"]["ES"][0]))
    AV_displacement[1] = np.linalg.norm(np.array(AV["ventricle"]["ED"][1]) - np.array(AV["ventricle"]["ES"][1]))
    AV_displacement[2] = np.linalg.norm(np.array(AV["atrium"]["ED"][0]) - np.array(AV["atrium"]["ES"][0]))
    AV_displacement[3] = np.linalg.norm(np.array(AV["atrium"]["ED"][1]) - np.array(AV["atrium"]["ES"][1]))

    # We exclude the minimum and larges, then take the mean

    AV_displacement = np.sort(AV_displacement)
    AVPD = np.mean(AV_displacement[1:-1])

    if display is True:
        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Landmarks at ED")
        plt.imshow(label_ED, cmap="gray")
        plt.scatter(*zip(*ventricle_lm_ED), c="red", s=8, label="Landmarks (Ventricle)")
        plt.scatter(*zip(*atrium_lm_ED), c="blue", s=8, label="Landmarks (Atrium)")
        plt.legend(loc="lower right")

        plt.subplot(1, 2, 2)
        plt.title("Landmarks at ES")
        plt.imshow(label_ES, cmap="gray")
        plt.scatter(*zip(*ventricle_lm_ES), c="red", s=8, label="Landmarks (Ventricle)")
        plt.scatter(*zip(*atrium_lm_ES), c="blue", s=8, label="Landmarks (Atrium)")
        # overlay landmarks at ED
        plt.scatter(*zip(*ventricle_lm_ED), s=8, edgecolors="red", facecolors="none")
        plt.scatter(*zip(*atrium_lm_ED), s=8, edgecolors="blue", facecolors="none")
        # connect landmarks at ED and ES
        for i in range(2):
            plt.plot(
                [ventricle_lm_ES[i][0], ventricle_lm_ED[i][0]],
                [ventricle_lm_ES[i][1], ventricle_lm_ED[i][1]],
                color="red",
                linestyle="--",
            )

            plt.plot(
                [atrium_lm_ES[i][0], atrium_lm_ED[i][0]], [atrium_lm_ES[i][1], atrium_lm_ED[i][1]], color="blue", linestyle="--"
            )

        plt.legend(loc="lower right")

        return (AVPD * 1e-1, fig)  # unit: cm
    else:
        return AVPD * 1e-1  # unit: cm


def evaluate_radius_thickness(seg_sa_s_t: Tuple[float, float], nim_sa: nib.nifti1.Nifti1Image, BSA_value: float):
    if seg_sa_s_t.ndim != 2:
        raise ValueError("The seg_sa should be a 2D image.")
    NUM_SEGMENTS = 6

    seg_z = seg_sa_s_t.astype(np.uint8)

    labels = {"BG": 0, "LV": 1, "Myo": 2, "RV": 3}
    # * The segmentation masks are divided into 6 segmeents, depending on which interval [ k*pi/3, (k+1)*pi/3 )
    # * the angle between vectors B_LP and B_LB_R is

    seg_z_LV = seg_z == labels["LV"]
    seg_z_RV = seg_z == labels["RV"]
    seg_z_Myo = seg_z == labels["Myo"]

    # np.argwhere is similar to np.nonzero, except the format of returns
    barycenter_LV = np.mean(np.argwhere(seg_z_LV), axis=0)
    barycenter_RV = np.mean(np.argwhere(seg_z_RV), axis=0)
    # plt.imshow(seg_Z_RV, cmap="gray")
    # plt.scatter(barycenter_RV[1], barycenter_RV[0], c="r")

    myo_points = np.argwhere(seg_z_Myo)
    angles = np.zeros(myo_points.shape[0])
    zones = np.zeros(myo_points.shape[0])
    boundaries = np.zeros(myo_points.shape[0])  # -1: inner boundary, 1: outer boundary

    for i in range(myo_points.shape[0]):
        myo_point = myo_points[i]
        vector1 = myo_point - barycenter_LV
        vector1 = vector1 / np.linalg.norm(vector1)
        vector2 = barycenter_RV - barycenter_LV
        vector2 = vector2 / np.linalg.norm(vector2)
        angle = np.arccos(np.dot(vector1, vector2))
        if np.cross(vector1, vector2) < 0:
            angle = 2 * np.pi - angle
        angles[i] = angle
        zones[i] = int(angle // (np.pi / 3)) + 1  # 6 segments

        neighbors = []
        for j in range(-1, 2):
            for k in range(-1, 2):
                if abs(j) == abs(k):
                    continue
                if (
                    myo_point[0] + j < 0
                    or myo_point[0] + j >= seg_z.shape[0]
                    or myo_point[1] + k < 0
                    or myo_point[1] + k >= seg_z.shape[1]
                ):
                    continue
                neighbors.append(seg_z[myo_point[0] + j, myo_point[1] + k])
        if labels["LV"] in neighbors:
            boundaries[i] = -1
        elif (labels["BG"] in neighbors) or (labels["RV"] in neighbors):
            boundaries[i] = 1

    # plt.figure(figsize=(20, 10))
    # plt.subplot(1,2,1)
    # plt.imshow(seg_z, cmap="gray")
    # plt.scatter(myo_points[:, 1], myo_points[:, 0], c=zones)
    # plt.subplot(1,2,2)
    # plt.imshow(seg_z, cmap="gray")
    # plt.scatter(myo_points[:, 1], myo_points[:, 0], c=boundaries, s=0.5)

    barycenter_segments_inner = []  # define barycenter of inner boundary of segment
    barycenter_segments_outer = []  # define barycenter of outer boundary of segment

    # seg_z_segments = np.ascontiguousarray(np.zeros_like(seg_z))
    # seg_z_segments[seg_z_Myo] = zones
    # plt.imshow(seg_z_segments, cmap='gray')

    for i in range(1, NUM_SEGMENTS + 1):
        segment_points = myo_points[zones == i]
        segment_boundary = boundaries[zones == i]
        segment_inner_barycenter = np.mean(segment_points[segment_boundary == -1], axis=0)
        segment_outer_barycenter = np.mean(segment_points[segment_boundary == 1], axis=0)

        # seg_z_segment = seg_z_segments == i
        # plt.figure(figsize=(20, 10))
        # plt.imshow(seg_z_segment, cmap="gray")
        # plt.scatter(segment_inner_barycenter[1], segment_inner_barycenter[0], c="r",s=1)
        # plt.scatter(segment_outer_barycenter[1], segment_outer_barycenter[0], c="b",s=1)

        barycenter_segments_inner.append(segment_inner_barycenter)
        barycenter_segments_outer.append(segment_outer_barycenter)

    radius_segments = []  # define RA=|BI|/BSA
    thickness_segments = []  # define T=|BO|/BSA-RA

    for i in range(NUM_SEGMENTS):
        barycenter_inner_real = np.dot(
            nim_sa.affine, np.array([barycenter_segments_inner[i][0], barycenter_segments_inner[i][1], 0, 1])
        )[:3]
        barycenter_outer_real = np.dot(
            nim_sa.affine, np.array([barycenter_segments_outer[i][0], barycenter_segments_outer[i][1], 0, 1])
        )[:3]
        barycenter_LV_real = np.dot(nim_sa.affine, np.array([barycenter_LV[0], barycenter_LV[1], 0, 1]))[:3]

        radius = np.linalg.norm(barycenter_inner_real - barycenter_LV_real) / BSA_value * 1e-1  # unit: cm
        thickness = np.linalg.norm(barycenter_outer_real - barycenter_LV_real) / BSA_value * 1e-1 - radius
        if thickness <= 0:
            raise ValueError("Thickness should be positive.")
        radius_segments.append(radius)
        thickness_segments.append(thickness)

    return (np.array(radius_segments), np.array(thickness_segments))


def evaluate_radius_thickness_disparity(
    seg_sa: Tuple[float, float, float, float], nim_sa: nib.nifti1.Nifti1Image, BSA_value: float
):
    if seg_sa.ndim != 4:
        raise ValueError("The seg_sa should be a 4D (3D+t) image.")
    T = seg_sa.shape[3]
    radius = []
    thickness = []
    for t in tqdm(range(T)):
        seg_sa_t = seg_sa[:, :, :, t]
        try:
            basal_slice = determine_sa_basal_slice(seg_sa_t)
            apical_slice = determine_sa_apical_slice(seg_sa_t)
        except ValueError:
            continue

        radius_t = []
        thickness_t = []
        for s in range(basal_slice, apical_slice + 1):
            seg_sa_s_t = seg_sa_t[:, :, s]
            radius_segments, thickness_segments = evaluate_radius_thickness(seg_sa_s_t, nim_sa, BSA_value)
            radius_t.append(radius_segments)
            thickness_t.append(thickness_segments)
        radius.append(np.array(radius_t).ravel().tolist())
        thickness.append(np.array(thickness_t).ravel().tolist())

    # We cannot use numpy manipulation here as the slice*segment is variable for each timepoint
    T_used = len(radius)
    radius_disparity = np.zeros(T_used)
    thickness_disparity = np.zeros(T_used)
    for t in range(T_used):
        # define Disparity: max(value_t/value_0) - min(value_t/value_0)
        # Since we have lots of elements, we remove top 10% and bottom 10% to avoid possible outliers

        radius_t = radius[t]
        radius_t_0 = radius_t[0]
        radius_t = sorted(radius_t)
        radius_t = radius_t[int(0.1 * len(radius_t)) : int(0.9 * len(radius_t))]
        radius_t = [r / radius_t_0 for r in radius_t]
        radius_disparity_t = max(radius_t) - min(radius_t)
        radius_disparity[t] = radius_disparity_t

        thickness_t = thickness[t]
        thickness_t_0 = thickness_t[0]
        thickness_t = sorted(thickness_t)
        thickness_t = thickness_t[int(0.1 * len(thickness_t)) : int(0.9 * len(thickness_t))]
        thickness_t = [t / thickness_t_0 for t in thickness_t]
        thickness_disparity_t = max(thickness_t) - min(thickness_t)
        thickness_disparity[t] = thickness_disparity_t

    radius_motion_disparity = np.max(radius_disparity)  # maximum over time
    thickness_motion_disparity = np.max(thickness_disparity)

    return radius_motion_disparity, thickness_motion_disparity, radius, thickness


def fractal_dimension(seg_sa_ED: Tuple[float, float, float], nim_sa_ED: nib.nifti1.Nifti1Image):
    fds = []

    img_sa = nim_sa_ED.get_fdata()
    for i in range(0, seg_sa_ED.shape[2]):
        seg_endo = seg_sa_ED[:, :, i] == 1
        img_endo = img_sa[:, :, i] * seg_endo
        if img_endo.sum() == 0:
            continue
        img_endo = (255 * (img_endo - np.min(img_endo)) / (np.max(img_endo) - np.min(img_endo))).astype(np.uint8)
        # plt.imshow(img_endo, cmap="gray")
        seg_backgroud = (img_endo > 0).astype(np.uint8)
        adaptive_thresh = cv2.adaptiveThreshold(img_endo, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        seg_endo_trabeculation = cv2.bitwise_and(adaptive_thresh, seg_backgroud)
        # plt.imshow(seg_endo_trabeculation, cmap="gray")

        # Find a bounding box that contain the endocardium and trabeculation
        coords = cv2.findNonZero(seg_endo_trabeculation)
        x, y, w, h = cv2.boundingRect(coords)
        # print(f"{i}: {w}, {h}")
        if w < 20 or h < 20:
            continue
        x -= 10
        y -= 10
        w += 20
        h += 20
        seg_endo_trabeculation_cropped = seg_endo_trabeculation[y : y + h, x : x + w]

        contours, _ = cv2.findContours(seg_endo_trabeculation_cropped, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        seg_endo_trabeculation_contour = np.zeros_like(seg_endo_trabeculation_cropped)
        cv2.drawContours(seg_endo_trabeculation_contour, contours, -1, 255, 1)

        scale = np.arange(0.05, 0.5, 0.01)
        bins = scale * seg_endo_trabeculation_cropped.shape[0]
        ps.metrics.boxcount(seg_endo_trabeculation_contour, bins=bins)
        boxcount_data = ps.metrics.boxcount(seg_endo_trabeculation_contour, bins=bins)
        slope, _, _, _, _ = linregress(np.log(scale), np.log(boxcount_data.count))
        slope = abs(slope)
        if slope < 1 or slope > 2:
            raise ValueError("Fractal dimension should lie between 1 and 2.")
        fds.append(slope)

    return np.mean(np.array(fds))


def calculate_LV_center(seg_sa, affine, time, slice):
    """
    Determine the real-world coordinates of the center of the LV in the given slice and time.
    """
    if seg_sa.ndim != 4:
        raise ValueError("seg_sa should be 4D (3D+t) array")

    labels = {"BG": 0, "LV": 1, "Myo": 2, "RV": 3}

    seg_z_t = seg_sa[:, :, slice, time]
    seg_z_t_LV = seg_z_t == labels["LV"]

    barycenter_LV = np.mean(np.argwhere(seg_z_t_LV), axis=0)

    barycenter_LV = np.dot(affine, np.array([*barycenter_LV, slice, 1]))[:3]

    return barycenter_LV


def evaluate_torsion(seg_sa: Tuple[float, float, float, float], nim_sa: nib.nifti1.Nifti1Image, contour_name_stem):
    if seg_sa.ndim != 4:
        raise ValueError("seg_sa should be 4D (3D+t) array")
    Z = seg_sa.shape[2]
    T = seg_sa.shape[3]
    n_slices_total = Z + 1
    affine = nim_sa.affine

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(f"{contour_name_stem}00.vtk")
    reader.Update()
    poly = reader.GetOutput()

    points = poly.GetPoints()
    lines = poly.GetLines()
    lines_dir = poly.GetCellData().GetArray("Direction ID")

    n_points = points.GetNumberOfPoints()
    n_lines = lines.GetNumberOfCells()

    points_base_ED = {"epi": [], "endo": []}  # define used to determine the angle through arcsin of vector
    points_apex_ED = {"epi": [], "endo": []}

    # Determine the three slices used
    affine_inv = np.linalg.inv(affine)
    z_cnt = np.zeros(Z)

    for i in range(n_points):
        point = points.GetPoint(i)
        (_, _, z) = np.dot(affine_inv, np.append(point, 1))[:3]
        z = int(round(z))
        z_cnt[z] += 1
        # plt.imshow(seg_sa_ED[:, :, z], cmap='gray')
        # plt.scatter(y, x, c='r')

    basal_slice = np.min(np.nonzero(z_cnt)[0])
    apical_slice = np.max(np.nonzero(z_cnt)[0])
    # print(basal_slice, apical_slice)

    # Ref https://dx.plos.org/10.1371/journal.pone.0109164
    SLICE_THICKNESS = 8  # unit: mm
    SLICE_GAP = 2
    n_slices_used = apical_slice - basal_slice + 1

    if nim_sa.header["pixdim"][3] != SLICE_THICKNESS + SLICE_GAP:
        raise ValueError("The slice thickness and gap does not aligns with protocol")

    # * Determine the vector at ED
    LV_barycenter_basal_ED = calculate_LV_center(seg_sa, affine, 0, basal_slice)
    LV_barycenter_apical_ED = calculate_LV_center(seg_sa, affine, 0, apical_slice)

    lines.InitTraversal()
    for i in range(n_lines):
        ids = vtk.vtkIdList()
        lines.GetNextCell(ids)

        dir_id = lines_dir.GetValue(i)
        if dir_id != 1:
            # non-radial line
            continue

        # define p1 is on endo and p2 is on epi
        p1 = np.array(points.GetPoint(ids.GetId(0)))
        p2 = np.array(points.GetPoint(ids.GetId(1)))

        (_, _, z1) = np.dot(affine_inv, np.append(p1, 1))[:3]
        (_, _, z2) = np.dot(affine_inv, np.append(p2, 1))[:3]
        z1 = round(z1)
        z2 = round(z2)

        if z1 != z2:
            raise ValueError("The line is not on the same slice")

        if z1 == basal_slice:
            points_base_ED["endo"].append(p1)
            points_base_ED["epi"].append(p2)
        elif z1 == apical_slice:
            points_apex_ED["endo"].append(p1)
            points_apex_ED["epi"].append(p2)

    # * Note that since the slice we chosen are not 0% and 100%, the twist here should be smaller than normal
    # * Only the torsion should be considered

    torsion_endo = {"base": np.zeros(T), "apex": np.zeros(T), "twist": np.zeros(T), "torsion": np.zeros(T)}

    torsion_epi = {"base": np.zeros(T), "apex": np.zeros(T), "twist": np.zeros(T), "torsion": np.zeros(T)}

    torsion_global = {"base": np.zeros(T), "apex": np.zeros(T), "twist": np.zeros(T), "torsion": np.zeros(T)}

    for fr in range(1, T):
        reader = vtk.vtkPolyDataReader()
        filename_myo = f"{contour_name_stem}{fr:02d}.vtk"
        reader.SetFileName(filename_myo)
        reader.Update()
        poly = reader.GetOutput()
        points = poly.GetPoints()
        lines = poly.GetLines()
        lines_dir = poly.GetCellData().GetArray("Direction ID")
        n_lines = lines.GetNumberOfCells()

        points_base = {"epi": [], "endo": []}
        points_apex = {"epi": [], "endo": []}

        LV_barycenter_basal = calculate_LV_center(seg_sa, affine, fr, basal_slice)
        LV_barycenter_apical = calculate_LV_center(seg_sa, affine, fr, apical_slice)

        lines.InitTraversal()
        for i in range(n_lines):
            ids = vtk.vtkIdList()
            lines.GetNextCell(ids)

            dir_id = lines_dir.GetValue(i)
            if dir_id != 1:
                # non-radial line
                continue

            p1 = np.array(points.GetPoint(ids.GetId(0)))
            p2 = np.array(points.GetPoint(ids.GetId(1)))

            (_, _, z1) = np.dot(affine_inv, np.append(p1, 1))[:3]
            (_, _, z2) = np.dot(affine_inv, np.append(p2, 1))[:3]
            z1 = round(z1)
            z2 = round(z2)

            if z1 != z2:
                logger.error(f"{z1} and {z2} are not on the same slice")
                raise ValueError("The line is not on the same slice")

            if z1 == basal_slice:
                points_base["endo"].append(p1)
                points_base["epi"].append(p2)
            elif z1 == apical_slice:
                points_apex["endo"].append(p1)
                points_apex["epi"].append(p2)

        if len(points_base_ED["epi"]) != len(points_base["epi"]) or len(points_base_ED["endo"]) != len(points_base["endo"]):
            raise ValueError(f"The number of points at frame {fr} is different from ED")

        # Ref https://www.sciencedirect.com/science/article/pii/S1097664723012437?via%3Dihub

        rotations_base_z = {"epi": np.zeros(len(points_base["epi"])), "endo": np.zeros(len(points_base["endo"]))}
        rotations_apex_z = {"epi": np.zeros(len(points_apex["epi"])), "endo": np.zeros(len(points_apex["endo"]))}

        n_base = len(points_base["epi"])
        n_apex = len(points_apex["epi"])

        for i in range(n_base):
            v1 = points_base_ED["epi"][i] - LV_barycenter_basal_ED
            v1 = [v1[0], v1[1]]
            v2 = points_base["epi"][i] - LV_barycenter_basal
            v2 = [v2[0], v2[1]]
            rotations_base_z["epi"][i] = np.degrees(np.arcsin(np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))

        for i in range(n_base):
            v1 = points_base_ED["endo"][i] - LV_barycenter_basal_ED
            v1 = [v1[0], v1[1]]
            v2 = points_base["endo"][i] - LV_barycenter_basal
            v2 = [v2[0], v2[1]]
            rotations_base_z["endo"][i] = np.degrees(np.arcsin(np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))

        for i in range(n_apex):
            v1 = points_apex_ED["epi"][i] - LV_barycenter_apical_ED
            v1 = [v1[0], v1[1]]
            v2 = points_apex["epi"][i] - LV_barycenter_apical
            v2 = [v2[0], v2[1]]
            rotations_apex_z["epi"][i] = np.degrees(np.arcsin(np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))

        for i in range(n_apex):
            v1 = points_apex_ED["endo"][i] - LV_barycenter_apical_ED
            v1 = [v1[0], v1[1]]
            v2 = points_apex["endo"][i] - LV_barycenter_apical
            v2 = [v2[0], v2[1]]
            rotations_apex_z["endo"][i] = np.degrees(np.arcsin(np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))

        torsion_endo["base"][fr] = np.mean(rotations_base_z["endo"])
        torsion_endo["apex"][fr] = -np.mean(rotations_apex_z["endo"])
        torsion_endo["twist"][fr] = torsion_endo["apex"][fr] - torsion_endo["base"][fr]
        torsion_endo["torsion"][fr] = torsion_endo["twist"][fr] / (
            ((SLICE_THICKNESS + SLICE_GAP) * (n_slices_used - 1)) * 0.1
        )  # unit: degree/cm

        torsion_epi["base"][fr] = np.mean(rotations_base_z["epi"])
        torsion_epi["apex"][fr] = -np.mean(rotations_apex_z["epi"])
        torsion_epi["twist"][fr] = torsion_epi["apex"][fr] - torsion_epi["base"][fr]
        torsion_epi["torsion"][fr] = torsion_epi["twist"][fr] / (
            ((SLICE_THICKNESS + SLICE_GAP) * (n_slices_used - 1)) * 0.1
        )

        torsion_global["base"][fr] = (n_base * torsion_endo["base"][fr] + n_apex * torsion_epi["base"][fr]) / (n_base + n_apex)
        torsion_global["apex"][fr] = (n_base * torsion_endo["apex"][fr] + n_apex * torsion_epi["apex"][fr]) / (n_base + n_apex)
        torsion_global["twist"][fr] = torsion_global["apex"][fr] - torsion_global["base"][fr]
        torsion_global["torsion"][fr] = torsion_global["twist"][fr] / (
            ((SLICE_THICKNESS + SLICE_GAP) * (n_slices_used - 1)) * 0.1
        )

    return torsion_endo, torsion_epi, torsion_global, basal_slice, apical_slice


def evaluate_t1_uncorrected(img_ShMOLLI: Tuple[float, float], seg_ShMOLLI: Tuple[float, float], labels):
    """
    Determine the global native T1 values as well as those for inter-ventricular septum (IVS), free-wall (FW) and blood pools.
    """
    seg_LV = np.zeros_like(seg_ShMOLLI)
    seg_myo = np.zeros_like(seg_ShMOLLI)
    seg_RV = np.zeros_like(seg_ShMOLLI)
    seg_LV = np.where(seg_ShMOLLI == labels["LV"], 1, seg_LV)
    seg_myo = np.where(seg_ShMOLLI == labels["Myo"], 1, seg_myo)
    seg_RV = np.where(seg_ShMOLLI == labels["RV"], 1, seg_RV)
    seg_LV = get_largest_cc(seg_LV)
    seg_RV = get_largest_cc(seg_RV)
    seg_LV_dilated = seg_LV.copy()
    seg_RV_dilated = seg_RV.copy()
    dilation_threshold = 12
    intersection_threshold = 400
    dilation_value = 0
    while dilation_value < dilation_threshold:
        seg_LV_dilated = binary_dilation(seg_LV_dilated)
        seg_RV_dilated = binary_dilation(seg_RV_dilated)
        dilation_value += 1
        # Stop if there is intersection between LV and RV
        if np.sum(seg_LV_dilated * seg_RV_dilated) > intersection_threshold:
            logger.info(f"Times of dilation to determine septum: {dilation_value}")
            # plt.imshow(seg_LV_dilated, cmap='gray')
            # plt.imshow(seg_RV_dilated, cmap='gray', alpha=0.5)
            break

    if dilation_value == dilation_threshold:
        raise ValueError("Exceeds dilation threshold when trying to determine septum.")

    # Determine two landmarks
    seg_intersection = seg_LV_dilated * seg_RV_dilated
    seg_intersection_contours, _ = cv2.findContours(seg_intersection.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seg_intersection_points = np.vstack([contour.reshape(-1, 2) for contour in seg_intersection_contours])

    # Compute the cross distance between two sets
    dist_matrix = cdist(seg_intersection_points, seg_intersection_points, "euclidean")
    # Convert flat index to 2D index
    max_distance_idx = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)

    lm1 = seg_intersection_points[max_distance_idx[0]]
    lm2 = seg_intersection_points[max_distance_idx[1]]
    logger.info("Landmarks for IVS are determined.")

    LV_barycenter = np.mean(np.argwhere(seg_LV_dilated), axis=0)
    # swap axis to align with lm
    LV_barycenter = np.array([LV_barycenter[1], LV_barycenter[0]])

    # seg_LV_dilated_for_myo = binary_dilation(seg_LV, iterations=6)
    # seg_myo = np.where((seg_LV_dilated_for_myo == 1) & (seg_LV == 0), 1, 0)

    seg_myo_IVS = np.zeros_like(seg_ShMOLLI, dtype=np.uint8)  # define intraventricular septum
    seg_myo_IVS = np.ascontiguousarray(seg_myo_IVS)
    seg_myo_FW = np.zeros_like(seg_ShMOLLI, dtype=np.uint8)  # define free-wall
    seg_myo_FW = np.ascontiguousarray(seg_myo_FW)

    # We need to make sure the triangle covers the mask
    # Draw a line between lm and barycenter to get a more distant point
    dir_lm1_barycenter = np.array(lm1) - np.array(LV_barycenter)
    lm1_distant = lm1 + dir_lm1_barycenter * 2
    dir_lm2_barycenter = np.array(lm2) - np.array(LV_barycenter)
    lm2_distant = lm2 + dir_lm2_barycenter * 2

    triangle_contour = np.array([lm1_distant, lm2_distant, LV_barycenter]).astype(int)
    cv2.fillPoly(seg_myo_IVS, [triangle_contour], 1)
    seg_myo_IVS = np.where(seg_myo == 1, seg_myo_IVS, 0)
    seg_myo_IVS = get_largest_cc(seg_myo_IVS)
    seg_myo_FW = seg_myo - seg_myo_IVS
    seg_myo_FW = get_largest_cc(seg_myo_FW)

    seg_myo_IVS_eroded = binary_erosion(get_largest_cc(seg_myo_IVS), iterations=2)
    seg_myo_FW_eroded = binary_erosion(get_largest_cc(seg_myo_FW), iterations=2)
    seg_myo_eroded = seg_myo_IVS_eroded + seg_myo_FW_eroded

    # * The LV/RV blood pool segmentations are eroded until 1/3 area of original mask
    area_LV = np.sum(seg_LV)
    area_RV = np.sum(seg_RV)
    seg_LV_eroded = seg_LV.copy()
    seg_RV_eroded = seg_RV.copy()
    while np.sum(seg_LV_eroded) > 0.34 * area_LV:
        seg_LV_eroded = binary_erosion(seg_LV_eroded)
    while np.sum(seg_RV_eroded) > 0.34 * area_RV:
        seg_RV_eroded = binary_erosion(seg_RV_eroded)
    logger.info("Blood pools are eroded.")

    # * To ensure no papillary muscles are included, any pixel whose T1 value was less than Q1-1.5*IQR is excluded
    T1_LV_raw = img_ShMOLLI[seg_LV_eroded == 1]
    Q1_LV_raw = np.percentile(T1_LV_raw, 25)
    Q3_LV_raw = np.percentile(T1_LV_raw, 75)
    IQR_LV_raw = Q3_LV_raw - Q1_LV_raw
    seg_LV_eroded = np.where(img_ShMOLLI < Q1_LV_raw - 1.5 * IQR_LV_raw, 0, seg_LV_eroded)
    T1_RV_raw = img_ShMOLLI[seg_RV_eroded == 1]
    Q1_RV_raw = np.percentile(T1_RV_raw, 25)
    Q3_RV_raw = np.percentile(T1_RV_raw, 75)
    IQR_RV_raw = Q3_RV_raw - Q1_RV_raw
    seg_RV_eroded = np.where(img_ShMOLLI < Q1_RV_raw - 1.5 * IQR_RV_raw, 0, seg_RV_eroded)
    seg_blood_eroded = seg_LV_eroded + seg_RV_eroded

    T1_global = img_ShMOLLI[seg_myo_eroded == 1]
    T1_IVS = img_ShMOLLI[seg_myo_IVS_eroded == 1]
    T1_FW = img_ShMOLLI[seg_myo_FW_eroded == 1]
    T1_blood = img_ShMOLLI[seg_blood_eroded == 1]

    # visualization
    figure = plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_ShMOLLI, cmap="gray")
    plt.imshow(seg_myo_IVS_eroded, cmap="gray", alpha=0.5)
    plt.scatter(lm1[0], lm1[1], c="r", s=8)
    plt.scatter(lm2[0], lm2[1], c="r", s=8)
    plt.scatter(LV_barycenter[0], LV_barycenter[1], c="blue", s=8)
    plt.title("Intraventricular septum (IVS)")
    plt.subplot(1, 3, 2)
    plt.imshow(img_ShMOLLI, cmap="gray")
    plt.imshow(seg_myo_FW_eroded, cmap="gray", alpha=0.5)
    plt.title("Free wall (FW)")
    plt.subplot(1, 3, 3)
    plt.imshow(img_ShMOLLI, cmap="gray")
    plt.imshow(seg_blood_eroded, cmap="gray", alpha=0.5)
    plt.title("Blood pool")
    return (T1_global.mean(), T1_IVS.mean(), T1_FW.mean(), T1_blood.mean(), figure)


def evaluate_velocity_flow(seg_morphology: Tuple[float, float, float], 
                           img_phase: Tuple[float, float, float], 
                           VENC,
                           square_per_pix
):
    if seg_morphology.ndim != 3 or img_phase.ndim != 3:
        raise ValueError("The input should be 3D image.")

    T = seg_morphology.shape[2]

    velocity = []
    flow = []
    flow_center = []  # define Center of velocity of forward flow
    velocity_map_all = np.zeros_like(img_phase)

    for t in range(T):
        mask = seg_morphology[:, :, t]
        phase = img_phase[:, :, t]

        # Ref Nayak, Krishna S., et al. “Cardiovascular Magnetic Resonance Phase Contrast Imaging.”
        phase_normalized = (phase / 4096.0) * (2 * np.pi) - np.pi
        velocity_map = (phase_normalized / np.pi) * VENC
        velocity_map_roi = velocity_map[mask == 1]

        velocity.append(np.mean(velocity_map_roi))  # unit: cm/s
        flow.append(np.sum(velocity_map_roi) * square_per_pix)  # unit: cm^3/s=mL/s

        # Ref Systolic Flow Displacement Correlates With Future Ascending Aortic Growth in Patients With Bicuspid Aortic Valves
        # Calculate the center of velocity: weighted by the absolute velocity information
        velocity_map_y_coords, velocity_map_x_coords = np.indices(velocity_map.shape)
        weights = np.where(mask == 1, np.abs(velocity_map), 0)
        total_weight = np.sum(weights)
        center_x = np.sum(velocity_map_x_coords * weights) / total_weight
        center_y = np.sum(velocity_map_y_coords * weights) / total_weight
        flow_center.append((center_x, center_y))

        velocity_map_all[:, :, t] = velocity_map

    return np.array(velocity), np.array(flow), np.array(flow_center), velocity_map_all
