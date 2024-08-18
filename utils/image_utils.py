# Copyright 2017, Wenjia Bai. All Rights Reserved.
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
import cv2
import os
import numpy as np
import nibabel as nib
from scipy import ndimage
import scipy.ndimage.measurements as measure
from matplotlib import pyplot as plt
import seaborn as sns

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from utils.log_utils import setup_logging

logger = setup_logging("image-utils")

def crop_image(image, cx, cy, size):
    """Crop a 3D image using a bounding box centred at (cx, cy) with specified size"""
    X, Y = image.shape[:2]
    r = int(size / 2)
    x1, x2 = cx - r, cx + r
    y1, y2 = cy - r, cy + r
    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)
    # Crop the image
    crop = image[x1_:x2_, y1_:y2_]
    # Pad the image if the specified size is larger than the input image size
    if crop.ndim == 3:
        crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)), "constant")
    elif crop.ndim == 4:
        crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)), "constant")
    else:
        print("Error: unsupported dimension, crop.ndim = {0}.".format(crop.ndim))
        exit(0)
    return crop


def normalise_intensity(image, thres_roi=10.0):
    """Normalise the image intensity by the mean and standard deviation"""
    val_l = np.percentile(image, thres_roi)
    roi = image >= val_l
    mu, sigma = np.mean(image[roi]), np.std(image[roi])
    eps = 1e-6
    image2 = (image - mu) / (sigma + eps)
    return image2


def rescale_intensity(image, thres=(1.0, 99.0)):
    """Rescale the image intensity to the range of [0, 1]"""
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2


def data_augmenter(image, label, shift, rotate, scale, intensity, flip):
    """
    Online data augmentation
    Perform affine transformation on image and label,
    which are 4D tensor of shape (N, H, W, C) and 3D tensor of shape (N, H, W).
    """
    image2 = np.zeros(image.shape, dtype=np.float32)
    label2 = np.zeros(label.shape, dtype=np.int32)
    for i in range(image.shape[0]):
        # For each image slice, generate random affine transformation parameters
        # using the Gaussian distribution
        shift_val = [np.clip(np.random.normal(), -3, 3) * shift, np.clip(np.random.normal(), -3, 3) * shift]
        rotate_val = np.clip(np.random.normal(), -3, 3) * rotate
        scale_val = 1 + np.clip(np.random.normal(), -3, 3) * scale
        intensity_val = 1 + np.clip(np.random.normal(), -3, 3) * intensity

        # Apply the affine transformation (rotation + scale + shift) to the image
        row, col = image.shape[1:3]
        M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_val, 1.0 / scale_val)
        M[:, 2] += shift_val
        for c in range(image.shape[3]):
            image2[i, :, :, c] = ndimage.interpolation.affine_transform(image[i, :, :, c], M[:, :2], M[:, 2], order=1)

        # Apply the affine transformation (rotation + scale + shift) to the label map
        label2[i, :, :] = ndimage.interpolation.affine_transform(label[i, :, :], M[:, :2], M[:, 2], order=0)

        # Apply intensity variation
        image2[i] *= intensity_val

        # Apply random horizontal or vertical flipping
        if flip:
            if np.random.uniform() >= 0.5:
                image2[i] = image2[i, ::-1, :, :]
                label2[i] = label2[i, ::-1, :]
            else:
                image2[i] = image2[i, :, ::-1, :]
                label2[i] = label2[i, :, ::-1]
    return image2, label2


def aortic_data_augmenter(image, label, shift, rotate, scale, intensity, flip):
    """
    Online data augmentation
    Perform affine transformation on image and label,

    image: NXYC
    label: NXY
    """
    image2 = np.zeros(image.shape, dtype=np.float32)
    label2 = np.zeros(label.shape, dtype=np.int32)

    # For N image. which come come from the same subject in the LSTM model,
    # generate the same random affine transformation parameters.
    shift_val = [np.clip(np.random.normal(), -3, 3) * shift, np.clip(np.random.normal(), -3, 3) * shift]
    rotate_val = np.clip(np.random.normal(), -3, 3) * rotate
    scale_val = 1 + np.clip(np.random.normal(), -3, 3) * scale
    intensity_val = 1 + np.clip(np.random.normal(), -3, 3) * intensity

    # The affine transformation (rotation + scale + shift)
    row, col = image.shape[1:3]
    M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_val, 1.0 / scale_val)
    M[:, 2] += shift_val

    # Apply the transformation to the image
    for i in range(image.shape[0]):
        for c in range(image.shape[3]):
            image2[i, :, :, c] = ndimage.interpolation.affine_transform(image[i, :, :, c], M[:, :2], M[:, 2], order=1)

        label2[i, :, :] = ndimage.interpolation.affine_transform(label[i, :, :], M[:, :2], M[:, 2], order=0)

        # Apply intensity variation
        image2[i] *= intensity_val

        # Apply random horizontal or vertical flipping
        if flip:
            if np.random.uniform() >= 0.5:
                image2[i] = image2[i, ::-1, :, :]
                label2[i] = label2[i, ::-1, :]
            else:
                image2[i] = image2[i, :, ::-1, :]
                label2[i] = label2[i, :, ::-1]
    return image2, label2


def np_categorical_dice(pred, truth, k):
    """
    Dice overlap metric for label k
    """
    A = (pred == k).astype(np.float32)
    B = (truth == k).astype(np.float32)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B))


def distance_metric(seg_A, seg_B, dx):
    """
    Measure the distance errors between the contours of two segmentations.
    The manual contours are drawn on 2D slices.
    We calculate contour to contour distance for each slice.
    """
    table_md = []
    table_hd = []
    _, _, Z = seg_A.shape
    for z in range(Z):
        # Binary mask at this slice
        slice_A = seg_A[:, :, z].astype(np.uint8)
        slice_B = seg_B[:, :, z].astype(np.uint8)

        # The distance is defined only when both contours exist on this slice
        if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
            # Find contours and retrieve all the points
            contours, _ = cv2.findContours(cv2.inRange(slice_A, 1, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pts_A = contours[0]
            for i in range(1, len(contours)):
                pts_A = np.vstack((pts_A, contours[i]))

            contours, _ = cv2.findContours(cv2.inRange(slice_B, 1, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pts_B = contours[0]
            for i in range(1, len(contours)):
                pts_B = np.vstack((pts_B, contours[i]))

            # Distance matrix between point sets
            M = np.zeros((len(pts_A), len(pts_B)))
            for i in range(len(pts_A)):
                for j in range(len(pts_B)):
                    M[i, j] = np.linalg.norm(pts_A[i, 0] - pts_B[j, 0])

            # Mean distance and hausdorff distance
            md = 0.5 * (np.mean(np.min(M, axis=0)) + np.mean(np.min(M, axis=1))) * dx
            hd = np.max([np.max(np.min(M, axis=0)), np.max(np.min(M, axis=1))]) * dx
            table_md += [md]
            table_hd += [hd]

    # Return the mean distance and Hausdorff distance across 2D slices
    mean_md = np.mean(table_md) if table_md else None
    mean_hd = np.mean(table_hd) if table_hd else None
    return mean_md, mean_hd


def get_largest_cc(binary):
    """Get the largest connected component in the foreground."""
    cc, n_cc = measure.label(binary)
    max_n = -1
    max_area = 0
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area > max_area:
            max_area = area
            max_n = n
    largest_cc = cc == max_n
    return largest_cc


def remove_small_cc(binary, thres=10):
    """Remove small connected component in the foreground."""
    cc, n_cc = measure.label(binary)
    binary2 = np.copy(binary)
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area < thres:
            binary2[cc == n] = 0
    return binary2


def split_sequence(image_name, output_name):
    """Split an image sequence into a number of time frames."""
    nim = nib.load(image_name)
    T = nim.header["dim"][4]
    affine = nim.affine
    image = nim.get_fdata()

    for t in range(T):
        image_fr = image[:, :, :, t]
        nim2 = nib.Nifti1Image(image_fr, affine)
        nib.save(nim2, "{0}{1:02d}.nii.gz".format(output_name, t))


def make_sequence(image_names, dt, output_name):
    """Combine a number of time frames into one image sequence."""
    nim = nib.load(image_names[0])
    affine = nim.affine
    X, Y, Z = nim.header["dim"][1:4]
    T = len(image_names)
    image = np.zeros((X, Y, Z, T))

    for t in range(T):
        image[:, :, :, t] = nib.load(image_names[t]).get_fdata()

    nim2 = nib.Nifti1Image(image, affine)
    nim2.header["pixdim"][4] = dt
    nib.save(nim2, output_name)


def split_volume(image_name, output_name):
    """
    Split an image volume into a number of slices.
    """
    nim = nib.load(image_name)
    Z = nim.header["dim"][3]
    affine = nim.affine
    image = nim.get_fdata()

    for z in range(Z):
        image_slice = image[:, :, z]
        image_slice = np.expand_dims(image_slice, axis=2)
        affine2 = np.copy(affine)
        affine2[:3, 3] += z * affine2[:3, 2]
        nim2 = nib.Nifti1Image(image_slice, affine2)
        nib.save(nim2, "{0}{1:02d}.nii.gz".format(output_name, z))


def image_apply_mask(input_name, output_name, mask_image, pad_value=-1):
    # Assign the background voxels (mask == 0) with pad_value
    nim = nib.load(input_name)
    image = nim.get_fdata()
    image[mask_image == 0] = pad_value
    nim2 = nib.Nifti1Image(image, nim.affine)
    nib.save(nim2, output_name)


def padding(input_A_name, input_B_name, output_name, value_in_B, value_output):
    """
    Pad the image A with the value_output where the image B has the value_in_B.
    """
    nim = nib.load(input_A_name)
    image_A = nim.get_fdata()
    image_B = nib.load(input_B_name).get_fdata()
    image_A[image_B == value_in_B] = value_output
    nim2 = nib.Nifti1Image(image_A, nim.affine)
    nib.save(nim2, output_name)


def auto_crop_image(input_name, output_name, reserve):
    """
    Automatically crop the image to remove the background.

    Parameters:
    - input_name: string, input image file name
    - output_name: string, output image file name
    - reserve: int, number of pixels to reserve outside the bounding box
    """
    nim = nib.load(input_name)
    image = nim.get_fdata()
    X, Y, Z = image.shape[:3]

    # Detect the bounding box of the foreground
    idx = np.nonzero(image > 0)
    x1, x2 = idx[0].min() - reserve, idx[0].max() + reserve + 1
    y1, y2 = idx[1].min() - reserve, idx[1].max() + reserve + 1
    z1, z2 = idx[2].min() - reserve, idx[2].max() + reserve + 1
    x1, x2 = max(x1, 0), min(x2, X)
    y1, y2 = max(y1, 0), min(y2, Y)
    z1, z2 = max(z1, 0), min(z2, Z)
    logger.info("Auto-crop image with bounding box:")
    logger.info(f"  Bottom-left corner = ({x1},{y1},{z1})")
    logger.info(f"  Top-right corner = ({x2},{y2},{z2})")

    # Crop the image
    image = image[x1:x2, y1:y2, z1:z2]

    # Update the affine matrix so that the origin is at the bottom-left corner
    affine = nim.affine
    # * Note that it's the fourth column!
    # * The first three columns are for rotation and scaling, while fourth column is for translation
    affine[:3, 3] = np.dot(affine, np.array([x1, y1, z1, 1]))[:3]
    nim2 = nib.Nifti1Image(image, affine)
    nib.save(nim2, output_name)


def overlay_contour(img, seg, RGBforLabel: dict = None, title=None, fill=False, scatters: dict = None):
    """
    Overlay segmentation contour on the img image.

    Parameters:
    - img: 2D numpy array, original image
    - seg: 2D numpy array, segmentation mask
    - RGBforLabel: dictionary, mapping labels to RGB colors (optional)
    - title: string, title for the plot (optional)
    - fill: boolean, whether to fill the contours or not (optional)
    """
    img = cv2.convertScaleAbs(img)  # avoid unsupported depth of input image

    assert (
        len(img.shape) == 2 or len(img.shape) == 3
    ), "img and seg must be 2D images, or 2D images where content is the same in each channel"
    if len(img.shape) == 3:
        img = img[:, :, 0]
        img = np.reshape(img, img.shape + (1,))
        if len(seg.shape) == 3:
            seg = seg[:, :, 0]
            seg = np.reshape(seg, seg.shape + (1,))
        elif len(seg.shape) == 2:
            seg = np.reshape(seg, seg.shape + (1,))
    assert img.shape == seg.shape, "img and seg must have same shape"

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    palette = sns.color_palette("Set1", 8)
    values = np.unique(seg)[1:]

    all_contours = []
    for value in values:
        binary_mask = np.where(seg == value, 1, 0).astype(np.uint8)
        contour, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.append(contour)

    if title is not None:
        plt.title(title)

    for i, contour in enumerate(all_contours):
        # opencv requires color to be in 0~255, whilc cmaps are in 0~1
        contour_color = palette[i]
        contour_color = [int(255 * c) for c in contour_color]
        plt.imshow(cv2.drawContours(img, contour, -1, color=contour_color, thickness=-1 if fill else 1), cmap="gray")
        if scatters is not None:
            scatter = scatters[values[i]]  # ignore ground truth
            for idx, s in enumerate(scatter):
                plt.scatter(s[0], s[1], color=palette[-(idx + 1)])
