import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import argparse
import sys
import ants

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config
from utils.log_utils import setup_logging

logger = setup_logging("preprocess_lvot")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Folder for one subject that contains Nifti files", required=True)


def check_brightness_anomaly(nii):
    """
    Check if there is a brightness anomaly in the image based on the standard deviation of the left and right halves of the image.
    The abnormal images will go through bias correction.
    """

    if nii.ndim != 4:
        raise ValueError("Input image must be 4D")

    if nii.shape[2] != 1 or nii.shape[3] != 50:
        raise ValueError("Input image must be of shape (H, W, 1, 50)")

    std_diff_list = []

    for t in range(nii.shape[3]):
        image = nii[:, :, 0, t]

        # After rotation, this becomes left and right
        left_std = np.std(image[: image.shape[0] // 2, :])
        right_std = np.std(image[image.shape[0] // 2 :, :])
        std_diff = abs(right_std - left_std)
        std_diff_list.append(std_diff)

    std_diff_mean = np.mean(std_diff_list)
    return std_diff_mean > 35, std_diff_mean


if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = args.data_dir
    subject = os.path.basename(data_dir)
    preprocess_dir = os.path.join(data_dir, "preprocess")

    model_dir = config.model_dir

    lvot_name = os.path.join(data_dir, "lvot.nii.gz")
    if not os.path.exists(lvot_name):
        logger.error(f"LVOT MRI file for {subject} does not exist")
        sys.exit(1)

    # * Before segmentation, we need to first pre-process the Nifti file.

    nii_lvot = nib.load(lvot_name)
    lvot = nii_lvot.get_fdata()
    lvot_3d = lvot[:, :, 0, :]
    lvot_3d += 1  # add offset

    # * Step1: Bias correction
    abnormaly, std_diff_mean = check_brightness_anomaly(lvot)
    if abnormaly:
        logger.info(f"{subject}: Brightness anomaly found. Applying bias correction.")
        os.makedirs(preprocess_dir, exist_ok=True)
        if std_diff_mean < 70:
            # Run one time
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(lvot_3d[:, :, 0], cmap="gray")
            plt.title("Original")
            plt.axis("off")

            lvot_3d_bias_corrected = ants.n4_bias_field_correction(ants.from_numpy(lvot_3d), verbose=True)
            lvot_3d_bias_corrected = lvot_3d_bias_corrected.numpy()
            plt.subplot(1, 2, 2)
            plt.imshow(lvot_3d_bias_corrected[:, :, 0], cmap="gray")
            plt.title("Bias Corrected 1st")
            plt.axis("off")

        elif std_diff_mean >= 70 and std_diff_mean < 100:
            # Run two times
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(lvot_3d[:, :, 0], cmap="gray")
            plt.title("Original")
            plt.axis("off")

            lvot_3d_bias_corrected = ants.n4_bias_field_correction(ants.from_numpy(lvot_3d), verbose=True)
            lvot_3d_bias_corrected = lvot_3d_bias_corrected.numpy()
            plt.subplot(1, 3, 2)
            plt.imshow(lvot_3d_bias_corrected[:, :, 0], cmap="gray")
            plt.title("Bias Corrected 1st")
            plt.axis("off")

            lvot_3d_bias_corrected = ants.n4_bias_field_correction(ants.from_numpy(lvot_3d_bias_corrected), verbose=True)
            lvot_3d_bias_corrected = lvot_3d_bias_corrected.numpy()
            plt.subplot(1, 3, 3)
            plt.imshow(lvot_3d_bias_corrected[:, :, 0], cmap="gray")
            plt.title("Bias Corrected 2nd")
            plt.axis("off")

        else:
            # # Run three times
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 4, 1)
            plt.imshow(lvot_3d[:, :, 0], cmap="gray")
            plt.title("Original")
            plt.axis("off")

            lvot_3d_bias_corrected = ants.n4_bias_field_correction(
                ants.from_numpy(lvot_3d), convergence={"iters": [100, 100, 100, 50], "tol": 1e-8}, verbose=True
            )
            lvot_3d_bias_corrected = lvot_3d_bias_corrected.numpy()
            plt.subplot(1, 4, 2)
            plt.imshow(lvot_3d_bias_corrected[:, :, 0], cmap="gray")
            plt.title("Bias Corrected 1st")
            plt.axis("off")

            lvot_3d_bias_corrected = ants.n4_bias_field_correction(
                ants.from_numpy(lvot_3d_bias_corrected), convergence={"iters": [100, 100, 100, 50], "tol": 1e-8}, verbose=True
            )
            lvot_3d_bias_corrected = lvot_3d_bias_corrected.numpy()
            plt.subplot(1, 4, 3)
            plt.imshow(lvot_3d_bias_corrected[:, :, 0], cmap="gray")
            plt.title("Bias Corrected 2nd")
            plt.axis("off")

            lvot_3d_bias_corrected = ants.n4_bias_field_correction(
                ants.from_numpy(lvot_3d_bias_corrected), convergence={"iters": [100, 100, 100, 50], "tol": 1e-8}, verbose=True
            )
            lvot_3d_bias_corrected = lvot_3d_bias_corrected.numpy()
            plt.subplot(1, 4, 4)
            plt.imshow(lvot_3d_bias_corrected[:, :, 0], cmap="gray")
            plt.title("Bias Corrected 3rd")
            plt.axis("off")

        plt.savefig(f"{preprocess_dir}/lvot_bias_correction.png")
        plt.close()

        lvot_3d = lvot_3d_bias_corrected
    else:
        logger.info(f"{subject}: No have brightness anomaly. Skip bias correction.")

    # * Step2: Rotate
    nii_rotated = np.rot90(lvot_3d, 3, (0, 1))  # rotate on first and second axes
    nii_final = np.expand_dims(nii_rotated, axis=2)

    rotation_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    nii_final_affine = nii_lvot.affine.copy()
    nii_final_affine[:3, :3] = np.dot(nii_final_affine[:3, :3], rotation_matrix)

    nii_final = nib.Nifti1Image(nii_final, nii_final_affine, nii_lvot.header)
    nii_final.header["pixdim"][1:4] = nii_lvot.header["pixdim"][1:4]
    lvot_preprocessed_name = os.path.join(data_dir, "lvot_processed.nii.gz")
    nib.save(nii_final, lvot_preprocessed_name)

    logger.info(f"{subject}: LVOT cine rotated. Preprocessing finished.")
