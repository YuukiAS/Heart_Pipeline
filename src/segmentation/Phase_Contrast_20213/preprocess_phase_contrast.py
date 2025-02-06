import os
import shutil
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import argparse
import sys
import ants

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config
from utils.log_utils import setup_logging

logger = setup_logging("preprocess_phase_contrast")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Folder for one subject that contains Nifti files", required=True)


def check_brightness_anomaly(nii):
    """
    Check if there is a brightness anomaly in the image based on the standard deviation of the left and right halves of the image.
    The abnormal images will go through bias correction.
    """

    if nii.ndim != 4:
        raise ValueError("Input image must be 4D")

    if nii.shape[2] != 1 or nii.shape[3] != 30:
        raise ValueError("Input image must be of shape (H, W, 1, 30)")

    std_diff_list = []

    for t in range(nii.shape[3]):
        image = nii[:, :, 0, t]

        lower_std = np.std(image[:, : image.shape[1] // 2])
        upper_std = np.std(image[:, image.shape[1] // 2 :])
        std_diff = abs(lower_std - upper_std)
        std_diff_list.append(std_diff)

    std_diff_mean = np.mean(std_diff_list)
    return std_diff_mean > 15, std_diff_mean  # empirical threshold


if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = args.data_dir
    subject = os.path.basename(data_dir)
    preprocess_dir = os.path.join(data_dir, "preprocess")

    model_dir = config.model_dir

    img_morphology_name = os.path.join(data_dir, "aortic_flow.nii.gz")
    if not os.path.exists(img_morphology_name):
        logger.error(f"{subject}: Phase Contrast Cine MRI file does not exist")
        sys.exit(1)

    # * Before segmentation, we need to first correct the bias of the Nifti file.

    nim_img = nib.load(img_morphology_name)
    img_morphology = nim_img.get_fdata()
    img_morphology_3d = img_morphology[:, :, 0, :]
    img_morphology_3d += 1  # add offset

    # * Step1: Bias correction
    abnormaly, std_diff_mean = check_brightness_anomaly(img_morphology)
    if abnormaly:
        logger.info(f"{subject}: Brightness anomaly found. Applying bias correction.")
        os.makedirs(preprocess_dir, exist_ok=True)
        if std_diff_mean < 15:
            # Run one time
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(img_morphology_3d[:, :, 0], cmap="gray")
            plt.title("Original")
            plt.axis("off")

            img_morphology_3d_bias_corrected = ants.n4_bias_field_correction(ants.from_numpy(img_morphology_3d), verbose=True)
            img_morphology_3d_bias_corrected = img_morphology_3d_bias_corrected.numpy()
            plt.subplot(1, 2, 2)
            plt.imshow(img_morphology_3d_bias_corrected[:, :, 0], cmap="gray")
            plt.title("Bias Corrected 1st")
            plt.axis("off")

        elif std_diff_mean >= 15 and std_diff_mean < 20:
            # Run two times
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(img_morphology_3d[:, :, 0], cmap="gray")
            plt.title("Original")
            plt.axis("off")

            img_morphology_3d_bias_corrected = ants.n4_bias_field_correction(ants.from_numpy(img_morphology_3d), verbose=True)
            img_morphology_3d_bias_corrected = img_morphology_3d_bias_corrected.numpy()
            plt.subplot(1, 3, 2)
            plt.imshow(img_morphology_3d_bias_corrected[:, :, 0], cmap="gray")
            plt.title("Bias Corrected 1st")
            plt.axis("off")

            img_morphology_3d_bias_corrected = ants.n4_bias_field_correction(
                ants.from_numpy(img_morphology_3d_bias_corrected), verbose=True
            )
            img_morphology_3d_bias_corrected = img_morphology_3d_bias_corrected.numpy()
            plt.subplot(1, 3, 3)
            plt.imshow(img_morphology_3d_bias_corrected[:, :, 0], cmap="gray")
            plt.title("Bias Corrected 2nd")
            plt.axis("off")

        else:
            # # Run three times
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 4, 1)
            plt.imshow(img_morphology_3d[:, :, 0], cmap="gray")
            plt.title("Original")
            plt.axis("off")

            img_morphology_3d_bias_corrected = ants.n4_bias_field_correction(
                ants.from_numpy(img_morphology_3d), convergence={"iters": [100, 100, 100, 50], "tol": 1e-8}, verbose=True
            )
            img_morphology_3d_bias_corrected = img_morphology_3d_bias_corrected.numpy()
            plt.subplot(1, 4, 2)
            plt.imshow(img_morphology_3d_bias_corrected[:, :, 0], cmap="gray")
            plt.title("Bias Corrected 1st")
            plt.axis("off")

            img_morphology_3d_bias_corrected = ants.n4_bias_field_correction(
                ants.from_numpy(img_morphology_3d_bias_corrected),
                convergence={"iters": [100, 100, 100, 50], "tol": 1e-8},
                verbose=True,
            )
            img_morphology_3d_bias_corrected = img_morphology_3d_bias_corrected.numpy()
            plt.subplot(1, 4, 3)
            plt.imshow(img_morphology_3d_bias_corrected[:, :, 0], cmap="gray")
            plt.title("Bias Corrected 2nd")
            plt.axis("off")

            img_morphology_3d_bias_corrected = ants.n4_bias_field_correction(
                ants.from_numpy(img_morphology_3d_bias_corrected),
                convergence={"iters": [100, 100, 100, 50], "tol": 1e-8},
                verbose=True,
            )
            img_morphology_3d_bias_corrected = img_morphology_3d_bias_corrected.numpy()
            plt.subplot(1, 4, 4)
            plt.imshow(img_morphology_3d_bias_corrected[:, :, 0], cmap="gray")
            plt.title("Bias Corrected 3rd")
            plt.axis("off")

        plt.savefig(f"{preprocess_dir}/aortic_flow_bias_correction.png")
        plt.close()

        img_morphology_3d = img_morphology_3d_bias_corrected
        img_morphology_final = np.expand_dims(img_morphology_3d, axis=2)

        nii_final_affine = nim_img.affine.copy()
        nii_final = nib.Nifti1Image(img_morphology_final, nii_final_affine, nim_img.header)
        nii_final.header["pixdim"][1:4] = nim_img.header["pixdim"][1:4]
        img_morphology_name = os.path.join(data_dir, "aortic_flow_processed.nii.gz")
        nib.save(nii_final, img_morphology_name)
        logger.info(f"{subject}: Phase contrast cine MRI has gone through bias correction. Preprocessing finished.")
    else:
        # copy the file
        shutil.copy(img_morphology_name, os.path.join(data_dir, "aortic_flow_processed.nii.gz"))
        logger.info(f"{subject}: No have brightness anomaly. Skip bias correction.")
