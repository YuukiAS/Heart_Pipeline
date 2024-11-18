import os
import shutil
import numpy as np
import nibabel as nib
import sys
import re

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


def check_existing_file(files_to_check, existing_path):
    """
    Check if the files in files_to_check already exist in existing_path.
    If so, return True, otherwise return False.
    """
    files_in_dir = os.listdir(existing_path)
    files_to_check = [x for x in files_to_check if x not in files_in_dir]
    if len(files_to_check) == 0:
        return True
    return False


def setup_segmentation_folders(ID: int, name: str, trained_model_path: str):
    """
    Set up the folders to run the segmentation models: nnUNet or U-Mamba.

    Arguments:
    - ID: ID of modality, e.g. 20212
    - name: Name of the modality, e.g. LVOT
    - trained_model_path: The path to the trained model,
        e.g. /work/users/y/u/yuukias/Heart_Pipeline/model/LVOT_20212/Dataset20212_LVOT/nnUNetTrainer__nnUNetPlans__2d
    """
    temp_dir = config.temp_dir

    raw_folder = os.path.join(temp_dir, "nnUNet_raw")
    preprocessed_folder = os.path.join(temp_dir, "nnUNet_preprocessed")
    results_folder = os.path.join(temp_dir, "nnUNet_results")
    target_model_path = os.path.join(results_folder, f"Dataset{ID}_{name}", os.path.basename(trained_model_path))

    if not os.path.exists(raw_folder):
        os.makedirs(raw_folder)

    if not os.path.exists(preprocessed_folder):
        os.makedirs(preprocessed_folder)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    if not os.path.exists(target_model_path):
        shutil.copytree(trained_model_path, target_model_path)


def prepare_files_to_segment(nii_path: str, subject: str, ID: int, name: str):
    nii = nib.load(nii_path).get_fdata()
    nii_affine = nib.load(nii_path).affine.copy()
    if nii.ndim != 4:
        raise ValueError("The input nii file should have 4 dimensions")

    temp_dir = config.temp_dir
    imagesTs_folder = os.path.join(temp_dir, "nnUNet_raw", f"Dataset{ID}_{name}", "imagesTs")
    labelsTs_folder = os.path.join(temp_dir, "nnUNet_raw", f"Dataset{ID}_{name}", "labelsTs")  # used to store predictions

    os.makedirs(imagesTs_folder, exist_ok=True)
    os.makedirs(labelsTs_folder, exist_ok=True)

    for s in range(nii.shape[2]):
        for t in range(nii.shape[3]):
            nii_s_t = nii[:, :, s, t]
            nii_s_t_name = os.path.join(imagesTs_folder, f"{name}_{subject}_{s:02}_{t:02}_0000.nii.gz")
            nii_s_t_file = nib.Nifti1Image(nii_s_t, nii_affine, nib.load(nii_path).header)
            nib.save(nii_s_t_file, nii_s_t_name)


def run_segment_code(subject: str, ID: int, name: str, model):
    temp_dir = config.temp_dir
    imagesTs_folder = os.path.join(temp_dir, "nnUNet_raw", f"Dataset{ID}_{name}", "imagesTs")
    labelsTs_folder = os.path.join(temp_dir, "nnUNet_raw", f"Dataset{ID}_{name}", "labelsTs")  # used to store predictions

    if model == "nnUNet":
        # nnUNetv2_predict -d $dataset \
        #                  -i "/work/users/y/u/yuukias/Annotation/nnUNet/data/nnUNet_raw/${dataset}/imagesTs" \
        #                  -o "/work/users/y/u/yuukias/Annotation/nnUNet/data/nnUNet_raw/${dataset}/labelsTs" \
        #                  -f  all \
        #                  -tr nnUNetTrainer \
        #                  -c 2d \
        #                  -p nnUNetPlans
        os.system(
            "nnUNetv2_predict "
            f"-d Dataset{ID}_{name} "
            f"-i {imagesTs_folder} "
            f"-o {labelsTs_folder} "
            "-f all "
            "-tr nnUNetTrainer "
            "-c 2d "
            "-p nnUNetPlans"
        )
    elif model == "UMamba":
        # nnUNetv2_predict -i "/work/users/y/u/yuukias/Annotation/UMamba/U-Mamba/data/nnUNet_raw/${dataset}/imagesTs" \
        #                  -o "/work/users/y/u/yuukias/Annotation/UMamba/U-Mamba/data/nnUNet_raw/${dataset}/labelsTs" \
        #                  -d $ID \
        #                  -c 2d \
        #                  -f all \
        #                  -tr nnUNetTrainerUMambaBot \
        #                  --disable_tta
        os.system(
            "nnUNetv2_predict "
            f"-d Dataset{ID}_{name} "
            f"-i {imagesTs_folder} "
            f"-o {labelsTs_folder} "
            "-f all "
            "-tr nnUNetTrainerUMambaBot "
            "-c 2d "
            "--disable_tta"
        )
    else:
        raise ValueError("Model should be either nnUNet or UMamba")


def obtain_files_segmented(subject: str, ID: int, name: str, target_name: str):
    temp_dir = config.temp_dir
    imagesTs_folder = os.path.join(temp_dir, "nnUNet_raw", f"Dataset{ID}_{name}", "imagesTs")
    labelsTs_folder = os.path.join(temp_dir, "nnUNet_raw", f"Dataset{ID}_{name}", "labelsTs")  # used to store predictions

    pattern_image = re.compile(r".*_(\d{2})_(\d{2})_0000\.nii\.gz$")
    pattern_label = re.compile(r".*_(\d{2})_(\d{2})\.nii\.gz$")
    slice_indices = []
    timeframe_indices = []

    file_first = None
    for file_name in os.listdir(labelsTs_folder):
        match = pattern_label.match(file_name)
        if match:
            s = int(match.group(1))
            t = int(match.group(2))
            slice_indices.append(s)
            timeframe_indices.append(t)

            if not file_first:
                file_first = os.path.join(labelsTs_folder, file_name)

    S = max(slice_indices) + 1
    T = max(timeframe_indices) + 1
    H = nib.load(file_first).shape[0]
    W = nib.load(file_first).shape[1]

    # Merge the segmented images

    nii = np.zeros((H, W, S, T), dtype=np.uint8)
    nii_affine = nib.load(file_first).affine.copy()

    cnt = 0
    for file_name in os.listdir(labelsTs_folder):
        match = pattern_label.match(file_name)
        if match:
            cnt += 1
            s = int(match.group(1))
            t = int(match.group(2))
            nii_s_t = nib.load(os.path.join(labelsTs_folder, file_name)).get_fdata()
            nii[:, :, s, t] = nii_s_t

    if cnt != S * T:
        raise ValueError("The number of segmented images does not match the expected number of slices and timeframes")

    for file_name in os.listdir(imagesTs_folder):
        match = pattern_image.match(file_name)
        if match:
            os.remove(os.path.join(imagesTs_folder, file_name))

    for file_name in os.listdir(labelsTs_folder):
        match = pattern_label.match(file_name)
        if match:
            os.remove(os.path.join(labelsTs_folder, file_name))
    
    nii_file = nib.Nifti1Image(nii, nii_affine, nib.load(file_first).header)
    nii_file.header["pixdim"][1:4] = nib.load(file_first).header["pixdim"][1:4]
    nib.save(nii_file, target_name)
