import os
import argparse
import sys
import SimpleITK as sitk

# sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from lib.Aortic_Scout_20207.train_operations.inference_2d_seg import single_image_segmentation

import config
from utils.log_utils import setup_logging

logger = setup_logging("segment_aortic_scout")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type = str, help="Folder for one subject that contains Nifti files", required=True)


if __name__ == "__main__":
    args = parser.parse_args()

    model_dir = config.model_dir

    data_dir = args.data_dir
    subject = os.path.basename(data_dir)

    aorta_name = os.path.join(data_dir, "aortic_scout.nii.gz")
    if not os.path.exists(aorta_name):
        logger.error(f"Aorta structure file for {subject} does not exist")
        sys.exit(1)
    
    # * Resample the aorta_volume from (240, 240, 20) to (238, 238, 136) so that code can be implemented
    new_spacing = [1.66667, 1.66667, 1.66667]
    new_size = [238, 238, 136]
    interpolator = sitk.sitkLinear

    aorta_volume = sitk.ReadImage(aorta_name)
    aorta_volume = sitk.DICOMOrient(aorta_volume, desiredCoordinateOrientation="LPS")  # adjust orientation to LPS

    original_spacing = aorta_volume.GetSpacing()
    original_size = aorta_volume.GetSize()

    aorta_volume_resampled = sitk.Resample(
        aorta_volume,
        new_size,
        sitk.Transform(),
        interpolator,
        aorta_volume.GetOrigin(),
        new_spacing,
        aorta_volume.GetDirection(),
        0,
        aorta_volume.GetPixelID(),
    )
    sitk.WriteImage(aorta_volume_resampled, aorta_name)  # update the volume

    single_image_segmentation(
        image_path=aorta_name, 
        data_output_path=data_dir,
        weight_path=os.path.join(config.model_dir, "Aortic_Scout_20207", "alt_unet.pth")
    )