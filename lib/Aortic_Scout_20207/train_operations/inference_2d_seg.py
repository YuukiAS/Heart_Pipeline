import numpy as np
import os
import torch
import torchio as tio
import SimpleITK as sitk
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from configuration import config
from data_operations.segmentation_dataset import (
    process_slice_inferencing,
    prepare_3d_for_2d,
    process_slice_mask_inferencing,
)

# from networks.UNet import *
from networks.Slice_UNet import Slice_UNet
from train_operations.make_experiment import make_experiment
from data_operations.utils import process_excel
from data_operations.image_utils import generate_DSC, itk_image_stack, channelwise_normalization
from data_operations.phenotypes import Aorta


def perform_inferencing(weight_path="./weights/alt_unet.pth"):
    """
    Currently for inputs of 3D images to be converted then normalized
    """
    print("[INFO] 2D image segmentation inferencing")
    make_experiment()  # create experiment directories

    test_images, test_masks, test_pid = process_excel(config.test_path, return_pid=True)
    prediction_path = config.output_path + config.experiment_name + "/predictions/"

    mask_volume_arr, prediction_volume_arr, dsc_arr, image_arr, mask_arr, prediction_arr = [], [], [], [], [], []
    model = Slice_UNet(1, 1)

    model.load_state_dict(torch.load(weight_path))
    model = model.cuda()

    model.eval()

    for img, msk in zip(test_images, test_masks):
        itk_image = sitk.ReadImage(img)
        itk_mask = sitk.ReadImage(msk)

        spacing = itk_image.GetSpacing()
        slice_arr = itk_image_stack(itk_image=itk_image)

        dataset = process_slice_inferencing(slice_arr)
        mask_dataset = process_slice_mask_inferencing(itk_image_stack(itk_mask))
        output_arr = []
        with torch.no_grad():
            for subject in dataset:
                x = subject["image"][tio.DATA]
                x = torch.squeeze(x, dim=-1)
                x = torch.unsqueeze(x, dim=0)
                x = x.cuda()
                x = x.float()
                y_pred = model(x)
                y_pred = torch.softmax(y_pred, dim=1)
                prediction = y_pred.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
                prediction = torch.squeeze(prediction)
                # prediction = torch.unsqueeze(prediction, dim=0)
                prediction = prediction.cpu().numpy()
                prediction = np.swapaxes(prediction, 0, -1)

                prediction = sitk.GetImageFromArray(prediction[1:-1, 1:-1])
                prediction.SetSpacing(spacing)

                output_arr.append(prediction)

        result_image = sitk.JoinSeries(output_arr)
        result_image.SetSpacing(spacing)

        itk_mask = sitk.Cast(itk_mask, sitk.sitkInt64)
        if config.generate_metrics:
            dsc_arr.append(generate_DSC(result_image, itk_mask))
            print(generate_DSC(result_image, itk_mask))

    print("[INFO] DSC average: {}".format(np.mean(dsc_arr)))
    print("[INFO] DSC std: {}".format(np.std(dsc_arr)))


# * We separate the original `single_image_inference` function into two parts:
# * one for segmentation and one for inference (feature extraction)
def single_image_segmentation(image_path, data_output_path, weight_path="./weights/alt_unet.pth"):
    """
    Function processes a single image and generates and writes output data.
    :image_path: path to a NIFTI (.nii or .nii.gz) image file
    :data_output_path: path for writing output to
    :weight_path: path to a U-Net weight file (model should be predefined in code currently)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # extract pid and generate prediction path
    # pid = extract_pid(image_path)
    # print("[INFO] performing inferencing on subject: {}".format(pid))
    prediction_path = os.path.join(data_output_path, "seg_aortic_scout.nii.gz")

    # read and extract z slice stack of images
    itk_image = sitk.ReadImage(image_path)
    slice_arr = itk_image_stack(itk_image=itk_image)

    # extract spacing and origin
    spacing = itk_image.GetSpacing()
    # origin = itk_image.GetOrigin()

    dataset = process_slice_inferencing(slice_arr)

    # load model and place on cpu
    model = Slice_UNet(1, 1)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    output_arr = []
    with torch.no_grad():
        for subject in dataset:
            x = subject["image"][tio.DATA]
            x = torch.squeeze(x, dim=-1)
            x = torch.unsqueeze(x, dim=0)

            x = x.float().to(device)
            y_pred = model(x)
            y_pred = torch.softmax(y_pred, dim=1)
            prediction = y_pred.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
            prediction = torch.squeeze(prediction)
            # prediction = torch.unsqueeze(prediction, dim=0)
            prediction = prediction.cpu().numpy()
            prediction = np.swapaxes(prediction, 0, -1)
            prediction = sitk.GetImageFromArray(prediction)
            prediction.SetSpacing(spacing)

            output_arr.append(prediction)

    result_image = sitk.JoinSeries(output_arr)
    result_image.SetSpacing(spacing)

    sitk.WriteImage(result_image, prediction_path)  # (240, 208, 100)


def single_image_inference(image_path, label_path, data_output_path):
    """
    Function processes a single image and generates and writes output data.
    :image_path: path to a NIFTI (.nii or .nii.gz) image file
    :data_output_path: path for writing output to
    :weight_path: path to a U-Net weight file (model should be predefined in code currently)
    """

    label = sitk.ReadImage(label_path)

    aorta_1 = Aorta(label, image_path, data_output_path)
    # generate vtk_mesh and centerline
    aorta_1.process_itk_data()
    # extract ascending and descending aorta regions
    aorta_1.generate_aorta_regions()
    aorta_1.generate_diameter()
    aorta_1.generate_centerline_length()
    aorta_1.generate_arch_height_width()
    aorta_1.generate_tortuosity()
    aorta_1.generate_curvature_torsion()
    return aorta_1.get_data()


def single_image_mask(image_path, data_output_path, weight_path="./weights/alt_unet.pth"):
    """
    Function processes a single image and generates and writes output data.
    :image_path: path to a NIFTI (.nii or .nii.gz) image file
    :data_output_path: path for writing output to
    :weight_path: path to a U-Net weight file (model should be predefined in code currently)
    """
    # extract pid and generate prediction path
    pid = extract_pid(image_path)
    print("[INFO] performing inferencing on subject: {}".format(pid))
    prediction_path = data_output_path + pid + "_prediction.nii"

    # read and extract z slice stack of images
    itk_image = sitk.ReadImage(image_path)
    slice_arr = itk_image_stack(itk_image=itk_image)

    # extract spacing and origin
    spacing = itk_image.GetSpacing()
    origin = itk_image.GetOrigin()

    dataset = process_slice_inferencing(slice_arr)

    # load model and placce on cpu
    model = Slice_UNet(1, 1)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device("cpu")))
    model.eval()
    output_arr = []
    with torch.no_grad():
        for subject in dataset:
            x = subject["image"][tio.DATA]
            x = torch.squeeze(x, dim=-1)
            x = torch.unsqueeze(x, dim=0)

            x = x.float()
            y_pred = model(x)
            y_pred = torch.softmax(y_pred, dim=1)
            prediction = y_pred.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
            prediction = torch.squeeze(prediction)
            # prediction = torch.unsqueeze(prediction, dim=0)
            prediction = prediction.cpu().numpy()
            prediction = np.swapaxes(prediction, 0, -1)
            prediction = sitk.GetImageFromArray(prediction)
            prediction.SetSpacing(spacing)

            output_arr.append(prediction)

    result_image = sitk.JoinSeries(output_arr)
    result_image.SetSpacing(spacing)

    # sitk.WriteImage(result_image, prediction_path)

    aorta_1 = Aorta(result_image, image_path, data_output_path)
    aorta_1.process_itk_data()
    aorta_1.save_vtk_mesh()


def old_single_image_inference(image_path, data_output_path, weight_path="./weights/alt_unet.pth"):
    # extract pid and generate prediction path
    pid = extract_pid(image_path)
    prediction_path = data_output_path + pid + "_prediction.nii"

    itk_image = sitk.ReadImage(image_path)

    spacing = itk_image.GetSpacing()
    origin = itk_image.GetOrigin()
    itk_image = channelwise_normalization(itk_image)

    subject = tio.Subject(image=tio.ScalarImage.from_sitk(itk_image))
    dataset = prepare_3d_for_2d(subject=subject)

    # load model and placce on cpu
    model = Slice_UNet(1, 1)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device("cpu")))
    model.eval()

    with torch.no_grad():
        for subject in dataset:
            print("[INFO] performing inferencing on subject: {}".format(pid))

            grid_sampler = tio.inference.GridSampler(subject, patch_size=config.patch_size, patch_overlap=0)
            aggregator = tio.inference.GridAggregator(grid_sampler)
            grid_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)

            for batch in grid_loader:
                x = batch["image"][tio.DATA]
                locations = batch[tio.LOCATION]
                if config.use_slice:
                    x = torch.squeeze(x, dim=-1)

                x = x.float()
                y_pred = model(x)
                y_pred = torch.softmax(y_pred, dim=1)
                y_pred = y_pred.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
                if config.use_slice:
                    y_pred = torch.unsqueeze(y_pred, dim=-1)

                aggregator.add_batch(y_pred, locations)

            prediction = aggregator.get_output_tensor()
            prediction = prediction.squeeze().cpu().numpy()
            prediction = np.swapaxes(prediction, 0, -1)

    # Completed inferencing one subject

    # prediction = np.swapaxes(prediction, 0, -1)

    # need to set spacing and origin
    img = sitk.GetImageFromArray(prediction)
    img.SetSpacing(spacing=spacing)
    # img.SetOrigin(origin=origin)
    print(img)

    sitk.WriteImage(img, prediction_path)


# used for extracting image inferencing with pathname only
def extract_pid(path):
    f_name = path.split("/")[-1]
    pid = f_name.split(".nii")[0]
    return pid
