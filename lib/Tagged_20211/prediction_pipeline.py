import time
import os
import re
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import config
from utils.log_utils import setup_logging

from PredictionResult import PredictionResult
import LocalisationCNN.loader as local_loader
import LandmarkTrackingRNNCNN.loader as rnncnn_loader
import prediction_utils as utils
import gif_utils as gif_utils

logger = setup_logging("eval_strain_tagged_pipeline")

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", help="Path to the HDF5 data", type=str, required=True)
parser.add_argument("--show_images", help="Show the first cine to screen or save", type=bool, default=True)
parser.add_argument("--save_to_gif", help="Save to gif is only used when show_images is True", type=bool, default=True)
parser.add_argument("--gif_path", help="Path to save the gif", type=str, default=None)


# ============== Data path ==============
file_ext_pattern = r"^(?!.*_result).*\.h5$"
batch_size = 20

# ============== Network models config ==============
# 1. Localisation Network
# localisation_network_path = './models/LocalCNN'
localisation_network_path = f"{config.model_dir}/Tagged_20211/LocalCNN/"
localisation_network_name = "localizer"
# 2. RNNCNN Network
# rnncnn_network_path = './models/LandmarkTrackingRNNCNN'
rnncnn_network_path = f"{config.model_dir}/Tagged_20211/LandmarkTrackingRNNCNN/"
rnncnn_network_name = "rnncnn"


if __name__ == "__main__":
    args = parser.parse_args()

    show_images = args.show_images
    save_to_gif = args.save_to_gif
    data_path = args.data_path
    gif_path = args.gif_path

    output_path = data_path
    subject_path = os.path.dirname(data_path)

    # traverse the input folder, keep as array to combine later
    files = os.listdir(data_path)
    # get all the .h5 input filenames that are not results
    input_files = [f for f in files if re.match(file_ext_pattern, f)]

    logger.info("{} files found!".format(len(input_files)))
    logger.info(input_files)

    # Load both networks here..let's see
    logger.info("Loading networks...")
    localnet = local_loader.NetworkModelCNN(localisation_network_path, localisation_network_name)
    landmarknet = rnncnn_loader.NetworkModelRNNCNN(rnncnn_network_path, rnncnn_network_name)

    # loop through all the input files, and run the network
    for num, input_file in enumerate(input_files):
        logger.info("\n--------------------------")
        logger.info("\nProcessing {} ({}/{}) - {}".format(input_file, num + 1, len(input_files), time.ctime()))
        start_time = time.time()

        # 0. Load the data
        input_filepath = os.path.join(data_path, input_file)
        # Check datasize for batching
        data_size = utils.get_dataset_len(input_filepath)

        # Do the prediction per batch
        for pos in range(0, data_size, batch_size):
            logger.info(f"\rProcessed {pos}/{data_size} ...")
            img_sequences = utils.load_dataset(input_filepath, pos, batch_size)

            # 1. Predict bounding box
            # we only need the ED frames
            ed_imgs = img_sequences[:, 0, :, :]  # frame t0 only, ED frame
            corners = localnet.predict_corners(ed_imgs)

            # 2. Localize the image based on predicted bounding box
            cropped_frames, resize_ratios = utils.crop_and_resize_all_frames(img_sequences, corners)

            # 3. Predict localized landmarks
            landmarks = landmarknet.predict_landmark_sequences(cropped_frames)

            # 4. Prepare to save results
            results = PredictionResult(corners, landmarks, resize_ratios)

            # 5. Calculate strains
            results.calculate_strains()

            # 6. Save results
            output_prefix = input_file[: -len(".h5")]  # strip the extension
            results.save_predictions(output_path, output_prefix)
        # End of batch loop
        logger.info(f"\rProcessed {data_size}/{data_size} ...")
        # ----------- Elapsed time -----------
        time_taken = time.time() - start_time
        fps = data_size / time_taken
        logger.info(f"Prediction saved as {output_path}/{output_prefix}")
        logger.info(
            "Prediction pipeline - {} cines: {:.2f} seconds ({:.2f} cines/second)".format(data_size, time_taken, fps)
        )

        if show_images:
            # Just an example, we do this on the first case of the  last batch
            logger.info("Showing/saving first cine case to GIF")
            if not os.path.isdir(gif_path):
                os.makedirs(gif_path)
            gif_filename = f"{gif_path}/{input_file.split('.')[0]}.gif"
            gif_utils.prepare_animation(
                img_sequences[0],
                cropped_frames[0],
                results.landmark_sequences[0],
                results.cc_strains[0],
                results.rr_strains[0],
                save_to_gif=save_to_gif,
                gif_filepath=gif_filename,
            )

    logger.info("\n====== Done! ======")
