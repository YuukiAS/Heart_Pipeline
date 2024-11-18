import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import argparse
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ECG_6025_20205.ecg_processor import ECG_Processor

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import config
from utils.log_utils import setup_logging
from utils.analyze_utils import analyze_time_series_derivative

logger = setup_logging("eval_strain_tagged")

parser = argparse.ArgumentParser()
parser.add_argument("--retest", action="store_true")
parser.add_argument("--data_list", help="List of subjects to be processed", nargs="*", type=int, required=True)
parser.add_argument("--file_name", help="Name of the csv file to save the features")


def prepare_h5_file(nii):
    """
    Prepare the h5 file for the tagged MRI.
    Each frame was zero-padded to 256*256
    Slices with fewer than 20 frames were padded with empty frames
    Slices with more than 20 frames were truncated by taking the first 20 frames
    """
    if nii.ndim != 4:
        raise ValueError(f"Data shape is {nii.shape}, which is not 4D")
    data = nii[:, :, 0, :]  # only one slice for each file
    data = np.transpose(data, (2, 0, 1))

    if data.shape[1] > 256 or data.shape[2] > 256:
        raise ValueError(f"Data shape is {data.shape}, which is larger than 256*256")

    if data.shape[1] < 256 or data.shape[2] < 256:
        # pad the data
        pad_x = (256 - data.shape[1]) // 2
        pad_y = (256 - data.shape[2]) // 2
        data = np.pad(data, ((0, 0), (pad_x, pad_x), (pad_y, pad_y)), mode="constant")

    if data.shape[0] < 20:
        # pad with empty frames
        data = np.concatenate([data, np.zeros((20 - data.shape[0], 256, 256))], axis=0)
    elif data.shape[0] > 20:
        # truncate the data
        data = data[:20]
    data = data.reshape(1, 20, 256, 256)

    return data


if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = config.data_visit2_dir if args.retest else config.data_visit1_dir

    df = pd.DataFrame()

    for subject in tqdm(args.data_list):
        subject = str(subject)
        ID = int(subject)
        logger.info(f"Calculating circumferential and radial strain using Tagged MRI for subject {subject}")
        sub_dir = os.path.join(data_dir, subject)

        tag1_name = os.path.join(sub_dir, "tag_1.nii.gz")  # basal
        tag2_name = os.path.join(sub_dir, "tag_2.nii.gz")  # mid-ventricle
        tag3_name = os.path.join(sub_dir, "tag_3.nii.gz")  # apical

        if not os.path.exists(tag1_name) or not os.path.exists(tag2_name) or not os.path.exists(tag3_name):
            logger.error(f"At least one tagged MRI file for {subject} does not exist")
            continue

        tag1 = nib.load(tag1_name).get_fdata()
        tag2 = nib.load(tag2_name).get_fdata()
        tag3 = nib.load(tag3_name).get_fdata()
        tags = [tag1, tag2, tag3]

        temporal_resolution = nib.load(tag1_name).header["pixdim"][4]
        time_grid_point = np.arange(20)  # since we only uses 20 frames
        time_grid_real = time_grid_point * temporal_resolution  # unit: s

        feature_dict = {
            "eid": subject,
        }

        # * Circumferential and Radial Strain
        # Ref Ferdian, Edward, et al. “Fully Automated Myocardial Strain Estimation from Cardiovascular MRI–Tagged...”

        os.makedirs(os.path.join(sub_dir, "tag_hdf5"), exist_ok=True)
        for i, tag_raw in enumerate(tags):
            tag = prepare_h5_file(tag_raw)
            with h5py.File(os.path.join(sub_dir, "tag_hdf5", f"tag{i + 1}.h5"), "w") as h5_file:
                h5_file.create_dataset("image_seqs", data=tag)

        logger.info(f"{subject}: Running prediction pipeline for tagged MRI")
        os.makedirs(os.path.join(sub_dir, "visualization", "ventricle"), exist_ok=True)
        gif_path = os.path.join(sub_dir, "visualization", "ventricle")
        os.system(
            f"python {config.lib_dir}/Tagged_20211/prediction_pipeline.py --data_path={sub_dir}/tag_hdf5 --gif_path={gif_path}"
        )

        logger.info(f"{subject}: Extracting strain features from prediction pipeline")
        names = ["basal", "mid", "apical"]
        os.makedirs(os.path.join(sub_dir, "timeseries"), exist_ok=True)
        for i, _ in enumerate(tags):
            with h5py.File(os.path.join(sub_dir, "tag_hdf5", f"tag{i + 1}_result.h5"), "r") as h5_file:
                # circumferential strain
                circum_strain = h5_file["cc_linear_strains"][:]  # linear strain
                circum_strain = circum_strain[0, :, 7]
                # Ref landmark tracking section
                circum_strain_squared = h5_file["cc_strains"][:]  # squared strain, we temporarily not use this
                circum_strain_squared = circum_strain_squared[0, :, 7]
                plt.plot(circum_strain)  # last channel is the average for circumferential strain
                plt.plot(circum_strain_squared)
                plt.legend(["linear strain", "squared strain"])
                plt.title(f"{ID}: Tagged Circumferential Strain in {names[i]} slice")
                plt.xlabel("Time [frame]")
                plt.ylabel("Strain [%]")
                plt.savefig(os.path.join(sub_dir, f"timeseries/tagged_circum_strain_{names[i]}.png"))
                plt.close()

                # radial strain
                radial_strain = h5_file["rr_linear_strains"][:]
                radial_strain = radial_strain[0, :, 0]
                radial_strain_squared = h5_file["rr_strains"][:]
                radial_strain_squared = radial_strain_squared[0, :, 0]
                plt.plot(radial_strain)
                plt.plot(radial_strain_squared)
                plt.legend(["linear strain", "squared strain"])
                plt.title(f"{ID}: Tagged Radial Strain in {names[i]} slice")
                plt.xlabel("Time [frame]")
                plt.ylabel("Strain [%]")
                plt.savefig(os.path.join(sub_dir, f"timeseries/tagged_radial_strain_{names[i]}.png"))
                plt.close()

                feature_dict.update(
                    {
                        f"Tagged MRI-{names[i]}: Circumferential strain [%]": -circum_strain.min() * 100,
                        f"Tagged MRI-{names[i]}: Radial strain [%]": radial_strain.max() * 100,
                    }
                )

                # * We replicate the setting for short axis, using only linear strains

                # * Feature 1 Strain Rate
                # * Since there are only 20 frames, not covering the entire cardiac cycle, we don't calculate late diastole SR

                try:
                    T_GCSR_pos, T_GCSR_neg, GCSR_pos, GCSR_neg = analyze_time_series_derivative(
                        time_grid_real,
                        circum_strain,
                        n_pos=1,
                        n_neg=1,
                        method="loess",  # for strain rate, we don't use the moving average method
                    )
                    print(GCSR_neg, GCSR_pos)
                    GCSR_S = GCSR_neg[0]
                    GCSR_E = GCSR_pos[0]
                    feature_dict.update(
                        {
                            f"Tagged MRI-{names[i]}: Circumferential Strain Rate (Peak-systolic) [1/s]": GCSR_S,
                            f"Tagged MRI-{names[i]}: Circumferential Strain Rate (Early-diastole) [1/s]": GCSR_E,
                        }
                    )
                    logger.info(f"{subject}: Global circumferential strain rate calculated.")
                except IndexError:  # in some cases there will even be no early diastole strain rate
                    feature_dict.update(
                        {
                            f"Tagged MRI-{names[i]}: Circumferential Strain Rate (Peak-systolic) [1/s]": GCSR_S,
                        }
                    )
                    logger.warning(f"{subject}: Only peak-systolic circumferential strain rate calculated.")
                except ValueError as e:
                    logger.warning(f"{subject}: {e}  No global circumferential strain rate calculated.")

                try:
                    T_GRSR_pos, T_GRSR_neg, GRSR_pos, GRSR_neg = analyze_time_series_derivative(
                        time_grid_real,
                        radial_strain,
                        n_pos=1,
                        n_neg=1,
                        method="loess",
                    )
                    GRSR_S = GRSR_pos[0]
                    GRSR_E = GRSR_neg[0]
                    feature_dict.update(
                        {
                            "Tagged MRI-{names[i]}: Radial Strain Rate (Peak-systolic) [1/s]": GRSR_S,
                            "Tagged MRI-{names[i]}: Radial Strain Rate (Early-diastole) [1/s]": GRSR_E,
                        }
                    )
                    logger.info(f"{subject}: Global radial strain rate calculated.")
                except IndexError:
                    feature_dict.update(
                        {
                            "Tagged MRI-{names[i]}: Radial Strain Rate (Peak-systolic) [1/s]": GRSR_S,
                        }
                    )
                    logger.warning(f"{subject}: Only peak-systolic radial strain rate calculated.")
                except ValueError as e:
                    logger.warning(f"{subject}: {e} No global radial strain rate calculated.")

                # * Feature: Time to peak interval
                T_circum_strain_peak = np.argmin(circum_strain)
                T_radial_strain_peak = np.argmax(radial_strain)

                t_circum_strain_peak = T_circum_strain_peak * temporal_resolution * 1000  # unit: ms
                t_radial_strain_peak = T_radial_strain_peak * temporal_resolution * 1000

                feature_dict.update(
                    {
                        f"Tagged MRI-{names[i]}: Circumferential Strain: Time to Peak [ms]": t_circum_strain_peak,
                        f"Tagged MRI-{names[i]}: Radial Strain: Time to Peak [ms]": t_radial_strain_peak,
                    }
                )

                ecg_processor = ECG_Processor(subject, args.retest)

                if config.useECG and not ecg_processor.check_data_rest():
                    logger.warning(f"{subject}: No ECG rest data, time to peak strain index will not be calculated.")
                else:
                    logger.info(f"{subject}: ECG rest data exists, calculate time to peak strain index.")
                    RR_interval = ecg_processor.determine_RR_interval()  # should be close to MeanNN in neurokit2

                    t_circum_strain_peak_index = t_circum_strain_peak / RR_interval
                    t_radial_strain_peak_index = t_radial_strain_peak / RR_interval

                    feature_dict.update(
                        {
                            f"Tagged MRI-{names[i]}: Circumferential Strain: Time to Peak Index": t_circum_strain_peak_index,
                            f"Tagged MRI-{names[i]}: Radial Strain: Time to Peak Index": t_radial_strain_peak_index,
                        }
                    )

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "strain")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df = df[[col for col in df.columns if col != "eid"] + ["eid"]]  # move 'eid' to the last column
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), index=False)
