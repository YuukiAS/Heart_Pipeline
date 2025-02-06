"""
Most published papers calculate strain using tagged MRI based on the harmonic phase (HARP) using the Diagnosoft Main Software.
Instead, we direcly base on an existing pipeline to calculate the strain from tagged MRI.

For Reference:
Imaging heart motion using harmonic phase MRI.https://doi.org/10.1109/42.845177
Fully Automated Myocardial Strain Estimation from Cardiovascular MRI-tagged Images https://doi.org/10.1148/ryct.2020190032
"""

import os
import shutil
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import argparse
from tqdm import tqdm
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ECG_6025_20205.ecg_processor import ECG_Processor

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import config
from utils.log_utils import setup_logging
from utils.analyze_utils import plot_time_series_dual_axes, analyze_time_series_derivative

logger = setup_logging("eval_strain_tagged")

parser = argparse.ArgumentParser()
parser.add_argument("--retest", action="store_true")
parser.add_argument("--data_list", help="List of subjects to be processed", nargs="*", type=int, required=True)
parser.add_argument("--file_name", help="Name of the csv file to save the features")


def prepare_h5_file(nii):
    """
    Prepare the h5 file for the Tagged MRI.
    Each frame was zero-padded to 256*256
    Slices with fewer than 20 frames were padded with empty frames
    Slices with more than 20 frames were truncated by taking the first 20 frames
    """
    if nii.ndim != 4:
        raise ValueError(f"Data shape is {nii.shape}, which is not 4D")

    data = nii[:, :, 0, :]  # only one slice in each file
    data = np.transpose(data, (2, 0, 1))
    data_frame = data.shape[0]

    if data.shape[1] > 256 or data.shape[2] > 256:
        raise ValueError(f"Data shape is {data.shape}, which is larger than 256*256")

    if data.shape[1] < 256 or data.shape[2] < 256:
        # zero pad the data
        logger.info(f"Zero padding the data from {data.shape[1]}*{data.shape[2]} to 256*256.")
        pad_x = (256 - data.shape[1]) // 2
        pad_y = (256 - data.shape[2]) // 2
        data = np.pad(data, ((0, 0), (pad_x, pad_x), (pad_y, pad_y)), mode="constant")

    if data.shape[0] < 20:
        # pad with empty frames
        logger.info(f"Padding the data from {data.shape[0]} to 20 frames.")
        data = np.concatenate([data, np.zeros((20 - data.shape[0], 256, 256))], axis=0)
    elif data.shape[0] > 20:
        # truncate the data
        logger.info(f"Truncating the data from {data.shape[0]} to 20 frames.")
        data = data[:20]
    data = data.reshape(1, 20, 256, 256)

    return data, min(data_frame, 20)


if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = config.data_visit2_dir if args.retest else config.data_visit1_dir

    df = pd.DataFrame()

    for subject in tqdm(args.data_list):
        subject = str(subject)
        ID = int(subject)
        logger.info(f"Calculating circumferential and radial strain using Tagged MRI for subject {subject}")
        sub_dir = os.path.join(data_dir, subject)

        tag1_path = os.path.join(sub_dir, "tag_1.nii.gz")  # basal
        tag2_path = os.path.join(sub_dir, "tag_2.nii.gz")  # mid-ventricle
        tag3_path = os.path.join(sub_dir, "tag_3.nii.gz")  # apical

        if not os.path.exists(tag1_path) or not os.path.exists(tag2_path) or not os.path.exists(tag3_path):
            logger.error(f"At least one Tagged MRI file for {subject} does not exist")
            continue

        tag1 = nib.load(tag1_path).get_fdata()
        tag2 = nib.load(tag2_path).get_fdata()
        tag3 = nib.load(tag3_path).get_fdata()
        tag_files = [tag1, tag2, tag3]
        tag_names = ["basal", "mid", "apical"]

        feature_dict = {
            "eid": subject,
        }

        # * Obtain Circumferential and Radial Strain
        # Ref Fully Automated Myocardial Strain Estimation from Cardiovascular MRI-tagged Images https://doi.org/10.1148/ryct.2020190032

        os.makedirs(os.path.join(sub_dir, "temp"), exist_ok=True)
        for i, tag_file in enumerate(tag_files):
            tag_data, T = prepare_h5_file(tag_file)
            # create h5 file for each of three slices.
            with h5py.File(os.path.join(sub_dir, "temp", f"tag_{tag_names[i]}.h5"), "w") as h5_file:
                h5_file.create_dataset("image_seqs", data=tag_data)

        logger.info(f"{subject}: Running prediction pipeline for tagged MRI")
        os.makedirs(os.path.join(sub_dir, "visualization", "ventricle"), exist_ok=True)
        gif_path = os.path.join(sub_dir, "visualization", "ventricle")
        os.system(f"python {config.lib_dir}/Tagged_20211/prediction_pipeline.py --data_path={sub_dir}/temp --gif_path={gif_path}")
        logger.info(f"{subject}: Circumferential and radial strain features for tagged MRI extracted.")

        temporal_resolution = nib.load(tag1_path).header["pixdim"][4] * 1000
        time_grid_point = np.arange(T)
        time_grid_real = time_grid_point * temporal_resolution  # unit: ms

        circum_strain = {}
        radial_strain = {}
        os.makedirs(os.path.join(sub_dir, "timeseries"), exist_ok=True)
        for i, _ in enumerate(tag_files):
            # obtain the results from the returned files generated by the pipeline
            with h5py.File(os.path.join(sub_dir, "temp", f"tag_{tag_names[i]}_result.h5"), "r") as h5_file:
                # Details are described in the landmark tracking section in the paper

                # Circumferential linear strain strain
                # define 7 Circumferential rings of landmarks, the 8th dimension is the mean of all rings
                circum_strain[tag_names[i]] = h5_file["cc_linear_strains"][0, :, 7] * 100  # unit: %
                # If the number of frames is less than 20, we will not make use of empty frames
                if T < 20:
                    circum_strain[tag_names[i]] = circum_strain[tag_names[i]][:T]

                # radial strain
                radial_strain[tag_names[i]] = h5_file["rr_linear_strains"][0, :, 0] * 100
                if T < 20:
                    radial_strain[tag_names[i]] = radial_strain[tag_names[i]][:T]

                # We currently do not make use of squared strains
                # circum_strain_squared = h5_file["cc_strains"][:]
                # circum_strain_squared = circum_strain_squared[0, :, 7]
                # radial_strain_squared = h5_file["rr_strains"][:]
                # radial_strain_squared = radial_strain_squared[0, :, 0]

        circum_strain["global"] = np.mean([circum_strain[tag] for tag in tag_names], axis=0)
        radial_strain["global"] = np.mean([radial_strain[tag] for tag in tag_names], axis=0)

        shutil.rmtree(os.path.join(sub_dir, "temp"))

        tag_names.append("global")

        data_strain = {}

        for tag_name in tag_names:
            tag_name_cap = tag_name.capitalize()

            data_strain.update(
                {
                    f"Strain-Tagged: {tag_name_cap} Circumferential strain [%]": circum_strain[tag_name],
                    f"Strain-Tagged: {tag_name_cap} Radial strain [%]": radial_strain[tag_name],
                }
            )

            feature_dict.update(
                {
                    f"Strain-Tagged: Circumferential strain ({tag_name}) [%]": circum_strain[tag_name].min(),
                    f"Strain-Tagged: Radial strain ({tag_name}) [%]": radial_strain[tag_name].max(),
                }
            )
        np.savez(f"{sub_dir}/timeseries/strain_tagged.npz", **data_strain)

        # * We replicate the setting for short axis, using only the global values

        # We will use smoothed strain for all advanced features
        fANCOVA = importr("fANCOVA")
        x_r = FloatVector(time_grid_real)
        y_r_gcs = FloatVector(circum_strain["global"])
        loess_fit_gcs = fANCOVA.loess_as(x_r, y_r_gcs, degree=2, criterion="gcv")
        GCS_loess_x = np.array(loess_fit_gcs.rx2("x")).reshape(
            T,
        )
        GCS_loess_y = np.array(loess_fit_gcs.rx2("fitted"))

        y_r_grs = FloatVector(radial_strain["global"])
        loess_fit_grs = fANCOVA.loess_as(x_r, y_r_grs, degree=2, criterion="gcv")
        GRS_loess_x = np.array(loess_fit_grs.rx2("x")).reshape(
            T,
        )
        GRS_loess_y = np.array(loess_fit_grs.rx2("fitted"))

        T_circum_strain_peak = np.argmax(abs(GCS_loess_y))
        circum_strain_peak = np.max(abs(GCS_loess_y))

        feature_dict.update({"Strain-SAX: Peak Systolic Circumferential Strain (Absolute Value) [%]": circum_strain_peak})

        T_radial_strain_peak = np.argmax(abs(GRS_loess_y))
        radial_strain_peak = np.max(abs(GRS_loess_y))

        feature_dict.update(
            {
                "Strain-Tagged: Peak Systolic Radial Strain (Absolute Value) [%]": radial_strain_peak,
            }
        )

        logger.info(f"{subject}: Plot time series of global strains.")
        fig, ax1, ax2 = plot_time_series_dual_axes(
            time_grid_point,
            circum_strain["global"],
            "Time [frame]",
            "Time [ms]",
            "Strain [%]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Global Circumferential Strain (GCS) Time Series using Tagged MRI",
        )
        ax1.axvline(x=T_circum_strain_peak, color="red", linestyle="--", alpha=0.7)
        ax1.text(
            T_circum_strain_peak,
            ax1.get_ylim()[0],
            "Peak Systolic Strain",
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=6,
            color="black",
        )
        box_text = "Global Circumferential Strain\n" f"Peak Systolic Strain: {circum_strain_peak:.2f} %"
        box_props = dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8)
        ax1.text(
            0.98,
            0.95,
            box_text,
            transform=ax1.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=box_props,
        )
        fig.savefig(f"{sub_dir}/timeseries/gcs_tagged.png")
        plt.close(fig)

        fig, ax1, ax2 = plot_time_series_dual_axes(
            time_grid_point,
            radial_strain["global"],
            "Time [frame]",
            "Time [ms]",
            "Strain [%]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Global Radial Strain (GRS) Time Series using Tagged MRI",
        )
        ax1.axvline(x=T_radial_strain_peak, color="red", linestyle="--", alpha=0.7)
        ax1.text(
            T_radial_strain_peak,
            ax1.get_ylim()[0],
            "Peak Systolic Strain",
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=6,
            color="black",
        )
        box_text = "Global Radial Strain\n" f"Peak Systolic Strain: {radial_strain_peak:.2f} %"
        box_props = dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8)
        ax1.text(
            0.98,
            0.95,
            box_text,
            transform=ax1.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=box_props,
        )
        fig.savefig(f"{sub_dir}/timeseries/grs_tagged.png")
        plt.close(fig)

        # * Feature 1: Strain Rate

        # Sometimes there will be extreme value for the last frame, we exclude it for calculation
        time_grid_point = time_grid_point[:-1]
        time_grid_real = time_grid_real[:-1]
        radial_strain["global"] = radial_strain["global"][:-1]
        circum_strain["global"] = circum_strain["global"][:-1]
        GCS_loess_x = GCS_loess_x[:-1]
        GCS_loess_y = GCS_loess_y[:-1]
        GRS_loess_x = GRS_loess_x[:-1]
        GRS_loess_y = GRS_loess_y[:-1]

        # Instead of using np.diff, we use np.gradient to ensure the same length
        GCS_diff_y = np.gradient(GCS_loess_y, GCS_loess_x) * 1000  # unit: %/s
        GRS_diff_y = np.gradient(GRS_loess_y, GRS_loess_x) * 1000 

        # Circumferential strain rate
        T_GCSR_S = T_GCSR_E = T_GRSR_S = T_GRSR_E = None
        try:
            # * Note that there are only 20 frames, not covering the entire cardiac cycle, we don't calculate late diastole SR
            T_GCSR_pos, T_GCSR_neg, GCSR_pos, GCSR_neg = analyze_time_series_derivative(
                time_grid_real / 1000,
                circum_strain["global"] / 100,  # Since strain is in %
                n_pos=1,  # set n_pos to 1 instead 2 for SAX
                n_neg=1,
            )
        except IndexError:  # in some cases there will even be no early diastole strain rate
            T_GCSR_pos, T_GCSR_neg, GCSR_pos, GCSR_neg = analyze_time_series_derivative(
                time_grid_real / 1000,
                circum_strain["global"] / 100,
                n_pos=0,
                n_neg=1,
            )
        except ValueError as e:
            logger.warning(f"{subject}: {e}  No global circumferential strain rate calculated.")

        try:
            T_GCSR_S = T_GCSR_neg[0]
            GCSR_S = GCSR_neg[0]
            logger.info(f"{subject}: Peak systolic circumferential strain rate calculated.")
            feature_dict.update(
                {
                    "Strain-Tagged: Peak Systolic Circumferential Strain Rate [1/s]": GCSR_S,
                }
            )
        except IndexError:
            logger.warning(f"{subject}: No peak systolic circumferential strain rate calculated.")
        try:
            T_GCSR_E = T_GCSR_pos[0]
            GCSR_E = GCSR_pos[0]
            logger.info(f"{subject}: Early diastolic circumferential strain rate calculated.")
            feature_dict.update(
                {
                    "Strain-Tagged: Early Diastolic Circumferential Strain Rate [1/s]": GCSR_E,
                }
            )
        except IndexError:
            logger.warning(f"{subject}: No early diastolic circumferential strain rate calculated.")

        # Radial strain rate
        try:
            T_GRSR_pos, T_GRSR_neg, GRSR_pos, GRSR_neg = analyze_time_series_derivative(
                time_grid_real / 1000, radial_strain["global"] / 100, n_pos=1, n_neg=1
            )
        except IndexError:
            T_GRSR_pos, T_GRSR_neg, GRSR_pos, GRSR_neg = analyze_time_series_derivative(
                time_grid_real / 1000, radial_strain["global"] / 100, n_pos=1, n_neg=0
            )
        except ValueError as e:
            logger.warning(f"{subject}: {e} No global radial strain rate calculated.")

        try:
            T_GRSR_S = T_GRSR_pos[0]
            GRSR_S = GRSR_pos[0]
            logger.info(f"{subject}: Peak systolic radial strain rate calculated.")
            feature_dict.update(
                {
                    "Strain-Tagged: Peak Systolic Radial Strain Rate [1/s]": GRSR_S,
                }
            )
        except IndexError:
            logger.warning(f"{subject}: No peak systolic radial strain rate calculated.")
        try:
            T_GRSR_E = T_GRSR_neg[0]
            GRSR_E = GRSR_neg[0]
            logger.info(f"{subject}: Early diastolic radial strain rate calculated.")
            feature_dict.update(
                {
                    "Strain-Tagged: Early Diastolic Radial Strain Rate [1/s]": GRSR_E,
                }
            )
        except IndexError:
            logger.warning(f"{subject}: No early diastolic radial strain rate calculated.")

        # Plot strain rate
        if T_GCSR_S:
            colors = ["blue"] * len(GCS_loess_x)
            colors[T_GCSR_S] = "deepskyblue"
            if T_GCSR_E:
                colors[T_GCSR_E] = "aqua"
            fig, ax1, ax2 = plot_time_series_dual_axes(
                time_grid_point,
                GCS_diff_y / 100,
                "Time [frame]",
                "Time [ms]",
                "Circumferential Strain Rate [1/s]",
                lambda x: x * temporal_resolution,
                lambda x: x / temporal_resolution,
                title=f"Subject {subject}: Global Circumferential Strain Rate Time Series using Tagged MRI",
                colors=colors,
            )
            ax1.axvline(x=T_GCSR_S, color="deepskyblue", linestyle="--", alpha=0.7)
            ax1.text(
                T_GCSR_S,
                ax1.get_ylim()[1],
                "Peak Systolic SR",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=6,
                color="black",
            )
            if T_GCSR_E:
                ax1.axvline(x=T_GCSR_E, color="aqua", linestyle="--", alpha=0.7)
                ax1.text(
                    T_GCSR_E,
                    ax1.get_ylim()[1],
                    "Early Diastolic SR",
                    verticalalignment="bottom",
                    horizontalalignment="center",
                    fontsize=6,
                    color="black",
                )
                box_text = (
                    "Global Circumferential Strain Rate\n"
                    f"Peak Systolic Strain Rate: {GCSR_S:.2f} 1/s\n"
                    f"Early Diastolic Strain Rate: {GCSR_E:.2f} 1/s"
                )
            else:
                box_text = "Global Circumferential Strain Rate\n" f"Peak Systolic Strain Rate: {GCSR_S:.2f} 1/s"
            box_props = dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8)
            ax1.text(
                0.98,
                0.95,
                box_text,
                transform=ax1.transAxes,
                fontsize=8,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=box_props,
            )
            fig.savefig(f"{sub_dir}/timeseries/gcsr_tagged.png")
            plt.close(fig)

        if T_GRSR_S:
            colors = ["blue"] * len(GRS_loess_x)
            colors[T_GRSR_S] = "deepskyblue"
            if T_GRSR_E:
                colors[T_GRSR_E] = "aqua"
            fig, ax1, ax2 = plot_time_series_dual_axes(
                time_grid_point,
                GRS_diff_y / 100,
                "Time [frame]",
                "Time [ms]",
                "Radial Strain Rate [1/s]",
                lambda x: x * temporal_resolution,
                lambda x: x / temporal_resolution,
                title=f"Subject {subject}: Global Radial Strain Rate Time Series using Tagged MRI",
                colors=colors,
            )
            ax1.axvline(x=T_GRSR_S, color="deepskyblue", linestyle="--", alpha=0.7)
            ax1.text(
                T_GRSR_S,
                ax1.get_ylim()[1],
                "Peak Systolic SR",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=6,
                color="black",
            )
            if T_GRSR_E:
                ax1.axvline(x=T_GRSR_E, color="aqua", linestyle="--", alpha=0.7)
                ax1.text(
                    T_GRSR_E,
                    ax1.get_ylim()[1],
                    "Early Diastolic SR",
                    verticalalignment="bottom",
                    horizontalalignment="center",
                    fontsize=6,
                    color="black",
                )
                box_text = (
                    "Global Radial Strain Rate\n"
                    f"Peak Systolic Strain Rate: {GRSR_S:.2f} 1/s\n"
                    f"Early Diastolic Strain Rate: {GRSR_E:.2f} 1/s"
                )
            else:
                box_text = "Global Radial Strain Rate\n" f"Peak Systolic Strain Rate: {GRSR_S:.2f} 1/s"
            box_props = dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8)
            ax1.text(
                0.98,
                0.95,
                box_text,
                transform=ax1.transAxes,
                fontsize=8,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=box_props,
            )
            fig.savefig(f"{sub_dir}/timeseries/grsr_tagged.png")
            plt.close(fig)

        # * Feature 2: Time to Peak interval
        t_circum_strain_peak = T_circum_strain_peak * temporal_resolution  # unit: ms
        t_radial_strain_peak = T_radial_strain_peak * temporal_resolution

        feature_dict.update(
            {
                "Strain-Tagged: Time to Peak Circumferential Strain [ms]": t_circum_strain_peak,
                "Strain-Tagged: Time to Peak Radial Strain [ms]": t_radial_strain_peak,
            }
        )

        ecg_processor = ECG_Processor(subject, args.retest)

        if not config.useECG or not ecg_processor.check_data_rest():
            logger.warning(f"{subject}: No ECG rest data, time to peak strain index will not be calculated.")
        else:
            RR_interval = ecg_processor.determine_RR_interval()  # should be close to MeanNN in neurokit2
            logger.info(f"{subject}: RR interval is {RR_interval:.2f} ms.")

            t_circum_strain_peak_index = t_circum_strain_peak / RR_interval
            t_radial_strain_peak_index = t_radial_strain_peak / RR_interval

            feature_dict.update(
                {
                    "Strain-Tagged: Time to Peak Circumferential Strain Index": t_circum_strain_peak_index,
                    "Strain-Tagged: Time to Peak Radial Strain Index": t_radial_strain_peak_index,
                }
            )

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "strain_tagged")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df = df[[col for col in df.columns if col != "eid"] + ["eid"]]  # move 'eid' to the last column
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), index=False)
