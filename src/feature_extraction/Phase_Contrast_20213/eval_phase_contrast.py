import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import argparse
from tqdm import tqdm
import sys
from pydicom.filereader import dcmread
from csa_header import CsaHeader

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config

from utils.log_utils import setup_logging
from utils.cardiac_utils import evaluate_velocity_flow
from utils.analyze_utils import plot_time_series_double_x

logger = setup_logging("eval_phase_contrast")

parser = argparse.ArgumentParser()
parser.add_argument("--retest", action="store_true")
parser.add_argument("--data_list", help="List of subjects to be processed", nargs="*", type=int, required=True)
parser.add_argument("--file_name", help="Name of the csv file to save the features")


if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = config.data_visit2_dir if args.retest else config.data_visit1_dir

    df = pd.DataFrame()

    for subject in tqdm(args.data_list):
        subject = str(subject)
        sub_dir = os.path.join(data_dir, subject)

        img_morphology_name = f"{sub_dir}/aortic_flow.nii.gz"  # define CINE mode
        seg_morphology_name = f"{sub_dir}/seg_aortic_flow.nii.gz"
        img_phase_name = f"{sub_dir}/aortic_flow_pha.nii.gz"
        img_magnitude_name = f"{sub_dir}/aortic_flow_mag.nii.gz"

        if (
            not os.path.exists(img_morphology_name)
            or not os.path.exists(seg_morphology_name)
            or not os.path.exists(img_phase_name)
            or not os.path.exists(img_magnitude_name)
        ):
            logger.error(f"{subject}: At least one modality for phase contrast MRI files is missing")
            continue

        nim_img = nib.load(img_morphology_name)
        img_morephology = nib.load(img_morphology_name).get_fdata()
        seg_morphology = nib.load(seg_morphology_name).get_fdata()
        seg_morphology_nan = np.where(seg_morphology == 0, np.nan, seg_morphology)
        img_phase = nib.load(img_phase_name).get_fdata()
        img_magnitude = nib.load(img_magnitude_name).get_fdata()

        T = nim_img.header["dim"][4]  # number of time frames
        temporal_resolution = nim_img.header["pixdim"][4] * 1000  # unit: ms
        pixdim = nim_img.header["pixdim"][1:4]  # spatial resolution
        square_per_pix = pixdim[0] * pixdim[1] * 1e-2  # unit: cm^2

        logger.info(f"{subject}: Generating a comparison of all modalities in phase-contrast MRI")
        os.makedirs(f"{sub_dir}/visualization/aorta", exist_ok=True)

        for t in range(T):
            plt.figure(figsize=(18, 5))
            plt.subplot(1, 5, 1)
            plt.imshow(img_morephology[:, :, 0, 0], cmap="gray")
            plt.imshow(seg_morphology_nan[:, :, 0, 0], cmap="jet", alpha=0.5)
            plt.title("Morphology (CINE)")
            plt.subplot(1, 5, 2)
            plt.imshow(img_magnitude[:, :, 0, 0], cmap="gray")
            plt.title("Magnitude")
            plt.subplot(1, 5, 3)
            plt.imshow(img_magnitude[:, :, 0, 0], cmap="gray")
            plt.imshow(seg_morphology_nan[:, :, 0, 0], cmap="jet", alpha=0.5)
            plt.title("Magnitude with label")
            plt.subplot(1, 5, 4)
            plt.imshow(img_phase[:, :, 0, 0], cmap="gray")
            plt.title("Phase")
            plt.subplot(1, 5, 5)
            plt.imshow(img_phase[:, :, 0, 0], cmap="gray")
            plt.imshow(seg_morphology_nan[:, :, 0, 0], cmap="jet", alpha=0.5)
            plt.title("Phase with label")
            plt.suptitle(f"Aortic Flow: Time {t}", fontsize=16)
            plt.savefig(f"{sub_dir}/visualization/aorta/aortic_flow_segmentation_{t:02d}.png")
            plt.close()

        feature_dict = {
            "eid": subject,
        }

        logger.info(f"{subject}: Determining Aliasing velocity (VENC) from DICOM files")
        dicom_folder = os.path.join(os.path.dirname(os.path.dirname(sub_dir)), "dicom", subject)
        dicom_phase_folder = os.path.join(dicom_folder, "flow_250_tp_AoV_bh_ePAT@c_P")

        # Just choose the first file
        dicom_files = [f for f in os.listdir(dicom_phase_folder) if f.endswith(".dcm")]
        dicom_file = dcmread(os.path.join(dicom_phase_folder, dicom_files[0]))
        raw_csa = dicom_file.get((0x29, 0x1010)).value
        parsed_csa = CsaHeader(raw_csa).read()
        VENC = parsed_csa["FlowVenc"]["value"]
        logger.info(f"{subject}: VENC is set to {VENC}cm/s")

        # * Feature1: Aortic flow velocity and flow

        logger.info(f"{subject}: Obtaining aortic velocities and flows")
        velocity, flow, flow_center, velocity_map = evaluate_velocity_flow(
            seg_morphology[:, :, 0, :],
            img_phase[:, :, 0, :],
            VENC,
            square_per_pix,
        )  # velocity unit: cm/s, flow unit: cm^3/s=mL/s

        time_grid_point = np.arange(T)
        time_grid_real = time_grid_point * temporal_resolution  # unit: ms

        for t in range(T):
            plt.imshow(velocity_map[:, :, t], cmap="jet")
            plt.colorbar()
            plt.title(f"Subject {subject}: Aortic Velocity Map at Time {t}")
            plt.contour(seg_morphology[:, :, 0, t], levels=[0.5], colors="red", linewidths=0.5)
            plt.savefig(f"{sub_dir}/visualization/aorta/aortic_velocity_map_{t:02d}.png")
            plt.close()

        # * Determing stenosis using peak velocity

        logger.info(f"{subject}: Determining velocity features, gradient, and level of aortic stenosis ")

        peak_velocity = np.max(velocity)  # cm/s
        print(peak_velocity)
        # * Gradient is determined by using the Bernoulli formula, first convert speed to m/s
        peak_gradient = 4 * (peak_velocity / 100) ** 2  # unit: mmHg

        ventricle = np.load(f"{sub_dir}/timeseries/ventricle.npz")
        T_ES_real = ventricle["LV: T_ES [ms]"]
        T_ES = int(T_ES_real / temporal_resolution)
        T_systole = np.argmax(velocity)

        if T_systole >= T_ES:
            logger.error(f"{subject}: Time for peak systole should be ahead of end systole")
            continue

        # define Mean velocity is the average of the numbers from ED to ES
        # Ref Biederman, Robert W., et al. The cardiovascular MRI tutorial: lectures and learning.
        mean_velocity = np.mean(velocity[: T_ES + 1])
        mean_gradient = 4 * (mean_velocity / 100) ** 2

        feature_dict.update(
            {
                "Aortic Flow: Vlocity encoding factor (VENC) [cm/s]": VENC,
                "Aortic Flow: Peak Velocity [cm/s]": peak_velocity,
                "Aortic Flow: Peak Gradient [mmHg]": peak_gradient,
                "Aortic Flow: Mean Velocity [cm/s]": mean_velocity,
                "Aortic Flow: Mean Gradient [mmHg]": mean_gradient,
            }
        )

        os.makedirs(f"{sub_dir}/timeseries", exist_ok=True)
        fig, ax1, ax2 = plot_time_series_double_x(
            time_grid_point,
            time_grid_real,
            velocity,
            "Time [frame]",
            "Time [ms]",
            "Velocity [cm/s]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Aortic Velocity",
        )
        fig.savefig(f"{sub_dir}/timeseries/aortic_velocity.png")
        plt.close(fig)

        # Ref Nishimura et al. 2014 AHA/ACC guideline for the management of patients with valvular heart disease
        # Ref Nishimura et al. 2017 AHA/ACC Focused Update of the 2014 AHA/ACC Guideline for the Management of Patients
        if peak_velocity / 100 < 3:
            stenosis = "None or mild"
        elif peak_velocity / 100 < 4:
            stenosis = "Moderate"
        elif peak_velocity / 100 < 5:
            stenosis = "Severe"
        else:
            stenosis = "Very severe"
        feature_dict.update({"Aortic Flow: Level of Stenosis": stenosis})

        # * Determining regurgitation
        logger.info(f"{subject}: Determining flow features and level of aortic regurgitation")

        flow_positive = np.where(flow > 0, flow, 0)
        flow_negative = np.where(flow < 0, flow, 0)

        forward_flow = np.trapz(flow_positive, dx=temporal_resolution)  # unit: mL
        backward_flow = np.trapz(flow_negative, dx=temporal_resolution)

        regurgitant_fraction = abs(backward_flow) / forward_flow * 100

        feature_dict.update(
            {
                "Aortic Flow: Forward Flow [mL]": forward_flow,
                "Aortic Flow: Backward Flow [mL]": abs(backward_flow),
                "Aortic Flow: Regurgitant Fraction [%]": regurgitant_fraction,
            }
        )

        fig, ax1, ax2 = plot_time_series_double_x(
            time_grid_point,
            time_grid_real,
            flow,
            "Time [frame]",
            "Time [ms]",
            "Flow [mL/s]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Aortic Flow",
        )
        fig.savefig(f"{sub_dir}/timeseries/aortic_flow.png")
        fig.text(
            0.15,
            0.8,
            f"Forward Flow: {forward_flow:.2f} mL\nBackward Flow: {backward_flow:.2f} mL",
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.5),
        )
        plt.close(fig)

        if regurgitant_fraction < 30:
            regurgitation = "None or mild"
        elif regurgitant_fraction < 50:
            regurgitation = "Moderate"
        else:
            regurgitation = "Severe"

        feature_dict.update({"Aortic Flow: Level of Regurgitation": regurgitation})

        # * Feature2: Flow displacement average
        # define This is a quantitative parameter for measuring eccentric aortic systolic flow
        # Ref Comparison of Four-Dimensional Flow Parameters for Quantification of Flow Eccentricity in the Ascending Aorta
        flow_displacement = []
        for t in range(T):
            # We will keep all distance in image space rather than the real distances
            lumen_coords_t = np.argwhere(seg_morphology[:, :, 0, t] > 0)
            lumen_center_t = np.mean(lumen_coords_t, axis=0)
            # swap the x and y axis
            lumen_center_t = np.array([lumen_center_t[1], lumen_center_t[0]])
            lumen_diameter_t = np.max(np.linalg.norm(lumen_coords_t - lumen_center_t, axis=1)) * 2
            flow_center_t = flow_center[t]
            flow_dispacement_t = np.linalg.norm(flow_center_t - lumen_center_t) / lumen_diameter_t * 100
            flow_displacement.append(flow_dispacement_t)

        # Ref Automated Quantification of Simple and Complex Aortic Flow Using 2D Phase Contrast MRI
        flow_displacement = np.array(flow_displacement)
        FDs = np.mean(flow_displacement[: T_ES + 1])  # flow displacement systolic average
        FDls = np.mean(flow_displacement[T_systole : T_ES + 1])  # flow displacement late systolic average

        fig, ax1, ax2 = plot_time_series_double_x(
            time_grid_point,
            time_grid_real,
            flow_displacement,
            "Time [frame]",
            "Time [ms]",
            "Flow Displacement [%]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Aortic Flow Displacement",
        )
        fig.savefig(f"{sub_dir}/timeseries/aortic_flow_displacement.png")
        plt.close(fig)

        # todo: Too small, fix it and generate velocity map
        feature_dict.update(
            {
                "Aortic Flow: Flow Displacement Systolic Average [%]": FDs,
                "Aortic Flow: Flow Displacement Late Systolic Average [%]": FDls,
            }
        )

        # todo: Delta RA (rotation angle)
        # define: Late systole is defined as the time from the peak systolic velocity to the end of systole

        # todo: Aortic valve size & area

        # todo: Number of valve cusp

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "aortic_flow")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), index=False)
