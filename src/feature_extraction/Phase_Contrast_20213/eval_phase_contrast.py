import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import find_objects
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
import argparse
from tqdm import tqdm
import sys
from pydicom.filereader import dcmread
import warnings
from csa_header import CsaHeader

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config

from utils.log_utils import setup_logging
from utils.cardiac_utils import evaluate_velocity_flow
from utils.analyze_utils import plot_time_series_dual_axes, plot_time_series_dual_axes_multiple_y, analyze_time_series_root

warnings.filterwarnings("ignore", category=cm.cbook.MatplotlibDeprecationWarning)

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
            or not os.path.exists(img_phase_name)
            or not os.path.exists(img_magnitude_name)
        ):
            logger.error(f"{subject}: At least one modality for phase contrast MRI files is missing")
            continue

        if not os.path.exists(seg_morphology_name):
            logger.error(f"{subject}: Segmentation of phase contrast MRI files is missing")
            continue

        nii_img = nib.load(img_morphology_name)
        img_morephology = nib.load(img_morphology_name).get_fdata()
        seg_morphology = nib.load(seg_morphology_name).get_fdata()
        seg_morphology_nan = np.where(seg_morphology == 0, np.nan, seg_morphology)
        img_phase = nib.load(img_phase_name).get_fdata()
        img_magnitude = nib.load(img_magnitude_name).get_fdata()

        T = nii_img.header["dim"][4]  # number of time frames
        temporal_resolution = nii_img.header["pixdim"][4] * 1000  # unit: ms
        pixdim = nii_img.header["pixdim"][1:4]  # spatial resolution
        square_per_pix = pixdim[0] * pixdim[1] * 1e-2  # unit: cm^2

        aortic_area = []

        logger.info(f"{subject}: Visualizing aortic flow segmentation of CINE, magnitude and phase")
        os.makedirs(f"{sub_dir}/visualization/aorta", exist_ok=True)
        for t in range(T):
            aortic_area.append(np.sum(seg_morphology[:, :, 0, t]) * square_per_pix)
            plt.figure(figsize=(20, 8))
            plt.subplot(1, 5, 1)
            plt.imshow(img_morephology[:, :, 0, t], cmap="gray")
            plt.imshow(seg_morphology_nan[:, :, 0, t], cmap="jet", alpha=0.5)
            plt.title(f"Frame {t}: Morphology (CINE)")
            plt.subplot(1, 5, 2)
            plt.imshow(img_magnitude[:, :, 0, t], cmap="gray")
            plt.title(f"Frame {t}: Magnitude")
            plt.subplot(1, 5, 3)
            plt.title(f"Frame {t}: Magnitude with label")
            plt.imshow(img_magnitude[:, :, 0, t], cmap="gray")
            plt.imshow(seg_morphology_nan[:, :, 0, t], cmap="jet", alpha=0.5)
            plt.subplot(1, 5, 4)
            plt.title(f"Frame {t}: Phase")
            plt.imshow(img_phase[:, :, 0, t], cmap="gray")
            plt.subplot(1, 5, 5)
            plt.title(f"Frame {t}: Phase with label")
            plt.imshow(img_phase[:, :, 0, t], cmap="gray")
            plt.imshow(seg_morphology_nan[:, :, 0, t], cmap="jet", alpha=0.5)
            plt.tight_layout()
            plt.savefig(f"{sub_dir}/visualization/aorta/aortic_flow_segmentation_{(t + 1):02d}.png")
            plt.close()

        feature_dict = {
            "eid": subject,
        }

        # * Feature1: Aortic area (not aortic valve area)
        aortic_area = np.array(aortic_area)
        feature_dict.update(
            {
                "Aortic Flow: Maximum Area [mm^2]": np.max(aortic_area),
                "Aortic Flow: Minimum Area [mm^2]": np.min(aortic_area),
            }
        )

        logger.info(f"{subject}: Determining Aliasing velocity (VENC) from DICOM files")
        dicom_folder = os.path.join(os.path.dirname(os.path.dirname(sub_dir)), "dicom", subject)
        dicom_phase_folder = os.path.join(dicom_folder, "flow_250_tp_AoV_bh_ePAT@c_P")

        # Just choose the first file
        dicom_files = [f for f in os.listdir(dicom_phase_folder) if f.endswith(".dcm")]
        dicom_file = dcmread(os.path.join(dicom_phase_folder, dicom_files[0]))
        raw_csa = dicom_file.get((0x29, 0x1010)).value
        parsed_csa = CsaHeader(raw_csa).read()
        VENC = parsed_csa["FlowVenc"]["value"]  # unit: cm/s
        logger.info(f"{subject}: VENC is set to {VENC}cm/s")

        # * Feature2: Aortic flow velocity, gradient, flow

        logger.info(f"{subject}: Obtaining aortic velocity, gradient and flow information")
        velocity, gradient, flow, flow_positive, flow_negative, velocity_center, velocity_map = evaluate_velocity_flow(
            seg_morphology[:, :, 0, :],
            img_phase[:, :, 0, :],
            VENC,
            square_per_pix,
        )  # velocity unit: cm/s, flow unit: cm^3/s=mL/s

        time_grid_point = np.arange(T)
        time_grid_real = time_grid_point * temporal_resolution  # unit: ms

        # * Determine timepoint of peak systole and end systole using flow curve
        # * Note the information extracted using short axis/long axis will be used as reference

        # Ref Aortic flow is associated with aging and exercise capacity https://doi.org/10.1093/ehjopen/oead079
        # define Systole phase: T_ED ~ T_ES
        # define Late systole phase: T_peak_systole ~ T_ES
        # define Diastole phase: T_ES ~ T_last
        try:
            n_root = 1
            root = analyze_time_series_root(time_grid_point, flow, n_root)  # linear interpolation by default
            while root < 5:
                # This is a heuristic threshold, as the ES time point is usually around 10~11 (320~350ms)
                n_root += 1
                root = analyze_time_series_root(time_grid_point, flow, n_root)
            T_ES = analyze_time_series_root(time_grid_point, flow, n_root, method="round")
            T_ES_real = T_ES * temporal_resolution
            T_peak_systole = np.argmax(flow)
        except ValueError:
            logger.error(f"{subject}: Unable to determine end-systole point from the flow curve")
            continue

        if T_peak_systole >= T_ES:
            logger.error(f"{subject}: Time for peak systole should be ahead of end systole")
            continue

        try:
            ventricle = np.load(f"{sub_dir}/timeseries/ventricle.npz")
            logger.info(f"{subject}: Reference time from ventricle segmentation")
            T_ES_real_ref = ventricle["LV: T_ES [ms]"]
            T_ES_ref = int(T_ES_real / temporal_resolution)
            # a very rough check
            if max(T_ES_real_ref, T_ES_real) > 2 * min(T_ES_real_ref, T_ES_real):
                logger.error(f"{subject}: Time extracted from flow curve is not consistent with one from ventricle segmentation")
                continue
        except FileNotFoundError:
            logger.info(f"{subject}: No reference time from ventricle segmentation")

        logger.info(f"{subject}: End-systole time point is {T_ES} ({T_ES_real:.2f}ms)")
        logger.info(f"{subject}: Peak systole time point is {T_peak_systole} ({T_peak_systole * temporal_resolution:.2f}ms)")

        os.makedirs(f"{sub_dir}/timeseries", exist_ok=True)
        logger.info(f"{subject}: Saving aortic valve time data")
        data_time = {
            "Aortic Flow: T_ES [frame]": T_ES,
            "Aortic Flow: T_peak_systole [frame]": T_peak_systole,
            "Aortic Flow: T_ES [ms]": T_ES_real,
            "Aortic Flow: T_peak_systole [ms]": T_peak_systole * temporal_resolution,
        }
        np.savez(f"{sub_dir}/timeseries/aorta.npz", **data_time)

        # * Visualize all velocity maps in one figure
        # * For better visualization, we first determine a largest bounding box for the aortic valve
        logger.info(f"{subject}: Visualizing aortic flow velocity maps")
        fig = plt.figure(figsize=(12, 10))

        bounding_boxes = []

        for t in range(T):
            objects = find_objects(seg_morphology[:, :, 0, t].astype(np.uint8))
            col_start = objects[0][1].start
            col_end = objects[0][1].stop
            row_start = objects[0][0].start
            row_end = objects[0][0].stop
            bounding_boxes.append((row_start, row_end, col_start, col_end))

        bounding_boxes = np.array(bounding_boxes)
        bounding_box = (
            np.min(bounding_boxes[:, 0]),
            np.max(bounding_boxes[:, 1]),
            np.min(bounding_boxes[:, 2]),
            np.max(bounding_boxes[:, 3]),
        )

        width = bounding_box[3] - bounding_box[2] + 1
        height = bounding_box[1] - bounding_box[0] + 1

        velocity_map_roi = np.zeros((height, width, T))

        for t in range(T):
            velocity_map_roi[:, :, t] = (velocity_map[:, :, t] * seg_morphology[:, :, 0, t])[
                bounding_box[0] : bounding_box[1] + 1, bounding_box[2] : bounding_box[3] + 1
            ]

        x = np.linspace(0, 10, width)
        y = np.linspace(0, 10, height)
        X, Y = np.meshgrid(x, y)
        rows, cols = 6, 5
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(rows, cols, figure=fig, wspace=0, hspace=0)

        for t in range(T):
            row, col = divmod(t, cols)
            ax = fig.add_subplot(gs[row, col], projection="3d")
            ax.plot_surface(X, Y, velocity_map_roi[:, :, t], cmap="viridis", edgecolor="none")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.axis("off")
            ax.set_title(f"Frame {t + 1}: Peak Velocity: {velocity[t]:.2f} cm/s", fontsize=8)

        # add a supertitle
        plt.suptitle(f"Aortic Valve Velocity (VENC is set to {VENC} cm/s)", fontsize=16, y=0.98)

        norm = cm.colors.Normalize(vmin=np.min(velocity_map_roi), vmax=np.max(velocity_map_roi))
        sm = cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm)
        cbar.set_label("Velocity (cm/s)", fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{sub_dir}/visualization/aorta/aortic_flow_velocity.png")

        # define Mean velocity is the average during systole
        peak_velocity = np.max(velocity)
        feature_dict.update(
            {
                "Aortic Flow: Peak Velocity [cm/s]": peak_velocity,
                "Aortic Flow: Mean Gradient [mmHg]": np.mean(gradient[:T_ES]),
            }
        )

        os.makedirs(f"{sub_dir}/timeseries", exist_ok=True)
        fig, ax1, ax2 = plot_time_series_dual_axes(
            time_grid_point,
            velocity,
            "Time [frame]",
            "Time [ms]",
            "Peak Velocity [cm/s]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Aortic Velocity",
        )
        box_text = (
            "Velocity and Gradient\n"
            f"Peak Velocity: {peak_velocity:.2f} cm/s\n"
            f"Mean Gradient: {np.mean(gradient[:T_ES]):.2f} mmHg"
        )
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
        ax1.axvline(x=T_ES, color="purple", linestyle="--", alpha=0.7)
        ax1.text(
            T_ES,
            ax1.get_ylim()[1],
            "End Systole",
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=6,
            color="black",
        )
        ax1.axvline(x=T_peak_systole, color="pink", linestyle="--", alpha=0.7)
        ax1.text(
            T_peak_systole,
            ax1.get_ylim()[1],
            "Peak Systole",
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=6,
            color="black",
        )
        arrow_systole = FancyArrowPatch(
            (0, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
            (T_ES, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
            arrowstyle="<->",
            linestyle="--",
            color="black",
            alpha=0.7,
            mutation_scale=15,
        )
        ax1.add_patch(arrow_systole)
        ax1.text(
            T_ES / 2,
            ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
            "Systole",
            fontsize=6,
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )
        arrow_late_systole = FancyArrowPatch(
            (T_peak_systole, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 6),
            (T_ES, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 6),
            arrowstyle="<->",
            linestyle="--",
            color="black",
            alpha=0.7,
            mutation_scale=15,
        )
        ax1.add_patch(arrow_late_systole)
        ax1.text(
            (T_peak_systole + T_ES) / 2,
            ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 8,
            "Late Systole",
            fontsize=6,
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )
        arrow_diastole = FancyArrowPatch(
            (T_ES, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
            (ax1.get_xlim()[1], ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
            arrowstyle="<->",
            linestyle="--",
            color="black",
            alpha=0.7,
            mutation_scale=15,
        )
        ax1.add_patch(arrow_diastole)
        ax1.text(
            (T_ES + ax1.get_xlim()[1]) / 2,
            ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
            "Diastole",
            fontsize=6,
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )
        fig.savefig(f"{sub_dir}/timeseries/aortic_velocity.png")
        plt.close(fig)

        # * Determining forward flow and backward from the flow curve, which can be used to calculate regurgitation fraction
        logger.info(f"{subject}: Determining flow features and level of aortic regurgitation")

        forward_flow = np.trapz(np.where(flow > 0, flow, 0), dx=temporal_resolution / 1000)  # unit: mL
        backward_flow = abs(np.trapz(np.where(flow < 0, flow, 0)[T_ES + 1 :], dx=temporal_resolution / 1000))

        regurgitant_fraction = backward_flow / forward_flow * 100
        if regurgitant_fraction > 100:
            logger.error(f"{subject}: Regurgitant fraction should be less than 100%")
            continue

        feature_dict.update(
            {
                "Aortic Flow: Forward Flow [mL]": forward_flow,
                "Aortic Flow: Backward Flow [mL]": backward_flow,
                "Aortic Flow: Regurgitant Fraction [%]": regurgitant_fraction,
            }
        )

        # * Levels of aortic stenosis and regurgitation can be determined from the following reference:
        # Ref 2014 AHA/ACC Guideline for the Management of Patients With Valvular Heart Disease. https://doi.org/10.1016/j.jacc.2014.02.536

        fig, ax1, ax2 = plot_time_series_dual_axes_multiple_y(
            time_grid_point,
            [flow, flow_positive, flow_negative],
            ["Net Flow", "Forward Flow", "Backward Flow"],
            "Time [frame]",
            "Time [ms]",
            "Flow [mL/s]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Aortic Flow",
            colors=["red", "yellow", "blue"],
        )
        box_text = "Aortic Flow\n" f"Forward Flow: {forward_flow:.2f} mL\n" f"Backward Flow: {backward_flow:.2f} mL"
        box_props = dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8)
        ax1.text(
            0.98,
            0.80,
            box_text,
            transform=ax1.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=box_props,
        )
        ax1.axvline(x=T_ES, color="purple", linestyle="--", alpha=0.7)
        ax1.text(
            T_ES,
            ax1.get_ylim()[1],
            "End Systole",
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=6,
            color="black",
        )
        ax1.axvline(x=T_peak_systole, color="pink", linestyle="--", alpha=0.7)
        ax1.text(
            T_peak_systole,
            ax1.get_ylim()[1],
            "Peak Systole",
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=6,
            color="black",
        )
        arrow_systole = FancyArrowPatch(
            (0, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
            (T_ES, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
            arrowstyle="<->",
            linestyle="--",
            color="black",
            alpha=0.7,
            mutation_scale=15,
        )
        ax1.add_patch(arrow_systole)
        ax1.text(
            T_ES / 2,
            ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
            "Systole",
            fontsize=6,
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )
        arrow_late_systole = FancyArrowPatch(
            (T_peak_systole, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 6),
            (T_ES, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 6),
            arrowstyle="<->",
            linestyle="--",
            color="black",
            alpha=0.7,
            mutation_scale=15,
        )
        ax1.add_patch(arrow_late_systole)
        ax1.text(
            (T_peak_systole + T_ES) / 2,
            ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 8,
            "Late Systole",
            fontsize=6,
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )
        arrow_diastole = FancyArrowPatch(
            (T_ES, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
            (ax1.get_xlim()[1], ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
            arrowstyle="<->",
            linestyle="--",
            color="black",
            alpha=0.7,
            mutation_scale=15,
        )
        ax1.add_patch(arrow_diastole)
        ax1.text(
            (T_ES + ax1.get_xlim()[1]) / 2,
            ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
            "Diastole",
            fontsize=6,
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )
        fig.savefig(f"{sub_dir}/timeseries/aortic_flow.png")
        plt.close(fig)

        # * Feature3: Velocity-Time Integral (VTI) and Aortic valve area (AVA)

        # Ref A Simplified Continuity Equation Approach to the Quantification of Stenotic Bicuspid Aortic Valves https://doi.org/10.1080/10976640701693717
        # Ref Practical Value of Cardiac Magnetic Resonance Imaging for Clinical Quantification of Aortic Valve Stenosis https://doi.org/10.1161/01.CIR.0000095268.47282.A1

        VTI = np.trapz(velocity[: T_ES + 1], dx=temporal_resolution / 1000)  # unit: cm. Add 1 to complete the integral
        AVA = forward_flow / VTI

        feature_dict.update(
            {
                "Aortic Flow: Aortic Valve Area [cm^2]": AVA,
            }
        )

        if VTI > 100:
            logger.warning(f"{subject}: Extremely high VTI detected, skipped")
        else:
            feature_dict.update(
                {
                    "Aortic Flow: Velocity Time Integral [cm]": VTI,
                }
            )

        # * Feature4: Flow displacement
        # This is a quantitative parameter for measuring eccentric aortic systolic flow
        # Ref Comparison of Four-Dimensional Flow Parameters for Quantification of Flow Eccentricity in the Ascending Aorta https://pubmed.ncbi.nlm.nih.gov/21928387/
        flow_displacement = []
        lumen_center = []
        for t in range(T):
            # We will keep all distance in image space rather than the real distances
            lumen_coords_t = np.argwhere(seg_morphology[:, :, 0, t] > 0)
            lumen_center_t = np.mean(lumen_coords_t, axis=0)
            # swap the x and y axis
            lumen_center_t = np.array([lumen_center_t[1], lumen_center_t[0]])
            lumen_center.append(lumen_center_t)
            lumen_diameter_t = np.max(np.linalg.norm(lumen_coords_t - lumen_center_t, axis=1)) * 2
            velocity_center_t = velocity_center[t]
            # * The average vessel radius is used to normalize the flow displacement
            flow_dispacement_t = np.linalg.norm(velocity_center_t - lumen_center_t) / (lumen_diameter_t / 2) * 100
            flow_displacement.append(flow_dispacement_t)

        lumen_center = np.array(lumen_center)

        logger.info(f"{subject}: Visualizing aortic flow displacement")
        for t in range(T):
            plt.figure(figsize=(12, 8))
            plt.imshow(velocity_map[:, :, t], cmap="jet")
            plt.contour(seg_morphology[:, :, 0, t], levels=[0.5], colors="black", linewidths=0.5)
            plt.scatter(
                velocity_center[t][0],
                velocity_center[t][1],
                c="red",
                s=3,
                label="Center of velocity",
            )
            plt.scatter(
                lumen_center[t][0],
                lumen_center[t][1],
                c="blue",
                s=3,
                label="Center of lumen",
            )
            plt.plot(
                [velocity_center[t][0], lumen_center[t][0]],
                [velocity_center[t][1], lumen_center[t][1]],
                color="black",
                linewidth=1,
            )
            plt.legend(loc="lower right")
            plt.title(f"Frame {t}: Flow Displacement is {flow_displacement[t]:.2f}%")
            norm = cm.colors.Normalize(vmin=np.min(velocity_map), vmax=np.max(velocity_map))
            sm = cm.ScalarMappable(cmap="viridis", norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label("Velocity (cm/s)")
            plt.tight_layout()
            plt.savefig(f"{sub_dir}/visualization/aorta/aortic_flow_displacement_{(t + 1):02d}.png")
            plt.close()

        flow_displacement = np.array(flow_displacement)

        # Ref Automated Quantification of Simple and Complex Aortic Flow Using 2D Phase Contrast MRI https://doi.org/10.3390/medicina60101618
        FDs = np.mean(flow_displacement[:T_ES])  # flow displacement systolic average
        FDls = np.mean(flow_displacement[T_peak_systole:T_ES])  # flow displacement late systolic average
        FDd = np.mean(flow_displacement[T_ES:])  # flow displacement diastolic average

        feature_dict.update(
            {
                "Aortic Flow: Flow Displacement Systolic Average [%]": FDs,
                "Aortic Flow: Flow Displacement Late Systolic Average [%]": FDls,
                "Aortic Flow: Flow Displacement Diastolic Average [%]": FDd,
            }
        )
        fig, ax1, ax2 = plot_time_series_dual_axes(
            time_grid_point,
            flow_displacement,
            "Time [frame]",
            "Time [ms]",
            "Flow Displacement [%]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Aortic Flow Displacement",
        )
        box_text = (
            "Aortic Flow Displacement\n"
            f"Systolic Average: {FDs:.2f} %\n"
            f"Late Systolic Average: {FDls:.2f}%\n"
            f"Diastolic Average: {FDd:.2f} %"
        )
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
        ax1.axvline(x=T_ES, color="purple", linestyle="--", alpha=0.7)
        ax1.text(
            T_ES,
            ax1.get_ylim()[1],
            "End Systole",
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=6,
            color="black",
        )
        ax1.axvline(x=T_peak_systole, color="pink", linestyle="--", alpha=0.7)
        ax1.text(
            T_peak_systole,
            ax1.get_ylim()[1],
            "Peak Systole",
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=6,
            color="black",
        )
        arrow_systole = FancyArrowPatch(
            (0, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
            (T_ES, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
            arrowstyle="<->",
            linestyle="--",
            color="black",
            alpha=0.7,
            mutation_scale=15,
        )
        ax1.add_patch(arrow_systole)
        ax1.text(
            T_ES / 2,
            ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
            "Systole",
            fontsize=6,
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )
        arrow_late_systole = FancyArrowPatch(
            (T_peak_systole, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 6),
            (T_ES, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 6),
            arrowstyle="<->",
            linestyle="--",
            color="black",
            alpha=0.7,
            mutation_scale=15,
        )
        ax1.add_patch(arrow_late_systole)
        ax1.text(
            (T_peak_systole + T_ES) / 2,
            ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 8,
            "Late Systole",
            fontsize=6,
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )
        arrow_diastole = FancyArrowPatch(
            (T_ES, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
            (ax1.get_xlim()[1], ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 10),
            arrowstyle="<->",
            linestyle="--",
            color="black",
            alpha=0.7,
            mutation_scale=15,
        )
        ax1.add_patch(arrow_diastole)
        ax1.text(
            (T_ES + ax1.get_xlim()[1]) / 2,
            ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) / 12,
            "Diastole",
            fontsize=6,
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )
        fig.savefig(f"{sub_dir}/timeseries/aortic_flow_displacement.png")
        plt.close(fig)

        # * Feature5: Rotation angle (RA)

        logger.info(f"{subject}: Determining aortic flow rotation angle")
        # Ref Aortic flow is associated with aging and exercise capacity https://doi.org/10.1093/ehjopen/oead079
        # Ref Aortic flow is abnormal in HFpEF https://pmc.ncbi.nlm.nih.gov/articles/PMC10940846/pdf/wellcomeopenres-8-23435.pdf
        rotation_angle = []

        for t in range(T):
            # * Since rotation angle is sensitive to errors, FD=12% is chosen as the threshold when setting RA=0
            if flow_displacement[t] < 12:
                angle = 0
            else:
                angle = (
                    np.arctan2(velocity_center[t][0] - lumen_center[t][0], velocity_center[t][1] - lumen_center[t][1])
                    / np.pi
                    * 180
                )
                angle = (360 - angle) % 360  # make sure mapping to (0, 360°)
            rotation_angle.append(angle)

        fig, ax1, ax2 = plot_time_series_dual_axes(
            time_grid_point,
            rotation_angle,
            "Time [frame]",
            "Time [ms]",
            "FD Rotation Angle [°]",
            lambda x: x * temporal_resolution,
            lambda x: x / temporal_resolution,
            title=f"Subject {subject}: Aortic Flow Displacement Rotation Angle",
        )
        ax1.axvline(x=T_ES, color="purple", linestyle="--", alpha=0.7)
        ax1.text(
            T_ES,
            ax1.get_ylim()[1],
            "End Systole",
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=6,
            color="black",
        )
        ax1.axvline(x=T_peak_systole, color="pink", linestyle="--", alpha=0.7)
        ax1.text(
            T_peak_systole,
            ax1.get_ylim()[1],
            "Peak Systole",
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=6,
            color="black",
        )
        fig.savefig(f"{sub_dir}/timeseries/aortic_flow_rotation_angle.png")
        plt.close(fig)

        # todo: ΔRA, RA at ES - RA when flow angle stablized after peak systole can be determined from the curve

        # * Feature6: Systolic flow reversal ratio (SFR)

        # This is very similar to regurgitant fraction, but we only consider the systolic phase
        systolic_forward_flow = np.trapz(flow_positive[:T_ES], dx=temporal_resolution / 1000)
        systolic_backward_flow = abs(np.trapz(flow_negative[:T_ES], dx=temporal_resolution / 1000))

        feature_dict.update(
            {
                "Aortic Flow: Systolic Forward Flow [mL]": systolic_forward_flow,
                "Aortic Flow: Systolic Reverse Flow [mL]": systolic_backward_flow,  # a bit different name
            }
        )
        SFR = systolic_backward_flow / systolic_forward_flow * 100
        if SFR > 80:
            logger.warning(f"{subject}: Systolic flow reversal ratio is extremely high, skipped.")
        else:
            feature_dict.update({"Aortic Flow: Systolic Flow Reversal Ratio [%]": SFR})

        df_row = pd.DataFrame([feature_dict])
        df = pd.concat([df, df_row], ignore_index=True)

    target_dir = config.features_visit2_dir if args.retest else config.features_visit1_dir
    target_dir = os.path.join(target_dir, "aortic_flow")
    os.makedirs(target_dir, exist_ok=True)
    df.sort_index(axis=1, inplace=True)  # sort the columns according to alphabet orders
    df.to_csv(os.path.join(target_dir, f"{args.file_name}.csv"), index=False)
