import os
import shutil
import argparse
from tqdm import tqdm

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config

# several functions to simplify the redundant code
from utils.os_utils import check_existing_file
from utils.slurm_utils import generate_header_cpu, generate_header_gpu

from utils.log_utils import setup_logging

logger = setup_logging("main: segment")

parser = argparse.ArgumentParser()
parser.add_argument("--overwrite", action="store_true", help="Overwrite the existing segmentation files")


def generate_scripts(
    pipeline_dir, data_dir, code_dir, modality, num_subjects_per_file=100, retest_suffix=None, cpu=True, overwrite=False
):
    if retest_suffix is None:
        code_step2_dir = os.path.join(code_dir, "segment_visit1")
    else:
        code_step2_dir = os.path.join(code_dir, "segment_visit2")

    if os.path.exists(code_step2_dir):
        shutil.rmtree(code_step2_dir)
    os.makedirs(code_step2_dir)

    sub_total = os.listdir(data_dir)
    sub_total = sorted(sub_total)
    length_total = len(sub_total)
    logger.info(f"Total number of subjects: {length_total}")
    num_files = length_total // num_subjects_per_file + 1

    with open(os.path.join(code_step2_dir, "batAll.sh"), "w") as file_submit:
        file_submit.write("#!/bin/bash\n")
        for file_i in tqdm(range(1, num_files + 1)):
            file_submit.write(f"sbatch bat{file_i}.pbs\n")
            with open(os.path.join(code_step2_dir, f"bat{file_i}.pbs"), "w") as file_script:
                if cpu:
                    generate_header_cpu("Heart_Segment", pipeline_dir, file_i, file_script, retest_suffix=retest_suffix)
                else:
                    generate_header_gpu("Heart_Segment", pipeline_dir, file_i, file_script, retest_suffix=retest_suffix)

                for sub_i in range(
                    (file_i - 1) * num_subjects_per_file + 1,
                    min(file_i * num_subjects_per_file + 1, length_total + 1),
                ):
                    subject = sub_total[sub_i - 1]  # subject ID
                    sub_dir = os.path.join(data_dir, subject)

                    write_script = False

                    if "aortic_scout" in modality:
                        if not check_existing_file(["seg_aortic_scout.nii.gz"], sub_dir) or overwrite:
                            if check_existing_file(["aortic_scout.nii.gz"], sub_dir):
                                # no original file
                                file_script.write(f"echo '{subject}: Generate segmentation for aortic scout'\n")
                                file_script.write(
                                    "python ./src/segmentation/Aortic_Scout_20207/segment_aortic_scout.py "
                                    f"--data_dir {sub_dir}\n"
                                )
                                write_script = True
                    if "la" in modality:
                        if not check_existing_file(["seg_la_2ch.nii.gz"], sub_dir) or overwrite:
                            if check_existing_file(["la_2ch.nii.gz"], sub_dir):
                                file_script.write(f"echo '{subject}: Generate segmentation for vertical long axis'\n")
                                file_script.write(
                                    "python ./src/segmentation/Long_Axis_20208/segment_la.py "
                                    f"--modality 2ch --data_dir {sub_dir}\n"
                                )
                                write_script = True
                        else:
                            logger.info(f"{subject}: Segmentation for vertical long axis already exists, skip.")

                        # Segmentation that include only atriums
                        if not check_existing_file(["seg_la_4ch.nii.gz"], sub_dir) or overwrite:
                            if check_existing_file(["la_4ch.nii.gz"], sub_dir):
                                file_script.write(
                                    f"echo '{subject}: Generate segmentation for horizontal long axis (2 chambers)'\n"
                                )
                                file_script.write(
                                    "python ./src/segmentation/Long_Axis_20208/segment_la.py "
                                    f"--modality 4ch --data_dir {sub_dir}\n"
                                )
                                write_script = True
                        else:
                            logger.info(f"{subject}: Segmentation for horizontal long axis (2 chambers) already exists, skip.")

                        # Segmentation that include all four ventricles and atriums
                        if not check_existing_file(["seg4_la_4ch.nii.gz"], sub_dir) or overwrite:
                            if check_existing_file(["la_4ch.nii.gz"], sub_dir):
                                file_script.write(
                                    f"echo '{subject}: Generate segmentation for horizontal long axis (4 chambers)'\n"
                                )
                                file_script.write(
                                    "python ./src/segmentation/Long_Axis_20208/segment_la.py "
                                    f"--modality 4ch_4chamber --data_dir {sub_dir}\n"
                                )
                                write_script = True
                        else:
                            logger.info(f"{subject}: Segmentation for horizontal long axis (4 chambers) already exists, skip.")

                    if "sa" in modality:
                        if not check_existing_file(["seg_sa.nii.gz"], sub_dir) or overwrite:
                            if check_existing_file(["sa.nii.gz"], sub_dir):
                                file_script.write(f"echo '{subject}: Generate segmentation for short axis'\n")
                                file_script.write(
                                    f"python ./src/segmentation/Short_Axis_20209/segment_sa.py --data_dir {sub_dir}\n"
                                )
                                write_script = True
                        else:
                            logger.info(f"{subject}: Segmentation for short axis already exists, skip.")

                    if "aortic_dist" in modality:
                        if not check_existing_file(["seg_aortic_dist.nii.gz"], sub_dir) or overwrite:
                            if check_existing_file(["aortic_dist.nii.gz"], sub_dir):
                                file_script.write(f"echo '{subject}: Generate segmentation for aortic distensibility'\n")
                                file_script.write(
                                    "python ./src/segmentation/Aortic_Distensibility_20210/segment_aortic_dist.py "
                                    f"--data_dir {sub_dir}\n"
                                )
                                write_script = True
                        else:
                            logger.info(f"{subject}: Segmentation for aortic distensibility already exists, skip.")

                    # * Tagged MRI incorporates segmentation during processing, its segmentation will not be handled here

                    if "lvot" in modality:
                        if not check_existing_file(["seg_lvot.nii.gz"], sub_dir) or overwrite:
                            if check_existing_file(["lvot.nii.gz"], sub_dir):
                                file_script.write(f"echo '{subject}: Generate segmentation for LVOT'\n")
                                file_script.write(
                                    "python ./src/segmentation/LVOT_20212/preprocess_lvot.py " f"--data_dir {sub_dir}\n"
                                )
                                file_script.write(f"conda activate {config.model_envs[config.model_used['lvot']]}\n")
                                file_script.write("source ./env_variable.sh\n")
                                file_script.write(
                                    "python ./src/segmentation/LVOT_20212/segment_lvot.py "
                                    f"--data_dir {sub_dir} --model {config.model_used['lvot']}\n"
                                )
                                file_script.write("conda deactivate\n")
                                write_script = True
                        else:
                            logger.info(f"{subject}: Segmentation for LVOT already exists, skip.")
                    if "aortic_blood_flow" in modality:
                        if not check_existing_file(["seg_aortic_flow.nii.gz"], sub_dir) or overwrite:
                            if check_existing_file(["aortic_flow.nii.gz"], sub_dir):
                                file_script.write(f"echo '{subject}: Generate segmentation for Phase Contrast MRI'\n")
                                file_script.write(
                                    "python ./src/segmentation/Phase_Contrast_20213/preprocess_phase_contrast.py "
                                    f"--data_dir {sub_dir}\n"
                                )
                                file_script.write(f"conda activate {config.model_envs[config.model_used['aortic_blood_flow']]}\n")
                                file_script.write("source ./env_variable.sh\n")
                                file_script.write(
                                    "python ./src/segmentation/Phase_Contrast_20213/segment_phase_contrast.py "
                                    f"--data_dir {sub_dir} --model {config.model_used['aortic_blood_flow']}\n"
                                )
                                file_script.write("conda deactivate\n")
                                write_script = True
                        else:
                            logger.info(f"{subject}: Segmentation for Phase Contrast MRI already exists, skip.")
                    if "shmolli" in modality:
                        if not check_existing_file(["seg_shmolli_t1map.nii.gz"], sub_dir) or overwrite:
                            if check_existing_file(["shmolli_t1map.nii.gz"], sub_dir):
                                file_script.write(f"echo '{subject}: Generate segmentation for Native T1 Mapping'\n")
                                file_script.write(f"conda activate {config.model_envs[config.model_used['shmolli']]}\n")
                                file_script.write("source ./env_variable.sh\n")
                                file_script.write(
                                    "python ./src/segmentation/Native_T1_20214/segment_native_t1.py "
                                    f"--data_dir {sub_dir} --model {config.model_used['shmolli']}\n"
                                )
                                file_script.write("conda deactivate\n")
                                write_script = True
                        else:
                            logger.info(f"{subject}: Segmentation for Native T1 Mapping already exists, skip.")

                    if write_script:
                        file_script.write("\n")

                file_script.write("echo 'Finished!'\n")


if __name__ == "__main__":
    pipeline_dir = config.pipeline_dir
    data_visit1_dir = config.data_visit1_dir
    data_visit2_dir = config.data_visit2_dir
    code_dir = config.code_dir
    modality = config.modality
    retest_suffix = config.retest_suffix

    os.makedirs(code_dir, exist_ok=True)

    # Clean up existing files in temporary directory
    logger.info("Clean up existing files in temporary directory")
    temp_dir = config.temp_dir
    raw_folder = os.path.join(temp_dir, "nnUNet_raw")
    if os.path.exists(raw_folder):
        shutil.rmtree(raw_folder)

    logger.info("Generate scripts to segment for visit1 data")
    generate_scripts(pipeline_dir, data_visit1_dir, code_dir, modality, cpu=False)
    if retest_suffix is not None:
        logger.info("Generate scripts to segment for visit2 data")
        generate_scripts(pipeline_dir, data_visit2_dir, code_dir, modality, retest_suffix=retest_suffix, cpu=False)
    else:
        logger.info("No retest_suffix is provided, only visit1 data will be segmented")
