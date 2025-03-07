"""
This script is used to prepare the CMR data for the pipeline using zip files
"""

import os
import shutil
import argparse
from tqdm import tqdm

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config

from utils.slurm_utils import generate_header_cpu, generate_header_gpu
from utils.log_utils import setup_logging

logger = setup_logging("main: prepare_data_cmr")

parser = argparse.ArgumentParser()
parser.add_argument("--overwrite", action="store_true", help="Overwrite the existing files")
parser.add_argument("--keepdicom", action="store_true", help="Keep the DICOM files after converting to Nifti")


def generate_scripts(
    pipeline_dir,
    data_raw_dir,
    code_dir,
    out_dir,
    modality,
    num_subjects_per_file=100,
    retest_suffix=None,
    cpu=True,
    overwrite=False,
    keepdicom=False,
):
    if retest_suffix is None:
        code_step1_dir = os.path.join(code_dir, "prepare_data_visit1")
        out_step1_dir = os.path.join(out_dir, "visit1/")

        scout = os.path.join(data_raw_dir, "20207/")
        long_axis = os.path.join(data_raw_dir, "20208/")
        short_axis = os.path.join(data_raw_dir, "20209/")
        aortic = os.path.join(data_raw_dir, "20210/")
        tagging = os.path.join(data_raw_dir, "20211/")
        LVOT = os.path.join(data_raw_dir, "20212/")
        blood_flow = os.path.join(data_raw_dir, "20213/")
        shmolli = os.path.join(data_raw_dir, "20214/")
    else:
        code_step1_dir = os.path.join(code_dir, "prepare_data_visit2")
        out_step1_dir = os.path.join(out_dir, "visit2/")

        scout = os.path.join(data_raw_dir, f"20207_{retest_suffix}/")
        long_axis = os.path.join(data_raw_dir, f"20208_{retest_suffix}/")
        short_axis = os.path.join(data_raw_dir, f"20209_{retest_suffix}/")
        aortic = os.path.join(data_raw_dir, f"20210_{retest_suffix}/")
        tagging = os.path.join(data_raw_dir, f"20211_{retest_suffix}/")
        LVOT = os.path.join(data_raw_dir, f"20212_{retest_suffix}/")
        blood_flow = os.path.join(data_raw_dir, f"20213_{retest_suffix}/")
        shmolli = os.path.join(data_raw_dir, f"20214_{retest_suffix}/")

    if os.path.exists(code_step1_dir):
        shutil.rmtree(code_step1_dir)
    os.makedirs(code_step1_dir)

    sub_total = []
    if "aortic_scout" in modality:
        sub_aortic_scout = os.listdir(scout)
        sub_aortic_scout = [x.split("_")[0] for x in sub_aortic_scout]
        sub_total = sub_total + sub_aortic_scout
    if "la" in modality:
        sub_la = os.listdir(long_axis)
        sub_la = [x.split("_")[0] for x in sub_la]
        sub_total = sub_total + sub_la
    if "sa" in modality:
        sub_sa = os.listdir(short_axis)
        sub_sa = [x.split("_")[0] for x in sub_sa]
        sub_total = sub_total + sub_sa
    if "aortic_dist" in modality:
        sub_aortic_dist = os.listdir(aortic)
        sub_aortic_dist = [x.split("_")[0] for x in sub_aortic_dist]
        sub_total = sub_total + sub_aortic_dist
    if "tag" in modality:
        sub_tag = os.listdir(tagging)
        sub_tag = [x.split("_")[0] for x in sub_tag]
        sub_total = sub_total + sub_tag
    if "lvot" in modality:
        sub_lvot = os.listdir(LVOT)
        sub_lvot = [x.split("_")[0] for x in sub_lvot]
        sub_total = sub_total + sub_lvot
    if "aortic_blood_flow" in modality:
        sub_aortic_flow = os.listdir(blood_flow)
        sub_aortic_flow = [x.split("_")[0] for x in sub_aortic_flow]
        sub_total = sub_total + sub_aortic_flow
    if "shmolli" in modality:
        sub_shmolli = os.listdir(shmolli)
        sub_shmolli = [x.split("_")[0] for x in sub_shmolli]
        sub_total = sub_total + sub_shmolli

    sub_total = list(set(sub_total))
    sub_total = sorted(sub_total)
    length_total = len(sub_total)
    logger.info(f"Total number of zip files: {length_total}")
    num_files = length_total // num_subjects_per_file + 1

    with open(os.path.join(code_step1_dir, "batAll.sh"), "w") as file_submit:
        file_submit.write("#!/bin/bash\n")
        for file_i in tqdm(range(1, num_files + 1)):
            file_submit.write(f"sbatch bat{file_i}.pbs\n")
            with open(os.path.join(code_step1_dir, f"bat{file_i}.pbs"), "w") as file_script:
                if cpu:
                    generate_header_cpu("Heart_Prepare", pipeline_dir, file_i, file_script, retest_suffix=retest_suffix)
                else:
                    generate_header_gpu("Heart_Prepare", pipeline_dir, file_i, file_script, retest_suffix=retest_suffix)

                for sub_i in range(
                    (file_i - 1) * num_subjects_per_file + 1,
                    min(file_i * num_subjects_per_file + 1, length_total + 1),
                ):
                    sub_id = sub_total[sub_i - 1]
                    options_str = ""
                    if "aortic_scout" in modality:
                        options_str += " --aortic_scout=" + scout
                    if "la" in modality:
                        options_str += " --long_axis=" + long_axis
                    if "sa" in modality:
                        options_str += " --short_axis=" + short_axis
                    if "aortic_dist" in modality:
                        options_str += " --aortic_dist=" + aortic
                    if "tag" in modality:
                        options_str += " --tag=" + tagging
                    if "lvot" in modality:
                        options_str += " --lvot=" + LVOT
                    if "aortic_blood_flow" in modality:
                        options_str += " --aortic_blood_flow=" + blood_flow
                    if "shmolli" in modality:
                        options_str += " --shmolli=" + shmolli
                    if overwrite:
                        options_str += " --overwrite"
                    if keepdicom:
                        options_str += " --keepdicom"
                    file_script.write(
                        f"python ./script/prepare_data.py --out_dir={out_step1_dir} --sub_id={sub_id} {options_str}\n"
                    )

                file_script.write("echo 'Finished!'\n")


if __name__ == "__main__":
    pipeline_dir = config.pipeline_dir
    data_raw_dir = config.data_raw_dir
    out_dir = config.data_dir
    code_dir = config.code_dir
    modality = config.modality
    retest_suffix = config.retest_suffix

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(code_dir, exist_ok=True)

    args = parser.parse_args()
    overwrite = args.overwrite
    keepdicom = args.keepdicom

    if overwrite:
        logger.info("Overwrite is set to True, existing files will be overwritten")
    logger.info("Generate scripts for preparing visit1 data")
    generate_scripts(pipeline_dir, data_raw_dir, code_dir, out_dir, modality, overwrite=overwrite, keepdicom=keepdicom)
    if retest_suffix is not None:
        logger.info("Generate scripts for preparing visit2 data")
        generate_scripts(
            pipeline_dir,
            data_raw_dir,
            code_dir,
            out_dir,
            modality,
            retest_suffix=retest_suffix,
            overwrite=overwrite,
            keepdicom=keepdicom,
        )
    else:
        logger.info("No retest_suffix is provided, only visit1 data will be prepared")
