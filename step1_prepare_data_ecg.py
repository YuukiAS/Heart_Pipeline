"""
This script is used to prepare the ECG data for the pipeline using xml files.
Please run step1_prepare_data_cmr.py before running this script.
For the format of XML file, please refer to https://biobank.ndph.ox.ac.uk/ukb/refer.cgi?id=123481.
"""

import os
import shutil
from tqdm import tqdm

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config

from utils.log_utils import setup_logging
logger = setup_logging("main: prepare_data_ecg")

def prepare_files_rest(pipeline_dir, data_raw_dir, out_dir, retest_suffix=None):
    if retest_suffix is None:
        out_step1_dir = config.data_visit1_dir
        ecg_dir = os.path.join(data_raw_dir, "20205")
    else:
        out_step1_dir = config.data_visit2_dir
        ecg_dir = os.path.join(data_raw_dir, f"20205_{retest_suffix}/")

    sub_total = os.listdir(ecg_dir)
    sub_total = [x.split("_")[0] for x in sub_total]

    logger.info(f"Total number of subjects with ECG rest XML: {len(sub_total)}")

    for sub in tqdm(sub_total):
        if retest_suffix is None:
            xml_file_target = os.path.join(ecg_dir, f"{sub}_20205_2_0.xml")
        else:
            xml_file_target = os.path.join(ecg_dir, f"{sub}_20205_3_0.xml")

        xml_file_dest_folder = os.path.join(out_step1_dir, sub)
        xml_file_dest = os.path.join(xml_file_dest_folder, "ecg_rest.xml")
        if not os.path.exists(xml_file_dest_folder):
            logger.warning(f"Subject {sub} does not have corresponding CMR data, skip.")
            continue
        shutil.copy(xml_file_target, xml_file_dest)

def prepare_files_exercise(pipeline_dir, data_raw_dir, out_dir, retest_suffix=None):
    if retest_suffix is None:
        out_step1_dir = config.data_visit1_dir
        ecg_dir = os.path.join(data_raw_dir, "6025")
    else:
        out_step1_dir = config.data_visit2_dir
        ecg_dir = os.path.join(data_raw_dir, f"6025_{retest_suffix}/")

    sub_total = os.listdir(ecg_dir)
    sub_total = [x.split("_")[0] for x in sub_total]

    logger.info(f"Total number of subjects with ECG exercise XML: {len(sub_total)}")

    for sub in tqdm(sub_total):
        if retest_suffix is None:
            xml_file_target = os.path.join(ecg_dir, f"{sub}_20205_2_0.xml")
        else:
            xml_file_target = os.path.join(ecg_dir, f"{sub}_20205_3_0.xml")

        xml_file_dest_folder = os.path.join(out_step1_dir, sub)
        xml_file_dest = os.path.join(xml_file_dest_folder, "ecg_exercise.xml")
        if not os.path.exists(xml_file_dest_folder):
            logger.warning(f"Subject {sub} does not have corresponding CMR data, skip.")
            continue
        shutil.copy(xml_file_target, xml_file_dest)


if __name__ == "__main__":
    pipeline_dir = config.pipeline_dir
    data_raw_dir = config.data_raw_dir
    out_dir = config.data_dir
    retest_suffix = config.retest_suffix

    os.makedirs(out_dir, exist_ok=True)

    logger.info("Copying ECG files for visit1 data")
    prepare_files_rest(pipeline_dir, data_raw_dir, out_dir)
    prepare_files_exercise(pipeline_dir, data_raw_dir, out_dir)
    if retest_suffix is not None:
        logger.info("Copying ECG files for visit2 data")
        prepare_files_rest(pipeline_dir, data_raw_dir, out_dir, retest_suffix=retest_suffix)
        prepare_files_exercise(pipeline_dir, data_raw_dir, out_dir, retest_suffix=retest_suffix)
    else:
        logger.info("No retest_suffix is provided, only visit1 data will be copied")