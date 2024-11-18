import os
import shutil
from tqdm import tqdm

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config

from utils.slurm_utils import generate_header_cpu, generate_header_gpu, generate_aggregate
from utils.log_utils import setup_logging

logger = setup_logging("main: extract_feature_combined")


def generate_scripts(pipeline_dir, data_dir, code_dir, modality, num_subjects_per_file=100, retest_suffix=None, cpu=True):
    if retest_suffix is None:
        code_step4_dir = os.path.join(code_dir, "extract_feature_combined_visit1")
    else:
        code_step4_dir = os.path.join(code_dir, "extract_feature_combined_visit2")

    if os.path.exists(code_step4_dir):
        shutil.rmtree(code_step4_dir)
    os.makedirs(code_step4_dir)

    sub_total = os.listdir(data_dir)
    length_total = len(sub_total)
    logger.info(f"Total number of subjects: {length_total}")
    num_files = length_total // num_subjects_per_file + 1

    retest_str = "" if not retest_suffix else " --retest"  # most scripts only accept "--retest"
    with (
        open(os.path.join(code_step4_dir, "batAll.sh"), "w") as file_submit,
        open(os.path.join(code_step4_dir, "aggregate.pbs"), "w") as file_aggregate,
    ):
        file_submit.write("#!/bin/bash\n")

        if cpu:
            generate_header_cpu("Heart_Aggregate", pipeline_dir, None, file_aggregate, retest_suffix=retest_suffix)
        else:
            generate_header_gpu("Heart_Aggregate", pipeline_dir, None, file_aggregate, retest_suffix=retest_suffix)

        for file_i in tqdm(range(1, num_files + 1)):
            sub_file_i = sub_total[
                (file_i - 1) * num_subjects_per_file : min(file_i * num_subjects_per_file, length_total)
            ]
            sub_file_i_str = " ".join(map(str, sub_file_i))

            file_submit.write(f"sbatch bat{file_i}.pbs\n")
            with open(os.path.join(code_step4_dir, f"bat{file_i}.pbs"), "w") as file_script:
                if cpu:
                    generate_header_cpu("Heart_Extract", pipeline_dir, file_i, file_script, retest_suffix=retest_suffix)
                else:
                    generate_header_gpu("Heart_Extract", pipeline_dir, file_i, file_script, retest_suffix=retest_suffix)

                if "la" in modality and "sa" in modality:
                    file_script.write("echo 'Extract combined features that use both short axis and long axis'\n")
                    file_script.write(
                        f"python -u ./src/feature_extraction/Combined_Features/eval_ventricular_atrial_feature.py "
                        f"{retest_str} --file_name=ventricular_atrial_feature_{file_i} --data_list {sub_file_i_str} \n"
                    )

                # need the mean in first round
                if "shmolli" in modality:
                    file_script.write("echo 'Extract features for native T1 (corrected)'\n")
                    file_script.write(
                        f"python -u ./src/feature_extraction/Combined_Features/eval_native_t1.py "
                        f"{retest_str} --file_name=native_t1_corrected_{file_i} --data_list {sub_file_i_str} \n"
                    )
                
                if file_i == 1:

                    if "la" in modality and "sa" in modality:
                        generate_aggregate(file_aggregate, "ventricular_atrial_feature", retest_suffix=retest_suffix)
                    
                    if "shmolli" in modality:
                        generate_aggregate(file_aggregate, "native_t1_corrected", retest_suffix=retest_suffix)

                file_script.write("echo 'Done!'\n")

        file_aggregate.write("echo 'Done!'\n")


if __name__ == "__main__":
    pipeline_dir = config.pipeline_dir
    data_visit1_dir = config.data_visit1_dir
    data_visit2_dir = config.data_visit2_dir
    code_dir = config.code_dir
    modality = config.modality
    retest_suffix = config.retest_suffix

    os.makedirs(code_dir, exist_ok=True)

    logger.info("Generate scripts to extract combined (advanced) features for visit1 data")
    generate_scripts(pipeline_dir, data_visit1_dir, code_dir, modality)
    if retest_suffix is not None:
        logger.info("Generate scripts tp extract combined (advanced) features for visit2 data")
        generate_scripts(pipeline_dir, data_visit2_dir, code_dir, modality, retest_suffix=retest_suffix)
    else:
        logger.info("No retest_suffix is provided, only features for visit1 data will be extracted")
