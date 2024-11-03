import os
import shutil
from tqdm import tqdm

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config

from utils.log_utils import setup_logging

logger = setup_logging("main: extract_feature_combined")


def generate_scripts(pipeline_dir, data_dir, code_dir, modality, num_subjects_per_file=200, retest_suffix=None):
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

        file_aggregate.write("#!/bin/bash\n")
        file_aggregate.write("#SBATCH --ntasks=1\n")
        if retest_suffix is None:
            file_aggregate.write("#SBATCH --job-name=Heart_Aggregate\n")
            file_aggregate.write("#SBATCH --output=Heart_Aggregate.out\n")
        else:
            file_aggregate.write(f"#SBATCH --job-name=Heart_Aggregate_{retest_suffix}\n")
            file_aggregate.write(f"#SBATCH --output=Heart_Aggregate_{retest_suffix}.out\n")
        file_aggregate.write("#SBATCH --cpus-per-task=4\n")
        file_aggregate.write("#SBATCH --mem=8G\n")
        file_aggregate.write("#SBATCH --time=24:00:00\n")
        file_aggregate.write("#SBATCH --partition=general\n")
        file_aggregate.write("\n")
        file_aggregate.write("\n")
        file_aggregate.write(f"cd {pipeline_dir}\n")

        for file_i in tqdm(range(1, num_files + 1)):
            sub_file_i = sub_total[
                (file_i - 1) * num_subjects_per_file : min(file_i * num_subjects_per_file, length_total)
            ]
            sub_file_i_str = " ".join(map(str, sub_file_i))

            file_submit.write(f"sbatch bat{file_i}.pbs\n")
            with open(os.path.join(code_step4_dir, f"bat{file_i}.pbs"), "w") as file_script:
                file_script.write("#!/bin/bash\n")
                file_script.write("#SBATCH --ntasks=1\n")
                if retest_suffix is None:
                    file_script.write(f"#SBATCH --job-name=Heart_Extract_{file_i}\n")
                    file_script.write(f"#SBATCH --output=Heart_Extract_{file_i}.out\n")
                else:
                    file_script.write(f"#SBATCH --job-name=Heart_Extract_{file_i}_{retest_suffix}\n")
                    file_script.write(f"#SBATCH --output=Heart_Extract_{file_i}_{retest_suffix}.out\n")
                file_script.write("#SBATCH --cpus-per-task=4\n")
                file_script.write("#SBATCH --mem=8G\n")
                file_script.write("#SBATCH --time=48:00:00\n")
                file_script.write("#SBATCH --partition=general\n")
                file_script.write("\n")
                file_script.write("\n")
                file_script.write(f"cd {pipeline_dir}\n")

                if "la" in modality and "sa" in modality:
                    file_script.write("echo 'Extract combined features that use both short axis and long axis'\n")
                    file_script.write(
                        f"python -u ./src/feature_extraction/combined/eval_ventricular_atrial_feature.py "
                        f"{retest_str} --file_name=ventricular_atrial_feature_{file_i} --data_list {sub_file_i_str} \n"
                    )
                    if file_i == 1:
                        if retest_suffix is None:
                            file_aggregate.write(
                                "python ./script/aggregate_csv.py "
                                f"--csv_dir={os.path.join(config.features_visit1_dir, 'combined')} "
                                f"--target_dir={os.path.join(config.features_visit1_dir, 'aggregated')} "
                                "--prefix=ventricular_atrial_feature\n"
                            )
                        else:
                            file_aggregate.write(
                                "python ./script/aggregate_csv.py "
                                f"--csv_dir={os.path.join(config.features_visit2_dir, 'combined')} "
                                f"--target_dir={os.path.join(config.features_visit2_dir, 'aggregated')} "
                                "--prefix=ventricular_atrial_feature\n"
                            )

                if "t1" in modality:
                    file_script.write("echo 'Extract features for native T1 (corrected)'\n")
                    file_script.write(
                        f"python -u ./src/feature_extraction/combined/eval_native_t1.py "
                        f"{retest_str} --file_name=native_t1_corrected_{file_i} --data_list {sub_file_i_str} \n"
                    )
                    if file_i == 1:
                        if retest_suffix is None:
                            file_aggregate.write(
                                "python ./script/aggregate_csv.py "
                                f"--csv_dir={os.path.join(config.features_visit1_dir, 'combined')} "
                                f"--target_dir={os.path.join(config.features_visit1_dir, 'aggregated')} "
                                "--prefix=native_t1_corrected\n"
                            )
                        else:
                            file_aggregate.write(
                                "python ./script/aggregate_csv.py "
                                f"--csv_dir={os.path.join(config.features_visit2_dir, 'combined')} "
                                f"--target_dir={os.path.join(config.features_visit2_dir, 'aggregated')} "
                                "--prefix=native_t1_corrected\n"
                            )

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
