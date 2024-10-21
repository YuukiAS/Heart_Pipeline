import os
import shutil
from tqdm import tqdm

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config

from utils.log_utils import setup_logging

logger = setup_logging("main: extract_feature_basic")


def generate_scripts(pipeline_dir, data_dir, code_dir, modality, useECG, num_subjects_per_file=200, retest_suffix=None):
    if retest_suffix is None:
        code_step3_dir = os.path.join(code_dir, "extract_feature_visit1")
    else:
        code_step3_dir = os.path.join(code_dir, "extract_feature_visit2")

    if os.path.exists(code_step3_dir):
        shutil.rmtree(code_step3_dir)
    os.makedirs(code_step3_dir)

    sub_total = os.listdir(data_dir)
    length_total = len(sub_total)
    logger.info(f"Total number of subjects: {length_total}")
    num_files = length_total // num_subjects_per_file + 1

    retest_str = "" if not retest_suffix else " --retest"  # most scripts only accept "--retest"
    with (
        open(os.path.join(code_step3_dir, "batAll.sh"), "w") as file_submit,
        open(os.path.join(code_step3_dir, "aggregate.pbs"), "w") as file_aggregate,
        open(os.path.join(code_step3_dir, "batECG.sh"), "w") as file_ecg,  # optional
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

        if useECG is True:
            file_ecg.write("#!/bin/bash\n")
            file_ecg.write("#SBATCH --ntasks=1\n")
            if retest_suffix is None:
                file_ecg.write("#SBATCH --job-name=Heart_ECG\n")
                file_ecg.write("#SBATCH --output=Heart_ECG.out\n")
            else:
                file_ecg.write(f"#SBATCH --job-name=Heart_ECG_{retest_suffix}\n")
                file_ecg.write(f"#SBATCH --output=Heart_ECG_{retest_suffix}.out\n")
            file_ecg.write("#SBATCH --cpus-per-task=4\n")
            file_ecg.write("#SBATCH --mem=8G\n")
            file_ecg.write("#SBATCH --time=48:00:00\n")
            file_ecg.write("#SBATCH --partition=general\n")
            file_ecg.write("\n")
            file_ecg.write("\n")
            file_ecg.write(f"cd {pipeline_dir}\n")
            file_ecg.write("echo 'Extract ECG features'\n")
            file_ecg.write(f"python -u ./src/feature_extraction/ecg/ecg_neurokit.py {retest_str}\n")
            file_ecg.write("echo 'Done!'\n")
            file_submit.write("sbatch batECG.sh\n")

        for file_i in tqdm(range(1, num_files + 1)):  # start from 1
            sub_file_i = sub_total[
                (file_i - 1) * num_subjects_per_file : min(file_i * num_subjects_per_file, length_total)
            ]
            sub_file_i_str = " ".join(map(str, sub_file_i))

            file_submit.write(f"sbatch bat{file_i}.pbs\n")
            with open(os.path.join(code_step3_dir, f"bat{file_i}.pbs"), "w") as file_script:
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

                if "la" in modality:
                    file_script.write("echo 'Extract features for atrial volume'\n")
                    file_script.write(
                        f"python -u ./src/feature_extraction/long_axis/eval_atrial_volume.py "
                        f"{retest_str} --file_name=atrial_volume_{file_i} --data_list {sub_file_i_str} \n"
                    )
                    file_script.write("echo 'Extract features for longitudinal strain'\n")
                    file_script.write(
                        f"python -u ./src/feature_extraction/long_axis/eval_strain_lax.py "
                        f"{retest_str} --file_name=strain_la_{file_i} --data_list {sub_file_i_str} \n"
                    )

                    # Script for aggregating separate feature files
                    if file_i == 1:
                        if retest_suffix is None:
                            file_aggregate.write(
                                "python ./script/aggregate_csv.py "
                                f"--csv_dir={os.path.join(config.features_visit1_dir, 'atrium')} "
                                f"--target_dir={os.path.join(config.features_visit1_dir, 'comprehensive')} "
                                "--prefix=atrial_volume\n"
                            )
                            file_aggregate.write(
                                "python ./script/aggregate_csv.py "
                                f"--csv_dir={os.path.join(config.features_visit1_dir, 'strain')} "
                                f"--target_dir={os.path.join(config.features_visit1_dir, 'comprehensive')} "
                                "--prefix=strain_la\n"
                            )
                        else:
                            file_aggregate.write(
                                "python ./script/aggregate_csv.py "
                                f"--csv_dir={os.path.join(config.features_visit2_dir, 'atrium')} "
                                f"--target_dir={os.path.join(config.features_visit2_dir, 'comprehensive')} "
                                "--prefix=atrial_volume\n"
                            )
                            file_aggregate.write(
                                "python ./script/aggregate_csv.py "
                                f"--csv_dir={os.path.join(config.features_visit2_dir, 'strain')} "
                                f"--target_dir={os.path.join(config.features_visit2_dir, 'comprehensive')} "
                                "--prefix=strain_la\n"
                            )
                if "sa" in modality:
                    file_script.write("echo 'Extract features for ventricular volume'\n")
                    file_script.write(
                        f"python -u ./src/feature_extraction/short_axis/eval_ventricular_volume.py "
                        f"{retest_str} --file_name=ventricular_volume_{file_i} --data_list {sub_file_i_str} \n"
                    )
                    file_script.write("echo 'Extract features for wall thickness'\n")
                    file_script.write(
                        f"python -u ./src/feature_extraction/short_axis/eval_wall_thickness.py "
                        f"{retest_str} --file_name=wall_thickness_{file_i} --data_list {sub_file_i_str} \n"
                    )
                    file_script.write("echo 'Extract features for cirumferential and radial strain'\n")
                    file_script.write(
                        f"python -u ./src/feature_extraction/short_axis/eval_strain_sax.py "
                        f"{retest_str} --file_name=strain_sa_{file_i} --data_list {sub_file_i_str} \n"
                    )

                    # Script for aggregating separate feature files
                    if file_i == 1:
                        if retest_suffix is None:
                            file_aggregate.write(
                                "python ./script/aggregate_csv.py "
                                f"--csv_dir={os.path.join(config.features_visit1_dir, 'ventricle')} "
                                f"--target_dir={os.path.join(config.features_visit1_dir, 'comprehensive')} "
                                "--prefix=ventricular_volume\n"
                            )
                            file_aggregate.write(
                                "python ./script/aggregate_csv.py "
                                f"--csv_dir={os.path.join(config.features_visit1_dir, 'wall_thickness')} "
                                f"--target_dir={os.path.join(config.features_visit1_dir, 'comprehensive')} "
                                "--prefix=wall_thickness\n"
                            )
                            file_aggregate.write(
                                "python ./script/aggregate_csv.py "
                                f"--csv_dir={os.path.join(config.features_visit1_dir, 'strain')} "
                                f"--target_dir={os.path.join(config.features_visit1_dir, 'comprehensive')} "
                                "--prefix=strain_sa\n"
                            )
                        else:
                            file_aggregate.write(
                                "python ./script/aggregate_csv.py "
                                f"--csv_dir={os.path.join(config.features_visit2_dir, 'ventricle')} "
                                f"--target_dir={os.path.join(config.features_visit2_dir, 'comprehensive')} "
                                "--prefix=ventricular_volume\n"
                            )
                            file_aggregate.write(
                                "python ./script/aggregate_csv.py "
                                f"--csv_dir={os.path.join(config.features_visit2_dir, 'wall_thickness')} "
                                f"--target_dir={os.path.join(config.features_visit2_dir, 'comprehensive')} "
                                "--prefix=wall_thickness\n"
                            )
                            file_aggregate.write(
                                "python ./script/aggregate_csv.py "
                                f"--csv_dir={os.path.join(config.features_visit2_dir, 'strain')} "
                                f"--target_dir={os.path.join(config.features_visit2_dir, 'comprehensive')} "
                                "--prefix=strain_sa\n"
                            )
                if "aor" in modality:
                    pass
                if "tag" in modality:
                    pass
                if "lvot" in modality:
                    pass
                if "blood" in modality:
                    pass
                if "t1" in modality:
                    file_script.write("echo 'Extract features for native T1 (uncorrected)'\n")
                    file_script.write(
                        f"python -u ./src/feature_extraction/native_t1/eval_native_t1.py "
                        f"{retest_str} --file_name=native_t1_uncorrected_{file_i} --data_list {sub_file_i_str} \n"
                    )

                    # Script for aggregating separate feature files
                    if file_i == 1:
                        if retest_suffix is None:
                            file_aggregate.write(
                                "python ./script/aggregate_csv.py "
                                f"--csv_dir={os.path.join(config.features_visit1_dir, 'native_t1')} "
                                f"--target_dir={os.path.join(config.features_visit1_dir, 'comprehensive')} "
                                "--prefix=native_t1_uncorrected\n"
                            )
                        else:
                            file_aggregate.write(
                                "python ./script/aggregate_csv.py "
                                f"--csv_dir={os.path.join(config.features_visit2_dir, 'native_t1')} "
                                f"--target_dir={os.path.join(config.features_visit2_dir, 'comprehensive')} "
                                "--prefix=native_t1_uncorrected\n"
                            )

                file_script.write("echo 'Done!'\n")

        file_aggregate.write("echo 'Done!'\n")


if __name__ == "__main__":
    pipeline_dir = config.pipeline_dir
    data_visit1_dir = config.data_visit1_dir
    data_visit2_dir = config.data_visit2_dir
    code_dir = config.code_dir
    modality = config.modality
    useECG = config.useECG
    retest_suffix = config.retest_suffix

    os.makedirs(code_dir, exist_ok=True)

    logger.info("Generate scripts to extract features for visit1 data")
    generate_scripts(pipeline_dir, data_visit1_dir, code_dir, modality, useECG)
    if retest_suffix is not None:
        logger.info("Generate scripts tp extract features for visit2 data")
        generate_scripts(pipeline_dir, data_visit2_dir, code_dir, modality, useECG, retest_suffix=retest_suffix)
    else:
        logger.info("No retest_suffix is provided, only features for visit1 data will be extracted")
