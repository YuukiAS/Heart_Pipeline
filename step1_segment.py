import os
import shutil
from tqdm import tqdm
import logging
import logging.config

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
from utils.os_utils import check_existing_file

logging.config.fileConfig(config.logging_config)
logger = logging.getLogger("main")
logging.basicConfig(level=config.logging_level)


def generate_scripts(pipeline_dir, data_dir, code_dir, modality, num_subjects_per_file=500, retest_suffix=None):
    if retest_suffix is None:
        code_step1_dir = os.path.join(code_dir, "segment_visit1")
    else:
        code_step1_dir = os.path.join(code_dir, "segment_visit2")

    if os.path.exists(code_step1_dir):
        shutil.rmtree(code_step1_dir)
    os.makedirs(code_step1_dir)

    sub_total = os.listdir(data_dir)
    length_total = len(sub_total)
    logger.info(f"Total number of subjects: {length_total}")
    num_files = length_total // num_subjects_per_file + 1

    with open(os.path.join(code_step1_dir, "batAll.sh"), "w") as file_submit:
        file_submit.write("#!/bin/bash\n")
        for file_i in tqdm(range(num_files)):
            file_submit.write(f"sbatch bat{file_i}.pbs\n")
            with open(os.path.join(code_step1_dir, f"bat{file_i}.pbs"), "w") as file_script:
                file_script.write("#!/bin/bash\n")
                file_script.write("#SBATCH --ntasks=1\n")
                if retest_suffix is None:
                    file_script.write(f"#SBATCH --job-name=Heart_Segment_{file_i}\n")
                    file_script.write(f"#SBATCH --output=Heart_Segment_{file_i}.out\n")
                else:
                    file_script.write(f"#SBATCH --job-name=Heart_Segment_{file_i}_{retest_suffix}\n")
                    file_script.write(f"#SBATCH --output=Heart_Segment_{file_i}_{retest_suffix}.out\n")
                file_script.write("#SBATCH --cpus-per-task=4\n")
                file_script.write("#SBATCH --mem=8G\n")
                file_script.write("#SBATCH --time=48:00:00\n")
                file_script.write("#SBATCH --partition=general\n")
                file_script.write("\n")
                file_script.write("\n")
                file_script.write(f"cd {pipeline_dir}\n")

                for sub_i in range(
                    file_i * num_subjects_per_file + 1,
                    min((file_i + 1) * num_subjects_per_file + 1, length_total + 1),
                ):
                    sub_id = sub_total[sub_i - 1]
                    sub_id_dir = os.path.join(data_dir, sub_id)

                    if "la" in modality:
                        if not check_existing_file(["seg_la_2ch.nii.gz"], sub_id_dir):
                            file_script.write(
                                f"echo ' {sub_id}: Generate segmentation scripts for horizontal long axis'\n"
                            )
                            file_script.write(
                                f"python ./src/segmentation/deploy_network.py --seq_name la_2ch "
                                f"--data_dir {sub_id_dir} --model_path ./model/FCN_la_2ch\n"
                            )
                        else:
                            logger.warning(f"{sub_id}: Segmentation for horizontal long axis already exists.")

                        if not check_existing_file(["seg_la_4ch.nii.gz"], sub_id_dir):
                            file_script.write(
                                f"echo ' {sub_id}: Generate segmentation scripts for vertical long axis (2chamber)'\n"
                            )
                            file_script.write(
                                f"python ./src/segmentation/deploy_network.py --seq_name la_4ch "
                                f"--data_dir {sub_id_dir} --model_path ./model/FCN_la_4ch\n"
                            )
                        else:
                            logger.warning(f"{sub_id}: Segmentation for vertical long axis (2chamber) already exists.")

                        if not check_existing_file(["seg4_la_4ch.nii.gz"], sub_id_dir):
                            file_script.write(
                                f"echo ' {sub_id}: Generate segmentation scripts for vertical long axis (4chamber)'\n"
                            )
                            file_script.write(
                                f"python ./src/segmentation/deploy_network.py --seq_name la_4ch "
                                f"--data_dir {sub_id_dir} --seg4 1 --model_path ./model/FCN_la_4ch_seg4\n"
                            )
                        else:
                            logger.warning(f"{sub_id}: Segmentation for vertical long axis (4chamber) already exists.")
                    if "sa" in modality:
                        # file_script.write(
                        #     f"echo ' {sub_id}: Generate segmentation scripts for short axis'\n"
                        # )
                        # file_script.write(
                        #     f"python ./src/segmentation/deploy_network.py --seq_name sa "\
                        #     f"--data_dir {sub_id_dir} --model_path ./model/FCN_sa\n"
                        # )
                        pass
                    if "aor" in modality:
                        # file_script.write(
                        #     f"echo ' {sub_id}: Generate segmentation scripts for aorta'\n"
                        # )
                        # file_script.write(
                        #     f"python ./src/segmentation/deploy_network.py --seq_name ao "\
                        #     f"--data_dir {sub_id_dir} --model_path ./model/UNet-LSTM_ao\n"
                        # )
                        pass
                    if "tag" in modality:
                        # todo No segmentation network right now
                        pass
                    if "lvot" in modality:
                        # todo No segmentation network right now
                        pass
                    if "blood" in modality:
                        # todo No segmentation network right now
                        pass
                    if "t1" in modality:
                        # todo No segmentation network right now
                        pass


if __name__ == "__main__":
    pipeline_dir = config.pipeline_dir
    data_visit1_dir = config.data_visit1_dir
    data_visit2_dir = config.data_visit2_dir
    code_dir = config.code_dir
    modality = config.modality
    retest_suffix = config.retest_suffix

    os.makedirs(code_dir, exist_ok=True)

    logger.info("Generate scripts to segment for visit1 data")
    generate_scripts(pipeline_dir, data_visit1_dir, code_dir, modality)
    if retest_suffix is not None:
        logger.info("Generate scripts to segment for visit2 data")
        generate_scripts(pipeline_dir, data_visit2_dir, code_dir, modality, retest_suffix=retest_suffix)
    else:
        logger.info("No retest_suffix is provided, only visit1 data will be segmented")
