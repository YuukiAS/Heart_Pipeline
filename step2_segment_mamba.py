import os
import shutil

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
from utils.log_utils import setup_logging
logger = setup_logging("main: segment(mamba)")


def generate_scripts(pipeline_dir, code_dir, modality_dict: dict, retest_suffix=None):
    if retest_suffix is None:
        code_step2_dir = os.path.join(code_dir, "segment_visit1_mamba")
    else:
        code_step2_dir = os.path.join(code_dir, "segment_visit2_mamba")

    if os.path.exists(code_step2_dir):
        shutil.rmtree(code_step2_dir)
    os.makedirs(code_step2_dir)

    # Sub-step2: Prepare data in nnUNet format
    # Make sure nii folder has already been prepared in step0
    with open(os.path.join(code_step2_dir, "prepareAll.sh"), "w") as file_submit:
        file_submit.write("#!/bin/bash\n")
        for dataset_id in modality_dict:
            modality = modality_dict[dataset_id]
            file_submit.write(f"sbatch prepare_{modality}.sh\n")
            with open(os.path.join(code_step2_dir, f"prepare_{modality}.sh"), "w") as file_script:
                file_script.write("#!/bin/bash\n")
                file_script.write("#SBATCH --ntasks=1\n")
                if retest_suffix is None:
                    file_script.write(f"#SBATCH --job-name=Heart_Segment(Mamba)_Prepare_{modality}\n")
                    file_script.write(f"#SBATCH --output=Heart_Segment(Mamba)_Prepare_{modality}.out\n")
                else:
                    file_script.write(f"#SBATCH --job-name=Heart_Segment(Mamba)_Prepare_{modality}_{retest_suffix}\n")
                    file_script.write(f"#SBATCH --output=Heart_Segment(Mamba)_Prepare_{modality}_{retest_suffix}.out\n")
                file_script.write("#SBATCH --cpus-per-task=4\n")
                file_script.write("#SBATCH --mem=8G\n")
                file_script.write("#SBATCH --time=48:00:00\n")
                file_script.write("#SBATCH --partition=general\n")
                file_script.write("\n")
                file_script.write("\n")
                file_script.write(f"cd {pipeline_dir}/script\n")
                if retest_suffix is None:
                    file_script.write(f"python prepare_data_nnunet.py -d {dataset_id} -m {modality}\n")
                else:
                    file_script.write(f"python prepare_data_nnunet.py -d {dataset_id} -m {modality} --retest\n")

    # Sub-step2: Preprocess the data
    with open(os.path.join(code_step2_dir, "preprocessAll.sh"), "w") as file_submit:
        file_submit.write("#!/bin/bash\n")
        for dataset_id in modality_dict:
            modality = modality_dict[dataset_id]
            file_submit.write(f"sbatch preprocess_{modality}.sh\n")
            with open(os.path.join(code_step2_dir, f"preprocess_{modality}.sh"), "w") as file_script:
                file_script.write("#!/bin/bash\n")
                file_script.write("#SBATCH --ntasks=1\n")
                if retest_suffix is None:
                    str1 = f"#SBATCH --job-name=Heart_Segment(Mamba)_Preprocess_{modality}\n"
                    str2 = f"#SBATCH --output=Heart_Segment(Mamba)_Preprocess_{modality}.out\n"
                else:
                    str1 = f"#SBATCH --job-name=Heart_Segment(Mamba)_Preprocess_{modality}_{retest_suffix}\n"
                    str2 = f"#SBATCH --output=Heart_Segment(Mamba)_Preprocess_{modality}_{retest_suffix}.out\n"
                file_script.write(str1)
                file_script.write(str2)
                file_script.write("#SBATCH --cpus-per-task=4\n")
                file_script.write("#SBATCH --mem=8G\n")
                file_script.write("#SBATCH --time=24:00:00\n")
                file_script.write("#SBATCH --partition=volta-gpu\n")
                file_script.write("#SBATCH --gres=gpu:1\n")
                file_script.write("#SBATCH --qos=gpu_access\n")
                file_script.write("\n")
                file_script.write("\n")
                if retest_suffix is None:
                    nnunet_dir = config.data_visit1_nnunet_dir
                else:
                    nnunet_dir = config.data_visit2_nnunet_dir
                file_script.write(
                    f"export nnUNet_raw={os.path.join(nnunet_dir, 'nnUNet_raw')}\n")
                file_script.write(
                    f"export nnUNet_preprocessed={os.path.join(nnunet_dir, 'nnUNet_preprocessed')}\n")
                file_script.write(
                    f"export nnUNet_results={os.path.join(nnunet_dir, 'nnUNet_results')}\n")
                file_script.write(f"source {config.env_mamba}\n")
                file_script.write(f"nnunetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity\n")


if __name__ == "__main__":
    pipeline_dir = config.pipeline_dir
    code_dir = config.code_dir
    modality = config.modality
    retest_suffix = config.retest_suffix

    modality_dict_visit1 = {}
    if "sa" in modality:
        modality_dict_visit1[100] = "sa"
    if "la" in modality:
        modality_dict_visit1[101] = "la_2ch"
        modality_dict_visit1[102] = "la_4ch"

    logger.info("Generate scripts to segment for visit1 data")
    generate_scripts(pipeline_dir, code_dir, modality_dict_visit1)
    if retest_suffix is not None:
        logger.info("Generate scripts to segment for visit2 data")
        modality_dict_visit2 = {}
        if "sa" in modality:
            modality_dict_visit2[110] = "sa"
        if "la" in modality:
            modality_dict_visit2[111] = "la_2ch"
            modality_dict_visit2[112] = "la_4ch"
        generate_scripts(pipeline_dir, code_dir, modality_dict_visit2, retest_suffix)
    else:
        logger.info("No retest_suffix is provided, only visit1 data will be segmented")
