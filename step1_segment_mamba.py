import os
import shutil

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
import logging
import logging.config

logging.config.fileConfig(config.logging_config)
logger = logging.getLogger("main")
logging.basicConfig(level=config.logging_level)

def generate_scripts(pipeline_dir, code_dir, modality_dict: dict, retest_suffix = None):
    if retest_suffix is None:
        code_step1_dir = os.path.join(code_dir, "segment_visit1_mamba")
    else:
        code_step1_dir = os.path.join(code_dir, "segment_visit2_mamba")

    if os.path.exists(code_step1_dir):
        shutil.rmtree(code_step1_dir)
    os.makedirs(code_step1_dir)

    # Sub-step1: Prepare data in nnUNet format
    # Make sure nii folder has already been prepared in step0
    with open(os.path.join(code_step1_dir, "prepareAll.sh"), "w") as file_submit:
        file_submit.write("#!/bin/bash\n")
        for dataset_id in modality_dict:
            modality = modality_dict[dataset_id]
            file_submit.write(f"sbatch prepare_{modality}.sh\n")
            with open(os.path.join(code_step1_dir, f"prepare_{modality}.sh"), "w") as file_script:
                file_script.write("#!/bin/bash\n")
                file_script.write("#SBATCH --ntasks=1\n")
                if retest_suffix is None:
                    file_script.write(f"#SBATCH --job-name=Heart_Segment_{modality}_mamba\n")
                    file_script.write(f"#SBATCH --output=Heart_Segment_{modality}_mamba.out\n")
                else:
                    file_script.write(f"#SBATCH --job-name=Heart_Segment_{modality}_mamba_{retest_suffix}\n")
                    file_script.write(f"#SBATCH --output=Heart_Segment_{modality}_mamba_{retest_suffix}.out\n")
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