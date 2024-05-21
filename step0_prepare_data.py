import os
import shutil
from tqdm import tqdm
import logging
import logging.config

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config

logging.config.fileConfig(config.logging_config)
logger = logging.getLogger('main')
logging.basicConfig(level=config.logging_level)

# This file is used to prepare the data for the pipeline using zip files

def generate_scripts(pipeline_dir, data_raw_dir, code_dir, out_dir, 
                     modality, num_subjects_per_file = 1000, retest_suffix = None):
    if retest_suffix is None:
        code_step0_dir = os.path.join(code_dir, "prepare_data_visit1")
        out_step0_dir = os.path.join(out_dir, "visit1/")

        long_axis = os.path.join(data_raw_dir, "20208/")
        short_axis = os.path.join(data_raw_dir, "20209/")
        aortic = os.path.join(data_raw_dir, "20210/")
        tagging = os.path.join(data_raw_dir, "20211/")
        LVOT = os.path.join(data_raw_dir, "20212/")
        blood_flow = os.path.join(data_raw_dir, "20213/")
        T1 = os.path.join(data_raw_dir, "20214/")
    else:
        code_step0_dir = os.path.join(code_dir, "prepare_data_visit2")
        out_step0_dir = os.path.join(out_dir, "visit2/")

        long_axis = os.path.join(data_raw_dir, f"20208_{retest_suffix}/")
        short_axis = os.path.join(data_raw_dir, f"20209_{retest_suffix}/")
        aortic = os.path.join(data_raw_dir, f"20210_{retest_suffix}/")
        tagging = os.path.join(data_raw_dir, f"20211_{retest_suffix}/")
        LVOT = os.path.join(data_raw_dir, f"20212_{retest_suffix}/")
        blood_flow = os.path.join(data_raw_dir, f"20213_{retest_suffix}/")
        T1 = os.path.join(data_raw_dir, f"20214_{retest_suffix}/")

    if os.path.exists(code_step0_dir):
        shutil.rmtree(code_step0_dir)
    os.makedirs(code_step0_dir)

    sub_total = []
    if "la" in modality:
        sub_la = os.listdir(long_axis)
        sub_la = [x.split("_")[0] for x in sub_la]

        sub_total = sub_total + sub_la
    if "sa" in modality:
        sub_sa = os.listdir(short_axis)
        sub_sa = [x.split("_")[0] for x in sub_sa]
    
        sub_total = sub_total + sub_sa
    if "aor" in modality:
        sub_aor = os.listdir(aortic)
        sub_aor = [x.split("_")[0] for x in sub_aor]

        sub_total = sub_total + sub_aor
    if "tag" in modality:
        sub_tag = os.listdir(tagging)
        sub_tag = [x.split("_")[0] for x in sub_tag]

        sub_total = sub_total + sub_tag
    if "lvot" in modality:
        sub_lvot = os.listdir(LVOT)
        sub_lvot = [x.split("_")[0] for x in sub_lvot]

        sub_total = sub_total + sub_lvot
    if "blood" in modality:
        sub_blood = os.listdir(blood_flow)
        sub_blood = [x.split("_")[0] for x in sub_blood]

        sub_total = sub_total + sub_blood
    if "t1" in modality:
        sub_t1 = os.listdir(T1)
        sub_t1 = [x.split("_")[0] for x in sub_t1]

        sub_total = sub_total + sub_t1


    length_total = len(sub_total)
    logger.info(f"Total number of subjects: {length_total}")
    num_files = length_total // num_subjects_per_file + 1

    with open(os.path.join(code_step0_dir, "batAll.sh"), "w") as file_submit:
        file_submit.write("#!/bin/bash\n")        
        for file_i in tqdm(range(num_files)):
            file_submit.write(f"sbatch bat{file_i}.pbs\n")
            with open(os.path.join(code_step0_dir, f"bat{file_i}.pbs"), "w") as file_script:
                file_script.write("#!/bin/bash\n")
                file_script.write("#SBATCH --ntasks=1\n")
                if retest_suffix is None:
                    file_script.write(f"#SBATCH --job-name=Heart_Prepare_{file_i}\n")
                    file_script.write(f"#SBATCH --output=Heart_Prepare_{file_i}.out\n")
                else:
                    file_script.write(f"#SBATCH --job-name=Heart_Prepare_{file_i}_{retest_suffix}\n")
                    file_script.write(f"#SBATCH --output=Heart_Prepare_{file_i}_{retest_suffix}.out\n")
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
                    option_str = ""
                    if "la" in modality:
                        option_str += " --long_axis=" + long_axis
                    if "sa" in modality:
                        option_str += " --short_axis=" + short_axis
                    if "aor" in modality:
                        option_str += " --aortic=" + aortic
                    if "tag" in modality:
                        option_str += " --tag=" + tagging
                    if "lvot" in modality:
                        option_str += " --lvot=" + LVOT
                    if "blood" in modality:
                        option_str += " --blood_flow=" + blood_flow
                    if "t1" in modality:
                        option_str += " --T1=" + T1
                    file_script.write(
                        f"python ./script/prepare_data.py --out_dir={out_step0_dir} --sub_id={sub_id} {option_str}\n"
                    )
    
if __name__ == "__main__":
    pipeline_dir = config.pipeline_dir
    data_raw_dir = config.data_raw_dir
    out_dir = config.data_out_dir
    code_dir = config.code_dir
    modality = config.modality
    retest_suffix = config.retest_suffix

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(code_dir, exist_ok=True)

    logger.info("Generate scripts for preparing visit1 data")
    generate_scripts(pipeline_dir, data_raw_dir, code_dir, out_dir, modality)
    if retest_suffix is not None:
        logger.info("Generate scripts for preparing visit2 data")
        generate_scripts(pipeline_dir, data_raw_dir, code_dir, out_dir, modality, retest_suffix=retest_suffix)
    else:
        logger.info("No retest_suffix is provided, only visit1 data will be prepared")