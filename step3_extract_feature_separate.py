import os
import shutil
from tqdm import tqdm

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config

from utils.slurm_utils import generate_header_cpu, generate_header_gpu, generate_aggregate
from utils.log_utils import setup_logging

logger = setup_logging("main: extract_feature_separate")


def generate_scripts(
    pipeline_dir,
    data_dir,
    code_dir,
    modality,
    useECG,
    num_subjects_per_file=100,
    num_subjects_per_unit=5,
    retest_suffix=None,
    cpu=True,
):
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

        if cpu:
            generate_header_cpu("Heart_Aggregate", pipeline_dir, None, file_aggregate, retest_suffix=retest_suffix)
        else:
            generate_header_gpu("Heart_Aggregate", pipeline_dir, None, file_aggregate, retest_suffix=retest_suffix)

        if useECG is True:
            if cpu:
                generate_header_cpu("Heart_ECG", pipeline_dir, None, file_ecg, retest_suffix=retest_suffix)
            else:
                generate_header_gpu("Heart_ECG", pipeline_dir, None, file_ecg, retest_suffix=retest_suffix)

            file_ecg.write("echo 'Extract ECG features'\n")
            file_ecg.write(f"python -u ./src/feature_extraction/ecg/ecg_neurokit.py {retest_str}\n")
            file_ecg.write("echo 'Done!'\n")
            file_submit.write("sbatch batECG.sh\n")

        for file_i in tqdm(range(1, num_files + 1)):  # start from 1
            sub_file_i = sub_total[(file_i - 1) * num_subjects_per_file : min(file_i * num_subjects_per_file, length_total)]
            num_units = len(sub_file_i) // num_subjects_per_unit + (len(sub_file_i) % num_subjects_per_unit > 0)

            file_submit.write(f"sbatch bat{file_i}.pbs\n")
            with open(os.path.join(code_step3_dir, f"bat{file_i}.pbs"), "w") as file_script:
                if cpu:
                    generate_header_cpu("Heart_Extract", pipeline_dir, file_i, file_script, retest_suffix=retest_suffix)
                else:
                    generate_header_gpu("Heart_Extract", pipeline_dir, file_i, file_script, retest_suffix=retest_suffix)

                for unit in range(num_units):
                    start_idx = unit * num_subjects_per_unit
                    end_idx = min((unit + 1) * num_subjects_per_unit, len(sub_file_i))
                    sub_file_unit = sub_file_i[start_idx:end_idx]
                    sub_file_unit_str = " ".join(map(str, sub_file_unit))

                    file_script.write(f"echo 'Extract features for {sub_file_unit_str}'\n")

                    if "aortic_scout" in modality:
                        file_script.write("echo 'Extract features for aortic structure'\n")
                        file_script.write(
                            f"python -u ./src/feature_extraction/Aortic_Scout_20207/eval_aortic_structure.py "
                            f"{retest_str} --file_name=aortic_structure_{file_i} --data_list {sub_file_unit_str} \n"
                        )

                    if "la" in modality:
                        file_script.write("echo 'Extract features for atrial volume'\n")
                        file_script.write(
                            f"python -u ./src/feature_extraction/Long_Axis_20208/eval_atrial_volume.py "
                            f"{retest_str} --file_name=atrial_volume_{file_i} --data_list {sub_file_unit_str} \n"
                        )

                    if "sa" in modality:
                        file_script.write("echo 'Extract features for ventricular volume'\n")
                        file_script.write(
                            f"python -u ./src/feature_extraction/Short_Axis_20209/eval_ventricular_volume.py "
                            f"{retest_str} --file_name=ventricular_volume_{file_i} --data_list {sub_file_unit_str} \n"
                        )
                        file_script.write("echo 'Extract features for wall thickness'\n")
                        file_script.write(
                            f"python -u ./src/feature_extraction/Short_Axis_20209/eval_wall_thickness.py "
                            f"{retest_str} --file_name=wall_thickness_{file_i} --data_list {sub_file_unit_str} \n"
                        )

                    if "la" in modality and "sa" in modality:
                        file_script.write("echo 'Extract features for longitudinal strain using long axis'\n")
                        file_script.write(
                            f"python -u ./src/feature_extraction/Long_Axis_20208/eval_strain_lax.py "
                            f"{retest_str} --file_name=strain_la_{file_i} --data_list {sub_file_unit_str} \n"
                        )
                        file_script.write("echo 'Extract features for cirumferential and radial strain using short axis'\n")
                        file_script.write(
                            f"python -u ./src/feature_extraction/Short_Axis_20209/eval_strain_sax.py "
                            f"{retest_str} --file_name=strain_sa_{file_i} --data_list {sub_file_unit_str} \n"
                        )

                    if "aortic_dist" in modality:
                        file_script.write("echo 'Extract features for aortic distensibility'\n")
                        file_script.write(
                            f"python -u ./src/feature_extraction/Aortic_Distensibility_20210/eval_aortic_dist.py "
                            f"{retest_str} --file_name=aortic_dist_{file_i} --data_list {sub_file_unit_str} \n"
                        )

                    if "tag" in modality:
                        file_script.write("echo 'Extract features for cirumferential and radial strain using tagged MRI'\n")
                        file_script.write(
                            f"python -u ./src/feature_extraction/Tagged_20211/eval_strain_tagged.py "
                            f"{retest_str} --file_name=strain_tagged_{file_i} --data_list {sub_file_unit_str} \n"
                        )

                    if "lvot" in modality:
                        file_script.write("echo 'Extract features for LVOT diameters'\n")
                        file_script.write(
                            f"python -u ./src/feature_extraction/LVOT_20212/eval_LVOT.py "
                            f"{retest_str} --file_name=LVOT_{file_i} --data_list {sub_file_unit_str} \n"
                        )

                    if "aortic_blood_flow" in modality:
                        file_script.write("echo 'Extract features for aortic blood flow features'\n")
                        file_script.write(
                            f"python -u ./src/feature_extraction/Phase_Contrast_20213/eval_phase_contrast.py "
                            f"{retest_str} --file_name=aortic_flow_{file_i} --data_list {sub_file_unit_str} \n"
                        )

                    if "shmolli" in modality:
                        file_script.write("echo 'Extract features for Native T1 features'\n")
                        file_script.write(
                            f"python -u ./src/feature_extraction/Native_T1_20214/eval_native_t1.py "
                            f"{retest_str} --file_name=native_T1_{file_i} --data_list {sub_file_unit_str} \n"
                        )

                    if modality is not None:
                        file_script.write("\n")

                if file_i == 1:
                    if "aortic_scout" in modality:
                        generate_aggregate(file_aggregate, "aortic_structure", retest_suffix=retest_suffix)

                    if "la" in modality:
                        generate_aggregate(file_aggregate, "atrial_volume", retest_suffix=retest_suffix)
                        generate_aggregate(file_aggregate, "strain_la", retest_suffix=retest_suffix)

                    if "sa" in modality:
                        generate_aggregate(file_aggregate, "ventricular_volume", retest_suffix=retest_suffix)
                        generate_aggregate(file_aggregate, "wall_thickness", retest_suffix=retest_suffix)
                        generate_aggregate(file_aggregate, "strain_sa", retest_suffix=retest_suffix)

                    if "aortic_dist" in modality:
                        generate_aggregate(file_aggregate, "aortic_dist", retest_suffix=retest_suffix)

                    if "tag" in modality:
                        generate_aggregate(file_aggregate, "strain_tagged", retest_suffix=retest_suffix)

                    if "lvot" in modality:
                        generate_aggregate(file_aggregate, "LVOT", retest_suffix=retest_suffix)

                    if "aortic_blood_flow" in modality:
                        generate_aggregate(file_aggregate, "aortic_flow", retest_suffix=retest_suffix)

                    if "shmolli" in modality:
                        generate_aggregate(file_aggregate, "native_T1", retest_suffix=retest_suffix)

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
