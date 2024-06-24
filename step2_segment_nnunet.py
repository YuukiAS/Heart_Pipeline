import os
import shutil

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
from utils.log_utils import setup_logging
logger = setup_logging("main: segment(nnunet)")


def generate_scripts(pipeline_dir, data_dir, data_nnunet_dir, code_dir, modality, retest_suffix=None):
    if retest_suffix is None:
        code_step2_dir = os.path.join(code_dir, "segment_visit1_nnunet")
        retest_str = ""
    else:
        code_step2_dir = os.path.join(code_dir, "segment_visit2_nnunet")
        retest_str = "--retest"
    nnUNet_raw_path = os.path.join(data_nnunet_dir, "nnUNet_raw")
    nnUNet_preprocessed_path = os.path.join(data_nnunet_dir, "nnUNet_preprocessed")
    nnUNet_results_path = os.path.join(data_nnunet_dir, "nnUNet_results")

    if os.path.exists(code_step2_dir):
        shutil.rmtree(code_step2_dir)
    os.makedirs(code_step2_dir)

    sub_total = os.listdir(data_dir)
    length_total = len(sub_total)
    logger.info(f"Total number of subjects: {length_total}")

        
    with (
        open(os.path.join(code_step2_dir, "prepare.sh"), "w") as file_prepare,
        open(os.path.join(code_step2_dir, "preprocess.sh"), "w") as file_preprocess,
        open(os.path.join(code_step2_dir, "train.sh"), "w") as file_train,
        open(os.path.join(code_step2_dir, "inference.sh"), "w") as file_inference,
        open(os.path.join(code_step2_dir, "postprocess.sh"), "w") as file_postprocess,
    ):
    
        files = [file_prepare, file_preprocess, file_train, file_inference, file_postprocess]
        for file_script in files:
            file_script.write("#!/bin/bash\n")
            file_script.write("#SBATCH --cpus-per-task=4\n")
            file_script.write("#SBATCH --mem=32G\n")
            file_script.write("#SBATCH --time=48:00:00\n")
            file_script.write("#SBATCH --partition=htzhulab\n")
            file_script.write("#SBATCH --gres=gpu:1\n")
            file_script.write("#SBATCH --qos=gpu_access\n")

        file_prepare.write(f"#SBATCH --job-name=Heart_Segment_Prepare_{retest_suffix}(nnUNet)\n")
        file_prepare.write(f"#SBATCH --output=Heart_Segment_Prepare_{retest_suffix}.out\n")
        file_preprocess.write(f"#SBATCH --job-name=Heart_Segment_Preprocess_{retest_suffix}(nnUNet)\n")
        file_preprocess.write(f"#SBATCH --output=Heart_Segment_preprocess_{retest_suffix}.out\n")
        file_train.write(f"#SBATCH --job-name=Heart_Segment_Train_{retest_suffix}(nnUNet)\n")
        file_train.write(f"#SBATCH --output=Heart_Segment_Train_{retest_suffix}.out\n")
        file_inference.write(f"#SBATCH --job-name=Heart_Segment_Inference_{retest_suffix}(nnUNet)\n")
        file_inference.write(f"#SBATCH --output=Heart_Segment_Inference_{retest_suffix}.out\n")
        file_postprocess.write(f"#SBATCH --job-name=Heart_Segment_Postprocess_{retest_suffix}(nnUNet)\n")
        file_postprocess.write(f"#SBATCH --output=Heart_Segment_Postprocess_{retest_suffix}.out\n")

        for file_script in files:
            file_script.write(f"export nnUNet_raw={nnUNet_raw_path}\n")
            file_script.write(f"export nnUNet_preprocessed={nnUNet_preprocessed_path}\n")
            file_script.write(f"export nnUNet_results={nnUNet_results_path}\n")
            file_script.write(f"cd {pipeline_dir}\n")


        if "la" in modality:
            # if not check_existing_file(["seg_la_2ch.nii.gz"], sub_id_dir):
            #     file_script.write(
            #         f"echo ' {sub_id}: Generate segmentation scripts for horizontal long axis'\n"
            #     )
            #     file_script.write(
            #         f"python ./src/segmentation/deploy_network.py --seq_name la_2ch "
            #         f"--data_dir {sub_id_dir} --model_path ./model/FCN_la_2ch\n"
            #     )
            # else:
            #     logger.warning("Segmentation for horizontal long axis already exists.")

            # if not check_existing_file(["seg_la_4ch.nii.gz"], sub_id_dir):
            #     file_script.write(
            #         f"echo ' {sub_id}: Generate segmentation scripts for vertical long axis (2chamber)'\n"
            #     )
            #     file_script.write(
            #         f"python ./src/segmentation/deploy_network.py --seq_name la_4ch "
            #         f"--data_dir {sub_id_dir} --model_path ./model/FCN_la_4ch\n"
            #     )
            # else:
            #     logger.warning("Segmentation for vertical long axis (2chamber) already exists.")

            # if not check_existing_file(["seg4_la_4ch.nii.gz"], sub_id_dir):
            #     file_script.write(
            #         f"echo ' {sub_id}: Generate segmentation scripts for vertical long axis (4chamber)'\n"
            #     )
            #     file_script.write(
            #         f"python ./src/segmentation/deploy_network.py --seq_name la_4ch "
            #         f"--data_dir {sub_id_dir} --seg4 1 --model_path ./model/FCN_la_4ch_seg4\n"
            #     )
            # else:
            #     logger.warning("Segmentation for vertical long axis (4chamber) already exists.")
            pass
        if "sa" in modality:
            dataset_id = 100 if retest_suffix is None else 110

            for file_script in files:
                file_script.write("echo 'Prepare sa modality'\n")

            # todo change this to a function
            file_prepare.write(f"python ./script/prepare_data_nnunet.py -d {dataset_id} -m sa {retest_str}\n")
            file_preprocess.write(f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity\n")
            file_train.write(f"nnUNetv2_train {dataset_id} 3d_fullres all\n")
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
    data_visit1_nnunet_dir = config.data_visit1_nnunet_dir
    data_visit2_dir = config.data_visit2_dir
    data_visit2_nnunet_dir = config.data_visit2_nnunet_dir
    code_dir = config.code_dir
    modality = config.modality
    retest_suffix = config.retest_suffix

    os.makedirs(code_dir, exist_ok=True)

    logger.info("Generate scripts to segment for visit1 data")
    generate_scripts(pipeline_dir, data_visit1_dir, data_visit1_nnunet_dir,
                     code_dir, modality)
    if retest_suffix is not None:
        logger.info("Generate scripts to segment for visit2 data")
        generate_scripts(pipeline_dir, data_visit2_dir, data_visit2_nnunet_dir,
                         code_dir, modality, retest_suffix=retest_suffix)
    else:
        logger.info("No retest_suffix is provided, only visit1 data will be segmented")



# nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity
# nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD
# nnUNetv2_train DATASET_NAME_OR_ID 3d_lowres FOLD
# nnUNetv2_train DATASET_NAME_OR_ID 3d_cascade_fullres FOLD

# nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS 
# nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities