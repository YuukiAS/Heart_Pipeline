import os
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
from utils.log_utils import setup_logging

logger = setup_logging("segment_la")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type = str, help="Folder for one subject that contains Nifti files", type=int, required=True)
parser.add_argument("--modality", type = str, help="Modality of the long axis",required=True)


if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = args.data_dir
    modality = args.modality

    model_dir = config.model_dir

    if modality == "2ch":
        os.system("python ./src/segmentation/deploy_network.py --seq_name la_2ch "
                  f"--data_dir {data_dir} --model_path {model_dir}/Long_Axis_20208/FCN_la_2ch\n")
        
    elif modality == "4ch":
        os.system("python ./src/segmentation/deploy_network.py --seq_name la_4ch "
                  f"--data_dir {data_dir} --model_path {model_dir}/Long_Axis_20208/FCN_la_4ch\n")
    elif modality == "4ch_4chamber":
        os.system("python ./src/segmentation/deploy_network.py --seq_name la_4ch "
                  f"--data_dir {data_dir} --seg4 1 --model_path {model_dir}/Long_Axis_20208/FCN_la_4ch_seg4\n")
    elif modality == "3ch":
        # todo Need to be implemented
        pass
    else:
        raise ValueError(f"Modality {modality} is not supported")
    
