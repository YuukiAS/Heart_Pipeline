import os
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import config
from utils.log_utils import setup_logging

logger = setup_logging("segment_sa")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type = str, help="Folder for one subject that contains Nifti files", required=True)


if __name__ == "__main__":
    args = parser.parse_args()

    data_dir = args.data_dir

    model_dir = config.model_dir
    pipeline_dir = config.pipeline_dir

    os.system(f"cd {pipeline_dir}/src/segmentation/Short_Axis_20209 && "
          f"python ./deploy_network.py --seq_name sa "
          f"--data_dir {data_dir} --model_path {model_dir}/Short_Axis_20209/FCN_sa")
