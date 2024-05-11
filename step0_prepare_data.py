import os
import shutil
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config

# This file is used to prepare the data for the pipeline using zip files

pipeline_dir = config.pipeline_dir
data_raw_dir = config.data_raw_dir
out_dir = config.out_dir
code_dir = config.code_dir

os.makedirs(out_dir, exist_ok=True)
if os.path.exists(code_dir):
    shutil.rmtree(code_dir)
os.makedirs(code_dir, exist_ok=True)

long_axis = os.path.join(data_raw_dir, "20208/")
short_axis = os.path.join(data_raw_dir, "20209/")
aortic = os.path.join(data_raw_dir, "20210/")
tagging = os.path.join(data_raw_dir, "20211/")
LVOT = os.path.join(data_raw_dir, "20212/")
blood_flow = os.path.join(data_raw_dir, "20213/")
T1 = os.path.join(data_raw_dir, "20214/")

sub_la = os.listdir(os.path.join(data_raw_dir, "20208"))
sub_la = [x.split("_")[0] for x in sub_la]

sub_sa = os.listdir(os.path.join(data_raw_dir, "20209"))
sub_sa = [x.split("_")[0] for x in sub_sa]

sub_aor = os.listdir(os.path.join(data_raw_dir, "20210"))
sub_aor = [x.split("_")[0] for x in sub_aor]

sub_tag = os.listdir(os.path.join(data_raw_dir, "20211"))
sub_tag = [x.split("_")[0] for x in sub_tag]

sub_lvot = os.listdir(os.path.join(data_raw_dir, "20212"))
sub_lvot = [x.split("_")[0] for x in sub_lvot]

sub_blood = os.listdir(os.path.join(data_raw_dir, "20213"))
sub_blood = [x.split("_")[0] for x in sub_blood]

sub_t1 = os.listdir(os.path.join(data_raw_dir, "20214"))
sub_t1 = [x.split("_")[0] for x in sub_t1]

length_la = len(sub_la)
length_sa = len(sub_sa)
length_aor = len(sub_aor)
length_tag = len(sub_tag)
length_lvot = len(sub_lvot)
length_blood = len(sub_blood)
length_t1 = len(sub_t1)

# modify here to select the subjects you want to use
modality = ["la"]
# modality = ['la', 'sa', 'aor', 'tag', 'lvot', 'blood', 't1']
sub_total = []
if "la" in modality:
    sub_total = sub_total + sub_la
if "sa" in modality:
    sub_total = sub_total + sub_sa
if "aor" in modality:
    sub_total = sub_total + sub_aor
if "tag" in modality:
    sub_total = sub_total + sub_tag
if "lvot" in modality:
    sub_total = sub_total + sub_lvot
if "blood" in modality:
    sub_total = sub_total + sub_blood
if "t1" in modality:
    sub_total = sub_total + sub_t1


length_total = len(sub_total)
print(f"Total number of subjects: {length_total}")

num_subjects_per_file = 1000
num_files = length_total // num_subjects_per_file + 1

with open(os.path.join(code_dir, "batAll.sh"), "w") as file_submit:
    file_submit.write("#!/bin/bash\n")        
    for file_i in tqdm(range(num_files)):
        file_submit.write(f"sbatch bat{file_i}.pbs\n")
        with open(os.path.join(code_dir, "bat{}.pbs".format(file_i)), "w") as file_script:
            file_script.write("#!/bin/bash\n")
            file_script.write("#SBATCH --ntasks=1\n")
            file_script.write(f"#SBATCH --job-name=Heart_Prepare_{file_i}\n")
            file_script.write("#SBATCH --cpus-per-task=4\n")
            file_script.write("#SBATCH --mem=8G\n")
            file_script.write("#SBATCH --time=48:00:00\n")
            file_script.write("#SBATCH --partition=general\n")
            file_script.write(f"#SBATCH --output=Heart_Prepare_{file_i}.out\n")
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
                    f"python {config.pipeline_dir}/script/prepare_data.py "
                    f"--out_dir={out_dir} --sub_id={sub_id} {option_str}\n"
                )
