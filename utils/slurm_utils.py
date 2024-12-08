import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


def generate_header_cpu(jobname, pipeline_dir, file_i, file_script, retest_suffix=None):
    file_script.write("#!/bin/bash\n")
    file_script.write("#SBATCH --ntasks=1\n")
    file_script.write("#SBATCH --cpus-per-task=16\n")
    if retest_suffix is None:
        if file_i:
            file_script.write(f"#SBATCH --job-name={jobname}_{file_i}\n")
            file_script.write(f"#SBATCH --output={jobname}_{file_i}.out\n")
        else:
            file_script.write(f"#SBATCH --job-name={jobname}\n")
            file_script.write(f"#SBATCH --output={jobname}.out\n")
    else:
        if file_i:
            file_script.write(f"#SBATCH --job-name={jobname}_{file_i}_{retest_suffix}\n")
            file_script.write(f"#SBATCH --output={jobname}_{file_i}_{retest_suffix}.out\n")
        else:
            file_script.write(f"#SBATCH --job-name={jobname}_{retest_suffix}\n")
            file_script.write(f"#SBATCH --output={jobname}_{retest_suffix}.out\n")
    file_script.write("#SBATCH --mem=16G\n")
    file_script.write("#SBATCH --time=48:00:00\n")
    file_script.write("#SBATCH --partition=general\n")
    file_script.write("\n")
    file_script.write("\n")
    file_script.write(f"cd {pipeline_dir}\n")
    file_script.write("\n")


def generate_header_gpu(jobname, pipeline_dir, file_i, file_script, retest_suffix=None):
    file_script.write("#!/bin/bash\n")
    file_script.write("#SBATCH --ntasks=1\n")
    file_script.write("#SBATCH --cpus-per-task=16\n")
    if retest_suffix is None:
        if file_i:
            file_script.write(f"#SBATCH --job-name={jobname}_{file_i}\n")
            file_script.write(f"#SBATCH --output={jobname}_{file_i}.out\n")
        else:
            file_script.write(f"#SBATCH --job-name={jobname}\n")
            file_script.write(f"#SBATCH --output={jobname}.out\n")
    else:
        if file_i:
            file_script.write(f"#SBATCH --job-name={jobname}_{file_i}_{retest_suffix}\n")
            file_script.write(f"#SBATCH --output={jobname}_{file_i}_{retest_suffix}.out\n")
        else:
            file_script.write(f"#SBATCH --job-name={jobname}_{retest_suffix}\n")
            file_script.write(f"#SBATCH --output={jobname}_{retest_suffix}.out\n")
    file_script.write("#SBATCH --mem=16G\n")
    file_script.write("#SBATCH --time=48:00:00\n")
    file_script.write(f"#SBATCH --partition={config.partition}\n")
    file_script.write("#SBATCH --gres=gpu:1\n")
    file_script.write("#SBATCH --qos=gpu_access\n")
    file_script.write("\n")
    file_script.write('source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"\n')
    file_script.write("\n")
    file_script.write("\n")
    file_script.write(f"cd {pipeline_dir}\n")
    file_script.write("\n")


def generate_aggregate(file_aggregate, csv_folder_name, feature_name=None, retest_suffix=None):
    if retest_suffix is None:
        file_aggregate.write(
            "python ./script/aggregate_csv.py "
            f"--csv_dir={os.path.join(config.features_visit1_dir, csv_folder_name)} "
            f"--target_dir={os.path.join(config.features_visit1_dir, 'aggregated')} "
            f"--prefix={csv_folder_name if not feature_name else feature_name}\n"
        )
        file_aggregate.write(f"rm -r {os.path.join(config.features_visit1_dir, csv_folder_name)}\n\n")

    else:
        file_aggregate.write(
            "python ./script/aggregate_csv.py "
            f"--csv_dir={os.path.join(config.features_visit2_dir, csv_folder_name)} "
            f"--target_dir={os.path.join(config.features_visit2_dir, 'aggregated')} "
            f"--prefix={csv_folder_name if not feature_name else feature_name}\n"
        )
        file_aggregate.write(f"rm -r {os.path.join(config.features_visit2_dir, csv_folder_name)}\n\n")
