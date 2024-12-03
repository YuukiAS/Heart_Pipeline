import pandas as pd
import os
import glob
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.log_utils import setup_logging
logger = setup_logging("aggregate_csv")


def aggregate_csv(csv_dir, target_dir, prefix):
    """
    Aggregate csv files with the same prefix in the csv_dir. `eid` column must be present in each csv files.
    """
    csv_files = glob.glob(os.path.join(csv_dir, f"{prefix}*.csv"))
    logger.info(f"Used CSV files: {csv_files}")
    if len(csv_files) == 0:
        logger.error(f"No available csv files found with prefix {prefix} in {csv_dir}")
        raise ValueError(f"No available csv files found with prefix {prefix} in {csv_dir}")
    comprehensive_df = pd.DataFrame()
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            comprehensive_df = pd.concat([comprehensive_df, df], ignore_index=True)
        except pd.errors.EmptyDataError:
            logger.error(f"Empty csv file {csv_file} found. The file is skipped.")
            continue

    comprehensive_df = comprehensive_df.sort_values("eid")
    os.makedirs(target_dir, exist_ok=True)
    comprehensive_df.to_csv(os.path.join(target_dir, f"{prefix}.csv"), index=False)
    print(f"Aggregated csv file saved to {os.path.join(target_dir, f'{prefix}.csv')}")


parser = argparse.ArgumentParser()
parser.add_argument("--csv_dir", help="Directory of the csv files to be aggregated", required=True)
parser.add_argument("--target_dir", help="Directory to store the aggregated csv file", required=True)
parser.add_argument("--prefix", help="Prefix of the csv files to be aggregated", required=True)

if __name__ == "__main__":
    args = parser.parse_args()

    aggregate_csv(args.csv_dir, args.target_dir, args.prefix)
