import pandas as pd
import os
import glob
import argparse


def aggregate_csv(csv_dir, target_dir, prefix):
    """
    Aggregate csv files with the same prefix in the csv_dir. `eid` column must be present in each csv files.
    """
    csv_files = glob.glob(os.path.join(csv_dir, f"{prefix}*.csv"))

    comprehensive_df = pd.DataFrame()
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, index_col=0)
        comprehensive_df = pd.concat([comprehensive_df, df], ignore_index=True)
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
