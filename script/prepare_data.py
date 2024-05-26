# Copyright 2017, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import argparse
import glob
import pandas as pd
import numpy as np
import shutil
import dateutil.parser
import logging
import logging.config
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
from utils.biobank_utils import process_manifest, Biobank_Dataset
from utils.os_utils import check_existing_file
import utils.parse_cvi42_xml as parse_cvi42_xml

logging.config.fileConfig(config.logging_config)
logger = logging.getLogger("prepare_data")
logging.basicConfig(level=config.logging_level)

contour_gt_dir = config.contour_gt_dir

argparser = argparse.ArgumentParser("Prepare data for the pipeline")
argparser.add_argument("--out_dir", required=True, type=str, help="Directory to store the data")
argparser.add_argument("--sub_id", required=True, type=str, help="ID of the subject")
argparser.add_argument("--long_axis", help="Directory to store the long axis data")
argparser.add_argument("--short_axis", help="Directory to store the short axis data")
argparser.add_argument("--aortic", help="Directory to store the aortic data")
argparser.add_argument("--tag", help="Directory to store the tagging data")
argparser.add_argument("--lvot", help="Directory to store the LVOT data")
argparser.add_argument("--blood_flow", help="Directory to store the blood flow data")
argparser.add_argument("--T1", help="Directory to store the experimental shMOLLI data")
# action="store_true" means that if the flag is present, the value is set to True
argparser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite all existing zip files",
)

args = argparser.parse_args()

logger.info(f"Subject ID: {args.sub_id}")

dicom_dir = os.path.join(args.out_dir, "dicom", args.sub_id)  # temporary directory to store DICOM files
cvi42_contours_dir = os.path.join(args.out_dir, "contour", args.sub_id)  # temporary directory to CVI42 store files
nii_dir = os.path.join(args.out_dir, "nii", args.sub_id)

if os.path.exists(dicom_dir):
    os.system("rm -rf {0}".format(dicom_dir))
if os.path.exists(cvi42_contours_dir):
    os.system("rm -rf {0}".format(cvi42_contours_dir))
if args.overwrite and os.path.exists(nii_dir):
    os.system("rm -rf {0}".format(nii_dir))
os.makedirs(dicom_dir, exist_ok=True)
os.makedirs(cvi42_contours_dir, exist_ok=True)
os.makedirs(nii_dir, exist_ok=True)

zip_files = []
if args.long_axis:
    if len(glob.glob(args.long_axis + args.sub_id + "_*.zip")) == 0:
        logger.error(f"No long axis zip found for subject {args.sub_id}")
        sys.exit(1)
    if check_existing_file(["la_2ch.nii.gz", "la_3ch.nii.gz", "la_4ch.nii.gz"], nii_dir):
        logger.warning(f"All long axis files in {nii_dir} already exists. Use --overwrite to overwrite existing files")
    else:
        zip_files = zip_files + glob.glob(args.long_axis + args.sub_id + "_*.zip")
if args.short_axis:
    if len(glob.glob(args.short_axis + args.sub_id + "_*.zip")) == 0:
        logger.error(f"No short axis zip found for subject {args.sub_id}")
        sys.exit(1)
    if check_existing_file(["sa.nii.gz"], nii_dir):
        logger.warning(f"All short axis files in {nii_dir} already exists. Use --overwrite to overwrite existing files")
    else:
        zip_files = zip_files + glob.glob(args.short_axis + args.sub_id + "_*.zip")
if args.aortic:
    if len(glob.glob(args.aortic + args.sub_id + "_*.zip")) == 0:
        logger.error(f"No aortic zip found for subject {args.sub_id}")
        sys.exit(1)
    if check_existing_file(["aor.nii.gz"], nii_dir):
        logger.warning(f"All aortic files in {nii_dir} already exists. Use --overwrite to overwrite existing files")
    else:
        zip_files = zip_files + glob.glob(args.aortic + args.sub_id + "_*.zip")
if args.tag:
    if len(glob.glob(args.tag + args.sub_id + "_*.zip")) == 0:
        logger.error(f"No tagging zip found for subject {args.sub_id}")
        sys.exit(1)
    zip_files = zip_files + glob.glob(args.tag + args.sub_id + "_*.zip")
    # todo check existing file
if args.lvot:
    if len(glob.glob(args.lvot + args.sub_id + "_*.zip")) == 0:
        logger.error(f"No LVOT zip found for subject {args.sub_id}")
        sys.exit(1)
    zip_files = zip_files + glob.glob(args.lvot + args.sub_id + "_*.zip")
    # todo check existing file
if args.blood_flow:
    if len(glob.glob(args.blood_flow + args.sub_id + "_*.zip")) == 0:
        logger.error(f"No blood flow zip found for subject {args.sub_id}")
        sys.exit(1)
    zip_files = zip_files + glob.glob(args.blood_flow + args.sub_id + "_*.zip")
    # todo check existing file
if args.T1:
    if len(glob.glob(args.T1 + args.sub_id + "_*.zip")) == 0:
        logger.error(f"No T1 zip found for subject {args.sub_id}")
        sys.exit(1)
    zip_files = zip_files + glob.glob(args.T1 + args.sub_id + "_*.zip")
    # todo check existing file

if len(zip_files) == 0:
    logger.warning(f"No zip files needed for subject {args.sub_id}. The subject is skipped.")
    sys.exit(1)
else:
    logger.info(f"Found {len(zip_files)} zip files for subject {args.sub_id}")

for file_i in np.arange(len(zip_files)):
    os.system("unzip -o " + zip_files[file_i] + " -d " + dicom_dir)
    # All original zip_files has a typo in the extension, so we need to rename them
    if os.path.exists(os.path.join(dicom_dir, "manifest.cvs")):
        os.system("cp " + dicom_dir + "manifest.cvs " + dicom_dir + "manifest.csv")
    process_manifest(
        os.path.join(dicom_dir, "manifest.csv"),
        os.path.join(dicom_dir, "manifest2.csv"),
    )  # remove comma
    df2 = pd.read_csv(os.path.join(dicom_dir, "manifest2.csv"))
    pid = df2.at[0, "patientid"]
    date = dateutil.parser.parse(df2.at[0, "date"][:11]).date().isoformat()
    for series_name, series_df in df2.groupby("series discription"):
        series_dir = os.path.join(dicom_dir, series_name)
        if not os.path.exists(series_dir):
            os.mkdir(series_dir)
        series_files = [os.path.join(dicom_dir, x) for x in series_df["filename"]]
        os.system("mv {0} {1}".format(" ".join(series_files), series_dir))

cvi42_contours_file = os.path.join(contour_gt_dir, f"{args.sub_id}.cvi42wsx")
if os.path.exists(cvi42_contours_file):
    print(f"Found cvi42 contours for subject {args.sub_id}")
    parse_cvi42_xml.parseFile(cvi42_contours_file, cvi42_contours_dir)

# It will automatically determines the modality according to manifest.csv
dset = Biobank_Dataset(dicom_dir, cvi42_contours_dir)
dset.read_dicom_images()
dset.convert_dicom_to_nifti(nii_dir)

# clean up the temporary directories
shutil.rmtree(dicom_dir)
shutil.rmtree(cvi42_contours_dir)

logger.info(f"Nii files for subject {args.sub_id} has been stored in {nii_dir}")
