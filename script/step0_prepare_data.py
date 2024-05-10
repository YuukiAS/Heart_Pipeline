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
"""
module load python/2.7.12;cd /proj/tengfei/pipeline/UKB_Heart;
export PATH=/proj/tengfei/pipeline/UKB_Heart/data:${PATH}
export PYTHONPATH=/proj/tengfei/pipeline/UKB_Heart/data:${PYTHONPATH}
python data/step0_dataPrepare.py /proj/tengfei/pipeline/UKB_Heart/downloaded/ 1002592
"""
import os
import glob
import pandas as pd
from biobank_utils import *
import dateutil.parser
import sys
import parse_cvi42_xml

datadir = sys.argv[1]  
ID = sys.argv[2]  
print(ID)
long_axis = sys.argv[3]
short_axis = sys.argv[4]
aortic = sys.argv[5]
files = glob.glob(long_axis + ID + "_*.zip") + glob.glob(short_axis + ID + "_*.zip") + glob.glob(aortic + ID + "_*.zip")
dicom_dir = datadir + "/dicom/" + ID + "/"  # this folder is temporary, use ID to allow parallel computing
cvi42_contours_dir = datadir + "/label/" + ID + "/"   # from return 2541
niidir = datadir + "/nii/" + ID + "/"
os.system("mkdir -p " + dicom_dir)
os.system("mkdir -p " + cvi42_contours_dir)
os.system("mkdir -p " + niidir)
for ii in np.arange(len(files)):
    os.system("unzip -o " + files[ii] + " -d " + dicom_dir)
    if os.path.exists(os.path.join(dicom_dir, "manifest.cvs")):
        os.system("cp " + dicom_dir + "manifest.cvs " + dicom_dir + "manifest.csv")  # All original files has a typo
    process_manifest(
        os.path.join(dicom_dir, "manifest.csv"),
        os.path.join(dicom_dir, "manifest2.csv"),
    )
    df2 = pd.read_csv(os.path.join(dicom_dir, "manifest2.csv"))
    pid = df2.at[0, "patientid"]
    date = dateutil.parser.parse(df2.at[0, "date"][:11]).date().isoformat()
    for series_name, series_df in df2.groupby("series discription"):
        series_dir = os.path.join(dicom_dir, series_name)
        if not os.path.exists(series_dir):
            os.mkdir(series_dir)
        series_files = [os.path.join(dicom_dir, x) for x in series_df["filename"]]
        os.system("mv {0} {1}".format(" ".join(series_files), series_dir))


label_dir = '/work/users/y/u/yuukias/database/UKBiobank/data_reference/label'
cvi42_contours_file = os.path.join(label_dir, '{0}.cvi42wsx'.format(ID))
if os.path.exists(cvi42_contours_file):
    print("Found cvi42 contours for subject {0}".format(ID))
    parse_cvi42_xml.parseFile(cvi42_contours_file, cvi42_contours_dir)


dset = Biobank_Dataset(dicom_dir, cvi42_contours_dir)
dset.read_dicom_images()
dset.convert_dicom_to_nifti(niidir)
os.system("rm -rf {0}".format(dicom_dir))
# os.system("rm -rf {0}".format(cvi42_contours_dir))
# os.system("rm -f " + datadir + ID + "_*.zip")
