import os
import sys
# import urllib.request as urllib
# import shutil
# import pandas as pd
# from common.cardiac_utils import *

# Deploy the segmentation network
print("Deploying the segmentation network ...")
data_path = sys.argv[1] + "/"  # /database/UKBiobank/data_reference/out/nii
ID = sys.argv[2]  # patient ID
bpressure = sys.argv[3]
data_path = data_path + ID + "/"
out = data_path + "out"
raw_path = data_path + "raw"

print("******************************")
print(f"Start Cardiac Analysis for {ID}")
print("******************************")

os.system("mkdir -p " + raw_path)
os.system("mv " + data_path + "/*.nii.gz " + raw_path + " 2>/dev/null")  # if no file to move, fail silently

# Status = os.path.exists("{0}/table_ventricular_volume.csv".format(out))
# Status = Status + os.path.exists("{0}/table_wall_thickness.csv".format(out))
# Status = Status + os.path.exists("{0}/table_atrial_volume.csv".format(out))
# Status = Status + os.path.exists("{0}/table_aortic_area.csv".format(out))
# print("Number of existing results:" + str(Status))
# if Status < 4:
#     print("Starting...")
# else:
#     print("Overwriting...")

# remove previous results
os.system("rm -fr {0}".format(out))  
os.system("rm -f {0}/seg_*.nii.gz".format(raw_path))
os.system("rm -f {0}/wall_thickness_ED.*".format(raw_path))

os.system("mkdir -p " + out)


print("-------------")
print("Segmentations")
print("-------------")

print("Short axis: Segmentations")
os.system(
    "python3 common/deploy_network.py --seq_name sa --data_dir {0} --model_path trained_model/FCN_sa".format(
        data_path
    )
)

print("Long axis: Segmentations")
os.system(
    "python3 common/deploy_network.py --seq_name la_2ch --data_dir {0} "
    "--model_path trained_model/FCN_la_2ch".format(data_path)
)
os.system(
    "python3 common/deploy_network.py --seq_name la_4ch --data_dir {0} "
    "--model_path trained_model/FCN_la_4ch".format(data_path)
)
os.system(
    "python3 common/deploy_network.py --seq_name la_4ch --data_dir {0} "
    "--seg4 1 --model_path trained_model/FCN_la_4ch_seg4".format(data_path)
)

print("Aorta: Segmentations")
os.system(
    "python3 common/deploy_network_ao.py --seq_name ao --data_dir {0} "
    "--model_path trained_model/UNet-LSTM_ao".format(data_path)
)

print("----------------------")
print("Extracting 2D features")
print("----------------------")

print("Short axis: Extracting 2D features")
os.system(
    "python3 short_axis/eval_ventricular_volume.py --data_dir {0} --output_csv {1}/table_ventricular_volume.csv".format(
        data_path, out
    )
)
os.system(
    "python3 short_axis/eval_wall_thickness.py --data_dir {0} --output_csv {1}/table_wall_thickness.csv".format(
        data_path, out
    )
)
print("Long axis: Extracting 2D features")
os.system(
    "python3 long_axis/eval_atrial_volume.py --data_dir {0} --output_csv {1}/table_atrial_volume.csv".format(
        data_path, out
    )
)
print("Aorta: Extracting 2D features")
os.system(
    f"python3 aortic/eval_aortic_area.py "
    f"--data_dir {data_path} --pressure_csv {bpressure}  --id {ID} --output_csv {out}/table_aortic_area.csv"
)


#####################################################################################
# 4D features.
#####################################################################################
# todo these two failed
# print("----------------------")
# print("Extracting 4D features")
# print("----------------------")
os.system('python3 short_axis/eval_strain_sax.py '
          f'--data_dir {data_path} --par_dir par --output_csv {out}/table_strain_sax.csv')
os.system('python3 long_axis/eval_strain_lax.py '
          f'--data_dir {data_path} --par_dir par --output_csv {out}/table_strain_lax.csv')
os.system('mv {0}/{1}/dice*.csv {2}/'.format(data_path,ID,out))

print('All Done')
