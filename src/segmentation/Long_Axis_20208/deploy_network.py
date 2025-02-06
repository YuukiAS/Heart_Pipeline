# Copyright 2017, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by licable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import argparse
import time
import math
import numpy as np
import nibabel as nib
import tensorflow as tf
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from utils.image_utils import rescale_intensity


""" Deployment parameters """
parser = argparse.ArgumentParser()

parser.add_argument("--seq_name", type=str, default="sa", choices=["sa", "la_2ch", "la_4ch"], help="Sequence name.")
parser.add_argument(
    "--data_dir",
    type=str,
    help="Path to the data set directory, under which images " "are organised in subdirectories for each subject.",
)
parser.add_argument("--model_path", type=str, default="", help="Path to the saved trained model.")
parser.add_argument("--process_seq", type=int, default=1, help="Process a time sequence of images.")
parser.add_argument("--save_seg", type=int, default=1, help="Save segmentation.")
parser.add_argument(
    "--seg4",
    type=int,
    default=0,
    help="Segment all the 4 chambers in long-axis 4 chamber view. "
    "This seg4 network is trained using 200 subjects from location 18545."
    "By default, for all the other tasks (ventricular segmentation"
    "on short-axis images and atrial segmentation on long-axis images,"
    "the networks are trained using 3,975 subjects from location 2964.",
)

args = parser.parse_args()

if __name__ == "__main__":
    gd = tf.compat.v1.MetaGraphDef()
    with open("{0}.meta".format(args.model_path), "rb") as f:
        gd.ParseFromString(f.read())
    for node in gd.graph_def.node:
        if "_output_shapes" in node.attr:
            del node.attr["_output_shapes"]

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # * Import the trained model
        # saver = tf.compat.v1.train.import_meta_graph('{0}.meta'.format(args.model_path))
        saver = tf.compat.v1.train.import_meta_graph(gd)
        saver.restore(sess, "{0}".format(args.model_path))

        print("Start deployment on the data set using model {0}".format(args.model_path.split("/")[-1]))
        start_time = time.time()

        data_dir = os.path.join(args.data_dir)

        if args.seq_name == "la_4ch" and args.seg4:
            seg_name = "{0}/seg4_{1}.nii.gz".format(data_dir, args.seq_name)
        else:
            seg_name = "{0}/seg_{1}.nii.gz".format(data_dir, args.seq_name)
        if os.path.exists(seg_name):
            print("Segmentation already exists, skip")
            sys.exit(0)

        if args.process_seq:  # Process the temporal sequence
            image_name = "{0}/{1}.nii.gz".format(data_dir, args.seq_name)

            if not os.path.exists(image_name):
                print(
                    "Warning: Directory {0} does not contain an image with file " "name {1}. Skip.".format(
                        data_dir, os.path.basename(image_name)
                    )
                )
                sys.exit(1)

            # Read the image
            print("Reading {} ...".format(image_name))
            nim = nib.load(image_name)
            image = nim.get_fdata()
            X, Y, Z, T = image.shape
            orig_image = image

            print("Segmenting full sequence ...")

            # Intensity rescaling
            image = rescale_intensity(image, (1, 99))

            # Prediction (segmentation)
            pred = np.zeros(image.shape)

            # Pad the image size to be a factor of 16 so that the
            # downsample and upsample procedures in the network will
            # result in the same image size at each resolution level.
            X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
            x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
            x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
            image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0), (0, 0)), "constant")

            # Process each time frame
            for t in range(T):
                # Transpose the shape to NXYC
                image_fr = image[:, :, :, t]
                image_fr = np.transpose(image_fr, axes=(2, 0, 1)).astype(np.float32)
                image_fr = np.expand_dims(image_fr, axis=-1)

                # Evaluate the network
                prob_fr, pred_fr = sess.run(["prob:0", "pred:0"], feed_dict={"image:0": image_fr, "training:0": False})

                # Transpose and crop segmentation to recover the original size
                pred_fr = np.transpose(pred_fr, axes=(1, 2, 0))
                pred_fr = pred_fr[x_pre : x_pre + X, y_pre : y_pre + Y]
                pred[:, :, :, t] = pred_fr

            # * ED frame defaults to be the first time frame.
            # * Determine ES frame according to the minimum LV volume.
            k = {}
            k["ED"] = 0
            if args.seq_name == "sa" or (args.seq_name == "la_4ch" and args.seg4):
                k["ES"] = np.argmin(np.sum(pred == 1, axis=(0, 1, 2)))
            else:
                k["ES"] = np.argmax(np.sum(pred == 1, axis=(0, 1, 2)))
            print("  ED frame = {:d}, ES frame = {:d}".format(k["ED"], k["ES"]))

            # Save the segmentation
            if args.save_seg:
                print("  Saving segmentation ...")
                nim2 = nib.Nifti1Image(pred, nim.affine)
                nim2.header["pixdim"] = nim.header["pixdim"]
                if args.seq_name == "la_4ch" and args.seg4:
                    seg_name = "{0}/seg4_{1}.nii.gz".format(data_dir, args.seq_name)
                else:
                    seg_name = "{0}/seg_{1}.nii.gz".format(data_dir, args.seq_name)
                nib.save(nim2, seg_name)

                for fr in ["ED", "ES"]:
                    nib.save(
                        nib.Nifti1Image(orig_image[:, :, :, k[fr]], nim.affine),
                        "{0}/{1}_{2}.nii.gz".format(data_dir, args.seq_name, fr),
                    )
                    if args.seq_name == "la_4ch" and args.seg4:
                        seg_name = "{0}/seg4_{1}_{2}.nii.gz".format(data_dir, args.seq_name, fr)
                    else:
                        seg_name = "{0}/seg_{1}_{2}.nii.gz".format(data_dir, args.seq_name, fr)
                    nib.save(nib.Nifti1Image(pred[:, :, :, k[fr]], nim.affine), seg_name)
        else:  # Process only ED and ES time frames
            image_ED_name = "{0}/{1}_{2}.nii.gz".format(data_dir, args.seq_name, "ED")
            image_ES_name = "{0}/{1}_{2}.nii.gz".format(data_dir, args.seq_name, "ES")
            if not os.path.exists(image_ED_name) or not os.path.exists(image_ES_name):
                print(
                    "Warning: Directory {0} does not contain an image with " "file name {1} or {2}. Skip.".format(
                        data_dir, os.path.basename(image_ED_name), os.path.basename(image_ES_name)
                    )
                )
                sys.exit(1)

            measure = {}
            for fr in ["ED", "ES"]:
                image_name = "{0}/{1}_{2}.nii.gz".format(data_dir, args.seq_name, fr)

                # Read the image
                print("  Reading {} ...".format(image_name))
                nim = nib.load(image_name)
                image = nim.get_fdata()
                X, Y = image.shape[:2]
                if image.ndim == 2:
                    image = np.expand_dims(image, axis=2)

                print("  Segmenting {} frame ...".format(fr))

                # Intensity rescaling
                image = rescale_intensity(image, (1, 99))

                # Pad the image size to be a factor of 16 so that
                # the downsample and upsample procedures in the network
                # will result in the same image size at each resolution
                # level.
                X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
                x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
                x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
                image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0)), "constant")

                # Transpose the shape to NXYC
                image = np.transpose(image, axes=(2, 0, 1)).astype(np.float32)
                image = np.expand_dims(image, axis=-1)

                # Evaluate the network
                prob, pred = sess.run(["prob:0", "pred:0"], feed_dict={"image:0": image, "training:0": False})

                # Transpose and crop the segmentation to recover the original size
                pred = np.transpose(pred, axes=(1, 2, 0))
                pred = pred[x_pre : x_pre + X, y_pre : y_pre + Y]

                # Save the segmentation
                if args.save_seg:
                    print("  Saving segmentation ...")
                    nim2 = nib.Nifti1Image(pred, nim.affine)
                    nim2.header["pixdim"] = nim.header["pixdim"]
                    if args.seq_name == "la_4ch" and args.seg4:
                        seg_name = "{0}/seg4_{1}_{2}.nii.gz".format(data_dir, args.seq_name, fr)
                    else:
                        seg_name = "{0}/seg_{1}_{2}.nii.gz".format(data_dir, args.seq_name, fr)
                    nib.save(nim2, seg_name)
