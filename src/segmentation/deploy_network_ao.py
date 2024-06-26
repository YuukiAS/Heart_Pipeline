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
# ==============================================================================
import os
import argparse
import time
import math
import numpy as np
import nibabel as nib
import tensorflow as tf
from common.image_utils import *


""" Deployment parameters """
# args = tf.compat.v1.args.args
# tf.compat.v1.args.DEFINE_integer('time_step', 1,
#                             'Time step during deployment of LSTM.')
# tf.compat.v1.args.DEFINE_enum('seq_name', 'ao', ['ao'],
#                          'Sequence name.')
# tf.compat.v1.args.DEFINE_enum('model', 'UNet-LSTM', ['UNet', 'UNet-LSTM', 'Temporal-UNet'],
#                          'Model name.')
# tf.compat.v1.args.DEFINE_string('data_dir',
#                            '/vol/medic02/users/wbai/data/cardiac_atlas/Biobank_ao/validation',
#                            'Path to the test set directory, under which images '
#                            'are organised in subdirectories for each subject.')
# tf.compat.v1.args.DEFINE_string('model_path',
#                            '/vol/biomedic2/wbai/ukbb_cardiac/UKBB_18545/model/UNet-LSTM_ao_level5_filter16_22222_batch1_iter20000_lr0.001_zscore_tw9_h16_bidir_seq2seq_wR5_wr0.1_joint/UNet-LSTM_ao_level5_filter16_22222_batch1_iter20000_lr0.001_zscore_tw9_h16_bidir_seq2seq_wR5_wr0.1_joint.ckpt-20000',
#                            'Path to the saved trained model.')
# tf.compat.v1.args.DEFINE_boolean('process_seq', True,
#                             'Process a time sequence of images.')
# tf.compat.v1.args.DEFINE_boolean('save_seg', True,
#                             'Save segmentation.')
# tf.compat.v1.args.DEFINE_boolean('z_score', True,
#                             'Normalise the image intensity to z-score. '
#                             'Otherwise, rescale the intensity.')
# tf.compat.v1.args.DEFINE_integer('weight_R', 5,
#                             'Radius of the weighting window.')
# tf.compat.v1.args.DEFINE_float('weight_r', 0.1,
#                           'Power of weight for the seq2seq loss. 0: uniform; 1: linear; 2: square.')

parser = argparse.ArgumentParser()

parser.add_argument('--time_step', type=int, default=1,
                    help='Time step during deployment of LSTM.')
parser.add_argument('--seq_name', type=str, default='ao',
                    choices=['ao'],
                    help='Sequence name.')
parser.add_argument('--model', type=str, default='UNet-LSTM',
                    choices=['UNet', 'UNet-LSTM', 'Temporal-UNet'],
                    help='Model name.')
parser.add_argument('--data_dir', type=str,
                    default='/vol/medic02/users/wbai/data/cardiac_atlas/Biobank_ao/validation',
                    help='Path to the test set directory, under which images '
                         'are organised in subdirectories for each subject.')
parser.add_argument('--model_path', type=str,
                    default='/vol/biomedic2/wbai/ukbb_cardiac/UKBB_18545/model/UNet-LSTM_ao_level5_filter16_22222_batch1_iter20000_lr0.001_zscore_tw9_h16_bidir_seq2seq_wR5_wr0.1_joint/UNet-LSTM_ao_level5_filter16_22222_batch1_iter20000_lr0.001_zscore_tw9_h16_bidir_seq2seq_wR5_wr0.1_joint.ckpt-20000',
                    help='Path to the saved trained model.')
parser.add_argument('--process_seq', type=int, default=1,
                    help='Process a time sequence of images.')
parser.add_argument('--save_seg', type=int, default=1,
                    help='Save segmentation.')
parser.add_argument('--z_score', type=int, default=1,
                    help='Normalise the image intensity to z-score. '
                         'Otherwise, rescale the intensity.')
parser.add_argument('--weight_R', type=int, default=5,
                    help='Radius of the weighting window.')
parser.add_argument('--weight_r', type=float, default=0.1,
                    help='Power of weight for the seq2seq loss. 0: uniform; 1: linear; 2: square.')

args = parser.parse_args()

if __name__ == '__main__':
    gd = tf.compat.v1.MetaGraphDef()
    with open('{0}.meta'.format(args.model_path), "rb") as f:
        gd.ParseFromString(f.read())
    for node in gd.graph_def.node:
        if '_output_shapes' in node.attr:
            del node.attr['_output_shapes']

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # * Import the trained model
        # saver = tf.compat.v1.train.import_meta_graph('{0}.meta'.format(args.model_path))
        saver = tf.compat.v1.train.import_meta_graph(gd)
        saver.restore(sess, '{0}'.format(args.model_path))

        print('Start deployment on the data set using model {0}'.format(args.model))

        # Process each subject subdirectory
        data_list = sorted(os.listdir(args.data_dir))
        processed_list = []
        table = []
        for data in data_list:
            if data == "out":
                continue
            data_dir = os.path.join(args.data_dir, data)

            if args.process_seq:
                # Process the temporal sequence
                image_name = '{0}/{1}.nii.gz'.format(data_dir, args.seq_name)

                if not os.path.exists(image_name):
                    print('  Directory {0} does not contain an image with file name {1}. '
                          'Skip.'.format(data_dir, os.path.basename(image_name)))
                    continue

                # Read the image
                print('  Reading {} ...'.format(image_name))
                nim = nib.load(image_name)
                dx, dy, dz, dt = nim.header['pixdim'][1:5]
                area_per_pixel = dx * dy
                image = nim.get_fdata()
                X, Y, Z, T = image.shape
                orig_image = image

                print('  Segmenting full sequence ...')

                # Intensity normalisation
                if args.z_score:
                    image = normalise_intensity(image, 10.0)
                else:
                    image = rescale_intensity(image, (1.0, 99.0))

                # Probability (segmentation)
                n_class = 3
                prob = np.zeros((X, Y, Z, T, n_class), dtype=np.float32)

                # Pad the image size to be a factor of 16 so that the downsample and upsample procedures
                # in the network will result in the same image size at each resolution level.
                # X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
                X2, Y2 = 256, 256
                x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
                x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
                image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0), (0, 0)), 'constant')

                # Process each time frame
                if args.model == 'UNet':
                    # For each time frame
                    for t in range(T):
                        # Transpose the shape to NXYC
                        image_fr = image[:, :, :, t]
                        image_fr = np.transpose(image_fr, axes=(2, 0, 1)).astype(np.float32)
                        image_fr = np.expand_dims(image_fr, axis=-1)

                        # Evaluate the network
                        # prob_fr: NXYC
                        prob_fr = sess.run('prob:0',
                                           feed_dict={'image:0': image_fr, 'training:0': False})

                        # Transpose and crop to recover the original size
                        # prob_fr: XYNC
                        prob_fr = np.transpose(prob_fr, axes=(1, 2, 0, 3))
                        prob_fr = prob_fr[x_pre:x_pre + X, y_pre:y_pre + Y]
                        prob[:, :, :, t, :] = prob_fr
                elif args.model == 'UNet-LSTM' or args.model == 'Temporal-UNet':
                    time_window = args.weight_R * 2 - 1
                    rad = int((time_window - 1) / 2)
                    weight = np.zeros((1, 1, 1, T, 1))

                    w = []
                    for t in range(time_window):
                        d = abs(t - rad)
                        if d <= args.weight_R:
                            w_t = pow(1 - float(d) / args.weight_R, args.weight_r)
                        else:
                            w_t = 0
                        w += [w_t]

                    w = np.array(w)
                    w = np.reshape(w, (1, 1, 1, time_window, 1))

                    # For each time frame after a time_step
                    for t in range(0, T, args.time_step):
                        # Get the frames in the time window
                        t1 = t - rad
                        t2 = t + rad
                        idx = []
                        for i in range(t1, t2 + 1):
                            if i < 0:
                                idx += [i + T]
                            elif i >= T:
                                idx += [i - T]
                            else:
                                idx += [i]

                        # image_idx: NTXYC
                        image_idx = image[:, :, :, idx]
                        image_idx = np.transpose(image_idx, axes=(2, 3, 0, 1)).astype(np.float32)
                        image_idx = np.expand_dims(image_idx, axis=-1)

                        # Evaluate the network
                        # Curious: can we deploy the LSTM model more efficiently by utilising the state variable?
                        # Currently, we have to feed all the time frames in the time window and we can not just
                        # feed one time frame, because the LSTM is an unrolled model in the dataflow graph.
                        # It needs all the input from the time window.
                        # prob_idx: NTXYC
                        prob_idx = sess.run('prob:0',
                                            feed_dict={'image:0': image_idx, 'training:0': False})

                        # Transpose and crop the segmentation to recover the original size
                        # prob_idx: XYNTC
                        prob_idx = np.transpose(prob_idx, axes=(2, 3, 0, 1, 4))

                        # Tile the overling probability maps
                        prob[:, :, :, idx] += prob_idx[x_pre:x_pre + X, y_pre:y_pre + Y] * w
                        weight[:, :, :, idx] += w

                    # Average probability
                    prob /= weight
                else:
                    print('Error: unknown model {0}.'.format(args.model))
                    exit(0)

                # Segmentation
                pred = np.argmax(prob, axis=-1).astype(np.int32)

                # Save the segmentation
                if args.save_seg:
                    print('  Saving segmentation ...')
                    nim2 = nib.Nifti1Image(pred, nim.affine)
                    nim2.header['pixdim'] = nim.header['pixdim']
                    nib.save(nim2, '{0}/seg_{1}.nii.gz'.format(data_dir, args.seq_name))

                processed_list += [data]
            else:
                if args.model == 'UNet-LSTM':
                    print('UNet-LSTM does not support frame-wise segmentation. '
                          'Please use the -process_seq flag.')
                    exit(0)

                # Process ED and ES time frames
                image_ED_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, args.seq_name, 'ED')
                image_ES_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, args.seq_name, 'ES')
                if not os.path.exists(image_ED_name) or not os.path.exists(image_ES_name):
                    print('  Directory {0} does not contain an image with file name {1} or {2}. '
                          'Skip.'.format(data_dir, os.path.basename(image_ED_name), os.path.basename(image_ES_name)))
                    continue

                measure = {}
                for fr in ['ED', 'ES']:
                    image_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, args.seq_name, fr)

                    # Read the image
                    # image: XYZ
                    print('  Reading {} ...'.format(image_name))
                    nim = nib.load(image_name)
                    dx, dy, dz, dt = nim.header['pixdim'][1:5]
                    area_per_pixel = dx * dy
                    image = nim.get_fdata()
                    X, Y = image.shape[:2]

                    print('  Segmenting {} frame ...'.format(fr))

                    # Intensity normalisation
                    if args.z_score:
                        image = normalise_intensity(image, 10.0)
                    else:
                        image = rescale_intensity(image, (1.0, 99.0))

                    # Pad the image size to be a factor of 16 so that the downsample and upsample procedures
                    # in the network will result in the same image size at each resolution level.
                    X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
                    x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
                    x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
                    image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0)), 'constant')

                    # Transpose the shape to NXYC
                    # image: NXY
                    image = np.transpose(image, axes=(2, 0, 1)).astype(np.float32)
                    # image: NXYC
                    image = np.expand_dims(image, axis=-1)

                    # Evaluate the network
                    # pred: NXY
                    prob, pred = sess.run(['prob:0', 'pred:0'],
                                          feed_dict={'image:0': image, 'training:0': False})

                    # Transpose and crop the segmentation to recover the original size
                    pred = np.transpose(pred, axes=(1, 2, 0))
                    pred = pred[x_pre:x_pre + X, y_pre:y_pre + Y]

                    # Save the segmentation
                    if args.save_seg:
                        print('  Saving segmentation ...')
                        nim2 = nib.Nifti1Image(pred, nim.affine)
                        nim2.header['pixdim'] = nim.header['pixdim']
                        nib.save(nim2, '{0}/seg_{1}_{2}.nii.gz'.format(data_dir, args.seq_name, fr))

                processed_list += [data]
