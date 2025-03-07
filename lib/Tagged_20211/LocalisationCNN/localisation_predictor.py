"""
Predictor for localization CNN
Author: Edward Ferdian
Date:   01/06/2018
"""

import tensorflow as tf
import h5py
import numpy as np

import LocalisationCNN as cnn
import InputPipelineHandler as ip

assert tf.__version__ >= "2.0", "Tensorflow version must be at least 2.0"


def load_data(input_filepath):
    with h5py.File(input_filepath, mode="r") as hdf5:
        data_nr = len(hdf5["image_seqs"])

    indexes = np.arange(data_nr)
    filenames = [input_filepath] * len(indexes)
    print("Dataset: {} rows".format(len(indexes)))
    return filenames, indexes


if __name__ == "__main__":
    base_path = "../../data"
    test_set = "{}/train_example.h5".format(base_path)
    batch_size = 50

    model_dir = "../../../../../model/Tagged_20212/LocalCNN"
    # model_dir = "/work/users/y/u/yuukias/Heart_Pipeline/model/Tagged_20212/LocalCNN"
    model_name = "LocalCNN"  # only for printing

    # ----------------- TensorFlow stuff -------------------
    # Reset all the tensor variables
    tf.compat.v1.reset_default_graph()

    # Initialize dataset
    ds = ip.InputPipelineHandler(batch_size)

    # Prepare data iterator
    test_files, test_indexes = load_data(test_set)
    test_iterator = ds.initialize_dataset(test_files, test_indexes, training=False)

    # Initialize the network
    network = cnn.LocalisationCNN()
    network.restore_model(model_dir, model_name)
    network.predict(test_iterator)

    print("Done")
