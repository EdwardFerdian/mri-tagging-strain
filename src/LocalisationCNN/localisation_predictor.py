"""
Predictor for localization CNN
Author: Edward Ferdian
Date:   01/06/2018
"""
import tensorflow as tf
import h5py
import numpy as np

import packages.LocalisationCNN as cnn
import packages.InputPipelineHandler as ip
import packages.utils as utils


def load_data(input_filepath):
    with h5py.File(input_filepath, mode = 'r' ) as hdf5:
        data_nr = len(hdf5['image_seqs'])

    indexes = np.arange(data_nr)
    filenames = [input_filepath] * len(indexes)
    print("Dataset: {} rows".format(len(indexes)))
    return filenames, indexes

if __name__ == "__main__":
    base_path = "[INSERT DIRECTORY HERE]"    
    test_set = "{}/test.h5".format(base_path)
    batch_size = 50
    
    model_dir = "[INSERT MODEL DIRECTORY HERE]"
    model_name = "LocalCNN"

    # prepare log file
    utils.prepare_logfile("localisation_predictor")

    # ----------------- TensorFlow stuff -------------------
    # Reset all the tensor variables
    tf.reset_default_graph()  # We need to do this here before creating any tensor -> Yep, Dataset is a tensor object

    # Initialize dataset
    ds = ip.InputPipelineHandler(batch_size)
    
    # Prepare data iterator
    test_files, test_indexes = load_data(test_set)
    test_iterator = ds.initialize_dataset(test_files, test_indexes, training=False)
    
    # Initialize the network
    network = cnn.LocalisationCNN()
    network.initialize()
    network.restore_model(model_dir, model_name)
    network.predict(test_iterator)
    
    print("Done")
