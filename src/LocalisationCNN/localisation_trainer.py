"""
Localisation CNN trainer
Author: Edward Ferdian
Date:   01/06/2018
"""
import tensorflow as tf
import h5py
import numpy as np

import LocalisationCNN as cnn
import InputPipelineHandler as ip
import utils

# Hyperparameters optimisation variables
initial_learning_rate = 1e-3
epochs = 100
batch_size = 50
training_keep_prob = 0.8


def load_data(input_filepath):
    with h5py.File(input_filepath, mode = 'r' ) as hdf5:
        data_nr = len(hdf5['image_seqs'])

    indexes = np.arange(data_nr)
    filenames = [input_filepath] * len(indexes)
    print("Dataset: {} rows".format(len(indexes)))
    return filenames, indexes

if __name__ == "__main__":
    base_path = "../../data"
    
    training_set = "{}/train_example.h5".format(base_path)
    validation_set = "{}/train_example.h5".format(base_path) # this is just an example

    # prepare log file
    utils.prepare_logfile("localisation_trainer")

    # ----------------- TensorFlow stuff -------------------
    # Reset all the tensor variables
    tf.reset_default_graph()  # We need to do this here before creating any tensor -> Yep, Dataset is a tensor object

    # Initialize dataset
    ds = ip.InputPipelineHandler(batch_size)
    
    # Prepare training data iterator
    train_files, train_indexes = load_data(training_set)
    train_iterator = ds.initialize_dataset(train_files, train_indexes, training=True)
    
    # Prepare validation data iterator
    validation_files, validation_indexes = load_data(validation_set)
    val_iterator = ds.initialize_dataset(validation_files, validation_indexes, training=False)

    # Initialize the network
    network = cnn.LocalisationCNN(initial_learning_rate, training_keep_prob)
    network.init_model_dir()
    network.train_network(train_iterator, val_iterator, epochs)
