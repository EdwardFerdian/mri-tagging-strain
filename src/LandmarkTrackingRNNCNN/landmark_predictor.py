"""
Landmark predictor
Author: Edward Ferdian
Date:   01/06/2018
"""
import tensorflow as tf
import h5py
import numpy as np

import LandmarkTrackingNetwork as rnncnn
import CroppedInputPipelineHandler as cp

batch_size = 30

def load_data(input_filepath):
    with h5py.File(input_filepath, mode = 'r' ) as hdf5:
        data_nr = len(hdf5['cropped_image_seqs'])

    indexes = np.arange(data_nr)
    filenames = [input_filepath] * len(indexes)
    print("Dataset: {} rows".format(len(indexes)))
    return filenames, indexes

if __name__ == "__main__":
    base_path = "../../data"
    test_set = "{}/train_example.h5".format(base_path)
    
    
    model_dir = "../models/LandmarkTrackingRNNCNN_[INSERT_TEXT_HERE]"
    model_name = "LandmarkTrackingRNNCNN"

    # Reset all the tensor variables
    tf.reset_default_graph() 

    # Initialize dataset
    ds = cp.CroppedInputPipelineHandler(batch_size)
    
    # Prepare data iterator
    test_files, test_indexes = load_data(test_set)
    test_iterator = ds.initialize_dataset(test_files, test_indexes, training=False)
    
    # Initialize the network
    network = rnncnn.LandmarkTrackingNetwork(batch_size)
    network.restore_model(model_dir, model_name)
    network.predict(test_iterator)
    
    print("Done")
