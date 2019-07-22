"""
Input Pipeline Handler for Landmark tracking network
Author: Edward Ferdian
Date:   01/06/2018
"""
import tensorflow as tf
import h5py
import numpy as np
import math
import cv2

class CroppedInputPipelineHandler:
    # constructor
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def initialize_dataset(self, all_files, indexes, training=False):
        """
            Input pipeline.
            This function accepts a list of filenames with index to read.
            The _training_data_load_wrapper will read the filename-index pair and load the data.
        """
        ds = tf.data.Dataset.from_tensor_slices((all_files, indexes))

        if training:
            ds = ds.shuffle(5000)

        # Run the mapping functions on each data
        ds = ds.map(self.training_data_load_wrapper, num_parallel_calls=4)
        ds = ds.map(self._reshape_frames)

        ds = ds.batch(batch_size=self.batch_size).prefetch(10)
        
        return ds.make_initializable_iterator()

    def training_data_load_wrapper(self, data_path, idx):
        return tf.py_func(func=self.load_img_and_coord_sequence, 
            inp=[data_path, idx], Tout=[tf.float32, tf.float32])

    def load_img_and_coord_sequence(self, fpath, idx):
        """
            Load an image sequence and coords sequence for a certain filename-index pair
            output shape: [t, 256, 256], [t, 2, 168], [4]
            t is number of frames (t=20)
        """
        with h5py.File(fpath, 'r') as hl:
            image_sequence = np.asarray(hl.get('cropped_image_seqs')[idx])
            landmark_coords = np.asarray(hl.get('cropped_landmark_coords')[idx])
        
        return image_sequence.astype('float32'), landmark_coords.astype('float32')
        
    def _reshape_frames(self, imgs, coords):
        img_seq = tf.reshape(imgs, [20, 128, 128, 1])
        coords_seq = tf.reshape(coords, [20, 2 * 168])

        return img_seq, coords_seq



