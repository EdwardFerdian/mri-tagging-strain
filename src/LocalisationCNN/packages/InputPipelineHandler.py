"""
InputPipelineHandler
Author: Edward Ferdian
Date:   01/06/2018
"""
import tensorflow as tf
import h5py
import numpy as np
from random import randint
import scipy.ndimage as ndimage
import math

class InputPipelineHandler:
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
            ds = ds.shuffle(1000)

        # Run the mapping functions on each data
        ds = ds.map(self.training_data_load_wrapper, num_parallel_calls=4)

        if training:
            ds = ds.map(self.augmentation_wrapper)
            # TODO: add more augmentation here (e.g. flip, translation, normalize, contrast, zoom, etc)

        ds = ds.map(self._resize_function)

        ds = ds.batch(batch_size=self.batch_size).prefetch(10)

        # Do not use repeat with initializable operator. Use 'while True' to exhaust the list
        return ds.make_initializable_iterator()

    # --- Wrapper functions for the tf.dataset input pipeline ---
    def training_data_load_wrapper(self, fpath, idx):
        return tf.py_func(func=self.load_hd5_img_and_bbox_from_sequence, 
            inp=[fpath, idx], Tout=[tf.float32, tf.float32])

    def augmentation_wrapper(self, img, coords):
        return tf.py_func(func=self._rotate_img_and_bbox, 
            inp=[img, coords], Tout=[tf.float32, tf.float32])

    # --- end of wrapper functions ---

    def load_hd5_img_and_bbox_from_sequence(self, fpath, idx):
        '''
            Load an image from a pair of filename-index.
            Only the ED frame is taken from the image sequence.

        '''
        hd5path = fpath
        with h5py.File(hd5path, 'r') as hl:
            # [n, time=0, 256, 256]
            ed_img = np.asarray(hl.get('image_seqs')[idx,0,:,:]) # we only need the first frame
            # [n, 4]
            bbox = np.asarray(hl.get('bbox_corners')[idx])
        # We need to cast this to the proper data type like below
        # Default type is double/float64
        return ed_img.astype('float32'), bbox.astype('float32')

    def _resize_function(self, img, coords):
        img = tf.reshape(img, [256, 256])
        coords = tf.reshape(coords, [4])
        return img, coords

    def _rotate_img_and_bbox(self, img, bbox):
        """
            Rotate the image and bounding box.
            Currently it only does 90, 180, and 270 degrees.
        """
        rnd = randint(0,7) 
        if (rnd == 0 or rnd > 3):
            # we give higher chance the image is not being rotated
            return img, bbox
        else:
            # we do only rotation for 90, 180, and 270 for now
            angle = rnd * 90
            new_img = ndimage.rotate(img, angle, reshape=False)
            new_bbox = self.rotate_coords_bbox(img.shape, bbox, -angle)

            return new_img, new_bbox.astype('float32')

    def rotate_coords_bbox(self, img_shape, coords, angle):
        center_point = np.asarray(img_shape)/2

        point = [coords[0], coords[1]]
        x1 = self.rotate_single_point(point, angle, center_point)
        
        point = [coords[2], coords[3]]
        x2 = self.rotate_single_point(point, angle, center_point)

        min_x = min(x1[0], x2[0])
        min_y = min(x1[1], x2[1])

        max_x = max(x1[0], x2[0])
        max_y = max(x1[1], x2[1])

        new_coords = np.asarray([min_x, min_y, max_x, max_y])
        return new_coords

    def rotate_single_point(self, point,angle, centerPoint):
        """Rotates a point around another centerPoint. Angle is in degrees.
        Rotation is counter-clockwise"""
        angle = math.radians(angle)
        temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
        temp_point = ( temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle) , temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
        temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
        return temp_point




