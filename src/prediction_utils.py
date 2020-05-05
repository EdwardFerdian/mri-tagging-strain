import numpy as np
import time
import h5py
import os
import math
import cv2

def get_dataset_len(filepath):
    with h5py.File(filepath, 'r') as hl:
        data_size = hl['image_seqs'].shape[0]
    return data_size

def load_dataset(filepath, start_index, batch_size):
    with h5py.File(filepath, 'r') as hl:
        img_sequences = np.asarray(hl.get('image_seqs'))[start_index:start_index+batch_size]
    return img_sequences

def crop_image_bbox(img, bbox_corners):
    left_x = math.floor(bbox_corners[0])
    low_y  = math.floor(bbox_corners[1])

    right_x = math.ceil(bbox_corners[2])
    high_y  = math.ceil(bbox_corners[3])

    # special case where low_x or left_x is negative, we need to add some padding
    pad_x = 0
    pad_y = 0
    if (left_x < 0):
        pad_x = 0 - left_x
    if (low_y < 0):
        pad_y = 0 - low_y
    
    if (pad_x == 0 and pad_y == 0):
        return img[low_y:high_y, left_x:right_x ]
    else:
        print('Cropping image with extra padding due to negative start index')
        new_img = np.pad(img, ((pad_y,pad_y),(pad_x,pad_x)), 'constant')
        return new_img[pad_y+low_y:pad_y+high_y, pad_x+left_x:pad_x+right_x ]

def resize_image(img, new_size):
    new_img = cv2.resize(img, dsize=(new_size,new_size), interpolation=cv2.INTER_CUBIC)
    
    ratio_x = new_size / img.shape[0]
    # coords = np.array(coords)
    # # assumption x and y has the same ratio
    # new_coords = coords * ratio_x
    return new_img, ratio_x



def crop_and_resize_all_frames(img_sequences, corners, new_img_size=128):
    cropped_frames = np.zeros(shape=(img_sequences.shape[0], img_sequences.shape[1], new_img_size, new_img_size))
    resize_ratios = np.ones(shape=len(img_sequences))

    # for every image sequences
    for i in range(len(img_sequences)):
        bbox_corners = corners[i]

        # for every time frame, we crop it using the same bounding box
        for t in range(len(img_sequences[i])):    
            frame = img_sequences[i,t,:,:]

            new_img = crop_image_bbox(frame, bbox_corners)
            # Resample the image and keep the ratio
            new_img, ratio_x = resize_image(new_img, new_img_size)

            cropped_frames[i][t] = new_img
        
        # save the ratio, on index level, not on frame level
        resize_ratios[i] = ratio_x

    return cropped_frames, resize_ratios