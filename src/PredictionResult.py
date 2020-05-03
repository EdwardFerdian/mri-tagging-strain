import numpy as np
import math
import os
import h5py

class PredictionResult: 
    def __init__(self, bbox_corners, landmark_sequences, resize_ratios):
        self.bbox_corners = bbox_corners
        self.landmark_sequences = landmark_sequences
        self.resize_ratios = resize_ratios
        
        # revert the position back
        self.uncropped_landmark_sequences = self.revert_coords_to_original()
        self.rr_strains = []
        self.cc_strains = []
        self.rr_linear_strains = []
        self.cc_linear_strains = []

    
    def save_predictions(self, output_path, input_file_pattern):
        new_filename = input_file_pattern
        if (input_file_pattern.endswith('.h5')):
            new_filename = input_file_pattern[:-3] # strip the .h5

        output_filename = '{}.result.h5'.format(new_filename)
        print('Saving prediction as {}'.format(output_filename))
        
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        with h5py.File(os.path.join(output_path, output_filename), 'w') as hf:
            hf.create_dataset("bbox_predictions", data=self.bbox_corners, dtype='float32')
            hf.create_dataset("landmark_local_predictions", data=self.landmark_sequences, dtype='float32')
            hf.create_dataset("landmark_predictions", data=self.uncropped_landmark_sequences, dtype='float32')
            hf.create_dataset("resize_ratios", data=self.resize_ratios, dtype='float32')
            hf.create_dataset("rr_strains", data=self.rr_strains, dtype='float32')
            hf.create_dataset("cc_strains", data=self.cc_strains, dtype='float32')
            hf.create_dataset("rr_linear_strains", data=self.rr_linear_strains, dtype='float32')
            hf.create_dataset("cc_linear_strains", data=self.cc_linear_strains, dtype='float32')
        # print('Prediction saved!')

    def revert_coords_to_original(self):
        # create a zero copy of the landmark sequences
        new_coords = np.zeros(shape=self.landmark_sequences.shape)

        # TODO: maybe we don't need the loop, because of numpy
        for idx in range(len(self.landmark_sequences)):
            # time_steps x 2 x 168
            landmarks = self.landmark_sequences[idx]
            bbox = self.bbox_corners[idx]
            ratio = self.resize_ratios[idx]

            left_x = math.floor(bbox[0])
            low_y  = math.floor(bbox[1])

            # rescale back
            new_coords[idx] = landmarks / ratio

            # add the top left corner
            new_coords[idx,:,0,:] = new_coords[idx,:,0,:] + left_x
            new_coords[idx,:,1,:] = new_coords[idx,:,1,:] + low_y
        # end of for loop
        
        # left_x = np.floor(self.bbox_corners[:,0])
        # low_y  = np.floor(self.bbox_corners[:,1])

        # new_coords = self.landmark_sequences / self.resize_ratios[:, np.newaxis]
        # # # add the top left corner
        # new_coords[:,:,0,:] = new_coords[:,:,0,:] + left_x
        # new_coords[:,:,1,:] = new_coords[:,:,1,:] + low_y

        
        return new_coords

    def calculate_strains(self):
        self.rr_strains = self.calculate_radial_strain(self.uncropped_landmark_sequences, use_linear_strain=False)
        self.rr_linear_strains = self.calculate_radial_strain(self.uncropped_landmark_sequences, use_linear_strain=True)

        ccs = []
        linear_ccs = []
        for wall_index in range(0,7):
            # cc strain the L^2 version
            cc_strains = self.calculate_circumferential_strain(self.uncropped_landmark_sequences, wall_index, use_linear_strain=False)
            ccs.append(cc_strains)

            # cc strain the linear version
            cc_linear_strains = self.calculate_circumferential_strain(self.uncropped_landmark_sequences, wall_index, use_linear_strain=True)
            linear_ccs.append(cc_linear_strains)
        
        stacked_ccs = np.stack(ccs, axis=2)
        # calculate the avg cc
        avg_cc = np.mean(stacked_ccs, axis=2)
        stacked_ccs = np.concatenate((stacked_ccs, avg_cc[...,np.newaxis]), axis=2)
        stacked_ccs = np.squeeze(stacked_ccs, axis=3)
        self.cc_strains = stacked_ccs

        stacked_linear_ccs = np.stack(linear_ccs, axis=2)
        avg_linear_cc = np.mean(stacked_linear_ccs, axis=2)
        stacked_linear_ccs = np.concatenate((stacked_linear_ccs, avg_linear_cc[...,np.newaxis]), axis=2)
        stacked_linear_ccs = np.squeeze(stacked_linear_ccs, axis=3)
        self.cc_linear_strains = stacked_linear_ccs

    '''
        Calculate rr strain for a batch of image sequences
        flattened_coords => [batch_size, nr_frames, 2, 168]
    '''
    def calculate_radial_strain(self, coords_batch, use_linear_strain=False):
        # point 0 is epi, point 6 is endo, do this for all the 'radials'
        endo_batch = coords_batch[:, :, :, ::7]
        epi_batch =  coords_batch[:, :, :, 6::7]

        # batch x time x 2 x 24 radials
        diff = (epi_batch - endo_batch) ** 2
        # print('diff', diff.shape)
        
        # batch x time x 24 sqrdiff
        summ = diff[:,:,0,:] + diff[:,:,1,:] # x^2 + y^2
        # print('summ', summ.shape)

        if use_linear_strain:
            # use L instead of L^2
            summ = np.sqrt(summ)

        # grab the frame 0 (ED) for all data, and 24 RR strains
        summ_ed = summ[:,0,:]

        # division through a certain column, without np.split
        # batch x time x 24 rr strains
        divv = summ/summ_ed[:,np.newaxis] # this is the trick, add new axis

        if use_linear_strain:
            rr_strains = divv - 1
        else:
            rr_strains = (divv - 1) / 2

        rr_strains = np.mean(rr_strains, axis=2)

        # batch x time x strain
        rr_strains = np.expand_dims(rr_strains, axis=2)
        return rr_strains

    def calculate_circumferential_strain(self, coords_batch, wall_index, use_linear_strain=False):
        # batch x time x 2 x 24
        midwall_points = coords_batch[:,:,:, wall_index::7]  # get point index 3 for every radial
        # print(midwall_points.shape)

        # we will have to calculate the strain between every points

        points_arr = np.split(midwall_points, 24, axis=3)

        # strain formula: ((l^2/L^2)-1) / 2  --> l^2 = x^2 + y^2
        # with x and y is the difference between x and y coords of 2 points
        ccs = []
        # the cc strain is circular, so we going through all of them and back to point 0
        for r in range(0,len(points_arr)):
            # for the last point, calculate between point_r and point_0
            if r+1 == len(points_arr):
                cc_diff = np.square(points_arr[r] - points_arr[0])
            else:
                cc_diff = np.square(points_arr[r] - points_arr[r+1])

            # do the sum: x^2 + y^2
            cc_sum = cc_diff[:,:,0] + cc_diff[:,:,1]

            if use_linear_strain:
                # use L instead of L^2
                cc_sum = np.sqrt(cc_sum)

            cc_sum_ed = cc_sum[:,0]
            
            # do the strain calculation
            partial_cc = cc_sum/cc_sum_ed[:, np.newaxis]
            if use_linear_strain:
                partial_cc = (partial_cc - 1)
            else:
                partial_cc = (partial_cc - 1) / 2

            # put the partial_cc in every time frame back together
            ccs.append(partial_cc)
        # stack the partial_cc for every links together
        stacked_ccs = np.stack(ccs, axis=2)

        # calculate the mean cc for every time frame
        mid_cc = np.mean(stacked_ccs, axis=2)
        # print(mid_cc.shape)
        # print(mid_cc[0][0:5])
        return mid_cc