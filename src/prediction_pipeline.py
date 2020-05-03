import time
import os
import h5py
from PredictionResult import PredictionResult
import LocalisationCNN.loader as local_loader
import LandmarkTrackingRNNCNN.loader as rnncnn_loader
import prediction_utils as utils
import gif_utils as gif_utils

# =========== Option ==============
# ImageMagick is required to save to gif
show_images = False # show the first cine to screen or save
save_to_gif = False # save_to_gif is only used when show_images is True
gif_path = './result'

# ============== Data path ============== 
data_path = './data'
file_ext_pattern = '.h5'
output_path = './result'

# ============== Network models config ==============
# 1. Localisation Network
localisation_network_path  = './model/LocalCNN'
localisation_network_name = 'localizer'
# 2. RNNCNN Network
rnncnn_network_path = './model/LandmarkTrackingRNNCNN'
rnncnn_network_name = 'rnncnn'


if __name__ == "__main__":    
    # traverse the input folder, keep as array to combine later
    files  = os.listdir(data_path)
    # get all the .h5 input filenames
    input_files =  [f for f in files if file_ext_pattern in f]

    print('{} files found!'.format(len(input_files)))
    print(input_files)

    # Load both networks here..let's see
    print('Loading networks...')
    localnet = local_loader.NetworkModelCNN(localisation_network_path, localisation_network_name)
    landmarknet = rnncnn_loader.NetworkModelRNNCNN(rnncnn_network_path, rnncnn_network_name)

    # loop through all the input files, and run the network
    for num, input_file in enumerate(input_files):
        print("\n--------------------------")
        print('\nProcessing {} ({}/{}) - {}'.format(input_file, num+1, len(input_files), time.ctime()))
        start_time = time.time()

        # 0. Load the data
        input_filepath = os.path.join(data_path,input_file)
        img_sequences = utils.load_dataset(input_filepath)
        
        # 1. Predict bounding box
        # we only need the ED frames for first pipeline
        ed_imgs = img_sequences[:,0,:,:] # frame t0 only, ED frame
        corners = localnet.predict_corners(ed_imgs)
        
        # 2. Localize the image based on predicted bounding box
        cropped_frames, resize_ratios = utils.crop_and_resize_all_frames(img_sequences, corners)

        # 3. Predict localized landmarks 
        landmarks = landmarknet.predict_landmark_sequences(cropped_frames)
        
        # 4. Prepare to save results
        results = PredictionResult(corners, landmarks, resize_ratios)
        
        # 5. Calculate strains
        results.calculate_strains()

        # 6. Save results
        output_prefix = input_file[:-len(file_ext_pattern)] # strip the extension
        results.save_predictions(output_path, output_prefix)
        
        # ----------- Elapsed time ----------- 
        time_taken = time.time() - start_time
        fps = len(img_sequences) / time_taken
        print("Prediction pipeline - {} cines: {:.2f} seconds ({:.2f} cines/second)".format(len(img_sequences), time_taken, fps))

        if show_images:
            print("Showing/saving first cine case to GIF")
            if not os.path.isdir(gif_path):
                os.makedirs(gif_path)
            gif_filename = '{}/{}-0.gif'.format(gif_path, input_file)
            gif_utils.prepare_animation(img_sequences[0], cropped_frames[0], results.landmark_sequences[0], results.cc_strains[0], results.rr_strains[0], save_to_gif=save_to_gif, gif_filepath=gif_filename)


    print("\n====== Done! ======")