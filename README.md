# mri-tagging-strain

This is an implementation of the paper [Fully Automated Myocardial Strain Estimation from Cardiovascular MRI–tagged Images Using a Deep Learning Framework in the UK Biobank](https://pubs.rsna.org/doi/10.1148/ryct.2020190032) using Tensorflow 1.8.0. 

The framework consists of 2 different networks:
- Localisation CNN (available): it takes a single MRI Image (256x256) and performs a regression to output bounding box coordinates of the top left and bottom right, enclosing the myocardium with extra 30% space on each side.
- Tracking Landmark RNNCNN it takes a sequence of MRI image (20 frames) and use shared-weight CNN to extract the spatial features of every frames, which are then passed on to the RNN to incorporate the temporal feature extraction. The RNNCNN network is trained end-to-end.

Please find the pre-trained networks weights here:
[MRI-tagging network models](https://auckland.figshare.com/collections/Fully_Automated_Myocardial_Strain_Estimation_from_Cardiovascular_MRI_tagged_Images_Using_a_Deep_Learning_Framework_in_the_UK_Biobank/4962155).

If you are using later Tensorflow 1.x version that is not compatible with this version, please refer to Tensorflow backwards compatibility (tf.compat module). 

 
## Overall framework workflow
![Imgur](https://i.imgur.com/HNS3uRB.png)


## 1. Installation
#### 1.1 Prerequisites
Please make sure that your machine is equipped with GPUs that support CUDA.

Python 3.6 is recommended. Dependencies are listed in requirements.txt

#### 1.2 Requirements
We provided an example data (HDF5 format) with 1 row of data.

Localization Network requires the following data column: 
- image_seqs - MRI image sequence with n_rows x 20 frames x 256 x 256)
- bbox_corners -  Bounding box coordinates nrows x 4 (x1, y1, x2, y2) for top left and bottom right corner.

Landmark Tracking Network requires the following data column:
- cropped_image_seqs - Cropped image sequence using the provided bounding box n_rows x 20 frames x 128 x 128
- cropped_landmark_coords - 168 Landmark coordinates on the myocardium n_rows x 20 frames x 2 x 168

#### 1.3 Prepare data
We provided a script to prepare the dataset from CIM_TAG_2D v6.0 output files and Dicom images. Configure these two directories first within the prepare_data.py then run the script.


## 2. Usage

#### 2.1 Run the code

#### 2.1.1 Localization Network
Training:

    Open localization_trainer.py
    Configure the training and validation data path
    Run the localization_trainer.py script
    Model is saved by default under models/ directory

Prediction:
    
    Open localization_predictor.py
    Configure the model directory
    Run the localization_predictor.py script

#### 2.1.2 Landmark Tracking Network

#### Architecture:
![Imgur](https://i.imgur.com/15QjrWI.png)

Training:

    Open landmark_trainer.py
    Configure the training and validation data path
    Run the landmark_trainer.py script
    Model is saved by default under models/ directory

Prediction:

    Open landmark_predictor.py
    Configure the model directory
    Run the landmark_predictor.py script

#### 2.2 Prediction pipeline

The script will run the whole prediction using the 2 networks. The first network predicts the bounding box and localize the region of interests (ROI).
The ROI will then cropped and resize before passing them to the second network. The second network will predict the local landmark coordinates from the cropped frames.
Finally, radial and circumferential strains are calculated.

To run the example script, do the following:
    
    1. From the project root, create a new directory /models
    2. Place the pre-trained models in the models directory
    3. Run prediction_pipeline.py

We provided an option to save the results to a GIF file. For this purpose, ImageMagick is required.

#### 2.3 Example Result

![Imgur](https://i.imgur.com/gyenhs4.gif)

## 3. Contact Information

If you encounter any problems in using these codes, please open an issue in this repository.
Author: Edward Ferdian (edwardferdian03@gmail.com).

Thanks for your interest!
