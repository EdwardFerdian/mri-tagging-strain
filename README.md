# mri-tagging-strain
Deep learning framework for strain estimation on CMR Tagging images
version 1.0 by Edward Ferdian

This is an implementation of "CMR Tagging Strain Analysis using Deep Learning Framework" with Tensorflow 1.5.0. The framework consists of 2 different networks:
- Localisation CNN (available): it takes a single MRI Image (256x256) and performs a regression to output bounding box coordinates of the top left and bottom right, enclosing the myocardium with extra 30% space on each side.
- Tracking Landmark RNNCNN ( :warning: not available yet): it takes a sequence of MRI image (20 frames) and use shared-weight CNN to extract the spatial features of every frames, which are then passed on to the RNN to incorporate the temporal feature extraction. The RNNCNN network is trained end-to-end.


![Imgur](https://i.imgur.com/HNS3uRB.png)

![Imgur](https://i.imgur.com/gyenhs4.gif)

## 1. Installation
#### 1.1 Prerequisites
Please make sure that your machine is equipped with GPUs that support CUDA.
Python 3.6 is recommended.

#### 1.2 Requirements
(in progress)

## 2. Usage

#### 2.1 Data preparation
(In progress)

#### 2.2 Run the code
    

## 3. Contact Information

If you encounter any problems in using these codes, please open an issue in this repository.
You may also contact Edward Ferdian (edwardferdian03@gmail.com).

Thanks for your interest!
