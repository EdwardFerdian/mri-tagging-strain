import os
import cim_utils as cutils
import pydicom as dicom
import numpy as np
import math
import h5py
import cv2


class DataSetModel:
   
    def __init__(self):
        self.patients = []
        self.model_names = []
        self.slices = []

        self.image_seqs = []
        self.landmark_coords = []
        self.cropped_image_seqs = []
        self.cropped_landmark_coords = []

        self.px_spaces = []
        self.bbox_corners = []
        self.centroid_widths = []
        self.regions = []
        
        self.ed_frame_idx = []
        self.es_frame_idx = []

def _save_to_h5(output_path, col_name, dataset, string_dt=False):
    dataset = np.asarray(dataset)

    # convert float64 to float32 to save space
    if dataset.dtype == 'float64':
        dataset = np.array(dataset, dtype='float32')
    
    with h5py.File(output_path, 'a') as hf:    
        if col_name not in hf:
            datashape = (None, )
            if (dataset.ndim > 1):
                datashape = (None, ) + dataset.shape[1:]
            # special case
            if string_dt:
                string_dtype = h5py.special_dtype(vlen=str)
                hf.create_dataset(col_name, data=dataset, maxshape=datashape, dtype=string_dtype)
            else:
                hf.create_dataset(col_name, data=dataset, maxshape=datashape)
        else:
            hf[col_name].resize((hf[col_name].shape[0]) + dataset.shape[0], axis = 0)
            hf[col_name][-dataset.shape[0]:] = dataset

def get_filepaths(directory):
    """
        This function will generate the file names in a directory 
        tree by walking the tree either top-down or bottom-up. For each 
        directory in the tree rooted at directory top (including top itself), 
        it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.


def pad(matrix):
    '''
        Add some padding with equal width on the sides or top/bottom of the image.
        Image is padded until the width/height is equal to the largest number

        Input- matrix: a 2 dimensional matrix
        Output- 2 dimensional matrix with height = width
    '''
    nx, ny = np.shape(matrix)
    if (nx != ny):
        max_size = max(nx,ny)
        pad_x = (max_size-nx)//2
        pad_y = (max_size-ny)//2
        new_matrix = np.pad(matrix, ((pad_x,pad_x),(pad_y,pad_y)), 'constant')
        return new_matrix
    else:
        return matrix

def find_centroid(coords):
    '''
        Find a centroid, given a list of coordinates
        Coords: a (2, n) shaped numpy array
        Output: center point (x, y)
    '''
    center_x = (max(coords[0]) + min(coords[0])) /2  
    center_y = (max(coords[1]) + min(coords[1])) /2 
    return center_x, center_y


def adjust_single_bounding_box(corners, adjustment_fraction=0.3):
    '''
        Adjust the bounding box by adding a fraction of the width to sides and height to top/bottom
        Corners: The original bounding box corner (x1, y1, x2, y2)
    '''
    width  = corners[2]-corners[0]
    height = corners[3]-corners[1]

    # Add 30% space from the actual outer points
    diff = np.array((-width, -height, width, height)) * adjustment_fraction

    return corners + diff


def get_bounding_box_corners(coords):
    '''
        This function does the following:
        1. Get the bounding box corners as minimum and maximum x and y pairs from the 168 landmark coordinates
        2. Adjust the bounding box with an additional percentage
        3. Check if there is any value that is not inside the image, adjust them
      x1,y1 *------
            |     |
            |_____* x2,y2
    '''    
    corners = np.array((min(coords[0]), min(coords[1]), max(coords[0]), max(coords[1])))
    corners = adjust_single_bounding_box(corners)
    corners = assure_single_corner_within_img(corners)
    return corners

def assure_single_corner_within_img(corners):
    '''
        This function is used for single numpy 'row'
        Check if there is any value outside the image size
        Value <0 is set to 0
        Value >max is set to max
    '''
    corners[0] = np.maximum(corners[0],0) # set negative value to zero
    corners[1] = np.maximum(corners[1],0) # set negative value to zero

    corners[2] = np.minimum(corners[2],original_size) # set exceeding value to original_size
    corners[3] = np.minimum(corners[3],original_size)
    return corners

def crop_image_bbox(img, coords, bbox_corners):
    '''
        Crop the img based on the bbox_corners
        Coordinates are also adjusted based on the new coordinates after being cropped
    '''
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
        coords[0] = coords[0] - left_x
        coords[1] = coords[1] - low_y
        return img[low_y:high_y, left_x:right_x ], coords
    else:
        print('Cropping image with extra padding due to negative start index')
        new_img = np.pad(img, ((pad_y,pad_y),(pad_x,pad_x)), 'constant')
        coords[0] = coords[0] - left_x + pad_y
        coords[1] = coords[1] - low_y + pad_x
        return new_img[pad_y+low_y:pad_y+high_y, pad_x+left_x:pad_x+right_x ], coords


def crop_image_centroid(img, coords, center_x, center_y, crop_size):
    '''
        Crop image based on centroid
        Crop size is in pixels, we assume width and height are the same size.
    '''
    half_size = crop_size / 2
    #print("cropsize", crop_size)
    #print("halfsize", half_size)

    left_x = math.floor(center_x - half_size)
    right_x = math.ceil(center_x + half_size)
    
    low_y = math.floor(center_y - half_size)
    high_y = math.ceil(center_y + half_size)
    
    # special case where low_x or left_x is negative, we need to add some padding
    pad_x = 0
    pad_y = 0
    if (left_x < 0):
        pad_x = 0 - left_x
    if (low_y < 0):
        pad_y = 0 - low_y
    
    if (pad_x == 0 and pad_y == 0):
        coords[0] = coords[0] - left_x
        coords[1] = coords[1] - low_y
        return img[low_y:high_y, left_x:right_x ], coords
    else:
        print('Cropping image with extra padding due to negative start index')
        new_img = np.pad(img, ((pad_y,pad_y),(pad_x,pad_x)), 'constant')
        coords[0] = coords[0] - left_x + pad_y
        coords[1] = coords[1] - low_y + pad_x
        return new_img[pad_y+low_y:pad_y+high_y, pad_x+left_x:pad_x+right_x ], coords

def resize_image(img, coords, new_size):
    """
        Resample the image to the new_size
        Rescale the coordinates to the new_size coordinates
    """
    new_img = cv2.resize(img, dsize=(new_size,new_size), interpolation=cv2.INTER_CUBIC)
    
    ratio_x = new_size / img.shape[0]
    #ratio_y = new_img.shape(1) / img.shape(1)

    coords = np.array(coords)
    # assumption x and y has the same ratio
    new_coords = coords * ratio_x
    return new_img, new_coords

def generate_image_from_cim_analysis_files(filepath, img_filepath, patient_name, series_index=0):
    '''
        This is the main extractor function. It does the following:
        1. Check the system folder for .model file(s)
        2. Check the system folder for the .img_imageptr file
        3. From the .img_imageptr file, retrieve the path to the dicom files, in reference to the series, slice, and index
        4. For every model found (refer to step 1), retrieve the ED and ES index per series and slice
        5. Again, for every model, every time frame, we retrieve the following:
            a. DICOM file path
            b. Image pixel values from DICOM file
            c. Landmark coordinates from _strain.dat file
            d. Landmark regions from _strain.dat file
            e. Append to dataset_model
    '''
    max_frame = 20
    img_size  = 256
    crop_size = 128

    dataset_model = DataSetModel()

    # read out the system folder to check which frame is ES and ED
    system_dir = "{}/{}/system".format(filepath, patient_name)
    files = get_filepaths(system_dir)

    #print("Checking system folder for model description files...")
    model_files = [f for f in files if ".model" in f]
    # Get 1 ptr file from a case
    ptr_file = [f for f in files if f.endswith(".img_imageptr")][0]
    #print(ptr_file)
    datatype = [('series', '<i4'), ('slice', '<i4'), ('index', '<i4'), ('path', 'U255')]
    image_files = np.genfromtxt(ptr_file, delimiter='\t', names='series, slice, index, path', skip_header=1, dtype=datatype)
    
    condition = image_files['series'] == series_index
    image_files = image_files[condition]
    
    
    models = []
    for model_file in model_files:
        mod = cutils.read_model_file(model_file, system_dir, patient_name)
        models.append(mod)

    for model in models:
        # model.description()
        for num, series in enumerate(model.series_dirs):
            if (series.lower().startswith('series_2')):
                # print('skipping series_2')
                continue

            data_path = "{}/{}/{}/{}".format(filepath, patient_name, model.model_name, series.lower())

            slice_files = get_filepaths(data_path)
            all_frame_files = [f for f in slice_files if "_strain.dat" in f]

            imgs = []
            coords = []
            cropped_imgs = []
            cropped_coords = []
            regions = []
            
            corners = (0,0,0,0)
            px_space = 0

            nr_frames = len(all_frame_files)  # actual nr of frames of the sequence
            if (nr_frames > 0 and nr_frames != max_frame):
                print('Number of frames:', nr_frames)

            for frame_idx in range(0,max_frame):
                # read the dat file
                #file_prefix = "{}/{}_{}_".format(data_path, patient_name, (frame_idx+1)) # Linux version
                file_prefix = "{}\\{}_{}_".format(data_path, patient_name, (frame_idx+1)) # Windows version
                
                # read the dicom file
                if (nr_frames < max_frame and nr_frames > 0 and frame_idx >= nr_frames):
                    # just in case the sequence is less than max_frame, we pad it
                    # print('Number of frames', nr_frames, 'add some padding')
                    imgs.append(np.zeros((img_size,img_size)))
                    coords.append(np.zeros((2,168)))

                    cropped_imgs.append(np.zeros((crop_size,crop_size)))
                    cropped_coords.append(np.zeros((2,168)))
                    regions.append(np.zeros(168))
                else:
                    tmp_img, tmp_coords, corners, px_space, tmp_regions = prepare_binary(img_filepath, file_prefix, model, frame_idx, num, series, image_files, all_frame_files, corners)
                    
                    # Resample the image and recalculate the coords
                    tmp_crop_img, tmp_crop_coords = crop_image_bbox(tmp_img, tmp_coords, corners)            
                    tmp_crop_img, tmp_crop_coords = resize_image(tmp_crop_img, tmp_crop_coords, crop_size)
                    
                    if tmp_img is not None and tmp_coords is not None:
                        imgs.append(tmp_img)
                        coords.append(tmp_coords)
                        
                        cropped_imgs.append(tmp_crop_img)
                        cropped_coords.append(tmp_crop_coords)
                        regions.append(tmp_regions)
                    else:
                        if (frame_idx == 0): 
                            #print(patient_name, model.model_name, series, "The first image does not exist, don't bother, continue to next sequence")
                            break # don't bother continue on this slice
                        
                        # print('Missing image...', file_prefix)
                        imgs.append(np.zeros((img_size,img_size)))
                        coords.append(np.zeros((2,168)))
                        
                        cropped_imgs.append(np.zeros((crop_size,crop_size)))
                        cropped_coords.append(np.zeros((2,168)))
                        regions.append(np.zeros(168))

            if len(imgs) > 0:
                dataset_model.patients.append(patient_name)
                dataset_model.model_names.append(model.model_name)
                dataset_model.slices.append(series.lower())
                
                dataset_model.image_seqs.append(imgs)
                dataset_model.landmark_coords.append(coords)               
                dataset_model.px_spaces.append(px_space)
                
                dataset_model.bbox_corners.append(corners) # without the centroid, we always get this now
                dataset_model.regions.append(regions)

                dataset_model.ed_frame_idx.append(model.ed_indexes[num])
                dataset_model.es_frame_idx.append(model.es_indexes[num])

               

                dataset_model.cropped_image_seqs.append(cropped_imgs)
                dataset_model.cropped_landmark_coords.append(cropped_coords)
    
    return dataset_model

                
def prepare_binary(img_filepath, fileprefix, model, frame_idx, num, series, image_files, all_frame_files, bbox_corners):
    '''
        Retrieve the following information from the DICOM file and _strain.dat file
        - dicom pixel values
        - landmark coordinates
        - bounding box corners 
        - pixel spacing
        - landmark regions
    '''
    plot_file = [f for f in all_frame_files if f.startswith(fileprefix)]
    
    # print(all_frame_files)
    # print(fileprefix)
    # print('len(plot_file)',len(plot_file))
    # Sometimes the ed_file name does not match the pattern, it is good to check like this
    if (len(plot_file) > 0):
        # print('check DICOM image')
        # check DICOM image
        tmp = image_files[image_files['series']==(model.series_nrs[num]-1)]
        tmp = tmp[tmp['slice']==(model.slice_nrs[num]-1)]
        
        frame = frame_idx       
        tmp = tmp[tmp['index'] == frame]
        if (len(tmp) > 0):
            # load the DICOM IMAGE
            original_path = tmp['path'][0]
            dicom_path = original_path
            dicom_path = dicom_path.replace("IMAGEPATH\\\\", img_filepath+"\\") # to handle weird case below, we do this first
            dicom_path = dicom_path.replace("IMAGEPATH\\", img_filepath+"\\") # there is a weird case like this, yes
            dicom_path = dicom_path.replace("\\", "/")
                        
            if not(os.path.isfile(dicom_path)):
                # If file does not exist
                print('missing ',frame_idx, dicom_path)
                return None, None, None, None, None
            #title = f"{model.model_name}-{series}-FR{frame+1}"

            ds = dicom.dcmread(dicom_path)
            images = pad(ds.pixel_array)

            labeled_points = cutils.get_model_points_and_label_from_file(plot_file[0])
            
            coords = cutils.to_dicom_coords(labeled_points, np.shape(images))
            regions = cutils.get_regions(labeled_points)

            ConstPixelSpacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), float(ds.SliceThickness))
            if frame_idx == 0: # first slice
                # center_x, center_y = find_centroid(coords)
                corners = get_bounding_box_corners(coords)
            else:
                # center_x, center_y = centroid[0], centroid[1]
                corners = bbox_corners

            

            # print(ConstPixelSpacing[0])
            # print(bbox_points)
            return images, coords, corners, ConstPixelSpacing[0], regions
    return None, None, None, None, None


def save_to_h5(output_filepath, dm, start_idx, end_idx):
    
    
    patients = np.array(dm.patients, dtype=object)
    models = np.array(dm.model_names, dtype=object)
    slices = np.array(dm.slices, dtype=object)
    
    # Patient info
    _save_to_h5(output_filepath, '_patients', patients, string_dt=True)
    _save_to_h5(output_filepath, '_models', models, string_dt=True)
    _save_to_h5(output_filepath, '_slices', slices, string_dt=True)
    
    # For the localisation network
    _save_to_h5(output_filepath, 'image_seqs', dm.image_seqs)
    _save_to_h5(output_filepath, 'landmark_coords', dm.landmark_coords)
    _save_to_h5(output_filepath, 'bbox_corners', dm.bbox_corners)
    _save_to_h5(output_filepath, 'px_spaces', dm.px_spaces)

    # This is for the landmark tracking network
    _save_to_h5(output_filepath, 'cropped_image_seqs', dm.cropped_image_seqs)
    _save_to_h5(output_filepath, 'cropped_landmark_coords', dm.cropped_landmark_coords)
    
    # Other stuff
    _save_to_h5(output_filepath, 'ed_frame_idx', dm.ed_frame_idx)
    _save_to_h5(output_filepath, 'es_frame_idx', dm.es_frame_idx)
    _save_to_h5(output_filepath, 'regions', dm.regions)

    

def prepare_sequence_data(filepath, img_filepath, output_path, output_file):
    '''
        Retrieve and save all the needed information from the image filepath and cim filepath
        The file is saved in the output_path with a certain prefix
    '''
    # For all files
    print('Scanning folder:', filepath)
    subfolders = [f.name for f in os.scandir(filepath) if f.is_dir() ]  
    print('Total patients:',len(subfolders))

    total_saved = 0

    
    for index, patient_name in enumerate(subfolders):
        print ((index+1),': Processing ', patient_name)
        # dm = DataSetModel() 
        dm = generate_image_from_cim_analysis_files(filepath, img_filepath, patient_name)

        output_filepath = os.path.join(output_path, output_file)
        save_to_h5(output_filepath, dm, 0, len(dm.patients))

        total_saved += len(dm.patients)
        if index > 0:
            break
        

    print('Total image sequences saved:', total_saved)
    print('==============================\n')
    
    


if __name__ == "__main__":
    
    original_size = 256

    cropping = False # when True, the data is cropped based on the bounding box, coordinates are adjusted accordingly
    img_size = 128 # Resampled image size (used when cropping=True)

    if not cropping:
        # if not cropping, we set the img_size to the original. This is used in frame padding 
        print('No cropping, image size is set to 256x256')
        img_size = original_size

    base_path = "C:/Users/efer502/Documents/ukb_tags/40casesICC-EL+AB_after cleaning 1"
    dicom_filepath = "C:/Users/efer502/Documents/ukb_tags/40cases_pat_data"
    output_path = "./data"

    h5file = 'test.h5'
    cim_filepath = base_path

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    prepare_sequence_data(cim_filepath, dicom_filepath, output_path, h5file)
  
    print('Done!')

