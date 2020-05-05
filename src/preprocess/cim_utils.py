import re
import numpy as np
import pydicom
import math
import h5py
import cv2
import os

class AnalysisModel:
    model_name = ""
    series_dirs = []
    series_nrs = []
    slice_nrs = []
    ed_indexes = []
    es_indexes = []

    def description(self):
        print('----- Model {} -----'.format(self.model_name))
        print(self.series_dirs)
        print('Series: {}'.format(self.series_nrs))
        print('Slices: {}'.format(self.slice_nrs))
        print('Ed indexes: {}'.format(self.ed_indexes))
        print('Es indexes: {}'.format(self.es_indexes))
        print()


def read_model_file(model_file, system_file_path, case_name):
    '''
        Read the .model file, extract the ED and ES frame index using Regex
        Grab the series and slice name
    '''
    # read the desc file
    pattern_ed = r"(Series(.+?)Slice(.+?))End-diastolic Frame\s*?.*?(\d+)"
    pattern_es = r"(Series(.+?)Slice(.+?))End-systolic Frame\s*?.*?(\d+)"
    reg_ed = re.compile(pattern_ed)
    reg_es = re.compile(pattern_es)

    all_lines = []
    # substring and get the model name only
    # model_name = model_file[len(f"{system_file_path}\\{case_name}."):]
    model_name = model_file[len("{}/{}.".format(system_file_path, case_name)):]

    #print("Checking model file for index of ED and ES frame....")
    with open(model_file) as myFile:
        all_lines = myFile.read().splitlines()

    # Retrieve the lines containing the patterns
    ed_frames = list(filter(reg_ed.match, all_lines))
    es_frames = list(filter(reg_es.match, all_lines))

    ed_indexes = []
    es_indexes = []
    series_folders = []
    series_numbers = []
    slice_numbers = []

    model = AnalysisModel()
    # Retrieve the index of the ed frames and es frames
    for ed_frame in ed_frames:
        m = reg_ed.match(ed_frame)
        series_folder = m.group(1).replace(" ", "_").rstrip("_")
        series_folders.append(series_folder)
        series_numbers.append(int(m.group(2).strip()))
        slice_numbers.append(int(m.group(3).strip()))
        ed_index = int(m.group(4))
        ed_indexes.append(ed_index)

    for es_frame in es_frames:
        m = reg_es.match(es_frame)
        es_index = int(m.group(4))
        es_indexes.append(es_index)

    model.model_name = model_name
    model.series_nrs = series_numbers
    model.slice_nrs = slice_numbers
    model.ed_indexes = ed_indexes
    model.es_indexes = es_indexes
    model.series_dirs = series_folders
    
    #model.description()
    return model

def get_model_points_and_label_from_file(filepath, delimiter= ' ', names=True):
    '''
        Get the list of coordinates, regions, and other properties from the _strain.dat file
        Return a tuple of (x coord, y coords, LV (AHA) segment, elements (4 segment), division, radial index )
    '''
    data = np.genfromtxt(filepath, delimiter=' ', names=True)
    points = (data['imageX'], data['imageY'], data['Region'],data['Element'],data['SubDivXi1'],data['SubDivXi2'])
    return points

def to_dicom_coords(labeled_points, dimensions):
    '''
        Points is an array with 6 elements
        See: get_model_points_and_label_from_file
        ['imageX'], ['imageY'], ['Region'], ['Element'],['SubDivXi1'],['SubDivXi2'])
    '''
    coords = [labeled_points[0], (dimensions[0]-labeled_points[1])]
    return coords


def get_regions(labeled_points):
    return labeled_points[2]

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
        # self.centroid_widths = []
        self.regions = []
        
        self.ed_frame_idx = []
        self.es_frame_idx = []

    def save(self, output_filepath, save_patient_info=False):
        # For the localisation network
        _save_to_h5(output_filepath, 'image_seqs', self.image_seqs)
        _save_to_h5(output_filepath, 'landmark_coords', self.landmark_coords)
        _save_to_h5(output_filepath, 'bbox_corners', self.bbox_corners)
        _save_to_h5(output_filepath, 'px_spaces', self.px_spaces)

        # This is for the landmark tracking network
        _save_to_h5(output_filepath, 'cropped_image_seqs', self.cropped_image_seqs)
        _save_to_h5(output_filepath, 'cropped_landmark_coords', self.cropped_landmark_coords)
        
        # Other stuff
        _save_to_h5(output_filepath, 'ed_frame_idx', self.ed_frame_idx)
        _save_to_h5(output_filepath, 'es_frame_idx', self.es_frame_idx)
        _save_to_h5(output_filepath, 'regions', self.regions)

        if save_patient_info:    
            patients = np.array(self.patients, dtype=object)
            models = np.array(self.model_names, dtype=object)
            slices = np.array(self.slices, dtype=object)
            
            # Patient info
            _save_to_h5(output_filepath, '_cases', patients, string_dt=True)
            _save_to_h5(output_filepath, '_models', models, string_dt=True)
            _save_to_h5(output_filepath, '_slices', slices, string_dt=True)

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

    corners[2] = np.minimum(corners[2],256) # set exceeding value to original_size
    corners[3] = np.minimum(corners[3],256)
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

def generate_image_from_cim_analysis_files(cim_dir, dicom_dir, case_name, series_index=0):
    '''
        This is the main extractor function. It does the following:
        1. From the cim case file, check the system folder for .model file(s)
        2. Check the system folder for the .img_imageptr file
        3. From the .img_imageptr file, retrieve the path to the dicom files, in reference to the series, slice, and index
            IMAGEPATH in this file refers to your dicom_dir, please put your dicoms in the designated folders.
        4. For every cim analysis model found (refer to step 1), retrieve the ED and ES index per series and slice
            We only process series 1, as series 2 refers to longitudinal
        5. For every CIM analysis model, go to every time frame, we retrieve the following:
            a. DICOM file path
            b. Image pixel values from DICOM file
            c. Landmark coordinates from _strain.dat file
            d. Landmark regions from _strain.dat file
            e. Return a dataset_model
    '''
    max_frame = 20
    img_size  = 256
    crop_size = 128

    dataset_model = DataSetModel()

    # 0. read out the system folder to check which frame is ES and ED
    system_dir = "{}/{}/system".format(cim_dir, case_name)
    files = get_filepaths(system_dir)

    # 1 .print("Checking system folder for model description files...")
    model_files = [f for f in files if ".model" in f]
    
    # 2. Get 1 ptr file from a case
    ptr_file = [f for f in files if f.endswith(".img_imageptr")][0]

    # 3. Get the dicom path list from the pointer file
    datatype = [('series', '<i4'), ('slice', '<i4'), ('index', '<i4'), ('path', 'U255')]
    image_files = np.genfromtxt(ptr_file, delimiter='\t', names='series, slice, index, path', skip_header=1, dtype=datatype)
    
    condition = image_files['series'] == series_index
    image_files = image_files[condition]
    
    # 4. Get the ED and ES info from the model file
    models = []
    for model_file in model_files:
        mod = read_model_file(model_file, system_dir, case_name)
        models.append(mod)

    for model in models:
        # model.description()
        for num, series in enumerate(model.series_dirs):
            if (series.lower().startswith('series_2')):
                # print('skipping series_2')
                continue
            
            # The strain.dat files are in the cim_filepath/case/model_case/series
            strain_dat_path = "{}/{}/{}/{}".format(cim_dir, case_name, model.model_name, series.lower())

            slice_files = get_filepaths(strain_dat_path)
            all_frame_files = [f for f in slice_files if "_strain.dat" in f]

            nr_frames = len(all_frame_files)  # actual nr of frames of the sequence
            if nr_frames == 0:
                continue

            if (nr_frames != max_frame):
                # Print out any model that does not have max_frame (default: 20)
                print('Number of frames:', nr_frames)

            imgs = []
            coords = []
            cropped_imgs = []
            cropped_coords = []
            regions = []

            # Go through every frame, any frame exceeding max_frame won't be read
            for frame_idx in range(0,max_frame):
                # read the dat file
                #file_prefix = "{}/{}_{}_".format(data_path, case_name, (frame_idx+1)) # Linux version
                file_prefix = "{}\\{}_{}_".format(strain_dat_path, case_name, (frame_idx+1)) # Windows version

                # go through the list of collected filenames
                strain_files = [f for f in all_frame_files if f.startswith(file_prefix)]
                dicom_files = find_dicom_images(model, frame_idx, num, series, image_files)

                tmp_img, tmp_coords, px_space, tmp_regions = extract_img_coords_data(dicom_dir, dicom_files, strain_files)
               

                if tmp_img is None or tmp_coords is None:
                    if (frame_idx == 0): 
                        #print(case_name, model.model_name, series, "The first image does not exist, don't bother, continue to next cine")
                        break # don't bother continue on this slice
                    
                    # print('Missing image...', file_prefix)
                    imgs.append(np.zeros((img_size,img_size)))
                    coords.append(np.zeros((2,168)))
                    
                    cropped_imgs.append(np.zeros((crop_size,crop_size)))
                    cropped_coords.append(np.zeros((2,168)))
                    regions.append(np.zeros(168))
                else:
                    if frame_idx == 0:
                        # We calculate bounding box during ED frame only
                        # center_x, center_y = find_centroid(coords)
                        corners = get_bounding_box_corners(tmp_coords)
                    imgs.append(tmp_img)
                    coords.append(tmp_coords)

                    # Resample the image and recalculate the coords
                    tmp_crop_img, tmp_crop_coords = crop_image_bbox(tmp_img, tmp_coords, corners)            
                    tmp_crop_img, tmp_crop_coords = resize_image(tmp_crop_img, tmp_crop_coords, crop_size)
                    
                    cropped_imgs.append(tmp_crop_img)
                    cropped_coords.append(tmp_crop_coords)
                    regions.append(tmp_regions)
                
            # after the long else if
            if len(imgs) > 0:
                dataset_model.patients.append(case_name)
                dataset_model.model_names.append(model.model_name)
                dataset_model.slices.append(series.lower())
                
                dataset_model.image_seqs.append(imgs)
                dataset_model.landmark_coords.append(coords)               
                dataset_model.px_spaces.append(px_space)
                
                dataset_model.bbox_corners.append(corners)
                dataset_model.regions.append(regions)

                dataset_model.ed_frame_idx.append(model.ed_indexes[num])
                dataset_model.es_frame_idx.append(model.es_indexes[num])

                dataset_model.cropped_image_seqs.append(cropped_imgs)
                dataset_model.cropped_landmark_coords.append(cropped_coords)
    
    return dataset_model

def find_dicom_images(model, frame_idx, num, series, image_files):
    # check DICOM image
    tmp = image_files[image_files['series']==(model.series_nrs[num]-1)]
    tmp = tmp[tmp['slice']==(model.slice_nrs[num]-1)]
    
    frame = frame_idx       
    tmp = tmp[tmp['index'] == frame]
    return tmp
                
def extract_img_coords_data(dicom_dir, dicom_files, strain_files):
    '''
        Retrieve the following information from the DICOM file and _strain.dat file
        - dicom pixel values
        - landmark coordinates
        - pixel spacing
        - landmark regions
    '''
        
    # check DICOM image
    if (len(dicom_files) > 0):
        # load the DICOM IMAGE
        original_path = dicom_files['path'][0]
        dicom_path = original_path
        dicom_path = dicom_path.replace("IMAGEPATH\\\\", dicom_dir+"\\") # to handle weird case below, we do this first
        dicom_path = dicom_path.replace("IMAGEPATH\\", dicom_dir+"\\") # there is a weird case like this, yes
        dicom_path = dicom_path.replace("\\", "/")
                    
        if not(os.path.isfile(dicom_path)):
            # Missing image
            print('missing ', dicom_path)
            return None, None, None, None
        #title = f"{model.model_name}-{series}-FR{frame+1}"

        # Get the image data
        ds = pydicom.dcmread(dicom_path)
        images = pad(ds.pixel_array)
        # This is the px spacing (in mm)
        ConstPixelSpacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), float(ds.SliceThickness))
        px_space = ConstPixelSpacing[0]

        if len(strain_files) == 0 :
            print('Missing landmark file')
            return images, None, px_space, None

        # Get the landmark coordinates
        labeled_points = get_model_points_and_label_from_file(strain_files[0])
        coords = to_dicom_coords(labeled_points, np.shape(images))

        # Get the landmark regions
        regions = get_regions(labeled_points)

        return images, coords, px_space, regions
    