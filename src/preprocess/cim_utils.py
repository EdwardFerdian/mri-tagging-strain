import re
import numpy as np

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


def read_model_file(model_file, system_file_path, patient_name):
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
    # model_name = model_file[len(f"{system_file_path}\\{patient_name}."):]
    model_name = model_file[len("{}/{}.".format(system_file_path, patient_name)):]

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