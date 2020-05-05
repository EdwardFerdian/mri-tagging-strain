import os
import cim_utils


if __name__ == "__main__":
    # Configure the CIM directory here, CIM directory contains a number of analysis cases
    cim_filepath = "./CIM_analysis_models"
    # This is the DicomDir where the original Dicoms used by CIM are located
    dicom_filepath = "./DicomDir"
    
    # Saving option
    save_patient_info = True
    output_path = "./data"
    output_file = 'train.h5'
    output_filepath = os.path.join(output_path, output_file)

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # Scan all files within the CIM folder
    print('Scanning CIM folder:', cim_filepath)
    subfolders = [f.name for f in os.scandir(cim_filepath) if f.is_dir() ]  
    print('Total cases:', len(subfolders))

    total_saved = 0
    for index, case_name in enumerate(subfolders):
        print ((index+1),': Processing ', case_name)
        
        dm = cim_utils.generate_image_from_cim_analysis_files(cim_filepath, dicom_filepath, case_name)
        dm.save(output_filepath, save_patient_info=save_patient_info)

        total_saved += len(dm.patients)
        
    print('Total cines saved:', total_saved)
    print('==============================\n')
    print('Done!')

