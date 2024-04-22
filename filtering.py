import os
import numpy as np
from scipy.signal import medfilt

def median_filter(input_path, output_path=None, kernel_size=5):
    # Apply a median filter independently to all landmarks time series

    sgn = np.load(input_path)

    for keypoint in range(sgn.shape[1]):
        for axis in range(sgn.shape[2]):
            sgn[:, keypoint, axis] = medfilt(sgn[:, keypoint, axis], kernel_size)
    
    if output_path is not None:
        np.save(output_path, sgn)    


def median_folder(input_directory_path, output_directory_path):
    # Apply the median_filter function to all files for one patient

    fnames = os.listdir(input_directory_path)
    if not os.path.exists(output_directory_path):
        os.mkdir(output_directory_path)

    for fname in fnames:
        if 'time' not in fname:
            fpath = input_directory_path + fname
            output_path = output_directory_path + fname.split(".")[0]
            median_filter(fpath, output_path=output_path)


def smooth_data(patient_ids, results_dir):
    # Apply the medain filters to all files of all patients

    for patient_id in patient_ids:
        # The smoothed time series will be written in 'median' folders for each patient
        input_folder = results_dir + str(patient_id) + "/landmarks/"
        output_folder = results_dir + str(patient_id) + "/median/"
        
        if not os.path.exists(results_dir + str(patient_id)):
            os.mkdir(results_dir + str(patient_id))

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
            
        median_folder(input_folder, output_folder)
