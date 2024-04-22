import os

from filtering import smooth_data
from angles import extract_angles
from landmarks_extraction import extract_landmarks
from affine_calibration import calibrate_files

# Absolute or relative path of the folder in which the output files will be written
results_dir = "IAAAA_results/" # TO MODIFY

# Absolute (not relative) path of the folder containing the data folders
data_dir = "data/" # TO MODIFY

# Absolute or relative path of the file with the angles measured with the goniometre
ground_truth_path = "gonio.csv" # TO MODIFY

# Absolute or relative path of the mediapipe model
model_path = 'models/pose_landmarker_heavy.task' # TO MODIFY

if not os.path.exists(results_dir):
    os.mkdir(results_dir)

# Create the list of the patient numbers
patient_ids = []
for f_name in os.listdir(data_dir):
    if os.path.isdir(os.path.join(data_dir, f_name)):
        try:
            patient_ids.append(int(f_name))
        except:
            pass

print("Patient ids:")
print(patient_ids)

# Use the realsense and mediapipe libraries to extract the landmarks (longest part to run)
print("Extract landmarks")
extract_landmarks(patient_ids, data_dir, results_dir, model_path)

# Smooth the landmarks time series using a median filter
print("Smooth data")
smooth_data(patient_ids, results_dir)

# Process the smoothed landmarks files to compute the angles
print("Extract angles")
extract_angles(patient_ids, results_dir)

# Postprocess the estimations to make an affine calibration with the leave-one-out strategy
print("Calibrate files")
f_names = ["angles_2D", "angles_3D_d", "angles_3D_mp"]
calibrate_files(f_names, results_dir, ground_truth_path)
