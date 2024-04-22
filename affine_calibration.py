from sklearn.linear_model import LinearRegression
import os
import pandas as pd
import numpy as np


def calibrate(ground_truth_path, estimation_path, output_path):
    """
    Read the estimations in 'estimation_path', calibrate them using 
    'ground_truth_path' and write the results in 'output_path'
    """

    ground_truth_df = pd.read_csv(ground_truth_path)
    estimation_df = pd.read_csv(estimation_path)

    patient_ids = estimation_df[estimation_df.columns[0]]
    ground_truth_ids = list(ground_truth_df[ground_truth_df.columns[0]])
    ids_to_select = []
    for id in patient_ids:
        ids_to_select.append(ground_truth_ids.index(id))

    estimation_array = np.array(estimation_df)[:, 1:]
    
    ground_truth_array = np.array(ground_truth_df)[ids_to_select, 1:]

    corrected_estimations = np.zeros_like(estimation_array)
    for angle_id in range(corrected_estimations.shape[1]):
        for patient_id in range(corrected_estimations.shape[0]):
        
            X = estimation_array[:, angle_id].copy()
            X[patient_id:-1] = X[patient_id+1:]
            X = X[:-1].reshape(-1, 1)

            Y = ground_truth_array[:, angle_id].copy()
            Y[patient_id:-1] = Y[patient_id+1:]
            Y = Y[:-1]

            linReg = LinearRegression()
            linReg.fit(X, Y)

            corrected_estimations[patient_id, angle_id] = linReg.predict(estimation_array[patient_id, angle_id].reshape(1, 1))[0]
        
    pd.DataFrame(corrected_estimations.astype(int), index=patient_ids).to_csv(output_path)


def calibrate_files(f_names, results_dir, ground_truth_path):
    # Calibrate a list of files using the same ground truth file
    for f_name in f_names:
        estimation_path = os.path.join(results_dir, f_name + ".csv")
        output_path = os.path.join(results_dir, f_name + "_calib.csv")
        calibrate(ground_truth_path, estimation_path, output_path)
