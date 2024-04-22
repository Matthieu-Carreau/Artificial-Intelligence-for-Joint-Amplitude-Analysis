import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Indices of each landmark of interet in the output of the mediapipe model

LEFT_ANKLE = 27
LEFT_KNEE = 25
LEFT_HIP = 23
LEFT_SHOULDER = 11
RIGHT_ANKLE = 28
RIGHT_KNEE = 26
RIGHT_HIP = 24
RIGHT_SHOULDER = 12


def angle_vectors(u, v):
    # Computes the angle between two vectors in degrees
    dot = np.dot(u, v)
    cos = dot / np.linalg.norm(u) / np.linalg.norm(v)
    return np.round(np.arccos(cos)*180 / np.pi)


def series_angle_vectors(u, v):
    # Computes the angles time series (in degrees) between two time series of vectors
    dot = np.sum(u*v, axis=1)
    cos = dot / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)
    return np.round(np.arccos(cos)*180 / np.pi)


def angle_separated_file(results_dir, patient_id, side, clip_name, knee=False, plot=False):
    """
    Extract the angles of interest from landmarks time series stored in 'clip_name'
    side: 'R' or 'L'
    clip_name: 'ABD', 'ADD', 'EXT', 'FLX', 'RE', 'RI'
    """
    base_path = results_dir + str(patient_id) + "/median/" 
    base_path += str(patient_id) + " " + side + " " + clip_name + "_"
    crop_time = 5 # number of time frames that are not taken into account at the beginning and at the end
    time_angle = None
    angle_2D = np.nan
    angle_3D_d = np.nan # Using depth information
    angle_3D_mp = np.nan # Estimation using mediapipe only

    # According to the clip name, we choose the segments between which the angle is computed
    if clip_name in ['ABD', 'ADD', 'EXT', 'FLX']:
        if side == 'R':
            a_start = RIGHT_SHOULDER
            a_end = RIGHT_HIP
            b_start = RIGHT_HIP
            b_end = RIGHT_KNEE

        else:
            a_start = LEFT_SHOULDER
            a_end = LEFT_HIP
            b_start = LEFT_HIP
            b_end = LEFT_KNEE
    
    elif clip_name in ['RE', 'RI']:
        # For hip rotation, the vertical reference is the line between the akle on the floor and the associated 
        if side == 'R':
            a_start = RIGHT_KNEE
            a_end = RIGHT_ANKLE
            b_start = LEFT_SHOULDER
            b_end = LEFT_ANKLE

        else:
            a_start = LEFT_KNEE
            a_end = LEFT_ANKLE
            b_start = RIGHT_SHOULDER
            b_end = RIGHT_ANKLE

    else:
        print("Error in the clip_name:", clip_name)
    
    if knee:
        # Add points for the knee flexion
        if side == 'R':
            a_start = RIGHT_HIP
            a_end = RIGHT_KNEE
            b_start = RIGHT_KNEE
            b_end = RIGHT_ANKLE

        else:
            a_start = LEFT_HIP
            a_end = LEFT_KNEE
            b_start = LEFT_KNEE
            b_end = LEFT_ANKLE
    
    for suffix in ["raw", "cor", "wor"]:
        fpath = base_path + suffix + ".npy"
        
        if os.path.exists(fpath):
            landmarks = np.load(fpath)

            if suffix == "raw": # Only use 2D
                landmarks = landmarks[:, :, :2]
                landmarks[:, 0] *= 4/3 # Aspect ratio correction

                # Select the maximum
                angles = series_angle_vectors(landmarks[:, a_end] - landmarks[:, a_start],
                                              landmarks[:, b_end] - landmarks[:, b_start])
                
                # Avoid boundary effects
                mask = np.zeros(landmarks.shape[0])
                mask[crop_time: -crop_time] = 1
                time_angle = np.argmax(angles*mask)

                angle_2D = angles[time_angle]

                if plot:
                    plt.plot([time_angle]*2, [0, 70], color="red", linestyle="--")
                    plt.plot(angles, color="black")
                    plt.scatter([time_angle], [angles[time_angle]], color="red")
                    plt.xlabel("Frame number")
                    plt.ylabel("Degrees")
                    plt.show()
                
            
            elif suffix == "cor":
                frame = landmarks[time_angle]
                angle_3D_d = angle_vectors(frame[a_end] - frame[a_start],
                                           frame[b_end] - frame[b_start])
            else:
                frame = landmarks[time_angle]
                angle_3D_mp = angle_vectors(frame[a_end] - frame[a_start],
                                           frame[b_end] - frame[b_start])
                
    return time_angle, angle_2D, angle_3D_d, angle_3D_mp


def extract_angles(patient_ids, results_dir):
    """
    Extract the angles from all the patient ids in patient_ids and write them in csv files
    """

    # Dataframes to contain the angles
    angles_2D = pd.DataFrame(index=patient_ids)
    angles_3D_d = pd.DataFrame(index=patient_ids)
    angles_3D_mp = pd.DataFrame(index=patient_ids)

    for patient_id in patient_ids:
        for side in ['R', 'L']:
            
            # Flexion Genou
            time, angle_2D, angle_3D_d, angle_3D_mp = angle_separated_file(results_dir, patient_id, side, 'FLX', knee=True)
            col = side + ' FLX GEN'
            angles_2D.at[patient_id, col] = int(angle_2D)
            angles_3D_d.at[patient_id, col] = int(angle_3D_d)
            angles_3D_mp.at[patient_id, col] = int(angle_3D_mp)

            # Flexion Hanche
            time, angle_2D, angle_3D_d, angle_3D_mp = angle_separated_file(results_dir, patient_id, side, 'FLX')
            col = side + ' FLX HAN'
            angles_2D.at[patient_id, col] = int(angle_2D)
            angles_3D_d.at[patient_id, col] = int(angle_3D_d)
            angles_3D_mp.at[patient_id, col] = int(angle_3D_mp)
            
            # Other angles: Adduction, Rotation interne, Extension, Abduction, Rotation externe
            for clip_name in ['ADD', 'RI', 'EXT', 'ABD', 'RE']:
                time, angle_2D, angle_3D_d, angle_3D_mp = angle_separated_file(results_dir, patient_id, side, clip_name)
                col = side + ' ' + clip_name
                angles_2D.at[patient_id, col] = int(angle_2D)
                angles_3D_d.at[patient_id, col] = int(angle_3D_d)
                angles_3D_mp.at[patient_id, col] = int(angle_3D_mp)

    # Write the raw angles in csv files
    angles_2D.to_csv(os.path.join(results_dir, "angles_2D.csv"))
    angles_3D_d.to_csv(os.path.join(results_dir, "angles_3D_d.csv"))
    angles_3D_mp.to_csv(os.path.join(results_dir, "angles_3D_mp.csv"))