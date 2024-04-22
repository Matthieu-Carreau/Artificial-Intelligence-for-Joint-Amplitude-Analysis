Artificial Intelligence for Joint Amplitude Analysis 

This repository contains the code for the project "Intelligence Articulaire pour l'Analyse des Amplitudes Articulaire" (IAAAA), made with Jules Descamps, Timothée Maire and Valerian Fiodiere from Hôpital Lariboisière, during a module of the MVA master.

The objective is to estimate angular amplitude of articular motions of the leg using RGB-D videos.
We extract time series of landmarks from the videos using Mediapipe Pose estimation models.


# Installation

1. Install the following python libraries:
- numpy 
- matplotlib
- pandas
- scipy
- mediapipe
- pyrealsense2
- sklearn (only required for the affine calibration)


2. Download one of the mediapipe landmarker models, the one used in the tests was:

https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

Liter models are also available at https://developers.google.com/mediapipe/solutions/vision/pose_landmarker . 


3. Organize your bag recordings with RGB-D data in separated folders for each patient, the name of the folder should be an integer identifying the patient.
Each folder should contain 12 bag files named with the convention "id_patient side angle.bag", for example: "1 L ABD.bag", where *id_patient* is the integer patient identifier, *side* is "R" or "L" and *angle* is among the following list: 
- "ABD" for abduction, 
- "ADD" for adduction, 
- "EXT" for extension, 
- "FLX" for flexions, 
- "RI" for Rotation Interne,
- "RE" for Rotation Externe.


4. Specify the input and output paths in the main.py file (4 paths to specify before the "TO MODIFY" comments, according to the instructions).

5. Run the file main.py. 6 files will be created at the root of the results folder.
First, the three following files contain the angles estimated using respectively the 2D image only, the depth information from the RGB-D camera, and the depth estimation from the mediapipe model (without using the real depth information).
- angles_2D.csv
- angles_3D_d.csv
- angles_3D_mp.csv

Then, the three last files contain the result after the affine calibration with the leave-one-out strategy:
- angles_2D_calib.csv
- angles_3D_d_calib.csv
- angles_3D_mp_calib.csv

