from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import pyrealsense2 as rs
import matplotlib.pyplot as plt


def get_detector(model_path):
    # Load the mediapipe pose estimation model
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
    return detector


def init_bag_reader(bag_file_path):
    # Create the bag reader object
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file_path)
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)
    cfg = pipeline.start(config)
    profile = cfg.get_stream(rs.stream.color)
    playback = cfg.get_device().as_playback()
    playback.set_real_time(False)
    playback.pause()

    return pipeline, playback, profile.as_video_stream_profile().get_intrinsics()


def np_landmarks(landmarks):
    # Create a numpy array with the coordinates from the landmarks object
    array = np.zeros((len(landmarks), 3))
    for i, lm in enumerate(landmarks):
       array[i, 0] = lm.x
       array[i, 1] = lm.y
       array[i, 2] = lm.z
       
    return array


def draw_landmarks_on_image(rgb_image, detection_result):
    # Plot an image with the landmarks
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def process_bag_file(bag_file_path, model_path, plot=False, step=20, max_frames=10**5, output_path=None, print_time=False):
    """
    Process a bag file to extract landmarks time series
    """
    detector = get_detector(model_path)
    pipeline, playback, intrinsics = init_bag_reader(bag_file_path)
    align = rs.align(rs.stream.color)
    n_frames = 0
    start_time = 0
    timestamp = 0
    last_timestamp = 0
    raw_landmarks = []
    raw_world_landmarks = []
    corrected_landmarks = []
    timestamps = []

    while n_frames < max_frames:
        playback.resume()
        frames = pipeline.wait_for_frames()
        playback.pause()

        timestamp = frames.get_timestamp()
        aligned_frames = align.process(frames)
        
        if n_frames == 0:
            start_time = timestamp
        
        if timestamp < last_timestamp:
            print("End of the recording")
            break

        last_timestamp = timestamp

        if n_frames % step == 0:
            depth = aligned_frames.get_depth_frame()

            color = np.array(aligned_frames.get_color_frame().get_data())

            rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=color)
            try:
                detection_result = detector.detect(rgb_frame)
            except:
                print("Detection impossible")
            
            landmarks = np_landmarks(detection_result.pose_landmarks[0])
            world_landmarks = np_landmarks(detection_result.pose_world_landmarks[0])
            raw_landmarks.append(landmarks)
            raw_world_landmarks.append(world_landmarks)
            timestamps.append((timestamp - start_time) / 1000)

            # Correct landmarks
            corrected_lm = np.zeros(landmarks.shape)
            h, w, _ = color.shape
            for i, lm in enumerate(landmarks):
                pix_x = np.clip(int(np.round(lm[0]*w)), 0, w-1)
                pix_y = np.clip(int(np.round(lm[1]*h)), 0, h-1)
                dist = depth.get_distance(pix_x, pix_y)
                
                if dist == 0:
                    pass

                point = rs.rs2_deproject_pixel_to_point(intrinsics, [pix_x, pix_y], dist)
                corrected_lm[i] = point

            corrected_landmarks.append(corrected_lm)
                    
            if plot:
                colorizer = rs.colorizer()
                depth_image = colorizer.colorize(depth).get_data()

                segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
                visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) 
                print(np.min(visualized_mask), np.max(visualized_mask))
                mask = (0.4 + visualized_mask) / 1.4 / 255

                print(np.min(mask), np.max(mask))

                annotated_image = draw_landmarks_on_image(color, detection_result)
                plt.imshow(annotated_image * mask)
                plt.show()
                print(np.shape(annotated_image), np.shape(mask), np.shape(annotated_image * mask))
                print(np.max(annotated_image), np.max(annotated_image * mask))
                
                annotated_depth_image = draw_landmarks_on_image(depth_image, detection_result)
                plt.imshow(annotated_depth_image * mask)
                plt.show()
            
            if print_time:
                print(np.round((timestamp - start_time)/ 1000, 3))

        n_frames += 1
        
    print("Number of frames =", n_frames)

    raw_landmarks = np.array(raw_landmarks)
    raw_world_landmarks = np.array(raw_world_landmarks)
    corrected_landmarks = np.array(corrected_landmarks)
    timestamps = np.array(timestamps)

    if output_path is not None:
        np.save(output_path + "_raw.npy", raw_landmarks)
        np.save(output_path + "_wor.npy", raw_world_landmarks)
        np.save(output_path + "_cor.npy", corrected_landmarks)
        np.save(output_path + "_time.npy", timestamps)

    return timestamps, raw_landmarks, raw_world_landmarks, corrected_landmarks


def process_folder(input_directory_path, output_directory_path, model_path):
    # Process all bag files in one patient folder
    fnames = os.listdir(input_directory_path)

    for fname in fnames:
        fpath = input_directory_path + fname
        print("Processing", fpath)
        output_path = output_directory_path + fname.split(".")[0]
        print("Output", output_path)
        process_bag_file(fpath, 
                         model_path,
                         output_path=output_path, 
                         step=1,
                         plot=False)


def extract_landmarks(patient_ids, data_dir, results_dir, model_path):
    # Extract landmarks from all patient folders
    for patient_id in patient_ids:
        input_folder = data_dir + str(patient_id) + "/"
        output_folder = results_dir + str(patient_id) + "/landmarks/"
        
        if not os.path.exists(results_dir + str(patient_id)):
            os.mkdir(results_dir + str(patient_id))

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
            
        process_folder(input_folder, output_folder, model_path)


