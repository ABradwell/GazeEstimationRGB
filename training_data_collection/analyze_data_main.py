import argparse
import numpy as np

from Tracker.ptgaze import GazeEstimator, GazeEstimationMethod
from Tracker.ptgaze import Face, FacePartsName, FaceParts
from pygame.locals import *
from Tracker.ptgaze import get_default_config
from Tracker.ptgaze import update_default_config, update_config
import math
import os
import cv2

import dlib
import numpy as np

# General Concept:
"""
    For each subset of data, calculate the direction the user is looking, and store in a text file specific to that subset. 
"""


def dist_between_points(p1, p2):

    x1 = p1.x
    y1 = p1.y

    x2 = p2.x
    y2 = p2.y

    max_x = max(x1, x2)
    min_x = min(x1, x2)
    max_y = max(y1, y2)
    min_y = min(y1, y2)

    distance = math.sqrt(abs(math.pow(max_x - min_x, 2) + math.pow(max_y - min_y, 2)))

    return distance


def landmarks_to_list(landmarks):
    int_coords = []
    for i in range(0, 68):
        int_coords.append((landmarks.part(i).x, landmarks.part(i).y))

    return int_coords


def get_raw_vector(face: Face):
    return face.gaze_vector


def check_blinking(face):
    # Return false if blinking, and true if not blinking
    good_to_measure = face.either_blinking()
    print("Either blinking returning {}\n".format(good_to_measure))

    return not good_to_measure


def dlib_process_image(image, face_det, land_det):
    # Return width of face, height of face, distance between eyes, and bounding square around head.
    # (bool, float, float, float, (float, float, float, float))

    faces_detected = face_det(image)

    if len(faces_detected) == 0:
        print("NO FACES DETECTED. IMAGE BEING DISCARDED.")
        return False, (-1, -1, -1, -1), [], -1, -1, -1

    elif len(faces_detected) > 1:
        print("TOO MANY FACES DETECTED. DISREGARD IMAGE.")
        return False, (-1, -1, -1, -1), [], -1, -1, -1

    face_rect = dlib.rectangle(
        int(faces_detected[0].left()),
        int(faces_detected[0].top()),
        int(faces_detected[0].right()),
        int(faces_detected[0].bottom())
    )

    landmarks_detected = land_det(image, face_rect)

    top_of_nose = landmarks_detected.part(27)
    bottom_of_jaw = landmarks_detected.part(8)

    left_cheek = landmarks_detected.part(0)
    right_cheek = landmarks_detected.part(16)

    face_height = dist_between_points(top_of_nose, bottom_of_jaw)
    face_width = dist_between_points(left_cheek, right_cheek)

    inner_left_eye = landmarks_detected.part(39)
    inner_right_eye = landmarks_detected.part(42)
    eye_distance = dist_between_points(inner_left_eye, inner_right_eye)

    return True, face_rect, landmarks_to_list(landmarks_detected), eye_distance, face_height, face_width


def process_image(image, get_pw=True, get_raw=False) -> (float, float):
    if image is not None:
        faces = gaze_estimator.detect_faces(image)
        for face in faces:

            # if check_blinking(face):
            #     print("Either blinking returning True\n")
            #     return None, None

            gaze_estimator.estimate_gaze(image, face)
            # print(gaze_estimator.get_distance(face))

            if get_pw and get_raw:
                return get_gaze_vector(face), get_raw_vector(face)
            elif get_pw:
                return get_gaze_vector(face)
            elif get_raw:
                return get_raw_vector(face)

    return None, None


def get_config():
    """Create the tracking device using the PtGaze tracker modified"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='Config file for YACS. When using a config file, all the other '
                             'commandline arguments are ignored. '
                             'See https://github.com/hysts/pytorch_mpiigaze_demo/configs/demo_mpiigaze.yaml')
    parser.add_argument(
        '--mode', type=str, default='face', choices=['eye', 'face'],
        help='With \'eye\', MPIIGaze model will be used. With \'face\', '
             'MPIIFaceGaze model will be used. (default: \'face\')')
    parser.add_argument(
        '--face-detector', type=str, default='dlib', choices=['dlib', 'face_alignment_dlib', 'face_alignment_sfd'],
        help='The method used to detect faces and find face landmarks (default: \'dlib\')')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Device used for model inference.')
    parser.add_argument('--image', type=str, help='Path to an input image file.')
    parser.add_argument('--video', type=str, help='Path to an input video file.')
    parser.add_argument('--camera', type=str, help='Camera calibration file. '
                                                   'See https://github.com/hysts/pytorch_mpiigaze_demo/ptgaze/data/calib/sample_params.yaml')
    parser.add_argument('--output-dir', '-o', type=str,
                        help='If specified, the overlaid video will be saved to this directory.')
    parser.add_argument('--ext', '-e', type=str, choices=['avi', 'mp4'], help='Output video file extension.')
    parser.add_argument('--no-screen', action='store_true',
                        help='If specified, the video is not displayed on screen, and saved '
                             'to the output directory.')
    args = parser.parse_args()
    config = get_default_config()

    if args.config:
        config.merge_from_file(args.config)
    else:
        update_default_config(config, args)

    update_config(config)
    return config


def get_gaze_vector(face: Face):
    if config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
        pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))

    elif config.mode == GazeEstimationMethod.MPIIGaze.name:
        pitches = []
        yaws = []
        for key in [FacePartsName.REYE, FacePartsName.LEYE]:
            eye = getattr(face, key.name.lower())
            curpitch, curyaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
            pitches.append(curpitch)
            yaws.append(curyaw)
        pitch = (np.sum(np.array(pitches)) / len(pitches))
        yaw = (np.sum(np.array(yaws)) / len(yaws))

    else:
        raise ValueError
    return [pitch, yaw]


def get_coords(frame) -> (float, float):
    pass


if __name__ == "__main__":
    config = get_config()
    gaze_estimator = GazeEstimator(config)

    # location of the model (path of the model).
    model_PATH = "trained_networks/shape_predictor_68_face_landmarks.dat"
    frontalFaceDetector = dlib.get_frontal_face_detector()
    faceLandmarkDetector = dlib.shape_predictor(model_PATH)

    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, "data")
    all_file_names = os.listdir(data_dir)

    with open("data/train_file.txt", "w") as out:

        # TODO: GENERALIZE
        num_horizontal_classifications = 4
        num_vertical_classifications = 4
        number_of_samples_to_take = 12
        num_data_per_calib = 10

        out.write("HOR:{}\n".format(num_horizontal_classifications))
        out.write("VERT:{}\n".format(num_vertical_classifications))
        out.write("SAMPLES:{}\n".format(number_of_samples_to_take))
        out.write("DATA:{}\n\n".format(num_data_per_calib))

        calib_tag = ["A", "B", "C", "D"]
        for sample_num in range(number_of_samples_to_take):
            print("\t\tSample {}".format(sample_num))

            calib_frames = []
            data_frames = []

            data_image_names = [name for name in all_file_names if
                                (name.replace("_", " ").strip().split()[0] == str(sample_num)) and ("data" in name)]
            calib_image_names = [name for name in all_file_names if
                                 (name.replace("_", " ").strip().split()[0] == str(sample_num)) and ("calib" in name)]

            print("\n\nNEW CALIB: {} ({})".format(sample_num, calib_image_names))
            print("NEW DATA: {} ({})\n\n".format(sample_num, data_image_names))

            for calib_name in calib_image_names:
                calib_name = "data/{}".format(calib_name)
                print("\t\t\t\tCALIB FILE {}".format(calib_name))

                calib_attributes = calib_name.replace("_", " ").replace(".jpg", "").strip().split()

                frame = cv2.imread(calib_name, cv2.IMREAD_COLOR)
                (pitch, yaw) = process_image(frame)
                face_found, face_rect, landmarks_detected, eye_distance, face_height, face_width = dlib_process_image(
                    frame, frontalFaceDetector, faceLandmarkDetector)

                if not face_found:
                    print("No Face on calibration, skipping data set.")
                    continue

                if pitch is None:
                    print("No Pitch Found on calibration. Blinking. Skipping data set.")
                    continue

                calib_frames.append((frame, pitch, yaw))
                out.write("C{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\n".format(sample_num, calib_attributes[2], pitch, yaw,
                                                                         eye_distance, face_height, face_width,
                                                                         face_rect, landmarks_detected))

            for data_name in data_image_names:
                data_name = "data/{}".format(data_name)
                print("\t\t\t\tSub Sample {}".format(data_name))

                data_atrributes = data_name.replace("_", " ").replace(".jpg", "").strip().split()

                frame = cv2.imread(data_name, cv2.IMREAD_COLOR)
                data_frames.append(frame)
                (pitch, yaw) = process_image(frame)
                face_found, face_rect, landmarks_detected, eye_distance, face_height, face_width = dlib_process_image(
                    frame, frontalFaceDetector, faceLandmarkDetector)

                if not face_found:
                    print("No Face, skipping data point.")
                    continue

                if pitch is None:
                    print("No Pitch Found. Blinking. Skipping data point.")
                    continue

                data_frames.append((frame, pitch, yaw))
                out.write(
                    "D{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\n".format(sample_num, data_atrributes[3], data_atrributes[4],
                                                                   pitch, yaw, eye_distance, face_height, face_width,
                                                                   face_rect, landmarks_detected))
