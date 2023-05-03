from typing import Optional

from scipy.spatial import distance as dist
import numpy as np
import math

from .face_parts import FaceParts, FacePartsName
from .eye import Eye


class Face(FaceParts):
    def __init__(self, bbox: np.ndarray, landmarks: np.ndarray):
        super().__init__(FacePartsName.FACE)
        self.bbox = bbox
        self.landmarks = landmarks

        self.reye: Eye = Eye(FacePartsName.REYE)
        self.leye: Eye = Eye(FacePartsName.LEYE)

        self.head_position: Optional[np.ndarray] = None
        self.model3d: Optional[np.ndarray] = None

        self.LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
        self.RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]
        self.EYE_AR_THRESH = 0.2

    @staticmethod
    def change_coordinate_system(euler_angles: np.ndarray) -> np.ndarray:
        return euler_angles * np.array([-1, 1, -1])

    def either_blinking(self):
        """Returns true if the user closes his eyes"""
        left_eye = self.get_eye(self.LEFT_EYE_POINTS)
        right_eye = self.get_eye(self.RIGHT_EYE_POINTS)

        leftear = self.eye_aspect_ratio(left_eye)
        rightear = self.eye_aspect_ratio(right_eye)
        ear = (leftear + rightear) / 2

        if ear < self.EYE_AR_THRESH:

            print("BLINKING...")
            return True

        return False

    def get_eye(self, points):
        eye = []
        for point in points:
            eye.append(self.landmarks[point])
        return eye

    @staticmethod
    def eye_aspect_ratio(eye):
        # compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        return ear
