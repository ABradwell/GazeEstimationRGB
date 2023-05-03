
import argparse
import threading
import cv2
from queue import Queue
import pygame
import numpy as np
import datetime
from CONSTANTS import *
from Tracker.ptgaze import GazeEstimator, GazeEstimationMethod
from Tracker.ptgaze import Face, FacePartsName, FaceParts
from ...ptgaze import get_default_config
from ...ptgaze import update_default_config, update_config
from .. import settings
from .baseeyetracker import BaseEyeTracker
from pygame.locals import *
import math
import os


class CamEyeTracker(BaseEyeTracker):

    """The CamEyeTracker class uses your webcam as an eye tracker"""

    def __init__(self, logfile=None,
                 eventdetectionfile=None,
                 eventdetection=settings.EVENTDETECTION, saccade_velocity_threshold=35,
                 saccade_acceleration_threshold=9500, blink_threshold=settings.BLINKTHRESH, **args):

        super().__init__()

        if logfile is None:
            logfile = "output/GAZE_DATA/p{}_gaze_data.txt".format(PATIENT_ID)
        if eventdetectionfile is None:
            eventdetectionfile = 'output/GAZE_DATA/p{}_gaze_detection.txt'.format(PATIENT_ID)



        self.frames = Queue()

        self.cap = cv2.VideoCapture(0)
        self.status, self.frame = self.cap.read()  # Take an initial frame to load the cameras attributes
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))  # Full fps
        width, height = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.size = (int(width), int(height))
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Format of video encoding

        # This thread saves the new frames given by the previous thread
        self.lock = threading.Lock()  # Prevents reading and simultanious writing
        self.display = pygame.display.get_surface()
        self.dispsize = pygame.display.get_surface().get_size()

        config = self.get_config()
        self.config = config
        self.gaze_estimator = GazeEstimator(config)
        self.eye_used = "LEFT"

        self.draw_calibration_target = None
        self.draw_drift_correction_target = None

        self.pointAngles = {
            "topLeft": (0, 0),
            "bottomLeft": (0, 0),
            "topRight": (0, 0),
            "bottomRight": (0, 0),
            "topMid": (0, 0),
            "bottomMid": (0, 0),
            "leftMid": (0, 0),
            "rightMid": (0, 0),
            "center": (0, 0)
        }

        self.left_yaw = -1
        self.right_yaw = -1

        self.top_pitch = -1
        self.bottom_pitch = -1

        self.mid_pitch = -1
        self.mid_yaw = -1

        self.fps = FPS
        self.clock = pygame.time.Clock()

        self.x = -1
        self.y = -1

        location = VIDEO_OUTPUT_LOCATION
        self.out = cv2.VideoWriter(location, self.fourcc, 30, self.size)

        self._get_frame_thread = threading.Thread(target=self.read, args=())
        self.recording = False
        self._full_is_retrieving = False
        self.record_play = False
        self.retrieve_play = False
        self.isconnected = True

        self.log_file = open(logfile, 'w+')
        self.detection_file = open(eventdetectionfile, 'w+')

        # Section for gaze analysis
        self.frameCount = 0
        self.event = ""
        self.start_x = -1
        self.start_y = -1
        self.start_time = 0
        self.start_frame = 0

        self.last_x = -1
        self.last_y = -1
        self.last_time = 0

        self.fixation_threshold = FIXATION_THRESH
        self.velocity_threshold = VELOCITY_THRESH_ACC
        self.newevent = True

        self.video_start_time = None

        self._write_headers()
        self.calibrate()


    def calibrate(self):

        """   """

        calibration_file = open('{}p{}_webcam_calibration_log.txt'.format(WEBCAM_CALIBRATION_LOCATION, PATIENT_ID),'w+')
        calibration_file.write("Corner, \tTimestamp-start, \tTimestamp-end\n")
        horMid = self.dispsize[0] / 2 - 13
        verMid = self.dispsize[1] / 2 - 13
        right = self.dispsize[0] - 25
        bottom = self.dispsize[1] - 25

        calibration_rects = [(0, 0, 25, 25), (0, self.dispsize[1] - 25, 25, 25),
                             (self.dispsize[0], 0, -25, 25), (self.dispsize[0], self.dispsize[1], -25, -25),
                             (horMid, 0, 25, 25), (horMid, bottom, 25, 25),
                             (0, verMid, 25, 25), (right, verMid, 25, 25),
                             ((self.dispsize[0] - 25) / 2, (self.dispsize[1] - 25) / 2, 25, 25)]

        corners = ["topLeft", "bottomLeft",
                   "topRight", "bottomRight",
                   "topMid", "bottomMid",
                   "leftMid", "rightMid", "center"]

        for rect in range(len(calibration_rects)):
            corner = corners[rect]
            calibration_box = calibration_rects[rect]

            line = "LOOK AT THE GREEN SQUARE & PRESS SPACE"

            x = self.dispsize[0] / 2
            y = self.dispsize[1] - 50

            font = pygame.font.Font(None, FONTSIZE)
            rendered = font.render(line, True, COLOR_TEXT)
            self.display.blit(rendered, (x, y))

            mark = pygame.Rect(calibration_box[0], calibration_box[1], calibration_box[2], calibration_box[3])
            pygame.draw.rect(self.display, GREEN, mark)
            pygame.display.update()

            while True:

                # self.wait_time_ms(CALIBRATION_PAUSE)
                self.wait_space_key()
                print("KEYPRESS")
                start_time = self.create_timestamp()
                stats = [[], []]

                for i in range(1):
                    ret, frame = self.cap.read()
                    tempPitch, temYaw = self.get_pw(frame)
                    stats[0].append(tempPitch)
                    stats[1].append(temYaw)

                    # TODO This only save the first frame out of X taken
                    filename = "{}p{}_{}.jpg".format(WEBCAM_CALIBRATION_LOCATION, PATIENT_ID, corners[rect])
                    cv2.imwrite(filename, frame)

                end_time = self.create_timestamp()
                if len(stats[0]) > 0 and None not in stats[0]:
                    self.pointAngles[corner] = ((np.sum(np.array(stats[0])) / len(stats[0])),
                                                (np.sum(np.array(stats[1])) / len(stats[1])))
                    break

            calibration_file.write("{} \t{} \t{}".format(corners[rect], start_time, end_time) + '\n')

            self.display.fill(COLOR_BACKGROUND)

        for a in self.pointAngles.keys():
            if self.pointAngles[a][0] is None:
                self.calibrate()
                break

        self.process_edges()

    def process_edges(self):

        tlPitch, tlYaw = self.pointAngles["topLeft"]
        trPitch, trYaw = self.pointAngles["topRight"]
        blPitch, blYaw = self.pointAngles["bottomLeft"]
        brPitch, brYaw = self.pointAngles["bottomRight"]
        tmPitch, tmYaw = self.pointAngles["topMid"]
        bmPitch, bmYaw = self.pointAngles["bottomMid"]
        lmPitch, lmYaw = self.pointAngles["leftMid"]
        rmPitch, rmYaw = self.pointAngles["rightMid"]
        cPitch, cYaw = self.pointAngles["center"]

        # Now adjust the mid points based on the center

        self.left_yaw = (tlYaw + blYaw + lmYaw) / 3
        self.right_yaw = (trYaw + brYaw + rmYaw) / 3

        self.top_pitch = (tlPitch + trPitch + tmPitch) / 3
        self.bottom_pitch = (blPitch + brPitch + bmPitch) / 3

        self.mid_yaw = (self.left_yaw + self.right_yaw) / 2
        self.mid_pitch = (self.top_pitch + self.bottom_pitch) / 2

        self.pointAngles["topLeft"] = (self.top_pitch, self.left_yaw)
        self.pointAngles["topRight"] = (self.top_pitch, self.right_yaw)
        self.pointAngles["bottomLeft"] = (self.bottom_pitch , self.left_yaw)
        self.pointAngles["bottomRight"] = (self.bottom_pitch , self.right_yaw)
        self.pointAngles["topMid"] = (self.top_pitch , self.mid_yaw)
        self.pointAngles["bottomMid"] = (self.bottom_pitch, self.mid_yaw)
        self.pointAngles["leftMid"] = (self.mid_pitch, self.left_yaw)
        self.pointAngles["rightMid"] = (self.mid_pitch, self.right_yaw)
        self.pointAngles["center"] = (self.mid_pitch, self.mid_yaw)

    def close(self):

        self.out.release()
        return self.connected()

    def connected(self):
        self.isconnected = self.cap.isOpened()
        return self.isconnected

    def drift_correction(self, pos=None, fix_triggered=True):

        if pos is not None:
            self.draw_drift_correction_target(pos[0], pos[1])
        else:
            centerx = self.dispsize[0]/2
            centery = self.dispsize[1]/2
            self.draw_drift_correction_target(centerx, centery)

    def fix_triggered_drift_correction(self, pos=None, min_samples=30, max_dev=60, reset_threshold=10):

        self.retrieve_play = False
        self.record_play = False
        run_correction = True

        while run_correction:

            errkeys = []
            total_err = 0
            err = 0
            last_point = pos

            for i in range(min_samples):
                (_, frame) = self.cap.read()  # Pull new frame from webcam
                (x, y) = self.get_coords(frame)

                # xDif = pos[0] - x
                # yDif = pos[1] - y

                err = math.hypot((pos[0] - x), (pos[1] - y))
                if i > 1:
                    if abs(last_point[0] - x) > reset_threshold or abs(last_point[1] - y) > reset_threshold:
                        print("RESET DUE TO MAX DEVIATION")
                        break

                last_point = (x, y)
                total_err += err

            run_correction = False
            err = total_err/min_samples

            self.retrieve_play = True
            self.record_play = True
            return err <= max_dev

    def get_eyetracker_clock_async(self):
        """
        desc:
            Returns the difference between tracker time and PyGaze time, which
            can be used to synchronize timing

        returns:
            desc:    The difference between eyetracker time and PyGaze time.
            type:    [int, float]
        """

        pass

    def log(self, msg):

        if len(msg) is 3:
            outLine = "{}, {}, {} \n".format(str(msg[0]), str(msg[1]), str(msg[2]))
            self.log_file.write(outLine)
        else:
            raise SyntaxWarning

    def log_var(self, var, val):

        pass

    def pupil_size(self):

        return -1

    def send_command(self, cmd):
        raise SyntaxWarning("No commands can be sent to a webcam tracker")

    def set_eye_used(self):
        """
        desc:
            Logs the `eye_used` variable, based on which eye was specified
            (if both eyes are being tracked, the left eye is used). Does not
            return anything.
        """

        self.eye_used = "LEFT"

    def draw_drift_correction_target(self, x, y):

        target = pygame.Rect(x, y, 25, 25)
        pygame.draw.rect(self.display, WHITE, target)

    def draw_calibration_target(self, x, y):

        target = pygame.Rect(x, y, 25, 25)
        pygame.draw.rect(self.display, GREEN, target)

    def set_draw_calibration_target_func(self, func):

        self.draw_calibration_target = func

    def set_draw_drift_correction_target_func(self, func):

        self.draw_drift_correction_target = func

    def start_recording(self):

        self.retrieve_play = True
        self._full_is_retrieving = True
        if self._get_frame_thread.ident is None:
            self._get_frame_thread.start()

        self.isconnected = True

    def status_msg(self, msg):

        pass

    def stop_recording(self):

        if self._get_frame_thread.is_alive():
            self.retrieve_play = False
            self._full_is_retrieving = False

        self.cap.release()
        self._get_frame_thread.join()
        self.close()

    def wait_for_event(self, event):

        if event == 3:
            self.wait_for_blink_start()
        elif event == 4:
            self.wait_for_blink_end()
        elif event == 5:
            self.wait_for_saccade_start()
        elif event == 6:
            self.wait_for_saccade_end()
        elif event == 7:
            self.wait_for_fixation_start()
        elif event == 8:
            self.wait_for_fixation_end()


    def wait_for_blink_end(self):
        """
        desc: |
            Waits for a blink end and returns the blink ending time.
            Detection based on Dalmaijer et al. (2013) if EVENTDETECTION is set
            to 'pygaze', or using native detection functions if EVENTDETECTION
            is set to 'native' (NOTE: not every system has native functionality;
            will fall back to ;pygaze' if 'native' is not available!)

        returns:
            desc:    Blink ending time in milliseconds, as measured from
                    experiment begin time.
            type:    [int, float]
        """

        pass

    def wait_for_blink_start(self):
        """
        desc: |
            Waits for a blink start and returns the blink starting time.
            Detection based on Dalmaijer et al. (2013) if EVENTDETECTION is set
            to 'pygaze', or using native detection functions if EVENTDETECTION
            is set to 'native' (NOTE: not every system has native functionality;
            will fall back to ;pygaze' if 'native' is not available!)

        returns:
            desc:     Blink starting time in milliseconds, as measured from
                    experiment begin time
            type:    [int, float]
        """

        pass

    def wait_for_fixation_end(self):
        """
        desc: |
            Returns time and gaze position when a fixation has ended;
            function assumes that a 'fixation' has ended when a deviation of
            more than self.pxfixtresh from the initial fixation position has
            been detected (self.pxfixtresh is created in self.calibration,
            based on self.fixtresh, a property defined in self.__init__).
            Detection based on Dalmaijer et al. (2013) if EVENTDETECTION is set
            to 'pygaze', or using native detection functions if EVENTDETECTION
            is set to 'native' (NOTE: not every system has native functionality;
            will fall back to ;pygaze' if 'native' is not available!)

        returns:
            desc:     A `time, gazepos` tuple. Time is the end time in
                    milliseconds (from expstart), gazepos is a (x,y) gaze
                    position tuple of the position from which the fixation was
                    initiated.
            type:    tuple
        """

        pass

    def wait_for_fixation_start(self):
        """
        desc: |
            Returns starting time and position when a fixation is started;
            function assumes a 'fixation' has started when gaze position
            remains reasonably stable (i.e. when most deviant samples are
            within self.pxfixtresh) for five samples in a row (self.pxfixtresh
            is created in self.calibration, based on self.fixtresh, a property
            defined in self.__init__).
            Detection based on Dalmaijer et al. (2013) if EVENTDETECTION is set
            to 'pygaze', or using native detection functions if EVENTDETECTION
            is set to 'native' (NOTE: not every system has native functionality;
            will fall back to ;pygaze' if 'native' is not available!)

        returns:
            desc:     A `time, gazepos` tuple. Time is the starting time in
                    milliseconds (from expstart), gazepos is a (x,y) gaze
                    position tuple of the position from which the fixation was
                    initiated.
            type:    tuple
        """

        pass

    def wait_for_saccade_end(self):
        """
        desc: |
            Returns ending time, starting and end position when a saccade is
            ended; based on Dalmaijer et al. (2013) online saccade detection
            algorithm if EVENTDETECTION is set to 'pygaze', or using native
            detection functions if EVENTDETECTION is set to 'native' (NOTE: not
            every system has native functionality; will fall back to ;pygaze'
            if 'native' is not available!)

        returns:
            desc:    An `endtime, startpos, endpos` tuple. Endtime in
                    milliseconds (from expbegintime); startpos and endpos
                    are (x,y) gaze position tuples.
            type:    tuple
        """

        pass

    def wait_for_saccade_start(self):
        """
        desc: |
            Returns starting time and starting position when a saccade is
            started; based on Dalmaijer et al. (2013) online saccade detection
            algorithm if EVENTDETECTION is set to 'pygaze', or using native
            detection functions if EVENTDETECTION is set to 'native' (NOTE: not
            every system has native functionality; will fall back to ;pygaze'
            if 'native' is not available!)

        returns:
            desc:    An `endtime, startpos` tuple. Endtime in milliseconds (from
                    expbegintime); startpos is an (x,y) gaze position tuple.
            type:    tuple
        """

        pass

    def get_pw(self, frame) -> (float, float):
        return self.process_image(frame)

    def process_image(self, image, get_pw=True, get_raw=False) -> (float, float):
        if image is not None:
            faces = self.gaze_estimator.detect_faces(image)
            for face in faces:

                if self.check_blinking(face):
                    return None, None

                self.gaze_estimator.estimate_gaze(image, face)
                print(self.gaze_estimator.get_distance(face))

                if get_pw and get_raw:
                    return self.get_gaze_vector(face), self.get_raw_vector(face)
                elif get_pw:
                    return self.get_gaze_vector(face)
                elif get_raw:
                    return self.get_raw_vector(face)

        return None, None

    def update_position(self, x, y):
        self.x = x
        self.y = y

    def get_coords(self, frame) -> (float, float):
        (pitch, yaw) = self.process_image(frame)
        if pitch is not None:
            return self.pw_to_coords(pitch, yaw)
        return None, None

    def pw_to_coords(self, pitch, yaw):

        lowest_pitch, highest_pitch = min(self.top_pitch, self.bottom_pitch), max(self.top_pitch, self.bottom_pitch)
        lowest_yaw, highest_yaw = min(self.left_yaw, self.right_yaw), max(self.left_yaw, self.right_yaw)

        a = (pitch - lowest_pitch)
        b = (highest_pitch - lowest_pitch)
        c = (yaw - lowest_yaw)
        d = (highest_yaw - lowest_yaw)

        x = (c / d) * self.dispsize[0]
        y = self.dispsize[1] - (a / b) * self.dispsize[1]

        return x, y

    def sample(self):
        return self.x, self.y

    def read(self):

        """Pull new frame from the webcam, and add it to the frame queue"""

        self.video_start_time = self.create_timestamp()

        while self._full_is_retrieving:
            if self.retrieve_play and self.cap.isOpened():
                (_, frame) = self.cap.read()  # Pull new frame from webcam
                time = self.create_timestamp()
                self.frames.put([frame, time])  # Add to frame queue
                print(self.frames.qsize())

        self.record_frames()

    def record_frames(self):

        """Call frame-gather thread, and store given frame to output file"""

        while self.frames.qsize() > 0:
            print(self.frames.qsize())
            [frame, time] = self.frames.get()
            print("NEW TIME ON GAZE PULLED at {}".format(time))
            self.out.write(frame)
            (self.x, self.y) = self.get_coords(frame)
            self.log([self.x, self.y, time])

            self.frameCount += 1

            cur_x = self.x
            cur_y = self.y
            cur_time = time

            if self.newevent is True:
                if cur_x is not None or cur_y is not None:

                    self.start_x = cur_x
                    self.start_y = cur_y
                    self.start_time = cur_time
                    self.start_frame = self.frameCount

                    self.last_x = self.start_x
                    self.last_y = self.start_y
                    self.last_time = self.start_time

                self.newevent = False

            else:

                if cur_x is None or cur_y is None:
                    if self.event is not "BLINKING":
                        frame_duration = self.frameCount - self.start_frame
                        time_duration = self.time_difference(self.start_time, cur_time)
                        self.detection_file.write(
                            '\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{}'.format(self.start_x, self.start_y,
                                                                                  self.start_time, cur_x,
                                                                                  cur_y, cur_time,
                                                                                  self.event, frame_duration,
                                                                                  time_duration) + '\n')

                        # Write buffer to the datafile
                        self.event = "BLINKING"
                        self.newevent = True

                else:
                    dist = math.hypot((cur_x - self.last_x), (cur_y - self.last_y))
                    x_velocity = (cur_x - self.last_x) / self.time_difference(cur_time, self.last_time)
                    y_velocity = (cur_y - self.last_y) / self.time_difference(cur_time, self.last_time)
                    velocity = (x_velocity, y_velocity)

                    if dist < self.fixation_threshold:
                        if self.event is not "FIXATION":
                            frame_duration = self.frameCount - self.start_frame
                            time_duration = self.time_difference(self.start_time, cur_time)
                            self.detection_file.write(
                                '\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{}'.format(self.start_x, self.start_y,
                                                                                      self.start_time, cur_x,
                                                                                      cur_y, cur_time,
                                                                                      self.event, frame_duration,
                                                                                      time_duration) + '\n')
                            self.event = "FIXATION"
                            self.newevent = True

                        else:
                            self.last_x = cur_x
                            self.last_y = cur_y
                            self.last_time = cur_time

                    elif velocity[0] >= self.velocity_threshold or velocity[1] >= self.velocity_threshold:
                        if self.event is not "SACCADE":
                            frame_duration = self.frameCount - self.start_frame
                            time_duration = self.time_difference(self.start_time, cur_time)
                            self.detection_file.write(
                                '\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{}'.format(self.start_x, self.start_y,
                                                                                       self.start_time, cur_x,
                                                                                       cur_y, cur_time,
                                                                                       self.event, frame_duration,
                                                                                       time_duration) + '\n')
                            self.event = "SACCADE"
                            self.newevent = True

                        else:
                            self.last_x = cur_x
                            self.last_y = cur_y
                            self.last_time = cur_time
                    else:
                        self.newevent = True
                        frame_duration = self.frameCount - self.start_frame
                        time_duration = self.time_difference(self.start_time, cur_time)
                        self.detection_file.write(
                            '\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{}'.format(self.start_x, self.start_y,
                                                                                  self.start_time, cur_x,
                                                                                  cur_y, cur_time,
                                                                                  self.event, frame_duration,
                                                                                  time_duration) + '\n')

    @staticmethod
    def wait_time_ms(ms):
        pygame.time.delay(ms)

    @staticmethod
    def wait_time_seconds(sec):
        pygame.time.delay(sec * 1000)

    @staticmethod
    def create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d%H%M%S.%f')

    def get_gaze_vector(self, face: Face):
        if self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))

        elif self.config.mode == GazeEstimationMethod.MPIIGaze.name:
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

    @staticmethod
    def get_raw_vector(face: Face):
        return face.gaze_vector

    @staticmethod
    def time_difference(t1, t2):
        t1 = float(t1[-9:].strip())
        t2 = float(t2[-9:].strip())
        if abs(t2 - t1) == 0:
            return 0.0000001
        return abs(t2 - t1)

    @staticmethod
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
                 'MPIIFaceGaze model will be used. (default: \'eye\')')
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

    @staticmethod
    def wait_space_key():
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                if event.type == KEYDOWN and event.key == K_SPACE:
                    return

    def check_blinking(self, face):
        face.either_blinking()

    def _flush_to_file(self):
        # write data to disk
        self.log_file.flush()  # internal buffer to RAM
        os.fsync(self.log_file.fileno())  # RAM file cache to disk

    def _write_headers(self):

        self.log_file.write(',\t'.join(['X Coordinate',
                                        'Y Coordinate',
                                        'Timestamp']) + '\n')

        self.detection_file.write(',\t'.join(['Start LocationX',
                                              'Start LocationY',
                                              'Start Time',
                                              'End LocationX',
                                              'End LocationY',
                                              'End Time',
                                              'Event',
                                              'Frame Duration',
                                              'Time Duration']) + '\n')
        self._flush_to_file()
