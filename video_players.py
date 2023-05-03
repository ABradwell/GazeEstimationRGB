"""
    Author: Aiden Stevenson Bradwell
    Date: 2021-11-19
    Affiliation: University of Ottawa, Ottawa, Ontario (Student)

    Description:
        Declare two class...
            VideoGetter: Read frames from webcam
            VideoShower: Display frames after being filtered

    Libraries required:
        N/A
"""

from threading import Thread
import cv2


class VideoGetter:
    """ Read frames from webcam """

    def __init__(self, src):
        self.camera = src
        (self.grabbed, self.frame) = self.camera.read()
        self.stopped = False
        self.frame_queue = []

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:

            if not self.grabbed:
                self.stop()
            else:

                (self.grabbed, cur_frame) = self.camera.read()
                self.frame_queue.append(cur_frame)

    def stop(self):
        self.stopped = True


class VideoShower:
    """ Display frames after being filtered """

    def __init__(self, frame_queue, gui_frame):
        self.frame_queue = frame_queue
        self.gui = gui_frame
        self.stopped = False
        self.cur_image = None
        self.panel = None
        self.flip = True

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):

        while not self.stopped:

            cv2.waitKey(10)

            # if the panel is not None, we need to initialize it
            if len(self.frame_queue) > 0:
                image = self.frame_queue.pop()
                if self.flip:
                    image = cv2.flip(image, 0)
                cv2.imshow("Video Feed...", image)

    def stop(self):
        self.stopped = True
