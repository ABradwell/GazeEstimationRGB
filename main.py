"""
    Author: Aiden Stevenson Bradwell
    Date: 2021-11-19
    Affiliation: University of Ottawa, Ottawa, Ontario (Student)

    Description:
        Create two objects, one records frames, one filters the frames and displays them.
        Launch both in parallel processing approach
        User is able to add or remove filters using an OpenCv button bank

    Libraries required:
        opencv-python
        tkinter
        numpy
"""



from video_players import *
from tkinter import *
from cv2 import *
import numpy as np




if __name__ == "__main__":

    # Initialization Steps
    top = Tk()

    l = Label(top, text="").grid(row=0, column=0, columnspan=15)
    # Add all currently supported filtering methods
    row = 0

    # Initialize webcam
    camera = cv2.VideoCapture(0)

    # Create webcam object, which adds all frames to the filter's frame-queue
    video_getter = VideoGetter(camera).start()
    video_shower = VideoShower(video_getter.frame_queue, top).start()
    top.mainloop()

    # If reached, program has been ended. Destroy all windows.
    cv2.destroyAllWindows()
