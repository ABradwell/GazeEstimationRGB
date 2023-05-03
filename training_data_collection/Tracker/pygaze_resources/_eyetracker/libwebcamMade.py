# Webcam EyeTracker classes
# Edwin S. Dalmaijer
# version 0.1, 12-10-2013

__author__ = "Edwin Dalmaijer"

# in DEBUG mode, images of the calibration are saved as strings in a textfile,
# after the calibration, the textfile will be read and PNG images will be
# produced based on the content of the file
DEBUG = False
BUFFSEP = 'brad'


# # # # #
# imports

import os.path

import cv2
import pygame
import pygame.camera

pygame.init()

import pygaze
from pygaze._display.basedisplay import BaseDisplay


class CamEyeTracker(BaseDisplay):
    
    """The CamEyeTracker class uses your webcam as an eye tracker"""
    
    def __init__(self, device=None, camres=(640,480)):
        
        """Initializes a CamEyeTracker instance
        
        arguments
        None
        
        keyword arguments
        device      --  a string or an integer, indicating either
                        device name (e.g. '/dev/video0'), or a
                        device number (e.g. 0); None can be passed
                        too, in this case Setup will autodetect a
                        useable device (default = None)
        camres      --  the resolution of the webcam, e.g.
                        (640,480) (default = (640,480))
        """

        self.cam = cv2.VideoCapture(0)

        if device == None:

            if self.cam is None or not self.cam.isOpened():
                device = self.cam
                raise Exception("Error in camtracker.CamEyeTracker.__init__")
            
        # get the webcam resolution (get_size not available on all systems)
        try:
            w = self.cam.get(cv2.CV_CAP_PROP_FRAME_WIDTH)
            h = self.cam.get(cv2.CV_CAP_PROP_FRAME_HEIGHT)
            self.camres = (w, h)
        except:
            self.camres = camres

        # default settings
        self.settings = {'pupilcol':(0,0,0), \
                    'threshold':100, \
                    'nonthresholdcol':(100,100,255,255), \
                    'pupilpos':(-1,-1), \
                    'pupilrect':pygame.Rect(self.camres[0]/2-50,self.camres[1]/2-25,100,50), \
                    'pupilbounds': [0,0,0,0], \
                    '':None                 
                    }
    
    
    def get_size(self):
        
        """Returns a (w,h) tuple of the image size
        
        arguments
        None
        
        keyword arguments
        None
        
        returns
        imgsize     --  a (width,height) tuple indicating the size
                        of the images produced by the webcam
        """
        return self.camres

    
    def get_snapshot(self):
        
        """Returns a snapshot, without doing any any processing
        
        arguments
        None
        
        keyword arguments
        None
        
        returns
        snapshot        --  a pygame.surface.Surface instance,
                        containing a snapshot taken with the webcam
        """

        success, image = self.cam.read()

        return image
        
        
    def threshold_image(self, image):
        
        """Applies a threshold to an image and returns the thresholded
        image
        
        arguments
        image           --  the image that should be thresholded, a
                        pygame.surface.Surface instance
        
        returns
        thresholded     --  the thresholded image,
                        a pygame.surface.Surface instance
        """
        
        # surface to apply threshold to surface
        thimg = pygame.surface.Surface(self.get_size(), 0, image)
        
        # perform thresholding
        th = (self.settings['threshold'],self.settings['threshold'],self.settings['threshold'])
        pygame.transform.threshold(thimg, image, self.settings['pupilcol'], th, self.settings['nonthresholdcol'], 1)
        
        return thimg
    
    
    def find_pupil(self, thresholded, pupilrect=False):
        
        """Get the pupil center, bounds, and size, based on the thresholded
        image; please note that the pupil bounds and size are very
        arbitrary: they provide information on the pupil within the
        thresholded image, meaning that they would appear larger if the
        camera is placed closer towards a subject, even though the
        subject's pupil does not dilate
        
        arguments
        thresholded     --  a pygame.surface.Surface instance, as
                        returned by threshold_image
        
        keyword arguments
        pupilrect       --  a Boolean indicating whether pupil searching
                        rect should be applied or not
                        (default = False)
        
        returns
        pupilcenter, pupilsize, pupilbounds
                    --  pupilcenter is an (x,y) position tuple that
                        gives the pupil center with regards to the
                        image (where the top left is (0,0))
                        pupilsize is the amount of pixels that are
                        considered to be part of the pupil in the
                        thresholded image; when no pupilbounds can
                        be found, this will return (-1,-1)
                        pupilbounds is a (x,y,width,height) tuple,
                        specifying the size of the largest square
                        in which the pupil would fit
        """
        
        
        # cut out pupilrect (but only if pupil bounding rect option is on)
        if pupilrect:
            # pupil rect boundaries
            rectbounds = pygame.Rect(self.settings['pupilrect'])
            # correct rect edges that go beyond image boundaries
            if self.settings['pupilrect'].left < 0:
                rectbounds.left = 0
            if self.settings['pupilrect'].right > self.camres[0]:
                rectbounds.right = self.camres[0]
            if self.settings['pupilrect'].top < 0:
                rectbounds.top = 0
            if self.settings['pupilrect'].bottom > self.camres[1]:
                rectbounds.bottom = self.camres[1]
            # cut rect out of image
            thresholded = thresholded.subsurface(rectbounds)
            ox, oy = thresholded.get_offset()
        
        # find potential pupil areas based on threshold
        th = (self.settings['threshold'],self.settings['threshold'],self.settings['threshold'])
        mask = pygame.mask.from_threshold(thresholded, self.settings['pupilcol'], th)
        
        # get largest connected area within mask (which should be the pupil)
        pupil = mask.connected_component()
        
        # get pupil center
        pupilcenter = pupil.centroid()
        
        # if we can only look within a rect around the pupil, do so
        if pupilrect:
            # compensate for subsurface offset
            pupilcenter = pupilcenter[0]+ox, pupilcenter[1]+oy
            # check if the pupil position is within the rect
            if (self.settings['pupilrect'].left < pupilcenter[0] < self.settings['pupilrect'].right) and (self.settings['pupilrect'].top < pupilcenter[1] < self.settings['pupilrect'].bottom):
                # set new pupil and rect position
                self.settings['pupilpos'] = pupilcenter
                x = pupilcenter[0] - self.settings['pupilrect'][2]/2
                y = pupilcenter[1] - self.settings['pupilrect'][3]/2
                self.settings['pupilrect'] = pygame.Rect(x,y,self.settings['pupilrect'][2],self.settings['pupilrect'][3])
            # if the pupil is outside of the rect, return missing
            else:
                self.settings['pupilpos'] = (-1,-1)
        else:
            self.settings['pupilpos'] = pupilcenter
        
        # get pupil bounds (sometimes failes, hence try-except)
        try:
            self.settings['pupilbounds'] = pupil.get_bounding_rects()[0]
            # if we're using a pupil rect, compensate offset
            if pupilrect:
                self.settings['pupilbounds'].left += ox
                self.settings['pupilbounds'].top += oy
        except:
            # if it fails, we simply use the old rect
            pass
        
        return self.settings['pupilpos'], pupil.count(), self.settings['pupilbounds']


    def give_me_all(self, pupilrect=False):
        
        """Returns snapshot, thresholded image, pupil position, pupil area,
        and pupil bounds
        
        arguments
        None
        
        keyword arguments
        pupilrect       --  a Boolean indicating whether pupil searching
                        rect should be applied or not
                        (default = False)
        
        returns
        snapshot, thresholded, pupilcenter, pupilbounds, pupilsize
            snapshot    --  a pygame.surface.Surface instance,
                        containing a snapshot taken with the webcam
            thresholded --  the thresholded image,
                        a pygame.surface.Surface instance
            pupilcenter --  pupilcenter is an (x,y) position tuple that
                        gives the pupil center with regards to the
                        image (where the top left is (0,0))
            pupilsize   --  pupilsize is the amount of pixels that are
                        considered to be part of the pupil in the
                        thresholded image; when no pupilbounds can
                        be found, this will return (-1,-1)
            pupilbounds --  pupilbounds is a (x,y,width,height) tuple,
                        specifying the size of the largest square
                        in which the pupil would fit
        """
        
        img = self.get_snapshot()
        thimg = self.threshold_image(img)
        ppos, parea, pbounds = self.find_pupil(thimg, pupilrect)
        
        return img, thimg, ppos, parea, pbounds


    def close(self):
        
        """Shuts down connection to the webcam and closes logfile
        
        arguments
        None
        
        keyword arguments
        None
        
        returns
        None
        """
        
        # close camera
        self.cam.release()

    def calibrate(self):
        pass
