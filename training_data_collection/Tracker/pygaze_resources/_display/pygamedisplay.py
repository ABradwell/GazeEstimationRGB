# -*- coding: utf-8 -*-
#
# This file is part of PyGaze - the open-source toolbox for eye tracking
#
#    PyGaze is a Python module for easily creating gaze contingent experiments
#    or other software (as well as non-gaze contingent experiments/software)
#    Copyright (C) 2012-2013  Edwin S. Dalmaijer
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>

from .. import *
from ..libtime import clock

import copy
import math
import os.path

import pygame
import pygame.display
import pygame.draw
import pygame.image

from .._display.basedisplay import BaseDisplay
# we try importing the copy_docstr function, but as we do not really need it
# for a proper functioning of the code, we simply ignore it when it fails to
# be imported correctly
try:
    from .._misc.misc import copy_docstr
except:
    pass


class PyGameDisplay(BaseDisplay):

    # See _display.basedisplay.BaseDisplay for documentation

    def __init__(self, dispsize=settings.DISPSIZE, fgc=settings.FGC, bgc=settings.BGC, screen=None, **args):

        # See _display.basedisplay.BaseDisplay for documentation
        
        # try to import copy docstring (but ignore it if it fails, as we do
        # not need it for actual functioning of the code)
        try:
            copy_docstr(BaseDisplay, PyGameDisplay)
        except:
            # we're not even going to show a warning, since the copied
            # docstring is useful for code editors; these load the docs
            # in a non-verbose manner, so warning messages would be lost
            pass

        self.dispsize = dispsize
        self.fgc = fgc
        self.bgc = bgc
        self.mousevis = False

        # initialize PyGame display-module
        pygame.display.init()
        # make mouse invisible (should be so per default, but you never know)
        pygame.mouse.set_visible(self.mousevis)
        if settings.FULLSCREEN:
            mode = pygame.FULLSCREEN|pygame.HWSURFACE|pygame.DOUBLEBUF
        else:
            mode = pygame.HWSURFACE|pygame.DOUBLEBUF
        # create surface for full screen displaying        
        self.disp = pygame.display.set_mode(self.dispsize, mode)

        # blit screen to display surface (if user entered one)
        if screen:
            self.disp.blit(screen.screen, (0, 0))
        else:
            self.disp.fill(self.bgc)


    def show(self):

        # See _display.basedisplay.BaseDisplay for documentation

        pygame.display.flip()
        return clock.get_time()


    def show_part(self, rect, screen=None):

        # See _display.basedisplay.BaseDisplay for documentation

        if len(rect) > 1:
            for r in rect:
                self.disp.set_clip(r)
                if screen:
                    self.disp.blit(screen.screen, (0,0))
                pygame.display.update(r)
                self.disp.set_clip(None)
                
        elif len(rect) == 1:
            self.disp.clip(rect)
            if screen:
                self.disp.blit(screen.screen, (0,0))
            pygame.display.update(rect)
            self.disp.set_clip(None)

        else:
            raise Exception("Error in libscreen.Display.show_part: rect should be a single rect (i.e. a (x,y,w,h) tuple) or a list of rects!")
        
        return clock.get_time()


    def fill(self, screen=None):

        # See _display.basedisplay.BaseDisplay for documentation

        self.disp.fill(self.bgc)
        if screen != None:
            self.disp.blit(screen.screen, (0, 0))


    def close(self):

        # See _display.basedisplay.BaseDisplay for documentation

        pygame.display.quit()


    def make_screenshot(self, filename="screenshot.png"):

        # See _display.basedisplay.BaseDisplay for documentation

        pygame.image.save(self.disp, filename)
