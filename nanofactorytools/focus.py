##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides a focus detector class for a laser direct writing
# system.
#
##########################################################################

import numpy as np
import cv2 as cv
from skimage.filters import threshold_otsu
from scidatacontainer import Container
from nanofactorysystem import Parameter, popargs

from . import image
np.seterr(invalid="ignore")

# Status codes
STATUS = ("focus", "no focus", "no contour", "off center",
          "non circular", "offset")
S_FOCUS, S_NOFOCUS, S_NOCONTOUR, S_OFFCENTER, S_NONCIRCULAR, S_OFFSET \
         = range(len(STATUS))


##########################################################################
def mark_focus(dc, lw=2, **args):
    
    """ Take focus container object and return a normalized differential
    image with markings. Recognized keyword arguments:

    * center:  draw center area circle
    * contour: draw contour of focus area
    * focus:   mark focus center by cross hair lines
    """

    # Radius of center area
    radius = dc["eval/parameter.json"]["centerRadius"]
    
    # Results dictionary
    result = dc["eval/result.json"]

    # Normalized difference false color image
    diff = dc["eval/image_diff.png"]
    h, w = diff.shape
    img = image.normcolor(diff)

    # BGR line color
    color = [(255, 255, 255), (153, 153, 153), (153, 153, 153),
             (102, 255, 255), (255, 255, 102), (153, 153, 153)]
    color = color[result["status"]]

    # Draw center area circle
    if "center" in args and args["center"]:
        r = round(radius)
        x = round(0.5*(w-1))
        y = round(0.5*(h-1))
        cv.circle(img, (x, y), r, color, lw)

    # Draw contour line
    if "contour" in args and args["contour"]:
        if result["focusContour"] is not None:
            contour = np.array(result["focusContour"])
            cv.drawContours(img, [contour], -1, color, lw)
        
    # Draw focus position
    if "focus" in args and args["focus"]:
        if result["focusOffset"] is not None:
            xf, yf = result["focusOffset"]
            x = round(xf + 0.5*(w-1))
            y = round(yf + 0.5*(h-1))
            cv.line(img, (x, 0), (x, w-1), color, lw)
            cv.line(img, (0, y), (h-1, y), color, lw)

    # Return image
    return img


##########################################################################
class Focus(Parameter):

    """ Focus detector class. After initializing with a camera
    background image, the method detect() may be called with a pair of
    pre- and post-exposure images. """

    _defaults = {
        "shape": (512, 512),
        "centerRadius": 80,
        "blurInput": 1.0,
        "stdFactor": 2.0,
        "upScale": 100,
        "blurOutput": 2.0,
        "blurThreshold": 4.0,
        "blurContour": 12.0,
        "factorContour": 1.0,
        "peakContour": 0.2,
        "minCircularity": 0.85,
        }

    _resultkeys = [ "status", "statusString", "diffOffset", "diffMin",
        "diffMax", "avgBack", "stdBack", "avgCenter", "stdCenter",
        "avgContrast", "threshold", "thresholdSource", "focusContour",
        "focusOffset", "focusArea", "circularity" ]


    def __init__(self, system, logger=None, **kwargs):

        """ Initialize focus detector parameters using _defaults and the
        optional keyword arguments. Store the background camera image.
        """

        # Store system object
        self.system = system
        user = self.system.user["key"]
        
        # Initialize parameter class
        args = popargs(kwargs, "focus")
        super().__init__(user, logger, **args)
        self.log.info("Initializing focus detector.")

        # Background image
        dz = self.system["backOffset"]
        self.imgBack = self.background(dz)
        shape = self["shape"]
        self.back = image.crop(self.imgBack.img, shape)

        # Mask of the center region
        r = self["centerRadius"]
        self.mask = image.circle_mask(shape, r, 0.0, 0.0)
        
        # No result yet
        self.exposure = None
        self.imgPre = None
        self.imgPost = None
        self.diff = None
        self.result = None
        self.log.info("Initialized focus detector.")

        
    def background(self, dz):

        """ Eventually move to given relative z position and
        take a background camera image. Find the exposure
        time to get camera images with an average pixel value
        of 127. Return image as image container object. """


        self.log.info("Take background image.")

        # Current z position        
        z0 = self.system.position("Z")

        # Move to background z position
        fast = self.system["speed"]
        delay = self.system["delay"]
        self.system.moveabs(fast, delay, z=z0+dz)

        # Set exposure time for normalized image
        self.system.optexpose(127)
            
        # Take background image
        img = self.system.getimage()

        # Move to initial z position
        self.system.moveabs(fast, delay, z=z0)

        # Return images
        self.log.info("Got background image.")
        return img


    def run(self, x, y, z, dz, power, speed, duration):
        
        """ Exposed an axial line or point (dz=0) at given position and
        run focus detection. """

        # Delete previous result
        self.exposure = None
        self.imgPre = None
        self.imgPost = None
        self.diff = None
        self.result = None

        # Expose axial line
        self.imgPre, self.imgPost, self.exposure \
            = self.zline(x, y, z, dz, power, speed, duration)
        
        # Detect focus spot
        self.diff, self.result \
            = self.detect(self.imgPre, self.imgPost)

            
    def zline(self, x, y, z, dz, power, speed, duration):
        
        """ Exposed an axial line or point (dz=0) at given position. """

        # Fast positioning speed
        fast = self.system["speed"]
        
        # Delay time after stages reached their destination
        delay = self.system["delay"]

        # Move to center position
        self.system.moveabs(fast, delay, x=x, y=y, z=z)

        # Take pre exposure camera image
        img0 = self.system.getimage()

        # Expose axial line
        if dz != 0.0:
            v = min(speed, dz/duration)
            dt = dz/v
            self.system.zline(power, fast, v, dz)
            self.system.wait("XYZ", delay)

        # Expose a dot
        else:
            v = 0.0
            dz = 0.0
            dt = duration
            self.system.pulse(power, dt)
            
        # Take post exposure camera image
        img1 = self.system.getimage()

        # Exposure data
        exposure = {
            "x": x,
            "y": y,
            "zCenter": z,
            "zLength": dz,
            "laserPower": power,
            "fastSpeed": fast,
            "destinationDelay": delay,
            "setSpeed": speed,
            "setDuration": duration,
            "speed": v,
            "duration": dt,
            }
        
        # Done.
        return img0, img1, exposure
            

    def detect(self, img0, img1):

        """ Main access point: apply the focus detection algorithm to
        the given pre- and post-exposure images. Return the difference
        image and the result dictionary. """

        img0 = img0.img
        img1 = img1.img
        result = {k: None for k in self._resultkeys}
        diff = self._difference(img0, img1, result)
        self._find(diff, result)
        result["statusString"] = STATUS[result["status"]]
        return diff, result
            
            
    def _difference(self, img0, img1, result):

        """ Return the difference image of the given pre- and
        post-exposure images. """

        # Get sub-images
        img0_sub = image.crop(img0, self["shape"])
        img1_sub = image.crop(img1, self["shape"])

        # Subtract camera background image
        img0 = image.diff(img0_sub, self.back, self["blurInput"])
        img1 = image.diff(img1_sub, self.back, self["blurInput"])

        # Prepare thresholded registration images
        avg0, std0 = image.stats(img0, self.mask)
        avg1, std1 = image.stats(img1, self.mask)
        thres = 0.5*(avg0 + self["stdFactor"]*std0 + avg1 + self["stdFactor"]*std1)
        img0_reg = np.where(img0 < thres, 0, img0)
        img1_reg = np.where(img1 < thres, 0, img1)

        # Register img1 on img0
        dx, dy = image.register(img0_reg, img1_reg, self["upScale"])
        result["diffOffset"] = dx, dy

        # Shift each half-way with sub-pixel resolution and calculate
        # the difference image
        diff = image.subdiff(img0, img1, dx, dy, self["blurOutput"])
        diff, imin, imax = image.norm(diff)
        result["diffMin"] = imin
        result["diffMax"] = imax

        # Done
        return diff


    def _find(self, diff, result):

        """ Focus detection algorithm acting on the given difference
        image. """

        # Determine empty image based on noise level in center and
        # background
        img = image.blur(diff, self["blurThreshold"])
        avgb, stdb = image.stats(img, self.mask)
        avgc, stdc = image.stats(img, ~self.mask)
        result["avgBack"] = avgb
        result["stdBack"] = stdb
        result["avgCenter"] = avgc
        result["stdCenter"] = stdc
        result["avgContrast"] = avgc/avgb
        if avgc < avgb + 0.5*stdb and stdc < 1.3*stdb:
            result["status"] = S_NOFOCUS
            return

        # Threshold for contour detection
        img = image.blur(diff, self["blurContour"])
        thres0 = threshold_otsu(img, 256)
        avg, std = image.stats(img, self.mask)
        thres1 = avg + self["factorContour"]*std
        thres1 += self["peakContour"] * (img.max() - thres1)
        if thres0 > thres1:
            thres = thres0
            result["thresholdSource"] = "otsu"
        else:
            thres = thres1
            result["thresholdSource"] = "noise"
        result["threshold"] = thres
        
        # Find contours
        contours = image.get_contours(img, thres)
        if len(contours) < 1:
            result["status"] = S_NOCONTOUR
            return

        # Contour with maximum average brightness
        contour = None
        mask = None
        bmax = 0.0
        for c in contours:
            m = image.contour_mask(diff.shape, c)
            bright = np.ma.array(img, mask=~m).mean()
            if bright > bmax:
                contour = c
                mask = m
                bmax = bright
        result["focusContour"] = contour.astype(int).tolist()

        # Focus position as center of mass
        x, y = image.coordinates(diff.shape)
        weight = np.ma.array(diff, mask=~mask).astype(img.dtype)
        weight /= weight.sum()
        dx = np.ma.array(weight*x, mask=~mask).sum()
        dy = np.ma.array(weight*y, mask=~mask).sum()
        result["focusOffset"] = dx, dy
        if dx**2 + dy**2 > self["centerRadius"]**2:
            result["status"] = S_OFFCENTER
            return

        # Contour area and equivalent circle
        area = int(mask.astype(int).sum())
        result["focusArea"] = area
        r = round(np.sqrt(area/np.pi))
        cmask = image.circle_mask(diff.shape, r, dx, dy)

        # Circularity is relative overlap of contour and circle
        both = int((mask & cmask).astype(int).sum())
        circularity = both/area
        result["circularity"] = circularity
        if circularity < self["minCircularity"]:
            result["status"] = S_NONCIRCULAR
            return

        # Done
        result["status"] = S_FOCUS


    def container(self, config=None, **kwargs):

        """ Return results as SciDataContainer. """

        if self.result is None:
            raise RuntimeError("No results!")
            
        # Collect UUIDs of background, pre- and post-exposure images as
        # references
        refs = {
            "imageBack": self.imgBack.uuid,
            "imagePre": self.imgPre.uuid,
            "imagePost": self.imgPost.uuid,
            }

        # General metadata
        content = {
            "containerType": {"name": "FocusDetect", "version": 1.1},
            }
        meta = {
            "title": "Focus Detection Data",
            "description": "Detection of laser focus spot on microscope image.",
            }

        # Container dictionary
        items = self.system.items() | {
            "content.json": content,
            "meta.json": meta,
            "references.json": refs,
            "data/exposure.json": self.exposure,
            "data/image_back.json": self.imgBack.params,
            "data/image_pre.json": self.imgPre.params,
            "data/image_post.json": self.imgPost.params,
            "data/focus.json": self.parameters(),
            "meas/image_diff.png": self.diff,
            "meas/result.json": self.result,
            }

        # Return container object
        config = config or self.config
        return Container(items=items, config=config, **kwargs)
