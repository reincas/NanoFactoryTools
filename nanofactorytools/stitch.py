##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import json
import numpy as np
import cv2 as cv
import skimage

from . import image


##########################################################################
# Utility functions
##########################################################################

def readdiff(fn, back):

    """ Read image and subtract given background image. """

    img = image.read(fn)
    img = cv.absdiff(img, back)
    return img


##########################################################################
# Image stitching canvas
##########################################################################

class Canvas(object):

    def __init__(self, width, height):

        """ Initialize empty canvas. """

        # Canvas size in pixels
        self.width = abs(int(width))
        self.height = abs(int(height))

        # Initialize image registry
        self.count = np.zeros((self.height, self.width), dtype=int)
        self.images = []


    def add(self, img, offsetx, offsety):

        """ Register given image on the canvas with given offset. """

        # Size of the given image
        height, width = img.shape

        # Horizontal image position outside the canvas
        if offsetx + width <= 0 or offsetx >= self.width:
            return

        # Vertical image position outside the canvas
        if offsety + height <= 0 or offsety >= self.height:
            return

        # Partial overlap on the left side of the canvas
        if offsetx < 0:
            dx = -offsetx
            img = img[:,dx:]
            offsetx = 0

        # Partial overlap on the right side of the canvas
        if offsetx + width > self.width:
            dx = offsetx + width - self.width
            img = img[:,:-dx]
            
        # Partial overlap on the bottom of the canvas
        if offsety < 0:
            dy = -offsety
            img = img[dy:,:]
            offsety = 0

        # Partial overlap on the top of the canvas
        if offsety + height > self.height:
            dy = offsety + height - self.height
            img = img[:-dy,:]

        # Size of the potentially cropped image
        height, width = img.shape

        # Bounding box of the image location on the canvas
        x1, y1 = offsetx, offsety
        x2, y2 = x1 + width, y1 + height

        # Register the image on the canvas
        self.count[y1:y2,x1:x2] += 1
        self.images.append((img, x1, y1, x2, y2))


    def get(self):

        """ Return superposition of all registered images. """

        canvas = np.zeros((self.height, self.width), dtype=float)

        for img, x1, y1, x2, y2 in self.images:
            canvas[y1:y2,x1:x2] += img

        cnt = np.where(self.count == 0, 1, self.count)
        canvas = np.where(self.count == 0, 0, canvas/cnt)
        canvas = canvas.astype(np.uint8)
        
        return canvas


##########################################################################
# Determination of transformation matrix
##########################################################################

def _intersect(img1, img2, x, y):

    """ Return intersection part of both images based on the given
    offset guess for img2 relative to img1. """

    # Sanity check
    if img1.shape != img2.shape:
        raise RuntimeError("Image shapes must match!")

    # Image size in pixels
    h, w = img1.shape

    # Horizontal slices
    if x < 0:
        x1 = slice(0, w+x)
        x2 = slice(-x, w)
    elif x > 0:
        x1 = slice(x, w)
        x2 = slice(0, w-x)
    else:
        x1 = slice(None)
        x2 = slice(None)
    
    # Vertical slices
    if y < 0:
        y1 = slice(0, h+y)
        y2 = slice(-y, h)
    elif y > 0:
        y1 = slice(y, h)
        y2 = slice(0, h-y)
    else:
        y1 = slice(None)
        y2 = slice(None)

    # Return partial images
    return img1[y1,x1], img2[y2,x2]


def _register(img1, img2, dx_raw, dy_raw, resolution=100):

    """ Return stitching offset between the two given images. """

    # Get intersection part of both images based on the given offset
    # guess for img2 relative to img1
    img1, img2 = _intersect(img1, img2, dx_raw, dy_raw)

    # Suppress noise
    img1 = np.where(img1 >= 16, img1, 0)
    img2 = np.where(img2 >= 16, img2, 0)

    # Determine residual stiching offset between the two partial
    # images
    shifts = skimage.registration.phase_cross_correlation(img1, img2,
                                            upsample_factor=resolution)[0]
    dy, dx = shifts

    # Much slower alternative for reference
    #mask1 = img1 >= 16
    #mask2 = img2 >= 16
    #dy, dx = skimage.registration.phase_cross_correlation(img1, img2,
    #            reference_mask=mask1, moving_mask=mask2, upsample_factor=resolution)

    # Return stiching offset
    return dx_raw+dx, dy_raw+dy


def get_shear(resolution, pitch_x, pitch_y, nx, ny, fn, fn_base):

    """ Determine transformation matrix. """

    # Camera resolution in pixel per micrometer
    resolution = float(resolution)
    
    # Offset between adjacent images in micrometers in stage coordinate
    # system
    pitch_x = float(pitch_x)
    pitch_y = float(pitch_y)

    # Number of images in horizontal and vertical direction
    nx = int(nx)
    ny = int(ny)

    # Background image
    img_b = image.read(fn_base)

    # Horizontal translation vector
    print("Horizontal translation vector...")
    dx_raw = round(pitch_x * resolution)
    dy_raw = 0
    a11 = []
    a21 = []
    for j in range(ny):
        for i in range(nx-1):
            img1 = readdiff(fn % (i, j), img_b)
            img2 = readdiff(fn % (i+1, j), img_b)
            dx, dy = _register(img1, img2, dx_raw, dy_raw)
            #print(dx, dy)
            if np.sqrt((dx_raw-dx)**2+(dy_raw-dy)**2) > 20:
                raise RuntimeError("Shear deviation larger than expected!")
            #print(i, j, dx, dy)
            a11.append(dx)
            a21.append(dy)
    a11 = np.mean(a11) / pitch_x
    a21 = np.mean(a21) / pitch_x
    
    # Vertical translation vector
    print("Vertical translation vector...")
    dx_raw = 0
    dy_raw = round(pitch_y * resolution)
    a12 = []
    a22 = []
    for j in range(ny-1):
        for i in range(nx):
            img1 = readdiff(fn % (i, ny-1-j), img_b)
            img2 = readdiff(fn % (i, ny-2-j), img_b)
            dx, dy = _register(img1, img2, dx_raw, dy_raw)
            #print(dx, dy)
            if np.sqrt((dx_raw-dx)**2+(dy_raw-dy)**2) > 20:
                raise RuntimeError("Shear deviation larger than expected!")
            #print(i, j, dx, dy)
            a12.append(dx)
            a22.append(dy)
    a12 = np.mean(a12) / pitch_y
    a22 = np.mean(a22) / pitch_y

    # Return shear matrix
    shear = Shear(a11, a12, a21, a22)
    print(shear)
    print("Done.")
    return shear


##########################################################################
# Coordinate transformation class
##########################################################################

class Shear(object):

    def __init__(self, mat=None, file=None):

        """ Store coordinate transformation matrix. """

        if file:
            mat = self.load(file)
            
        self.a11 = float(mat[0])
        self.a12 = float(mat[1])
        self.a21 = float(mat[2])
        self.a22 = float(mat[3])


    def __repr__(self):

        return "Shear(%.8f, %.8f, %.8f, %.8f)" % (self.a11, self.a12, self.a21, self.a22)


    def pixel(self, dx, dy):

        """ Convert given stage offset in micrometers to camera
        coordinates in pixels. """

        px = round(self.a11*dx + self.a12*dy)
        py = round(self.a21*dx + self.a22*dy)
        return px, py        


    def save(self, fn):

        """ Save coordinate transforamtion matrix to JSON file. """

        data = json.dumps((self.a11, self.a12, self.a21, self.a22))
        with open(fn, "w") as fp:
            fp.write(data)


    def load(self, fn):
    
        """ Restore shear object from given JSON data file. """
    
        with open(fn, "r") as fp:
            data = fp.read()
        return json.loads(data)


##########################################################################
# Image stitching canvas with coodinate transformation
##########################################################################

class ShearCanvas(Canvas):

    """ Image stitching canvas with coodinate transformation from stage
    coordinates to camera coordinates. """

    def __init__(self, pitch_x, pitch_y, nx, ny, w, h, shear):

        """ Initialize a canvas covering nx times ny images of witdh w
        and height h with given horizontal and vertical pitches using
        the given coordinate transformation shear.  """

        # Offset between adjacent images in micrometers in stage coordinate
        # system
        self.pitch_x = float(pitch_x)
        self.pitch_y = float(pitch_y)

        # Number of images in horizontal and vertical direction
        self.nx = int(nx)
        self.ny = int(ny)

        # Size of sub-images in pixels
        self.w = int(w)
        self.h = int(h)
        
        # Coordinate mapping object
        self.shear = shear
        
        # Bounding box of the canvas in camera pixel coordinates
        x = (self.nx-1) * self.pitch_x
        y = (self.ny-1) * self.pitch_y
        bbox = [(0, 0), (x, 0), (x, y), (0, y)]
        v = [self.shear.pixel(x, y) for x, y in bbox]
        vx = [x for x, y in v]
        vy = [y for x, y in v]
        vx[1] += self.w
        vx[2] += self.w
        vy[2] += self.h
        vy[3] += self.h
        self.x0 = min(vx)
        x1 = max(vx)
        self.y0 = min(vy)
        y1 = max(vy)

        # Initialize canvas
        super().__init__(x1-self.x0, y1-self.y0)
        

    def add(self, img, i, j):

        """ Register given image on the canvas on given horizontal and
        vertical index position. """
        
        dx = i * self.pitch_x
        dy = j * self.pitch_y
        dx, dy = self.shear.pixel(dx, dy)
        super().add(img, round(dx-self.x0), round(dy-self.y0))


    def add_all(self, fn, img_back=None):

        """ Read image from every image position and add it to the
        canvas. Return the superposition image. """
    
        for j in range(self.ny):
            for i in range(self.nx):
                img = readdiff(fn % (i, self.ny-1-j), img_back)
                self.add(img, i, j)
        return self.get()
