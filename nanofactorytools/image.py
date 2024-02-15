##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module contains convenience functions for image processing
#
##########################################################################

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.registration import phase_cross_correlation


def read(fn):

    """ Rear image from file and eventually convert it to grayscale. """

    img = cv.imread(fn)
    img = gray(img)
    return img


def write(fn, img):

    """ Write image to file. """

    cv.imwrite(fn, img)
    

def gray(img):

    """ Eventually convert image to grayscale. """

    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def diff(img0, img1, r=None):

    """ Return floating point difference image blurred with Gaussian
    kernel. """
    
    img = cv.absdiff(img0, img1)
    if img.dtype != float:
        img = img.astype(float)
    img = blur(img, r)
    return img


def norm(img):

    """ Spread image values to the fullrange of 0-255 and return it as a
    unsigned 8-bit integer image. Also return the minimum and maximum
    values of the initial image. """

    imin = img.min()
    imax = img.max()
    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    return img, imin, imax


def crop(img, shape, offset=None):

    """ Return central sub-image with given shape (tuple) or quadratic
    size (int). Eventually reduce color space to gray scale. """

    if isinstance(shape, int):
        shape = (shape, shape)
    height, width = shape

    if offset is not None:
        offx, offy = offset
    else:
        offx, offy = 0, 0
    
    img = gray(img)
    h, w = img.shape
    if h < height or w < width:
        raise RuntimeError("Image too small!")

    x0 = offx + (w-width) // 2
    y0 = offy + (h-height) // 2
    x1 = x0 + width
    y1 = y0 + height
    return img[y0:y1,x0:x1]


def blur(img, r):

    """ Apply a Gaussian blur to the given image. """

    if img.dtype != float:
        img = img.astype(float)
    if r:
        img = cv.GaussianBlur(img, (0,0), r, borderType=cv.BORDER_DEFAULT)
    return img

def gauss(size, sigma, dx=0.0, dy=0.0, norm=False):

    """ Quadratic Gaussian kernel array with optional offset. """

    x, y = coordinates((size, size))
    gauss = np.exp(-((x+dx)**2+(y+dy)**2)/sigma**2)
    if norm:
        gauss /= gauss.sum()
    return gauss

    
def register(img0, img1, ups=100):

    """ Register img1 on img0 with sub-pixel resolution. """

    ## RuntimeWarning: invalid value encountered in cdouble_scalars
    dy, dx = phase_cross_correlation(img0, img1, upsample_factor=ups)[0]
    return dx, dy


def subshift(img, dx, dy, sigma):

    """ Shift image with sub-pixel accuracy by filtering with an offset
    Gaussian kernel. """

    r = round(np.sqrt(dx*dx+dy*dy)+2*sigma)
    size = 2*r + 1
    kernel = gauss(size, sigma, dx, dy, norm=True)
    img = cv.filter2D(img, -1, kernel)

    #maxval = max(kernel[:,(0,-1)].max(), kernel[(0,-1),:].max())
    #print("kernel size:    %d" % size)
    #print("max edge value: %g" % maxval)
    return img


def subdiff(img0, img1, dx, dy, sigma=2.0):

    """ Shift each image half way and return the differencial image.
    Parameters dx and dy are expected as img1 registered on img0. """

    if img0.dtype != float:
        img0 = img0.astype(float)
    if img1.dtype != float:
        img1 = img1.astype(float)
    img0 = subshift(img0, -0.5*dx, -0.5*dy, sigma)
    img1 = subshift(img1, 0.5*dx, 0.5*dy, sigma)
    img = diff(img0, img1, None)
    return img


def normcolor(img, cmap=None):

    """ Normalize given floating point image and convert it to 8-bit BGR
    image. """

    if cmap is None:
        cmap = "viridis"
    img = cv.normalize(img, None, 0.0, 1.0, cv.NORM_MINMAX, cv.CV_64F)
    img = plt.get_cmap(cmap)(img)[:,:,:3]
    img *= 255
    img = img.astype(np.uint8)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    return img


def get_contours(img, thres):

    """ Return a list of contours from the image at given threshold. """

    img = np.where(img <= thres, 0, 1).astype(np.uint8)
    return cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]

    
def contour_mask(shape, contour):

    """ Return boolean array with True for all elements inside the
    contour. """

    img = np.zeros(shape, dtype=int)
    cv.drawContours(img, [contour], -1, 255, -1)
    return img.astype(bool)


def circle_mask(shape, r, dx, dy):

    """ Return boolean array with True for all elements inside a circle
    with radius r and given center offset. """

    x, y = coordinates(shape)
    return (x-dx)**2 + (y-dy)**2 <= r**2


def coordinates(shape):

    """ Return centered x and y coordinate matrices. """

    h, w = shape
    x = np.arange(w, dtype=float) - 0.5*(w-1) 
    y = np.arange(h, dtype=float) - 0.5*(h-1)
    x, y = np.meshgrid(x, y, indexing="xy")
    return x, y


def stats(img, mask=None):

    """ Return average value and standard deviation of flattened and
    optionally masked array. """

    if mask is not None:
        if mask.dtype != bool:
            raise RuntimeError("Boolean mask reqired!")
        img = np.ma.array(img, mask=mask)
    return img.mean(), img.std()
