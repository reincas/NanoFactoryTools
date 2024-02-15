##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import cv2 as cv
import matplotlib.pyplot as plt
from nanofactorytools import image, Shear, ShearCanvas


def plot1(img, cb=False):

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(img)
    if cb:
        fig.colorbar(im, ax=ax)
    plt.show()


def plot2(img1, img2, cb=False):

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        im = ax[0].imshow(img1)
        if cb:
            fig.colorbar(im, ax=ax)
        im = ax[1].imshow(img2)
        if cb:
            fig.colorbar(im, ax=ax)
        plt.show()

# Camera background image
fn_base = "focus-2/base.png"

# Filanames of sub-images
fn = "focus-2/plane-1/img-%02d-%02d.png"

# Offset between adjacent images in micrometers in stage coordinate
# system
pitch_x = 80.0
pitch_y = 80.0

# Number of images in horizontal and vertical direction
nx = 6
ny = 6

##    # Determine coordinate transformation matrix based on camera
##    # resolution in pixels per micrometer
##    resolution = 5.0
##    shear = get_shear(resolution, pitch_x, pitch_y, nx, ny, fn, fn_base)
##    shear.save("focus-2/shear-1.json")

# Determine coordinate transformation matrix
shear = Shear(file="focus-2/shear-1.json")

# Load background image
img_b = image.read(fn_base)
h, w = img_b.shape

# Stitch all sub-images
canvas = ShearCanvas(pitch_x, pitch_y, nx, ny, w, h, shear)
img = canvas.add_all(fn, img_b)
#plot1(img)

# Normalized false color image
norm_img = image.norm(img)[0]
cv.imwrite("focus-2/img-1.png", norm_img)