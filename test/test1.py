##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

from skimage.registration import phase_cross_correlation
from scidatacontainer import load_config
from nanofactorysystem import System, getLogger, mkdir
from nanofactorytools import Layer, image

config = load_config(
    author = "Reinhard Caspary",
    email = "reinhard.caspary@phoenixd.uni-hannover.de",
    organization = "Leibniz Universität Hannover",
    orcid = "0000-0003-0460-6088")

sample = {
    "name": "#1",
    "orientation": "top",
    "substrate": "boro-silicate glass",
    "substrateThickness": 700.0,
    "material": "SZ2080",
    "materialThickness": 75.0,
    }

system_args = {
    "name": "Laser Nanofactory",
    "manufacturer": "Femtika",
    "wavelength": 0.515,
    "objective": "Zeiss 20x, NA 0.8",
    "zMax": 25700.0,
    }

focus_args = {}

layer_args = {
    "laserPower": 0.7,
    "stageSpeed": 200.0,
    "duration": 0.2,
    "dzCoarseDefault": 100.0,
    "dzFineDefault": 10.0,
    "resolution": 0.5,
    "beta": 0.6,
    }

zlo = zup = 25200.0
path = mkdir("test/off")
logger = getLogger(logfile="%s/console.log" % path)

logger.info("Initialize system object...")
with System(sample, logger, config, **system_args) as system:

    sx = 0.2
    sy = -0.2 
    if 1:
        logger.info("Initialize layer object...")
        subpath = mkdir("%s/layer" % path)
        layer = Layer(system, focus_args, logger, config, **layer_args)

        logger.info("Store background image...")
        layer.focus.imgBack.write("%s/back.zdc" % subpath)

        logger.info("Run layer detection...")
        x = system.x0
        y = system.y0
        layer.run(x, y, zlo, zup, path=subpath)

        logger.info("Store layer results...")
        dc = layer.container()
        dc.write("%s/layer.zdc" % subpath)

        result = dc["meas/result.json"]        
        zlo = result["zLower"]
        zup = result["zUpper"]
        x_off = sx * result["xOffset"]
        y_off = sy * result["yOffset"]
        z_off = result["zOffset"]
    else:
        x = system.x0
        y = system.y0
        zlo = 25229.6
        zup = 25261.0
        x_off = sx * -48
        y_off = sy * 5
        z_off = -7.0
        
    z = 0.5 * (zlo + zup)

    delay = system["delay"]

    dx = 2.0
    dy = 2.0
    size = 256
    system.moveabs(x=x+x_off-dx, y=y+y_off-dy, z=z+z_off, wait=delay)
    img0 = system.getimage()["meas/image.png"]
    img0 = image.crop(img0, size)

    system.moveabs(x=x+x_off, y=y+y_off, z=z+z_off, wait=delay)
    img1 = system.getimage()["meas/image.png"]
    img1 = image.crop(img1, size)

    system.moveabs(x=x+x_off+dx, y=y+y_off+dy, z=z+z_off, wait=delay)
    img2 = system.getimage()["meas/image.png"]
    img2 = image.crop(img2, size)

    diff0 = image.diff(img0, img1)
    diff1 = image.diff(img1, img2)
    y, x = phase_cross_correlation(diff0, diff1, upsample_factor=20,
                                   overlap_ratio=0.3)[0]
    sx = dx / x
    sy = dy / y
    print(x, y)
    print(sx, sy)

    image.write("%s/img0.png" % path, img0)
    image.write("%s/img1.png" % path, img1)
    image.write("%s/img2.png" % path, img2)
    image.write("%s/diff0.png" % path, diff0)
    image.write("%s/diff1.png" % path, diff1)

    logger.info("Done.")
