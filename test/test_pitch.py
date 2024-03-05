##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import h5py
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
path = mkdir("test/pitch")
logger = getLogger(logfile="%s/console.log" % path)

logger.info("Initialize system object...")
with System(sample, logger, config, **system_args) as system:

    sx = 0.2
    sy = -0.2
    z_off = -7.0
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
    else:
        x = system.x0
        y = system.y0
        zlo = 25229.6
        zup = 25261.0
        x_off = sx * -48
        y_off = sy * 5
    
    z = 0.5 * (zlo + zup)
    logger.info("Layer lo:   %.3f µm" % zlo)
    logger.info("      up:   %.3f µm" % zup)
    logger.info("Position x: %.3f µm" % x)
    logger.info("         y: %.3f µm" % y)
    logger.info("         z: %.3f µm" % z)
    logger.info("Offset x:   %.3f µm" % x_off)
    logger.info("       y:   %.3f µm" % y_off)
    logger.info("       z:   %.3f µm" % z_off)
    logger.info("Pitch x:    %.3f µm/px" % sx)
    logger.info("      y:    %.3f µm/px" % sy)

    delay = system["delay"]

    dx = 2.0
    dy = 2.0
    size = 256
    system.moveabs(x=x+x_off, y=y+y_off, z=z+z_off, wait=delay)
    imgC = system.getimage()["meas/image.png"]
    imgC = image.crop(imgC, size)

    system.moveabs(x=x+x_off-dx, y=y+y_off, z=z+z_off, wait=delay)
    imgW = system.getimage()["meas/image.png"]
    imgW = image.crop(imgW, size)

    system.moveabs(x=x+x_off+dx, y=y+y_off, z=z+z_off, wait=delay)
    imgE = system.getimage()["meas/image.png"]
    imgE = image.crop(imgE, size)

    system.moveabs(x=x+x_off, y=y+y_off-dy, z=z+z_off, wait=delay)
    imgS = system.getimage()["meas/image.png"]
    imgS = image.crop(imgS, size)

    system.moveabs(x=x+x_off, y=y+y_off+dy, z=z+z_off, wait=delay)
    imgN = system.getimage()["meas/image.png"]
    imgN = image.crop(imgN, size)

    diffE = image.diff(imgC, imgE)
    diffW = image.diff(imgW, imgC)
    pxy, pxx = phase_cross_correlation(diffW, diffE, upsample_factor=20, overlap_ratio=0.3)[0]
    sx = dx / pxx

    diffN = image.diff(imgC, imgN)
    diffS = image.diff(imgS, imgC)
    pyy, pyx = phase_cross_correlation(diffS, diffN, upsample_factor=20, overlap_ratio=0.3)[0]
    sy = dy / pyy

    logger.info("Calibration xx: %.3f px / %.1f µm (x)" % (pxx, dx))
    logger.info("            xy: %.3f px / %.1f µm (x)" % (pxy, dx))
    logger.info("            yx: %.3f px / %.1f µm (y)" % (pyx, dy))
    logger.info("            yy: %.3f px / %.1f µm (y)" % (pyy, dy))
    logger.info("Pitch x:    %.3f µm/px" % sx)
    logger.info("      y:    %.3f µm/px" % sy)

    fn = "%s/images.hdf5" % path
    with h5py.File(fn, "w") as fp:
        fp.create_dataset("imgC", data=imgC)
        fp.create_dataset("imgW", data=imgW)
        fp.create_dataset("imgE", data=imgE)
        fp.create_dataset("imgS", data=imgS)
        fp.create_dataset("imgN", data=imgN)
        fp.create_dataset("diffW", data=diffW)
        fp.create_dataset("diffE", data=diffE)
        fp.create_dataset("diffS", data=diffS)
        fp.create_dataset("diffN", data=diffN)

    # with h5py.File(fn, "r") as fp:
    #     imgC = fp["imgC"][...]

    logger.info("Done.")
