##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

from nanofactorysystem import System, getLogger, mkdir
from nanofactorytools import Layer

args = {
    "attenuator": {
        "fitKind": "quadratic",
        },
    "controller": {
        "zMax": 25700.0,
        },
    "sample": {
        "name": "#1",
        "orientation": "top",
        "substrate": "boro-silicate glass",
        "substrateThickness": 700.0,
        "material": "SZ2080",
        "materialThickness": 75.0,
        },
    "focus": {},
    "layer": {
        "beta": 0.7,
        },
    }

zlo = zup = 25200.0

user = "Reinhard"
objective = "Zeiss 20x"
path = mkdir("test/layer")
logger = getLogger(logfile="%s/console.log" % path)

logger.info("Initialize system object...")
with System(user, objective, logger, **args) as system:

    logger.info("Initialize layer object...")
    layer = Layer(system, logger, **args)

    logger.info("Store background image...")
    layer.focus.imgBack.write("%s/back.zdc" % path)

    logger.info("Run layer detection...")
    x = system.x0
    y = system.y0
    layer.run(x, y, zlo, zup, path=path)

    logger.info("Run pitch detection...")
    ((pxx, pxy), (pyx, pyy)) = layer.pitch()
    logger.info("Camera pitch xx: %.3f µm/px" % pxx)
    logger.info("Camera pitch xy: %.3f µm/px" % pxy)
    logger.info("Camera pitch yx: %.3f µm/px" % pyx)
    logger.info("Camera pitch yy: %.3f µm/px" % pyy)
    
    logger.info("Store results...")
    dc = layer.container()
    dc.write("%s/layer.zdc" % path)
    logger.info("Done.")
