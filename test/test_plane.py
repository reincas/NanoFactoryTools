##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import json
from scidatacontainer import load_config
from nanofactorysystem import System, getLogger, mkdir
from nanofactorytools import Plane

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
    "beta": 0.7,
    }

plane_args = {}

dx = 80.0
dy = 80.0
nx = 2
ny = 2
zlo = zup = 25200.0
path = mkdir("test/plane")
logger = getLogger(logfile="%s/console.log" % path)

logger.info("Initialize system object...")
with System(sample, logger, config, **system_args) as system:

    logger.info("Initialize plane object...")
    plane = Plane(zlo, zup, system, focus_args, layer_args, logger, config, **plane_args)
        
    #logger.info("Load steps...")
    #with open("%s/steps.json" % path, "r") as fp:
    #    plane.steps = json.loads(fp.read())
    
    logger.info("Store background image...")
    plane.layer.focus.imgBack.write("%s/back.zdc" % path)

    logger.info("Run plane detection...")
    for j in range(ny):
        for i in range(nx):
            x = system.x0 + i*dx
            y = system.y0 + j*dy
            plane.run(x, y, path=path)

    #logger.info("Store steps...")
    #with open("%s/steps.json" % path, "w") as fp:
    #    fp.write(json.dumps(plane.steps))
    
    logger.info("Store results...")
    dc = plane.container()
    dc.write("%s/plane.zdc" % path)
    logger.info("Done.")