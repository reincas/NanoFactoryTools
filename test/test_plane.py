##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

from nanofactorysystem import System, getLogger, mkdir
from nanofactorytools import Plane

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
    "plane": {},
    }

dx = 80.0
dy = 80.0
nx = 2
ny = 2
zlo = zup = 25200.0

user = "Reinhard"
objective = "Zeiss 20x"
path = mkdir("test/plane")
logger = getLogger(logfile="%s/console.log" % path)

logger.info("Initialize system object...")
with System(user, objective, logger, **args) as system:

    logger.info("Initialize plane object...")
    plane = Plane(zlo, zup, system, logger, **args)
        
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