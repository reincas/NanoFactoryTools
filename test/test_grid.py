##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

from nanofactorysystem import System, getLogger, mkdir
from nanofactorytools import Layer, Grid

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
        "laserPower": 0.7,
        "stageSpeed": 200.0,
        "duration": 0.2,
        "dzCoarseDefault": 100.0,
        "dzFineDefault": 10.0,
        "resolution": 0.5,
        "beta": 0.7,
        },
    "grid": {
        "gridPitch": 40.0,
        "laserPower": 0.7,
        "duration": 0.2,
        },
    }

nx = 9
ny = 7
zlo = zup = 25200.0

user = "Reinhard"
objective = "Zeiss 20x"
path = mkdir("test/grid")
logger = getLogger(logfile="%s/console.log" % path)

logger.info("Initialize system object...")
with System(user, objective, logger, **args) as system:

    # Initialize layer object
    logger.info("Initialize layer object...")
    subpath = mkdir("%s/layer" % path)
    layer = Layer(system, logger, **args)

    # Store background image
    logger.info("Store background image...")
    layer.focus.imgBack.write("%s/back.zdc" % subpath)

    # Run layer detection
    pitch = args["grid"]["gridPitch"]
    x = system.x0 - (0.5*nx + 1)*pitch
    y = system.y0 - (0.5*ny + 1)*pitch
    layer.run(x, y, zlo, zup, path=subpath)

    # Run pitch detection
    logger.info("Run pitch detection...")
    ((pxx, pxy), (pyx, pyy)) = layer.pitch()
    logger.info("Camera pitch xx: %.3f µm/px" % pxx)
    logger.info("Camera pitch xy: %.3f µm/px" % pxy)
    logger.info("Camera pitch yx: %.3f µm/px" % pyx)
    logger.info("Camera pitch yy: %.3f µm/px" % pyy)

    # Store layer results
    logger.info("Store layer results...")
    dc = layer.container()
    dc.write("%s/layer.zdc" % subpath)
    result = dc["meas/result.json"]

    # Initialize grid object
    grid = Grid(system, logger, **args)
    
    # Run grid exposure
    x = system.x0
    y = system.y0
    zlo = result["zLower"]
    zup = result["zUpper"]
    z = 0.5 * (zlo + zup)
    grid.run(x, y, z, nx, ny)
    print(system.transform.P)
    
    # Store grid results
    logger.info("Store grid results...")
    dc = grid.container()
    dc.write("%s/grid.zdc" % path)
    print(dc)

    logger.info("Done.")
