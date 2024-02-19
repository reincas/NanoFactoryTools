##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

from scidatacontainer import load_config
from nanofactorysystem import System, getLogger, mkdir
from nanofactorytools import Layer, Grid

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
    "beta": 0.6,
    }

grid_args = {
    "zNum": 301,
    "xStep": 50.0,
    "yStep": 50.0,
    "zStep": 0.1,
    "zOffset": -7.0,
    "laserPower": layer_args["laserPower"],
    "duration": layer_args["duration"],
    }

nx = 9
ny = 7
zlo = zup = 25200.0
path = mkdir("test/grid")
logger = getLogger(logfile="%s/console.log" % path)

logger.info("Initialize system object...")
with System(sample, logger, config, **system_args) as system:

    logger.info("Initialize layer object...")
    subpath = mkdir("%s/layer" % path)
    layer = Layer(system, focus_args, logger, config, **layer_args)

    logger.info("Store background image...")
    layer.focus.imgBack.write("%s/back.zdc" % subpath)

    logger.info("Run layer detection...")
    x = system.x0 - (0.5*nx + 1)*grid_args["xStep"]
    y = system.y0 - (0.5*ny + 1)*grid_args["yStep"]
    layer.run(x, y, zlo, zup, path=subpath)

    logger.info("Store layer results...")
    dc = layer.container()
    dc.write("%s/layer.zdc" % subpath)
    
    logger.info("Initialize grid object...")
    result = dc["meas/result.json"]
    grid_args.update({
        "zLower": result["zLower"],
        "zUpper": result["zUpper"],
        "xOffset": result["xOffset"],
        "yOffset": result["yOffset"],
        })
    grid = Grid(system, config, **grid_args)
    
    logger.info("Run grid exposure...")
    x = system.x0
    y = system.y0
    grid.run(x, y, nx, ny)
    
    logger.info("Store grid results...")
    dc = grid.container()
    dc.write("%s/grid.zdc" % path)

    logger.info("Done.")
