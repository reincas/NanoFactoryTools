##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

from scidatacontainer import load_config
from nanofactorysystem import System, getLogger, mkdir
from nanofactorytools import Layer

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
    "beta": 0.6,
    }

zlo = zup = 25200.0
path = mkdir("test/layer")
logger = getLogger(logfile="%s/console.log" % path)

logger.info("Initialize system object...")
with System(sample, logger, config, **system_args) as system:

    logger.info("Initialize layer object...")
    layer = Layer(system, focus_args, logger, config, **layer_args)

    logger.info("Store background image...")
    layer.focus.imgBack.write("%s/back.zdc" % path)

    logger.info("Run layer detection...")
    x = system.x0
    y = system.y0
    layer.run(x, y, zlo, zup, path=path)

    logger.info("Store results...")
    dc = layer.container()
    dc.write("%s/layer.zdc" % path)
    logger.info("Done.")
