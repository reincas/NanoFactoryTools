##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

from scidatacontainer import load_config
from nanofactorysystem import System, getLogger, mkdir
from nanofactorytools import Focus

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

path = mkdir("test/focus")
logger = getLogger(logfile="%s/console.log" % path)

logger.info("Initialize system object...")
with System(sample, logger, config, **system_args) as system:
    
    logger.info("Initialize focus object...")
    focus = Focus(system, logger, config, **focus_args)
    
    logger.info("Store background image...")
    focus.imgBack.write("%s/back.zdc" % path)
    
    logger.info("Expose vertical line and detect focus...")
    x = system.x0
    y = system.y0
    z = system.z0
    dz = 0.0
    power = 0.7
    speed= 200.0
    duration= 0.2
    focus.run(x, y, z, dz, power, speed, duration)
    
    logger.info("Store images...")
    focus.imgPre.write("%s/image_pre.zdc" % path)
    focus.imgPost.write("%s/image_post.zdc" % path)
    
    logger.info("Store results...")
    dc = focus.container()
    dc.write("%s/focus.zdc" % path)
    logger.info("Done.")
