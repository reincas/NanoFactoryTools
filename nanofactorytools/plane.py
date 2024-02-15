##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides a photoresin interface plane class for a laser
# direct writing system.
#
##########################################################################

import numpy as np
from scidatacontainer import Container
from nanofactorysystem import Parameter, mkdir

from .layer import Layer


##########################################################################
class PlaneFit(object):

    """ Fit a plane to the given list of x,y,z values by least squares
    fitting of the z values. """

    def __init__(self, points):

        """ Run the fitting algorithm and store the results. """

        # Least squares plane fitting
        A = np.array(points, dtype=float)
        b = np.array(A[:,-1])
        A[:,-1] = 1.0
        self.params = np.linalg.lstsq(A, b, rcond=None)[0]

        # Average z deviation
        self.dev = np.dot(A, self.params) - b
        self.avg = np.sqrt(sum(self.dev*self.dev))/len(self.dev)
        self.maxdev = max(abs(self.dev))

        # Surface normal vector
        p0 = self.getvec(0, 0)
        p1 = self.getvec(1, 0) - p0
        p2 = self.getvec(0, 1) - p0
        x, y, z = np.cross(p1, p2)

        # Length of y projection and length of vector
        rxy = np.hypot(x, y)
        #r = np.hypot(rxy, z)

        # Plane slope and polar angle
        self.slope = rxy / z
        self.theta = np.arctan2(rxy, z) * 180.0 / np.pi

        # Azimuthal angle
        self.phi = np.arctan2(y, x) * 180.0 / np.pi
        while self.phi > 180.0:
            self.phi -= 360.0
        while self.phi <= -180.0:
            self.phi += 360.0


    def __str__(self):

        """ Return result string. """

        s = []
        s.append("polar angle:    %.1f° (%.2f %%)" % (self.theta, 100*self.slope))
        s.append("azimuth angle:  %.1f°" % self.phi)
        s.append("mean deviation: %.3f µm (max. %.3f µm)" % (self.avg, self.maxdev))
        return "\n".join(s)


    def log_results(self, func, name=None):

        """ Pass result string to given logger function line by line.
        Each line eventually preceeded by an optional name string. """

        for line in str(self).split("\n"):
            if name is not None:
                line = "%s %s" % (name, line)
            func(line)
            

    def getz(self, x, y):

        """ Return z value of the plane at the given lateral xy
        position. """

        return np.dot(self.params, [x, y, 1])
    

    def getvec(self, x, y):

        """ Return xyz vector with z value of the plane at the given
        lateral xy position. """

        z = self.getz(x, y)
        return np.array([x, y, z], dtype=float)


##########################################################################
class zPlane(object):

    def __init__(self, plane, key):

        """ Store plane parameters. """

        params = plane["meas/result.json"][key]
        sx = params["xSlope"]
        sy = params["ySlope"]
        z0 = params["z0"]
        self.params = np.array([sx, sy, z0], dtype=float)


    def getz(self, x, y):

        """ Return z value of the plane at the given lateral xy
        position. """

        return np.dot(self.params, [x, y, 1])
    
            
##########################################################################
class Plane(Parameter):

    """ Layer plane class. """

    _defaults = {
        "dzCoarseDefault": 100.0,
        "dzFineDefault": 10.0,
        }

    def __init__(self, zlo, zup, system, focus_args=None, layer_args=None, logger=None, config=None, **kwargs):

        """ Initialize the layer scan object. """

        # Initialize parameter class
        super().__init__(logger, config, **kwargs)
        self.log.info("Initializing plane detector.")

        # Store system object
        self.system = system

        # Initialize layer object
        layer_args = layer_args or {}
        self.layer = Layer(system, focus_args, logger, self.config, **layer_args)

        # Store initial values
        self.zlo = float(zlo)
        self.zup = float(zup)

        # No results yet
        self.steps = []
        self.log.info("Initialized plane detector.")


    def run(self, x, y, path=None, home=False):

        pos = len(self.steps)
        path = mkdir("%s/layer" % path, clean=False)
        path = mkdir("%s/layer-%02d" % (path, pos))

        # Store current xyz position
        x0, y0, z0 = self.system.position("XYZ")

        # Detect layer interfaces
        if self.zlo == self.zup:
            dz = self["dzCoarseDefault"]
        else:
            dz = self["dzFineDefault"]
        self.layer.run(x, y, self.zlo, self.zup, dz, path, home=False)
        l = self.layer.container()
        l.write("%s/layer.zdc" % path)

        # Append results of this step
        self.steps.append({
            "scan": pos,
            "x": x,
            "y": y,
            "zLowerInit": self.zlo,
            "zLower": result["zLower"],
            "dzLower": result["dzLower"],
            "zUpperInit": self.zup,
            "zUpper": result["zUpper"],
            "dzUpper": result["dzUpper"],
            "dzInit": dz,
            "layerUuid": l.uuid,
            })

        # Store results as estimates for the next scan
        self.zlo = result["zLower"]
        self.zup = result["zUpper"]
        
        # Move stages back to initial position
        if home:
            self.system.moveabs(wait="XYZ", x=x0, y=y0, z=z0)


    def _pop_results(self):

        steps = list(self.steps)
        points = [[s["x"], s["y"], s["zLower"], s["zUpper"]] for s in steps]
        points = np.array(points, dtype=float)
        lower = points[:,(0,1,2)]
        upper = points[:,(0,1,3)]

        result = {}
        for key, points, name in [("lower", lower, "Lower"),
                                  ("upper", upper, "Upper")]:
            plane = PlaneFit(points)
            plane.log_results(self.log.info, name)
            result[key] = {
                "xSlope": plane.params[0],
                "ySlope": plane.params[1],
                "z0": plane.params[2],
                "avgDeviation": plane.avg,
                "maxDeviation": plane.maxdev,
                "gradient": plane.slope,
                "polarAngle": plane.theta,
                "azimuthAngle": plane.phi,
                }
        
        self.steps = []
        return result, steps
    
    
    def container(self, config=None, **kwargs):

        """ Return results as SciDataContainer. """

        # Collect results
        if len(self.steps) == 0:
            raise RuntimeError("No results!")
        result, steps = self._pop_results()
        
        # Collect UUIDs of focus detections as references
        refs = {}
        for step in steps:
            key = "scan-%02d" % step["scan"]
            refs[key] = step["layerUuid"]

        # General metadata
        content = {
            "containerType": {"name": "DcLayerPlane", "version": 1.0},
            }
        meta = {
            "title": "TPP Layer Detection Data",
            "description": "",
            }

        # Container dictionary
        items = {
            "content.json": content,
            "meta.json": meta,
            "references.json": refs,
            "data/system.json": self.system.parameters(),
            "data/sample.json": self.system.sample,
            "data/parameter.json": self.parameters(),
            "meas/steps.json": steps,
            "meas/result.json": result,
            }

        # Return container object
        config = config or self.config
        return Container(items=items, config=config, **kwargs)
