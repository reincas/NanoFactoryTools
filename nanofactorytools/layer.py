##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides a photoresin layer detector class for a laser
# direct writing system.
#
##########################################################################

import math
#import random
import numpy as np
from scidatacontainer import Container
from nanofactorysystem import Parameter, mkdir

from .focus import S_FOCUS, S_NOFOCUS, Focus


##########################################################################
def flex_round(value, uncertainty):

    """ Round the given uncertainty value to 3 digits if its mantissa is
    in the range 1.0...2.9 and 2 digits in the range 3.0...9.9. Round
    the value to the same number of decimals as the uncertainty. """

    digits = 1-math.floor(math.log10(uncertainty/3.0))
    return round(value, digits), round(uncertainty, digits)


# ##########################################################################
# class FocusDetector(object):

#     """ Focus detection simulator for testing purposes. """

#     def __init__(self, lower, upper, jitter, fuzzy):

#         self.lower = float(lower)
#         self.upper = float(upper)
#         self.jitter_width = jitter
#         self.fuzzy_width = fuzzy

#     def jitter(self, z):

#         return z + self.jitter_width * (random.betavariate(5, 5) - 0.5)

#     def thres(self, x):

#         w = self.fuzzy_width
#         return 0.5*(1.0-np.sign(x)*(np.exp(-(abs(x)/w+np.log(np.e/2)-1))-1))
        
#     def detect(self, z, dz):

#         z0 = z - 0.5*dz
#         z1 = z + 0.5*dz

#         lower = self.jitter(self.lower)
#         upper = self.jitter(self.upper)
#         if z1 <= lower:
#             length = z1 - lower
#         elif z0 >= upper:
#             length = upper - z0
#         else:
#             length = min(z1, upper) - max(z0, lower)

#         if self.fuzzy_width <= 0.0:
#             result = length > 0.0
#         else:
#             thres = self.thres(length)
#             result = random.random() < thres
#         if (length > 0.0) != result:
#             print("=== simulated error: %.2f ===" % length)
#         return result


##########################################################################
class LayerBisect(object):

    """ Layer edge bisectioning class. It is used to determine the z
    position of the lower and upper photoresin interface.

    Usage:

    The method next() delivers the next z position and length to be
    exposed. The focus detection result of this exposure must be
    registered using the method register(). This method will return True
    when the bisectioning algorithm has finished and False otherwise.
    The method result() provides the current layer detection results.
    """

    # Translation of focus detection results to numerical values
    _hitvalue = {False: -1.0, None: 0.0, True: 1.0}
    
    def __init__(self, zmax, maxmult, zlo, zup, dz, beta, resolution):

        """ Initialize the layer scan object. """

        # Maximum allowed z value
        self.zmax = float(zmax)
        
        # Initial scan range is limited to maxmult*dz
        self.maxmult = float(maxmult)
        
        # Start from scratch with a single estimation for the layer
        # position
        if zlo == zup:
            
            # Initial z position and length of the scan
            self.z_init = float(zlo)
            self.dz_init = float(dz)

            # Initialize the array for the bisection estimates
            self.bisect = None
            self.edge = None

        # Skip the coarse detection phase with given estimations for the
        # lower and upper layer interface
        else:
            bisect = [[zlo-0.5*dz, zlo+0.5*dz],
                      [zup-0.5*dz, zup+0.5*dz]]
            self.bisect = np.array(bisect, dtype=float)

        # Scan length decrement factor
        self.beta = float(beta)

        # Required resolution
        self.resolution = float(resolution)

        # Initialize an empty list for the scan results
        self.results = None


    def stepline(self, mode="both"):

        """ Return x and y vector of the step function combining all
        current results. The weight of a focus detection is +1, the
        weight of a miss is -1. """

        # Start and end positions and weight value 
        z0 = self.results[:,0]
        z1 = self.results[:,1]
        value = self.results[:,2]

        # Sorted list of unique z positions
        pos = np.unique(np.concatenate((z0, z1)))

        # Mesh grids for start, end and value
        z, z0 = np.meshgrid(pos, z0)
        z, z1 = np.meshgrid(pos, z1)
        z, value = np.meshgrid(pos, value)

        # Value vector as sum of weights
        match = (z >= z0) & (z < z1)
        p = (match & (value > 0)).sum(axis=0)
        n = (match & (value < 0)).sum(axis=0)
        if mode == "hit":
            value = p[:-1]
        elif mode == "miss":
            value = -n[:-1]
        else:
            value = (p-n)[:-1]

        # x and y coordinates of the corner points of the step function
        x = np.array((pos[:-1], pos[1:])).T.ravel()
        y = np.array((value, value)).T.ravel()
        return x, y


    def bisect_init(self):

        """ Initialize the two layer edge bisectioning objects. """

        # Start and end positions and weight value 
        z0 = self.results[:,0]
        z1 = self.results[:,1]
        value = self.results[:,2]

        # Lower and upper edge of successful focus scans
        i = value > 0.0
        lower = np.min(z0[i])
        upper = np.max(z1[i])

        # Collect z ranges of failed focus scans
        i = value < 0.0
        z0_miss = z0[i]
        z1_miss = z1[i]

        # Determine all negative focus scans crossing the current lower
        # edge. Set corrected lower edge to the maximum of all these
        # scan ranges.
        i = (z0_miss < lower) & (z1_miss > lower)
        lower = np.max(z1_miss[i])
        
        # Determine all negative focus scans crossing the current upper
        # edge. Set corrected upper edge to the minimum of all these
        # scan ranges. The case upper == zmax (len(i) == 0) needs
        # special consideration.
        i = (z0_miss < upper) & (z1_miss > upper)
        if len(i) > 0:
            upper = np.max(z0_miss[i])

        # Initialize the array for the upper and lower estimates (axis
        # 1) for the lower and the upper edge (axis 0)
        self.bisect = np.zeros((2,2), dtype=float)
        self.bisect[:,0] = lower
        self.bisect[:,1] = upper


    def bisect_next(self):

        """ Return the center z coordinate and the length of the next
        scan range. """

        # Current center and length of range
        z = 0.5*(self.bisect[:,1] + self.bisect[:,0])
        dz = self.bisect[:,1] - self.bisect[:,0]

        # Take edge with larger estimated range
        edge = np.argmax(dz)

        # Lower edge: scan lower segment of the estimated range
        if edge == 0:
            z = z[edge] - 0.5*self.beta*dz[edge]
            dz = self.beta * dz[edge]
            
        # Upper edge: scan upper segment of the estimated range
        else:
            z = z[edge] + 0.5*self.beta*dz[edge]
            dz = self.beta * dz[edge]

        # Add some jitter to avoid series of ambiguous results at the
        # very edge of the layer
        #z = z + 0.05*dz*(random.random()-0.5)

        # Store current edge for registration
        self.edge = edge

        # Return the center z coordinate and the length of the next scan
        # range
        return z, dz


    def bisect_register(self, z, dz, hit):

        """ Set new lower and upper estimates for the layer edge based
        on the result of the last scan. """

        # Focus detection was ambiguous
        if hit is None:
            return

        # Focus detected: scan range is new estimate
        if hit:
            lower = z - 0.5*dz
            upper = z + 0.5*dz

        # No focus detected
        else:
            
            # Lower edge: section above scan range is new estimate
            if self.edge == 0:
                lower = z + 0.5*dz
                upper = z + 1.5*dz

            # Upper edge: section below scan range is new estimate
            else:
                lower = z - 1.5*dz
                upper = z - 0.5*dz

        # Store new estimation range            
        self.bisect[self.edge,0] = lower
        self.bisect[self.edge,1] = upper


    def clip(self, z, dz):

        """ Make sure that the range does not exceed zmax. Reduce the
        upper end accordingly. Throw an ValueError, if clipping is
        impossible, because the whole range exceeds zma.  """

        # Apply zmax clipping
        z0 = z - 0.5*dz
        z1 = min(self.zmax, z + 0.5*dz)

        # Clipping is impossible
        if z1 <= z0:
            raise ValueError("Focus scan failed!")

        # Return new range
        return 0.5*(z0+z1), z1-z0
    

    def next(self):

        """ Return center and length of next scan range. """

        # Step 4b: Edge bisectioning
        if self.bisect is not None:
            z, dz = self.bisect_next()

            # Failing at this point is virtually impossible
            return self.clip(z, dz)

        # Step 1: Start with initial guess
        if self.results is None or not any(self.results[:,-1] != 0.0):
            dz = self.dz_init
            z = self.z_init

            # Failing at this initial point is fatal
            return self.clip(z, dz)

        # Extract previous start and end positions and weight values
        z0 = self.results[:,0]
        z1 = self.results[:,1]
        value = self.results[:,2]

        # Step 2: Increase scanned range until at least one positive and
        # one negative focus scan was encountered
        if (np.all(value > 0.0) or np.all(value < 0.0)):

            # Avoid an endless search if there is something wrong with
            # the laser beam position.
            if z1.max() - z0.min() > self.maxmult*self.dz_init:
                raise RuntimeError("No layer found. Check laser!")

            # Upper end expansion
            if len(self.results) % 2 == 0:
                dz = self.dz_init
                z = np.max(z1) + 0.4*dz

                # Failing at this point forces lower end expansion
                try:
                    return self.clip(z, dz)
                except:
                    pass

            # Lower end expansion
            dz = self.dz_init
            z = np.min(z0) - 0.4*dz
            
            # Failing at this point is virtually impossible
            return self.clip(z, dz)

        # Step 3a: Run until the two lowest scans are negative
        imin = np.argsort(z0)
        if not np.all(value[imin[:2]] < 0.0):
            dz = self.dz_init
            z = z0[imin[0]] - 0.4*dz
            
            # Failing at this point is virtually impossible
            return self.clip(z, dz)
            
        # Step 3b: Run until the two highest scans are negative
        imax = np.argsort(z1)
        if not np.all(value[imax[-2:]] < 0.0):
            dz = self.dz_init
            z = z1[imax[-1]] + 0.4*dz

            # Failing at this point is not critical, proceed with 4a
            try:
                return self.clip(z, dz)
            except:
                pass

        # Step 4a: Initialize and start edge bisectioning
        self.bisect_init()
        z, dz = self.bisect_next()

        # Failing at this point is virtually impossible
        return self.clip(z, dz)
    

    def register(self, z, dz, hit):

        """ Register focus detection result (True = focus, None =
        ambiguous, False = no focus). Return True when the edge
        bisectioning process has finished. """

        # Not yet finished
        finished = False
        
        # Append result to the list of results
        row = [z-0.5*dz, z+0.5*dz, self._hitvalue[hit]]
        if self.results is None:
            self.results = np.array([row], dtype=float)
        else:
            self.results = np.concatenate((self.results, [row]), axis=0)

        # Update estimation range for edge bisection
        if self.bisect is not None:
            self.bisect_register(z, dz, hit)
            
            # End the layer scan when the uncertainty of the edge
            # estimates is smaller than the requested accuracy
            if np.all((self.bisect[:,1] - self.bisect[:,0]) < self.resolution):
                finished = True

        # Done
        return finished


    def result(self):

        """ Return the range of scanned z positions when the edge
        bisectioning has not yet started. In the edge bisectioning stage
        return the current estimation for the z positions of lower and
        upper edge. """

        # Default values
        dz_lower = 0.0
        dz_upper = 0.0
        bisect = False

        # Edge detection has not yet started
        if self.result is None:
            z_lower = self.z_init - 0.5*self.dz_init
            z_upper = self.z_init + 0.5*self.dz_init

        # Edge bisectioning has not yet started
        elif self.bisect is None:
            z_lower = np.min(self.results[:,0])
            z_upper = np.max(self.results[:,1])

        # Edge bisectioning is running
        else:
            z_lower, z_upper = 0.5 * (self.bisect[:,1] + self.bisect[:,0])
            dz_lower, dz_upper = self.bisect[:,1] - self.bisect[:,0]
            bisect = True

        # Done
        return bisect, z_lower, z_upper, dz_lower, dz_upper


##########################################################################
class Layer(Parameter):

    """ Layer edge detector class. After initializing just call the
    method detect() to run the edge detection. """

    _defaults = {
        "xCenter": None,
        "yCenter": None,
        "zLowerInit": None,
        "zUpperInit": None,
        "dzInit": None,
        "dzCoarseDefault": 100.0,
        "dzFineDefault": 10.0,
        "zMax": None,
        "maxMult": 5.0,
        "laserPower": 0.7,
        "stageSpeed": 200.0,
        "duration": 0.2,
        "beta": 0.7,
        "resolution": 0.5,
        "lateralPitch": 4.0,
        }

    # Translation of focus detection results to numerical values
    _hitvalue = {False: "no focus", None: "ambiguous", True: "focus"}
    
    
    def __init__(self, system, focus_args=None, logger=None, config=None, **kwargs):

        """ Initialize the layer scan object. """

        # Initialize parameter class
        super().__init__(logger, config, **kwargs)
        self.log.info("Initializing layer detector.")

        # Store system object
        self.system = system
        self["zMax"] = self.system["zMax"]
        
        # Initialize focus detection
        focus_args = focus_args or {}
        self.focus = Focus(self.system, logger, **focus_args)
        
        # No result yet
        self.steps = None
        self.result = None
        self.log.info("Initialized layer detector.")


    def _spiral(self, pos):

        """ Return x, y coordinates along a quadratic spiral starting at
        the given center point. The value of the lateral pitch is
        given by the respective detector parameter. """

        # Center point
        if pos == 0:
            i, j = 0, 0

        else:
            # Radius 1, 2, 3, ...
            fp, r = math.modf(0.5*(math.sqrt(pos)+1))
            r = round(r)

            # Side 0, 1, 2, or 3
            v = pos - (4*r*(r-1)+1)
            s = v // (2*r)

            # Position on the side -r+1 ... +r
            p = (v % (2*r)) - r + 1

            # Integer x, y coordinates relative to center point
            if s == 0:
                i, j = p, -r
            elif s == 1:
                i, j = r, p
            elif s == 2:
                i, j = -p, r
            else:
                i, j = -r, -p

        # Return the coordinates
        x = self["xCenter"] + i*self["lateralPitch"]
        y = self["yCenter"] + j*self["lateralPitch"]
        return x, y


    def run(self, x, y, zlo, zup, dz=None, path=None, home=True):

        """ Main method of the layer detector. Find the z coordinates of
        both interfaces of the photoresist layer. Return results as layer
        container object. """

        self.log.info("Photoresist layer detection started.")

        # Store current xyz position
        x0, y0, z0 = self.system.position("XYZ")

        # Detect interfaces
        if dz is None:
            if zlo == zup:
                dz = self["dzCoarseDefault"]
            else:
                dz = self["dzFineDefault"]
        self.result, self.steps = self._detect(x, y, zlo, zup, dz, path)
        self.log.info("Lower interface: %g [%g]" \
                      % flex_round(self.result["zLower"], self.result["dzLower"]))
        self.log.info("Upper interface: %g [%g]" \
                      % flex_round(self.result["zUpper"], self.result["dzUpper"]))
        self.log.info("Photoresist layer detection finished.")

        # Move stages back to initial position
        if home:
            self.system.moveabs(wait="XYZ", x=x0, y=y0, z=z0)
        

    def _detect(self, x, y, zlo, zup, dz, path=None):

        """ Run the layer detection algorithm and return the final and
        intermediate results. """

        # Center position of the layer detection spiral
        self["xCenter"] = float(x)
        self["yCenter"] = float(y)

        # Initial z position and length of the scan
        self["zLowerInit"] = float(zlo)
        self["zUpperInit"] = float(zup)
        self["dzInit"] = float(dz)

        # Initialize the bisectioning object
        zmax = self["zMax"]
        maxmult = self["maxMult"]
        beta = self["beta"]
        resolution = self["resolution"]
        bisect = LayerBisect(zmax, maxmult, zlo, zup, dz, beta, resolution)

        power = self["laserPower"]
        speed = self["stageSpeed"]
        duration = self["duration"]

        # Run layer detection algorithm
        if path:
            path = mkdir("%s/focus" % path)
        finished = False
        steps = []
        while not finished:
            pos = len(steps)
            if path:
                subpath = mkdir("%s/focus-%02d" % (path, pos))

            # Next exposure coordinates
            x, y = self._spiral(pos)
            z, dz = bisect.next()

            # Expose an axial line and detect the focus
            self.focus.run(x, y, z, dz, power, speed, duration)
            f = self.focus.container()
            if path:
                self.focus.imgPre.write("%s/image_pre.zdc" % subpath)
                self.focus.imgPost.write("%s/image_post.zdc" % subpath)
                f.write("%s/focus.zdc" % subpath)

            # Hit status
            status = self.focus.result["status"]
            if status == S_FOCUS:
                hit = True
            elif status == S_NOFOCUS:
                hit = False
            else:
                hit = None

            # Register the focus detection result and get the current
            # layer detection status
            finished = bisect.register(z, dz, hit)
            bi, z_lower, z_upper, dz_lower, dz_upper = bisect.result()

            # Store parameters and results of this step
            steps.append(dict(self.focus.exposure).update({
                "exposure": pos,
                "focusStatus": self._hitvalue[hit],
                "focusUuid": f.uuid,
                "bisection": bi,
                "zLower": z_lower,
                "dzLower": dz_lower,
                "zUpper": z_upper,
                "dzUpper": dz_upper,
                "finished": finished,
                }))

            # Send info line to logger
            if bi:
                zlo = "%g [%g]" % flex_round(z_lower, dz_lower)
                zup = "%g [%g]" % flex_round(z_upper, dz_upper)
                s = "bisect %16s %16s" % (zlo, zup)
            else:
                zlo, dzlo = flex_round(z_lower, dz)
                zup, dzup = flex_round(z_upper, dz)
                s = "prepare  %g ... %g [%g]" % (zlo, zup, dzup)
            step = "Step %d:" % pos
            z = "%g [%g]" % flex_round(z, dz)
            hit = self._hitvalue[hit]
            self.log.debug("%-9s %16s -> %-9s | %s" % (step, z, hit, s))

        # Final result
        result = {
            "zLower": z_lower,
            "dzLower": dz_lower,
            "zUpper": z_upper,
            "dzUpper": dz_upper,
            }

        # Return final and intermediate results
        return result, steps


    def container(self, config=None, **kwargs):

        """ Return results as SciDataContainer. """

        if self.result is None:
            raise RuntimeError("No results!")

        # Collect UUIDs of focus detections as references
        refs = {}
        for step in self.steps:
            key = "focus-%02d" % step["exposure"]
            refs[key] = step["focusUuid"]

        # General metadata
        content = {
            "containerType": {"name": "DcLayerDetect", "version": 1.0},
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
            "meas/steps.json": self.steps,
            "meas/result.json": self.result,
            }

        # Return container object
        config = config or self.config
        return Container(items=items, config=config, **kwargs)
