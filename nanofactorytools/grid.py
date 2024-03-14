##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides a grid detector for microscope images.
#
##########################################################################

import bisect
import numpy as np
import cv2 as cv
from scipy.ndimage import maximum_filter
from scidatacontainer import Container
from nanofactorysystem import Parameter, popargs

from . import image


def detectGrid(img, logger, **params):

    """ Point grid detector. The algorithm detects the focus spots in the given
    burred difference image. The algorithm estimates a rotated quadratic mesh
    grid on which the spots are located. The algorithm is designed to be
    roubust against missing grid points and outliers, which are rejected.

    Return values: Nx2 array of valid spot locations, Nx2 array of
    integer grid positions of these spots, floating point edge length of
    the quadratic mesh elements in pixels, rotation angle of the grid
    with respect to the image axes in degrees, and grid size in (nx, ny)
    determined by the outmost spots. """

    # Parameters for the determination of outliers
    mindist = params["detectMinDist"]
    minstep = params["detectMinStep"]
    minsize = params["detectMinSize"]
    minangle = params["detectMinAngle"]
    minoff = params["detectMinOff"]
    maxdist = params["detectMaxDist"]

    # Binary mask based on maximum filter
    img = maximum_filter(img, size=mindist)
    mean = 0.5 * (np.min(img) + np.max(img))
    img = np.where(img > mean, 1, 0).astype(np.uint8)

    # Detect all external contours in binary mask
    contours = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

    # Determine center point of each contour
    points = []
    for cnt in contours:
        M = cv.moments(cnt)
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])
        points.append((x, y))
    points = np.array(points, dtype=int)

    ### DEBUG: Remove and add some points
    #idx = (3, 16, 27, 33)
    #points = np.array([points[i,:] for i in range(len(points)) if i not in idx])
    #points = np.concatenate((points, [[100, 200], [345, 202]]), axis=0)
    ### DEBUG

    # Determine the distance vector and its magnitude for every
    # unordered pair of points
    n = len(points)
    i = np.indices((n, n)).reshape((2,n*n)).T
    i = i[i[:,0] < i[:,1],:]
    dist = points[i[:,0],:] - points[i[:,1],:]
    d = np.sqrt((dist*dist).sum(axis=1))

    # Determine dmean as estimated distance of adjacent points. Group
    # increasing distances by length increments exceeding minstep. The
    # group of smallest distances determines the grid pitch. However,
    # small distance groups with less than minsize members are
    # considered as outliers and discarded.
    dsort = np.sort(d)
    groups = np.split(dsort, np.where(np.diff(dsort) > minstep)[0] + 1)
    k = 0
    while len(groups[k]) < minsize:
        k += 1
    if k != 0:
        logger.warn("Outliers expected!")
    if k >= len(groups):
        logger.error("Detection of grid pitch failed!")
        raise RuntimeError("Detection of grid pitch failed!")
    dmean = groups[k].mean()

    # Get the indices of all point pairs in the determined group. Grid
    # pitch is the mean distance of the point pairs in the group.
    maxdiff = 1.2*(groups[k].max()-groups[k].min())
    jmin = np.argwhere(abs(d-dmean) < maxdiff).ravel()
    if len(jmin) < 4:
        logger.error("Detection of grid pitch failed!")
        raise RuntimeError("Detection of grid pitch failed!")
    pitch = d[jmin].mean()
    logger.info("Grid pitch: %.1f px" % pitch)

    # Rotation angles of the point pairs in the range between -pi/4 to
    # +pi/4. Valid pair distance vectors may direct in four directions:
    # angle + i*pi/2 with i=[0,1,2,3]. The rotation angles are
    # determined as the remainder of a division of the direction angles
    # by pi/2. Angles above pi/4 are fliped to pi/2-angle.
    pi2 = np.pi / 2
    pi4 = np.pi / 4
    dist = dist[jmin,:]
    angle = np.arctan2(dist[:,1], dist[:,0])
    angle = angle % pi2
    angle = np.where(angle < pi4, angle, pi2-angle)

    # Group the sorted list of rotation angles by increments larger than
    # minangle. The rotation angle of the grid is determined as the mean
    # angle of the largest group. All point pairs outside this group are
    # considered as outliers and being ignored.
    angle = np.sort(angle)
    groups = np.split(angle, np.where(np.diff(angle) > minangle)[0]+1)
    k = np.argmax([len(g) for g in groups])
    if len(groups) > 1:
        num = len(angle) - len(groups[k])
        angle = groups[k]
        logger.warn("%d outliers ignored (angle)!" % num)
    angle = angle.mean()
    alpha = angle * 180.0/np.pi
    logger.info("Grid angle: %.3f°" % (alpha))

    # Rotate points back to achieve horizontal and vertical grid
    # lines
    cosa = np.cos(angle)
    sina = np.sin(angle)
    pt = np.dot(points, np.array([[cosa, -sina], [sina, cosa]]))

    # Determine xy-offset of grid in pixels. All point offsets are
    # sorted and grouped by increments exceeding minoff relative to the
    # grid pitch. The offset is determined as mean value of the largest
    # group. All points outside this group are considered as outliers
    # and being ignored.
    offset = []
    for i in range(2):
        off = np.sort((pt[:,i] / pitch) % 1.0)
        groups = np.split(off, np.where(np.diff(off) > minoff)[0]+1)
        k = np.argmax([len(g) for g in groups])
        if len(groups) > 1:
            num = len(off) - len(groups[k])
            off = groups[k]
            axis = ("x", "y")[i]
            logger.warn("%d outliers ignored (offset %s)!" % (num, axis))
        offset.append(off.mean() * pitch)
    ox, oy = offset

    # Get grid indices of all points
    pt[:,0] = (pt[:,0] - ox) / pitch
    pt[:,1] = (pt[:,1] - oy) / pitch
    idx = np.round(pt).astype("int")

    # Determine distance of each point from the next grid point. All
    # points with a distance relative to the grid pitch above maxdist
    # are considered as outliers and being removed.
    dx = pt - idx
    dx = np.sqrt((dx*dx).sum(axis=1))
    i = np.argwhere(dx < maxdist).ravel()
    if len(i) != len(points):
        num = len(points) - len(i)
        points = points[i,:]
        idx = idx[i,:]
        logger.warn("%d outliers removed (grid)!" % num)

    # Determine grid size
    idx[:,0] -= idx[:,0].min()
    idx[:,1] -= idx[:,1].min()
    nx = idx[:,0].max() + 1
    ny = idx[:,1].max() + 1

    # Return points and parameters
    return points, idx, pitch, alpha, nx, ny


def getTransform(src, w, h, dst, nx, ny, dx, dy):

    """ Map the source coordinates in pixels to a grid in micrometres.
    The positive source coordinates lead to points on a (w, h) image.
    The destination coordinates are relative coordinates of a grid of
    size (nx, ny) and pitch (dx, dy) in micrometres. Both point sets are
    centered before the mapping.

    Return values: Transformation matrix A of size 2x3 and the average
    derivation of the points from their respective grid position in
    micrometres.

    x' = A[0,0]*x + A[0,1]*y + A[0,2]
    y' = A[1,0]*x + A[1,1]*y + A[1,2]

    with coordinates (x',y') in micrometres and (x,y) in pixels. """

    # Center image coordinates
    src = np.array(src)
    src[:,0] = src[:,0] - w/2
    src[:,1] = src[:,1] - h/2

    # Center destination grid coordinates in micrometres
    x = (dst[:,0] - (nx-1)/2) * dx
    y = (dst[:,1] - (ny-1)/2) * dy
    dst = np.stack((x, y), axis=1)

    # Get transformation matrix
    A, inliers = cv.estimateAffine2D(src, dst, True)
    if not all(inliers):
        num = len(src) - inliers.sum()
        print("***** Estimation of transform failed for %d points!" % num)
        #raise RuntimeError("Estimation of transform failed!")

    # Mean derivation of a single point
    x = np.concatenate((src, np.ones((len(src), 1))), 1).T
    dr = np.dot(A, x).T - dst
    dr = np.sqrt((dr*dr).sum(axis=1))
    dr = dr.sum() / len(dr)

    # Return results
    return A, dr


##########################################################################
# Grid class
##########################################################################

class Grid(Parameter):

    _defaults = {
        "xNum": None,
        "yNum": None,
        "gridPitch": None,
        "laserPower": None,
        "duration": None,
        "x": None,
        "y": None,
        "z": None,
        "detectMinDist": 16,
        "detectMinStep": 8.0,
        "detectMinSize": 12,
        "detectMinAngle": 5e-3,
        "detectMinOff": 2e-2,
        "detectMaxDist": 0.45,
        "focusCenterSize": 5,
        "focusInitialNum": 9,
        "focusInitialStep": 2.5,
        "focusMinStep": 0.1,
        }

    def __init__(self, system, logger=None, **kwargs):

        """ Initialize the camera calibration class. """

        # Store system object
        self.system = system
        user = self.system.user["key"]

        # Initialize parameter class
        args = popargs(kwargs, "grid")
        super().__init__(user, logger, **args)
        self.log.info("Initializing camera calibration.")

        # No results yet
        self.preImg = None
        self.postImg = None
        self.imgParams = None
        self.result = None
        self.log.info("Initialized camera calibration.")


    def _focusValue(self, z, points, size, values):

        """ Quantify the focus quality of the grid spots at given position of
        the z stage utilizing the fact that the peak value of the Airy pattern
        of a spot is maximized in the focal plane of the camera.

        Pick a small quadratic region with given size centered at each grid
        spot from the given list. Return the maximum value from the mean
        values of each pixel in the region. The size must be an odd positive
        number. """

        # Get camera image at given position of the z stage
        delay = self.system["delay"]
        self.system.moveabs(z=z, wait=delay)
        img = self.system.getimage().img

        # Difference image with pre-exposure image
        img = image.diff(self.preImg, img)

        # Determine quality factor
        r = (size - 1) // 2
        simg = np.zeros((size, size), dtype=float)
        for x, y in points:
            simg += img[y-r:y+r+1,x-r:x+r+1]
        simg /= len(points)
        q = simg.max()

        # Return quality factor
        values.append((z, q))
        return q


    def _autoFocus(self, z0, num, step, minstep, points, size):

        # Get num focus values at distance step around z0
        z = z0 - 0.5 * (num - 1) * step
        values = []
        while len(values) < num:
            self._focusValue(z, points, size, values)
            z += step

        # Maximum focus value
        imax = np.argmax([v for z,v in values])
        zc, vc = values[imax]

        # Next lower z position
        if imax > 0:
            zlo, vlo = values[imax-1]
        else:
            zlo, vlo = zc - step, None

        # Next higher z position
        if imax < len(values)-1:
            zhi, vhi = values[imax+1]
        else:
            zhi, vhi = zc + step, None

        # Eventually get focus value of next lower z position
        while vlo is None:
            vlo = self._focusValue(zlo, points, size, values)
            if vlo > vc:
                zhi, vhi = zc, vc
                zc, vc = zlo, vlo
                zlo, vlo = zlo - step, None

        # Eventually get focus value of next higher z position
        while vhi is None:
            vhi = self._focusValue(zhi, points, size, values)
            if vhi > vc:
                zlo, vlo = zc, vc
                zc, vc = zhi, vhi
                zhi, vhi = zhi + step, None

        # Bisectioning algorithm: Find maximum focus value
        while max(zc-zlo, zhi-zc) > minstep:
            self.log.info("%.1f, %.1f   %.1f, %.1f   %.1f, %.1f" % (zlo, vlo, zc, vc, zhi, vhi))

            if zc-zlo > zhi-zc:
                z = (zlo + zc) / 2
                v = self._focusValue(z, points, size, values)
                if v <= vc:
                    zlo, vlo = z, v
                else:
                    zhi, vhi = zc, vc
                    zc, vc = z, v
            else:
                z = (zc + zhi) / 2
                v = self._focusValue(z, points, size, values)
                if v <= vc:
                    zhi, vhi = z, v
                else:
                    zlo, vlo = zc, vc
                    zc, vc = z, v

        # Sort list of all determined focus quality factors
        i = np.argsort([z for z,v in values])
        values = np.array(values)[i]
        self.log.info("%.1f, %.1f   %.1f, %.1f   %.1f, %.1f" % (zlo, vlo, zc, vc, zhi, vhi))
        return zc, values


    def run(self, x0, y0, z0, nx, ny):

        """ Expose a grid of (nx,ny) spots in the center of the photoresin.
        Generate the list of pre-exposure images, the list of post-exposure
        images and the common image parameter dictionary. """

        # Dictionary for results
        self.result = {}
        
        # Store grid parameters
        self["x"] = x0
        self["y"] = y0
        self["z"] = z0
        self["xNum"] = nx
        self["yNum"] = ny

        # Take pre-exposure image
        self.log.info("Take pre-exposure image")
        delay = self.system["delay"]
        x, y, z = self.system.stage_pos([x0, y0, z0], [0, 0])
        self.system.moveabs(x=x, y=y, z=z, wait=delay)
        dc = self.system.getimage()
        self.preImg = dc.img
        self.imgParams = dc.params

        # Expose grid
        pitch_um = self["gridPitch"]
        power = self["laserPower"]
        dt = self["duration"]
        self.log.info("Expose %d x %d points (pitch: %.1f µm)" % (nx, ny, pitch_um))
        for i in range(nx):
            for j in range(ny):
                x = x0 + pitch_um*(i - (nx-1)/2)
                y = y0 + pitch_um*(j - (ny-1)/2)
                self.system.moveabs(x=x, y=y, z=z0, wait=delay)
                self.system.pulse(power, dt)

        # Take post exposure image
        self.log.info("Take post-exposure image")
        x, y, z = self.system.stage_pos([x0, y0, z0], [0, 0])
        self.system.moveabs(x=x, y=y, z=z, wait=delay)
        self.postImg = self.system.getimage().img

        # Localize grid spots on the difference image
        params = self.parameters()
        blur = params["detectMinDist"]
        diff = image.diff(self.preImg, self.postImg, blur)
        src, dst, pitch_px, angle, nxd, nyd = detectGrid(diff, self.log, **params)
        self.log.info("Detected grid size:  %d x %d" % (nxd, nyd))
        self.log.info("Detected grid pitch: %.1f px" % pitch_px)
        self.log.info("Detected grid angle: %.3f°" % angle)
        if nxd != nx or nyd != ny:
            self.log.error("Wrong grid size!")
            raise RuntimeError("Wrong grid size!")

        # Run autofocus on grid spots
        self.log.info("Start camera grid autofocus.")
        num = self["focusInitialNum"]
        step = self["focusInitialStep"]
        minstep = self["focusMinStep"]
        size = self["focusCenterSize"]
        assert isinstance(size, int) and size > 0 and size % 2 == 1
        zmax, values = self._autoFocus(z, num, step, minstep, src, size)
        z_off = zmax - z0
        values[:,0] -= z0
        self.result["focusOffset"] = z_off
        self.result["offsetValues"] = values.tolist()
        self.log.info("Camera grid autofocus finished.")
        self.log.info("Axial offset of camera focus: %.1f µm" % z_off)

        # Update current coordinate transformation matrix
        self.system.update_pos("auto", z_off, minstep)

        # Get coordinate transformation matrix
        h, w = diff.shape
        A, dr = getTransform(src, w, h, dst, nx, ny, pitch_um, pitch_um)

        # Update current coordinate transformation matrix
        self.system.update_pos("grid", A, dr)
        self.result["affineTransformation"] = self.system.transform.P.tolist()
        self.result["spotDerivation"] = dr
        


    def container(self, config=None, **kwargs):

        """ Return results as SciDataContainer. """

        if self.result is None:
            raise RuntimeError("No results!")

        # General metadata
        content = {
            "containerType": {"name": "PointGrid", "version": 1.1},
            }
        meta = {
            "title": "Camera Calibration",
            "description": "Grid-based camera calibration with auto-focus.",
            }

        # Package dictionary
        items = {
            "content.json": content,
            "meta.json": meta,
            "data/system.json": self.system.parameters(),
            "data/grid.json": self.parameters(),
            "data/camera.json": self.imgParams,
            "meas/image_pre.png": self.preImg,
            "meas/image_post.png": self.postImg,
            "meas/result.json": self.result,
            }

        # Return container object
        config = config or self.config
        return Container(items=items, config=config, **kwargs)

