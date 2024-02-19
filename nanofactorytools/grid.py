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
from scidatacontainer import Container
from nanofactorysystem import Parameter

from . import image


def detectGrid(img, debug=False):

    """ Point grid detector. The algorithm detects circular spots (Airy
    disks) on the given image after application of a Gaussian blur with
    given radius. The algorithm estimates a rotated quadratic mesh grid
    on which the spots are located. The algorithm is designed to be
    roubust against missing grid points and outliers, which are
    rejected.

    Return values: Nx2 array of valid spot locations, Nx2 array of
    integer grid positions of these spots, floating point edge length of
    the quadratic mesh elements in pixels, rotation angle of the grid
    with respect to the image axes in degrees, and grid size in (nx, ny)
    determined by the outmost spots. """

    # Parameters for the determination of outliers
    minstep = 8.0 # 10.0
    minsize = 12
    minangle = 5e-3 # 5e-3
    minoff = 2e-2 # 5e-3
    maxdist = 0.45 # 0.05
    
    # Blob detector parameters for circle detection
    params = cv.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 5.0 # 50.0
    #params.minRepeatability = 2
    #params.filterByColor = True
    #params.blobColor = 0
    params.filterByArea = True
    params.minArea = 50.0
    params.maxArea = 5000.0
    params.minThreshold = 10.0
    params.maxThreshold = 240.0
    params.thresholdStep = 10.0
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.maxCircularity = np.finfo("float32").max
    params.filterByInertia = False
    params.minInertiaRatio = 0.4
    params.maxInertiaRatio = np.finfo("float32").max
    params.filterByConvexity = True
    params.minConvexity = 0.9
    params.maxConvexity = np.finfo("float32").max
    #params.collectContours = False

    # Find circular points in the image using the blob detector
    detector = cv.SimpleBlobDetector.create(params)
    mask = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    points = detector.detect(255-mask)
    points = np.array([k.pt for k in points])
    detect = np.array(points)

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
    groups = np.split(dsort, np.where(np.diff(dsort) > minstep)[0]+1)
    k = 0
    while len(groups[k]) < minsize:
        k += 1
    if debug and k != 0:
        print("*** Warning: Outliers expected!")
    if k >= len(groups):
        raise RuntimeError("Grid detection failed!")
    dmean = groups[k].mean()

    # Get the indices of all point pairs in the determined group. Grid
    # pitch is the mean distance of the point pairs in the group.
    maxdiff = 1.2*(groups[k].max()-groups[k].min())
    jmin = np.argwhere(abs(d-dmean) < maxdiff).ravel()
    if len(jmin) < 4:
        raise RuntimeError("Grid detection failed!")
    pitch = d[jmin].mean()
    if debug:
        print("Grid pitch: %.1f px" % pitch)

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
        if debug:
            print("*** Warning: %d outliers ignored (angle)!" % num)
    angle = angle.mean()
    alpha = angle * 180.0/np.pi
    if debug:
        print("Grid angle: %.3f°" % (alpha))

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
            if debug:
                axis = ("x", "y")[i]
                print("*** Warning: %d outliers ignored (offset %s)!" % (num, axis))
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
        if debug:
            print("*** Warning: %d outliers removed (grid)!" % num)

    # Determine grid size
    idx[:,0] -= idx[:,0].min()
    idx[:,1] -= idx[:,1].min()
    nx = idx[:,0].max() + 1
    ny = idx[:,1].max() + 1

    # Return points and parameters
    return points, idx, pitch, alpha, nx, ny, detect


def focusRadius(img, center, blur=4, size=20, thres=0.2):

    """ Determine focus quality based on the size of a set of Airy disks
    at given locations on a camera image.

    Split the given image into quadratic tiles of edge length 2*size+1
    centered at the given center points. The size of tiles exceeding
    image borders are reduced symmetrically. For each tile the number of
    high level pixels is determined. The threshold for high level is
    given relative to the maximum pixel value in the respective tile.

    Return value: Average effective radius of the high level area in
    pixels. """

    # Blur the camera image
    img = image.blur(img, blur)

    # Determine and evaluate every tile
    h, w = img.shape
    rnew = []
    for i in range(len(center)):

        # Bounding box of the tile
        cx, cy = center[i]
        x0 = round(cx - size)
        x1 = round(cx + size)
        y0 = round(cy - size)
        y1 = round(cy + size)
    
        # Correct bounding box at image edges
        if x0 < 0:
            x0, x1 = 0, x0+x1
        if x1 > w:
            x0, x1 = x0+x1-w, w
        if y0 < 0:
            y0, y1 = 0, y0+y1
        if y1 > h:
            y0, y1 = y0+y1-h, h

        # Number of pixels above threshold
        v = np.sort(img[y0:y1,x0:x1].ravel())
        v /= v[-1]
        rnew.append(len(v) - bisect.bisect(v, thres))

    # Return the average effective radius of the high level area in
    # pixels
    return np.sqrt(np.array(rnew).mean() / np.pi)
    

def getTransform(src, w, h, dst, nx, ny, dx, dy):

    """ Map the source coordinates in pixels to a grid in micrometres.
    The positive source coordinates lead to points on a (w, h) image.
    The destination coordinates are relative coordinates of a grid of
    size (nx, ny) and pitch (dx, dy) in micrometres. Both point sets are
    centered before the mapping.

    Return values: Transformation matrix A of size 2x3, scaling factors
    (ax, ay) in micrometres per pixel, shear angles (bx, by) in degrees,
    center offset (Fx, Fy) in pixel, and the average derivation of the
    points from their respective grid position in micrometres.

    x' = A[0,0]*x + A[0,1]*y + A[0,2]
    y' = A[1,0]*x + A[1,1]*y + A[1,2]

    x' =  ax*cos(bx)*(x-Fx) + ay*sin(by)*(y-Fy) 
    y' = -ax*sin(bx)*(x-Fx) + ay*cos(by)*(y-Fy)

    with coordinates (x',y') in micrometres and (x,y) in pixels. """

    # Center image coordinates
    src = np.array(src)
    src[:,0] -= w/2
    src[:,1] -= h/2

    # Center destination grid coordinates in micrometres
    x = (dst[:,0] - (nx-1)/2) * dx
    y = (dst[:,1] - (ny-1)/2) * dy
    dst = np.stack((x, y), axis=1)

    # Get transformation matrix
    A, inliers = cv.estimateAffine2D(src, dst, True)
    if not all(inliers):
        num = len(src) - inliers.sum()
        print("Estimation of transform failed for %d points!" % num)
        #raise RuntimeError("Estimation of transform failed!")

    # Scale factors in µm/pixel
    ax, ay = np.sqrt((A[:,:2]*A[:,:2]).sum(axis=0))

    # Shear angles in degrees
    bx = np.arctan2(-A[1,0], A[0,0]) * 180.0/np.pi
    by = np.arctan2(A[0,1], A[1,1]) * 180.0/np.pi

    # Mean derivation of a single point
    x = np.concatenate((src, np.ones((len(src), 1))), 1).T
    dr = np.dot(A, x).T - dst
    dr = np.sqrt((dr*dr).sum(axis=1))
    dr = dr.sum() / len(dr)

    # Focus offset from camera center in pixel
    M, b = A[:,:2], -A[:,2]
    Fx, Fy = np.linalg.solve(M, b)

    # Return results
    return A, ax, ay, bx, by, Fx, Fy, dr


##########################################################################
# Grid class
##########################################################################

class Grid(Parameter):

    _defaults = {
        "xNum": None,
        "yNum": None,
        "zNum": None,
        "xStep": None,
        "yStep": None,
        "zStep": None,
        "xOffset": None,
        "yOffset": None,
        "zOffset": None,
        "laserPower": None,
        "duration": None,
        "zLower": None,
        "zUpper": None,
        "x": None,
        "y": None,
        "z": None
        }

    def __init__(self, system, config=None, logger=None, **kwargs):

        """ Initialize the camera calibration class. """

        # Initialize parameter class
        super().__init__(logger, config, **kwargs)
        self.log.info("Initializing camera calibration.")

        # Store system object
        self.system = system
        
        # No results yet
        self.preImg = None
        self.postImg = None
        self.imgParams = None
        self.log.info("Initialized camera calibration.")
        

    def _getimages(self, z0, nz, dz):
        
        """ Return a list of camera images and an image parameter
        dictionary. """

        images = []
        for i in range(nz):
            z = z0 + (i - (nz-1)/2)*dz
            self.system.moveabs(z=z, wait="Z")
            dc = self.system.getimage()
            img = dc["meas/image.png"]
            images.append(img)

        params = dc["meas/image.json"]
        return images, params


    def run(self, x0, y0, nx, ny):
        
        """ Expose a grid of (nx,ny) spots in the center of the photoresin. 
        Generate the list of pre-exposure images, the list of post-exposure
        images and the common image parameter dictionary. """

        z0 = 0.5 * (self["zLower"] + self["zUpper"]) + self["zOffset"]
        self["x"] = x0
        self["y"] = y0
        self["z"] = z0
        self["xNum"] = nx
        self["yNum"] = ny

        self.log.info("Take pre-exposure images")
        self.system.moveabs(x=x0, y=y0, wait="XY")
        nz = self["zNum"]
        dz = self["zStep"]
        self.preImg, _ = self._getimages(z0, nz, dz)

        self.log.info("Expose %d x %d points" % (nx, ny))
        dx = self["xStep"]
        dy = self["yStep"]
        power = self["laserPower"]
        dt = self["duration"]
        for i in range(nx):
            for j in range(ny):
                x = x0 + dx*(i - (nx-1)/2)
                y = y0 + dy*(j - (ny-1)/2)
                self.system.moveabs(x=x, y=y, z=z0, wait="XYZ")
                self.system.pulse(power, dt)

        self.log.info("Take post-exposure images")
        self.system.moveabs(x=x0, y=y0, wait="XY")
        self.postImg, self.imgParams = self._getimages(z0, nz, dz)


    def container(self, config=None, **kwargs):

        """ Return results as SciDataContainer. """

        if self.preImg is None:
            raise RuntimeError("No results!")

        # General metadata
        content = {
            "containerType": {"name": "DcPointGrid", "version": 1.0},
            }
        meta = {
            "title": "TPP Camera Calibration Grid Images",
            "description": "",
            }

        # Package dictionary
        items = {
            "content.json": content,
            "meta.json": meta,
            "data/system.json": self.system.parameters(),
            "data/sample.json": self.system.sample,
            "data/parameter.json": self.parameters(),
            "meas/image.json": self.imgParams,
            }

        # Store pre- and post-exposure images
        for i, img in enumerate(self.preImg):
            items["meas/pre/image-%04d.png" % i] = img
        for i, img in enumerate(self.postImg):
            items["meas/post/image-%04d.png" % i] = img

        # Return container object
        config = config or self.config
        return Container(items=items, config=config, **kwargs)

