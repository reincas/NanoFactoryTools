import matplotlib.pyplot as plt
import numpy as np
from scidatacontainer import Container
from nanofactorytools import detectGrid, focusRadius, getTransform, image


def step1(grid, blur):
    
    """ Detect grid on center image. """
    
    params = grid["data/parameter.json"]
    i = (params["zNum"] - 1) // 2
    img0 = grid["meas/pre/image-%04d.png" % i]
    img1 = grid["meas/post/image-%04d.png" % i]
    diff = image.diff(img0, img1, blur)
    
    src, dst, pitch, angle, nx, ny, detect = detectGrid(diff, True)
    print("Grid: %d x %d" % (nx, ny))
    if nx != params["xNum"] or ny != params["yNum"]:
        print("*** Warning: Wrong grid size!")
    print("Pitch: %.1f µm" % pitch)
    print("Angle: %.3f°" % angle)
    
    return src

    
def step2(grid, center, blur, size, thresh):
    
    """ Detect offset of camera focus """

    params = grid["data/parameter.json"]
    nz = params["zNum"]
    dz = params["zStep"]
    z_off = params["zOffset"]
    
    data = []
    for i in range(nz):
        z = (i - (nz-1)/2)*dz + z_off
        img0 = grid["meas/pre/image-%04d.png" % i]
        img1 = grid["meas/post/image-%04d.png" % i]
        diff = image.diff(img0, img1)
        r = focusRadius(diff, center, blur, size, thresh)
        data.append((z, r))
    data = np.array(data)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(data[:,0], data[:,1], "r")
    ax.grid(True)
    ax.set_xlabel("axial camera offset [µm]")
    ax.set_ylabel("effective spot radius [px]")
    plt.savefig("radius.png", bbox_inches="tight")
    plt.show()
    
    imin = np.argmin(data[:,1])
    z_off = data[imin, 0]
    r = data[imin, 1]
    print("Axial offset: %.1f µm" % z_off)
    print("Spot radius: %.1f px" % r)
    return z_off, imin


def step3(grid, blur, i):
    
    """ Detect transformation matrix on focussed image. """
    
    params = grid["data/parameter.json"]
    img0 = grid["meas/pre/image-%04d.png" % i]
    img1 = grid["meas/post/image-%04d.png" % i]
    diff = image.diff(img0, img1, blur)
    
    src, dst, pitch, angle, nx, ny, detect = detectGrid(diff, True)
    print("Grid: %d x %d" % (nx, ny))
    if nx != params["xNum"] or ny != params["yNum"]:
        print("*** Warning: Wrong grid size!")
    print("Pitch: %.1f µm" % pitch)
    print("Angle: %.3f°" % angle)

    h, w = diff.shape
    dx = params["xStep"]
    dy = params["yStep"]
    A, ax, ay, bx, by, Fx, Fy, dr = getTransform(src, w, h, dst, nx, ny, dx, dy)
    print("Scale:    %.4f x %.4f µm/px" % (ax, ay))
    print("Rotation: %.3f° x %.3f°" % (bx, by))
    print("Offset:   %.1f x %.1f px" % (Fx, Fy))
    print("Mean dev: %.3f µm" % dr)
    return A

    
##########################################################################
blur = 4
size = 20
thresh = 0.2

# Read grid exposure container
fn = "test/grid/grid.zdc"
dc = Container(file=fn)

# Detect grid on center image
src = step1(dc, blur)

# Detect offset of camera focus
z_off, i = step2(dc, src, blur, size, thresh)

# Detect transformation matrix on focussed image
A = step3(dc, blur, i)
