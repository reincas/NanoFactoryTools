from pathlib import Path
import h5py
import cv2 as cv
import numpy as np
import scipy
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt

from scidatacontainer import Container
from nanofactorytools import image
from nanofactorysystem import mkdir

np.seterr(divide='ignore')


GRID = Path(".", "test", "grid", "grid.zdc")
TMP = Path("C:/", "Dienst", "pxdSeafile", "pxdTmp")
DATA = Path(TMP, "data.hdf5")
#dst = r"C:\Dienst\pxdSeafile\pxdTmp\autofocus"


def get_diff():
    
    dc = Container(file=str(GRID))
    params = dc["data/parameter.json"]
    diff = []
    for i in range(params["zNum"]):
        img0 = dc["meas/pre/image-%04d.png" % i]
        img1 = dc["meas/post/image-%04d.png" % i]
        diff.append(cv.absdiff(img0, img1))
        #diff.append(image.diff(img0, img1, blur))
    return np.array(diff)


def store(dc, dst, blur):
    
    mkdir(dst)
    params = dc["data/parameter.json"]
    for i in range(params["zNum"]):
        img0 = dc["meas/pre/image-%04d.png" % i]
        img1 = dc["meas/post/image-%04d.png" % i]
        diff = image.diff(img0, img1, blur)
        fn = "%s/image-%04d.png" % (dst, i)
        image.write(fn, image.normcolor(diff))


def read_diff(size=16, nx=9, ny=7):

    with h5py.File(DATA, "a") as fp:
        
        if "diff" in fp.keys():
            print("Load diff...")
            diff = fp["diff"][...]
        else:
            print("Calculate diff...")
            diff = get_diff()
            print("Store diff...")
            fp.create_dataset("diff", data=diff)
        
        gkey = "grid-%03d" % size
        ckey = "center-%03d" % size
        if gkey in fp.keys() and ckey in fp.keys():
            print("Load grid...")
            grid = fp[gkey][...]
            center = fp[ckey][...]
        else:
            print("Calculate grid...")
            grid, center = full_grid(diff, size, nx, ny)
            print("Store grid...")
            fp.create_dataset(gkey, data=grid)
            fp.create_dataset(ckey, data=center)
            
    print("Done.")
    return diff, grid, center


def full_grid(diff, size, nx, ny):

    for i in range(diff.shape[0]):
        center = np.zeros((diff.shape[0], nx*ny, 2), dtype=int)

        c, img = get_grid(diff[i,:,:], size)
        if len(c) != nx*ny:
            raise RuntimeError("Layer %d: Found %d focus spots instead of %d!" % (i, len(center), nx*ny))

        if i == 0:
            grid = np.zeros((diff.shape[0], img.shape[0], img.shape[1]), dtype=img.dtype)
        grid[i,:,:] = img
        center[i,:,:] = c
    return grid, center


def get_grid(img, size=16):
    
    # Blur float image
    img = orig = img.astype(float)
    img = cv.GaussianBlur(img, None, size)
    
    # Binary mask based on maximum filter
    img = maximum_filter(img, size=size)
    mean = 0.5 * (np.min(img) + np.max(img))
    img = np.where(img > mean, 1, 0).astype(np.uint8)
    
    contours = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    
    center = []
    for cnt in contours:
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        center.append((cx, cy))
    
    r = 4*size
    img = np.zeros((2*r+1, 2*r+1), dtype=float)
    for cx, cy in center:
        img += orig[cy-r-1:cy+r,cx-r-1:cx+r]
    img /= len(contours)
    
    return center, img

    
def get_radial(img):
    
    r = (img.shape[0] - 1) // 2
    dr = 1.0
    bins = np.arange(0.0, r+0.9*dr, dr).astype(float)
    
    x = (np.arange(2*r+1) - r)
    y = np.array(x)
    x, y = np.meshgrid(x, y)
        
    R = np.sqrt(x*x + y*y).ravel()
    r1 = np.where(R - 0.5*dr < 0.0, 0.0, R - 0.5*dr)
    r2 = R + 0.5*dr
    V = img.ravel() / (r2*r2-r1*r1)
    
    radial = scipy.stats.binned_statistic(R, V, "sum", bins)[0]
    #radial[0] = 0.0
    s = np.cumsum(radial)
    #radial = np.array(s)
    s /= s[-1]
    s -= 0.5
    s *= s
    j = np.argmin(s)
    print("half width:", np.mean(bins[j:j+2]), j)
    
    return bins, radial
    

def optimize(bins, radial):
    
    def PSF(x, *args):
        
        P, W020 = x
        a, b, c, NA = args[:-1]
            
        A = np.exp(a * W020) * b
        B = scipy.special.jv(0, np.tensordot(c * NA, rho, axes=0))
        
        res = np.tensordot(B, A, axes=1)
        f = P + (res * res.conjugate()).real
        f /= np.sum(f)
        return f
    
    
    def func(x, *args):
    
        f0 = args[-1]   
        f = PSF(x, *args)        
        return (f0 - f).std()
    
    
    f0 = radial
    r = 0.5*(bins[1:]+bins[:-1])
    
    pitch = 0.204 / 0.650
    #W020 = 2.0 #(i-150) * 0.1 / 0.650
    #r = (img.shape[0] - 1) // 2
    #x = (np.arange(2*r+1) - r) * pitch
    #y = np.array(x)
    #x, y = np.meshgrid(x, y)
    
    num = 16
    NA = 0.8
    drho = 1.0 / num
    rho = np.arange(0.0, 1.0+0.9*drho, drho)
    a = 2j * np.pi * rho * rho
    b = rho * drho
    c = 2 * np.pi * r * pitch
    
    f0 /= np.sum(f0)
    args = (a, b, c, NA, f0)
    x0 = (1.0, 1.0)
    
    x = scipy.optimize.fmin(func, x0, args, disp=True)
    print("P:", x[0])
    print("W020:", x[1])
    f = PSF(x, *args)

    plt.plot(r, f0, r, f)


def optimize2(bins, radial):
    
    def PSF(x, *args):
        
        A, w = x
        r, f0 = args
        
        f = A * np.exp(-(r/w)**2)
        return f
    
    
    def func(x, *args):
    
        r, f0 = args   
        f = PSF(x, *args)        
        return (f0 - f).std()
    
    
    f0 = radial
    r = 0.5*(bins[1:]+bins[:-1])
    
    f0 /= np.sum(f0)
    args = (r, f0)
    x0 = (1.0, 1.0)
    
    x = scipy.optimize.fmin(func, x0, args, disp=True)
    print("A:", x[0])
    print("w:", x[1])
    f = PSF(x, *args)

    plt.plot(r, f0, r, f)


def get_radius(grid):

    radius = []
    for i in range(grid.shape[0]):
    
        img = grid[i,:,:]
        #img = image.blur(img, 2)
        
        A1 = np.count_nonzero(img > 0.5*np.max(img))
        r1 = np.sqrt(A1 / np.pi)
    
        img = np.where(img > 0.5*np.max(img), 1, 0).astype(np.uint8)
        contours = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        contours = np.concatenate(contours, axis=0)
        hull = cv.convexHull(contours)
        A2 = cv.contourArea(hull)
        r2 = np.sqrt(A2 / np.pi)
        #print("Radius %d: %.2f px" % (i, r))
        
        radius.append((r1, r2))
    radius = np.array(radius)
    return radius


#####################################################################
size = 16
nx = 9
ny = 7
diff, grid, center = read_diff(size, nx, ny)


def grad(img):
    
    #print(np.min(img), np.max(img))
    #img1 = image.norm(img)[0]
    #img = img.astype(np.uint8)
    #img1 = cv.Laplacian(img, cv.CV_8U, ksize=1)
    img1 = cv.Laplacian(img, cv.CV_64F, ksize=1)
    print("grad:", np.min(img1), np.max(img1), np.mean(np.abs(img1)))



#img = diff[250,:,:].astype(float) - diff[260,:,:].astype(float)
#print(np.min(img), np.max(img), scipy.stats.skew(img.ravel()))
# h, w = img.shape
# img2 = img2[:h//2,:w//2]
# #img = img1 = cv.Laplacian(img2, cv.CV_64F, ksize=1)
# img = cv.Sobel(img2,cv.CV_64F, 0, 1, ksize=5)
# print("skew:", scipy.stats.skew(img.ravel()))
# bins = np.arange(np.min(img), np.max(img), 1.0)
# hist = np.histogram(img, bins)[0]
# plt.plot(bins[:-1], hist)

#plt.imshow(image.normcolor(img))

#bins, radial = get_radial(img)
#optimize2(bins, radial)

radius = get_radius(grid)
n = grid.shape[0]
z = np.arange(n) - (n-1)/2
r1 = radius[:,0]
r2 = radius[:,1]
plt.plot(z, r1, z, r2)

plt.show()




    