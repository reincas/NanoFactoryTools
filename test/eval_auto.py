from pathlib import Path
import h5py
import cv2 as cv
import numpy as np
import scipy
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt
import skimage

from scidatacontainer import Container
from nanofactorytools import image
from nanofactorysystem import mkdir

np.seterr(divide='ignore')


GRID = Path(".", "test", "grid", "grid.zdc")
TMP = Path("C:/", "Dienst", "pxdSeafile", "pxdTmp")
DATA0 = Path(TMP, "data0.hdf5")
DATA1 = Path(TMP, "data1.hdf5")


def get_diff(full=True):
    
    dc = Container(file=str(GRID))
    params = dc["data/parameter.json"]
    num = params["zNum"]
    i = (num - 1) // 2
    img0 = dc["meas/pre/image-%04d.png" % i]
    diff = []
    for i in range(num):
        if full:
            img0 = dc["meas/pre/image-%04d.png" % i]
        img1 = dc["meas/post/image-%04d.png" % i]
        diff.append(cv.absdiff(img0, img1))
        #diff.append(image.diff(img0, img1, blur))
    return np.array(diff)


# def store(dst, blur, full=True):
    
#     dc = Container(file=str(GRID))
#     mkdir(dst)
#     params = dc["data/parameter.json"]
#     num = params["zNum"]
#     i = (num - 1) // 2
#     img0 = dc["meas/pre/image-%04d.png" % i]
#     for i in range(num):
#         if full:
#             img0 = dc["meas/pre/image-%04d.png" % i]
#         img1 = dc["meas/post/image-%04d.png" % i]
#         diff = image.diff(img0, img1, blur)
#         fn = "%s/image-%04d.png" % (dst, i)
#         image.write(fn, image.normcolor(diff))


def store(dst, images):
    
    mkdir(dst)
    for i, img in enumerate(images):
        fn = "%s/image-%04d.png" % (dst, i)
        image.write(fn, image.normcolor(img))


def read_diff(fn, size=16, nx=9, ny=7, full=True):

    with h5py.File(fn, "a") as fp:
        
        if "diff" in fp.keys():
            print("Load diff...")
            diff = fp["diff"][...]
        else:
            print("Calculate diff...")
            diff = get_diff(full)
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
            grid, center = full_grid(diff, size, nx, ny, full)
            print("Store grid...")
            fp.create_dataset(gkey, data=grid)
            fp.create_dataset(ckey, data=center)

    print("Done.")
    return diff, grid, center


def full_grid(diff, size, nx, ny, full=True):

    num = diff.shape[0]
    if full:
        center = np.zeros((num, nx*ny, 2), dtype=int)
    else:
        i = (num - 1) // 2
        center = get_center(diff[i], size)
    
    for i in range(num):
        if full:
            c = None
        else:
            c = center
        c, img = get_grid(diff[i], size, c)
        if len(c) != nx*ny:
            raise RuntimeError("Layer %d: Found %d focus spots instead of %d!" % (i, len(center), nx*ny))

        if i == 0:
            grid = np.zeros((num, img.shape[0], img.shape[1]), dtype=img.dtype)
        grid[i,:,:] = img
        if full:
            center[i,:,:] = c
    return grid, center


def get_center(img, size=16):
    
    # Blur float image
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
    
    return np.array(center, dtype=int)

    
def get_grid(img, size=16, center=None):
    
    img = img.astype(float)
    if center is None:
        center = get_center(img, size)
    
    r = 4*size
    simg = np.zeros((2*r+1, 2*r+1), dtype=float)
    for cx, cy in center:
        simg += img[cy-r:cy+r+1,cx-r:cx+r+1]
    simg /= len(center)
    
    return center, simg

    
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

# =============================================================================
#         img = np.where(img > thresh*np.max(img), 1, 0).astype(np.uint8)
#         contours = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
#         contours = np.concatenate(contours, axis=0)
#         hull = cv.convexHull(contours)
#         A2 = cv.contourArea(hull)
#         r2 = np.sqrt(A2 / np.pi)
#         #print("Radius %d: %.2f px" % (i, r))
#         
# =============================================================================

# def get_area(grid, thresh=0.5):

#     img = [i / i.max() for i in grid]    
#     area = np.array([np.count_nonzero(i > thresh) for i in img], dtype=float)
#     return area

def get_area2(grid, thresh=0.5):

    thresh = thresh * get_cval(grid, 2)
    img = [grid[i] - thresh[i] for i in range(thresh.shape[0])]
    area = np.array([np.count_nonzero(i > 0.0) for i in img], dtype=float)
    return area


def get_area(grid, d=2, thresh=0.1):

    ic = (grid.shape[1] - 1) // 2
    #center = get_cval(grid, d)
    #img = np.array([grid[i] / center[i] for i in range(grid.shape[0])])
    img = np.array([image.blur(i, d) for i in grid])
    img = np.array([i / i[ic,ic] for i in img])
    img = np.where(np.logical_and(img > 1.0-thresh, img < 1.0+thresh), 1, 0).astype(np.uint8)
    n, h, w = img.shape
    #mask = np.zeros((h+2, w+2), dtype=img.dtype)
    img = [cv.floodFill(i, None, (ic,ic), 2)[1] for i in img]
    area = np.array([np.count_nonzero(i == 2) for i in img], dtype=float)
    return area


def get_cval(grid, d=2):
    
    ic = (grid.shape[1] - 1) // 2
    img = grid[:,ic-d:ic+d+1,ic-d:ic+d+1]
    n,h,w = img.shape
    img = img.reshape((n,h*w))
    img = img.mean(axis=1)
    return img


def get_max(grid, d=2):
    
    n = 2 * d + 1
    k = np.ones((n,n), dtype=float)
    k /= k.size
    img = np.array([cv.filter2D(i, cv.CV_64F, k).ravel() for i in grid])
    img = img.max(axis=1)
    return img


def get_grad(grid, q=0.5):
    
    img = [cv.Laplacian(i, cv.CV_64F, ksize=7) for i in grid]
    img = np.array([i.ravel() for i in img])
    img = np.sort(img*img, axis=1)
    img = np.cumsum(img, axis=1)
    img = [np.searchsorted(i, i[0]+q*(i[-1]-i[0])) for i in img]
    img = np.array(img, dtype=float)
    return img
    #img = np.quantile(img, q, axis=1)
    #return np.quantile(img, q, axis=1)

# def plot_radius(grid):

#     radius = get_radius(grid)
#     n = grid.shape[0]
#     z = np.arange(n) - (n-1)/2
#     r1 = radius[:,0]
#     r2 = radius[:,1]
#     plt.plot(z, r1, z, r2)


def zero_cross(x, y):
    
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    assert len(x.shape) == 1
    assert x.shape == x.shape

    izero = y[1:-1] == 0.0
    x1 = x[:-2][izero]
    x2 = x[2:][izero]
    y1 = y[:-2][izero]
    y2 = y[2:][izero]
    mzero = (y2-y1) / (x2-x1)
    xzero = x[1:-1][izero]
    
    icross = y[1:] * y[:-1] < 0.0
    x1 = x[:-1][icross]
    x2 = x[1:][icross]
    y1 = y[:-1][icross]
    y2 = y[1:][icross]
    mcross = (y2-y1) / (x2-x1)
    xcross = x1 - y1/mcross
    
    z = np.concatenate((xzero, xcross), axis=0)
    m = np.concatenate((mzero, mcross), axis=0)
    i = np.argsort(z)
    z = z[i]
    m = m[i]
    
    return z, m


def find_focus(value, args, z0, num, zcoarse, zfine):

    n = (num - 1) // 2
    zinitial = np.arange(z0-n*zcoarse, z0+n*zcoarse+1, zcoarse, dtype=float)
    vinitial = [value(z, *args) for z in zinitial]

    imax = np.argmax(vinitial)    
    zc, vc = zinitial[imax], vinitial[imax]
        
    zlo, vlo = zc - zcoarse, None
    if imax > 0:
        vlo = vinitial[imax-1]

    zhi, vhi = zc + zcoarse, None
    if imax < len(vinitial)-1:
        vhi = vinitial[imax+1]
    
    while vlo is None:
        vlo = value(zlo, *args)
        if vlo > vc:
            zhi, vhi = zc, vc
            zc, vc = zlo, vlo
            zlo, vlo = zlo - zcoarse, None
    
    while vhi is None:
        vhi = value(zhi, *args)
        if vhi > vc:
            zlo, vlo = zc, vc
            zc, vc = zhi, vhi
            zhi, vhi = zhi + zcoarse, None
    
    while max(zc-zlo, zhi-zc) > zfine:
        print("%.1f, %.1f   %.1f, %.1f   %.1f, %.1f" % (zlo, vlo, zc, vc, zhi, vhi))
        
        if zc-zlo > zhi-zc:
            z = (zlo + zc) / 2
            v = value(z, *args)
            if v <= vc:
                zlo, vlo = z, v
            else:
                zhi, vhi = zc, vc
                zc, vc = z, v
        else:
            z = (zc + zhi) / 2
            v = value(z, *args)
            if v <= vc:
                zhi, vhi = z, v
            else:
                zlo, vlo = zc, vc
                zc, vc = z, v

    print("%.1f, %.1f   %.1f, %.1f   %.1f, %.1f" % (zlo, vlo, zc, vc, zhi, vhi))
    return zc


#####################################################################


size = 16
nx = 9
ny = 7
full = False
diff, grid, center = read_diff(DATA1, size, nx, ny, full)

#print("Store images...")
#store(Path(TMP, "autofocus"), grid)

def value(z, *args):
    grid, z0, dz = args
    
    ic = (grid.shape[0] - 1) // 2
    i = round(ic + (z-z0) / dz)
    img = grid[i]
    cy = (img.shape[0] - 1) // 2
    cx = (img.shape[1] - 1) // 2
    #return img[cy-2:cy+3,cx-2:cx+3].max()
    return img[cy-1:cy+2,cx-1:cx+2].max()
    #return img[cy,cx]

z0 = -7.0
dz = 0.1
args = (grid, z0, dz)
zopt = find_focus(value, args, z0=z0, num=9, zcoarse=2.5, zfine=dz)
ic = (grid.shape[0] - 1) // 2
iopt = round(ic + (zopt-z0) / dz)
print("opt: %.2f µm [%d]" % (zopt, iopt))

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

    

plt.close()

# r = (grid.shape[1] - 1) // 2
# i = 170
# x = []
# shift = []
# imin = 50
# imax = 250
# for i in range(imin, imax+1):
#     img1 = grid[i-5,:,:]
#     img2 = grid[i+5,:,:]
    
#     #img1 = image.blur(img1, 2)
#     #img2 = image.blur(img2, 2)
    
#     img1p = skimage.transform.warp_polar(img1, radius=r, scaling='log')
#     img2p = skimage.transform.warp_polar(img2, radius=r, scaling='log')
    
#     shifts, error, phasediff = skimage.registration.phase_cross_correlation(img1p, img2p,
#                                                        upsample_factor=100,
#                                                        normalization=None)
#     #print("shifts:", shifts)
#     #print("error:", error)
#     #print("phasediff:", phasediff)
    
#     x.append(i)
#     shift.append(shifts[1])
#     d = 2

# x = np.array(x)
# shift = np.array(shift)

#k = cv.getGaussianKernel(5, 2.0, cv.CV_64F).ravel()
#print(k)

#shift = np.convolve(k, shift, "same")
#shift = np.cumsum(shift)

d = 1
thresh = 0.1
x = np.arange(grid.shape[0], dtype=float)
cval = get_cval(grid, d)
mval = get_max(grid, d)
area = get_area(grid, d, thresh)
#area = get_area(grid, 0.5)[imin:imax+1]
#grad = get_grad(grid, 0.5)[imin:imax+1]
#grad -= grad.mean()
#grad = get_grad(grid, 0.5)
#xx = np.arange(grad.shape[1])
#plt.plot(xx, grad[50,:], xx, grad[173,:])

# shift /= shift.std()
# dmaxval /= dmaxval.std()
# maxval /= maxval.std()
# area /= area.std()
# max2 /= max2.std()
# grad /= grad.std()

plot = [cval, mval, area]
plot = [i for p in plot for i in (x, p)]
plt.plot(*plot)


#img = [grid[i,:,:] for i in (110, 146, 173)]
img = [grid[i,:,:] for i in (113, 146, 175)]
# img = [ skimage.transform.warp_polar(i, radius=2*r, scaling='log') for i in img]
# h,w = img[0].shape
# print(h,w)
# h = h // 2
# img = [ np.concatenate((i[h:,:], i[:h,:]), axis=0)for i in img]
img = np.concatenate(img, axis=1)
#plt.imshow(image.normcolor(img))

#plot_radius(grid)

plt.show()




    