#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:08:05 2023

@author: derik
"""

#%%
import numpy as np
import skimage
from glob import glob
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.optimize import minimize_scalar, least_squares


#%%

def fit_ellipse(x, y):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()


def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi

def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y


#%%
import csv

with open('training_set_pixel_size_and_HC.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
    data = data[1:]
    
df_filenames = []
pixel_sizes = []
hcf = []

for df in data:
    df_filenames.append(df[0][:-4])
    pixel_sizes.append(float(df[1]))
    hcf.append(float(df[2]))
    
pixel_sizes = np.array(pixel_sizes)
hcf = np.array(hcf)



#%%
filenames = sorted(glob('training_set/*Annotation.png'))
fh_filenames = []
fh_params = []
fh_pixel_sizes = []
fh_hcs = []

for num, file_, in enumerate(filenames):
    print(file_)
    filename_true = file_[13:-15]
    img = skimage.io.imread(file_, as_gray=True)
    ellipse_idx = np.array(np.nonzero(img)).T
    
    x = ellipse_idx[:,0]/1000.
    y = ellipse_idx[:,1]/1000.
    
    # Formulate and solve the least squares problem ||Ax - b ||^2
    coeffs = fit_ellipse(x, y)
    x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)
    phi = (phi)/(2*np.pi)
    params = np.array([x0,y0,ap,bp,phi])

    print(x0, y0, ap, bp, e, phi)
    
    if filename_true in df_filenames:
        location = df_filenames.index(filename_true)
        print(f"The string '{filename_true}' is located at index {location} in the list.")
    else:
        print(f"The string '{filename_true}' is not in the list.")
    
    fh_filenames.append(filename_true)
    fh_params.append(params.astype(np.float32))
    fh_pixel_sizes.append(pixel_sizes[location].astype(np.float32))
    fh_hcs.append(hcf[location].astype(np.float32))

    
    # if num>50:
    #     break


#%%
sio.savemat('fh_params2.mat',{"fh_name":fh_filenames,"fh_params":np.array(fh_params),
                              "fh_pixel_sizes":np.array(fh_pixel_sizes),"fhc_gt":np.array(fh_hcs)})



