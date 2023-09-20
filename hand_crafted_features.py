# -*- coding: utf-8 -*-
"""Copy of Hand Crafted Features.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1unoapbSYh9FDsEfH07FBOaiU7Tstxqw5

** Few Hand Crafted Feature extraction functions **


* Hu Moments
* Zerinke Moments 
* Haralick features
* Local Binary Pattern 
* Color Histogram 
* Global Features
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import mahotas
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans

#!pip install mahotas

"""Read Image

FEATURE EXTRACTION CODES

1. HU Moments ( Shape Descriptor )

The feature vector obtained after calling this function can be used to quantify and represent the shape of an object in an image.

[The workings and possible problems of using HU moments](https://learnopencv.com/shape-matching-using-hu-moments-c-python/)
"""

# Extract Hu Moments feature of an image. It returns the 7 HU moments of the image. Inp is BGR Format

def find_hu_moments(x):    
    
    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    hu_moment = cv2.HuMoments(cv2.moments(gray)).flatten()

    return hu_moment


"""2. Zerinke Moments ( Shape Descriptor )

These are more powerful and less computationally expensive as compared to the Hu moments. Again, it is a shape extracting feature. Could be used for boundaries.

[Zerinke Moments](https://cvexplained.wordpress.com/2020/07/21/10-5-zernike-moments/)
"""

# Extract Zerinke Moments feature of an image. It returns the 25 moments of the image. Inp is BGR Format

def find_zernike_moments(x, radius=21, degree=8):

    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    zk_moment = mahotas.features.zernike_moments(gray, radius, degree)

    return zk_moment

"""3. Haralick Moments ( Texture Descriptor )

Texture descriptors. More information in :

[Haralick](https://www.geeksforgeeks.org/mahotas-haralick-features/)
"""

# Extract Haralick feature of an image. Inp is BGR Format

def find_haralick(x):

    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)

    return haralick

"""4. Local Binary Patterns ( Texture Descriptors )

Texture descriptors. Read more at :

1. [Resource 1](https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html)

2. [Resource 2](https://www.geeksforgeeks.org/create-local-binary-pattern-of-an-image-using-opencv-python/#:~:text=Local%20Binary%20Pattern%2C%20also%20known,value%20of%20the%20centre%20pixel.)
"""

# Extract LBP HISTOGRAM feature of an image. Inp is BGR Format

def find_lbp(x, numPoints=24, radius=8):
    
    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, numPoints, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    result, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    return result

"""5. Colour Histograms """

# Extract colour HISTOGRAM feature of an image. Inp is BGR Format

def find_colour_histogram(img, n_bins=8):
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # convert the image to HSV color-space
    hist  = cv2.calcHist([hsv], [0, 1, 2], None, [n_bins, n_bins, n_bins], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    out = hist.flatten()

    return out

"""6. Extract Global Features

Put all into one for convenience
"""

# Input is BGR format and it returns global feature array containing shape, texture and colour features of image.

def find_global_features(img):
    
    hu_moments = find_hu_moments(img)
    zernike_moments = find_zernike_moments(img)
    haralick   = find_haralick(img)
    lbp_histogram  = find_lbp(img)
    color_histogram  = find_colour_histogram(img)
    global_feature = np.hstack([hu_moments, zernike_moments, haralick, lbp_histogram, color_histogram])
    
    return global_feature

