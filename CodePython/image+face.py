#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 22:53:01 2018

@author: Md. Fakrul Abedin Bhuiyan and S. Rayhan Kabir
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:25:57 2018

@author: Md. Fakrul Abedin Bhuiyan and S. Rayhan Kabir
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage.io
import io

from scipy.misc import imsave


from skimage import data, img_as_float
from skimage import exposure


matplotlib.rcParams['font.size'] = 8


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()
    ax_img.set_adjustable('box-forced')

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


# Load an example image
#img = data.imread('/home/fakrul/Downloads/kkkk.jpg')/home/fakrul/Documents/thesis /images

img = data.imread('F:/Study/Research, Thesis/DIU/Fakrul/code/images/image 1.jpg')
img_o = data.imread('F:/Study/Research, Thesis/DIU/Fakrul/code/images/image 1.jpg')
img_p = data.imread('F:/Study/Research, Thesis/DIU/Fakrul/code/images/image 1.jpg')
img_q = data.imread('F:/Study/Research, Thesis/DIU/Fakrul/code/images/image 1.jpg')

imsave('F:/Study/Research, Thesis/DIU/Fakrul/code/face/img.jpg',img)
#img = data.brick()
#img = skimage.io.imread(home/fakrul/Downloads)

# Contrast stretching
p2, p98 = np.percentile(img, (2, 98))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
imsave('F:/Study/Research, Thesis/DIU/Fakrul/code/face/Contrast stretching/test1.jpg',img_rescale)

# Equalization
img_eq = exposure.equalize_hist(img)
imsave('F:/Study/Research, Thesis/DIU/Fakrul/code/face/Histogram equalization/test2.jpg',img_eq)

# Adaptive Equalization
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
imsave('F:/Study/Research, Thesis/DIU/Fakrul/code/face/Adaptive equalization/test3.jpg',img_adapteq)


# Display results
fig = plt.figure(figsize=(8,5))
axes = np.zeros((2, 4), dtype=np.object)
axes[0, 0] = fig.add_subplot(2, 4, 1)
for i in range(1, 4):
    axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
for i in range(0, 4):
    axes[1, i] = fig.add_subplot(2, 4, 5+i)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
ax_img.set_title('Low contrast image')

y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Number of pixels')
ax_hist.set_yticks(np.linspace(0, y_max, 5))

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
ax_img.set_title('Contrast stretching')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
ax_img.set_title('Histogram equalization')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
ax_img.set_title('Adaptive equalization')

ax_cdf.set_ylabel('Fraction of total intensity')
ax_cdf.set_yticks(np.linspace(0, 1, 10))

# prevent overlap of y-axis labels
fig.tight_layout()
plt.show()
#img.dtype
#img.min(),img.max()




face_cascade = cv2.CascadeClassifier('F:/Study/Research, Thesis/DIU/Fakrul/code/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('F:/Study/Research, Thesis/DIU/Fakrul/code/haarcascade_eye.xml')

img = cv2.imread('F:/Study/Research, Thesis/DIU/Fakrul/code/face/Contrast stretching/test1.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,1.1, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.rectangle(img_o,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    cv2.imwrite('F:/Study/Research, Thesis/DIU/Fakrul/code/face/Contrast stretching/crop.jpg', roi_color)
    
    #cv2.imwrite('gyt.jpg',roi_color)
    #eyes = eye_cascade.detectMultiScale(roi_gray)
    cv2.imshow('img', img_o)
    imsave('F:/Study/Research, Thesis/DIU/Fakrul/code/face/Contrast stretching/detect.jpg',img_o)
        # for (ex,ey,ew,eh) in eyes:
        #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        #cv2.imshow('img',roi_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    
    
    
img = cv2.imread('F:/Study/Research, Thesis/DIU/Fakrul/code/face/Histogram equalization/test2.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,1.1, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.rectangle(img_p,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    cv2.imwrite('F:/Study/Research, Thesis/DIU/Fakrul/code/face/Histogram equalization/crop.jpg', roi_color)
    
    #cv2.imwrite('gyt.jpg',roi_color)
    #eyes = eye_cascade.detectMultiScale(roi_gray)
    cv2.imshow('img', img_p)
    imsave('F:/Study/Research, Thesis/DIU/Fakrul/code/face/Histogram equalization/detect.jpg',img_p)
        # for (ex,ey,ew,eh) in eyes:
        #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        #cv2.imshow('img',roi_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    

img = cv2.imread('F:/Study/Research, Thesis/DIU/Fakrul/code/face/Adaptive equalization/test3.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,1.1, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.rectangle(img_q,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    cv2.imwrite('F:/Study/Research, Thesis/DIU/Fakrul/code/face/Adaptive equalization/crop.jpg', roi_color)
    
    #cv2.imwrite('gyt.jpg',roi_color)
    #eyes = eye_cascade.detectMultiScale(roi_gray)
    cv2.imshow('img', img_q)
    imsave('F:/Study/Research, Thesis/DIU/Fakrul/code/face/Adaptive equalization/detect.jpg',img_q)
        # for (ex,ey,ew,eh) in eyes:
        #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        #cv2.imshow('img',roi_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
