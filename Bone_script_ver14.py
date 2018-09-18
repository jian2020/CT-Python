# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:46:59 2017

@author: elampert
"""

import os
from scipy import misc
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import dicom_reader_ver14
# from dicom_reader_ver14 import get_segmented_bones, load_ct, test_bones, largest_region
from dicom_reader_ver14 import load_ct, largest_region
import draw_image
from draw_image import draw

import skimage
from skimage import filters, transform
#import SimpleITK as sitk
from mayavi import mlab
#import morphsnakes
import scipy
import scipy.misc as misc
from skimage.morphology import closing
from skimage.segmentation import morphological_chan_vese as MorphACWE
from skimage.segmentation import circle_level_set
from mayavi import mlab

#path = r'C:\Users\elampert\Downloads\stratasys\CD1\CD1\6661\6663'
#path = r'C:\Users\elampert\Downloads\stratasys\CD1\CD1\6661\6665'
#path = r'C:\Users\elampert\Downloads\stratasys\CD1\CD1\6661\6666'
#path = r'C:\Users\elampert\Downloads\stratasys\DICOMDIR\73764152\13764151'

print("Input the DataPath")
path = input()
# path = r'E:\vertebrate-segmentation\datasets\test'
#path = r'C:\Users\elampert\Documents\DOI\DOI\ABD_LYMPH_001\61.7.22285965616260355338860879829667630274\61.7.167248355135476067044532759811631626828'
#filename = r'C:\Users\elampert\Documents\cephx\body\subset0\1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd' 
#filename = r'C:\Users\elampert\Documents\cephx\body\subset0\1.3.6.1.4.1.14519.5.2.1.6279.6001.313835996725364342034830119490.mhd'
#filename = r'C:\Users\elampert\Documents\Data\body\subset0\1.3.6.1.4.1.14519.5.2.1.6279.6001.430109407146633213496148200410.mhd'

#path =r'C:\Users\elampert\Documents\Code\Hadassa\34438666'


# main  - Segment
#slices = dicom_reader.LoadMhd(filename) # load from the Luna16 dataset
# load Dicom files
slices = load_ct(path)
# slices = slices[159:600] # remove no relevant data in 137
#slices = slices[19::] # remove no relevant data in 173
slices = slices.transpose(0,2,1)
count = len(slices)
#SegScan = slices*(slices>TH)~\Documents\cephx\Spine\spine_2\morphsnakes-master~\Documents\cephx\Spine\spine_2\morphsnakes-master
#smoothed_im = ndi.filters.median_filter(SegScan, size = 3)
#slices = slices.transpose(2,0,1) # move to coronal view
TH = 150 # MHD
TH = 1101 # Dicom - works well for 344386666, but too high for 137
Th = 1001 # Dicom - try for 137

Seg_scan =slices* (slices>TH)


# Spine detector
spine_radius = 50
#selem = disk(spine_radius)
Scan0=slices[0]*0
del slices
# 2D Morphological chan-vase to create the basic segmentation
for n in range(0,count):
    border = MorphACWE(Seg_scan[n],50, init_level_set='checkerboard',smoothing=4, lambda1=1, lambda2=1)
    # draw(border, n)
    border_tag=1-border
    if border.sum() > border_tag.sum(): # Relevant area is sometimes 1 and sometimes 0. So I use the fact it's the smaller area.
        border = 1- border
    Im = border*Seg_scan[n]
    # draw(Im, n)
    Scan0 = np.dstack((Scan0,Im))
    print(n)
del count
del border
del border_tag
del Im

# keep largest area only
largest_volume = Scan0*largest_region(Scan0>0)

# scale all 3 axis:
Scan2 = transform.rescale(largest_volume,1.414,preserve_range=True)
del largest_volume
print(len(Scan2))
Scan2 = Scan2.transpose([1,2,0])
print(len(Scan2))
print(90)
Scan2 = transform.rescale(Scan2,1.414,preserve_range=True)
print(len(Scan2))
Scan2 = Scan2.transpose([2,1,0])
print(len(Scan2))
print(95)
Scan2 = transform.rescale(Scan2,1.414,preserve_range=True)
print(len(Scan2))
Scan2 = Scan2.transpose([0,2,1])
print(len(Scan2))
print(100)

#Feature preserving Smoothing
Seg4= skimage.filters.gaussian(Scan2, sigma=1)
del Scan2
print(108)
#Closure
Seg5 = closing(Seg4,skimage.morphology.ball(2))
del Seg4
print(112)
mlab.figure()
print(114)
mlab.contour3d(Seg5)
print(116)

outputfile = r'E:\vertebrate-segmentation\result\\' + path[-8:-4] + str(TH) + '.stl'
dicom_reader_ver14.stl_export(Seg5,TH-1,outputfile,1)
#np.save(outputfile[:-3],Seg4)