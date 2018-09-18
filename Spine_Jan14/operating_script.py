# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 10:32:28 2017

@author: mmoshe
"""

import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np
from spine import get_points_from_stl, get_gaussian_histogram, \
            get_local_extrema_of_vector, get_curve_model
from spine_chord_finder import get_points_near_spine
from derivative_chord_finder import SpineChordTangent
from spine_segmentor import SpineSegmentor, plot_segmented_spine
    

stl_path = r"C:\Users\elampert\Documents\Code\results\strat_morph13761100.stl"
pre_pnts = get_points_from_stl(stl_path)
points = np.vstack((pre_pnts[:,2], pre_pnts[:,0], pre_pnts[:,1])).T
curve_model, points = get_curve_model(points, angle_threshold = 2, \
                                zy_threshold = 60, dbg = True)

spine_pnts = get_points_near_spine(curve_model, points, zy_threshold = 80, dbg = True)

#mlab.figure('tangent model')
tangent = SpineChordTangent(curve_model)
x_pnts = np.linspace(np.min(spine_pnts[:,0]), np.max(spine_pnts[:,0]), 200)
tan, sample_points = tangent.get_tangent(x_pnts, True)

x_hist = get_gaussian_histogram(tan, spine_pnts, sample_points, sigma = 10)
plt.figure('smoothed_histogram of x population'); plt.plot(sample_points[:,0], x_hist)
x_centers = get_local_extrema_of_vector(x_hist, sample_points[:,0], mode = np.less, dbg = True)
centers = curve_model.get_curve_points(x_centers)
mlab.points3d(centers[:,0], centers[:,1], centers[:,2], color = (1,0,1), scale_factor = 20)

mlab.figure()
spine_pnts = get_points_near_spine(curve_model, points, zy_threshold = 100, dbg = False)[::1, :]           
spine_pnts = spine_pnts
segmentor = SpineSegmentor(spine_pnts,centers[:, 0])
segments = segmentor.get_segmented_spine()
plot_segmented_spine(segments)

#segments = [spine_pnts] # to reduce to one segment
from stl_save import SegentsSaver
saver = SegentsSaver(stl_path, np.eye(4), 25) # no rotation
out_fold = r'C:\Users\elampert\Documents\code\results\segments\saved_segments_137_morph'
#saver.save_all_segements_to_folder(segments, out_fold)
