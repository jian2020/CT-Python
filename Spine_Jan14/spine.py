from stl import stl
from mayavi import mlab
import numpy as np
from spine_chord_finder import ZYOfXMedianModel
from canonical_rotation import get_canonicaly_rotated_pnts

def get_curve_model(points, angle_threshold, zy_threshold, dbg = False):
    
    curve_model = ZYOfXMedianModel(points, zy_threshold = zy_threshold, dbg = False)
    angle = 1
    while angle > angle_threshold:
        points, angle = get_canonicaly_rotated_pnts(points, curve_model)
        print "angle difference " + str(angle)
        curve_model = ZYOfXMedianModel(points, zy_threshold = zy_threshold, dbg = False)
    if dbg:
        curve_model.plot_curve(points[:, 0])
    return curve_model, points

def get_gaussian_histogram(normals, points, centers, sigma):
    
    x_hist = [get_gaussian_bin(points, centers[index], normals[index], sigma = sigma) for index in xrange(len(normals))]
    return np.array(x_hist)

def get_gaussian_bin(points, center, normal, sigma):
    
    argum = np.inner(points - center, normal)/sigma
    weights = np.exp(-argum**2)
    return np.sum(weights)
    
def get_normalized_points_from_stl(stl_path ,dbg = False):
    
    points = get_points_from_stl(stl_path)
    points -= np.average(points, axis = 0)
    points /= np.std(points, axis = 0)
    if dbg:
        mlab.points3d(points[:,0], points[:,1], points[:,2], mode = 'point', color = (1,0,0))
    return points

def get_points_from_stl(stl_path):    
    stl_mesh = stl.StlMesh(stl_path)
    points = np.vstack((stl_mesh.v0, stl_mesh.v1, stl_mesh.v2))
    return points[::15]
    
def get_local_extrema_of_vector(x_hist, x_pnts, mode, dbg = False):
    from scipy.signal import argrelextrema
    import matplotlib.pyplot as plt
    
    indexes = argrelextrema(x_hist, mode)
    if dbg:
        plt.plot(x_pnts[indexes], x_hist[indexes], '.')
    return x_pnts[indexes]
    
