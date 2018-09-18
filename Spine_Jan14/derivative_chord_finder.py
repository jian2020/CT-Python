# -*- coding: utf-8 -*-
"""
Created on Mon Jan 08 16:40:16 2018

@author: mmoshe
"""
import numpy as np

class SpineChordTangent:
    
    def __init__(self, curve_model):
        
        self.curve_model = curve_model
        
    def get_tangent(self, x_vector, dbg = False):

        x_sorted = sorted(x_vector)
        spn_pnts = self.curve_model.get_curve_points(x_sorted)
        spn_pnts_centers = (spn_pnts[1:, :] + spn_pnts[:-1, :])/2.
        result = np.diff(spn_pnts, axis = 0)
        result /= np.linalg.norm(result, axis = 1).reshape([-1,1])
        if dbg:
            self.__plot_quiver(result, spn_pnts_centers)
        return result, spn_pnts_centers
    
    def __plot_quiver(self, tangent, spn_pnts):
        from mayavi import mlab
        
        mlab.quiver3d(spn_pnts[:, 0], spn_pnts[:, 1], spn_pnts[:,2], 
                      tangent[:, 0], tangent[:, 1], tangent[:, 2])