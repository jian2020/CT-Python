# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 08:21:47 2017

@author: mmoshe
"""

import numpy as np
from stl import mesh
from stl import stl
from scipy.spatial import cKDTree

class SegentsSaver:
    
    def __init__(self, stl_path, rigid_tx_mm, max_dist_mm):
        
        self.max_dist_mm = max_dist_mm
        self.rigid_tx_mm = rigid_tx_mm
        spn_mesh = stl.StlMesh(stl_path)
        self.__set_mesh_points(spn_mesh)
        
    def save_all_segements_to_folder(self, segments, out_fold):
        import os
        
        if not os.path.exists(out_fold):
            os.makedirs(out_fold)
        for index, seg in enumerate(segments):
            fname = 'segment_' + str(index) + '.stl'
            out_path = os.path.join(out_fold, fname)
            self.__save_segment_stl(seg, out_path)
    
    def __save_segment_stl(self, segment, out_path):
        
        finder = TriNearPointsFinder(segment, self.max_dist_mm)
        seg0, seg1, seg2 = finder.get_near_triangles(self.v0, self.v1, self.v2)
        save_3points_to_stl(seg0, seg1, seg2, out_path)
        
    def __set_mesh_points(self, spn_mesh):
        
        self.v0 = self.__get_rotated_pnts(spn_mesh.v0)
        self.v1 = self.__get_rotated_pnts(spn_mesh.v1)
        self.v2 = self.__get_rotated_pnts(spn_mesh.v2)
        
    def __get_rotated_pnts(self, v):
        
        cv = np.vstack((v[:,2], v[:,0], v[:,1])).T
        cv = np.dot(cv, self.rigid_tx_mm[:3,:3].T) + self.rigid_tx_mm[:3, 3]
        return cv
    
class TriNearPointsFinder:
    
    def __init__(self, points, max_dist_mm):
        
        self.tree = cKDTree(points)
        self.max_dist_mm = max_dist_mm
    
    def get_near_triangles(self, v0, v1, v2):
        
        close_0 = self.__are_close(v0)
        close_1 = self.__are_close(v1)
        close_2 = self.__are_close(v2)
        are_near = np.logical_and.reduce((close_0, close_1, close_2))
        return v0[are_near], v1[are_near], v2[are_near]
        
    def __are_close(self, pnts):

        dists, _ = np.array(self.tree.query(pnts))
        are_close = dists < self.max_dist_mm
        return are_close

def save_3points_to_stl(v0, v1, v2, out_path):
    
    assert v0.shape == v1.shape == v2.shape, \
        'all triangles must have three vertices'
    assert v0.shape[1] == 3, 'needs to be 3 dimensional'
    
    length = v0.shape[0]
    out_mesh = mesh.Mesh(np.zeros(length, dtype=mesh.Mesh.dtype))
    
    for tri_index in xrange(length):
        out_mesh.vectors[tri_index][0] = v0[tri_index,:]
        out_mesh.vectors[tri_index][1] = v1[tri_index,:]
        out_mesh.vectors[tri_index][2] = v2[tri_index,:]
        
    out_mesh.save(out_path)