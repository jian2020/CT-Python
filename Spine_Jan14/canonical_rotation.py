import numpy as np
from transformation import rotation_matrix

def get_canonicaly_rotated_pnts(points_mm, curve_model):
    
    center, x_dir = get_x_direction(points_mm, curve_model)
    angle = -np.arccos(x_dir[0])
    normal_dir = np.cross(np.array((1,0,0)), x_dir)
    normal_dir /= np.linalg.norm(normal_dir)
    rot_mat = rotation_matrix(angle, normal_dir, center)
    rot_pnts_mm = np.dot(points_mm, rot_mat[:3, :3].T) + rot_mat[3, :3]
    return rot_pnts_mm, np.abs(np.mod(angle,np.pi))

def get_x_direction(points_mm, curve_model):
    
    x_vector = np.linspace(np.min(points_mm[:, 0]), np.max(points_mm[:, 0]), 200)
    chord_pnts_mm = curve_model.get_curve_points(x_vector)
    center = np.average(chord_pnts_mm, axis = 0)
    chord_pnts_mm -= center
    cov_mat = np.dot(chord_pnts_mm.T, chord_pnts_mm)
    vals, directions = np.linalg.eigh(cov_mat)
    index = np.argmax(vals)
    return center, directions[:, index]