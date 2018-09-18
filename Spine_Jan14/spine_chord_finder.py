from mayavi import mlab
import numpy as np
import pandas as pd

def zy_polynomial_fit_points(points, dbg = False):
    
    coeff = np.polyfit(points[:,0], points[:,(1,2)], 4)
    poly_y = np.poly1d(coeff[:,0])
    poly_z = np.poly1d(coeff[:,1])
    if dbg:
        x_min = np.min(points[:,0])
        x_max = np.max(points[:,0])
        xp = np.linspace(x_min, x_max, 100)
        mlab.plot3d(xp, poly_y(xp), poly_z(xp), color = (0,0,1), line_width = 100)
    return coeff, poly_y, poly_z
    
def get_points_near_spine(zy_model, points, zy_threshold, dbg = False):
    
    perd_pnts = zy_model.get_curve_points(points[:,0])
    T = np.linalg.norm(perd_pnts[:, (1,2)] - points[:, (1,2)], axis =1) < zy_threshold
    ans = points[T]
    if dbg:
        mlab.points3d(points[:,0], points[:,1], points[:,2], mode = 'point', color = (1,0,0))
        mlab.points3d(ans[:,0], ans[:,1], ans[:,2], mode = 'point', color = (0,1,0))
    return ans
    
class ZYOfXMedianModel:
    
    def __init__(self, points, zy_threshold = 1, dbg = False):
        
        self.__set_normaliztion(points)
        self.__construct_from_points(points)
        spine_pnts = get_points_near_spine(self, points, zy_threshold)
        self.__construct_from_points(spine_pnts)
        if dbg:
            self.plot_curve(points[:,0])
        
    def get_curve_points(self, x_vector):
        
        normed_x = (x_vector - self.center[0])/self.width[0]
        result = np.empty((normed_x.shape[0],3))
        dict_pnts = {'x': normed_x}
        result[:,1] = self.res_y.predict(dict_pnts)
        result[:,2] = self.res_z.predict(dict_pnts)
        result[:,0] = normed_x
        return result*self.width + self.center
    
    def __construct_from_points(self, points):
        import statsmodels.formula.api as smf
        
        normed_pnts = self.__get_normed_pnts(points)
        dat_pnts = pd.DataFrame(normed_pnts, columns = ('x', 'y', 'z'))
        mod_y = smf.quantreg('y ~ x + I(x**2.0) + I(x**3.0) + I(x**4.0)', dat_pnts)
        mod_z = smf.quantreg('z ~ x + I(x**2.0) + I(x**3.0) + I(x**4.0)', dat_pnts)
        self.res_y = mod_y.fit(q = 0.5)
        self.res_z = mod_z.fit(q = 0.5)
        
    def __set_normaliztion(self, points):
        
        self.center = np.average(points, axis = 0)
        self.width = np.std(points, axis = 0)
        
    def __get_normed_pnts(self, points):
        
        norm_pnts = points - self.center
        norm_pnts /= self.width
        return norm_pnts
        
    def plot_curve(self, x_vector):
        
        x_vector = np.array(sorted(x_vector))
        pred_pnts = self.get_curve_points(x_vector)
        mlab.points3d(pred_pnts[:,0], pred_pnts[:,1], pred_pnts[:,2], mode = 'point', color = (1,1,1))