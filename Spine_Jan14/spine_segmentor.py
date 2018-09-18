import numpy as np

class SpineSegmentor:
    
    def __init__(self, spine_pnts, x_maximas, x_scale = 4):
        
        self.spine_pnts = spine_pnts
        self.spine_pnts[:,0] *= x_scale
        self.x_maximas = x_maximas*x_scale
        self.x_maximas.sort()

    def get_segmented_spine(self):

        segemented_spine = list()
        T = self.spine_pnts[:, 0] <= self.x_maximas[0]
        segemented_spine.append(self.spine_pnts[T])
        for maxima_index in range(len(self.x_maximas) - 1):
            self.__add_segment(maxima_index, segemented_spine)
        S = self.spine_pnts[:, 0] > self.x_maximas[-1]
        segemented_spine[-1] = np.vstack((segemented_spine[-1], self.spine_pnts[S]))
        return np.array(segemented_spine)
        
    def __add_segment(self, maxima_index, segemented_spine):
        
        segmented_pnts = self.__segment_two_verts(maxima_index)
        segemented_spine[-1] = np.vstack((segemented_spine[-1], segmented_pnts['low_pnts']))
        segemented_spine.append(segmented_pnts['high_pnts'])
        print('segment ' + str(maxima_index) + ' added')
        
    def __segment_two_verts(self, maxima_index):
        from sklearn.cluster import AgglomerativeClustering
        
        T = np.logical_and(self.spine_pnts[:,0] > self.x_maximas[maxima_index], 
                           self.spine_pnts[:,0] <= self.x_maximas[maxima_index + 1])
        two_verts_pnts = self.spine_pnts[T]
        cluster_labels = AgglomerativeClustering(2).fit_predict(two_verts_pnts)
        a_vert = two_verts_pnts[cluster_labels == 0]
        b_vert = two_verts_pnts[cluster_labels == 1]
        if np.mean(a_vert[:,0]) < np.mean(b_vert[:,0]):
            return {'low_pnts': a_vert, 'high_pnts': b_vert}
        return {'low_pnts': b_vert, 'high_pnts': a_vert}
        
def plot_segmented_spine(segemented_spine_pnts):
    from mayavi import mlab
    
    for index, segment in enumerate(segemented_spine_pnts):
        color = [0,0,0]
        color[index%3] = 1
        mlab.points3d(segment[:,0] /4, segment[:,1], segment[:,2], color = tuple(color), mode = 'point')