import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def draw(border, n):
    plt.imsave(r'E:\vertebrate-segmentation\result\border\Im'+str(n)+'.jpg', np.array(border).reshape(512,512), cmap=cm.gray)
    return