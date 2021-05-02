# coding: utf-8
import numpy as np
from tools import dist_v

####### KNN ######
def kNN(data, k, plt_lim):
    cl_nb = len(data)
    data_len = 0
    for i in range(cl_nb):
        data_len += len(data[i])

    # vytvořit meshgrid
    grid_dim = 300
    X,Y = np.meshgrid(np.linspace(-plt_lim,plt_lim,grid_dim),np.linspace(-plt_lim,plt_lim,grid_dim))

    def f(x,y):
        z = np.zeros(cl_nb)
        for j in range(cl_nb):
            z[j] = np.sum(np.partition(dist_v(data[j],[x,y]),k)[:k])
        return np.argmin(z)

    vf = np.vectorize(f)

    Z = vf(X,Y)
    
    return X,Y,Z




