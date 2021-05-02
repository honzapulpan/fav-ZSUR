# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors

def min_dist(data, plt_lim):
    cl_nb = len(data)
    data_len = 0
    for i in range(cl_nb):
        data_len += len(data[i])

    # pro všechny třídy určit střední hodnotu
    mi = [] 
    for i in range(cl_nb):
        mi.append(np.mean(data[i],axis=0))

    # vytvořit meshgrid
    grid_dim = 300
    X,Y = np.meshgrid(np.linspace(-plt_lim,plt_lim,grid_dim),np.linspace(-plt_lim,plt_lim,grid_dim))

    # pro každý bod gridu určit 4 hodnoty funkce a vybrat tu nejmenší (označit číslem 0-3, nebo rovnou barvou)
    z = np.zeros((cl_nb,grid_dim,grid_dim))
    for j in range(cl_nb):
        for x in range(0,grid_dim):
            for y in range(0,grid_dim):
                x_mi = [X[x,y],Y[x,y]]-mi[j]
                z[j][x,y]= x_mi[0]**2+x_mi[1]**2

    Z = np.argmin(z, axis=0)
    
    return X,Y,Z

