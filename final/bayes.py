# coding: utf-8

import numpy as np


def bayes(data, plt_lim):
    """
    data ... pole roztříděných 2-rozměrných dat    
    
    return ... pole X,Y,Z s meshgridem a ohodnocením barvou na tomto gridu
    """
 
    cl_nb = len(data)
    data_len = 0
    for i in range(cl_nb):
        data_len += len(data[i])
        
    # pro všechny třídy odhadnu pravděpodobnost třídy a parametry normálního rozdělení
    Pw = []
    mi = [] 
    sigma_det = []
    sigma_inv = []

    for i in range(cl_nb):
        Pw.append(len(data[i])/data_len)
        mi.append(np.mean(data[i],axis=0))
        sigma = np.cov(data[i].T)
        sigma_det.append(np.linalg.det(sigma))
        sigma_inv.append(np.linalg.inv(sigma))

    # vytvořit meshgrid
    grid_dim = 300 # počet prvků dělení gridu
    X,Y = np.meshgrid(np.linspace(-plt_lim,plt_lim,grid_dim),np.linspace(-plt_lim,plt_lim,grid_dim))

    # pro každý bod gridu určím třídu, do které podle bayese patří
    z = np.zeros((cl_nb,grid_dim,grid_dim))
    for j in range(cl_nb):
        for x in range(0,grid_dim):
            for y in range(0,grid_dim):
                z[j][x,y] = np.exp(-1/2*np.dot(np.dot([X[x,y],Y[x,y]]-mi[j],sigma_inv[j]),[X[x,y],Y[x,y]]-mi[j]))/(np.sqrt((2*np.pi)**2 * sigma_det[j]))

    Z = np.argmax(z, axis=0)
    return X,Y,Z

