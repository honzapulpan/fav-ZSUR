# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from tools import dist_m

def retezova_mapa(data, pp):
    '''
    pp ... True/False - tisknout výslednou funkci ano/ne
    '''
    data_len = len(data)
    d = np.full([data_len-1], np.nan)
    
    # matice vzdáleností
    dist_matrix = dist_m(data)

    # řádek vybírat náhodně
    sel_row = np.random.randint(data_len)  #19 JE DOCELA VHODNÁ

    for i in range(0,data_len-1):
        min_ind = np.nanargmin(dist_matrix[sel_row,:], axis=None)
        d[i] =  dist_matrix[sel_row,min_ind]
        dist_matrix[sel_row,:] = np.inf
        dist_matrix[:,sel_row] = np.inf
        sel_row = min_ind
    
    if pp:
        plt.plot(d)
        plt.show()

    #stanovení počtu tříd
    d_sort = np.sort(d)
    cl_nb = len(d_sort[d_sort>=10])+1
    
    return(cl_nb)


