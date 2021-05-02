# coding: utf-8


import numpy as np

def shlukove_hladiny(data):
    
    data_len = len(data)

    h = np.full([data_len-1], np.nan)
    dist_matrix = np.full((data_len,data_len),np.nan)

    for i in range(0,data_len):
       dist_matrix[i,:] = np.sum((data-data[i,:])**2, axis=1)

    np.fill_diagonal(dist_matrix, np.nan)

    for i in range(0,data_len-1):

        #### nalezení těch, co k sobě mají nejblíže
        min_ind = np.unravel_index(np.nanargmin(dist_matrix, axis=None), dist_matrix.shape)

        h[i] = dist_matrix[min_ind[0],min_ind[1]]

        item1_dist_v = dist_matrix[:,min_ind[0]]
        item2_dist_v = dist_matrix[:,min_ind[1]]

        new_dist_v = np.minimum(item1_dist_v, item2_dist_v)

        #do sloupce a řádku prvního prvku vložím novy distanční vektor
        dist_matrix[:,min_ind[0]] = np.transpose(new_dist_v)
        dist_matrix[min_ind[0],:] = new_dist_v

        #smazání sloupce a řádku s druhým prvkem
        dist_matrix[min_ind[1],:] = np.inf
        dist_matrix[:,min_ind[1]] = np.inf
        
        #stanovení počtu tříd
        h_dist = np.floor(np.log10(np.ediff1d(h))).astype(int)
        cl_nb = len(h_dist[h_dist>=0])+1 #+1 protože top číslo je vzdálenost 2 tříd, všechny nižší jednu přidávaj

    return cl_nb