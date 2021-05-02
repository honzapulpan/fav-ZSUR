
# coding: utf-8
import numpy as np
from tools import dist_v, dist_m, metric

def maximin(data):
    #definice konstanty
    q = 0.5

    data_len = len(data)
    
    #volím náhodně jeden obraz = 1. střední hodnota
    mi = []
    mi0_ind = np.random.randint(data_len) 
    mi.append(data[mi0_ind])
    data = np.delete(data, mi0_ind, 0)

    mi1_ind = np.argmax(dist_v(data,mi[0]), axis=None)
    mi.append(data[mi1_ind])
    data = np.delete(data, mi1_ind, 0)

    d_max=1.
    d_test=0.

    while d_max > d_test:
        dist_vectors = []
        for i in range(0,len(mi)):
            dist_vectors.append(dist_v(data,mi[i]))

        min_vector = dist_vectors[0]    
        for i in range(1,len(mi)):
            min_vector = np.minimum(min_vector, dist_vectors[i]) 

        d_max_ind = np.argmax(min_vector)    
        d_max = min_vector[d_max_ind]

        d_count = 1.0
        d_val = 0.0
        for i in range(0,len(mi)-1):
            for j in range(i+1,len(mi)):
                d_val = d_val + metric(mi[i],mi[j])
                d_count += 1
        d_test = q * d_val / d_count

        if d_test <= d_max:
            mi.append(data[d_max_ind])
            data = np.delete(data, d_max_ind, 0)
    return len(mi)

