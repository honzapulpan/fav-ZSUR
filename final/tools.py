import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors

# vypočte matici vzdáleností 2-rozměrných dat v matici d
def dist_m(d):
    
    dl = len(d)
    dm = np.zeros((dl,dl))

    for i in range(0,dl):
       dm[i,:] = np.sum((d-d[i,:])**2, axis=1)

    np.fill_diagonal(dm, np.nan)
    
    return dm

#vypočte vzdálenostní vektor od vektoru d a prvku s (který není prvkem vektoru d) 
def dist_v(d,s):
    return np.sum((d-s)**2, axis=1)

# definice metriky 2 prvků (standardně vynechávám odmocninu, nemá na výsledek vliv)
def metric(x1,x2):
    return np.sum((x1-x2)**2)

def plot_classifier_result(data,X,Y,Z,plt_lim):

    cmap = pltcolors.LinearSegmentedColormap.from_list('', ['red','green','blue','orange','yellow','pink','brown','silver'])
    norm=plt.Normalize(0,7)
    plt.scatter(X,Y,c=Z, cmap=cmap, norm=norm)
    #plt.scatter(X,Y,c=Z)

    plt.scatter(data[:,0],data[:,1], color='black', marker='.', alpha=0.3)

    plt.grid(True, which='both')
    plt.axhline(y=0, color='gray')
    plt.axvline(x=0, color='gray')
    plt.xlim(-plt_lim,plt_lim)
    plt.ylim(-plt_lim,plt_lim)
    plt.show()


