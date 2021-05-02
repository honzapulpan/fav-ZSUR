# coding: utf-8
import numpy as np

def norm_x(x):
    return x[0]**2+x[1]**2+x[2]**2

def linear(data, method, plt_lim):
    '''
    method: 0...rosenblatt, 1...konstantní přírůstek 2...upravený konstatní přírůstek
    '''
    cl_nb = len(data)
    
    ck = 1 # učící konstanta pro rosenblattův algoritmus
    beta = 0.1 # konstanta pro metodu konstatních přírůstků
    delta = 0. # pásmo necitlivosti
    
    # přidání 1 na začátek všech dat
    w_data = []
    for i in range(cl_nb):
        ones = np.ones((len(data[i]),1))
        w_data.append(np.concatenate((ones,data[i]),axis=1))

    # vytvoření prázdné matice prvků diskriminačních funkcí fí
    fi = []
    for i in range(cl_nb): #n
        fi.append([])
        for c in range(cl_nb):
            # vzhledem k tomu, že jsou data centrovaná kolem (0,0), počáteční aproximace je [0,0,0] 
            fi[i].append(np.array([0,0,0])) 

    #napočítání jednotlivých fí funkcí
    ndone = True
    max_iter = 100
    cnt = 1
    iter_cnt=0
        
    for i in range(cl_nb-1):
        for j in range(i+1, cl_nb):
            ndone = True
            while ndone and cnt <= max_iter:
                min_len=np.amin([len(w_data[j]),len(w_data[i])])
                min_pos=np.argmin([len(w_data[j]),len(w_data[i])])
                
                #procházím po jednom na přeskáčku data ze 2 tříd
                #pokud klasifikátor klasifikuje špatně, měním ho
                for n in range(min_len):
                    if np.dot(fi[i][j],w_data[j][n,:].T) >= delta:
                        if method == 0:
                            iter_cnt+=1
                            fi[i][j] = fi[i][j] + ck*(-1)*w_data[j][n,:]
                        elif method == 1:
                            iter_cnt+=1
                            ck = beta/norm_x(w_data[j][n,:])
                            fi[i][j] = fi[i][j] + ck*(-1)*w_data[j][n,:]
                        elif method == 2:
                            while np.dot(fi[i][j],w_data[j][n,:].T) >= delta:
                                iter_cnt+=1
                                ck = beta/norm_x(w_data[j][n,:])
                                fi[i][j] = fi[i][j] + ck*(-1)*w_data[j][n,:]
                    if np.dot(fi[i][j],w_data[i][n,:].T) < delta:
                        if method == 0:
                            iter_cnt+=1
                            fi[i][j] = fi[i][j] + ck*w_data[i][n,:]
                        elif method == 1:
                            iter_cnt+=1
                            ck = beta/norm_x(w_data[i][n,:])
                            fi[i][j] = fi[i][j] + ck*w_data[i][n,:]
                        elif method == 2:
                            while np.dot(fi[i][j],w_data[i][n,:].T) < delta:
                                iter_cnt+=1
                                ck = beta/norm_x(w_data[i][n,:])
                                fi[i][j] = fi[i][j] + ck*w_data[i][n,:]
                if min_pos == 0:
                    max_pos = i
                    z = 1
                else:
                    max_pos = j
                    z = -1
                for n in range(min_len, len(w_data[max_pos])):
                    if np.dot(fi[i][j],w_data[max_pos][n,:].T)*z <= delta:
                        if method == 0:
                            iter_cnt+=1
                            fi[i][j] = fi[i][j] + ck*z*w_data[max_pos][n,:]
                        elif method == 1:
                            iter_cnt+=1
                            ck = beta/norm_x(w_data[max_pos][n,:])
                            fi[i][j] = fi[i][j] + ck*z*w_data[max_pos][n,:]
                        elif method == 2:
                            while np.dot(fi[i][j],w_data[max_pos][n,:].T)*z <= delta:
                                iter_cnt+=1
                                ck = beta/norm_x(w_data[max_pos][n,:])
                                fi[i][j] = fi[i][j] + ck*z*w_data[max_pos][n,:]
                fi[j][i] = -1*fi[i][j]

                ndone = False
                for n in range(len(w_data[i])):
                    if np.dot(fi[i][j],w_data[i][n,:].T) <= delta:
                        ndone = True
                        cnt += 1
                        break
                for n in range(len(w_data[j])):
                    if np.dot(fi[i][j],w_data[j][n,:].T) > delta:
                        ndone = True
                        cnt += 1
                        break 

    # vytvořit meshgrid
    grid_dim = 300
    X,Y = np.meshgrid(np.linspace(-plt_lim,plt_lim,grid_dim),np.linspace(-plt_lim,plt_lim,grid_dim))

    Z = np.zeros((grid_dim,grid_dim))

    for x in range(grid_dim):
        for y in range(grid_dim):
            fi_mx = np.zeros((cl_nb,cl_nb))
            for i in range(cl_nb):
                for j in range(cl_nb):
                    fi_mx[i,j] = np.sign(np.dot(fi[i][j],[1,X[x,y],Y[x,y]]))
            np.fill_diagonal(fi_mx,0) 

            fi_mx_sum_p = np.where(np.sum(fi_mx,axis=1) >= 3.)
            if len(fi_mx_sum_p[0])==1:
                Z[x,y] = fi_mx_sum_p[0]
            else:
                Z[x,y] = 4


    return X,Y,Z,iter_cnt