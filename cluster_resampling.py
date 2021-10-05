# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:28:23 2019

@author: Lionel Yelibi
Cluster resampling scheme, See Section V 
Agglomerative Likelihood Clustering (2019,2020,2021) See pre-print: https://arxiv.org/abs/1908.00951

"""

import numpy as np
import itertools
from sklearn import metrics
from timeseries_generator import onefactor
from function_aspcv3 import alc



sizes = range(1000,3500,500)
T = range(10,35,5)
aggdata = { i : {} for i in sizes}
clusters_number = 10
gs = 1 
model = 'normal'

for N, t in zip(sizes, T):    

    data, key = onefactor(N, clusters_number,t,coupling_parameter = gs, model=model)
    print('noise signal ratio', t/N)    
    
    ''' ALC '''
    sample_size = int(1/(0.1/t))
    
    spin = { i : {j : 0 for j in range(i+1,N)} for i in range(N-1)}
    freq = { i : {j : 0 for j in range(i+1,N)} for i in range(N-1)}
    ari=0
    tot_iter = 0
    while ari <.9 and tot_iter<=2000:
        
        for iterations in range(100):
            idx = np.random.choice(range(N),size=sample_size,replace=False)
            for i, j in list(itertools.combinations(idx,2)):
                mi, ma = min([i,j]),max([i,j])
                freq[mi][ma]+=1
            temp = data[idx]
            aspc_solution = alc(temp)
            labels = np.unique(aspc_solution)
            indices = [ idx[aspc_solution == label] for label in labels]
            for idx_ in indices:
                for i,j in list(itertools.combinations(idx_,2)):
                    mi, ma = min([i,j]),max([i,j])
                    spin[mi][ma]+=1
        tot_iter+=100
        pdf = np.zeros((N,N))
        for i,j in list(itertools.combinations(range(N),2)): 
            try: pdf[i,j] = spin[i][j]/freq[i][j]
            except:
                continue
        pdf_ = (pdf + pdf.T)
        
        thres=.5
        tracker = { i:i for i in range(N)}
        
        for i in range(N):
            idx = np.arange(N,dtype=int)[pdf_[i]>=thres]
            for j in idx:
                tracker[j] = tracker[i]
        t_solution = list(tracker.values())
        solution =np.ones(N)*-1
        k=0
        for i in np.unique(t_solution):
            solution[t_solution==i]=k
            k+=1
        ari = metrics.adjusted_rand_score(key,solution)
        aggdata[N][tot_iter] = ari
        print(ari)

# np.save('surrogate_ari05.npy', aggdata)
# data = np.load('surrogate_ari.npy',allow_pickle=True).item()
# T = list(T)
# k=0
# plt.figure()
# for key in data.keys():
#     x = list(data[key].keys())
#     y = list(data[key].values())
#     plt.scatter(x,y,label = r'N=%s n=%s t=%s' % (key,int(key/10),T[k]))
#     plt.plot(x,y)
#     plt.xlabel('iterations')
#     plt.ylabel('ari')
#     # plt.legend()
#     k+=1
# plt.tight_layout()
