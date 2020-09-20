# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 21:55:33 2020

@author: Asus
"""
#%%
from NNLS_SPAR import *
from static_questions import * 
from matplotlib.pyplot import *
import numpy as np
from pysptools import *

import pysptools.util as util
import pysptools.eea as eea
import pysptools.abundance_maps as amp

from scipy.io import loadmat

#%% Read Salina

Sal1 = loadmat('SalinasA.mat')
Sal2 = loadmat('SalinasA_corrected.mat');
Sal3 = loadmat('SalinasA_gt.mat');

A1 = Sal1["salinasA"]
A2 = Sal2["salinasA_corrected"]
A3 = Sal3["salinasA_gt"]

m, n, k = A2.shape
A = np.zeros((m*n,k))
b = np.zeros((m*n,1))
   
#USE N-FINDR for endmember matrix
ee = eea.NFINDR()

endmem = ee.extract(A2, 12, maxit=5, normalize=False, ATGP_init=True)
endmem = endmem.T
endmem = endmem.astype(np.float64)

b_temp = A2[45,10,:]
b_temp = b_temp.reshape(204,1)
b_temp = b_temp.astype(np.float64)

s = 6  #sparsity level

#%% Solver

#Solve
omega = NNLSSPAR(endmem,b_temp,s)
omega.solve_nnls(list(range(12)))

#solution in vector form 
w = np.zeros((12,1))
indices = omega.indexes[0]
values = omega.coefficients[0]

w[indices,0] = values

#reconstruct
yhat = endmem@w

#MSE can be confusing due to the scale atm
MSE = np.sum((b_temp - yhat)**2)
print('MSE:' , MSE)

plot(b_temp,label='b_temp')
plot(yhat,label='yhat')
legend()
show()

#%% Complete Solver
w_list = np.zeros((m,n,12))
w_list2 = np.zeros((m,n,12))
err_list = np.zeros((m,n))
err_list2 = np.zeros((m,n))
err_list3 = np.zeros((m,n))
reconstructed = np.zeros((m,n,k))
reconstructed2 = np.zeros((m,n,k))

for col in range(m):
    for row in range(n):
        #select pixel
        b = A2[col,row,:]
        b = b.reshape(k,1)
        b = b.astype(np.float64)
        
        #Solve
        omega = NNLSSPAR(endmem,b,s)
        omega.solve_nnls(list(range(12)))
        
        #solution in vector form 
        w = np.zeros((12,1))
        indices = omega.indexes[0]
        values = omega.coefficients[0]
        
        w[indices,0] = values
        w2 = w.reshape(12,)
        w_list[col,row,:] = w2
        
        #reconstruct
        yhat = endmem@w
        yhat2 = yhat.reshape(k,)
        reconstructed[col,row] = yhat2
        
        #MSE can be confusing due to the scale atm
        MSE = np.sum((b - yhat)**2)/204
        err_list[col,row] = MSE 
        
        
for col in range(m):
    for row in range(n):
        #select pixel
        b = A2[col,row,:]
        b = b.reshape(k,1)
        b = b.astype(np.float64)
        
        #Solve
        omega = NNLSSPAR(endmem,b,s)
        omega.gurobi_mip2(list(range(12)))
        
        #solution in vector form 
        w_mip = np.zeros((12,1))
        indices = omega.indexes[0]
        values = omega.coefficients[0]
        
        w_mip[indices,0] = values
        w2 = w_mip.reshape(12,)
        w_list2[col,row,:] = w2
        
        #reconstruct
        yhat = endmem@w_mip
        yhat2 = yhat.reshape(k,)
        reconstructed2[col,row] = yhat2
        
        #MSE can be confusing due to the scale atm
        MSE = np.sum((b - yhat)**2)/204
        err_list2[col,row] = MSE 
        
                     

imshow(err_list,cmap='hot')
figure()
imshow(err_list2,cmap='hot')
#%%

plot(w_list[55,50,:])
plot(w_list2[55,50,:])
show()

#%% Plots

err_plot = np.zeros((m,n,6))
err_plot[:,:,0] = err_list
err_plot[:,:,1] = err_list2
err_plot[:,:,3] = reconstructed[:,:,25]
err_plot[:,:,4] = reconstructed2[:,:,25]
err_plot[:,:,2] = np.zeros((83,86))
err_plot[:,:,5] = A2[:,:,25]


width = 5
height = 5
rows = 2
cols = 3
axes=[]
fig=figure()

for a in range(rows*cols):
    axes.append( fig.add_subplot(rows, cols, a+1) )
    subplot_title=("Image "+str(a))
    axes[-1].set_title(subplot_title)  
    imshow(err_plot[:,:,a])
fig.tight_layout()    
show()


errs = err_list.reshape(m*n,)
errs2 = err_list2.reshape(m*n,)

#min_idx = np.min(errs)

print('1-2=',np.sum(errs)-np.sum(errs2))

fig.savefig('destination_path.eps', format='eps')



