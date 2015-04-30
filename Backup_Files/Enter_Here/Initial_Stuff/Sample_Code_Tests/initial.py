from __future__ import division
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %matplotlib inline
# %precision 4
plt.style.use('ggplot')

np.random.seed(1234)
import scipy.stats as stats
import timeit

def IBP(N, alpha):
    """Indian Buffet Process (IBP) steps:
    Input: N is the number of customers (objects, images); alpha is the only parameter;
    Return: result is the binary matrix (prior); Kplus is the number of dishes (features)"""
    result = np.zeros((N,1000))
    
    # Step 1: First customer takes a Poisson(alpha) of dishes
    t = stats.poisson.rvs(alpha) # (set the random seed when calling the function)
    if t > 0:
        result[0,0:t] = 1
    
    # Kplus = the number of features for which m_k > 0 (m_k: the number of previous customers who sampled that dish)
    Kplus = t
    for i in range(1,N):
        for k in range(Kplus):
            # Step 2: The ith customer takes dish k with probability m_k/i
            p = np.sum(result[0:(i+1),k])/(i+1) # this is a probability, so should be between 0 and 1
            assert p <= 1 
            assert p >= 0
            if stats.uniform.rvs(0) < p:
                result[i,k] = 1
            else:
                result[i,k] = 0
                
        # Step 3: The ith customer tries a Poisson(alpha/i) number of new dishes
        t = stats.poisson.rvs(alpha/(i+1))
        if t > 0:
            result[i,Kplus:(Kplus+t)] = 1
        Kplus += t
    result = result[:,0:Kplus]
    
    return result, Kplus

# --------------------------------------------------------------------------------------------------

# Basis images
import Image
basis1 = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,1,0,0,0,0],[1,1,1,0,0,0],[0,1,0,0,0,0]])
basis2 = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
basis3 = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,1],[0,0,0,0,1,1],[0,0,0,1,1,1]])
basis4 = np.array([[0,0,0,1,0,0],[0,0,0,1,1,1],[0,0,0,1,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])

D = 36
b1 = basis1.reshape(D)
b2 = basis2.reshape(D)
b3 = basis3.reshape(D)
b4 = basis4.reshape(D)
A = np.array([b1,b2,b3,b4])

# These are heatmaps!
plt.figure(num=None, tight_layout=True, figsize=(12,3), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(141)
plt.pcolormesh(basis1,cmap=plt.cm.gray)     
plt.subplot(142)
plt.pcolormesh(basis2,cmap=plt.cm.gray)  
plt.subplot(143)
plt.pcolormesh(basis3,cmap=plt.cm.gray)  
plt.subplot(144)
plt.pcolormesh(basis4,cmap=plt.cm.gray)  
plt.savefig('basis_images.png')
print "Latent feature matrices (A):"