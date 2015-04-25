from __future__ import division
import matplotlib
matplotlib.use('Agg')
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %matplotlib inline
#%precision 4
plt.style.use('ggplot')

np.random.seed(1234)
import scipy.stats as stats
import timeit
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

fig = plt.figure(figsize=(12,3)) # (num=None, tight_layout=True, figsize=(12,3), dpi=80, facecolor='w', edgecolor='k')
fig1 = fig.add_subplot(141)
fig1.pcolormesh(basis1,cmap=plt.cm.gray)     
fig2 = fig.add_subplot(142)
fig2.pcolormesh(basis2,cmap=plt.cm.gray)  
fig3 = fig.add_subplot(143)
fig3.pcolormesh(basis3,cmap=plt.cm.gray)  
fig4 = fig.add_subplot(144)
fig4.pcolormesh(basis4,cmap=plt.cm.gray) 

fig.savefig('basis_images.png')
plt.close()
print "Latent feature matrices (A):"