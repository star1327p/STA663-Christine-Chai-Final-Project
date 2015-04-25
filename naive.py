from __future__ import division
import matplotlib
matplotlib.use('Agg')
import functions as func
# You need to convert them into .py files 
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

# -------------------------------------------------------------------------------------------------------

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

# Generate image data: 100 matrices of size 6*6
N = 100
D = 36
K = 4
sigmaX_orig = 0.5

# All K basis images, each of length D
# Generate N images (customers, objects)
np.random.seed(1234)
images = np.zeros((N,6,6)) # simulated image data
structure = np.zeros((N,6,6))  # 0/1 structure for each image
add = stats.bernoulli.rvs(0.5,size=(N,K)) # whether the K=4 latent bases are present in each image
epsilon = stats.norm.rvs(loc=0,scale=0.5,size = (N,6,6)) # random noise

for i in range(N):
    structure[i,:,:] = add[i,0]*basis1 + add[i,1]*basis2 + add[i,2]*basis3 + add[i,3]*basis4
    images[i,:,:] = structure[i,:,:] + epsilon[i,:,:]

Z_orig = add   

# print images.shape
print "Example image:\n",images[4]
# plt.figure(tight_layout=True, figsize=(3,3),dpi=80)
figEx = plt.figure(figsize=(3,3)) # (num=None, tight_layout=True, figsize=(3,3), dpi=80, facecolor='w', edgecolor='k')
fig1 = figEx.add_subplot(111)
fig1.pcolormesh(images[4],cmap=plt.cm.gray)
figEx.savefig('example_image.png')
plt.close()

# Generate the Harmonic numbers, but we only need the sum
from fractions import Fraction
sum_Harmonics = 0
Harmonics = 0
for i in range(N):
    sum_Harmonics += (N-i)*Fraction(1,i+1)
    Harmonics += Fraction(1,i+1)
# print "Sum of H_1 + ... + H_N:", float(sum_Harmonics)
# print "Harmonic number H_N:", float(Harmonics)

# -------------------------------------------------------------------------------------------------------

# Initialization
N = 100
D = 36
K = 4
sigmaA = 1
sigmaX = 1

np.random.seed(1005)
# alpha = stats.gamma.rvs(a = 1, loc = 0, scale = 1, size = 1)[0]
alpha = 1

K_inf = 1000
Z, Kplus = IBP(N, alpha)
print "Initial Kplus:", Kplus
print "Z.shape:",Z.shape # (100,4) = (N,Kplus) (latent)
print "A.shape:",A.shape # (4,36) = (Kplus,D)  (weight)

# Set MCMC steps
mcmc = 1000 # plan to sample for 1000 times

# Setup the array
Z_arr = np.zeros((mcmc,N,K_inf))
Kplus_arr = np.zeros(mcmc)
sigmaX_arr = np.zeros(mcmc)
sigmaA_arr = np.zeros(mcmc)
alpha_arr = np.zeros(mcmc)
rX_accept = 0
rA_accept = 0

# More initialization
np.random.seed(16)
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

Z_orig = np.zeros((N,4))
sigmaX_orig = 0.5
X = np.zeros((N,D))

for i in range(N):
    Z_orig[i,:] = stats.uniform.rvs(loc=0,scale=1,size=4) > 0.5
    while np.sum(Z_orig[i,:]) == 0:
        Z_orig[i,:] = stats.uniform.rvs(loc=0,scale=1,size=4) > 0.5
    X[i,:] = np.random.normal(size=D)*sigmaX_orig + np.dot(Z_orig[i,:],A)
    


# -----------------------------------------------------------------------------------------------------------------------------

# Gibbs Sampler -- Steps
np.random.seed(111)
elapsed = 0
elapsed1 = 0
elapsed2 = 0
elapsed1k_count = 0
elapsed1N_count = 0
elapsed1k_init = 0
elapsed1k_calc = 0

elapsed1_arr = np.zeros(mcmc)
elapsed2_arr = np.zeros(mcmc)
elapsed1k_count_arr = np.zeros(mcmc)
elapsed1N_count_arr = np.zeros(mcmc)
elapsed1k_init_arr = np.zeros(mcmc)
elapsed1k_calc_arr = np.zeros(mcmc)

start = timeit.default_timer()

# for mc in range(mcmc):
# for mc in range(1000):
for mc in range(1000): # just test for 10 iterations

    # Step 0: Save generated parameters to the MCMC array
    Z_arr[mc,:,0:Kplus] = Z[:,0:Kplus]
    # print "Shape of Z:",Z.shape
    
    alpha_arr[mc] = alpha
    Kplus_arr[mc] = Kplus
    sigmaX_arr[mc] = sigmaX
    sigmaA_arr[mc] = sigmaA
    
    print "At iteration",mc,": Kplus is",Kplus,", alpha is",alpha
    
    elapsed1_arr[mc] = elapsed1
    elapsed2_arr[mc] = elapsed2
    elapsed1k_count_arr[mc] = elapsed1k_count/N
    elapsed1N_count_arr[mc] = elapsed1N_count/N
    elapsed1k_init_arr[mc] = elapsed1k_init
    elapsed1k_calc_arr[mc] = elapsed1k_calc
    
    # print "Generating Z|alpha takes",elapsed1,"sec, and sampling sigmaX, sigmaA takes",elapsed2,"sec"
    # print "In generating Z|alpha, sampling from Kplus takes",elapsed1k_count/N,"sec, and sampling new dishes takes",elapsed1N_count/N,"sec"
    # print "In generating Z|alpha -- sampling from Kplus, initializing takes",elapsed1k_init,"sec; calculation takes",elapsed1k_calc,"sec"
    # print "----------------------------------------------------"
    
    # Step 1: Generate Z|alpha (Gibbs)
    start1 = timeit.default_timer()
    elapsed1k_count = 0
    elapsed1N_count = 0
    
    for i in range(N):
        # Save the matrix M so we won't need to calculate it again and again
        # naive.py
        M = func.calcM(Z,Kplus,sigmaX,sigmaA)
        
        start1k = timeit.default_timer()
        
        for k in range(Kplus):
            
            start1k_init = timeit.default_timer()
            
            # This is possible because Kplus may decrease in this loop (e.g. dropping redundant zeros)
            if (k+1) > Kplus:
                break
            if Z[i,k] > 0:
                # Take care of singular features
                # Get rid of the features not sampled (remove the zeros)
                if np.sum(Z[:,k]) - Z[i,k] <= 0: # whether the dish is sampled by other customers or not
                # if np.sum(Z[:,k]) - Z[i,k] == 0: # same as the code above since Z is binary
                    #Z[i,k] = 0
                    # Avoid Kplus to become zero!
#                     if Kplus == 1:
#                         Z[:,0] = Z[:,1]
#                     else: # Kplus > 1
                    Z[:,k:(Kplus-1)] = Z[:,(k+1):Kplus]
                    Kplus -= 1
                        # Z = Z[:,0:Kplus] # remove the last column
                    # naive.py
                    M = func.calcM(Z,Kplus,sigmaX,sigmaA)          
                    continue            
            
            elapsed1k_init = timeit.default_timer() - start1k_init
            
            start1k_calc = timeit.default_timer()
            
            # Effective inverse calculation from Griffiths and Ghahramani (2005; Equations 51 to 54)
            # M_(-i) = inv(inv(M) - zi.T * zi)
#             M0 = calcInverse(Z[:,0:Kplus], M, i, k, 0)
#             M1 = calcInverse(Z[:,0:Kplus], M, i, k, 1)
            
            # Then calculate the posterior distribution: prior * likelihood 
            # i.e. customers sample the dishes that have been previously sampled
            # Likelihood: P(X|Z_(-i,k),sigmaX,sigmaA)
            # Prior: P(z_ik = 1 | z_(-i,k)) = m_(-i,k) / N, where m_(-i,k) = number of objects possess feature k, excluding i
            P = np.zeros(2)
            Z[i,k] = 1
            M1 = func.calcM(Z,Kplus,sigmaX,sigmaA) 
            P[1] = func.log_likelihood(X,Z[:,0:Kplus],M1,sigmaA,sigmaX,Kplus,N,D) + np.log(sum(Z[:,k])-Z[i,k]) - np.log(N)
            Z[i,k] = 0
            M0 = func.calcM(Z,Kplus,sigmaX,sigmaA) 
            P[0] = func.log_likelihood(X,Z[:,0:Kplus],M0,sigmaA,sigmaX,Kplus,N,D) + np.log(N-sum(Z[:,k])) - np.log(N)
            P = np.exp(P - max(P))
            # Sample from the posterior distribution
            rand = stats.uniform.rvs(loc=0,scale=1,size=1)           
            if rand < P[0]/(P[0]+P[1]):
                Z[i,k] = 0
                #M = M0
            else:
                Z[i,k] = 1
                #M = M1

            elapsed1k_calc = timeit.default_timer() - start1k_calc
        elapsed1k = timeit.default_timer() - start1k
        elapsed1k_count += elapsed1k
        
        # Sample the number of new dishes Pois(alpha/i) for the current customer/object
        # Truncated prior: P(z_ik = 1 | z_(-i,k)) = (m_(-i,k) + alpha/Kplus) / (N + alpha/Kplus)
        
        start1N = timeit.default_timer()
        
        # trun = np.zeros(5)
        trun = np.zeros(4)
        # alphaN = alpha/N  # don't use alpha/i, or this can result in division by zero, but I don't know the details
        # alphaN = alpha/(i+1)
        alphaN = alpha/N
        # Note: in MATLAB, any matrix can be expanded => in Python, we need np.vstack and/or np.hstack       
        
        # for ki in range(5):
        for ki in range(4):
            if ki > 0:
                new_stack = np.zeros((N,ki))
                new_stack[i,:] = 1
                Z = np.hstack((Z[:,0:Kplus],new_stack))
            M = np.linalg.inv(np.dot(Z[:,0:(Kplus+ki)].T,Z[:,0:(Kplus+ki)])+((sigmaX/sigmaA)**2)*np.identity(Kplus+ki))
            # Prior: x ~ Pois(lambda): f(x) = ((lambda**x)/x!)*exp(-lambda), where x = ki, lambda = alphaN
            
            trun[ki] = (ki)*np.log(alphaN) - alphaN - np.log(np.math.factorial(ki)) 
            # posterior is proportional to prior x likelihood
            trun[ki] += func.log_likelihood(X,Z[:,0:(Kplus+ki)],M,sigmaA,sigmaX,Kplus+ki,N,D)
            
        # Z[i,Kplus:(Kplus+4)] = 0
        Z[i,Kplus:(Kplus+3)] = 0
        trun = np.exp(trun-max(trun))
        trun = trun/np.sum(trun)
        
        p = stats.uniform.rvs(loc=0,scale=1,size=1)  
        t = 0
        # for ki in range(5):
        for ki in range(4):
            t += trun[ki]
            if p < t:
                new_dishes = ki
                break
        Z[i,Kplus:(Kplus+new_dishes)] = 1
        Kplus += new_dishes
        
        elapsed1N = timeit.default_timer() - start1N
        elapsed1N_count += elapsed1N
        
    elapsed1 = timeit.default_timer() - start1
        
    # Step 2: Sample sigmaX_star (Metropolis)
    start2 = timeit.default_timer()
    
    # M = calcM(Z, Kplus+new_dishes, sigmaX, sigmaA)
    M = func.calcM(Z, Kplus, sigmaX, sigmaA)
    #logLik = log_likelihood(X, Z[:,0:(Kplus+new_dishes)], M, sigmaA, sigmaX, Kplus+new_dishes, N, D)
    logLik = func.log_likelihood(X, Z[:,0:Kplus], M, sigmaA, sigmaX, Kplus, N, D)
    epsilonX = stats.uniform.rvs(loc=0,scale=1,size=1) 
    if epsilonX < 0.5:
        # sigmaX_star = sigmaX - epsilonX/40
        # sigmaX_star = sigmaX - epsilonX/20
        sigmaX_star = sigmaX - stats.uniform.rvs(loc=0,scale=1,size=1)/20
    else:
        # sigmaX_star = sigmaX + epsilonX/20   
        sigmaX_star = sigmaX + stats.uniform.rvs(loc=0,scale=1,size=1)/20 
    # M_Xstar = calcM(Z, Kplus+new_dishes, sigmaX_star, sigmaA)
    M_Xstar = func.calcM(Z, Kplus, sigmaX_star, sigmaA)
    # logLikX_star = log_likelihood(X, Z[:,0:(Kplus+new_dishes)], M_Xstar, sigmaA, sigmaX_star, Kplus+new_dishes, N, D)
    logLikX_star = func.log_likelihood(X, Z[:,0:Kplus], M_Xstar, sigmaA, sigmaX_star, Kplus, N, D)
    acc_X = np.exp(min(0, logLikX_star-logLik))
    
    # Step 3: Sample sigmaA_star (Metropolis)
    epsilonA = stats.uniform.rvs(loc=0,scale=1,size=1)
    if epsilonA < 0.5:
        # sigmaA_star = sigmaA - epsilonA/20
        sigmaA_star = sigmaA - stats.uniform.rvs(loc=0,scale=1,size=1)/20
    else:
        # sigmaA_star = sigmaA + epsilonA/40 
        sigmaA_star = sigmaA + stats.uniform.rvs(loc=0,scale=1,size=1)/20
        # sigmaA_star = sigmaA + epsilonA/20   
    # M_Astar = calcM(Z, Kplus+new_dishes, sigmaX, sigmaA_star)
    M_Astar = func.calcM(Z, Kplus, sigmaX, sigmaA_star)
    # logLikA_star = log_likelihood(X, Z[:,0:(Kplus+new_dishes)], M_Astar, sigmaA_star, sigmaX, Kplus+new_dishes, N, D)
    logLikA_star = func.log_likelihood(X, Z[:,0:Kplus], M_Astar, sigmaA_star, sigmaX, Kplus, N, D)
    acc_A = np.exp(min(0, logLikA_star-logLik))
    
    randX = stats.uniform.rvs(loc=0,scale=1,size=1)
    if randX < acc_X:
        sigmaX = sigmaX_star
        rX_accept += 1
    randA = stats.uniform.rvs(loc=0,scale=1,size=1)
    if randA < acc_A:
        sigmaA = sigmaA_star
        rA_accept += 1
    
    elapsed2 = timeit.default_timer() - start2
    
    # Step 4: Sample alpha|Z ~ Ga(a=1+Kplus,scale=1+Harmonics)
    alpha = stats.gamma.rvs(a = 1+Kplus, loc = 0, scale = np.reciprocal(1+Harmonics),size=1)[0]
    
elapsed = timeit.default_timer() - start
print "It takes",elapsed,"sec to run 1000 iterations"

# ------------------------------------------------------------------------------------------

### Plot the traceplots for the IBP results

fig = plt.figure(figsize=(8,8))
fig1 = fig.add_subplot(411)
fig1.plot(Kplus_arr)
fig1.set_ylabel('Kplus')
fig2 = fig.add_subplot(412)
fig2.plot(alpha_arr)
fig2.set_ylabel('alpha')
fig3 = fig.add_subplot(413)
fig3.plot(sigmaX_arr)
fig3.set_ylabel('sigmaX')
fig4 = fig.add_subplot(414)
fig4.plot(sigmaA_arr)
fig4.set_ylabel('sigmaA')
fig4.set_xlabel('Index')
fig.savefig('IBP_plot_results.png')
plt.close()

###### Setup the array
Kplus_final = Kplus_arr[996] # in fact it is 5 =.=
# Kplus_final = 4
# Z_final = Z_arr[996,:,0:Kplus_final-1].reshape(N,Kplus_final-1)
# Z_final = Z_arr[996,:,1:Kplus_final].reshape(N,Kplus_final-1)
Z_final = Z_arr[996,:,0:Kplus_final].reshape(N,Kplus_final)
sigmaX_final = sigmaX_arr[996]
sigmaA_final = sigmaA_arr[996]
A_inf = np.dot(np.linalg.inv(np.dot(Z_final.T,Z_final) +  ((sigmaX_final/sigmaA_final)**2)*np.identity(Kplus_final)),np.dot(Z_final.T,X))

# A_inf[3,:].reshape(6,6)
# subplot(1,4,1); imagesc(reshape(A_inf(1,:),6,6)); colormap(gray); axis off
# subplot(1,4,2); imagesc(reshape(A_inf(2,:),6,6)); colormap(gray); axis off
# subplot(1,4,3); imagesc(reshape(A_inf(3,:),6,6)); colormap(gray); axis off
# subplot(1,4,4); imagesc(reshape(A_inf(4,:),6,6)); colormap(gray); axis off

# print "Example image:\n",A_inf[0,:].reshape(6,6)
fig = plt.figure(figsize=(15,3))
fig1 = fig.add_subplot(151)
fig1.pcolormesh(A_inf[0,:].reshape(6,6),cmap=plt.cm.gray)
fig2 = fig.add_subplot(152)
fig2.pcolormesh(A_inf[1,:].reshape(6,6),cmap=plt.cm.gray)
fig3 = fig.add_subplot(153)
fig3.pcolormesh(A_inf[2,:].reshape(6,6),cmap=plt.cm.gray)
fig4 = fig.add_subplot(154)
fig4.pcolormesh(A_inf[3,:].reshape(6,6),cmap=plt.cm.gray)
fig5 = fig.add_subplot(155)
fig5.pcolormesh(A_inf[4,:].reshape(6,6),cmap=plt.cm.gray)
fig.savefig("IBP_image_results.png")
plt.close()

##### Save the profiling results
step1 = np.mean(elapsed1_arr)
step2 = np.mean(elapsed2_arr)
step3 = np.mean(elapsed1k_count_arr)
step4 = np.mean(elapsed1N_count_arr)
step5 = np.mean(elapsed1k_calc_arr)
step6 = np.mean(elapsed1k_init_arr)
first = np.array((step1,step2,step3,step4,step5,step6)).reshape(6,1)
second = np.array((1,1,100,100,500,500)).reshape(6,1)
third = first*second

columns = ['Time (seconds)/action','Times performed','Total time (seconds)']
index = ['Generating Z given alpha','Sampling sigmaX, sigmaA','Sampling from K+',
         'Sampling new dishes','Calculation','Initialization']

df = pd.DataFrame(np.hstack((first,second,third)),columns=columns,index=index)
tab = df.to_latex()
text_file = open("Table_naive.tex", "w")
text_file.write(tab)
text_file.close()