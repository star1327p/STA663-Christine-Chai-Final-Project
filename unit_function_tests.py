from __future__ import division
import numpy as np
import numpy.testing as npt
import scipy.stats as stats

# Test the calcM function
def calcM(Z,Kplus,sigmaX,sigmaA):
    """Save the matrix M so we won't need to calculate it again and again"""
    return np.linalg.inv(np.dot(Z[:,0:Kplus].T,Z[:,0:Kplus])+((sigmaX/sigmaA)**2)*np.identity(Kplus))

# Test the calcInverse_org and calcInverse functions
def calcInverse_orig(Z, M, i, k, val):
    """Effective inverse calculation from Griffiths and Ghahramani (2005; Equations 51 to 54)
    M_(-i) = inv(inv(M) - zi.T * zi)"""
    M_i = M - np.dot(np.dot(M,Z[i,:].T),np.dot(Z[i,:],M))/(np.dot(np.dot(Z[i,:],M),Z[i,:].T)-1)
    Z[i,k] = val
    M = M_i - np.dot(np.dot(M_i,Z[i,:].T),np.dot(Z[i,:],M_i))/(np.dot(np.dot(Z[i,:],M_i),Z[i,:].T)+1)
    Inv = M
    return Inv

def calcInverse(Z, M, i, k, val):
    """New version to check: M_(-i) = inv(inv(M) - zi.T * zi) and M = inv(inv(M_(-i)) + zi.T * zi)"""
    M_i = np.linalg.inv(np.linalg.inv(M) - np.dot(Z[i,:].T,Z[i,:]))
    Z[i,k] = val
    M = np.linalg.inv(np.linalg.inv(M_i) + np.dot(Z[i,:].T,Z[i,:]))
    return M

np.random.seed(1234)

N = 100
sigmaX = 0.9
sigmaA = 0.7
Kplus = 5
Z = np.zeros((N,1000))
alpha = 20
ind = 0
while ind < N:
    t = stats.poisson.rvs(alpha)
    if t > 0:
        Z[ind,0:t] = 1
        ind += 1
        
Z = Z[:,0:Kplus]
        
# Test the calcM function
M = calcM(Z,Kplus,sigmaX,sigmaA)
invM = np.dot(Z[:,0:Kplus].T,Z[:,0:Kplus])+((sigmaX/sigmaA)**2)*np.identity(Kplus)

npt.assert_almost_equal(np.dot(M,invM),np.identity(Kplus))
print "The function calcM passed the assertion test!\n"

# Test the calcInverse functions
# print Z[17,:]
# print Z.shape
(i,k,val) = (17,0,0)
# print i
M1 = calcInverse_orig(Z, M, i, k, val)
print "The computed matrix M1 using calInverse_orig:\n", M1,"\n"

M2 = calcInverse(Z, M, i, k, val)
print "The computed matrix M2 using calInverse:\n", M2,"\n"

Z[i,k] = val
newM = calcM(Z,Kplus,sigmaX,sigmaA)
print "The correct matrix M:\n",newM,"\n"

print "Does M == M1?",np.allclose(M1,newM)
print "Does M == M2?",np.allclose(M2,newM)
print "The calcInverse functions did not pass the assertion test!"