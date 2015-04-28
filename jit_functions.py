from numba import jit
import numpy as np

@jit
def calcM(Z,Kplus,sigmaX,sigmaA):
    """Save the matrix M so we won't need to calculate it again and again"""
    return np.linalg.inv(np.dot(Z[:,0:Kplus].T,Z[:,0:Kplus])+((sigmaX/sigmaA)**2)*np.identity(Kplus))

def calcInverse_orig(Z, M, i, k, val):
    """Effective inverse calculation from Griffiths and Ghahramani (2005; Equations 51 to 54)
    M_(-i) = inv(inv(M) - zi.T * zi)"""
    M_i = M - np.dot(np.dot(M,Z[i,:].T),np.dot(Z[i,:],M))/(np.dot(np.dot(Z[i,:],M),Z[i,:].T)-1)
    Z[i,k] = val
    M = M_i - np.dot(np.dot(M_i,Z[i,:].T),np.dot(Z[i,:],M_i))/(np.dot(np.dot(Z[i,:],M_i),Z[i,:].T)+1)
    Inv = M
    return Inv

def calcInverse(Z, M, i, k, val):
    """New version to check: M_(-i) = inv(inv(M) - zi.T * zi)"""
    Z[i,k] = val
    return np.linalg.inv(np.linalg.inv(M) - np.dot(Z[i,:].T,Z[i,:]))

def log_likelihood(X,Z,M,sigmaA,sigmaX,Kplus,N,D):  
    """Calculate the log-likelihood: P(X|Z,M,sigmaA,sigmaX,Kplus,N,D)"""  
    determinant = np.linalg.det(np.dot(Z.T,Z)+((sigmaX/sigmaA)**2)*np.identity(Kplus))
    constant = N*D*0.5*np.log(2*np.pi) + (N-Kplus)*D*np.log(sigmaX) + Kplus*D*np.log(sigmaA) + D*0.5*np.log(determinant)
    
    middle = np.identity(N) - np.dot(np.dot(Z, M),Z.T)
    trace = np.trace(np.dot(np.dot(X.T,middle),X))
    kernel = -0.5*np.reciprocal(sigmaX**2)*trace
    
    log_lik = -constant + kernel
    return log_lik