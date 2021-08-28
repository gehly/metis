import numpy as np
from math import *

###############################################################################
# This file contains generic functions useful in filtering.
# Functions:
#  compute_gaussian
#  split_GMM
#  split_gaussian_library
#
###############################################################################


def compute_gaussian(x, m, P):
    '''
    This function computes the likelihood of the multivariate gaussian pdf
    for a given state x, assuming mean m and covariance P.

    Parameters
    ------
    x : nx1 numpy array
        instance of a random vector
    m : nx1 numpy array
        mean
    P : nxn numpy array
        covariance

    Returns
    ------
    pg : float
        multivariate gaussian likelihood
    '''

    K1 = np.sqrt(np.linalg.det(2*pi*(P)))
    K2 = np.exp(-0.5 * np.dot((x-m).T, np.dot(np.linalg.inv(P), (x-m))))
    pg = (1/K1) * K2
    pg = float(pg)

    return pg
    
    
def compute_gaussian_prob(x, w, m, P, N=100000):
    '''
    This function computes the probability of a point lying closer to a mean
    vector in a GMM than for a given state x.
    Uses MC integration (uniform sampling)

    Parameters
    ------
    x : nx1 numpy array
        instance of a random vector
    w : list
        weights
    m : list
        nx1 numpy array, means
    P : list
        nxn numpy array, covars
    N : int, optional
        Number of samples to draw (default=1e5)

    Returns
    ------
    prob : float
        probability of getting a point closer to a mean vector in the GMM
    '''

    
    
    # Draw N samples from the PDF
    prob = 1.
    for j in xrange(len(w)):
        wj = w[j]
        mj = m[j]
        Pj = P[j]
        
        invPj = np.linalg.inv(Pj)
        
        mc_points = np.random.multivariate_normal(mj.flatten(), Pj, N) #int(wj*N))

#        if j == 0:
#            mc_points = mcj
#        else:
#            mc_points = np.concatenate((mc_points, mcj))
    
        # Loop and check
        Nin = 0.
        #Ntot = int(mc_points.shape[0])
        for ii in xrange(N):
            mci = np.reshape(mc_points[ii,:],(2,1))
        
            
            # Association distances
            dx2 = np.dot((x-mj).T, np.dot(invPj, (x-mj)))
            di2 = np.dot((mci-mj).T, np.dot(invPj, (mci-mj)))
#            print dx2
#            print di2
#            print x
#            print mj
#            print mci
            if di2 < dx2:            
                Nin += 1.
        
        # Compute probability of getting a point closer to a mean vector in the GMM
        prob *= float(Nin)/N

    return prob
    

def cholesky_inv(P):
    '''
    This function computes a matrix inverse using cholesky decomposition.
    Improves numerical stability for positive-definite matrices.

    Parameters
    ------
    P : nxn numpy array
        positive definite matrix

    Returns
    ------
    invP : nxn numpy array
        inverse of input P matrix
    '''
    
    cholP = np.linalg.inv(np.linalg.cholesky(P))
    invP = np.dot(cholP.T, cholP)
    
    return invP


def split_GMM(GMM0, N=3):
    '''
    This function splits a single gaussian PDF into multiple components.
    For a multivariate PDF, it will split along the axis corresponding to the
    largest eigenvalue (greatest uncertainty).
    The function splits along only one axis.

    Parameters
    ------
    GMM0 : list
        GMM weight, mean, covar
    N : int, optional
        number of components to split into (3, 4, or 5, default=3)

    Returns
    ------
    GMM : list
        GMM weights, means, covars
    '''

    # Break out input GM component
    w0 = GMM0[0]
    m0 = GMM0[1]
    P0 = GMM0[2]
    n = len(m0)

    # Get splitting library info
    wbar, mbar, sigbar = split_gaussian_library(N)

    # Decompose covariance matrix
    lam, V = np.linalg.eig(P0)

    # Find largest eigenvalue and corresponding eigenvector
    k = np.argmax(lam)
    lamk = lam[k]
    vk = np.reshape(V[:,k], (n,1))

    # Compute updated weights
    w = [w0 * wi for wi in wbar]

    # All sigma values are equal, just use first entry
    lam[k] = lam[k]*sigbar[0]**2
    Lam = np.diag(lam)

    # Compute updated means, covars
    m = []
    P = []
    for i in xrange(0, N):
        mi = m0 + np.sqrt(lamk)*mbar[i]*vk
        Pi = np.dot(V, np.dot(Lam, V.T))
        m.append(mi)
        P.append(Pi)

    GMM = [w, m, P]

    return GMM


def split_gaussian_library(N=3):
    '''
    This function  outputs the splitting library for GM components. All outputs
    are given to split a univariate standard normal distribution
    (m=0, sig = 1).

    Parameters
    ------
    N : int, optional
        number of components to split into (3, 4, or 5, default=3)

    Returns
    ------
    w : list
        component weights
    m : list
        component means (univariate)
    sig : list
        component sigmas (univariate)

    '''

    if N == 3:
        w = [0.2252246249136750, 0.5495507501726501, 0.2252246249136750]
        m = [-1.057515461475881, 0., 1.057515461475881]
        sig = [0.6715662886640760]*3

    elif N == 4:
        w = [0.1238046161618835, 0.3761953838381165,
             0.3761953838381165, 0.1238046161618835]
        m = [-1.437464136328835, -0.455886223973523,
             0.455886223973523, 1.437464136328835]
        sig = [0.5276007226175397]*4
    elif N == 5:
        w = [0.0763216490701042, 0.2474417859474436,
             0.3524731299649044, 0.2474417859474436, 0.0763216490701042]
        m = [-1.689972911128078, -0.800928383429953,
             0., 0.800928383429953, 1.689972911128078]
        sig = [0.4422555386310084]*5

    return w, m, sig
