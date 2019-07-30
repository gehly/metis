import numpy as np
from math import floor



def interp_lagrange(X, Y, xx, p):
    '''
    This function interpolates data using Lagrange method of order P
    
    Parameters
    ------
    X : 1D numpy array
        x-values of data to interpolate
    Y : 2D numpy array
        y-values of data to interpolate
    xx : float
        single x value to interpolate at
    p : int
        order of interpolation
    
    Returns
    ------
    yy : 1D numpy array
        interpolated y-value(s)
        
    References
    ------
    [1] Kharab, A., An Introduction to Numerical Methods: A MATLAB 
        Approach, 2nd ed., 2005.
            
    '''
    
    # Number of data points to use for interpolation (e.g. 8,9,10...)
    N = p + 1

    if (len(X) < N):
        print('Not enough data points for desired Lagrange interpolation!')
        
    # Compute number of elements on either side of middle element to grab
    No2 = 0.5*N
    nn  = int(floor(No2))
    
    # Find index such that X[row0] < xx < X[row0+1]
    row0 = list(np.where(X < xx)[0])[-1]
    
    # Trim data set
    # N is even (p is odd)    
    if (No2-nn == 0): 
        
        # adjust row0 in case near data set endpoints
        if (N == len(X)) or (row0 < nn-1):
            row0 = nn-1
        elif (row0 > (len(X)-nn)):  # (row0 == length(X))            
            row0 = len(X) - nn - 1        
    
        # Trim to relevant data points
        X = X[row0-nn+1 : row0+nn+1]
        Y = Y[row0-nn+1 : row0+nn+1, :]


    # N is odd (p is even)
    else:
    
        # adjust row0 in case near data set endpoints
        if (N == len(X)) or (row0 < nn):
            row0 = nn
        elif (row0 > len(X)-nn):
            row0 = len(X) - nn - 1
        else:
            if (xx-X(row0) > 0.5) and (row0+1+nn < len(X)):
                row0 = row0 + 1
    
        # Trim to relevant data points
        X = X[row0-nn:row0+nn+1]
        Y = Y[row0-nn:row0+nn+1, :]
        
    # Compute coefficients
    Pj = np.ones((1,N))
    
    for jj in range(N):
        for ii in range(N):
            
            if jj != ii:
                Pj[0, jj] = Pj[0, jj] * (-xx+X[ii])/(-X[jj]+X[ii])
    
    
    yy = np.dot(Pj, Y)
    
    return yy


