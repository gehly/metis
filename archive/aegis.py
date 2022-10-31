import numpy as np
from math import *
import os
import copy
from integration_functions import *
import filters as filt
import pickle
from scipy.integrate import odeint
import scipy.linalg as sci
import conversions as conv
import matplotlib.pyplot as plt

#############################################################################
#    This file contains functions to implement the AEGIS Gaussian Mixture
#    algorithm.
#
#    Ref:
#    DeMars, K. "Nonlinear Orbit Uncertainty Prediction and Rectification for
#    Space Situational Awareness," PhD thesis, 2010.
#############################################################################

def aegis_ukf(GMM,obsfile,intfcn,inputs) :
    '''
    aegis_ukf(GMM,obsfile,intfcn,inputs)

    This function implements the AEGIS UKF to estimate an object's state and
    covariance using an adaptive Gaussian Mixtures scheme to approximate
    the state uncertainty.

    Inputs:
    obsfile = filename of pkl file containing measurements
    stationfile = filename of pkl file containing ground station locations
    inputs = dictionary of input parameters
    intfcn = name of integration function
    meas_types = list of measurement types to be used

    Outputs
    Xref_mat = nxL matrix, each column is state vector Xref at each time epoch
    P_mat = Lxnxn matrix, each nxn matrix is covariance at each time epoch
    resids = pxL matrix, each column is prefit residuals at each time epoch   

    '''

    
    #Break out inputs
    n = len(GMM[1][0])

    #Get measurement and other data
    pklFile = open( obsfile, 'rb' )
    data = pickle.load( pklFile )
    measurements = data[1]
    X_truth = data[2]
    sigma_dict = data[4]
    stations = data[5]
    pklFile.close()

    #Retrieve observation times and ground station id's
    t_obs = measurements['obs_time']
    stat_id = measurements['station']

    #Form list of measurements
    meas_types = inputs['meas_types']
    Y = []
    Rvec = []
    for meas in meas_types :
        Y.append(measurements[meas])
        Rvec.append(sigma_dict[meas]**2.)
    inputs['Rk'] = np.diag(Rvec)    

    #Number of epochs and observations per epoch
    L = len(t_obs)
    p = len(meas_types)
    
    #Initialize
    resids = []
    GMM_output = []
    t_prop = []

    #Step 1: Compute Weights
    alpha = 1.
    beta = 2.
    kappa = 3. - n
    lam = alpha**2 * (n + kappa) - n
    gam = np.sqrt(n + lam)

    Wm = 1./(2.*(n + lam)) * np.ones((1,2*n))
    Wm = list(Wm.flatten())
    Wc = copy.copy(Wm)
    Wm.insert(0,lam/(n + lam))
    Wc.insert(0,lam/(n + lam) + (1 - alpha**2 + beta))
    Wm = np.asarray(Wm)
    diagWc = np.diag(Wc)

    #Inputs
    inputs['Wm'] = Wm
    inputs['diagWc'] = diagWc
    inputs['gam'] = gam

    #Begin Kalman Filter
    for i in xrange(0,L) :

        #Initialize for UKF
        if i == 0 :
            t_prior = 0.
        else :
            t_prior = t_obs[i-1]

        GMM_prior = copy.deepcopy(GMM)     

        #Step 2: Read the next observation
        ti = t_obs[i]
        tin = [t_prior,ti]
        delta_t = ti -  t_prior
        stati = stat_id[i]
        stat_ecef = stations[stati]
        Yi = []
        for j in xrange(0,p) :
            Yi.append(Y[j][i])
        Yi = np.reshape(Yi,(p,1))
       
        #Predictor Step
        if ti > t_prior :
            GMM_list,ts_list = aegis_predictor(GMM_prior,tin,inputs,intfcn)
            GMM_bar = GMM_list[-1]
        else :
            GMM_bar = copy.deepcopy(GMM_prior)
            GMM_list = [GMM_bar]
            ts_list = [ti]

        #Corrector Step
        GMM,residsi = aegis_corrector(GMM_bar,ti,Yi,stat_ecef,meas_types,inputs)

        #Output
        resids.append(residsi)
        GMM_list[-1] = GMM
        GMM_output.extend(GMM_list)
        t_prop.extend(ts_list)

        print ti
        print 'corrector output'
        w = GMM[0]
        print 'Num Components = ',len(w)

        if ti > 2000 :
            mistake

    #Reshape
    resids = np.reshape(resids,(L,p))

    return t_obs,t_prop,GMM_output,resids


def aegis_predictor(GMM0,tin,inputs,intfcn) :

    '''
    aegis_predictor(GMM0,tin,inputs,intfcn)

    This function implements the predictor step for the AEGIS UKF.

    Inputs:
    GMM0 = list of GM component weights, means, covariance matrices
         = [w,m,P]
            w = list of weights
            m = list of means (numpy nx1 arrays)
            P = list of covars (numpy nxn arrays)
    tin = list of previous and current time [t_prior,ti]
    inputs = dictionary of input parameters
    intfcn = name of integration function

    Outputs
    GMM_list = list of GMM lists for each time step ts
        GMM  = list of GM component weights, means, covariance matrices
             = [w,m,P]
                w = list of weights
                m = list of means (numpy nx1 arrays)
                P = list of covars (numpy nxn arrays)

    ts_list = list of times for propagation step

    '''

    #Time values
    t_prior = tin[0]
    tk = tin[1]
    delta_k = tk - t_prior

    #Integrator Options
    int_tol = 1.0e-12
    intfcn_entropy = int_twobody_diff_entropy

    #Inputs
    split_T = inputs['split_T']
    delta_s = inputs['delta_s']
    gam = inputs['gam']
    Wm = inputs['Wm']
    diagWc = inputs['diagWc']

    #Number of states, components
    n = len(GMM0[1][0])
    L0 = len(GMM0[0])

    #Initialize GMM, entropy and chi lists
    GMM_list = []
    GMM_list.append(GMM0)
    entropy_list = []
    chi_v_list = []

    for j in xrange(0,L0) :
        
        #Break out GMM
        wj = GMM0[0][j]
        mj = GMM0[1][j]
        Pj = GMM0[2][j]

        #Compute entropy
        ej = compute_entropy(Pj)
        entropy_list.append(ej)
        
        #Get sigma points
        sqP = np.linalg.cholesky(Pj)
        Xrep = np.tile(mj,(1,n))
        chi = np.concatenate((mj, Xrep+(gam*sqP), Xrep-(gam*sqP)),axis=1)
        chi_v = np.reshape(chi,(n*(2*n+1),1),order='F')
        chi_v_list.append(chi_v)        

    #Run through time in small increments
    ts_prior = t_prior
    ts = t_prior
    ts_list = []
    
    while ts < tk :

        ts = ts_prior + delta_s
        if ts < tk :
            tin_s = [ts_prior,ts]
            ts_list.append(ts)
        else :
            tin_s = [ts_prior,tk]
            ts_list.append(tk)

        #Current GMM
        GMM = GMM_list[-1]

        #Current weights and number of components
        ws_list = copy.copy(GMM[0])
        L = len(ws_list)

        #Initialize mean and covar list to current size but dummy value
        ms_list = [0.]*L
        Ps_list = [0.]*L

        print 'current time step',tin_s
        print 'L=',L

        #Run through components one at a time
        for j in xrange(0,L) :     
                                    
            #Step 3: Propagate to ts
            #3A. Propagate entropy linearly
            ej_linear = entropy_list[j]
            mj = GMM[1][j]
            int0 = np.append(ej_linear,mj)
            intout = odeint(intfcn_entropy,int0,tin_s,args=(inputs,),rtol=int_tol,atol=int_tol)
            ej_linear = intout[-1,0]
            entropy_list[j] = ej_linear
    
            #3B. Propagate sigma points
            chi_vj = chi_v_list[j]
            int0 = chi_vj.flatten()
            intout = odeint(intfcn,int0,tin_s,args=(inputs,),rtol=int_tol,atol=int_tol)

            #Extract values for later calculations
            chi_v = intout[-1,:]
            chi = np.reshape(chi_v,(n,2*n+1),order='F')

            #Step 4: Compute Xbar, Pbar - No Process Noise
            Xbar = np.dot(chi,Wm.T)
            Xbar = np.reshape(Xbar,(n,1))        
            chi_diff = chi - np.dot(Xbar, np.ones((1,(2*n+1))))
            Pbar = np.dot(chi_diff, np.dot(diagWc,chi_diff.T))
                      
            #Step 5: Check for nonlinearity
            ej_nonlin = compute_entropy(Pbar)

##            print 'Comp = ',j
##            print ej_linear
##            print ej_nonlin
##            print mj
##            print Xbar
##            print Pj
##            print Pbar
##            mistake
            
            #If true split
            if abs(ej_nonlin - ej_linear) > split_T :
                wj = ws_list[j]
                GMM_in = [wj,Xbar,Pbar]
                GMM_out = split_GMM(GMM_in,3)

                split_w = GMM_out[0]
                split_m = GMM_out[1]
                split_P = GMM_out[2]

                #Replace current component and add others
                for k in xrange(0,len(split_w)) :

                    #Compute weights, entropy and sigma points
                    wk = split_w[k]   #Note: split_GMM function multiplies by wj, no need to repeat here
                    mk = split_m[k]
                    Pk = split_P[k]
                    ek = compute_entropy(Pk)

                    sqP = np.linalg.cholesky(Pk)
                    Xrep = np.tile(mk,(1,n))
                    chi = np.concatenate((mk, Xrep+(gam*sqP), Xrep-(gam*sqP)),axis=1)
                    chi_v = np.reshape(chi,(n*(2*n+1),1),order='F')
                   
                    if k == 0 :
                        ws_list[j] = wk
                        ms_list[j] = mk
                        Ps_list[j] = Pk
                        chi_v_list[j] = chi_v
                        entropy_list[j] = ek
                    else :
                        ws_list.append(wk)
                        ms_list.append(mk)
                        Ps_list.append(Pk)
                        chi_v_list.append(chi_v)
                        entropy_list.append(ek)

            else :
                #Save current components
                ms_list[j] = Xbar
                Ps_list[j] = Pbar
                chi_v_list[j] = chi_v
                
        #Increment time and save current GMM_list
        ts_prior = ts
        GMM_s = [ws_list,ms_list,Ps_list]
        GMM_list.append(GMM_s)

    #After reaching tk
    #Step 6: Compute final Pbar for each component
    GMM_f = GMM_list[-1]
    L = len(GMM_f[0])

    #print 'Lbar = ',L

    for j in xrange(0,L) :
    
        Pbar = GMM_f[2][j]
        
        #Process noise
        if delta_k <= 1000. :
            Qric = delta_k * inputs['Qric']
            Q1 = Qric[0:3,0:3]
            Q2 = Qric[3:6,3:6]
            rvec = GMM_f[1][j][0:3]
            vvec = GMM_f[1][j][3:6]
            Q = np.zeros((n,n))
            Q1 = conv.ric2eci(rvec,vvec,Q1)
            Q2 = conv.ric2eci(rvec,vvec,Q2)
            Q[0:3,0:3] = Q1
            Q[3:6,3:6] = Q2
            Pbar = Pbar + Q

        #Re-symmetric Positive Definite
        Pbar = 0.5 * (Pbar + Pbar.T)

        GMM_f[2][j] = Pbar

    #Final output - remove first entry
    GMM_list[-1] = GMM_f
    del GMM_list[0]
            
    return GMM_list, ts_list


def aegis_corrector(GMM0,ti,Yi,stat_ecef,meas_types,inputs) :
    '''
    aegis_corrector(GMM0,ti,Yi,stat_ecef,meas_types,inputs)

    This function implements the measurement update for the AEGIS UKF.

    Inputs:
    GMM0 = list of GM component weights, means, covariance matrices
         = [w,m,P]
            w = list of weights
            m = list of means (numpy nx1 arrays)
            P = list of covars (numpy nxn arrays)
    ti = current time
    Yi = px1 array of current measurements
    stat_ecef = 1x3 ground station location in ECEF [km]
    meas_types = list of measurement types to be used
    inputs = dictionary of input parameters
    intfcn = name of integration function

    Outputs:
    GMM = list of GM component weights, means, covariance matrices
         = [w,m,P]
            w = list of weights
            m = list of means (numpy nx1 arrays)
            P = list of covars (numpy nxn arrays)
    

    '''

    #Inputs
    gam = inputs['gam']
    Wm = inputs['Wm']
    diagWc = inputs['diagWc']
    Rk = inputs['Rk']
    theta = inputs['theta0'] + inputs['dtheta']*ti

    #Break out GMM
    w0 = GMM0[0]
    m0 = GMM0[1]
    P0 = GMM0[2]
    n = len(m0[0])

    #Loop over all components
    beta_list = []
    mf_list = []
    Pf_list = []
    L = len(w0)
    for j in xrange(0,L) :

        #Component weight, mean, covar
        wj = w0[j]
        mj = m0[j]
        Pj = P0[j]

        #Step 6: Recompute sigma points -
        #Includes process noise from predictor step
        sqP = np.linalg.cholesky(Pj)
        Xrep = np.tile(mj,(1,n))
        chi = np.concatenate((mj, Xrep+(gam*sqP), Xrep-(gam*sqP)),axis=1)
        chi_diff = chi - np.dot(mj, np.ones((1,(2*n+1))))

        #Step 7: Computed Measurements
        Gi = filt.H_fcn_ukf(chi,stat_ecef,meas_types,inputs,ti)
        ybar = np.dot(Gi, Wm.T)

        #Step 8: Innovation and Cross-Correlation
        p = len(meas_types)
        ybar = np.reshape(ybar,(p,1))
        y_diff = Gi - np.dot(ybar, np.ones((1,2*n+1)))
        Pyy = Rk + np.dot(y_diff, np.dot(diagWc,y_diff.T))
        Pxy = np.dot(chi_diff, np.dot(diagWc,y_diff.T))

        #Step 9: Measurement Update
        K = np.dot(Pxy,np.linalg.inv(Pyy))
        mf = mj + np.dot(K,Yi-ybar)       
        Pf = Pj - np.dot(K, np.dot(Pyy,K.T))

        beta = compute_gaussian(Yi,ybar,Pyy)
        assoc_dist = np.dot((Yi-ybar).T, np.dot(np.linalg.inv(Pyy),(Yi-ybar)))
        beta_list.append(beta)

        print 'Corrector'
        print 'ybar',ybar
        print 'Yi',Yi
        print 'resid',Yi-ybar
        print 'Pyy',Pyy

        #Re-symmetric pos def
        Pf = 0.5 * (Pf + Pf.T)

        mf_list.append(mf)
        Pf_list.append(Pf)

    #Normalize updated weights
    denom = np.dot(beta_list,w0)
    wf = [a1*a2/denom for a1,a2 in zip(w0,beta_list)]

    #Form list, merge and prune components
    GMM = [wf,mf_list,Pf_list]   
    GMM = merge_GMM(GMM,inputs)

    #Compute post-fit residuals - merge all components to get mean
    inputs2 = {}
    inputs2['prune_T'] = 0.
    inputs2['merge_U'] = 1e6
    GMM_resid = merge_GMM(GMM,inputs2)
    m_resid = GMM_resid[1][0]
    H_til,ybar = filt.H_fcn(m_resid,stat_ecef,meas_types,inputs,ti)
    resids = Yi - ybar

    return GMM, resids


def compute_gaussian(x,m,P) :
    '''
    compute_gaussian(x,m,P)

    This function computes the value of the multivariate gaussian pdf
    for a random vector x, assuming mean m and covariance P.  

    Inputs:
    x = nx1 array, instance of a random vector
    m = nx1 array, mean
    P = nxn array, covariance

    Outputs:
    pg = multivariate gaussian pdf value    

    '''

    K1 = np.sqrt(np.linalg.det(2*pi*(P)))
    K2 = np.exp(-0.5 * np.dot((x-m).T, np.dot(np.linalg.inv(P),(x-m))))
    pg = (1/K1) * K2
    pg = float(pg)

    return pg


def compute_NL2(GMM1,GMM2) :
    '''
    compute_NL2(GMM1,GMM2)
    
    This function computes the normalized L2 distance between two PDFs. Assumed
    form of inputs is a Gaussian Mixture Model (GMM) including multiple
    components but function works the same comparing individual Gaussian PDFs.

    Inputs:
    GMM# = list of GM component weights, means, covariance matrices
         = [w,m,P]
            w = list of weights
            m = list of means (numpy nx1 arrays)
            P = list of covars (numpy nxn arrays)

    Outputs:
    NL2 = scalar NL2 distance between the PDFs
          [0 = same, 1 = completely diff]

    '''

    #Compute d12, d11, d22
    d12 = compute_dxx(GMM1,GMM2)
    d11 = compute_dxx(GMM1,GMM1)
    d22 = compute_dxx(GMM2,GMM2)

    #Compute NL2
    NL2 = 1 - (2*d12)/(d11 + d22)  

    return NL2


def compute_dxx(GMM1,GMM2) :

    '''
    compute_dxx(GMM1,GMM2)
    
    This function computes the distance parameter d for two PDFs. Assumed
    form of inputs is a Gaussian Mixture Model (GMM) including multiple
    components but function works the same comparing individual Gaussian PDFs.

    Inputs:
    GMM# = list of GM component weights, means, covariance matrices
         = [w,m,P]
            w = list of weights
            m = list of means (numpy nx1 arrays)
            P = list of covars (numpy nxn arrays)

    Outputs:
    dxx = scalar distance between the PDFs.

    '''

    #Break out input pdf parameters
    w1 = GMM1[0]
    m1 = GMM1[1]
    P1 = GMM1[2]    

    w2 = GMM2[0]
    m2 = GMM2[1]
    P2 = GMM2[2]

    #Get number of components
    k1 = len(w1)
    k2 = len(w2)

    #Compute dxx
    dxx = 0.
    for i in xrange(0,k1) :
        wi = w1[i]
        mi = m1[i]
        Pi = P1[i]
        for j in xrange(0,k2) :
            wj = w2[j]
            mj = m2[j]
            Pj = P2[j]

            K1 = np.sqrt(np.linalg.det(2*pi*(Pi+Pj)))
            K2 = np.exp(-0.5 * np.dot((mi-mj).T, np.dot(np.linalg.inv(Pi+Pj),(mi-mj))))
            K = (1/K1) * K2
            
            dxx += wi*wj*K 

    return float(dxx)


def compute_entropy(P) :
    '''

    This function computes the entropy of a given PDF

    '''

    if np.linalg.det(2*pi*e*P) < 0. :
        print np.linalg.det(2*pi*e*P)
        print np.linalg.eig(P)
        P2 = sci.sqrtm(P)
        P3 = np.real(np.dot(P2,P2.T))
        print np.linalg.eig(P3)
        print P3 - P

    #Differential Entropy
    H = 0.5 * log(np.linalg.det(2*pi*e*P))

    #Renyi Entropy
    # R = 0.5 * log(np.linalg.det(2*pi*(kappa**(1/(1-kappa)))*P))

    entropy = H

    return entropy


def compute_likelihood(m1,GMM) :
    '''
    compute_likelihood(m1,GMM)

    This function computes the agreement between a set of points and a PDF
    described by a GMM.  The agreement measure is a likelihood function.

    Inputs:
    m1 = list of (nx1) arrays representing the locations of the sample points
    GMM = list of GM component weights, means, covariance matrices
        = [w,m,P]
            w = list of weights
            m = list of means (numpy nx1 arrays)
            P = list of covars (numpy nxn arrays)            

    Outputs:
    L = likelihood of agreement
    
    '''

    #Break out GMM
    w = GMM[0]
    m = GMM[1]
    P = GMM[2]
    n = len(m[0])

    #Number of MC points and GM components
    k1 = len(m1)
    wi = 1./k1
    k2 = len(w)

    #Compute L
    L = 0.
    for i in xrange(0,k1) :
        #Note DeMars has m1,j Eq. 2.19
        mi = np.reshape(m1[i],(n,1))
        
        for j in xrange(0,k2) :
            wj = w[j]
            mj = m[j]
            Pj = P[j]

            K1 = np.sqrt(np.linalg.det(2*pi*(Pj)))
            K2 = np.exp(-0.5 * np.dot((mi-mj).T, np.dot(np.linalg.inv(Pj),(mi-mj))))
            pg = (1/K1) * float(K2)
            
            L += wi*wj*pg
            
    return L


def split_gaussian_library(N=3) :
    '''
    split_gaussian_library(N=3)
    
    This function  outputs the splitting library for GM components. All outputs
    are given to split a univariate standard normal distribution (m=0, sig = 1).
    
    Inputs:
    (optional) N = number of components to split into (3, 4, or 5)
                 = 3 (default)

    Outputs:
    w = list of component weights
    m = list of component means (univariate)
    sig = list of component sigmas (univariate)

    '''

    if N == 3 :
        w = [0.2252246249136750, 0.5495507501726501, 0.2252246249136750]
        m = [-1.057515461475881, 0., 1.057515461475881]
        sig = [0.6715662886640760]*3

    elif N == 4 :
        w = [0.1238046161618835, 0.3761953838381165, 0.3761953838381165, 0.1238046161618835]
        m = [-1.437464136328835, -0.455886223973523, 0.455886223973523, 1.437464136328835]
        sig = [0.5276007226175397]*4
    elif N == 5 :
        w = [0.0763216490701042, 0.2474417859474436, 0.3524731299649044, 0.2474417859474436, 0.0763216490701042]
        m = [-1.689972911128078, -0.800928383429953, 0., 0.800928383429953, 1.689972911128078]
        sig = [0.4422555386310084]*5

    return w,m,sig


def split_GMM(GMM0,N=3) :
    '''
    split_GMM(GMM0,N=3)
    
    This function splits a single gaussian PDF into multiple components.
    For a multivariate PDF, it will split along the axis corresponding to the
    largest eigenvalue (greatest uncertainty). The function splits along only one
    axis.

    Inputs:
    GMM0 = list of PDF weight, mean, covar
         = [w,m,P]
            w = weight, scalar
            m = mean, numpy nx1 array
            P = covar, numpy nxn array

    N = number of components to split into (3,4,or 5)

    Outputs:
    GMM = list of GM component weight, mean, covariance matrix
         = [w,m,P]
            w = single weight (float)
            m = single mean (numpy nx1 array)
            P = single covar (numpy nxn array)
    '''

    #Break out input GM component
    w0 = GMM0[0]
    m0 = GMM0[1]
    P0 = GMM0[2]
    n = len(m0)

##    print w0
##    print m0
##    print P0

    #Get splitting library info
    wbar,mbar,sigbar = split_gaussian_library(N)

    #Decompose covariance matrix
    lam,V = np.linalg.eig(P0)

    #Find largest eigenvalue and corresponding eigenvector
    k = np.argmax(lam)
    lamk = lam[k]
    vk = np.reshape(V[:,k],(n,1))

    #print 'lam', lam
    #print 'V',V
    #print 'vk',vk

    #Compute updated weights
    w = [w0 * wi for wi in wbar]    

    #All sigma values are equal, just use first entry
    lam[k] = lam[k]*sigbar[0]**2
    Lam = np.diag(lam)

    #print 'lamk',lamk
    #print 'lam',lam

    #Compute updated means, covars
    m = []
    P = []
    for i in xrange(0,N) :
        mi = m0 + np.sqrt(lamk)*mbar[i]*vk
        Pi = np.dot(V, np.dot(Lam,V.T))
        m.append(mi)
        P.append(Pi)
    
    GMM = [w,m,P]


    return GMM



def merge_GMM(GMM0,inputs) :
    '''
    merge_GMM(GMM0,inputs)
    
    This function examines a GMM containing multiple components. It removes
    components with weights below a given threshold, and merges components that
    are close together (small NL2 distance).
    
    Inputs:
    GMM0 = list of PDF weights, means, covars
         = [w,m,P]
            w = list of weights
            m = list of means (numpy nx1 arrays)
            P = list of covars (numpy nxn arrays)

    inputs = dictionary of input parameters (includes thresholds)

    Outputs:
    GMM = list of GM component weights, means, covariance matrices
         = [w,m,P]
            w = list of weights
            m = list of means (numpy nx1 arrays)
            P = list of covars (numpy nxn arrays)

    Ref:
    1. DeMars Thesis
    2. Vo and Ma, "The Gaussian Mixture Probability Hypothesis Density
    Filter," 2006.
    
    '''


    #Break out inputs
    T = inputs['prune_T']
    U = inputs['merge_U']

    #Get GM Components
    w0 = GMM0[0]
    m0 = GMM0[1]
    P0 = GMM0[2]
    wmax = max(w0)

    print w0

    #Remove GM components whose weight is lower than threshold T*wmax    
    w = []
    m = []
    P = []
    n = len(m0[0])
    #Find indices where weight exceeds threshold and keep the component
    I = [i for i,v in enumerate(w0) if v > T*wmax]
    for i in I :
        w.append(w0[i])
        m.append(m0[i])
        P.append(P0[i])

    I = np.arange(0,len(w))

    #Loop to merge components that are close
    count = 0
    wf = []
    mf = []
    Pf = []
    while len(I) != 0 :
        j = np.argmax(w)
        GMMj = [[w[j]],[m[j]],[P[j]]]
        L = []

        #Loop over components to see if they are close to j
        #Note, at least one will be when i == j
        wi = []
        msum = np.zeros((n,1))
        for i in xrange(0,len(w)) :
            Pi = P[i]
            invP = np.linalg.inv(Pi)
            prod = np.dot((m[i] - m[j]).T, np.dot(invP,(m[i] - m[j])))
            if prod <= U :
                L.append(i)
                wi.append(w[i])
                msum = msum + w[i] * m[i]

##            GMMi = [[w[i]],[m[i]],[P[i]]]
##            NL2 = compute_NL2(GMMi,GMMj)          
##
##            if NL2 <= U :
##                L.append(i)
##                wi.append(w[i])
##                msum = msum + w[i] * m[i]
                

        #Compute final w,m,P
        wf.append(sum(wi))
        mfi = 1./wf[-1] * msum
        mf.append(mfi)

        sum_cov = 0
        for i in xrange(0,len(L)):
            sum_cov = sum_cov + w[L[i]]*(P[L[i]] + (mf[count] - m[L[i]]) \
                                          *(mf[count] - m[L[i]]).T)

        Pfi = 1./wf[-1] * sum_cov
        Pf.append(Pfi)

        #Reduce w,m,P
        I = list(set(I).difference(set(L)))
        wd = []
        md = []
        Pd = []
        for i in I :
            wd.append(w[i])
            md.append(m[i])
            Pd.append(P[i])

        w = wd
        m = md
        P = Pd

        #Reset I        
        I = np.arange(0,len(w))
        
        #Increment counter
        count += 1

    #Normalize weights
    wf = list(np.asarray(wf)/sum(wf))

    #Form Output
    GMM = [wf,mf,Pf]
    

    return GMM



def plot_pdf_contours(GMM,mc_points) :

    #Break out GMM
    w = GMM[0]
    m = GMM[1]
    P = GMM[2]
    L = len(w)
    n = len(m[0])

    print 'Num Comp',L
    print m
    print P[0]


##    #Set up max/min values
##    xmin = 1e8
##    xmax = -1e8
##    ymin = 1e8
##    ymax = -1e8
##
##    for j in xrange(0,L) :
##        mj = m[j]
##        Pj = P[j]
##        x = mj[0]
##        y = mj[1]
##        sigx = np.sqrt(Pj[0][0])
##        sigy = np.sqrt(Pj[1][1])
##
##        xmin_j = x - 3*sigx
##        xmax_j = x + 3*sigx
##        ymin_j = y - 3*sigy
##        ymax_j = y + 3*sigy
##
##        if xmin_j < xmin :
##            xmin = xmin_j
##        if xmax_j > xmax :
##            xmax = xmax_j
##        if ymin_j < ymin :
##            ymin = ymin_j
##        if ymax_j > ymax :
##            ymax = ymax_j

    N = len(mc_points)
    mc_mean = np.mean(mc_points,axis=0)
    mc_stdx = np.sqrt(1./N*sum((mc_points[:,0]-mc_mean[0])**2.))
    mc_stdy = np.sqrt(1./N*sum((mc_points[:,1]-mc_mean[1])**2.))

    #Histogram data from mc points
    H,xedges,yedges = np.histogram2d(mc_points[:,0],mc_points[:,1])
    print 'xedges',xedges
    print 'yedges',yedges

##    xmax = mc_mean[0] + 3.*mc_stdx
##    xmin = mc_mean[0] - 3.*mc_stdx
##    ymax = mc_mean[1] + 3.*mc_stdy
##    ymin = mc_mean[1] - 3.*mc_stdy

    xmax = max(xedges)
    xmin = min(xedges)
    ymax = max(yedges)
    ymin = min(yedges)

    print xmax
    print xmin
    print ymax
    print ymin
        
    #Create arrays
    x_vect = np.linspace(xmin,xmax,10)
    y_vect = np.linspace(ymin,ymax,10)
    pdf_array = np.zeros((len(y_vect),len(x_vect)))

    #Compute pdf value for different x,y
    i = -1
    for xi in x_vect :
        i += 1
        k = -1
        for yk in y_vect :
            k += 1
            p = 0.
            for j in xrange(0,L) :
                wj = w[j]
                mj = m[j]
                Pj = P[j]

                state = mj.copy()
                state[0] = xi
                state[1] = yk           

                pg = compute_gaussian(state,mj,Pj)
               
                p += wj*pg
                

            #Store value
            pdf_array[k,i] = p

    pdfmax = np.max(pdf_array)
    pdfmin = np.min(pdf_array)

    #Compute values to set level curves at
    V = []
    for xi in xedges :
        for yk in yedges :
            p = 0.
            for j in xrange(0,L) :
                wj = w[j]
                mj = m[j]
                Pj = P[j]

                state = mj.copy()
                state[0] = xi
                state[1] = yk           

                pg = compute_gaussian(state,mj,Pj)
               
                p += wj*pg

            #Store value
            V.append(p)

    V2 = V[100:-1:2]

    print 'V2',V2
    print 'V',V
    print len(V)
    print pdfmax
    print pdfmin

    V2 = np.linspace(pdfmax/10000.,pdfmax,10)
    print V2




    #Generate mc points from GMM
    mc_points2 = gen_mc_points(GMM,N)

    plt.figure()
    
    #plt.contour(x_vect,y_vect,pdf_array,linewidths=2)

    plt.plot(mc_points[:,0],mc_points[:,1],'.',ms=1)
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])

    plt.figure()
    plt.plot(mc_points2[:,0],mc_points2[:,1],'r.',ms=1)
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    #plt.axis('equal')
    #plt.show()

    #Compute mean and std from data and GMM
    w = np.asarray(w)
    m = np.reshape(m,(L,n)).T

    #print w
    print sum(w)
    #print m
    
    
    GMM_mean = np.dot(m,w.T)
    #mc_points2 = gen_mc_points(GMM,N)
    GMM_stdx = np.sqrt(1./N*sum((mc_points2[:,0]-GMM_mean[0])**2.))
    GMM_stdy = np.sqrt(1./N*sum((mc_points2[:,1]-GMM_mean[1])**2.))

    


    print 'GMM_mean = ',GMM_mean
    print 'GMM_stdx = ',GMM_stdx
    print 'GMM_stdy = ',GMM_stdy
    print 'MC_mean = ',mc_mean
    print 'MC_stdx = ',mc_stdx
    print 'MC_stdy = ',mc_stdy

    return


def gen_mc_points(GMM,N) :

    #Break out GMM
    w = GMM[0]
    m = GMM[1]
    P = GMM[2]
    L = len(w)

    for j in xrange(0,L) :
        wj = w[j]
        mj = m[j]
        Pj = P[j]

        mcj = np.random.multivariate_normal(mj.flatten(),Pj,int(wj*N))

        if j == 0 :
            mc_points = mcj
        else :
            mc_points = np.concatenate((mc_points,mcj))



    return mc_points














