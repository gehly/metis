import numpy as np
from math import *
import matplotlib.pyplot as plt
import copy
import os
import pickle
from scipy.integrate import odeint

import filters.unscented_functions as uf
import utilities.integration_functions as ifunc
import batch

###############################################################################
# Constant Acceleration Test (Ball Dropping)
###############################################################################

def balldrop_setup():
    
    # Define inputs
    acc = 1.  #m/s^2
    inputs = {}
    inputs['acc'] = acc
    inputs['Q'] = np.diag([1e-12,1e-12])
    int_tol = 1e-12
    
    # Time vector
    ti_list = np.arange(0.,100.1,1.)
    
    # Inital State
    X_true = np.array([[0.],[0.]])
    P = np.array([[4., 0.],[0., 1.]])
    pert_vect = np.multiply(np.sqrt(np.diag(P)), np.random.randn(2))
    X_init = X_true + np.reshape(pert_vect, (2, 1))
    
    state_dict = {}
    state_dict[ti_list[0]] = {}
    state_dict[ti_list[0]]['X'] = X_init
    state_dict[ti_list[0]]['P'] = P
    
    # Outlier indices
    percent_outliers = 30.
    n_outliers = int(round(0.01*percent_outliers*len(ti_list)))
    outlier_inds = []
    for ii in xrange(n_outliers):
        loop_pass = False
        while not loop_pass:
            ind = int(floor(np.random.rand()*len(ti_list)))
            if ind not in outlier_inds:
                outlier_inds.append(ind)
                loop_pass = True
    
    print 'outlier inds', outlier_inds    
    
    # Generate truth and measurements 
    truth_dict = {}
    meas_dict = {}
    sig_y = 0.01
    sig_dy = 0.001
    meas_dict['sigma_dict'] = {}
    meas_dict['sigma_dict']['y'] = sig_y
    meas_dict['sigma_dict']['dy'] = sig_dy
    meas_dict['meas_types'] = ['y', 'dy']
    meas_dict['meas'] = {}
    X = copy.copy(X_true)
    for ii in xrange(len(ti_list)):
        
        Xo = copy.copy(X)
        int0 = Xo.flatten()
        
        if ii > 0:
            tin = [ti_list[ii-1], ti_list[ii]]
            intout = odeint(int_balldrop, int0, tin, args=(inputs,),
                            rtol=int_tol, atol=int_tol)
                            
            X = np.reshape(intout[-1,:], (2, 1))
        
        truth_dict[ti_list[ii]] = X
        
        if ii in outlier_inds:
            y_noise = 100.*sig_y*np.random.randn()
        else:
            y_noise = sig_y*np.random.randn()
            
        dy_noise = sig_dy*np.random.randn()
        
        y_meas = float(X[0] + y_noise)
        dy_meas = float(X[1] + dy_noise)
        meas_dict['meas'][ti_list[ii]] = np.array([[y_meas], [dy_meas]])
        
    
    #print state_dict
    #print truth_dict
    #print meas_dict    
    
    # Save Data
    fdir = 'C:\Users\Steve\Documents\\research\lp_norm\lp_ukf\\test\\balldrop'
    
#    fname = 'balldrop_inputs_and_truth.pkl'
#    inputs_file = os.path.join(fdir, fname)
#    pklFile = open(inputs_file, 'wb')
#    pickle.dump([state_dict, inputs, truth_dict], pklFile, -1)
#    pklFile.close()
    
    fname = 'balldrop_meas_' + str(int(n_outliers)).zfill(2) + 'p.pkl'
    meas_file = os.path.join(fdir, fname)
    pklFile = open(meas_file, 'wb')
    pickle.dump([meas_dict], pklFile, -1)
    pklFile.close()
    
    
    return


def execute_balldrop_test():
    
    # Load initial state, inputs
    fdir = 'C:\Users\Steve\Documents\\research\lp_norm\\test\\balldrop'
    fname = 'balldrop_inputs_and_truth.pkl'
    inputs_file = os.path.join(fdir, fname)
    pklFile = open( inputs_file, 'rb' )
    data = pickle.load(pklFile)
    state_dict = data[0]
    inputs = data[1]
    truth_dict = data[2]
    pklFile.close()    
    
    # Load measurements
    percent_outliers = 10.
    ti_list = sorted(truth_dict.keys())
    n_outliers = int(round(0.01*percent_outliers*len(ti_list)))
    fname = 'balldrop_meas_' + str(int(n_outliers)).zfill(2) + 'p.pkl'
    meas_file = os.path.join(fdir, fname)
    pklFile = open( meas_file, 'rb' )
    data = pickle.load(pklFile)
    meas_dict = data[0]
    pklFile.close()
    
    # inputs['Q'] = 1e-8*np.diag([1e-2,1e-4])
    
    # Execute filter
    #intfcn = int_balldrop_stm2
    #meas_fcn = H_balldrop
    intfcn = int_balldrop_ukf
    meas_fcn = unscented_balldrop
    inputs['int_tol'] = 1e-12
    alpha = 1.
    pnorm = 1.5
#    filter_output = uf.lp_ukf(state_dict, meas_dict, inputs, intfcn, meas_fcn,
#                              alpha, pnorm)
#    filter_output = batch.batch(state_dict, meas_dict, inputs, intfcn,
#                                meas_fcn, pnorm)
    filter_output = batch.unscented_batch(state_dict, meas_dict, inputs,
                                          intfcn, meas_fcn, alpha, pnorm)    
    
    # Save results
    fname = 'ubatch_balldrop_output_' + str(pnorm) + 'norm_' + \
        str(int(n_outliers)).zfill(2) + 'p.pkl'
    out_file = os.path.join(fdir, fname)
    pklFile = open(out_file, 'wb')
    pickle.dump([filter_output], pklFile, -1)
    pklFile.close()
    
    return


###############################################################################
# Analysis and Plots
###############################################################################

def balldrop_analysis():
    
    # Load truth
    fdir = 'C:\Users\Steve\Documents\\research\lp_norm\\test\\balldrop'
    fname = 'balldrop_inputs_and_truth.pkl'
    inputs_file = os.path.join(fdir, fname)
    pklFile = open( inputs_file, 'rb' )
    data = pickle.load(pklFile)
    truth_dict = data[2]
    pklFile.close()   
    
    # Load filter results
    pnorm = 1.5
    n_outliers = 10.
    fname = 'ubatch_balldrop_output_' + str(pnorm) + 'norm_' + \
        str(int(n_outliers)).zfill(2) + 'p.pkl'
    out_file = os.path.join(fdir, fname)
    pklFile = open( out_file, 'rb' )
    data = pickle.load(pklFile)
    filter_output = data[0]
    pklFile.close()    
    
    # Compute errors
    ti_list = sorted(filter_output.keys())
    pos_err = np.zeros(len(ti_list))
    vel_err = np.zeros(len(ti_list))
    pos_sig = np.zeros(len(ti_list))
    vel_sig = np.zeros(len(ti_list))
    resid_array = np.zeros(len(ti_list))
    ii = 0
    for ti in ti_list:
        X_est = filter_output[ti]['X']
        P_est = filter_output[ti]['P']
        resids = filter_output[ti]['resids']
        
        X_true = truth_dict[ti]
        
        pos_err[ii] = float(X_est[0] - X_true[0])
        vel_err[ii] = float(X_est[1] - X_true[1])
        pos_sig[ii] = np.sqrt(P_est[0,0])
        vel_sig[ii] = np.sqrt(P_est[1,1])
        resid_array[ii] = float(resids[0])
        
        ii += 1
        
    # Compute RMS errors
    RMS_pos = np.sqrt(np.mean(np.square(pos_err)))
    RMS_vel = np.sqrt(np.mean(np.square(vel_err)))
    RMS_resid1 = np.sqrt(np.mean(np.square(resid_array)))
    
    print 'RMS Results'
    print 'RMS pos', RMS_pos
    print 'RMS vel', RMS_vel
    print 'RMS resid 1 (pos)', RMS_resid1
    

    # Plot Position and Velocity Errors
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(ti_list,pos_err,'k.')
    plt.plot(ti_list, 3.*pos_sig, 'k--')
    plt.plot(ti_list, -3.*pos_sig, 'k--')
    plt.ylabel('Position Error [m]')
    
    plt.subplot(3,1,2)
    plt.plot(ti_list, vel_err, 'k.')
    plt.plot(ti_list, 3.*vel_sig, 'k--')
    plt.plot(ti_list, -3.*vel_sig, 'k--')
    plt.ylabel('Velocity Error [m/s]')
    
    plt.subplot(3,1,3)
    plt.plot(ti_list, resid_array, 'k.')
    plt.ylabel('Post-Fit Resids [m]')
    plt.xlabel('Time [sec]')    
    
    plt.show()
    
    return



###############################################################################
# Measurement Function
###############################################################################

def unscented_balldrop(chi, inputs):
    '''
    Function for use with unscented_transform.
    Parameters
    ------
    chi : L x (2L+1) numpy array
      sigma point matrix
    inputs : dictionary
      input parameters

    Returns
    ------
    Y : m x (2L+1) numpy array
      transformed sigma point matrix
    '''

    # Size of input/output
    row = int(chi.shape[0])
    col = int(chi.shape[1])
    Y = np.zeros((row, col))

    for j in xrange(0, col):

        # Pull out column of chi
        y = chi[0,j]
        dy = chi[1,j]

        Y[0,j] = float(y)
        Y[1,j] = float(dy)

    return Y


def H_balldrop(Xref, inputs):
    
    # Break out state
    y = float(Xref[0])
    dy = float(Xref[1])
    
    # Hi_til and Gi
    Hi_til = np.diag([1.,1.])
    Gi = np.array([[y],[dy]])
    
    return Hi_til, Gi



###############################################################################
# Integration Routines
###############################################################################
  

def int_balldrop(X, t, inputs):
    
    y = float(X[0])
    dy = float(X[1])
    
    dX = np.zeros(2)
    dX[0] = dy
    dX[1] = inputs['acc']
    
    dX = dX.flatten()
    
    return dX
    


def int_balldrop_stm(X, t, inputs):
    '''
    
    '''

    # Input data
    acc = inputs['acc']

    # Compute number of states
    n = (-1 + np.sqrt(1 + 4*len(X)))/2

    # State Vector
    y = X[0]
    dy = X[1]

    # Generate A matrix
    A = np.zeros((n, n))
    A[0,1] = 1.
    

    # Compute STM components dphi = A*phi
    phi_mat = np.reshape(X[n:], (n, n))
    dphi_mat = np.dot(A, phi_mat)
    dphi_v = np.reshape(dphi_mat, (n**2, 1))

    # Derivative vector
    dX = np.zeros((n+n**2, 1))

    dX[0] = dy
    dX[1] = acc
    dX[n:] = dphi_v

    dX = dX.flatten()

    return dX
    
    
def int_balldrop_ukf(X, t, inputs):

    # Break out inputs
    acc = inputs['acc']

    # Initialize
    dX = [0]*len(X)
    n = int((-1 + np.sqrt(1. + 8.*len(X)))/4.)

    for ind in xrange(0, 2*n+1):

        # Pull out relevant values from X
        x1 = float(X[ind*n])
        x2 = float(X[ind*n + 1])
        

        # Solve for components of dX
        dX[ind*n] = copy.copy(x2)
        dX[ind*n + 1] = copy.copy(acc)
        

    return dX
    
    
###############################################################################
# Debugging functions
###############################################################################

def test_int_balldrop_ukf():
    
    intfcn = int_balldrop_ukf
    ti_prior = 0.
    ti = 100.
    
    chi = np.zeros((2,5))
    L = 2
    chi_v = np.reshape(chi, (L*(2*L+1), 1), order='F')

    print chi_v
    
    inputs = {}
    inputs['acc'] = 1.
    
    int_tol = 1e-12

    # Integrate chi
    if ti_prior == ti:
        intout = chi_v.T
    else:
        int0 = chi_v.flatten()
        tin = [ti_prior, ti]
        intout = odeint(intfcn, int0, tin, args=(inputs,), rtol=int_tol,
                        atol=int_tol)
                        
    print intout
    
    return
    
def test_odeint():
    
    intfcn = ifunc.int_twobody
    
    X = np.reshape([7000.,0.,500.,0.,7.5,0.], (6,1))
    P = np.diag([1.,1.,1.,1e-6,1e-6,1e-6])
    
    inputs = {}
    inputs['GM'] = 3.986e5
    
    int_tol = 1e-8
    
    tin = [0.,10.]
    
    # Two Body
    int0 = X.flatten()
    intout = odeint(intfcn, int0, tin, args=(inputs,), rtol=int_tol, 
                    atol=int_tol)
    
    print 'Normal Exec'
    print intout
    x1 = np.reshape(intout[-1,:], (6,1))
    print x1
    
    print 
    print
    
    
    # UKF Style
    L = 6
    pnorm = 2.
    alpha = 1.
    
    # Prior information about the distribution
    kurt = gamma(5./pnorm)*gamma(1./pnorm)/(gamma(3./pnorm)**2.)
    beta = kurt - 1.
    kappa = kurt - float(L)

    # Compute sigma point weights
    lam = alpha**2.*(L + kappa) - L
    gam = np.sqrt(L + lam)
    
    sqP = np.linalg.cholesky(P)
    Xrep = np.tile(X, (1, L))
    chi = np.concatenate((X, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
    chi_v = np.reshape(chi, (L*(2*L+1), 1), order='F')
    
    intfcn = ifunc.int_twobody_ukf
    int0 = chi_v.flatten()
    
    intout = odeint(intfcn, int0, tin, args=(inputs,), rtol=int_tol,
                    atol=int_tol)
    
    print 'UKF Exec'
    print intout
    x2 = np.reshape(intout[-1,0:6], (6,1))
    print x2
    
    test = x2 - x1
    print test
    
    return
    
###############################################################################
# Execute Functions
###############################################################################
  
# balldrop_setup()
  
execute_balldrop_test()

balldrop_analysis()
  
#test_int_balldrop_ukf()

#  test_odeint()