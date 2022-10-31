import numpy as np
from scipy import stats
import pickle
import matplotlib.pyplot as plt
import sys
import os
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

import estimation.estimation_functions as est
import utilities.coordinate_systems as coord
import utilities.tle_functions as tle

from utilities.constants import arcsec2rad





###############################################################################
# Linear Motion Analysis
###############################################################################

def compute_linear1d_errors(filter_output, truth_dict):
    
    # Compute errors
    n = 2
    p = 1
    X_err = np.zeros((n, len(filter_output)))
    resids = np.zeros((p, len(filter_output)))
    sig_x = np.zeros(len(filter_output),)
    sig_dx = np.zeros(len(filter_output),)
    tk_list = list(filter_output.keys())
    for kk in range(len(filter_output)):
        tk = tk_list[kk]
        X = filter_output[tk]['X']
        P = filter_output[tk]['P']
        resids[:,kk] = filter_output[tk]['resids'].flatten()
        
        X_true = truth_dict[tk]
        X_err[:,kk] = (X - X_true).flatten()
        sig_x[kk] = np.sqrt(P[0,0])
        sig_dx[kk] = np.sqrt(P[1,1])
        
    resids_max = np.ceil(np.max(np.abs(resids))) 
        
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tk_list, X_err[0,:], 'k.')
    plt.plot(tk_list, 3*sig_x, 'k--')
    plt.plot(tk_list, -3*sig_x, 'k--')
#    plt.ylim([-1, 1])
    plt.ylabel('Pos Err [m]')
    
    plt.subplot(3,1,2)
    plt.plot(tk_list, X_err[1,:], 'k.')
    plt.plot(tk_list, 3*sig_dx, 'k--')
    plt.plot(tk_list, -3*sig_dx, 'k--')
#    plt.ylim([-0.05, 0.05])
    plt.ylabel('Vel Err [m/s]')
    
    
    plt.subplot(3,1,3)
    plt.plot(tk_list, resids[0,:], 'k.')
    plt.ylim([-resids_max, resids_max])
    plt.ylabel('Range Resids [m]')
    plt.xlabel('Time [sec]')
    
    plt.show()
    
   
    print('\nError Statistics')
    print('Pos mean and std [m]: ' + '{:.3f}'.format(np.mean(X_err[0,:])) + ', {:.3f}'.format(np.std(X_err[0,:])))
    print('Vel mean and std [m/s]: ' + '{:.3f}'.format(np.mean(X_err[1,:])) + ', {:.3f}'.format(np.std(X_err[1,:])))
    print('Rg Resids mean and std [m]: ' + '{:.3f}'.format(np.mean(resids[0,:])) + ', {:.3f}'.format(np.std(resids[0,:])))
    
    
    
    return



###############################################################################
# Balldrop Analysis
###############################################################################

def compute_balldrop_errors(filter_output, truth_dict):
    
    # Compute errors
    n = 2
    p = 2
    X_err = np.zeros((n, len(filter_output)))
    resids = np.zeros((p, len(filter_output)))
    sig_y = np.zeros(len(filter_output),)
    sig_dy = np.zeros(len(filter_output),)
    tk_list = list(filter_output.keys())
    for kk in range(len(filter_output)):
        tk = tk_list[kk]
        X = filter_output[tk]['X']
        P = filter_output[tk]['P']
        resids[:,kk] = filter_output[tk]['resids'].flatten()
        
        X_true = truth_dict[tk]
        X_err[:,kk] = (X - X_true).flatten()
        sig_y[kk] = np.sqrt(P[0,0])
        sig_dy[kk] = np.sqrt(P[1,1])
        
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tk_list, X_err[0,:], 'k.')
    plt.plot(tk_list, 3*sig_y, 'k--')
    plt.plot(tk_list, -3*sig_y, 'k--')
    plt.ylabel('Pos Err [m]')
    
    plt.subplot(2,1,2)
    plt.plot(tk_list, X_err[1,:], 'k.')
    plt.plot(tk_list, 3*sig_dy, 'k--')
    plt.plot(tk_list, -3*sig_dy, 'k--')
    plt.ylabel('Vel Err [m/s]')
    plt.xlabel('Time [sec]')
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tk_list, resids[0,:], 'k.')
    plt.ylabel('Y Resids [m]')
    
    plt.subplot(2,1,2)
    plt.plot(tk_list, resids[1,:], 'k.')
    plt.ylabel('dY Resids [m/s]')
    plt.xlabel('Time [sec]')
    
    plt.show()
    
    print('\nError Statistics')
    print('Pos mean and std [m]: ' + '{:.3f}'.format(np.mean(X_err[0,:])) + ', {:.3f}'.format(np.std(X_err[0,:])))
    print('Vel mean and std [m/s]: ' + '{:.3f}'.format(np.mean(X_err[1,:])) + ', {:.3f}'.format(np.std(X_err[1,:])))
    print('Pos Resids mean and std [m]: ' + '{:.3f}'.format(np.mean(resids[0,:])) + ', {:.3f}'.format(np.std(resids[0,:])))
    print('Vel Resids mean and std [m]: ' + '{:.3f}'.format(np.mean(resids[1,:])) + ', {:.3f}'.format(np.std(resids[1,:])))
    
    
    return


###############################################################################
# PDF Analysis
###############################################################################

def compute_LAM(GMM_dict, mc_points):
    '''
    This function computes the likelihood agreement measure between a 
    Gaussian Mixture Model and a set of Monte-Carlo samples
    
    Parameters
    ------
    GMM_dict : dictionary
        contains lists of GMM component weights, means, and covars
    mc_points : Nxn numpy array
        each row corresponts to one MC sample
        
    Returns
    ------
    LAM : float
        likelihood agreement measure
        
    Reference
    ------
    DeMars, K.J., Bishop, R.H., Jah, M.K., "Entropy-Based Approach for 
        Uncertainty Propagation of Nonlinear Dynamical Systems," JGCD 2013.
    
    '''
    
    
    # Break out GMM
    w = GMM_dict['weights']
    m = GMM_dict['means']
    P = GMM_dict['covars']
    nstates = len(m[0])
    
    # Loop to compute Likelihood Agreement Measure (Eq 25)
    N = len(mc_points) 
    LAM = 0.
    for jj in range(len(w)):
        wj = w[jj]
        mj = m[jj]
        Pj = P[jj]
        
        for ii in range(N):
            xi = mc_points[ii].reshape(nstates, 1)
            pg = est.gaussian_likelihood(xi, mj, Pj)
            LAM += (1./N)*wj*pg

    return LAM


def plot_pdf_contours(GMM_dict, axis1=0, axis2=1):
    '''
    The function plots the PDF contours of a given Gaussian Mixture Model.
    
    Parameters
    ------
    GMM_dict : dictionary
        contains weights, means, and covars of GMM
    axis1 : int, optional
        index of coordinate to plot as x-axis (default=0)
    axis2 : int, optional
        index of coordinate to plot as y-axis (default=1)
    
    '''
    
    # Break out GMM
    weights = GMM_dict['weights']
    means = GMM_dict['means']
    covars = GMM_dict['covars']
    
    print('Ncomps', len(weights))
    print('wmax', max(weights))
    print('wmin', min(weights))
    
    # Merge GMM to compute overall mean and covar to establish plot parameters
    params = {}
    params['prune_T'] = 0.
    params['merge_U'] = 1e10
    GMM_merge = est.merge_GMM(GMM_dict, params)
    
    m = GMM_merge['means'][0].flatten()
    P = GMM_merge['covars'][0]
    sig1 = np.sqrt(P[axis1,axis1])
    sig2 = np.sqrt(P[axis2,axis2])
    
    xmin = m[axis1] - 10*sig1
    xmax = m[axis1] + 10*sig1
    ymin = m[axis2] - 10*sig2
    ymax = m[axis2] + 10*sig2
    
    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)
    
#    # Merge GMM to get cleaner plot
#    params = {}
#    params['prune_T'] = 0
#    params['merge_U'] = 0.1
#    GMM_merge = est.merge_GMM(GMM_dict, params)
#    
#    # Break out GMM
#    weights = GMM_merge['weights']
#    means = GMM_merge['means']
#    covars = GMM_merge['covars']
#    
#    print('Ncomps', len(weights))
    
    
    xgrid, ygrid = np.meshgrid(x, y)
    z = np.zeros((len(y), len(x)))
    
    for jj in range(len(weights)):
        
        wj = weights[jj]
        mj = np.array([means[jj][axis1], means[jj][axis2]]).flatten()
        Pj = np.array([[covars[jj][axis1,axis1], covars[jj][axis1,axis2]],
                       [covars[jj][axis2,axis1], covars[jj][axis2,axis2]]])
    
        z += wj * stats.multivariate_normal(mj, Pj).pdf(np.dstack((xgrid, ygrid)))
    
    print(z.shape)
    zmax = np.max(z)
    print(zmax)
    levels = np.logspace(-8,-1,8)*zmax
    
    print(levels)
    
    plt.contour(x, y, z, levels=5)

    return


def test_pdf_contours():
    
    
    x1 = np.reshape([3,-1], (2,1))
    P1 = np.array([[2., 1.],[1., 8.]])
    
    GMM_dict = {}
    GMM_dict['weights'] = [1.]
    GMM_dict['means'] = [x1]
    GMM_dict['covars'] = [P1]
    
    plot_pdf_contours(GMM_dict)
    
    
    m1 = np.reshape([0, 0, 0], (3,1))
    m2 = np.reshape([10, 10, 0], (3,1))
    m3 = np.reshape([-10, 0, 10], (3,1))
    
    P1 = np.array([[40., 0., 0.],
                   [0., 10., 0.],
                   [0., 0., 10.]])
    
    P2 = np.array([[10., 0., 0.],
                   [0., 40., 0.],
                   [0., 0., 10.]])
    
    P3 = np.array([[10., 0., 0.],
                   [0., 10., 0.],
                   [0., 0., 40.]])
    
    GMM_dict2 = {}
    GMM_dict2['weights'] = [1./3., 1./3., 1./3.]
    GMM_dict2['means'] = [m1, m2, m3]
    GMM_dict2['covars'] = [P1, P2, P3]
    
    mc_points = est.gmm_samples(GMM_dict2, 10000)
    
    plt.figure()
    plt.plot(mc_points[:,0], mc_points[:,1], 'k.', alpha=0.05)
    plot_pdf_contours(GMM_dict2, axis1=0, axis2=1)
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.figure()
    plt.plot(mc_points[:,1], mc_points[:,2], 'k.', alpha=0.05)
    plot_pdf_contours(GMM_dict2, axis1=1, axis2=2)
    plt.xlabel('y')
    plt.ylabel('z')
    
    plt.figure()
    plt.plot(mc_points[:,0], mc_points[:,2], 'k.', alpha=0.05)
    plot_pdf_contours(GMM_dict2, axis1=0, axis2=2)
    plt.xlabel('x')
    plt.ylabel('z')
    
    plt.show()
    
    
    return


###############################################################################
# Orbit Analysis
###############################################################################


def compute_orbit_errors(filter_output, full_state_output, truth_dict):
    
#    pklFile = open(fname, 'rb' )
#    data = pickle.load( pklFile )
#    filter_output = data[0]
#    full_state_output = data[1]
#    truth_dict = data[2]
#    pklFile.close()
    
    # Times
    tk_list = list(full_state_output.keys())
    t0 = sorted(truth_dict.keys())[0]
    
    print(t0)
    print(tk_list[0])
    thrs = [(tk - t0).total_seconds()/3600. for tk in tk_list]
    
    meas_tk_list = list(filter_output.keys())
    meas_t0 = sorted(meas_tk_list)[0]
    thrs_meas = [(tk - t0).total_seconds()/3600. for tk in meas_tk_list]
    
    # Number of states and measurements
    Xo = filter_output[meas_t0]['X']
    resids0 = filter_output[meas_t0]['resids']
    n = len(Xo)
    p = len(resids0)
    
#    print('Estimated Initial State Xo', Xo)

    # Compute state errors
    X_err = np.zeros((n, len(full_state_output)))
    X_err_ric = np.zeros((3, len(full_state_output)))
    X_err_meas = np.zeros((n, len(filter_output)))
    X_err_ric_meas = np.zeros((3, len(filter_output)))
    resids = np.zeros((p, len(filter_output)))
    sig_x = np.zeros(len(full_state_output),)
    sig_y = np.zeros(len(full_state_output),)
    sig_z = np.zeros(len(full_state_output),)
    sig_dx = np.zeros(len(full_state_output),)
    sig_dy = np.zeros(len(full_state_output),)
    sig_dz = np.zeros(len(full_state_output),)
    sig_r = np.zeros(len(full_state_output),)
    sig_i = np.zeros(len(full_state_output),)
    sig_c = np.zeros(len(full_state_output),)
    sig_beta = np.zeros(len(full_state_output),)
    
    meas_ind = 0 
    for kk in range(len(full_state_output)):
        tk = tk_list[kk]
        X = full_state_output[tk]['X']
        P = full_state_output[tk]['P']
                
        X_true = truth_dict[tk]
        X_err[:,kk] = (X - X_true).flatten()
        sig_x[kk] = np.sqrt(P[0,0])
        sig_y[kk] = np.sqrt(P[1,1])
        sig_z[kk] = np.sqrt(P[2,2])
        sig_dx[kk] = np.sqrt(P[3,3])
        sig_dy[kk] = np.sqrt(P[4,4])
        sig_dz[kk] = np.sqrt(P[5,5])
        
        if n > 6:
            sig_beta[kk] = np.sqrt(P[6,6])
        
        # RIC Errors and Covariance
        rc_vect = X_true[0:3].reshape(3,1)
        vc_vect = X_true[3:6].reshape(3,1)
        err_eci = X_err[0:3,kk].reshape(3,1)
        P_eci = P[0:3,0:3]
        
        err_ric = coord.eci2ric(rc_vect, vc_vect, err_eci)
        P_ric = coord.eci2ric(rc_vect, vc_vect, P_eci)
        X_err_ric[:,kk] = err_ric.flatten()
        sig_r[kk] = np.sqrt(P_ric[0,0])
        sig_i[kk] = np.sqrt(P_ric[1,1])
        sig_c[kk] = np.sqrt(P_ric[2,2])
        
        # Store data at meas times
        if tk in meas_tk_list:
            X_err_meas[:,meas_ind] = (X - X_true).flatten()
            X_err_ric_meas[:,meas_ind] = err_ric.flatten()
            resids[:,meas_ind] = filter_output[tk]['resids'].flatten()
            meas_ind += 1
        
        
    # Fix Units
    X_err *= 1000.      # convert to m, m/s
    X_err_meas *= 1000.
    X_err_ric *= 1000.
    X_err_ric_meas *= 1000.
    sig_x *= 1000.
    sig_y *= 1000.
    sig_z *= 1000.
    sig_dx *= 1000.
    sig_dy *= 1000.
    sig_dz *= 1000.
    sig_r *= 1000.
    sig_i *= 1000.
    sig_c *= 1000.
    
    if n > 6:
        X_err[6,:] *= 1000.
        X_err_meas[6,:] *= 1000.
        sig_beta *= 1e6
    
    if p == 1:
        resids[0,:] *= 1000.
    if p == 2:
        resids *= (1./arcsec2rad)
    if p == 3:
        resids[0,:] *= 1000.
        resids[1:3,:] *= (1./arcsec2rad)
    
    

    # Compute and print statistics
    print('\n\nState Error and Residuals Analysis')
    print('\n\t\t\t\t  Mean\t\tSTD')
    print('----------------------------------------')
    print('X ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[0,:])), '\t{0:0.2E}'.format(np.std(X_err[0,:])))
    print('Y ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[1,:])), '\t{0:0.2E}'.format(np.std(X_err[1,:])))
    print('Z ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[2,:])), '\t{0:0.2E}'.format(np.std(X_err[2,:])))
    print('dX ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[3,:])), '\t{0:0.2E}'.format(np.std(X_err[3,:])))
    print('dY ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[4,:])), '\t{0:0.2E}'.format(np.std(X_err[4,:])))
    print('dZ ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[5,:])), '\t{0:0.2E}'.format(np.std(X_err[5,:])))
    print('')
    print('Radial [m]\t\t', '{0:0.2E}'.format(np.mean(X_err_ric[0,:])), '\t{0:0.2E}'.format(np.std(X_err_ric[0,:])))
    print('In-Track [m]\t', '{0:0.2E}'.format(np.mean(X_err_ric[1,:])), '\t{0:0.2E}'.format(np.std(X_err_ric[1,:])))
    print('Cross-Track [m]\t', '{0:0.2E}'.format(np.mean(X_err_ric[2,:])), '\t{0:0.2E}'.format(np.std(X_err_ric[2,:])))
    print('')
    
    if n > 6:
        print('Beta [m^2/kg]\t', '{0:0.2E}'.format(np.mean(X_err[6,:])), '\t{0:0.2E}'.format(np.std(X_err[6,:])))
        print('')
    
    if p == 1:
        print('Range [m]\t\t', '{0:0.2E}'.format(np.mean(resids[0,:])), '\t{0:0.2E}'.format(np.std(resids[0,:])))
        
    if p == 2:
        print('RA [arcsec]\t\t', '{0:0.2E}'.format(np.mean(resids[0,:])), '\t{0:0.2E}'.format(np.std(resids[0,:])))
        print('DEC [arcsec]\t', '{0:0.2E}'.format(np.mean(resids[1,:])), '\t{0:0.2E}'.format(np.std(resids[1,:])))
    
    if p == 3:
        print('Range [m]\t\t', '{0:0.2E}'.format(np.mean(resids[0,:])), '\t{0:0.2E}'.format(np.std(resids[0,:])))
        print('RA [arcsec]\t\t', '{0:0.2E}'.format(np.mean(resids[1,:])), '\t{0:0.2E}'.format(np.std(resids[1,:])))
        print('DEC [arcsec]\t', '{0:0.2E}'.format(np.mean(resids[2,:])), '\t{0:0.2E}'.format(np.std(resids[2,:])))
        


    
    # State Error Plots   
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs, X_err[0,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[0,:], 'b.')
    plt.plot(thrs, 3*sig_x, 'k--')
    plt.plot(thrs, -3*sig_x, 'k--')
    plt.ylabel('X Err [m]')
    
    plt.subplot(3,1,2)
    plt.plot(thrs, X_err[1,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[1,:], 'b.')
    plt.plot(thrs, 3*sig_y, 'k--')
    plt.plot(thrs, -3*sig_y, 'k--')
    plt.ylabel('Y Err [m]')
    
    plt.subplot(3,1,3)
    plt.plot(thrs, X_err[2,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[2,:], 'b.')
    plt.plot(thrs, 3*sig_z, 'k--')
    plt.plot(thrs, -3*sig_z, 'k--')
    plt.ylabel('Z Err [m]')

    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs, X_err[3,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[3,:], 'b.')
    plt.plot(thrs, 3*sig_dx, 'k--')
    plt.plot(thrs, -3*sig_dx, 'k--')
    plt.ylabel('dX Err [m/s]')
    
    plt.subplot(3,1,2)
    plt.plot(thrs, X_err[4,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[4,:], 'b.')
    plt.plot(thrs, 3*sig_dy, 'k--')
    plt.plot(thrs, -3*sig_dy, 'k--')
    plt.ylabel('dY Err [m/s]')
    
    plt.subplot(3,1,3)
    plt.plot(thrs, X_err[5,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[5,:], 'b.')
    plt.plot(thrs, 3*sig_dz, 'k--')
    plt.plot(thrs, -3*sig_dz, 'k--')
    plt.ylabel('dZ Err [m/s]')

    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs, X_err_ric[0,:], 'k.')
    plt.plot(thrs_meas, X_err_ric_meas[0,:], 'b.')
    plt.plot(thrs, 3*sig_r, 'k--')
    plt.plot(thrs, -3*sig_r, 'k--')
    plt.ylabel('Radial [m]')
    
    plt.subplot(3,1,2)
    plt.plot(thrs, X_err_ric[1,:], 'k.')
    plt.plot(thrs_meas, X_err_ric_meas[1,:], 'b.')
    plt.plot(thrs, 3*sig_i, 'k--')
    plt.plot(thrs, -3*sig_i, 'k--')
    plt.ylabel('In-Track [m]')
    
    plt.subplot(3,1,3)
    plt.plot(thrs, X_err_ric[2,:], 'k.')
    plt.plot(thrs_meas, X_err_ric_meas[2,:], 'b.')
    plt.plot(thrs, 3*sig_c, 'k--')
    plt.plot(thrs, -3*sig_c, 'k--')
    plt.ylabel('Cross-Track [m]')

    plt.xlabel('Time [hours]')
    
    if n == 7:
        plt.figure()
        plt.plot(thrs, X_err[6,:], 'k.')
        plt.plot(thrs_meas, X_err_meas[6,:], 'b.')
        plt.plot(thrs, 3*sig_beta, 'k--')
        plt.plot(thrs, -3*sig_beta, 'k--')
        plt.ylabel('Beta [m^2/kg]')
        plt.title('Additional Parameters')
        
        plt.xlabel('Time [hours]')
    
    
    # Residuals
    plt.figure()
    if p == 1:
        plt.plot(thrs_meas, resids[0,:], 'k.')
        plt.ylabel('Range [m]')       
        plt.xlabel('Time [hours]')
    
    if p == 2:
        
        plt.subplot(2,1,1)
        plt.plot(thrs_meas, resids[0,:], 'k.')
        plt.ylabel('RA [arcsec]')
        
        plt.subplot(2,1,2)
        plt.plot(thrs_meas, resids[1,:], 'k.')
        plt.ylabel('DEC [arcsec]')
        
        plt.xlabel('Time [hours]')
    
    if p == 3:
        plt.subplot(3,1,1)
        plt.plot(thrs_meas, resids[0,:], 'k.')
        plt.ylabel('Range [m]')
        
        plt.subplot(3,1,2)
        plt.plot(thrs_meas, resids[1,:], 'k.')
        plt.ylabel('RA [arcsec]')
        
        plt.subplot(3,1,3)
        plt.plot(thrs_meas, resids[2,:], 'k.')
        plt.ylabel('DEC [arcsec]')
        
        plt.xlabel('Time [hours]')
        
    
        
    
    plt.show()
    
    
    
    
    
    return


def compute_aegis_errors(filter_output, full_state_output, truth_dict):
    
#    pklFile = open(fname, 'rb' )
#    data = pickle.load( pklFile )
#    filter_output = data[0]
#    full_state_output = data[1]
#    truth_dict = data[2]
#    pklFile.close()
    
    # Times
    tk_list = list(full_state_output.keys())
    t0 = sorted(truth_dict.keys())[0]
    
    print(t0)
    print(tk_list[0])
    thrs = [(tk - t0).total_seconds()/3600. for tk in tk_list]
    
    meas_tk_list = list(filter_output.keys())
    meas_t0 = sorted(meas_tk_list)[0]
    thrs_meas = [(tk - t0).total_seconds()/3600. for tk in meas_tk_list]
    
    # Number of states and measurements
    Xo = filter_output[meas_t0]['means'][0]
    resids0 = filter_output[meas_t0]['resids']
    n = len(Xo)
    p = len(resids0)
    
    # Merge GMM params
    params = {}
    params['prune_T'] = 0.
    params['merge_U'] = 1e12
    
#    print('Estimated Initial State Xo', Xo)

    # Compute state errors
    X_err = np.zeros((n, len(full_state_output)))
    X_err_ric = np.zeros((3, len(full_state_output)))
    X_err_meas = np.zeros((n, len(filter_output)))
    X_err_ric_meas = np.zeros((3, len(filter_output)))
    resids = np.zeros((p, len(filter_output)))
    sig_x = np.zeros(len(full_state_output),)
    sig_y = np.zeros(len(full_state_output),)
    sig_z = np.zeros(len(full_state_output),)
    sig_dx = np.zeros(len(full_state_output),)
    sig_dy = np.zeros(len(full_state_output),)
    sig_dz = np.zeros(len(full_state_output),)
    sig_r = np.zeros(len(full_state_output),)
    sig_i = np.zeros(len(full_state_output),)
    sig_c = np.zeros(len(full_state_output),)
    sig_beta = np.zeros(len(full_state_output),)
    ncomp_array = np.zeros(len(full_state_output),)
    sumweights = np.zeros(len(full_state_output),)
    maxweights = np.zeros(len(full_state_output),)
    
    meas_ind = 0 
    for kk in range(len(full_state_output)):
        tk = tk_list[kk]
        
        # Retrieve and form GMM
        weights = full_state_output[tk]['weights']
        means = full_state_output[tk]['means']
        covars = full_state_output[tk]['covars']
        GMM_dict = {}
        GMM_dict['weights'] = weights
        GMM_dict['means'] = means
        GMM_dict['covars'] = covars
        ncomp_array[kk] = len(weights)
        sumweights[kk] = sum(weights)
        maxweights[kk] = max(weights)
        
        # Merge GMM to get mean state and covar for plots
        GMM_merge = est.merge_GMM(GMM_dict, params)
        X = GMM_merge['means'][0]
        P = GMM_merge['covars'][0]
                
        # Compute errors
        X_true = truth_dict[tk]
        X_err[:,kk] = (X - X_true).flatten()
        sig_x[kk] = np.sqrt(P[0,0])
        sig_y[kk] = np.sqrt(P[1,1])
        sig_z[kk] = np.sqrt(P[2,2])
        sig_dx[kk] = np.sqrt(P[3,3])
        sig_dy[kk] = np.sqrt(P[4,4])
        sig_dz[kk] = np.sqrt(P[5,5])
        
        if n > 6:
            sig_beta[kk] = np.sqrt(P[6,6])
        
        # RIC Errors and Covariance
        rc_vect = X_true[0:3].reshape(3,1)
        vc_vect = X_true[3:6].reshape(3,1)
        err_eci = X_err[0:3,kk].reshape(3,1)
        P_eci = P[0:3,0:3]
        
        err_ric = coord.eci2ric(rc_vect, vc_vect, err_eci)
        P_ric = coord.eci2ric(rc_vect, vc_vect, P_eci)
        X_err_ric[:,kk] = err_ric.flatten()
        sig_r[kk] = np.sqrt(P_ric[0,0])
        sig_i[kk] = np.sqrt(P_ric[1,1])
        sig_c[kk] = np.sqrt(P_ric[2,2])
        
        # Store data at meas times
        if tk in meas_tk_list:
            X_err_meas[:,meas_ind] = (X - X_true).flatten()
            X_err_ric_meas[:,meas_ind] = err_ric.flatten()
            resids[:,meas_ind] = filter_output[tk]['resids'].flatten()
            meas_ind += 1
        
        
    # Fix Units
    X_err *= 1000.      # convert to m, m/s
    X_err_meas *= 1000.
    X_err_ric *= 1000.
    X_err_ric_meas *= 1000.
    sig_x *= 1000.
    sig_y *= 1000.
    sig_z *= 1000.
    sig_dx *= 1000.
    sig_dy *= 1000.
    sig_dz *= 1000.
    sig_r *= 1000.
    sig_i *= 1000.
    sig_c *= 1000.
    
    if n > 6:
        X_err[6,:] *= 1000.
        X_err_meas[6,:] *= 1000.
        sig_beta *= 1e6
    
    if p == 1:
        resids[0,:] *= 1000.
    if p == 2:
        resids *= (1./arcsec2rad)
    if p == 3:
        resids[0,:] *= 1000.
        resids[1:3,:] *= (1./arcsec2rad)
    
    

    # Compute and print statistics
    print('\n\nState Error and Residuals Analysis')
    print('\n\t\t\t\t  Mean\t\tSTD')
    print('----------------------------------------')
    print('X ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[0,:])), '\t{0:0.2E}'.format(np.std(X_err[0,:])))
    print('Y ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[1,:])), '\t{0:0.2E}'.format(np.std(X_err[1,:])))
    print('Z ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[2,:])), '\t{0:0.2E}'.format(np.std(X_err[2,:])))
    print('dX ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[3,:])), '\t{0:0.2E}'.format(np.std(X_err[3,:])))
    print('dY ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[4,:])), '\t{0:0.2E}'.format(np.std(X_err[4,:])))
    print('dZ ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[5,:])), '\t{0:0.2E}'.format(np.std(X_err[5,:])))
    print('')
    print('Radial [m]\t\t', '{0:0.2E}'.format(np.mean(X_err_ric[0,:])), '\t{0:0.2E}'.format(np.std(X_err_ric[0,:])))
    print('In-Track [m]\t', '{0:0.2E}'.format(np.mean(X_err_ric[1,:])), '\t{0:0.2E}'.format(np.std(X_err_ric[1,:])))
    print('Cross-Track [m]\t', '{0:0.2E}'.format(np.mean(X_err_ric[2,:])), '\t{0:0.2E}'.format(np.std(X_err_ric[2,:])))
    print('')
    
    if n > 6:
        print('Beta [m^2/kg]\t', '{0:0.2E}'.format(np.mean(X_err[6,:])), '\t{0:0.2E}'.format(np.std(X_err[6,:])))
        print('')
    
    if p == 1:
        print('Range [m]\t\t', '{0:0.2E}'.format(np.mean(resids[0,:])), '\t{0:0.2E}'.format(np.std(resids[0,:])))
        
    if p == 2:
        print('RA [arcsec]\t\t', '{0:0.2E}'.format(np.mean(resids[0,:])), '\t{0:0.2E}'.format(np.std(resids[0,:])))
        print('DEC [arcsec]\t', '{0:0.2E}'.format(np.mean(resids[1,:])), '\t{0:0.2E}'.format(np.std(resids[1,:])))
    
    if p == 3:
        print('Range [m]\t\t', '{0:0.2E}'.format(np.mean(resids[0,:])), '\t{0:0.2E}'.format(np.std(resids[0,:])))
        print('RA [arcsec]\t\t', '{0:0.2E}'.format(np.mean(resids[1,:])), '\t{0:0.2E}'.format(np.std(resids[1,:])))
        print('DEC [arcsec]\t', '{0:0.2E}'.format(np.mean(resids[2,:])), '\t{0:0.2E}'.format(np.std(resids[2,:])))
        


    
    # State Error Plots   
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs, X_err[0,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[0,:], 'b.')
    plt.plot(thrs, 3*sig_x, 'k--')
    plt.plot(thrs, -3*sig_x, 'k--')
    plt.ylabel('X Err [m]')
    
    plt.subplot(3,1,2)
    plt.plot(thrs, X_err[1,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[1,:], 'b.')
    plt.plot(thrs, 3*sig_y, 'k--')
    plt.plot(thrs, -3*sig_y, 'k--')
    plt.ylabel('Y Err [m]')
    
    plt.subplot(3,1,3)
    plt.plot(thrs, X_err[2,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[2,:], 'b.')
    plt.plot(thrs, 3*sig_z, 'k--')
    plt.plot(thrs, -3*sig_z, 'k--')
    plt.ylabel('Z Err [m]')

    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs, X_err[3,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[3,:], 'b.')
    plt.plot(thrs, 3*sig_dx, 'k--')
    plt.plot(thrs, -3*sig_dx, 'k--')
    plt.ylabel('dX Err [m/s]')
    
    plt.subplot(3,1,2)
    plt.plot(thrs, X_err[4,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[4,:], 'b.')
    plt.plot(thrs, 3*sig_dy, 'k--')
    plt.plot(thrs, -3*sig_dy, 'k--')
    plt.ylabel('dY Err [m/s]')
    
    plt.subplot(3,1,3)
    plt.plot(thrs, X_err[5,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[5,:], 'b.')
    plt.plot(thrs, 3*sig_dz, 'k--')
    plt.plot(thrs, -3*sig_dz, 'k--')
    plt.ylabel('dZ Err [m/s]')

    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs, X_err_ric[0,:], 'k.')
    plt.plot(thrs_meas, X_err_ric_meas[0,:], 'b.')
    plt.plot(thrs, 3*sig_r, 'k--')
    plt.plot(thrs, -3*sig_r, 'k--')
    plt.ylabel('Radial [m]')
    
    plt.subplot(3,1,2)
    plt.plot(thrs, X_err_ric[1,:], 'k.')
    plt.plot(thrs_meas, X_err_ric_meas[1,:], 'b.')
    plt.plot(thrs, 3*sig_i, 'k--')
    plt.plot(thrs, -3*sig_i, 'k--')
    plt.ylabel('In-Track [m]')
    
    plt.subplot(3,1,3)
    plt.plot(thrs, X_err_ric[2,:], 'k.')
    plt.plot(thrs_meas, X_err_ric_meas[2,:], 'b.')
    plt.plot(thrs, 3*sig_c, 'k--')
    plt.plot(thrs, -3*sig_c, 'k--')
    plt.ylabel('Cross-Track [m]')

    plt.xlabel('Time [hours]')
    
    if n == 7:
        plt.figure()
        plt.plot(thrs, X_err[6,:], 'k.')
        plt.plot(thrs_meas, X_err_meas[6,:], 'b.')
        plt.plot(thrs, 3*sig_beta, 'k--')
        plt.plot(thrs, -3*sig_beta, 'k--')
        plt.ylabel('Beta [m^2/kg]')
        plt.title('Additional Parameters')
        
        plt.xlabel('Time [hours]')
    
    
    # Residuals
    plt.figure()
    if p == 1:
        plt.plot(thrs_meas, resids[0,:], 'k.')
        plt.ylabel('Range [m]')       
        plt.xlabel('Time [hours]')
    
    if p == 2:
        
        plt.subplot(2,1,1)
        plt.plot(thrs_meas, resids[0,:], 'k.')
        plt.ylabel('RA [arcsec]')
        
        plt.subplot(2,1,2)
        plt.plot(thrs_meas, resids[1,:], 'k.')
        plt.ylabel('DEC [arcsec]')
        
        plt.xlabel('Time [hours]')
    
    if p == 3:
        plt.subplot(3,1,1)
        plt.plot(thrs_meas, resids[0,:], 'k.')
        plt.ylabel('Range [m]')
        
        plt.subplot(3,1,2)
        plt.plot(thrs_meas, resids[1,:], 'k.')
        plt.ylabel('RA [arcsec]')
        
        plt.subplot(3,1,3)
        plt.plot(thrs_meas, resids[2,:], 'k.')
        plt.ylabel('DEC [arcsec]')
        
        plt.xlabel('Time [hours]')
        
        
    # Components and weights
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs_meas, ncomp_array, 'k.')
    plt.ylabel('Num Comps')    
    plt.subplot(3,1,2)
    plt.plot(thrs_meas, maxweights, 'k.')
    plt.ylabel('Max Weight')    
    plt.subplot(3,1,3)
    plt.plot(thrs_meas, sumweights, 'k.')
    plt.ylabel('Sum Weights')
    plt.xlabel('Time [hours]')
        
    
    plt.show()
    
    
    
    
    
    return


def compute_real_orbit_errors(filter_output, full_state_output, truth_dict, norad_id):
    
#    pklFile = open(fname, 'rb' )
#    data = pickle.load( pklFile )
#    filter_output = data[0]
#    full_state_output = data[1]
#    truth_dict = data[2]
#    pklFile.close()
    
    # Times
    tk_list = list(full_state_output.keys())
    t0 = sorted(truth_dict.keys())[0]
    
    print(t0)
    print(tk_list[0])
    thrs = [(tk - t0).total_seconds()/3600. for tk in tk_list]
    
    meas_tk_list = list(filter_output.keys())
    meas_t0 = sorted(meas_tk_list)[0]
    thrs_meas = [(tk - t0).total_seconds()/3600. for tk in meas_tk_list]
    
    # Number of states and measurements
    Xo = filter_output[meas_t0]['X']
    resids0 = filter_output[meas_t0]['resids']
    n = len(Xo)
    p = len(resids0)
    
#    print('Estimated Initial State Xo', Xo)

    # Compute state errors
    X_err = np.zeros((n, len(full_state_output)))
    X_err_ric = np.zeros((3, len(full_state_output)))
    X_err_meas = np.zeros((n, len(filter_output)))
    X_err_ric_meas = np.zeros((3, len(filter_output)))
    resids = np.zeros((p, len(filter_output)))
    sig_x = np.zeros(len(full_state_output),)
    sig_y = np.zeros(len(full_state_output),)
    sig_z = np.zeros(len(full_state_output),)
    sig_dx = np.zeros(len(full_state_output),)
    sig_dy = np.zeros(len(full_state_output),)
    sig_dz = np.zeros(len(full_state_output),)
    sig_r = np.zeros(len(full_state_output),)
    sig_i = np.zeros(len(full_state_output),)
    sig_c = np.zeros(len(full_state_output),)
    sig_beta = np.zeros(len(full_state_output),)
    
    meas_ind = 0 
    for kk in range(len(full_state_output)):
        tk = tk_list[kk]
        X = full_state_output[tk]['X']
        P = full_state_output[tk]['P']
                
        r_true = truth_dict[tk]
        r_est = X[0:3]
        
        X_err[0:3,kk] = (r_est - r_true).flatten()
        sig_x[kk] = np.sqrt(P[0,0])
        sig_y[kk] = np.sqrt(P[1,1])
        sig_z[kk] = np.sqrt(P[2,2])
#        sig_dx[kk] = np.sqrt(P[3,3])
#        sig_dy[kk] = np.sqrt(P[4,4])
#        sig_dz[kk] = np.sqrt(P[5,5])
        
        if n > 6:
            sig_beta[kk] = np.sqrt(P[6,6])
        
        # RIC Errors and Covariance
        # Use estimated state as chief then flip sign to get err = est - true
        rc_vect = X[0:3].reshape(3,1)
        vc_vect = X[3:6].reshape(3,1)
        err_eci = r_true - r_est        
        P_eci = P[0:3,0:3]
        
        err_ric = -coord.eci2ric(rc_vect, vc_vect, err_eci)  # flip sign
        P_ric = coord.eci2ric(rc_vect, vc_vect, P_eci)
        X_err_ric[:,kk] = err_ric.flatten()
        sig_r[kk] = np.sqrt(P_ric[0,0])
        sig_i[kk] = np.sqrt(P_ric[1,1])
        sig_c[kk] = np.sqrt(P_ric[2,2])
        
        # Store data at meas times
        if tk in meas_tk_list:
            X_err_meas[0:3,meas_ind] = (r_est - r_true).flatten()
            X_err_ric_meas[:,meas_ind] = err_ric.flatten()
            resids[:,meas_ind] = filter_output[tk]['resids'].flatten()
            meas_ind += 1
            
            
    # TLE Errors
    # Retrieve and propagate TLE data to desired times
    tle_state = tle.propagate_TLE([norad_id], tk_list)
    
    # Compute errors at each time
    tle_eci_err = np.zeros((3, len(tk_list)))
    tle_ric_err = np.zeros((3, len(tk_list)))
    for ii in range(len(tk_list)):
        
        tk = tk_list[ii]        
        r_true = truth_dict[tk]
        tle_r_eci = tle_state[norad_id]['r_GCRF'][ii].reshape(3,1)
        tle_v_eci = tle_state[norad_id]['v_GCRF'][ii].reshape(3,1)
        
        # Compute RIC errors with TLE data acting as chief satellite
        rho_eci = r_true - tle_r_eci    
        rho_ric = coord.eci2ric(tle_r_eci, tle_v_eci, rho_eci)
        
        # Change sign to set SP3 data as chief (truth)
        rho_ric = -rho_ric      
        
        # Store output
        tle_eci_err[:,ii] = (tle_r_eci - r_true).flatten()
        tle_ric_err[:,ii] = rho_ric.flatten()
        
        
    # Fix Units
    X_err *= 1000.      # convert to m, m/s
    X_err_meas *= 1000.
    X_err_ric *= 1000.
    X_err_ric_meas *= 1000.
    tle_eci_err *= 1000.
    tle_ric_err *= 1000.
    sig_x *= 1000.
    sig_y *= 1000.
    sig_z *= 1000.
    sig_dx *= 1000.
    sig_dy *= 1000.
    sig_dz *= 1000.
    sig_r *= 1000.
    sig_i *= 1000.
    sig_c *= 1000.
    
    if n > 6:
        X_err[6,:] *= 1000.
        X_err_meas[6,:] *= 1000.
        sig_beta *= 1e6
    
    if p == 1:
        resids[0,:] *= 1000.
    if p == 2:
        resids *= (1./arcsec2rad)
    if p == 3:
        resids[0,:] *= 1000.
        resids[1:3,:] *= (1./arcsec2rad)
    
    

    # Compute and print statistics
    print('\n\nState Error and Residuals Analysis')
    print('\n\t\t\t\t  Mean\t\tSTD')
    print('----------------------------------------')
    print('X ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[0,:])), '\t{0:0.2E}'.format(np.std(X_err[0,:])))
    print('Y ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[1,:])), '\t{0:0.2E}'.format(np.std(X_err[1,:])))
    print('Z ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[2,:])), '\t{0:0.2E}'.format(np.std(X_err[2,:])))
#    print('dX ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[3,:])), '\t{0:0.2E}'.format(np.std(X_err[3,:])))
#    print('dY ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[4,:])), '\t{0:0.2E}'.format(np.std(X_err[4,:])))
#    print('dZ ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[5,:])), '\t{0:0.2E}'.format(np.std(X_err[5,:])))
    print('')
    print('Radial [m]\t\t', '{0:0.2E}'.format(np.mean(X_err_ric[0,:])), '\t{0:0.2E}'.format(np.std(X_err_ric[0,:])))
    print('In-Track [m]\t', '{0:0.2E}'.format(np.mean(X_err_ric[1,:])), '\t{0:0.2E}'.format(np.std(X_err_ric[1,:])))
    print('Cross-Track [m]\t', '{0:0.2E}'.format(np.mean(X_err_ric[2,:])), '\t{0:0.2E}'.format(np.std(X_err_ric[2,:])))
    print('')
    print('TLE X ECI [m]\t\t', '{0:0.2E}'.format(np.mean(tle_eci_err[0,:])), '\t{0:0.2E}'.format(np.std(tle_eci_err[0,:])))
    print('TLE Y ECI [m]\t\t', '{0:0.2E}'.format(np.mean(tle_eci_err[1,:])), '\t{0:0.2E}'.format(np.std(tle_eci_err[1,:])))
    print('TLE Z ECI [m]\t\t', '{0:0.2E}'.format(np.mean(tle_eci_err[2,:])), '\t{0:0.2E}'.format(np.std(tle_eci_err[2,:])))
#    print('dX ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[3,:])), '\t{0:0.2E}'.format(np.std(X_err[3,:])))
#    print('dY ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[4,:])), '\t{0:0.2E}'.format(np.std(X_err[4,:])))
#    print('dZ ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[5,:])), '\t{0:0.2E}'.format(np.std(X_err[5,:])))
    print('')
    print('TLE Radial [m]\t\t', '{0:0.2E}'.format(np.mean(tle_ric_err[0,:])), '\t{0:0.2E}'.format(np.std(tle_ric_err[0,:])))
    print('TLE In-Track [m]\t', '{0:0.2E}'.format(np.mean(tle_ric_err[1,:])), '\t{0:0.2E}'.format(np.std(tle_ric_err[1,:])))
    print('TLE Cross-Track [m]\t', '{0:0.2E}'.format(np.mean(tle_ric_err[2,:])), '\t{0:0.2E}'.format(np.std(tle_ric_err[2,:])))
    print('')
    
    if n > 6:
        print('Beta [m^2/kg]\t', '{0:0.2E}'.format(np.mean(X_err[6,:])), '\t{0:0.2E}'.format(np.std(X_err[6,:])))
        print('')
        
    if p == 1:
        print('Range [m]\t\t', '{0:0.2E}'.format(np.mean(resids[0,:])), '\t{0:0.2E}'.format(np.std(resids[0,:])))
    
    if p == 2:
        print('RA [arcsec]\t\t', '{0:0.2E}'.format(np.mean(resids[0,:])), '\t{0:0.2E}'.format(np.std(resids[0,:])))
        print('DEC [arcsec]\t', '{0:0.2E}'.format(np.mean(resids[1,:])), '\t{0:0.2E}'.format(np.std(resids[1,:])))
    
    if p == 3:
        print('Range [m]\t\t', '{0:0.2E}'.format(np.mean(resids[0,:])), '\t{0:0.2E}'.format(np.std(resids[0,:])))
        print('RA [arcsec]\t\t', '{0:0.2E}'.format(np.mean(resids[1,:])), '\t{0:0.2E}'.format(np.std(resids[1,:])))
        print('DEC [arcsec]\t', '{0:0.2E}'.format(np.mean(resids[2,:])), '\t{0:0.2E}'.format(np.std(resids[2,:])))
        


    
    # State Error Plots   
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs, X_err[0,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[0,:], 'b.')
    plt.plot(thrs, tle_eci_err[0,:], 'r.')
    plt.plot(thrs, 3*sig_x, 'k--')
    plt.plot(thrs, -3*sig_x, 'k--')
    ymax = max([abs(max(X_err[0,:])), abs(max(tle_eci_err[0,:])), 3*abs(max(sig_x))])
    plt.ylim([-1.5*ymax, 1.5*ymax])
    plt.ylabel('X Err [m]')
    
    plt.subplot(3,1,2)
    plt.plot(thrs, X_err[1,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[1,:], 'b.')
    plt.plot(thrs, tle_eci_err[1,:], 'r.')
    plt.plot(thrs, 3*sig_y, 'k--')
    plt.plot(thrs, -3*sig_y, 'k--')
    ymax = max([abs(max(X_err[1,:])), abs(max(tle_eci_err[1,:])), 3*abs(max(sig_y))])
    plt.ylim([-1.5*ymax, 1.5*ymax])
    plt.ylabel('Y Err [m]')
    
    plt.subplot(3,1,3)
    plt.plot(thrs, X_err[2,:], 'k.')
    plt.plot(thrs_meas, X_err_meas[2,:], 'b.')
    plt.plot(thrs, tle_eci_err[2,:], 'r.')
    plt.plot(thrs, 3*sig_z, 'k--')
    plt.plot(thrs, -3*sig_z, 'k--')
    ymax = max([abs(max(X_err[2,:])), abs(max(tle_eci_err[2,:])), 3*abs(max(sig_z))])
    plt.ylim([-1.5*ymax, 1.5*ymax])
    plt.ylabel('Z Err [m]')

    plt.xlabel('Time [hours]')
    
#    plt.figure()
#    plt.subplot(3,1,1)
#    plt.plot(thrs, X_err[3,:], 'k.')
#    plt.plot(thrs_meas, X_err_meas[3,:], 'b.')
#    plt.plot(thrs, 3*sig_dx, 'k--')
#    plt.plot(thrs, -3*sig_dx, 'k--')
#    plt.ylabel('dX Err [m/s]')
#    
#    plt.subplot(3,1,2)
#    plt.plot(thrs, X_err[4,:], 'k.')
#    plt.plot(thrs_meas, X_err_meas[4,:], 'b.')
#    plt.plot(thrs, 3*sig_dy, 'k--')
#    plt.plot(thrs, -3*sig_dy, 'k--')
#    plt.ylabel('dY Err [m/s]')
#    
#    plt.subplot(3,1,3)
#    plt.plot(thrs, X_err[5,:], 'k.')
#    plt.plot(thrs_meas, X_err_meas[5,:], 'b.')
#    plt.plot(thrs, 3*sig_dz, 'k--')
#    plt.plot(thrs, -3*sig_dz, 'k--')
#    plt.ylabel('dZ Err [m/s]')
#
#    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs, X_err_ric[0,:], 'k.')
    plt.plot(thrs_meas, X_err_ric_meas[0,:], 'b.')
    plt.plot(thrs, tle_ric_err[0,:], 'r.')
    plt.plot(thrs, 3*sig_r, 'k--')
    plt.plot(thrs, -3*sig_r, 'k--')
    ymax = max([abs(max(X_err_ric[0,:])), abs(max(tle_ric_err[0,:])), 3*abs(max(sig_r))])
    plt.ylim([-1.5*ymax, 1.5*ymax])
    plt.ylabel('Radial [m]')
    
    plt.subplot(3,1,2)
    plt.plot(thrs, X_err_ric[1,:], 'k.')
    plt.plot(thrs_meas, X_err_ric_meas[1,:], 'b.')
    plt.plot(thrs, tle_ric_err[1,:], 'r.')
    plt.plot(thrs, 3*sig_i, 'k--')
    plt.plot(thrs, -3*sig_i, 'k--')
    ymax = max([abs(max(X_err_ric[1,:])), abs(max(tle_ric_err[1,:])), 3*abs(max(sig_i))])
    plt.ylim([-1.5*ymax, 1.5*ymax])
    plt.ylabel('In-Track [m]')
    
    plt.subplot(3,1,3)
    plt.plot(thrs, X_err_ric[2,:], 'k.')
    plt.plot(thrs_meas, X_err_ric_meas[2,:], 'b.')
    plt.plot(thrs, tle_ric_err[2,:], 'r.')
    plt.plot(thrs, 3*sig_c, 'k--')
    plt.plot(thrs, -3*sig_c, 'k--')
    ymax = max([abs(max(X_err_ric[2,:])), abs(max(tle_ric_err[2,:])), 3*abs(max(sig_c))])
    plt.ylim([-1.5*ymax, 1.5*ymax])
    plt.ylabel('Cross-Track [m]')

    plt.xlabel('Time [hours]')
    
    if n == 7:
        plt.figure()
        plt.plot(thrs, X_err[6,:], 'k.')
        plt.plot(thrs_meas, X_err_meas[6,:], 'b.')
        plt.plot(thrs, 3*sig_beta, 'k--')
        plt.plot(thrs, -3*sig_beta, 'k--')
        plt.ylabel('Beta [m^2/kg]')
        plt.title('Additional Parameters')
        
        plt.xlabel('Time [hours]')
    
    
    # Residuals
    plt.figure()
    
    
    if p == 2:
        
        plt.subplot(2,1,1)
        plt.plot(thrs_meas, resids[0,:], 'k.')
        plt.ylabel('RA [arcsec]')
        
        plt.subplot(2,1,2)
        plt.plot(thrs_meas, resids[1,:], 'k.')
        plt.ylabel('DEC [arcsec]')
        
        plt.xlabel('Time [hours]')
    
    if p == 3:
        plt.subplot(3,1,1)
        plt.plot(thrs_meas, resids[0,:], 'k.')
        plt.ylabel('Range [m]')
        
        plt.subplot(3,1,2)
        plt.plot(thrs_meas, resids[1,:], 'k.')
        plt.ylabel('RA [arcsec]')
        
        plt.subplot(3,1,3)
        plt.plot(thrs_meas, resids[2,:], 'k.')
        plt.ylabel('DEC [arcsec]')
        
        plt.xlabel('Time [hours]')
        
    
        
    
    plt.show()
    
    
    
    
    
    return





if __name__ == '__main__':
    
    plt.close('all')
    
    test_pdf_contours()
    
    
    
    
    
    
    
    
    
    
    