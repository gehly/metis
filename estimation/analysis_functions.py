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
# Vo Coordinated Turn Test Case
###############################################################################

def compute_coordturn_errors(filter_output, full_state_output, truth_dict):
    '''
    
    '''
    
    # True cardinality and states
    tk_truth = sorted(list(truth_dict.keys()))
    N_truth = []
    plot_truth = {}
    full_obj_list = []
    for tk in tk_truth:
        obj_list_truth = list(truth_dict[tk].keys())
        N_truth.append(len(obj_list_truth))
        
        # Store object states and plot times
        for obj_id in obj_list_truth:            
            if obj_id not in plot_truth:
                plot_truth[obj_id] = {}
                plot_truth[obj_id]['tk_plot'] = []
                plot_truth[obj_id]['x_plot'] = []
                plot_truth[obj_id]['y_plot'] = []
                full_obj_list.append(obj_id)
            
            plot_truth[obj_id]['tk_plot'].append(tk)
            plot_truth[obj_id]['x_plot'].append(truth_dict[tk][obj_id][0])
            plot_truth[obj_id]['y_plot'].append(truth_dict[tk][obj_id][2])
            
            
    full_obj_list = sorted(list(set(full_obj_list)))        
        
    
    # Estimated cardinality and states
    tk_list = sorted(list(filter_output.keys()))
    N_est = []
    plot_est = {}
    full_label_list = []
    for tk in tk_list:
        N_est.append(filter_output[tk]['N'])
        LMB_k = filter_output[tk]['LMB_dict']
        card_k = filter_output[tk]['card']
        label_k = filter_output[tk]['label_list']
        rk_list = filter_output[tk]['rk_list']
        Xk_list = filter_output[tk]['Xk_list']
        Pk_list = filter_output[tk]['Pk_list']
        
        
        for jj in range(len(label_k)):
            label = label_k[jj]
            Xj = Xk_list[jj]
            Pj = Pk_list[jj]
            rj = rk_list[jj]
            
            if label not in plot_est:
                plot_est[label] = {}
                plot_est[label]['tk_plot'] = []
                plot_est[label]['r_plot'] = []
                plot_est[label]['x_plot'] = []
                plot_est[label]['y_plot'] = []
                plot_est[label]['x_sig'] = []
                plot_est[label]['y_sig'] = []
                full_label_list.append(label)
                
            plot_est[label]['tk_plot'].append(tk)
            plot_est[label]['r_plot'].append(rj)
            plot_est[label]['x_plot'].append(Xj[0])
            plot_est[label]['y_plot'].append(Xj[2])
            plot_est[label]['x_sig'].append(np.sqrt(Pj[0,0]))
            plot_est[label]['y_sig'].append(np.sqrt(Pj[2,2]))
    
    
    # Cardinality plot
    plt.figure()
    plt.plot(tk_truth, N_truth, 'k-')
    plt.plot(tk_list, N_est, 'k.')
    plt.ylabel('Cardinality')
    plt.xlabel('Time [sec]')
    plt.legend(['Truth', 'Est'])
    
    
    # True/Est State plot
    cm = plt.get_cmap('hsv')
    num_colors = len(full_obj_list)
    
    fig1 = plt.figure()
    fig2 = plt.figure()   
    ii = 0
    for obj_id in full_obj_list:
        
        color_ii = cm(ii/num_colors)
        
        tk_plot = plot_truth[obj_id]['tk_plot']
        x_plot = plot_truth[obj_id]['x_plot']
        y_plot = plot_truth[obj_id]['y_plot']
        
        plt.figure(fig1)
        plt.subplot(2,1,1)
        plt.plot(tk_plot, x_plot, '-', color=color_ii)
        plt.ylabel('x [m]')
        plt.subplot(2,1,2)
        plt.plot(tk_plot, y_plot, '-', color=color_ii)
        plt.ylabel('y [m]')
        plt.xlabel('Time [sec]')
        
        plt.figure(fig2)
        plt.plot(x_plot, y_plot, '-', color=color_ii)
        plt.plot(x_plot[0], y_plot[0], 'o', color=color_ii)
        plt.plot(x_plot[-1], y_plot[-1], 'x', color=color_ii)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.xlim([-2000., 2000.])
        plt.ylim([0., 2000.])
        
        
        ii += 1
        
    for label in full_label_list:
        
        tk_plot = plot_est[label]['tk_plot']
        x_plot = plot_est[label]['x_plot']
        y_plot = plot_est[label]['y_plot']
        
        plt.figure(fig1)
        plt.subplot(2,1,1)
        plt.plot(tk_plot, x_plot, 'ko', fillstyle='none', ms=3)
        plt.subplot(2,1,2)
        plt.plot(tk_plot, y_plot, 'ko', fillstyle='none', ms=3)
        
        plt.figure(fig2)
        plt.plot(x_plot, y_plot, 'ko', fillstyle='none', ms=3)
        
    
    
    plt.show()
    
    
    
    
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


###############################################################################
# Multitarget Orbit Analysis
###############################################################################

def multitarget_orbit_errors(filter_output, full_state_output, truth_dict):
    
    # OSPA parameters
    pnorm = 2.
    c = 100.    # km, penalty for cardinality errors
    
    # Times
    tk_list = list(full_state_output.keys())
    t0 = sorted(truth_dict.keys())[0]
    
    # print(t0)
    # print(tk_list[0])
    thrs = [(tk - t0).total_seconds()/3600. for tk in tk_list]
    
    meas_tk_list = list(filter_output.keys())
    meas_t0 = sorted(meas_tk_list)[0]
    thrs_meas = [(tk - t0).total_seconds()/3600. for tk in meas_tk_list]
    
    # Number of states and measurements
    Xo = filter_output[meas_t0]['means'][0]
    resids0 = filter_output[meas_t0]['resids'][0]
    n = len(Xo)
    p = len(resids0)
    
    # Compute state errors
    X_err = np.zeros((n, len(full_state_output)))
    X_err_ric = np.zeros((3, len(full_state_output)))
    X_err_meas = np.zeros((n, len(filter_output)))
    X_err_ric_meas = np.zeros((3, len(filter_output)))
    ospa = np.zeros(len(full_state_output),)
    ospa_pos = np.zeros(len(full_state_output),)
    ospa_vel = np.zeros(len(full_state_output),)
    ospa_card = np.zeros(len(full_state_output),)
    
    
    # resids = np.zeros((p, len(filter_output)))
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
        
        print('')
        print(tk)
        
        # Retrieve GMM and extracted state estimate
        weights = full_state_output[tk]['weights']
        means = full_state_output[tk]['means']
        covars = full_state_output[tk]['covars']
        wk_list = full_state_output[tk]['wk_list']
        Xk_list = full_state_output[tk]['Xk_list']
        Pk_list = full_state_output[tk]['Pk_list']

        ncomp_array[kk] = len(weights)
        sumweights[kk] = sum(weights)
        maxweights[kk] = max(weights)
        
        # Compute OSPA errors
        Xt_list = truth_dict[tk]['Xt_list']
        
        OSPA, OSPA_pos, OSPA_vel, OSPA_card, row_indices = \
            compute_ospa(Xt_list, Xk_list, pnorm, c)
            
        ospa[kk] = OSPA
        ospa_pos[kk] = OSPA_pos
        ospa_vel[kk] = OSPA_vel
        ospa_card[kk] = OSPA_card
        
        # # Choose 1 object as representative case for error/covariance plots
        # if len(Xt_list) >= len(Xk_list):
        #     ii = row_indices[0]            
        # else:
        #     ii = row_indices.index(0)

        # print(row_indices)
        # print(ii)
        # print(wk_list)
        # print(Xt_list)
        # print(Xk_list)
        

        # Xt = Xt_list[0]
        # wk = wk_list[ii]
        # Xk = Xk_list[ii]
        # Pk = Pk_list[ii]
        
        # X_err[:,kk] = (Xk - Xt).flatten()
        # sig_x[kk] = np.sqrt(Pk[0,0])
        # sig_y[kk] = np.sqrt(Pk[1,1])
        # sig_z[kk] = np.sqrt(Pk[2,2])
        # sig_dx[kk] = np.sqrt(Pk[3,3])
        # sig_dy[kk] = np.sqrt(Pk[4,4])
        # sig_dz[kk] = np.sqrt(Pk[5,5])

        # # RIC Errors and Covariance
        # rc_vect = Xt[0:3].reshape(3,1)
        # vc_vect = Xt[3:6].reshape(3,1)
        # err_eci = X_err[0:3,kk].reshape(3,1)
        # P_eci = Pk[0:3,0:3]
        
        # err_ric = coord.eci2ric(rc_vect, vc_vect, err_eci)
        # P_ric = coord.eci2ric(rc_vect, vc_vect, P_eci)
        # X_err_ric[:,kk] = err_ric.flatten()
        # sig_r[kk] = np.sqrt(P_ric[0,0])
        # sig_i[kk] = np.sqrt(P_ric[1,1])
        # sig_c[kk] = np.sqrt(P_ric[2,2])
        
        # # Store data at meas times
        # if tk in meas_tk_list:
        #     X_err_meas[:,meas_ind] = (Xk - Xt).flatten()
        #     X_err_ric_meas[:,meas_ind] = err_ric.flatten()
            
        #     # resids_k = filter_output[tk]['resids']
        #     # resids[:,meas_ind] = filter_output[tk]['resids'].flatten()
        #     meas_ind += 1
        
        
    # Fix Units
    X_err *= 1000.      # convert to m, m/s
    X_err_meas *= 1000.
    X_err_ric *= 1000.
    X_err_ric_meas *= 1000.
    ospa *= 1000.
    ospa_pos *= 1000.
    ospa_vel *= 1000.
    sig_x *= 1000.
    sig_y *= 1000.
    sig_z *= 1000.
    sig_dx *= 1000.
    sig_dy *= 1000.
    sig_dz *= 1000.
    sig_r *= 1000.
    sig_i *= 1000.
    sig_c *= 1000.
    

    
    # if p == 1:
    #     resids[0,:] *= 1000.
    # if p == 2:
    #     resids *= (1./arcsec2rad)
    # if p == 3:
    #     resids[0,:] *= 1000.
    #     resids[1:3,:] *= (1./arcsec2rad)
    
    

    # Compute and print statistics
    conv_ind = int(len(full_state_output)/2)
    print('\n\nState Error and Residuals Analysis')
    print('\n\t\t\t\t  Mean\t\tSTD')
    print('----------------------------------------')
    print('X ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[0,conv_ind:])), '\t{0:0.2E}'.format(np.std(X_err[0,conv_ind:])))
    print('Y ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[1,conv_ind:])), '\t{0:0.2E}'.format(np.std(X_err[1,conv_ind:])))
    print('Z ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[2,conv_ind:])), '\t{0:0.2E}'.format(np.std(X_err[2,conv_ind:])))
    print('dX ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[3,conv_ind:])), '\t{0:0.2E}'.format(np.std(X_err[3,conv_ind:])))
    print('dY ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[4,conv_ind:])), '\t{0:0.2E}'.format(np.std(X_err[4,conv_ind:])))
    print('dZ ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[5,conv_ind:])), '\t{0:0.2E}'.format(np.std(X_err[5,conv_ind:])))
    print('')
    print('Radial [m]\t\t', '{0:0.2E}'.format(np.mean(X_err_ric[0,conv_ind:])), '\t{0:0.2E}'.format(np.std(X_err_ric[0,conv_ind:])))
    print('In-Track [m]\t', '{0:0.2E}'.format(np.mean(X_err_ric[1,conv_ind:])), '\t{0:0.2E}'.format(np.std(X_err_ric[1,conv_ind:])))
    print('Cross-Track [m]\t', '{0:0.2E}'.format(np.mean(X_err_ric[2,conv_ind:])), '\t{0:0.2E}'.format(np.std(X_err_ric[2,conv_ind:])))
    print('')
    

    
    # if p == 1:
    #     print('Range [m]\t\t', '{0:0.2E}'.format(np.mean(resids[0,:])), '\t{0:0.2E}'.format(np.std(resids[0,:])))
        
    # if p == 2:
    #     print('RA [arcsec]\t\t', '{0:0.2E}'.format(np.mean(resids[0,:])), '\t{0:0.2E}'.format(np.std(resids[0,:])))
    #     print('DEC [arcsec]\t', '{0:0.2E}'.format(np.mean(resids[1,:])), '\t{0:0.2E}'.format(np.std(resids[1,:])))
    
    # if p == 3:
    #     print('Range [m]\t\t', '{0:0.2E}'.format(np.mean(resids[0,:])), '\t{0:0.2E}'.format(np.std(resids[0,:])))
    #     print('RA [arcsec]\t\t', '{0:0.2E}'.format(np.mean(resids[1,:])), '\t{0:0.2E}'.format(np.std(resids[1,:])))
    #     print('DEC [arcsec]\t', '{0:0.2E}'.format(np.mean(resids[2,:])), '\t{0:0.2E}'.format(np.std(resids[2,:])))
        


    
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
    

    
    # # Residuals
    # plt.figure()
    # if p == 1:
    #     plt.plot(thrs_meas, resids[0,:], 'k.')
    #     plt.ylabel('Range [m]')       
    #     plt.xlabel('Time [hours]')
    
    # if p == 2:
        
    #     plt.subplot(2,1,1)
    #     plt.plot(thrs_meas, resids[0,:], 'k.')
    #     plt.ylabel('RA [arcsec]')
        
    #     plt.subplot(2,1,2)
    #     plt.plot(thrs_meas, resids[1,:], 'k.')
    #     plt.ylabel('DEC [arcsec]')
        
    #     plt.xlabel('Time [hours]')
    
    # if p == 3:
    #     plt.subplot(3,1,1)
    #     plt.plot(thrs_meas, resids[0,:], 'k.')
    #     plt.ylabel('Range [m]')
        
    #     plt.subplot(3,1,2)
    #     plt.plot(thrs_meas, resids[1,:], 'k.')
    #     plt.ylabel('RA [arcsec]')
        
    #     plt.subplot(3,1,3)
    #     plt.plot(thrs_meas, resids[2,:], 'k.')
    #     plt.ylabel('DEC [arcsec]')
        
    #     plt.xlabel('Time [hours]')
        
        
        
    # OSPA
    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(thrs, ospa, 'k.')
    plt.ylabel('OSPA')
    plt.subplot(4,1,2)
    plt.plot(thrs, ospa_pos, 'k.')
    plt.ylabel('OSPA Pos [m]')
    plt.subplot(4,1,3)
    plt.plot(thrs, ospa_vel, 'k.')
    plt.ylabel('OSPA Vel [m/s]')
    plt.subplot(4,1,4)
    plt.plot(thrs, ospa_card, 'k.')
    plt.ylabel('OSPA Card')
    plt.xlabel('Time [hours]')
        
    # Resids
    plt.figure()
    clist = ['r', 'g', 'b', 'c', 'k', 'y', 'm', 'c']
    for kk in range(len(meas_tk_list)):
        
        tk = meas_tk_list[kk]
        resids_k = filter_output[tk]['resids']
        
        for ii in range(len(resids_k)):
            
            ind = int(ii % len(clist))
            color_ii = clist[ind]
            ra_arcsec = resids_k[ii][0]*(1./arcsec2rad)
            dec_arcsec = resids_k[ii][1]*(1./arcsec2rad)
            
            plt.subplot(2,1,1)
            plt.plot(thrs_meas[kk], ra_arcsec, '.', c=color_ii)
            plt.subplot(2,1,2)
            plt.plot(thrs_meas[kk], dec_arcsec, '.', c=color_ii)
            
    plt.subplot(2,1,1)
    plt.ylabel('RA [arcsec]')
    plt.subplot(2,1,2)
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



def lmb_orbit_errors(filter_output, full_state_output, truth_dict):
    
    # OSPA parameters
    pnorm = 2.
    c = 100.    # km, penalty for cardinality errors
    
    # Times
    tk_list = list(full_state_output.keys())
    t0 = sorted(truth_dict.keys())[0]
    
    obj_id_list = sorted(truth_dict[tk_list[0]].keys())
    if 42709 in obj_id_list:
        del obj_id_list[obj_id_list.index(42709)]
    
    print(obj_id_list)
    
    # print(t0)
    # print(tk_list[0])
    thrs = [(tk - t0).total_seconds()/3600. for tk in tk_list]
    
    meas_tk_list = list(filter_output.keys())
    meas_t0 = sorted(meas_tk_list)[0]
    thrs_meas = [(tk - t0).total_seconds()/3600. for tk in meas_tk_list]
    
    # Number of states and measurements
    obj_id = list(truth_dict[meas_t0].keys())[0]
    Xo = truth_dict[meas_t0][obj_id]
    resids0 = filter_output[meas_t0]['resids'][0]
    n = len(Xo)
    p = len(resids0)
    
    # Compute state errors
    X_err = np.zeros((n, len(full_state_output)))
    X_err_ric = np.zeros((3, len(full_state_output)))
    X_err_meas = np.zeros((n, len(filter_output)))
    X_err_ric_meas = np.zeros((3, len(filter_output)))
    ospa = np.zeros(len(full_state_output),)
    ospa_pos = np.zeros(len(full_state_output),)
    ospa_vel = np.zeros(len(full_state_output),)
    ospa_card = np.zeros(len(full_state_output),)
    
    
    resids_plot = np.empty((2,0))
    thrs_resids = []
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
    nlabel_array = np.zeros(len(full_state_output),)
    N_est = np.zeros(len(full_state_output),)
    N_true = np.zeros(len(full_state_output),)
    rksum_array = np.zeros(len(full_state_output),)
    
    meas_ind = 0 
    for kk in range(len(full_state_output)):
        tk = tk_list[kk]
        
        print('')
        print(tk)
        
        
        # Retrieve GMM and extracted state estimate
        LMB_dict = full_state_output[tk]['LMB_dict']
        card = full_state_output[tk]['card']
        Nk = full_state_output[tk]['N']
        label_list = full_state_output[tk]['label_list']
        rk_list = full_state_output[tk]['rk_list']
        Xk_list = full_state_output[tk]['Xk_list']
        Pk_list = full_state_output[tk]['Pk_list']
        resids_k = full_state_output[tk]['resids']

        # Cardinality related terms
        nlabel_array[kk] = len(label_list)
        N_est[kk] = Nk        
        rksum_array[kk] = sum(rk_list)
        
        # Compute OSPA errors
        # Xt_list = truth_dict[tk]['Xt_list']
        Xt_list = []
        for obj_id in obj_id_list:
            Xt_list.append(truth_dict[tk][obj_id])
        N_true[kk] = len(Xt_list)
        
        OSPA, OSPA_pos, OSPA_vel, OSPA_card, row_indices = \
            compute_ospa(Xt_list, Xk_list, pnorm, c)
            
        ospa[kk] = OSPA
        ospa_pos[kk] = OSPA_pos
        ospa_vel[kk] = OSPA_vel
        ospa_card[kk] = OSPA_card
        
        # Choose 1 object as representative case for error/covariance plots
        ii = 0
        label_plot = label_list[ii]
        
        # if len(Xt_list) >= len(Xk_list):            
        #     ii = row_indices[0]
        # else:
        #     ii = row_indices.index(0)

        print(row_indices)
        print(ii)
        print(Xt_list)
        print(Xk_list)        

        Xt = Xt_list[ii]
        Xk = Xk_list[0]
        Pk = Pk_list[0]
        
        print(label_list[0])
        print(obj_id_list[0])
        print('Xt', Xt)
        print('Xk', Xk)
        
        mistake
        
        X_err[:,kk] = (Xk - Xt).flatten()
        sig_x[kk] = np.sqrt(Pk[0,0])
        sig_y[kk] = np.sqrt(Pk[1,1])
        sig_z[kk] = np.sqrt(Pk[2,2])
        sig_dx[kk] = np.sqrt(Pk[3,3])
        sig_dy[kk] = np.sqrt(Pk[4,4])
        sig_dz[kk] = np.sqrt(Pk[5,5])

        # RIC Errors and Covariance
        rc_vect = Xt[0:3].reshape(3,1)
        vc_vect = Xt[3:6].reshape(3,1)
        err_eci = X_err[0:3,kk].reshape(3,1)
        P_eci = Pk[0:3,0:3]
        
        err_ric = coord.eci2ric(rc_vect, vc_vect, err_eci)
        P_ric = coord.eci2ric(rc_vect, vc_vect, P_eci)
        X_err_ric[:,kk] = err_ric.flatten()
        sig_r[kk] = np.sqrt(P_ric[0,0])
        sig_i[kk] = np.sqrt(P_ric[1,1])
        sig_c[kk] = np.sqrt(P_ric[2,2])
        
        # Store data at meas times
        if tk in meas_tk_list:
            X_err_meas[:,meas_ind] = (Xk - Xt).flatten()
            X_err_ric_meas[:,meas_ind] = err_ric.flatten()
            
            # resids_k = filter_output[tk]['resids']
            
            # for resids_ii in resids_k:
            #     resids_ii *= (1./arcsec2rad)
            #     resids_plot = np.append(resids_plot, resids_ii, axis=1)
            #     thrs_resids.append((tk-t0).total_seconds()/3600.)
            # resids = filter_output[tk]['resids'].flatten()
            meas_ind += 1
        
        
    # Fix Units
    X_err *= 1000.      # convert to m, m/s
    X_err_meas *= 1000.
    X_err_ric *= 1000.
    X_err_ric_meas *= 1000.
    ospa *= 1000.
    ospa_pos *= 1000.
    ospa_vel *= 1000.
    sig_x *= 1000.
    sig_y *= 1000.
    sig_z *= 1000.
    sig_dx *= 1000.
    sig_dy *= 1000.
    sig_dz *= 1000.
    sig_r *= 1000.
    sig_i *= 1000.
    sig_c *= 1000.
    

    
    # if p == 1:
    #     resids[0,:] *= 1000.
    # if p == 2:
    #     resids *= (1./arcsec2rad)
    # if p == 3:
    #     resids[0,:] *= 1000.
    #     resids[1:3,:] *= (1./arcsec2rad)
    
    

    # Compute and print statistics
    conv_ind = int(len(full_state_output)/2)
    print('\n\nState Error and Residuals Analysis')
    print('\n\t\t\t\t  Mean\t\tSTD')
    print('----------------------------------------')
    print('X ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[0,conv_ind:])), '\t{0:0.2E}'.format(np.std(X_err[0,conv_ind:])))
    print('Y ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[1,conv_ind:])), '\t{0:0.2E}'.format(np.std(X_err[1,conv_ind:])))
    print('Z ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[2,conv_ind:])), '\t{0:0.2E}'.format(np.std(X_err[2,conv_ind:])))
    print('dX ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[3,conv_ind:])), '\t{0:0.2E}'.format(np.std(X_err[3,conv_ind:])))
    print('dY ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[4,conv_ind:])), '\t{0:0.2E}'.format(np.std(X_err[4,conv_ind:])))
    print('dZ ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[5,conv_ind:])), '\t{0:0.2E}'.format(np.std(X_err[5,conv_ind:])))
    print('')
    print('Radial [m]\t\t', '{0:0.2E}'.format(np.mean(X_err_ric[0,conv_ind:])), '\t{0:0.2E}'.format(np.std(X_err_ric[0,conv_ind:])))
    print('In-Track [m]\t', '{0:0.2E}'.format(np.mean(X_err_ric[1,conv_ind:])), '\t{0:0.2E}'.format(np.std(X_err_ric[1,conv_ind:])))
    print('Cross-Track [m]\t', '{0:0.2E}'.format(np.mean(X_err_ric[2,conv_ind:])), '\t{0:0.2E}'.format(np.std(X_err_ric[2,conv_ind:])))
    print('')
    

    
    # if p == 1:
    #     print('Range [m]\t\t', '{0:0.2E}'.format(np.mean(resids[0,:])), '\t{0:0.2E}'.format(np.std(resids[0,:])))
        
    # if p == 2:
    #     print('RA [arcsec]\t\t', '{0:0.2E}'.format(np.mean(resids[0,:])), '\t{0:0.2E}'.format(np.std(resids[0,:])))
    #     print('DEC [arcsec]\t', '{0:0.2E}'.format(np.mean(resids[1,:])), '\t{0:0.2E}'.format(np.std(resids[1,:])))
    
    # if p == 3:
    #     print('Range [m]\t\t', '{0:0.2E}'.format(np.mean(resids[0,:])), '\t{0:0.2E}'.format(np.std(resids[0,:])))
    #     print('RA [arcsec]\t\t', '{0:0.2E}'.format(np.mean(resids[1,:])), '\t{0:0.2E}'.format(np.std(resids[1,:])))
    #     print('DEC [arcsec]\t', '{0:0.2E}'.format(np.mean(resids[2,:])), '\t{0:0.2E}'.format(np.std(resids[2,:])))
        


    
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
    

    
    # # Residuals
    # plt.figure()
    # if p == 1:
    #     plt.plot(thrs_meas, resids[0,:], 'k.')
    #     plt.ylabel('Range [m]')       
    #     plt.xlabel('Time [hours]')
    
    # if p == 2:
        
    #     plt.subplot(2,1,1)
    #     plt.plot(thrs_meas, resids[0,:], 'k.')
    #     plt.ylabel('RA [arcsec]')
        
    #     plt.subplot(2,1,2)
    #     plt.plot(thrs_meas, resids[1,:], 'k.')
    #     plt.ylabel('DEC [arcsec]')
        
    #     plt.xlabel('Time [hours]')
    
    # if p == 3:
    #     plt.subplot(3,1,1)
    #     plt.plot(thrs_meas, resids[0,:], 'k.')
    #     plt.ylabel('Range [m]')
        
    #     plt.subplot(3,1,2)
    #     plt.plot(thrs_meas, resids[1,:], 'k.')
    #     plt.ylabel('RA [arcsec]')
        
    #     plt.subplot(3,1,3)
    #     plt.plot(thrs_meas, resids[2,:], 'k.')
    #     plt.ylabel('DEC [arcsec]')
        
    #     plt.xlabel('Time [hours]')
        
        
        
    # OSPA
    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(thrs, ospa, 'k.')
    plt.ylabel('OSPA')
    plt.subplot(4,1,2)
    plt.plot(thrs, ospa_pos, 'k.')
    plt.ylabel('OSPA Pos [m]')
    plt.subplot(4,1,3)
    plt.plot(thrs, ospa_vel, 'k.')
    plt.ylabel('OSPA Vel [m/s]')
    plt.subplot(4,1,4)
    plt.plot(thrs, ospa_card, 'k.')
    plt.ylabel('OSPA Card')
    plt.xlabel('Time [hours]')
        
    # Resids
    plt.figure()
    clist = ['r', 'g', 'b', 'c', 'k', 'y', 'm', 'c']
    for kk in range(len(meas_tk_list)):
        
        tk = meas_tk_list[kk]
        resids_k = filter_output[tk]['resids']
        
        for ii in range(len(resids_k)):
            
            ind = int(ii % len(clist))
            color_ii = clist[ind]
            ra_arcsec = resids_k[ii][0]*(1./arcsec2rad)
            dec_arcsec = resids_k[ii][1]*(1./arcsec2rad)
            
            plt.subplot(2,1,1)
            plt.plot(thrs_meas[kk], ra_arcsec, '.', c=color_ii)
            plt.subplot(2,1,2)
            plt.plot(thrs_meas[kk], dec_arcsec, '.', c=color_ii)
            
    plt.subplot(2,1,1)
    plt.ylabel('RA [arcsec]')
    plt.subplot(2,1,2)
    plt.ylabel('DEC [arcsec]')
    plt.xlabel('Time [hours]')
    
    
    # Cardinaltiy/Number plots
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs, N_true, 'k--')
    plt.plot(thrs, N_est, 'k.')
    plt.legend(['True', 'Est'])
    plt.ylabel('Cardinality')
    plt.subplot(3,1,2)
    plt.plot(thrs, nlabel_array, 'k.')
    plt.ylabel('Num Labels')
    plt.subplot(3,1,3)
    plt.plot(thrs, rksum_array, 'k.')
    plt.ylabel('sum(rk)')
    plt.xlabel('Time [hours]')
    
    
    
    
        
    
    plt.show()
    
    
    
    
    return



def compute_ospa(X1_list,X2_list,p=2.,c=100.) :
    '''


    '''

    # Number of elements in RFSs, assign bigger set to Y,n
    nstates = len(X1_list[0])
    n1 = len(X1_list)
    n2 = len(X2_list)

    if n1 >= n2:
        Y = X1_list
        n = n1
        X = X2_list
        m = n2
    else:
        Y = X2_list
        n = n2
        X = X1_list
        m = n1

    # Step 0: Generate 2D Score Matrix
    A = np.zeros((n,n))
    for ii in range(n):
        for jj in range(n):
            
            # Note: if ii > m, dc = c, A[ii,jj] = c - c = 0
            if ii < m:
                dc = compute_dc(X[ii],Y[jj],p=p,c=c)
                A[ii,jj] = c - dc

    # Step 1: Execute 2D Assignment Algorithm
    if n == 1 :
        row_indices = [0]
    else :
        row_indices = ospa_auction(A)

    #Step 2: For the optimal assignment above, report the distance dc
    #or the cutoff c for each assignment/unassigned point
    alpha_full = np.zeros(n,)
    alpha_pos = np.zeros(n,)
    alpha_vel = np.zeros(n,)
    for jj in range(n):
        ii = row_indices[jj]
        if ii < m:
            alpha_full[ii] = compute_dc(X[ii],Y[jj],p=p,c=c)
            alpha_pos[ii] = compute_dc(X[ii][0:3],Y[jj][0:3],p=p,c=c)
            alpha_vel[ii] = compute_dc(X[ii][3:6],Y[jj][3:6],p=p,c=c)
        else:
            alpha_full[ii] = c

    #Step 3: Compute OSPA and components
    OSPA = ((1./n)*sum([ai**p for ai in alpha_full]))**(1./p)
    OSPA_card = ((1./n)*sum([ai**p for ai in alpha_full[m:n]]))**(1./p)
    OSPA_pos = ((1./n)*sum([ai**p for ai in alpha_pos[0:m]]))**(1./p)
    OSPA_vel = ((1./n)*sum([ai**p for ai in alpha_vel[0:m]]))**(1./p)

    return OSPA, OSPA_pos, OSPA_vel, OSPA_card, row_indices



def compute_dc(x,y,p=2.,c=100.) :
    '''


    '''

    z = x - y
    
    d = sum([abs(zi)**p for zi in z])**(1./p)

    dc = min([c,d])

    return dc


def ospa_auction(A) :
    '''

    '''

    #Get number of rows/columns
    n = int(A.shape[0])

    #Step 1: Initialize assignment matrix and track prices
    assign_mat = np.zeros((n,n))
    price = np.zeros((n,1))
    eps = 1./(2.*n)

    #Repeat until all assignments have been made
    while np.sum(assign_mat) < n:
        for jj in range(n):
            
            #Step 2: Check if column jj is unassigned
            if np.sum(assign_mat[:,jj]) == 0:

                #Step 3: Find the best row ii for column jj              
                jvec = np.reshape(A[:,jj],(n,1)) - price
                ii = np.argmax(jvec)

                #Step 4: Assign row ii to column jj
                assign_mat[ii,:] = np.zeros((1,n))
                assign_mat[ii,jj] = 1.

                #Step 5: Compute new price
                jvec2 = np.sort(list(np.reshape(jvec,(1,n))))
                yj = jvec2[0][-1] - jvec2[0][-2]                
                price[ii] = price[ii] + yj + eps

    #Set the column order to achieve assignment
    row_indices = []
    for jj in range(n):
        x = np.nonzero(assign_mat[:,jj])       
        row_indices.append(int(x[0]))

    return row_indices


###############################################################################
# Tracklet Correlation and IOD
###############################################################################

def evaluate_tracklet_correlation(correlation_file, ra_lim, dec_lim):
    
    # Load correlation data
    pklFile = open(correlation_file, 'rb' )
    data = pickle.load( pklFile )
    correlation_dict = data[0]
    tracklet_dict = data[1]
    params_dict = data[2]
    truth_dict = data[3]
    pklFile.close()
        
    # # Reformulate correlation dict according to cases
    # case_dict = {}
    # N_true = 0
    # N_false = 0
    # for ii in correlation_dict:
    #     count = correlation_dict[ii]['count']
        
    #     # If this is a new case, retrieve and store values
    #     if count not in case_dict:
    #         case_dict[count] = {}
    #         case_dict[count]['corr_truth_list'] = [correlation_dict[ii]['corr_truth']]
    #         case_dict[count]['Xo_true'] = correlation_dict[ii]['Xo_true']
    #         case_dict[count]['Xo_list'] = [correlation_dict[ii]['Xo']]
    #         case_dict[count]['M_list'] = [correlation_dict[ii]['M']]
    #         case_dict[count]['resids_list'] = [correlation_dict[ii]['resids']]
    #         case_dict[count]['ra_rms_list'] = [correlation_dict[ii]['ra_rms']]
    #         case_dict[count]['dec_rms_list'] = [correlation_dict[ii]['dec_rms']]
            
    #         if correlation_dict[ii]['obj1_id'] == correlation_dict[ii]['obj2_id']:
    #             N_true += 1
    #         else:
    #             N_false += 1
                    
            
    #     # If multiple entries are from the same case, append to lists for 
    #     # later evaluation
    #     else:
    #         case_dict[count]['corr_truth_list'].append(correlation_dict[ii]['corr_truth'])
    #         case_dict[count]['Xo_list'].append(correlation_dict[ii]['Xo'])
    #         case_dict[count]['M_list'].append(correlation_dict[ii]['M'])
    #         case_dict[count]['resids_list'].append(correlation_dict[ii]['resids'])
    #         case_dict[count]['ra_rms_list'].append(correlation_dict[ii]['ra_rms'])
    #         case_dict[count]['dec_rms_list'].append(correlation_dict[ii]['dec_rms'])
           

    # Compute number of true positives, true negatives and correlation 
    # performance
    N_cases = len(correlation_dict.keys())
    N_true = 0
    N_false = 0
    N_truepos = 0
    N_trueneg = 0
    N_falsepos = 0
    N_falseneg = 0
    for case_id in correlation_dict:
        corr_truth_list = correlation_dict[case_id]['corr_truth_list']
        
        # True correlation status determined by object id
        obj1_id = correlation_dict[case_id]['obj1_id']
        obj2_id = correlation_dict[case_id]['obj2_id']
        
        if obj1_id == obj2_id:
            N_true += 1
            corr_truth = True
        else:
            N_false += 1
            corr_truth = False
        
        # If no IOD solution fit, correlation is estimated to be false
        if len(correlation_dict[case_id]['M_list']) == 0:
            corr_est = False
        
        # Otherwise, estimate correlation status based on residuals
        else:            
            ra_rms_list = correlation_dict[case_id]['ra_rms_list']
            dec_rms_list = correlation_dict[case_id]['dec_rms_list']
            resids_list = correlation_dict[case_id]['resids_list']
            
            rms_list = []
            for ii in range(len(resids_list)):
                resids_ii = resids_list[ii]
                resids_rms = np.sqrt(np.mean(np.sum(np.multiply(resids_ii, resids_ii), axis=0)))
                rms_list.append(resids_rms)
                
            min_ind = rms_list.index(min(rms_list))
            ra_min = ra_rms_list[min_ind]
            dec_min = dec_rms_list[min_ind]
            tot_min = rms_list[min_ind]*(1./arcsec2rad)
            
            if ra_min < ra_lim and dec_min < dec_lim:
                corr_est = True
            else:
                corr_est = False
        
        # Evaluate correlation status for true/false positives and negatives
        if corr_truth:
            if corr_est:
                N_truepos += 1
            else:
                N_falseneg += 1
                print('')
                print('False negative case id', case_id)
                
        else:
            if corr_est:
                N_falsepos += 1
            else:
                N_trueneg += 1
        
        # Check for fail conditions, if Gooding IOD didn't produce a solution
        # with the correct rev number M
        
    
    
    
    print('')
    print('Total Number of Correlations: ', N_cases)
    print('Number of True Correlations: ', N_true)
    print('Number of False Correlations: ', N_false)
    
    print('')
    print('True Pos:  %5.2f%% (%3d/%3d)' % ((N_truepos/N_true)*100., N_truepos, N_true))
    print('True Neg:  %5.2f%% (%3d/%3d)' % ((N_trueneg/N_false)*100., N_trueneg, N_false))
    print('False Pos: %5.2f%% (%3d/%3d)' % ((N_falsepos/N_false)*100., N_falsepos, N_false))
    print('False Neg: %5.2f%% (%3d/%3d)' % ((N_falseneg/N_true)*100., N_falseneg, N_true))
    
    
    
    
    
    
    
    
    
    return




if __name__ == '__main__':
    
    plt.close('all')
    
    test_pdf_contours()
    
    
    
    
    
    
    
    
    
    
    