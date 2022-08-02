import numpy as np
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

import utilities.coordinate_systems as coord
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