import numpy as np
from math import pi
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pickle
from datetime import datetime
from pathlib import Path
import sys

sys.path.append('../')

from utilities.attitude import quat2dcm, dcm2euler321, dcm2euler123
from utilities.coordinate_systems import ric2eci, lvlh2ric, eci2ric
from utilities.time_systems import jd2dt


def compute_ukf_errors(filter_output_file, truth_file, error_file):
    
    # Load filter output
    pklFile = open(filter_output_file, 'rb')
    data = pickle.load(pklFile)
    filter_output = data[0]
    pklFile.close()
    
    # Load truth data
    pklFile = open(truth_file, 'rb')
    data = pickle.load(pklFile)
    truth_time = data[0]
    skyfield_time = data[1]
    state = data[2]
    pklFile.close()
    
    # Times
    filter_time = filter_output['time']
#    truth_time = [proptime2datetime(ti) for ti in sol_time]
    
    # Error output
    L = len(filter_time)
    n = len(filter_output['X'][0])
    t_hrs = np.zeros(L,)
    Xerr = np.zeros((n,L))
    sigs = np.zeros((6,L))
    resids = np.zeros((3,L))
    resids[0,:] = [filter_output['resids'][ii][0] for ii in range(L)]
    resids[1,:] = [filter_output['resids'][ii][1] for ii in range(L)]
#    resids[2,:] = [filter_output['resids'][ii][2] for ii in range(L)]
    
    roll = []
    pitch = []
    yaw = []
    omega1 = []
    omega2 = []
    omega3 = []
    qerr = np.zeros((4,L))
    werr = np.zeros((3,L))
    DCM_OL = lvlh2ric()
    for ii in range(L):
        
        # Retrieve filter state
        ti = filter_time[ii]
        Xest = filter_output['X'][ii]
        P = filter_output['P'][ii]
        t_hrs[ii] = (ti - truth_time[0]).total_seconds()/3600.
        
        # Retrieve true state        
        ind = truth_time.index(ti)
        Xtrue = state[ind,:].reshape(n,1)
        
        # Compute errors
        Xerr[:,ii] = (Xest - Xtrue).flatten()
        sigs[:,ii] = np.sqrt(np.diag(P))
        
        if n == 13:
        
            pos = Xest[0:3].reshape(3,1)
            vel = Xest[3:6].reshape(3,1)
            q_BN = Xest[6:10].reshape(4,1)
            q_true = Xtrue[6:10].reshape(4,1)
            DCM_BN= quat2dcm(q_BN)
            DCM_NO = ric2eci(pos, vel)
            DCM_BO = np.dot(DCM_BN, DCM_NO)
            DCM_BL = np.dot(DCM_BO, DCM_OL)
            
            w_BN_B = Xest[10:13].reshape(3,1)
            w_true = Xtrue[10:13].reshape(3,1)
            w_ON_O = np.array([[0.], [0.], [np.linalg.norm(vel)/np.linalg.norm(pos)]])  # rad/s
            w_LO_O = np.zeros((3,1))
            w_LN_O = w_LO_O + w_ON_O
            w_LN_B = np.dot(DCM_BO, w_LN_O)
            w_BL_B = w_BN_B - w_LN_B
            
            r, p, y = dcm2euler123(DCM_BL)
            
            roll.append(r*180/pi)
            pitch.append(p*180/pi)
            yaw.append(y*180/pi)
            
            omega1.append(float(w_BL_B[0])*180/pi)
            omega2.append(float(w_BL_B[1])*180/pi)
            omega3.append(float(w_BL_B[2])*180/pi)
    
    if n == 13:    
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(t_hrs, roll, 'k.')
        plt.ylabel('Roll [deg]')    
        plt.title('True Attitude')
        plt.subplot(3,1,2)
        plt.plot(t_hrs, pitch, 'k.')
        plt.ylabel('Pitch [deg]')
        plt.subplot(3,1,3)
        plt.plot(t_hrs, yaw, 'k.')
        plt.ylabel('Yaw [deg]')
        plt.xlabel('Time [hours]')
        
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(t_hrs, omega1, 'k.')
        plt.ylabel('Omega1 [deg/s]')    
        plt.title('True Angular Velocity')
        plt.subplot(3,1,2)
        plt.plot(t_hrs, omega2, 'k.')
        plt.ylabel('Omega2 [deg/s]')
        plt.subplot(3,1,3)
        plt.plot(t_hrs, omega3, 'k.')
        plt.ylabel('Omega3 [deg/s]')
        plt.xlabel('Time [hours]')
        
        plt.figure()
        plt.subplot(4,1,1)
        plt.plot(t_hrs, Xerr[6,:], 'k.')
        plt.title('Quat Error')
        plt.subplot(4,1,2)
        plt.plot(t_hrs, Xerr[7,:], 'k.')
        plt.subplot(4,1,3)
        plt.plot(t_hrs, Xerr[8,:], 'k.')
        plt.subplot(4,1,4)
        plt.plot(t_hrs, Xerr[9,:], 'k.')
        plt.xlabel('Time [hours]')
        
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(t_hrs, Xerr[10,:], 'k.')
        plt.ylabel('wBN1 [deg/s]')    
        plt.title('True Angular Velocity')
        plt.subplot(3,1,2)
        plt.plot(t_hrs, Xerr[11,:], 'k.')
        plt.ylabel('wBN2 [deg/s]')
        plt.subplot(3,1,3)
        plt.plot(t_hrs, Xerr[12,:], 'k.')
        plt.ylabel('wBN3 [deg/s]')
        plt.xlabel('Time [hours]')
        
        
        plt.show()
        
    
    # Save data
    pklFile = open( error_file, 'wb' )
    pickle.dump( [t_hrs, Xerr, sigs, resids], pklFile, -1 )
    pklFile.close()

    
    return


def compute_mm_errors(filter_output_file, truth_file, error_file):
    
    # Load filter output
    pklFile = open(filter_output_file, 'rb')
    data = pickle.load(pklFile)
    filter_output = data[0]
    pklFile.close()
    
    # Load truth data
    pklFile = open(truth_file, 'rb')
    data = pickle.load(pklFile)
    truth_time = data[0]
    skyfield_time = data[1]
    state = data[2]
    pklFile.close()
    
    # Times
    filter_JD = sorted(filter_output.keys())
    filter_time = [jd2dt(JD) for JD in filter_JD]
#    truth_time = [proptime2datetime(ti) for ti in sol_time]
    
    extracted_model0 = filter_output[filter_JD[0]]['extracted_model']
    model_bank0 = filter_output[filter_JD[0]]['model_bank']
    model_id_list = sorted(list(model_bank0.keys()))
    
    # Error output
    L = len(filter_time)
    m = len(model_bank0)
    n = len(extracted_model0['est_means'])
    
    print(extracted_model0)
    
    t_hrs = np.zeros(L,)
    Xerr = np.zeros((n+3,L))
    sigs = np.zeros((n+3,L))
    model_weights = np.zeros((m,L))
    est_mode = np.zeros(L,)
    resids = np.zeros((2,L-1))
    for ii in range(L):
        
        # Retrieve filter state
        ti = filter_time[ii]
        extracted_model = filter_output[filter_JD[ii]]['extracted_model']
        Xest = extracted_model['est_means']
        P = extracted_model['est_covars']
        t_hrs[ii] = (ti - truth_time[0]).total_seconds()/3600.
        
        # Retrieve true state
        check_inds = [abs((true - ti).total_seconds()) for true in truth_time]
        ind = check_inds.index(min(check_inds))
        Xtrue = state[ind,0:6].reshape(n,1)
        
        # Compute errors
        Xerr[0:6,ii] = (Xest - Xtrue).flatten()
        sigs[0:6,ii] = np.sqrt(np.diag(P))
        
        # Compute RIC errors
        r_vect = Xtrue[0:3].reshape(3,1)
        v_vect = Xtrue[3:6].reshape(3,1)
        pos_err = Xerr[0:3,ii].reshape(3,1)
        P_pos = P[0:3,0:3]
        ric_err = eci2ric(r_vect, v_vect, pos_err)
        P_ric = eci2ric(r_vect, v_vect, P_pos)
        Xerr[6:9,ii] = ric_err.flatten()
        sigs[6:9,ii] = np.sqrt(np.diag(P_ric))
        
        # Compute model weights
        model_bank = filter_output[filter_JD[ii]]['model_bank']
        jj = 0
        wmax = 0.
        for model_id in model_id_list:
            wj = float(model_bank[model_id]['weight'])
            model_weights[jj,ii] = wj
            
#            print(jj)
#            print(model_id)
#            print(wj)
#            
#            if ii > 0:
#                mistake

            if ii == 0:
                if wj >= wmax:
                    est_mode[ii] = jj
                    wmax = wj
                    jj += 1
                
                continue
            
            resids_ii = model_bank[model_id]['resids']
            if wj >= wmax:
                resids[:,ii-1] = resids_ii.flatten()
                est_mode[ii] = jj
                wmax = wj
                
            jj += 1

    # Save data
    pklFile = open( error_file, 'wb' )
    pickle.dump([t_hrs, Xerr, sigs, resids, model_weights, model_id_list,
                 est_mode], pklFile, -1 )
    pklFile.close()

    
    return


def plot_ukf_errors(error_file):
    
#    plt.close('all')
    
    # Load filter output
    pklFile = open(error_file, 'rb')
    data = pickle.load(pklFile)
    t_hrs = data[0]
    Xerr = data[1]
    sigs = data[2]
    resids = data[3]
    pklFile.close()    
    
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_hrs, Xerr[0,:], 'k.')
    plt.plot(t_hrs, 3*sigs[0,:], 'k--')
    plt.plot(t_hrs, -3*sigs[0,:], 'k--')
    plt.ylabel('X Error [km]')    
    plt.title('Position Errors')
    plt.subplot(3,1,2)
    plt.plot(t_hrs, Xerr[1,:], 'k.')
    plt.plot(t_hrs, 3*sigs[1,:], 'k--')
    plt.plot(t_hrs, -3*sigs[1,:], 'k--')
    plt.ylabel('Y Error [km]')
    plt.subplot(3,1,3)
    plt.plot(t_hrs, Xerr[2,:], 'k.')
    plt.plot(t_hrs, 3*sigs[2,:], 'k--')
    plt.plot(t_hrs, -3*sigs[2,:], 'k--')
    plt.ylabel('Z Error [km]')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_hrs, Xerr[3,:], 'k.')
    plt.plot(t_hrs, 3*sigs[3,:], 'k--')
    plt.plot(t_hrs, -3*sigs[3,:], 'k--')
    plt.ylabel('dX Error [m/s]')    
    plt.title('Velocity Errors')
    plt.subplot(3,1,2)
    plt.plot(t_hrs, Xerr[4,:], 'k.')
    plt.plot(t_hrs, 3*sigs[4,:], 'k--')
    plt.plot(t_hrs, -3*sigs[4,:], 'k--')
    plt.ylabel('dY Error [m/s]')
    plt.subplot(3,1,3)
    plt.plot(t_hrs, Xerr[5,:], 'k.')
    plt.plot(t_hrs, 3*sigs[5,:], 'k--')
    plt.plot(t_hrs, -3*sigs[5,:], 'k--')
    plt.ylabel('dZ Error [m/s]')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_hrs, resids[0,:]*3600., 'k.')
    plt.ylabel('Right Asc [arcsec]')    
    plt.title('Post-fit Residuals')
    plt.subplot(3,1,2)
    plt.plot(t_hrs, resids[1,:]*3600., 'k.')
    plt.ylabel('Declination [arcsec]')
    plt.subplot(3,1,3)
    plt.plot(t_hrs, resids[2,:], 'k.')
    plt.ylabel('Apparent Mag')
    plt.xlabel('Time [hours]')
    
    plt.show()
    
    return


def plot_mm_errors(error_file):
    
    plt.close('all')
    
    # Load filter output
    pklFile = open(error_file, 'rb')
    data = pickle.load(pklFile)
    t_hrs = data[0]
    Xerr = data[1]
    sigs = data[2]
    resids = data[3]
    model_weights = data[4]
    model_id_list = data[5]
    est_mode = data[6]
    pklFile.close()    
    
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_hrs, Xerr[0,:], 'k.')
    plt.plot(t_hrs, 3*sigs[0,:], 'k--')
    plt.plot(t_hrs, -3*sigs[0,:], 'k--')
    plt.ylabel('X Error [km]')    
    plt.title('Position Errors')
    plt.subplot(3,1,2)
    plt.plot(t_hrs, Xerr[1,:], 'k.')
    plt.plot(t_hrs, 3*sigs[1,:], 'k--')
    plt.plot(t_hrs, -3*sigs[1,:], 'k--')
    plt.ylabel('Y Error [km]')
    plt.subplot(3,1,3)
    plt.plot(t_hrs, Xerr[2,:], 'k.')
    plt.plot(t_hrs, 3*sigs[2,:], 'k--')
    plt.plot(t_hrs, -3*sigs[2,:], 'k--')
    plt.ylabel('Z Error [km]')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_hrs, Xerr[3,:], 'k.')
    plt.plot(t_hrs, 3*sigs[3,:], 'k--')
    plt.plot(t_hrs, -3*sigs[3,:], 'k--')
    plt.ylabel('dX Error [m/s]')    
    plt.title('Velocity Errors')
    plt.subplot(3,1,2)
    plt.plot(t_hrs, Xerr[4,:], 'k.')
    plt.plot(t_hrs, 3*sigs[4,:], 'k--')
    plt.plot(t_hrs, -3*sigs[4,:], 'k--')
    plt.ylabel('dY Error [m/s]')
    plt.subplot(3,1,3)
    plt.plot(t_hrs, Xerr[5,:], 'k.')
    plt.plot(t_hrs, 3*sigs[5,:], 'k--')
    plt.plot(t_hrs, -3*sigs[5,:], 'k--')
    plt.ylabel('dZ Error [m/s]')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_hrs, Xerr[6,:], 'k.')
    plt.plot(t_hrs, 3*sigs[6,:], 'k--')
    plt.plot(t_hrs, -3*sigs[6,:], 'k--')
    plt.ylabel('Radial [km]')    
    plt.subplot(3,1,2)
    plt.plot(t_hrs, Xerr[7,:], 'k.')
    plt.plot(t_hrs, 3*sigs[7,:], 'k--')
    plt.plot(t_hrs, -3*sigs[7,:], 'k--')
    plt.ylabel('In-Track [km]')
    plt.subplot(3,1,3)
    plt.plot(t_hrs, Xerr[8,:], 'k.')
    plt.plot(t_hrs, 3*sigs[8,:], 'k--')
    plt.plot(t_hrs, -3*sigs[8,:], 'k--')
    plt.ylabel('Cross-Track [km]')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_hrs[1:], resids[0,:]*3600., 'k.')
    plt.ylabel('Right Asc [arcsec]')    
    plt.title('Post-fit Residuals')
    plt.subplot(3,1,2)
    plt.plot(t_hrs[1:], resids[1,:]*3600., 'k.')
    plt.ylabel('Declination [arcsec]')
    plt.subplot(3,1,3)
#    plt.plot(t_hrs[1:], resids[2,:], 'k.')
#    plt.ylabel('Apparent Mag')
    plt.xlabel('Time [hours]')
    
    
    legend = ['Low Noise', 'High Noise']
#              'Cube Nadir', 'Cube Spin', 'Cube Tumble', 'BW Nadir', 'BW Spin',
#              'BW Tumble']
    
    legend = sorted(legend)
    print(legend)
    print(model_id_list)
    
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]
    
    colors[2] = 'c'
    colors[3] = 'm'
    colors[5] = 'g'
    colors[6] = 'b'
    colors[9] = 'r'
    
    plt.figure()
    plt.subplot(2,1,1)
    n = len(model_id_list)
    for ii in range(n):
        c=colors[ii]
        plt.plot(t_hrs, model_weights[ii,:], 'o--', c=c, lw=3)
    
    
    plt.ylabel('Model Weights', fontsize=14)
    plt.ylim([-0.1, 1.1])
#    plt.xlim([-5, 36])
#    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35])
    plt.legend(legend, loc='best')
    

    plt.subplot(2,1,2)
    plt.plot(t_hrs, est_mode, 'ko', lw=3)
    plt.yticks([0, 1], legend)
    plt.ylim([-0.1, 1.1])
    plt.xlabel('Time [hours]', fontsize=14)
    
    
    
    plt.show()
    
    return


def plot_truth(truth_file, true_params_file):
    
    plt.close('all')
    
    # Load truth data
    pklFile = open(truth_file, 'rb')
    data = pickle.load(pklFile)
    truth_time = data[0]
    state = data[1]
    visibility = data[2]
    pklFile.close()
    
    # Load true params
    pklFile = open(true_params_file, 'rb')
    data = pickle.load(pklFile)
    spacecraftConfig = data[0]
    forcesCoeff = data[1]
    brdfCoeff = data[2]
    pklFile.close()
    
    
    # Times
    t_hrs = [(ti - truth_time[0]).total_seconds()/3600. for ti in truth_time]
    
    # Plot true states
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_hrs, state[:,0]*0.001, 'k.')
    plt.ylabel('X [km]')    
    plt.title('True Position')
    plt.subplot(3,1,2)
    plt.plot(t_hrs, state[:,1]*0.001, 'k.')
    plt.ylabel('Y [km]')
    plt.subplot(3,1,3)
    plt.plot(t_hrs, state[:,2]*0.001, 'k.')
    plt.ylabel('Z [km]')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_hrs, state[:,3]*0.001, 'k.')
    plt.ylabel('dX [km/s]')    
    plt.title('True Velocity')
    plt.subplot(3,1,2)
    plt.plot(t_hrs, state[:,4]*0.001, 'k.')
    plt.ylabel('dY [km/s]')
    plt.subplot(3,1,3)
    plt.plot(t_hrs, state[:,5]*0.001, 'k.')
    plt.ylabel('dZ [km/s]')
    plt.xlabel('Time [hours]')
    
    if state.shape[1] == 13:
        
        yaw = []
        pitch = []
        roll = []
        for ii in range(len(t_hrs)):
            qscalar = state[ii,6]
            q1 = state[ii,7]
            q2 = state[ii,8]
            q3 = state[ii,9]
            
            spacecraftState = State(spacecraftConfig)
            
            # Transform to roll-pitch-yaw
            q = np.array([[q1], [q2], [q3], [qscalar]])
            DCM_BN = quat2dcm(q)
            y, p, r = dcm2euler321(DCM_BN)
            
            yaw.append(y*180./pi)
            pitch.append(p*180./pi)
            roll.append(r*180./pi)
        
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(t_hrs, roll, 'k.')
        plt.ylabel('Roll [deg]')    
        plt.title('True Attitude')
        plt.subplot(3,1,2)
        plt.plot(t_hrs, pitch, 'k.')
        plt.ylabel('Pitch [deg]')
        plt.subplot(3,1,3)
        plt.plot(t_hrs, yaw, 'k.')
        plt.ylabel('Yaw [deg]')
        plt.xlabel('Time [hours]')
        
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(t_hrs, state[:,10], 'k.')
        plt.ylabel('Omega1 [deg/s]')    
        plt.title('True Angular Velocity')
        plt.subplot(3,1,2)
        plt.plot(t_hrs, state[:,11], 'k.')
        plt.ylabel('Omega2 [deg/s]')
        plt.subplot(3,1,3)
        plt.plot(t_hrs, state[:,12], 'k.')
        plt.ylabel('Omega3 [deg/s]')
        plt.xlabel('Time [hours]')
    
    
    # Plot measurements
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t_hrs, visibility[:,0], 'k.')
    plt.ylabel('Apparent Mag')    
    plt.title('Photo Measurements')
    plt.subplot(2,1,2)
    plt.plot(t_hrs, visibility[:,1], 'k.')
    plt.ylabel('Flux')
    plt.xlabel('Time [hours]')
    
    plt.figure()    
    plt.subplot(2,1,1)
    plt.plot(t_hrs, visibility[:,3], 'k.')
    plt.ylabel('Az [deg]')
    plt.title('Angle Measurements')
    plt.subplot(2,1,2)
    plt.plot(t_hrs, visibility[:,4], 'k.')
    plt.ylabel('El [deg]')
    plt.xlabel('Time [hours]')
    
    return


###############################################################################
# Stand-alone execution
###############################################################################


if __name__ == '__main__':
    
   # General parameters
    obj_id = 25042
    UTC = datetime(2018, 7, 5, 0, 0, 0) 
    object_type = 'sphere_med'
    
    # Data directory
    datadir = Path('C:/Users/Steve/Documents/data/multiple_model/'
                   '2018_07_12_leo/sphere_med')
    
    # Filenames
    init_orbit_file = datadir / 'iridium39_orbit_2018_07_05.pkl'
    sensor_file = datadir / 'sensors_falcon_params.pkl'
    
    fname = 'leo_' + object_type + '_2018_07_05_true_params.pkl'
    true_params_file = datadir / fname
    
    fname = 'leo_' + object_type + '_2018_07_12_truth.pkl'
    truth_file = datadir / fname
    
    fname = 'leo_' + object_type + '_2018_07_05_meas.pkl'
    meas_file = datadir / fname
    
    fname = 'leo_' + object_type + '_2018_07_05_model_params.pkl'
    model_params_file = datadir / fname
    
    fname = 'leo_sphere_med_mmae_2018_07_12_filter_output.pkl'
    filter_output_file = datadir / fname
    
    fname = 'leo_sphere_med_mmae_2018_07_12_filter_error.pkl'
    error_file = datadir / fname
    
    # Plot truth data
#    plot_truth(truth_file)
    
    # Compute and plot errors
#    compute_ukf_errors(filter_output_file, truth_file, error_file)
#    plot_ukf_errors(error_file)
    
    compute_mmae_errors(filter_output_file, truth_file, error_file)
    plot_mmae_errors(error_file)
    
    

























