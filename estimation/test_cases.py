import numpy as np
from math import pi
from scipy.integrate import odeint
from datetime import datetime
import pickle
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append('../')

from utilities.tle_functions import propagate_TLE
from sensors.sensors import generate_sensor_file
from propagation.integration_functions import int_twobody
from propagation.orbit_propagation import propagate_orbit
from utilities.eop_functions import get_celestrak_eop_alldata


def generate_init_orbit_file(obj_id, UTC, orbit_file):
    
    
    # Retrieve latest TLE info and propagate to desired start time using SGP4
    obj_id_list = [obj_id]
    UTC_list = [UTC]
    output_state = propagate_TLE(obj_id_list, UTC_list)

    print(output_state)
    
    # Save data
    pklFile = open( orbit_file, 'wb' )
    pickle.dump( [output_state], pklFile, -1 )
    pklFile.close()
    
    return



def parameter_setup_sphere(orbit_file, obj_id, mass, radius):

    # Load parameters
    pklFile = open(orbit_file, 'rb')
    data = pickle.load(pklFile)
    output_state = data[0]
    pklFile.close() 

    # Initialize spacecraft configuation
    UTC = output_state[obj_id]['UTC'][0]
    pos = output_state[obj_id]['r_GCRF'][0]
    vel = output_state[obj_id]['v_GCRF'][0]
    spacecraftConfig = {}
    spacecraftConfig['type'] = '3DoF' # 6DoF or 3DoF
    spacecraftConfig['mass'] = mass  # kg
    spacecraftConfig['radius'] = radius # m
    spacecraftConfig['time'] = UTC  # UTC in datetime
    spacecraftConfig['X'] = \
        np.array([pos[0], pos[1], pos[2], vel[0] ,vel[1] ,vel[2]])  # km, GCRF
    
    
    # Drag parameters
    dragCoeff = 2.2

    # Gravity field parameters
    order = 2
    degree = 0

    # SRP parameters
    emissivity = 0.05

    # Dynamic Model parameters
    forcesCoeff = {}
    forcesCoeff['order'] = order
    forcesCoeff['degree'] = degree
    forcesCoeff['dragCoeff'] = dragCoeff
    forcesCoeff['emissivity'] = emissivity

    # Measurement Model Parameters
    brdfCoeff = {}
    brdfCoeff['cSunVis'] = 455  # W/m^2
    brdfCoeff['d'] = 1.
    brdfCoeff['rho'] = 0.75
    brdfCoeff['s'] = 1 - brdfCoeff['d']
    brdfCoeff['Fo'] = 0.5
    
    return spacecraftConfig, forcesCoeff, brdfCoeff


def generate_true_params_file(orbit_file, obj_id, object_type, param_file):    
    
    eop_alldata = get_celestrak_eop_alldata()
    
    
    if object_type == 'sphere_lamr':
        
        # Parameter setup
        mass = 100.     # kg
        radius = 1./pi     # m,  gives area = 1 m^2
        
        spacecraftConfig, forcesCoeff, brdfCoeff = \
            parameter_setup_sphere(orbit_file, obj_id, mass, radius)
            
    if object_type == 'sphere_hamr':
        
        # Parameter setup
        mass = 1.     # kg
        radius = 1./pi     # m,  gives area = 1 m^2
        
        spacecraftConfig, forcesCoeff, brdfCoeff = \
            parameter_setup_sphere(orbit_file, obj_id, mass, radius)
            
    if object_type == 'cubesat_nadir':
        
        mass = 5.  # kg
        wE = 7.29211514670639e-5 * 180./pi
        attitude = np.array([0.0,0.0,0.0,0.0,-wE,0.0]) #roll-pitch-yaw and omega w.r.t orbit frame
        
        spacecraftConfig, forcesCoeff, brdfCoeff = \
            parameter_setup_cubesat(orbit_file, obj_id, mass, attitude)
            
    if object_type == 'cubesat_spin':
        
        mass = 5.  # kg
        #wE = 7.29211514670639e-5 * 180./pi
        attitude = np.array([0.0,0.0,0.0,0.0,5.,0.0]) #roll-pitch-yaw and omega w.r.t orbit frame
        
        spacecraftConfig, forcesCoeff, brdfCoeff = \
            parameter_setup_cubesat(orbit_file, obj_id, mass, attitude)
            
    if object_type == 'cubesat_tumble':
        
        mass = 5.  # kg
        wE = 7.29211514670639e-5 * 180./pi
        attitude = np.array([0.0,0.0,0.0,0.0,5.,5.]) #roll-pitch-yaw and omega w.r.t orbit frame
        
        spacecraftConfig, forcesCoeff, brdfCoeff = \
            parameter_setup_cubesat(orbit_file, obj_id, mass, attitude)
            
    if object_type == 'boxwing_nadir':
        
        mass = 500.  # kg
        wE = 7.29211514670639e-5 * 180./pi
        attitude = np.array([0.0,0.0,0.0,0.0,-wE,0.0]) #roll-pitch-yaw and omega w.r.t orbit frame
        
        spacecraftConfig, forcesCoeff, brdfCoeff = \
            parameter_setup_boxwing(orbit_file, obj_id, mass, attitude)
            
    if object_type == 'boxwing_spin':
        
        mass = 500.  # kg
        wE = 7.29211514670639e-5 * 180./pi
        attitude = np.array([0.0,0.0,0.0,0.0,5.,0.0]) #roll-pitch-yaw and omega w.r.t orbit frame
        
        spacecraftConfig, forcesCoeff, brdfCoeff = \
            parameter_setup_boxwing(orbit_file, obj_id, mass, attitude)
            
    if object_type == 'boxwing_tumble':
        
        mass = 500.  # kg
        wE = 7.29211514670639e-5 * 180./pi
        attitude = np.array([0.0,0.0,0.0,0.0,5.,5.]) #roll-pitch-yaw and omega w.r.t orbit frame
        
        spacecraftConfig, forcesCoeff, brdfCoeff = \
            parameter_setup_boxwing(orbit_file, obj_id, mass, attitude)
        
        
    # Save data    
    pklFile = open( param_file, 'wb' )
    pickle.dump( [spacecraftConfig, forcesCoeff, brdfCoeff, eop_alldata], pklFile, -1 )
    pklFile.close()
    
    
    return


def generate_truth_file(true_params_file, truth_file, ndays, dt):


    # Load parameters
    pklFile = open(true_params_file, 'rb')
    data = pickle.load(pklFile)
    spacecraftConfig = data[0]
    forcesCoeff = data[1]
    brdfCoeff = data[2]
    pklFile.close()
    
    intfcn = int_twobody
   
    UTC_times, state = propagate_orbit(intfcn, spacecraftConfig, forcesCoeff,
                                       brdfCoeff, ndays, dt)
    
    
    
    pklFile = open( truth_file, 'wb' )
    pickle.dump( [UTC_times, state], pklFile, -1 )
    pklFile.close()


    # Generate plots
    t_hrs = [(UTC_times[ii] - UTC_times[0]).total_seconds()/3600. for 
             ii in range(len(UTC_times))]
    
    plt.close('all')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_hrs, state[:,0], 'k.')
    plt.ylabel('X [km]')    
    plt.title('True Position')
    plt.subplot(3,1,2)
    plt.plot(t_hrs, state[:,1], 'k.')
    plt.ylabel('Y [km]')
    plt.subplot(3,1,3)
    plt.plot(t_hrs, state[:,2], 'k.')
    plt.ylabel('Z [km]')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_hrs, state[:,3], 'k.')
    plt.ylabel('dX [km/s]')    
    plt.title('True Velocity')
    plt.subplot(3,1,2)
    plt.plot(t_hrs, state[:,4], 'k.')
    plt.ylabel('dY [km/s]')
    plt.subplot(3,1,3)
    plt.plot(t_hrs, state[:,5], 'k.')
    plt.ylabel('dZ [km/s]')
    plt.xlabel('Time [hours]')
    
    
    if spacecraftConfig['type'] == '6DoF':
    
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(t_hrs, roll, 'k.')
        plt.ylim([-180, 180])
        plt.ylabel('Roll [deg]')    
        plt.title('True Attitude')
        plt.subplot(3,1,2)
        plt.plot(t_hrs, pitch, 'k.')
        plt.ylim([-90, 90])
        plt.ylabel('Pitch [deg]')
        plt.subplot(3,1,3)
        plt.plot(t_hrs, yaw, 'k.')
        plt.ylim([-180, 180])
        plt.ylabel('Yaw [deg]')
        plt.xlabel('Time [hours]')
        
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(t_hrs, omega1, 'k.')
        plt.ylim([-10, 10])
        plt.ylabel('Omega1 [deg/s]')    
        plt.title('True Angular Velocity')
        plt.subplot(3,1,2)
        plt.plot(t_hrs, omega2, 'k.')
        plt.ylim([-10, 10])
        plt.ylabel('Omega2 [deg/s]')
        plt.subplot(3,1,3)
        plt.plot(t_hrs, omega3, 'k.')
        plt.ylim([-10, 10])
        plt.ylabel('Omega3 [deg/s]')
        plt.xlabel('Time [hours]')
        
        plt.figure()
        plt.subplot(4,1,1)
        plt.plot(t_hrs, state[:,6], 'k.')
        plt.ylim([-1, 1])
        plt.ylabel('q0')    
        plt.title('Quaternion')
        plt.subplot(4,1,2)
        plt.plot(t_hrs, state[:,7], 'k.')
        plt.ylim([-1, 1])
        plt.ylabel('q1')
        plt.subplot(4,1,3)
        plt.plot(t_hrs, state[:,8], 'k.')
        plt.ylim([-1, 1])
        plt.ylabel('q2')
        plt.subplot(4,1,4)
        plt.plot(t_hrs, state[:,9], 'k.')
        plt.ylim([-1, 1])
        plt.ylabel('q3')
        plt.xlabel('Time [hours]')
    

    plt.show()
    

    return


def generate_noisy_meas(truth_file, sensor_file, meas_file, ndays=7.):
    
    # Load truth data
    pklFile = open(truth_file, 'rb')
    data = pickle.load(pklFile)
    UTC_times = data[0]
    state = data[1]
    pklFile.close()
    
    # Reduce data set
#    print(len(truth_time))
#    truth_time = [ti for ti in truth_time if ti < (truth_time[0]+timedelta(days=ndays))]
#    print(len(truth_time))
    
    # Load sensor data
    pklFile = open(sensor_file, 'rb')
    data = pickle.load(pklFile)
    sensor_dict = data[0]
    pklFile.close()
    
    # Retrieve sensor parameters
    sensor_id = list(sensor_dict.keys())[0]
    sensor = sensor_dict[sensor_id]
    mapp_lim = sensor['mapp_lim']
    az_lim = sensor['az_lim']
    el_lim = sensor['el_lim']
    sig_ra = sensor['sigma_dict']['ra']
    sig_dec = sensor['sigma_dict']['dec']
    sig_mapp = sensor['sigma_dict']['mapp']
    geodetic_latlonht = sensor['geodetic_latlonht']
    
    # Loop over times and check visiblity conditions
    for ii in range(len(UTC_times)):
        
        # Retrieve time and object location in ECI
        UTC = UTC_times[ii]
        Xi = state[ii,:]
        
        # Compute measurements
        meas = compute_measurement(Xi, sensor, UTC)
    
    
    
    
    # Find all times when visible
    visState = visibility[:,2]
    mapp = visibility[:,0]
    az = visibility[:,3]
    el = visibility[:,4]
    vis_inds = set(np.where([ti < (truth_time[0]+timedelta(days=ndays)) for ti in truth_time])[0])
    vis_inds.intersection_update(set(np.where(visState > 0.)[0]))
    vis_inds.intersection_update(set(np.where(az < az_lim[1]*180./pi)[0]))
    vis_inds.intersection_update(set(np.where(az > az_lim[0]*180./pi)[0]))
    vis_inds.intersection_update(set(np.where(el < el_lim[1]*180./pi)[0]))
    vis_inds.intersection_update(set(np.where(el > el_lim[0]*180./pi)[0]))
    vis_inds.intersection_update(set(np.where(mapp < mapp_lim)[0]))
    vis_inds = sorted(list(vis_inds))
    
    vis_time = [truth_time[ii] for ii in vis_inds]
    vis_az = az[vis_inds]
    vis_el = el[vis_inds]
    vis_mapp = mapp[vis_inds]

#    print(vis_time)

    # Compute passes
    start, stop, pass_length, meas_inds = compute_passes(vis_time, sensor)
    
    print(start)
    print(stop)
    print(pass_length)
    print(meas_inds)
    print('Total Number Meas: ', len(meas_inds))
    
    # Add noise
    meas_time = [vis_time[ii] for ii in meas_inds]
    az_noise = vis_az[meas_inds] + sig_az*np.random.randn(len(meas_inds),)
    el_noise = vis_el[meas_inds] + sig_el*np.random.randn(len(meas_inds),)
    mapp_noise = vis_mapp[meas_inds] + sig_mapp*np.random.randn(len(meas_inds),)
    
    # Save measurement file
    pklFile = open( meas_file, 'wb' )
    pickle.dump( [meas_time, az_noise, el_noise, mapp_noise], pklFile, -1 )
    pklFile.close()
    
    # Plot noise
    t_hrs = [(ti - truth_time[0]).total_seconds()/3600. for ti in meas_time]
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_hrs, mapp_noise - vis_mapp[meas_inds], 'k.')
    plt.ylabel('Apparent Mag')    
    plt.title('Measurements')
    plt.subplot(3,1,2)
    plt.plot(t_hrs, az_noise - vis_az[meas_inds], 'k.')
    plt.ylabel('Az [deg]')
    plt.subplot(3,1,3)
    plt.plot(t_hrs, el_noise - vis_el[meas_inds], 'k.')
    plt.ylabel('El [deg]')
    plt.xlabel('Time [hours]')
    
    
    
    plt.show()
    
    return




###############################################################################
# Stand-alone execution
###############################################################################


if __name__ == '__main__':
    
    # General parameters
    obj_id = 25042
    UTC = datetime(2018, 7, 5, 0, 0, 0) 
    object_type = 'sphere_lamr'
    
    # Data directory
    datadir = Path('C:/Users/Steve/Documents/data/multiple_model/'
                   '2018_07_07_leo')
    
    # Filenames
    init_orbit_file = datadir / 'iridium39_orbit_2018_07_05.pkl'
    sensor_file = datadir / 'sensors_falcon_params.pkl'
    
    fname = 'leo_' + object_type + '_2018_07_05_true_params.pkl'
    true_params_file = datadir / fname
    
    fname = 'leo_' + object_type + '_2018_07_05_truth.pkl'
    truth_file = datadir / fname
    
    fname = 'leo_' + object_type + '_2018_07_05_meas.pkl'
    meas_file = datadir / fname
    
    fname = 'leo_' + object_type + '_2018_07_05_model_params.pkl'
    model_params_file = datadir / fname
    
    fname = 'leo_' + object_type + '_2018_07_05_filter_output.pkl'
    filter_output_file = datadir / fname
    
    fname = 'leo_' + object_type + '_2018_07_05_filter_error.pkl'
    error_file = datadir / fname
    
    
    # Generate initial orbit file       
#    generate_init_orbit_file(obj_id, UTC, init_orbit_file)
    
    # Generate sensor file
#    generate_sensor_file(sensor_file)

    # Generate true params file
    generate_true_params_file(init_orbit_file, obj_id, object_type, true_params_file)
    
    
    # Generate truth trajectory and measurements file
    ndays = 7.
    dt = 10.
    
#    generate_truth_file(true_params_file, truth_file, ndays, dt)
    
    # Generate noisy measurements file
#    generate_noisy_meas(truth_file, sensor_file, meas_file, ndays=3.)
    
    # Generate model parameters file
#    generate_model_params(true_params_file, model_params_file)
    
    
    
    # Run filter
#    run_filter(model_params_file, sensor_file, meas_file, filter_output_file,
#               alpha=1e-4)
    