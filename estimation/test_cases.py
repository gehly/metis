import numpy as np
from math import pi
from scipy.integrate import odeint
from datetime import datetime
import pickle
import os
import sys
from pathlib import Path

cwd = os.getcwd()
metis_dir = Path(cwd).parent
utilities_dir = os.path.join(metis_dir, 'utilities')
sensors_dir = os.path.join(metis_dir, 'sensors')
sys.path.append(metis_dir)
sys.path.append(utilities_dir)
sys.path.append(sensors_dir)

from utilities.tle_functions import propagate_TLE
from sensors.sensors import generate_sensor_file
from propagation.integration_functions import int_twobody


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
    pickle.dump( [spacecraftConfig, forcesCoeff, brdfCoeff], pklFile, -1 )
    pklFile.close()
    
    
    return


def generate_truth_file(true_params_file, truth_file, ndays, dt):


   
        
    
    pklFile = open( truth_file, 'wb' )
    pickle.dump( [truth_time, state, visibility], pklFile, -1 )
    pklFile.close()


    # Generate plots
    tdiff = [(proptime2datetime(sol_time[ii]) - proptime2datetime(sol_time[0]))
             for ii in range(len(sol_time))]
    t_hrs = [tdiff[ii].days*24. + tdiff[ii].seconds/3600. for ii in range(len(tdiff))]
    
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
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_hrs, visibility[:,0], 'k.')
    plt.ylabel('Apparent Mag')    
    plt.title('Measurements')
    plt.subplot(3,1,2)
    plt.plot(t_hrs, visibility[:,3], 'k.')
    plt.ylabel('Az [deg]')
    plt.subplot(3,1,3)
    plt.plot(t_hrs, visibility[:,4], 'k.')
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
    
    fname = 'leo_' + object_type + '_2018_07_05_truth_2.pkl'
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
#    generate_true_params_file(init_orbit_file, obj_id, object_type, true_params_file)
    
    
    # Generate truth trajectory and measurements file
    ndays = 3.
    dt = 10.
    
    generate_truth_file(true_params_file, sensor_file, truth_file, ndays, dt)
    
    # Generate noisy measurements file
#    generate_noisy_meas(truth_file, sensor_file, meas_file, ndays=3.)
    
    # Generate model parameters file
#    generate_model_params(true_params_file, model_params_file)
    
    
    
    # Run filter
#    run_filter(model_params_file, sensor_file, meas_file, filter_output_file,
#               alpha=1e-4)
    