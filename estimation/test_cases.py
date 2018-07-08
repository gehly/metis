import numpy as np
from math import pi
from scipy.integrate import odeint
from datetime import datetime, timedelta
import pickle
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

from skyfield.api import Loader, utc

sys.path.append('../')

from utilities.tle_functions import propagate_TLE
from utilities.eop_functions import get_celestrak_eop_alldata
from utilities.eop_functions import get_XYs2006_alldata
from utilities.eop_functions import get_eop_data
from utilities.coordinate_systems import latlonht2ecef
from utilities.coordinate_systems import gcrf2itrf
from sensors.sensors import generate_sensor_file
from sensors.brdf_models import lambertian_sphere
from sensors.measurements import compute_measurement
from sensors.measurements import ecef2azelrange_rad
from sensors.visibility import check_visibility
from propagation.integration_functions import int_twobody
from propagation.integration_functions import int_twobody_ukf
from propagation.orbit_propagation import propagate_orbit
from data_processing.errors import compute_ukf_errors
from data_processing.errors import plot_ukf_errors

from estimation import unscented_kalman_filter


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
    spacecraftConfig['radius'] = radius * 0.001 # km
    spacecraftConfig['time'] = UTC  # UTC in datetime
    spacecraftConfig['brdf_function'] = lambertian_sphere
    spacecraftConfig['intfcn'] = int_twobody
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
    surfaces = {}
    surfaces[0] = {}
    surfaces[0]['brdf_params'] = brdfCoeff
    
    return spacecraftConfig, forcesCoeff, surfaces


def generate_true_params_file(orbit_file, obj_id, object_type, param_file):    
    
    eop_alldata = get_celestrak_eop_alldata()
    XYs_df = get_XYs2006_alldata()
    
    if object_type == 'sphere_lamr':
        
        # Parameter setup
        mass = 100.     # kg
        radius = 1./np.sqrt(pi)     # m,  gives area = 1 m^2
        
        spacecraftConfig, forcesCoeff, surfaces = \
            parameter_setup_sphere(orbit_file, obj_id, mass, radius)
            
    
        
        
    # Save data    
    pklFile = open( param_file, 'wb' )
    pickle.dump( [spacecraftConfig, forcesCoeff, surfaces, eop_alldata,
                  XYs_df], pklFile, -1 )
    pklFile.close()
    
    
    return


def generate_truth_file(true_params_file, truth_file, ephemeris, ts, ndays, dt):


    # Load parameters
    pklFile = open(true_params_file, 'rb')
    data = pickle.load(pklFile)
    spacecraftConfig = data[0]
    forcesCoeff = data[1]
    surfaces = data[2]
    pklFile.close()
    
    UTC_times, state = propagate_orbit(spacecraftConfig, forcesCoeff,
                                       surfaces, ephemeris, ndays, dt)
    
    sec_array = [(UTC - UTC_times[0]).total_seconds() for UTC in UTC_times]
    
    UTC0 = ts.utc(UTC_times[0].replace(tzinfo=utc)).utc
    skyfield_times = ts.utc(UTC0[0], UTC0[1], UTC0[2],
                            UTC0[3], UTC0[4], sec_array)
    
    pklFile = open( truth_file, 'wb' )
    pickle.dump( [UTC_times, skyfield_times, state], pklFile, -1 )
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


def generate_noisy_meas(true_params_file, truth_file, sensor_file, meas_file,
                        ephemeris, ndays=1.):
    
    # Load parameters
    pklFile = open(true_params_file, 'rb')
    data = pickle.load(pklFile)
    spacecraftConfig = data[0]
    forcesCoeff = data[1]
    surfaces = data[2]
    eop_alldata = data[3]
    XYs_df = data[4]
    pklFile.close()
    
    # Load truth data
    pklFile = open(truth_file, 'rb')
    data = pickle.load(pklFile)
    UTC_times = data[0]
    skyfield_times = data[1]
    state = data[2]
    pklFile.close()
    
    # Reduce data set
    print(len(UTC_times))
    UTC_times = [ti for ti in UTC_times if ti < (UTC_times[0]+timedelta(days=ndays))]
#    skyfield_times = skyfield_times[0:len(UTC_times)]
    print(len(UTC_times))
    print(len(skyfield_times))
    
    # Load sensor data
    pklFile = open(sensor_file, 'rb')
    data = pickle.load(pklFile)
    sensor_dict = data[0]
    pklFile.close()
    
    # Retrieve sensor parameters
    sensor_id = list(sensor_dict.keys())[0]
    sensor = sensor_dict[sensor_id]
    meas_types = sensor['meas_types']
    sigs = []
    for mtype in meas_types:
        sigs.append(sensor['sigma_dict'][mtype])
    
    
    # Retrieve sun and moon positions for full timespan
    earth = ephemeris['earth']
    sun = ephemeris['sun']
    moon = ephemeris['moon']
    moon_gcrf_array = earth.at(skyfield_times).observe(moon).position.km
    sun_gcrf_array = earth.at(skyfield_times).observe(sun).position.km
    
    # Compute visible indicies    
    vis_inds = check_visibility(state, UTC_times, sun_gcrf_array,
                                moon_gcrf_array, sensor,
                                spacecraftConfig, surfaces, eop_alldata,
                                XYs_df)

    print(vis_inds)

    # Compute measurements
    meas = np.zeros((len(vis_inds), len(meas_types)))
    meas_true = np.zeros((len(vis_inds), len(meas_types)))
    meas_times = []
    row = 0
    for ii in vis_inds:
        
        # Retrieve time and current sun and object locations in ECI
        UTC = UTC_times[ii]
        Xi = state[ii,:]
        sun_gcrf = sun_gcrf_array[:,ii].reshape(3,1)
        
        # Compute measurements
        EOP_data = get_eop_data(eop_alldata, UTC)
        Yi = compute_measurement(Xi, sun_gcrf, sensor, spacecraftConfig,
                                 surfaces, UTC, EOP_data, meas_types, XYs_df)
        
        for jj in range(len(meas_types)):
            sig = sigs[jj]
            meas[row,jj] = float(Yi[jj]) + sig*np.random.randn()
            meas_true[row,jj] = float(Yi[jj])
        
        meas_times.append(UTC)
        row += 1
        
    # Save measurement file
    pklFile = open( meas_file, 'wb' )
    pickle.dump( [meas_times, meas], pklFile, -1 )
    pklFile.close()
    
    # Plot measurements
    t_hrs = [(ti - UTC_times[0]).total_seconds()/3600. for ti in meas_times]
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_hrs, meas_true[:,2], 'k.')
    plt.ylabel('Apparent Mag')    
    plt.title('Measurements')
    plt.subplot(3,1,2)
    plt.plot(t_hrs, meas_true[:,0]*180/pi, 'k.')
    plt.ylabel('RA [deg]')
    plt.subplot(3,1,3)
    plt.plot(t_hrs, meas_true[:,1]*180/pi, 'k.')
    plt.ylabel('DEC [deg]')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_hrs, meas[:,2] - meas_true[:,2], 'k.')
    plt.ylabel('Apparent Mag')    
    plt.title('Measurement Noise')
    plt.subplot(3,1,2)
    plt.plot(t_hrs, (meas[:,0] - meas_true[:,0])*206265, 'k.')
    plt.ylabel('RA [arcsec]')
    plt.subplot(3,1,3)
    plt.plot(t_hrs, (meas[:,1] - meas_true[:,1])*206265, 'k.')
    plt.ylabel('DEC [arcsec]')
    plt.xlabel('Time [hours]')
    
    
    plt.show()
    
    return



def generate_model_params(true_params_file, model_params_file):
    
    # Load parameters
    pklFile = open(true_params_file, 'rb')
    data = pickle.load(pklFile)
    spacecraftConfig = data[0]
    forcesCoeff = data[1]
    surfaces = data[2]
    eop_alldata = data[3]
    XYs_df = data[4]
    pklFile.close()
        
    
    # Spherical case
    if spacecraftConfig['type'] == '3DoF':
        
        # Integration function
        spacecraftConfig['intfcn'] = int_twobody_ukf
        
        # Initial covariance
        Po = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])  # km^2 and km^2/s^2
        spacecraftConfig['covar'] = Po
        
        # Perturb initial state
        pert_vect = np.multiply(np.sqrt(np.diag(Po)), np.random.randn(6,))
        print(pert_vect)
        print(spacecraftConfig['X'])
        spacecraftConfig['X'] += \
            pert_vect.reshape(spacecraftConfig['X'].shape)
        
        # Alter additional parameters as needed        
        forcesCoeff['Q'] = 1e-6*np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])
        
        
    
    else:
        mistake
        
        
    # Save data
    pklFile = open( model_params_file, 'wb' )
    pickle.dump( [spacecraftConfig, forcesCoeff, surfaces, eop_alldata,
                  XYs_df], pklFile, -1 )
    pklFile.close()
    
    
    
    return


def run_filter(model_params_file, sensor_file, meas_file, filter_output_file,
               ephemeris, ts, alpha=1e-4):
    
    filter_output = \
        unscented_kalman_filter(model_params_file, sensor_file, meas_file,
                                ephemeris, ts, alpha)
        
    
    # Save data
    pklFile = open( filter_output_file, 'wb' )
    pickle.dump( [filter_output], pklFile, -1 )
    pklFile.close()

    return

###############################################################################
# Stand-alone execution
###############################################################################


if __name__ == '__main__':
    
    # General parameters
    obj_id = 25042
    UTC = datetime(2018, 7, 8, 0, 0, 0) 
    object_type = 'sphere_lamr'
    
    # Data directory
    datadir = Path('C:/Users/Steve/Documents/data/multiple_model/'
                   '2018_07_08_leo')
    
    # Filenames
    init_orbit_file = datadir / 'iridium39_orbit_2018_07_08.pkl'
    sensor_file = datadir / 'sensors_falcon_params.pkl'
    
    fname = 'leo_' + object_type + '_2018_07_08_true_params.pkl'
    true_params_file = datadir / fname
    
    fname = 'leo_' + object_type + '_2018_07_08_truth.pkl'
    truth_file = datadir / fname
    
    fname = 'leo_' + object_type + '_2018_07_08_meas.pkl'
    meas_file = datadir / fname
    
    fname = 'leo_' + object_type + '_2018_07_08_model_params.pkl'
    model_params_file = datadir / fname
    
    fname = 'leo_' + object_type + '_2018_07_08_filter_output.pkl'
    filter_output_file = datadir / fname
    
    fname = 'leo_' + object_type + '_2018_07_08_filter_error.pkl'
    error_file = datadir / fname
    
    
    cwd = os.getcwd()
    metis_dir = cwd[0:-10]
    load = Loader(os.path.join(metis_dir, 'skyfield_data'))
    ephemeris = load('de430t.bsp')
    ts = load.timescale()
    
    
    # Generate initial orbit file       
#    generate_init_orbit_file(obj_id, UTC, init_orbit_file)
    
    # Generate sensor file
#    generate_sensor_file(sensor_file)

    # Generate true params file
#    generate_true_params_file(init_orbit_file, obj_id, object_type, true_params_file)
    
    
    # Generate truth trajectory and measurements file
    ndays = 7.
    dt = 10.
    
#    generate_truth_file(true_params_file, truth_file, ephemeris, ts, ndays, dt)
    
    # Generate noisy measurements file
#    generate_noisy_meas(true_params_file, truth_file, sensor_file, meas_file,
#                        ephemeris, ndays=3.)
    
    # Generate model parameters file
    generate_model_params(true_params_file, model_params_file)
    
    
    
    # Run filter
    run_filter(model_params_file, sensor_file, meas_file, filter_output_file,
               ephemeris, ts, alpha=1e-4)
    
    # Compute and plot errors
    compute_ukf_errors(filter_output_file, truth_file, error_file)
    plot_ukf_errors(error_file)
    
    
    
    
    
    
    
    
    
    
    
    

    