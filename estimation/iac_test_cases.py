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

from utilities.constants import Re
from utilities.tle_functions import propagate_TLE
from utilities.tle_functions import launch2tle
from utilities.eop_functions import get_celestrak_eop_alldata
from utilities.eop_functions import get_XYs2006_alldata
from utilities.eop_functions import get_eop_data
from utilities.coordinate_systems import latlonht2ecef
from utilities.coordinate_systems import gcrf2itrf
from utilities.coordinate_systems import ric2eci
from utilities.coordinate_systems import eci2ric
from utilities.coordinate_systems import lvlh2ric
from utilities.coordinate_systems import ric2lvlh
from utilities.attitude import euler_angles
from utilities.attitude import quat2dcm
from utilities.attitude import dcm2quat
from utilities.attitude import dcm2euler123
from sensors.sensors import generate_sensor_file
from sensors.brdf_models import lambertian_sphere
from sensors.brdf_models import ashikhmin_premoze
from sensors.measurements import compute_measurement
from sensors.visibility_functions import check_visibility
from propagation.integration_functions import int_twobody
from propagation.integration_functions import int_twobody_ukf
from propagation.integration_functions import int_euler_dynamics_notorque
from propagation.integration_functions import int_twobody_6dof_notorque
from propagation.integration_functions import ode_twobody
from propagation.integration_functions import ode_twobody_ukf
from propagation.integration_functions import ode_twobody_j2_drag_srp
from propagation.integration_functions import ode_twobody_j2_drag_srp_ukf
from propagation.integration_functions import ode_twobody_6dof_notorque
from propagation.integration_functions import ode_twobody_6dof_notorque_ukf
from propagation.integration_functions import ode_twobody_j2_drag_srp_notorque
from propagation.integration_functions import ode_twobody_j2_drag_srp_notorque_ukf
from propagation.propagation_functions import propagate_orbit
from data_processing.errors import compute_ukf_errors
from data_processing.errors import plot_ukf_errors
from data_processing.errors import compute_mmae_errors
from data_processing.errors import plot_mmae_errors


from estimation import unscented_kalman_filter
from multiple_model import multiple_model_filter


def generate_init_orbit_file(obj_id, UTC, orbit_file, tle_dict={}):
    
    
    # Retrieve latest TLE info and propagate to desired start time using SGP4
    obj_id_list = [obj_id]
    UTC_list = [UTC]
    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict)

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
    pos = output_state[obj_id]['r_GCRF'][0].flatten()
    vel = output_state[obj_id]['v_GCRF'][0].flatten()
    spacecraftConfig = {}
    spacecraftConfig['type'] = '3DoF' # 6DoF or 3DoF
    spacecraftConfig['mass'] = mass  # kg
    spacecraftConfig['radius'] = radius * 0.001 # km
    spacecraftConfig['time'] = UTC  # UTC in datetime
    spacecraftConfig['brdf_function'] = lambertian_sphere
    spacecraftConfig['intfcn'] = ode_twobody_j2_drag_srp
    spacecraftConfig['integrator'] = 'dop853'
    spacecraftConfig['X'] = np.concatenate((pos, vel))  # km, GCRF
    
    
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
    forcesCoeff['solar_flux'] = 1367.  # w/m^2

    # Measurement Model Parameters
    brdfCoeff = {}
    brdfCoeff['cSunVis'] = 455. # W/m^2
    brdfCoeff['d'] = 1.
    brdfCoeff['rho'] = 0.75
    brdfCoeff['s'] = 1 - brdfCoeff['d']
    brdfCoeff['Fo'] = 0.5
    
    surfaces = {}
    surfaces[0] = {}
    surfaces[0]['brdf_params'] = brdfCoeff
    
    Rdiff = brdfCoeff['d']*brdfCoeff['rho']
    Rspec = brdfCoeff['s']*brdfCoeff['Fo']
    if Rdiff + Rspec > 1.:
        mistake
    
    return spacecraftConfig, forcesCoeff, surfaces


def parameter_setup_cubesat(orbit_file, obj_id, mass, attitude, dim):

    
    # Load parameters
    pklFile = open(orbit_file, 'rb')
    data = pickle.load(pklFile)
    output_state = data[0]
    pklFile.close()
    
    # Transform attitude to quaternion
    attitude *= pi/180.  # rad
    roll = float(attitude[0])
    pitch = float(attitude[1])
    yaw = float(attitude[2])
    
    sequence = [1,2,3]
    DCM_BL = euler_angles(sequence, roll, pitch, yaw)
    
    print(DCM_BL)
    
    roll, pitch, yaw = dcm2euler123(DCM_BL)
    
    print(roll*180/pi)
    print(pitch*180/pi)
    print(yaw*180/pi)
    
    w_BL_B = attitude[3:6].reshape(3,1)
    
    # Initialize spacecraft configuation
    UTC = output_state[obj_id]['UTC'][0]
    pos = output_state[obj_id]['r_GCRF'][0]
    vel = output_state[obj_id]['v_GCRF'][0]
    
    # Compute frame transforms
    DCM_LO = ric2lvlh()
    DCM_ON = eci2ric(pos, vel)
    DCM_NO = DCM_ON.T
    DCM_BN = np.dot(DCM_BL, np.dot(DCM_LO, DCM_ON))
    q_BN = dcm2quat(DCM_BN)
    
    w_ON_O = np.array([[0.], [0.], [np.linalg.norm(vel)/np.linalg.norm(pos)]])  # rad/s
    w_LO_O = np.zeros((3,1))
    w_LN_O = w_LO_O + w_ON_O
    w_LN_N = np.dot(DCM_NO, w_LN_O)
    
    w_BN_B = w_BL_B + np.dot(DCM_BN, w_LN_N)
    
    print(q_BN)
    print(w_BN_B)
    
    
    
    spacecraftConfig = {}
    spacecraftConfig['type'] = '3att' # 6DoF or 3DoF
    spacecraftConfig['mass'] = mass  # kg
    spacecraftConfig['time'] = UTC  # UTC in datetime
    spacecraftConfig['brdf_function'] = ashikhmin_premoze
    spacecraftConfig['intfcn'] = ode_twobody_j2_drag_srp_notorque
    spacecraftConfig['integrator'] = 'dop853'
    spacecraftConfig['X'] = \
        np.concatenate((pos.flatten(), vel.flatten(), q_BN.flatten(),
                        w_BN_B.flatten()))   # km, rad GCRF
        
    print(spacecraftConfig)
    
    # Moment of Inertia matrix, m^2 wrt body frame
    xdim = float(dim[0])
    ydim = float(dim[1])
    zdim = float(dim[2])
    spacecraftConfig['moi'] = np.array([[(ydim**2 + zdim**2),   0.,      0.],
                                        [ 0.,    (xdim**2 + zdim**2),    0.],
                                        [ 0.,      0.,  (xdim**2 + ydim**2)]]) * (mass/12.)
    
    # Convert to km^2
    spacecraftConfig['moi'] *= 1e-6

    # Location of center of mass wrt body frame, km
    spacecraftConfig['comOffset'] = np.array([0., 0., 0.]) * 0.001

    # Drag parameters
    dragCoeff = 3.0  # Mehta 2014 cylinder 3-1 ratio

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
    forcesCoeff['solar_flux'] = 1367.  # w/m^2

    # Measurement Model Parameters
    brdfCoeff = {}
    brdfCoeff['cSunVis'] = 455  #W/m^2
    brdfCoeff['d'] = 0.5
    brdfCoeff['rho'] = 0.75
    brdfCoeff['s'] = 1. - brdfCoeff['d']
    brdfCoeff['Fo'] = 0.75
    brdfCoeff['nu'] = 10
    brdfCoeff['nv'] = 10

    
    Rdiff = brdfCoeff['d']*brdfCoeff['rho']
    Rspec = brdfCoeff['s']*brdfCoeff['Fo']
    if Rdiff + Rspec > 1.:
        mistake
    
    
    surfaces = {}
    
    # Positive x-panel
    surfaces[0] = {}
    surfaces[0]['brdf_params'] = brdfCoeff
    surfaces[0]['area'] = ydim * zdim * 1e-6    # km^2
    surfaces[0]['center'] = np.array([[xdim/2.], [0.], [0.]]) * 0.001 # km
    surfaces[0]['norm_body_hat'] = np.array([[1.], [0.], [0.]])
    surfaces[0]['u_body_hat'] = np.array([[0.], [1.], [0.]])
    surfaces[0]['v_body_hat'] = np.array([[0.], [0.], [1.]])
    
    # Negative x-panel
    surfaces[1] = {}
    surfaces[1]['brdf_params'] = brdfCoeff
    surfaces[1]['area'] = ydim * zdim * 1e-6    # km^2
    surfaces[1]['center'] = np.array([[-xdim/2.], [0.], [0.]]) * 0.001 # km
    surfaces[1]['norm_body_hat'] = np.array([[-1.], [0.], [0.]])
    surfaces[1]['u_body_hat'] = np.array([[0.], [1.], [0.]])
    surfaces[1]['v_body_hat'] = np.array([[0.], [0.], [-1.]])
    
    # Positive y-panel
    surfaces[2] = {}
    surfaces[2]['brdf_params'] = brdfCoeff
    surfaces[2]['area'] = xdim * zdim * 1e-6    # km^2
    surfaces[2]['center'] = np.array([[0.], [ydim/2.], [0.]]) * 0.001 # km
    surfaces[2]['norm_body_hat'] = np.array([[0.], [1.], [0.]])
    surfaces[2]['u_body_hat'] = np.array([[-1.], [0.], [0.]])
    surfaces[2]['v_body_hat'] = np.array([[0.], [0.], [1.]])
    
    # Negative y-panel
    surfaces[3] = {}
    surfaces[3]['brdf_params'] = brdfCoeff
    surfaces[3]['area'] = xdim * zdim * 1e-6    # km^2
    surfaces[3]['center'] = np.array([[0.], [-ydim/2.], [0.]]) * 0.001 # km
    surfaces[3]['norm_body_hat'] = np.array([[0.], [-1.], [0.]])
    surfaces[3]['u_body_hat'] = np.array([[1.], [0.], [0.]])
    surfaces[3]['v_body_hat'] = np.array([[0.], [0.], [1.]])
    
    # Positive z-panel
    surfaces[4] = {}
    surfaces[4]['brdf_params'] = brdfCoeff
    surfaces[4]['area'] = xdim * ydim * 1e-6    # km^2
    surfaces[4]['center'] = np.array([[0.], [0.], [zdim/2.]]) * 0.001 # km
    surfaces[4]['norm_body_hat'] = np.array([[0.], [0.], [1.]])
    surfaces[4]['u_body_hat'] = np.array([[1.], [0.], [0.]])
    surfaces[4]['v_body_hat'] = np.array([[0.], [1.], [0.]])
    
    # Negative z-panel
    surfaces[5] = {}
    surfaces[5]['brdf_params'] = brdfCoeff
    surfaces[5]['area'] = xdim * ydim * 1e-6    # km^2
    surfaces[5]['center'] = np.array([[0.], [0.], [-zdim/2.]]) * 0.001 # km
    surfaces[5]['norm_body_hat'] = np.array([[0.], [0.], [-1.]])
    surfaces[5]['u_body_hat'] = np.array([[1.], [0.], [0.]])
    surfaces[5]['v_body_hat'] = np.array([[0.], [-1.], [0.]])


    print(surfaces)

    return spacecraftConfig, forcesCoeff, surfaces


def parameter_setup_boxwing(orbit_file, obj_id, mass, attitude, dim, mpanel,
                            paneldim):

    
    # Load parameters
    pklFile = open(orbit_file, 'rb')
    data = pickle.load(pklFile)
    output_state = data[0]
    pklFile.close()
    
    # Transform attitude to quaternion
    attitude *= pi/180.  # rad
    roll = float(attitude[0])
    pitch = float(attitude[1])
    yaw = float(attitude[2])
    
    sequence = [1,2,3]
    DCM_BL = euler_angles(sequence, roll, pitch, yaw)
    
    print(DCM_BL)
    
    roll, pitch, yaw = dcm2euler123(DCM_BL)
    
    print(roll*180/pi)
    print(pitch*180/pi)
    print(yaw*180/pi)
    
    w_BL_B = attitude[3:6].reshape(3,1)
    
    # Initialize spacecraft configuation
    UTC = output_state[obj_id]['UTC'][0]
    pos = output_state[obj_id]['r_GCRF'][0]
    vel = output_state[obj_id]['v_GCRF'][0]
    
    # Compute frame transforms
    DCM_LO = ric2lvlh()
    DCM_ON = eci2ric(pos, vel)
    DCM_NO = DCM_ON.T
    DCM_BN = np.dot(DCM_BL, np.dot(DCM_LO, DCM_ON))
    q_BN = dcm2quat(DCM_BN)
    
    w_ON_O = np.array([[0.], [0.], [np.linalg.norm(vel)/np.linalg.norm(pos)]])  # rad/s
    w_LO_O = np.zeros((3,1))
    w_LN_O = w_LO_O + w_ON_O
    w_LN_N = np.dot(DCM_NO, w_LN_O)
    
    w_BN_B = w_BL_B + np.dot(DCM_BN, w_LN_N)
    
    print(q_BN)
    print(w_BN_B)
    
    
    
    spacecraftConfig = {}
    spacecraftConfig['type'] = '3att' # 6DoF or 3DoF
    spacecraftConfig['mass'] = mass  # kg
    spacecraftConfig['time'] = UTC  # UTC in datetime
    spacecraftConfig['brdf_function'] = ashikhmin_premoze
    spacecraftConfig['intfcn'] = ode_twobody_j2_drag_srp_notorque
    spacecraftConfig['integrator'] = 'dop853'
    spacecraftConfig['X'] = \
        np.concatenate((pos.flatten(), vel.flatten(), q_BN.flatten(),
                        w_BN_B.flatten()))   # km, rad GCRF
        
    print(spacecraftConfig)
    
    # Moment of Inertia matrix, m^2 wrt body frame
    xdim = float(dim[0])
    ydim = float(dim[1])
    zdim = float(dim[2])
    
    xpanel = float(paneldim[0])
    ypanel = float(paneldim[1])
    zpanel = float(paneldim[2])
    
     # m^2 wrt body frame
    mbody = mass - 2*mpanel
    
    moi_body = np.array([[(ydim**2 + zdim**2),   0.,      0.],
                          [ 0.,    (xdim**2 + zdim**2),    0.],
                          [ 0.,      0.,  (xdim**2 + ydim**2)]]) * (mbody/12.)
    

    # Parallel axis theorem for solar panel moi 
    # (panels attach to +/- y body surfaces)
    rx = 0.5*ydim + 0.5*ypanel
    rz = 0.5*ydim + 0.5*ypanel
    
    moi_panel_x = (mpanel/12.) * (ypanel**2 + zpanel**2) + mpanel*(rx**2)
    moi_panel_y = (mpanel/12.) * (xpanel**2 + zpanel**2)
    moi_panel_z = (mpanel/12.) * (xpanel**2 + ypanel**2) + mpanel*(rz**2)
    moi_panels = np.array([[2*moi_panel_x,   0.,      0.],
                           [ 0.,    2*moi_panel_y,    0.],
                           [ 0.,      0.,  2*moi_panel_z]])
    
    spacecraftConfig['moi'] = moi_body + moi_panels
    
    
    
    
    # Convert to km^2
    spacecraftConfig['moi'] *= 1e-6

    # Location of center of mass wrt body frame, km
    spacecraftConfig['comOffset'] = np.array([0., 0., 0.]) * 0.001

    # Drag parameters
    dragCoeff = 3.0  # Mehta 2014 cylinder 3-1 ratio

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
    forcesCoeff['solar_flux'] = 1367. # W/m^2

    # Measurement Model Parameters
    brdfCoeff = {}
    brdfCoeff['cSunVis'] = 455.  #W/m^2
    brdfCoeff['d'] = 0.5
    brdfCoeff['rho'] = 0.75
    brdfCoeff['s'] = 1. - brdfCoeff['d']
    brdfCoeff['Fo'] = 0.75
    brdfCoeff['nu'] = 10
    brdfCoeff['nv'] = 10

    
    Rdiff = brdfCoeff['d']*brdfCoeff['rho']
    Rspec = brdfCoeff['s']*brdfCoeff['Fo']
    if Rdiff + Rspec > 1.:
        mistake
    
    
    surfaces = {}
    
    # Positive x-panel
    surfaces[0] = {}
    surfaces[0]['brdf_params'] = brdfCoeff
    surfaces[0]['area'] = ydim * zdim * 1e-6    # km^2
    surfaces[0]['center'] = np.array([[xdim/2.], [0.], [0.]]) * 0.001 # km
    surfaces[0]['norm_body_hat'] = np.array([[1.], [0.], [0.]])
    surfaces[0]['u_body_hat'] = np.array([[0.], [1.], [0.]])
    surfaces[0]['v_body_hat'] = np.array([[0.], [0.], [1.]])
    
    # Negative x-panel
    surfaces[1] = {}
    surfaces[1]['brdf_params'] = brdfCoeff
    surfaces[1]['area'] = ydim * zdim * 1e-6    # km^2
    surfaces[1]['center'] = np.array([[-xdim/2.], [0.], [0.]]) * 0.001 # km
    surfaces[1]['norm_body_hat'] = np.array([[-1.], [0.], [0.]])
    surfaces[1]['u_body_hat'] = np.array([[0.], [1.], [0.]])
    surfaces[1]['v_body_hat'] = np.array([[0.], [0.], [-1.]])
    
    # Positive y-panel
    surfaces[2] = {}
    surfaces[2]['brdf_params'] = brdfCoeff
    surfaces[2]['area'] = xdim * zdim * 1e-6    # km^2
    surfaces[2]['center'] = np.array([[0.], [ydim/2.], [0.]]) * 0.001 # km
    surfaces[2]['norm_body_hat'] = np.array([[0.], [1.], [0.]])
    surfaces[2]['u_body_hat'] = np.array([[-1.], [0.], [0.]])
    surfaces[2]['v_body_hat'] = np.array([[0.], [0.], [1.]])
    
    # Negative y-panel
    surfaces[3] = {}
    surfaces[3]['brdf_params'] = brdfCoeff
    surfaces[3]['area'] = xdim * zdim * 1e-6    # km^2
    surfaces[3]['center'] = np.array([[0.], [-ydim/2.], [0.]]) * 0.001 # km
    surfaces[3]['norm_body_hat'] = np.array([[0.], [-1.], [0.]])
    surfaces[3]['u_body_hat'] = np.array([[1.], [0.], [0.]])
    surfaces[3]['v_body_hat'] = np.array([[0.], [0.], [1.]])
    
    # Positive z-panel
    surfaces[4] = {}
    surfaces[4]['brdf_params'] = brdfCoeff
    surfaces[4]['area'] = xdim * ydim * 1e-6    # km^2
    surfaces[4]['center'] = np.array([[0.], [0.], [zdim/2.]]) * 0.001 # km
    surfaces[4]['norm_body_hat'] = np.array([[0.], [0.], [1.]])
    surfaces[4]['u_body_hat'] = np.array([[1.], [0.], [0.]])
    surfaces[4]['v_body_hat'] = np.array([[0.], [1.], [0.]])
    
    # Negative z-panel
    surfaces[5] = {}
    surfaces[5]['brdf_params'] = brdfCoeff
    surfaces[5]['area'] = xdim * ydim * 1e-6    # km^2
    surfaces[5]['center'] = np.array([[0.], [0.], [-zdim/2.]]) * 0.001 # km
    surfaces[5]['norm_body_hat'] = np.array([[0.], [0.], [-1.]])
    surfaces[5]['u_body_hat'] = np.array([[1.], [0.], [0.]])
    surfaces[5]['v_body_hat'] = np.array([[0.], [-1.], [0.]])
    
    # Solar Panels
    # +Y +X panel
    surfaces[6] = {}
    surfaces[6]['brdf_params'] = brdfCoeff
    surfaces[6]['area'] = ypanel * zpanel * 1e-6    # km^2
    surfaces[6]['center'] = np.array([[xpanel/2.], [(ydim+ypanel)/2.], [0.]]) * 0.001 # km
    surfaces[6]['norm_body_hat'] = np.array([[1.], [0.], [0.]])
    surfaces[6]['u_body_hat'] = np.array([[0.], [1.], [0.]])
    surfaces[6]['v_body_hat'] = np.array([[0.], [0.], [1.]])
    
    # +Y -X panel
    surfaces[7] = {}
    surfaces[7]['brdf_params'] = brdfCoeff
    surfaces[7]['area'] = ypanel * zpanel * 1e-6    # km^2
    surfaces[7]['center'] = np.array([[-xpanel/2.], [(ydim+ypanel)/2.], [0.]]) * 0.001 # km
    surfaces[7]['norm_body_hat'] = np.array([[-1.], [0.], [0.]])
    surfaces[7]['u_body_hat'] = np.array([[0.], [1.], [0.]])
    surfaces[7]['v_body_hat'] = np.array([[0.], [0.], [-1.]])
    
    # -Y +X panel
    surfaces[8] = {}
    surfaces[8]['brdf_params'] = brdfCoeff
    surfaces[8]['area'] = ypanel * zpanel * 1e-6    # km^2
    surfaces[8]['center'] = np.array([[xpanel/2.], [-(ydim+ypanel)/2.], [0.]]) * 0.001 # km
    surfaces[8]['norm_body_hat'] = np.array([[1.], [0.], [0.]])
    surfaces[8]['u_body_hat'] = np.array([[0.], [1.], [0.]])
    surfaces[8]['v_body_hat'] = np.array([[0.], [0.], [1.]])
    
    # -Y -X panel
    surfaces[9] = {}
    surfaces[9]['brdf_params'] = brdfCoeff
    surfaces[9]['area'] = ypanel * zpanel * 1e-6    # km^2
    surfaces[9]['center'] = np.array([[-xpanel/2.], [-(ydim+ypanel)/2.], [0.]]) * 0.001 # km
    surfaces[9]['norm_body_hat'] = np.array([[-1.], [0.], [0.]])
    surfaces[9]['u_body_hat'] = np.array([[0.], [1.], [0.]])
    surfaces[9]['v_body_hat'] = np.array([[0.], [0.], [-1.]])
    
    

    print(surfaces)

    return spacecraftConfig, forcesCoeff, surfaces


def generate_true_params_file(orbit_file, obj_id, object_type, param_file):    
    
    eop_alldata = get_celestrak_eop_alldata()
    XYs_df = get_XYs2006_alldata()
    
            
    if object_type == 'sphere_low_drag':
        
        # Parameter setup
        mass = 5.     # kg
        radius = 0.1/np.sqrt(pi)    # m,  gives area = 0.01 m^2
        
        spacecraftConfig, forcesCoeff, surfaces = \
            parameter_setup_sphere(orbit_file, obj_id, mass, radius)
            
    if object_type == 'sphere_high_drag':
        
        # Parameter setup
        mass = 5.     # kg
        radius = 0.3/np.sqrt(pi)    # m,  gives area = 0.09 m^2
        
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
    eop_alldata = data[3]
    pklFile.close()
    
    UTC_times, state = propagate_orbit(spacecraftConfig, forcesCoeff,
                                       surfaces, eop_alldata, ndays, dt)
    
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
    
    
    if spacecraftConfig['type'] == '6DoF' or spacecraftConfig['type'] == '3att':
        
        roll = []
        pitch = []
        yaw = []
        omega1 = []
        omega2 = []
        omega3 = []
        DCM_OL = lvlh2ric()
        for ii in range(len(t_hrs)):
            
            pos = state[ii,0:3].reshape(3,1)
            vel = state[ii,3:6].reshape(3,1)
            q_BN = state[ii,6:10].reshape(4,1)
            DCM_BN= quat2dcm(q_BN)
            DCM_NO = ric2eci(pos, vel)
            DCM_BO = np.dot(DCM_BN, DCM_NO)
            DCM_BL = np.dot(DCM_BO, DCM_OL)
            
            w_BN_B = state[ii,10:13].reshape(3,1)
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
        plt.ylim([-2, 2])
        plt.ylabel('q1')    
        plt.title('Quaternion')
        plt.subplot(4,1,2)
        plt.plot(t_hrs, state[:,7], 'k.')
        plt.ylim([-2, 2])
        plt.ylabel('q2')
        plt.subplot(4,1,3)
        plt.plot(t_hrs, state[:,8], 'k.')
        plt.ylim([-2, 2])
        plt.ylabel('q3')
        plt.subplot(4,1,4)
        plt.plot(t_hrs, state[:,9], 'k.')
        plt.ylim([-2, 2])
        plt.ylabel('q4')
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
    
    tf_hrs = (UTC_times[-1] - UTC_times[0]).total_seconds()/3600.
    
    
    # Load sensor data
    pklFile = open(sensor_file, 'rb')
    data = pickle.load(pklFile)
    sensor_dict = data[0]
    pklFile.close()    
    
    # Retrieve sun and moon positions for full timespan
    earth = ephemeris['earth']
    sun = ephemeris['sun']
    moon = ephemeris['moon']
    moon_gcrf_array = earth.at(skyfield_times).observe(moon).position.km
    sun_gcrf_array = earth.at(skyfield_times).observe(sun).position.km
    
    # Retrieve sensor parameters
    meas_dict = {}
    sensor_id_list = sorted(list(sensor_dict.keys()))
    sensor_access = []
    for sensor_id in sensor_id_list:
        sensor = sensor_dict[sensor_id]
        meas_types = sensor['meas_types']
        sigs = []
        for mtype in meas_types:
            sigs.append(sensor['sigma_dict'][mtype])
    
        # Compute visible indicies    
        vis_inds = check_visibility(state, UTC_times, sun_gcrf_array,
                                    moon_gcrf_array, sensor,
                                    spacecraftConfig, surfaces, eop_alldata,
                                    XYs_df)
    
        print(vis_inds)
    
        t_hrs = np.asarray([(UTC_times[ii] - UTC_times[0]).total_seconds()/3600. for ii in vis_inds])
        sensor_access.append(t_hrs)
        
        # Compute measurements
#        meas = np.zeros((len(vis_inds), len(meas_types)))
#        meas_true = np.zeros((len(vis_inds), len(meas_types)))
#        meas_times = []
        for ii in vis_inds:

            # Retrieve time and current sun and object locations in ECI
            UTC = UTC_times[ii]
            Xi = state[ii,:]
            sun_gcrf = sun_gcrf_array[:,ii].reshape(3,1)
                        
            if UTC not in meas_dict:
                meas_dict[UTC] = {}

            # Compute measurements
            EOP_data = get_eop_data(eop_alldata, UTC)
            Yi = compute_measurement(Xi, sun_gcrf, sensor, spacecraftConfig,
                                     surfaces, UTC, EOP_data, meas_types, XYs_df)
            
            meas = []
            for jj in range(len(meas_types)):
                sig = sigs[jj]
                meas.append(float(Yi[jj]) + sig*np.random.randn())

            meas_dict[UTC][sensor_id] = np.reshape(meas, (len(meas),1))

      
        
    print(meas_dict)
    
    # Save measurement file
    pklFile = open( meas_file, 'wb' )
    pickle.dump( [meas_dict], pklFile, -1 )
    pklFile.close()
    
    plt.figure()
    for ii in range(len(sensor_id_list)):
        plt.plot(sensor_access[ii], np.ones(len(sensor_access[ii]))*(ii+1.), 'ks')
        
    plt.xlim([0., tf_hrs])
    plt.xlabel('Time [hours]')
    plt.yticks([1, 2, 3], ['Arequipa', 'Stromlo', 'Yarragadee'])
    
    print(sensor_dict)
    print(sensor_id_list)
    print(sensor_access)
    
    
    # Plot measurements
#    t_hrs = [(ti - UTC_times[0]).total_seconds()/3600. for ti in UTC_times]
    
#    plt.figure()
#    plt.subplot(3,1,1)
#    if 'mapp' in meas_types:
#        plt.plot(t_hrs, meas_true[:,2], 'k.')
#        plt.ylabel('Apparent Mag')    
#    plt.title('Measurements')
#    plt.subplot(3,1,2)
#    plt.plot(t_hrs, meas_true[:,0]*180/pi, 'k.')
#    plt.ylabel('RA [deg]')
#    plt.subplot(3,1,3)
#    plt.plot(t_hrs, meas_true[:,1]*180/pi, 'k.')
#    plt.ylabel('DEC [deg]')
#    plt.xlabel('Time [hours]')
#    
#    plt.figure()
#    plt.subplot(3,1,1)
#    if 'mapp' in meas_types:
#        plt.plot(t_hrs, meas[:,2] - meas_true[:,2], 'k.')
#        plt.ylabel('Apparent Mag')    
#    plt.title('Measurement Noise')
#    plt.subplot(3,1,2)
#    plt.plot(t_hrs, (meas[:,0] - meas_true[:,0])*206265, 'k.')
#    plt.ylabel('RA [arcsec]')
#    plt.subplot(3,1,3)
#    plt.plot(t_hrs, (meas[:,1] - meas_true[:,1])*206265, 'k.')
#    plt.ylabel('DEC [arcsec]')
#    plt.xlabel('Time [hours]')
    
    
    plt.show()
    
    return



def generate_ukf_params(true_params_file, model_params_file):
    
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
        spacecraftConfig['intfcn'] = ode_twobody_j2_drag_srp_ukf
        
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
        forcesCoeff['Q'] = np.eye(3) * 1e-12
        
    elif spacecraftConfig['type'] == '3att':
        
        # Integration function
        spacecraftConfig['intfcn'] = ode_twobody_j2_drag_srp_notorque_ukf
        
        # Initial covariance
        Po = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])  # km^2 and km^2/s^2
        spacecraftConfig['covar'] = Po
        
        # Perturb initial state
        pert_vect = np.array([ 3.77146717e-01, -4.92942379e-01,  1.29437936e+00, -1.70821160e-03,
                          5.60305375e-04,  1.15733694e-03])
#        pert_vect = np.multiply(np.sqrt(np.diag(Po)), np.random.randn(6,))
        print(pert_vect)
        print(spacecraftConfig['X'])
        spacecraftConfig['X'][0:6] += \
            pert_vect.flatten()
        
        print(spacecraftConfig['X'])
        
        
        # Alter additional parameters as needed        
        forcesCoeff['Q'] = np.eye(3) * 1e-10
        

    # Non-spherical case
    else:
        
        # Integration function
        spacecraftConfig['intfcn'] = ode_twobody_6dof_notorque_ukf
        
        # Initial covariance
        ang = (1.*pi/180.)**2
        angvel = (0.001*pi/180.)**2
        Po = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6,
                      ang, ang, ang, angvel, angvel, angvel])  # km, km/s, rad, rad/s
        spacecraftConfig['covar'] = Po
        
        # Perturb initial state
        pert_vect = np.multiply(np.sqrt(np.diag(Po[0:6, 0:6])), np.random.randn(6,))
        pert_vect = np.append(pert_vect, np.zeros(7,))
        print(pert_vect)
        print(spacecraftConfig['X'])
        spacecraftConfig['X'] += \
            pert_vect.reshape(spacecraftConfig['X'].shape)
        
        # Alter additional parameters as needed
        forcesCoeff['Q'] = np.eye(3) * 1e-10
        forcesCoeff['sig_u'] = 1e-12
        forcesCoeff['sig_v'] = 1e-12
        
        
    # Save data
    pklFile = open( model_params_file, 'wb' )
    pickle.dump( [spacecraftConfig, forcesCoeff, surfaces, eop_alldata,
                  XYs_df], pklFile, -1 )
    pklFile.close()
    
    
    
    return


def generate_mmae_params(true_params_file, orbit_file, model_params_file):
    
    # Load parameters
    pklFile = open(true_params_file, 'rb')
    data = pickle.load(pklFile)
    spacecraftConfig = data[0]
    forcesCoeff = data[1]
    surfaces = data[2]
    eop_alldata = data[3]
    XYs_df = data[4]
    pklFile.close()
    
    
    # Initial covariance
    Po = np.diag([1., 1., 1., 1e-6, 1e-6, 1e-6])  # km^2 and km^2/s^2
    
    
    # Perturb initial state
#    pert_vect = np.multiply(np.sqrt(np.diag(Po)), np.random.randn(6,))
#    print(pert_vect)
    
    pert_vect = np.array([ 3.77146717e-01, -4.92942379e-01,  1.29437936e+00, -1.70821160e-03,
                          5.60305375e-04,  1.15733694e-03])
        
    
    # Parameter setup
    model_bank = {}
    
    ###########################################################################
    # Sphere LAMR Big
    ###########################################################################
    
    model_id = 'sphere_lamr_big'
    model_bank[model_id] = {}
    model_bank[model_id]['weight'] = 0.1
    
    # Basic Parameters
    mass = 100.     # kg
    radius = 1./np.sqrt(pi)    # m,  gives area = 1 m^2
    
    spacecraftConfig, forcesCoeff, surfaces = \
        parameter_setup_sphere(orbit_file, obj_id, mass, radius)

    # Integration function
    spacecraftConfig['intfcn'] = ode_twobody_j2_drag_srp_ukf
    
    # Covariance, initial state, process noise
    spacecraftConfig['covar'] = Po
    
    print(spacecraftConfig['X'])
    spacecraftConfig['X'] += \
        pert_vect.reshape(spacecraftConfig['X'].shape)
    
    # Alter additional parameters as needed        
    forcesCoeff['Q'] = np.eye(3) * 1e-12
    
    model_bank[model_id]['spacecraftConfig'] = spacecraftConfig
    model_bank[model_id]['forcesCoeff'] = forcesCoeff
    model_bank[model_id]['surfaces'] = surfaces
    
    ###########################################################################
    # Sphere MAMR Big
    ###########################################################################
    
    model_id = 'sphere_mamr_big'
    model_bank[model_id] = {}
    model_bank[model_id]['weight'] = 0.1    
    
    # Basic Parameters
    mass = 10.     # kg
    radius = 1./np.sqrt(pi)    # m,  gives area = 1 m^2
    
    spacecraftConfig, forcesCoeff, surfaces = \
        parameter_setup_sphere(orbit_file, obj_id, mass, radius)

    # Integration function
    spacecraftConfig['intfcn'] = ode_twobody_j2_drag_srp_ukf
    
    # Covariance, initial state, process noise
    spacecraftConfig['covar'] = Po
    
    print(spacecraftConfig['X'])
    spacecraftConfig['X'] += \
        pert_vect.reshape(spacecraftConfig['X'].shape)
    
    # Alter additional parameters as needed        
    forcesCoeff['Q'] = np.eye(3) * 1e-12
    
    model_bank[model_id]['spacecraftConfig'] = spacecraftConfig
    model_bank[model_id]['forcesCoeff'] = forcesCoeff
    model_bank[model_id]['surfaces'] = surfaces
    
    ###########################################################################
    # Sphere LAMR Small
    ###########################################################################
#    
    model_id = 'sphere_lamr_small'
    model_bank[model_id] = {}
    model_bank[model_id]['weight'] = 0.1  
    
    # Basic Parameters
    mass = 1.     # kg
    radius = 0.1/np.sqrt(pi)    # m,  gives area = 0.01 m^2
    
    spacecraftConfig, forcesCoeff, surfaces = \
        parameter_setup_sphere(orbit_file, obj_id, mass, radius)

    # Integration function
    spacecraftConfig['intfcn'] = ode_twobody_j2_drag_srp_ukf
    
    # Covariance, initial state, process noise
    spacecraftConfig['covar'] = Po
    
    print(spacecraftConfig['X'])
    spacecraftConfig['X'] += \
        pert_vect.reshape(spacecraftConfig['X'].shape)
    
    # Alter additional parameters as needed        
    forcesCoeff['Q'] = np.eye(3) * 1e-12
    
    model_bank[model_id]['spacecraftConfig'] = spacecraftConfig
    model_bank[model_id]['forcesCoeff'] = forcesCoeff
    model_bank[model_id]['surfaces'] = surfaces
    
    ###########################################################################
    # Sphere MAMR Small
    ###########################################################################
    
    model_id = 'sphere_mamr_small'
    model_bank[model_id] = {}
    model_bank[model_id]['weight'] = 0.1  
    
    # Basic Parameters
    # Parameter setup
    mass = 0.1     # kg
    radius = 0.1/np.sqrt(pi)    # m,  gives area = 0.01 m^2
    
    spacecraftConfig, forcesCoeff, surfaces = \
        parameter_setup_sphere(orbit_file, obj_id, mass, radius)

    # Integration function
    spacecraftConfig['intfcn'] = ode_twobody_j2_drag_srp_ukf
    
    # Covariance, initial state, process noise
    spacecraftConfig['covar'] = Po
    
    print(spacecraftConfig['X'])
    spacecraftConfig['X'] += \
        pert_vect.reshape(spacecraftConfig['X'].shape)
    
    # Alter additional parameters as needed        
    forcesCoeff['Q'] = np.eye(3) * 1e-12
    
    model_bank[model_id]['spacecraftConfig'] = spacecraftConfig
    model_bank[model_id]['forcesCoeff'] = forcesCoeff
    model_bank[model_id]['surfaces'] = surfaces
    
#    ###########################################################################
#    # Cubesat Nadir
#    ###########################################################################    
#    
#    model_id = 'cubesat_nadir'
#    model_bank[model_id] = {}
#    model_bank[model_id]['weight'] = 0.1
#    
#    mass = 1.  # kg
#    attitude = np.array([0., 0., 0., 0., 0., 0.])  # deg, deg/s
#    dim = np.array([0.3, 0.1, 0.1])  # m
#    
#    spacecraftConfig, forcesCoeff, surfaces = \
#        parameter_setup_cubesat(orbit_file, obj_id, mass, attitude, dim)
#    
#    # Integration function
#    spacecraftConfig['intfcn'] = ode_twobody_j2_drag_srp_notorque_ukf
#    
#    # Initial covariance
#    spacecraftConfig['covar'] = Po
#    
#    # Perturb initial state
#    print(pert_vect)
#    print(spacecraftConfig['X'])
#    spacecraftConfig['X'][0:6] += \
#        pert_vect.flatten()
#    
#    print(spacecraftConfig['X'])    
#    
#    # Alter additional parameters as needed        
#    forcesCoeff['Q'] = np.eye(3) * 1e-10
#    
#    
#    model_bank[model_id]['spacecraftConfig'] = spacecraftConfig
#    model_bank[model_id]['forcesCoeff'] = forcesCoeff
#    model_bank[model_id]['surfaces'] = surfaces
#    
#    
#    ###########################################################################
#    # Cubesat Spin
#    ###########################################################################    
#    
#    model_id = 'cubesat_spin'
#    model_bank[model_id] = {}
#    model_bank[model_id]['weight'] = 0.1
#    
#    mass = 1.  # kg
#    attitude = np.array([0., 0., 0., 0., 1., 0.])  # deg, deg/s
#    dim = np.array([0.3, 0.1, 0.1])  # m
#    
#    spacecraftConfig, forcesCoeff, surfaces = \
#        parameter_setup_cubesat(orbit_file, obj_id, mass, attitude, dim)
#    
#    # Integration function
#    spacecraftConfig['intfcn'] = ode_twobody_j2_drag_srp_notorque_ukf
#    
#    # Initial covariance
#    spacecraftConfig['covar'] = Po
#    
#    # Perturb initial state
#    print(pert_vect)
#    print(spacecraftConfig['X'])
#    spacecraftConfig['X'][0:6] += \
#        pert_vect.flatten()
#    
#    print(spacecraftConfig['X'])    
#    
#    # Alter additional parameters as needed        
#    forcesCoeff['Q'] = np.eye(3) * 1e-10
#    
#    
#    model_bank[model_id]['spacecraftConfig'] = spacecraftConfig
#    model_bank[model_id]['forcesCoeff'] = forcesCoeff
#    model_bank[model_id]['surfaces'] = surfaces
#    
#    
#    ###########################################################################
#    # Cubesat Tumble
#    ###########################################################################    
#    
#    model_id = 'cubesat_tumble'
#    model_bank[model_id] = {}
#    model_bank[model_id]['weight'] = 0.1
#    
#    mass = 1.  # kg
#    attitude = np.array([0., 0., 0., 1., 1., 1.])  # deg, deg/s
#    dim = np.array([0.3, 0.1, 0.1])  # m
#    
#    spacecraftConfig, forcesCoeff, surfaces = \
#        parameter_setup_cubesat(orbit_file, obj_id, mass, attitude, dim)
#    
#    # Integration function
#    spacecraftConfig['intfcn'] = ode_twobody_j2_drag_srp_notorque_ukf
#    
#    # Initial covariance
#    spacecraftConfig['covar'] = Po
#    
#    # Perturb initial state
#    print(pert_vect)
#    print(spacecraftConfig['X'])
#    spacecraftConfig['X'][0:6] += \
#        pert_vect.flatten()
#    
#    print(spacecraftConfig['X'])    
#    
#    # Alter additional parameters as needed        
#    forcesCoeff['Q'] = np.eye(3) * 1e-10
#    
#    
#    model_bank[model_id]['spacecraftConfig'] = spacecraftConfig
#    model_bank[model_id]['forcesCoeff'] = forcesCoeff
#    model_bank[model_id]['surfaces'] = surfaces
#    
#    
#    ###########################################################################
#    # Boxwing Nadir
#    ###########################################################################    
#    
#    model_id = 'boxwing_nadir'
#    model_bank[model_id] = {}
#    model_bank[model_id]['weight'] = 0.1
#    
#    mass = 200.  # kg
#    mpanel = 5.  # kg
#    attitude = np.array([0., 0., 0., 0., 0., 0.])  # deg, deg/s
#    dim = np.array([1.0, 1.0, 1.0])  # m
#    paneldim = np.array([0.02, 1.0, 0.5])
#    
#    spacecraftConfig, forcesCoeff, surfaces = \
#        parameter_setup_boxwing(orbit_file, obj_id, mass, attitude, dim,
#                                mpanel, paneldim)
#    
#    # Integration function
#    spacecraftConfig['intfcn'] = ode_twobody_j2_drag_srp_notorque_ukf
#    
#    # Initial covariance
#    spacecraftConfig['covar'] = Po
#    
#    # Perturb initial state
#    print(pert_vect)
#    print(spacecraftConfig['X'])
#    spacecraftConfig['X'][0:6] += \
#        pert_vect.flatten()
#    
#    print(spacecraftConfig['X'])    
#    
#    # Alter additional parameters as needed        
#    forcesCoeff['Q'] = np.eye(3) * 1e-10
#    
#    
#    model_bank[model_id]['spacecraftConfig'] = spacecraftConfig
#    model_bank[model_id]['forcesCoeff'] = forcesCoeff
#    model_bank[model_id]['surfaces'] = surfaces
#    
#    
#    ###########################################################################
#    # Boxwing Spin
#    ###########################################################################    
#    
#    model_id = 'boxwing_spin'
#    model_bank[model_id] = {}
#    model_bank[model_id]['weight'] = 0.1
#    
#    mass = 200.  # kg
#    mpanel = 5.  # kg
#    attitude = np.array([0., 0., 0., 0., 1., 0.])  # deg, deg/s
#    dim = np.array([1.0, 1.0, 1.0])  # m
#    paneldim = np.array([0.02, 1.0, 0.5])
#    
#    spacecraftConfig, forcesCoeff, surfaces = \
#        parameter_setup_boxwing(orbit_file, obj_id, mass, attitude, dim,
#                                mpanel, paneldim)
#    
#    # Integration function
#    spacecraftConfig['intfcn'] = ode_twobody_j2_drag_srp_notorque_ukf
#    
#    # Initial covariance
#    spacecraftConfig['covar'] = Po
#    
#    # Perturb initial state
#    print(pert_vect)
#    print(spacecraftConfig['X'])
#    spacecraftConfig['X'][0:6] += \
#        pert_vect.flatten()
#    
#    print(spacecraftConfig['X'])    
#    
#    # Alter additional parameters as needed        
#    forcesCoeff['Q'] = np.eye(3) * 1e-10
#    
#    
#    model_bank[model_id]['spacecraftConfig'] = spacecraftConfig
#    model_bank[model_id]['forcesCoeff'] = forcesCoeff
#    model_bank[model_id]['surfaces'] = surfaces
#    
#            
#   ###########################################################################
#    # Boxwing Tumble
#    ###########################################################################    
#    
#    model_id = 'boxwing_tumble'
#    model_bank[model_id] = {}
#    model_bank[model_id]['weight'] = 0.1
#    
#    mass = 200.  # kg
#    mpanel = 5.  # kg
#    attitude = np.array([0., 0., 0., 1., 1., 1.])  # deg, deg/s
#    dim = np.array([1.0, 1.0, 1.0])  # m
#    paneldim = np.array([0.02, 1.0, 0.5])
#    
#    spacecraftConfig, forcesCoeff, surfaces = \
#        parameter_setup_boxwing(orbit_file, obj_id, mass, attitude, dim,
#                                mpanel, paneldim)
#    
#    # Integration function
#    spacecraftConfig['intfcn'] = ode_twobody_j2_drag_srp_notorque_ukf
#    
#    # Initial covariance
#    spacecraftConfig['covar'] = Po
#    
#    # Perturb initial state
#    print(pert_vect)
#    print(spacecraftConfig['X'])
#    spacecraftConfig['X'][0:6] += \
#        pert_vect.flatten()
#    
#    print(spacecraftConfig['X'])    
#    
#    # Alter additional parameters as needed        
#    forcesCoeff['Q'] = np.eye(3) * 1e-10
#    
#    
#    model_bank[model_id]['spacecraftConfig'] = spacecraftConfig
#    model_bank[model_id]['forcesCoeff'] = forcesCoeff
#    model_bank[model_id]['surfaces'] = surfaces
            

    
    
    
    # Save data
    pklFile = open( model_params_file, 'wb' )
    pickle.dump( [model_bank, eop_alldata, XYs_df], pklFile, -1 )
    pklFile.close()
    
    
    
    return
        


def run_filter(model_params_file, sensor_file, meas_file, filter_output_file,
               ephemeris, ts, alpha=1e-4):
    
#    filter_output = \
#        unscented_kalman_filter(model_params_file, sensor_file, meas_file,
#                                ephemeris, ts, alpha)
        
    
    method = 'mmae'
    filter_output = \
        multiple_model_filter(model_params_file, sensor_file, meas_file,
                              filter_output_file, ephemeris, ts, method, alpha)
        
    
    # Save data
    pklFile = open( filter_output_file, 'wb' )
    pickle.dump( [filter_output], pklFile, -1 )
    pklFile.close()

    return

###############################################################################
# Stand-alone execution
###############################################################################


if __name__ == '__main__':
    
#     General parameters
    obj_id = 90003
    UTC = datetime(2018, 12, 9, 12, 0, 0)
    object_type = 'sphere_low_drag'
    
    # Data directory
    datadir = Path('C:/Users/Steve/Documents/data/multiple_model/'
                   '2018_08_20_imm')
    
    # Filenames
    init_orbit_file = datadir / '500km_orbit_2018_12_09.pkl'
    sensor_file = datadir / 'sensors_ilrs_optical.pkl'
    
    fname = '500km_' + object_type + '_2018_12_09_true_params.pkl'
    true_params_file = datadir / fname
    
    fname = '500km_' + object_type + '_2018_07_12_truth.pkl'
    truth_file = datadir / fname
    
    fname = '500km_' + object_type + '_2018_07_12_meas.pkl'
    meas_file = datadir / fname
    
    fname = '500km_' + object_type + '_2018_07_12_model_params.pkl'
    model_params_file = datadir / fname
    
#    fname = 'leo_sphere_med_mmae_2018_07_12_model_params.pkl'
#    mmae_params_file = datadir / fname
#    
#    fname = 'leo_sphere_med_mmae_2018_07_12_filter_output.pkl'
#    filter_output_file = datadir / fname
#    
#    fname = 'leo_sphere_med_mmae_2018_07_12_filter_error.pkl'
#    error_file = datadir / fname
    
    
    cwd = os.getcwd()
    metis_dir = cwd[0:-10]
    load = Loader(os.path.join(metis_dir, 'skyfield_data'))
    ephemeris = load('de430t.bsp')
    ts = load.timescale()
    
    
#    # Generate initial orbit file
    obj_id = 90003
    launch_elem_dict = {}
    launch_elem_dict[obj_id] = {}
    launch_elem_dict[obj_id]['ra'] = Re + 505.
    launch_elem_dict[obj_id]['rp'] = Re + 500.
    launch_elem_dict[obj_id]['i'] = 97.6
    launch_elem_dict[obj_id]['RAAN'] = 318.
    launch_elem_dict[obj_id]['w'] = 0.
    launch_elem_dict[obj_id]['M'] = 0.
    launch_elem_dict[obj_id]['UTC'] = UTC
    obj_id_list = [90003]
    tle_dict, tle_df = launch2tle(obj_id_list, launch_elem_dict)
    print(tle_dict)
#    generate_init_orbit_file(obj_id, UTC, init_orbit_file, tle_dict)
    
    # Generate sensor file
#    sensor_id_list = ['Stromlo Laser', 'Zimmerwald Laser',
#                      'Arequipa Laser', 'Haleakala Laser',
#                      'Yarragadee Laser']
#    sensor_id_list = ['Stromlo Optical', 'Arequipa Optical', 'Yarragadee Optical']
#    generate_sensor_file(sensor_file, sensor_id_list)
    
    # Visibility Analysis
    

    # Generate true params file
#    generate_true_params_file(init_orbit_file, obj_id, object_type, true_params_file)
    
    
    # Generate truth trajectory and measurements file
    ndays = 7.
    dt = 10.
    
#    generate_truth_file(true_params_file, truth_file, ephemeris, ts, ndays, dt)
    
#    # Generate noisy measurements file
    ndays = 3.
#    generate_noisy_meas(true_params_file, truth_file, sensor_file, meas_file,
#                        ephemeris, ndays)
    
    # Generate model parameters file
#    generate_ukf_params(true_params_file, model_params_file)
    
#    generate_mmae_params(true_params_file, init_orbit_file, mmae_params_file)
    
    # Run filter
#    run_filter(mmae_params_file, sensor_file, meas_file, filter_output_file,
#               ephemeris, ts, alpha=1e-4)
#    
    
    
    
    # Compute and plot errors
#    compute_mmae_errors(filter_output_file, truth_file, error_file)
#    plot_mmae_errors(error_file)
    
    
#    compute_ukf_errors(filter_output_file, truth_file, error_file)
#    plot_ukf_errors(error_file)
    
    
    
    
    
    
    
    
    
    
    

    