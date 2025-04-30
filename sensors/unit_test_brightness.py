import numpy as np
import math
import matplotlib.pyplot as plt

# Load required tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.astro.time_conversion import DateTime
from tudatpy.astro import element_conversion
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime

# Load spice kernels
spice.load_standard_kernels()



def object_setup():
    
    # Initial time - Vernal Equninox
    epoch_tdb = DateTime(2025, 3, 20, 12, 0, 0).epoch()
    
    # Initial state
    # LEO SSO
    # elem = np.array([7000e3, 0.001, np.radians(98.), 0., 0., 0.])
    
    # GEO
    elem = np.array([42164e3, 0.001, 0.001, 0., 0., 0])
    
    Xo = element_conversion.keplerian_to_cartesian(elem, 3.986e14)
        
    # Tudat parameters
    bodies_to_create = ['Earth', 'Sun', 'Moon']
        
    rso_params = {}
    rso_params['epoch_tdb'] = epoch_tdb
    rso_params['state'] = Xo
    rso_params['sph_deg'] = 8
    rso_params['sph_ord'] = 8    
    rso_params['central_bodies'] = ['Earth']
    rso_params['bodies_to_create'] = bodies_to_create
    rso_params['Cd'] = 2.2
    rso_params['Cr'] = 1.3
    rso_params['area'] = 1.
    rso_params['mass'] = 100.
    
    int_params = {}
    int_params['tudat_integrator'] = 'rk4'
    int_params['step'] = 1.
    
    
    
    return rso_params, int_params


def plot_measurements(check_elevation=True, check_station_dark=True, check_eclipse=True):
    
    # Retrieve object data
    rso_params, int_params = object_setup() 
    t0 = rso_params['epoch_tdb']
    Xo = rso_params['state']    
    area = rso_params['area']
    Cr = rso_params['Cr']
    albedo = Cr - 1.0
    
    # Create sensor - Leiden Optical
    latitude_rad = 52.155*np.pi/180.
    longitude_rad = 4.485*np.pi/180.
    height_m = 0.
    
    sensor_params = define_optical_sensor(latitude_rad, longitude_rad, height_m)
    sensor_params['meas_types'] = ['rg', 'az', 'el', 'ra', 'dec']
    el_lim = sensor_params['el_lim']
    
    # Setup propagation
    # Initialize bodies
    bodies_to_create = rso_params['bodies_to_create']
    bodies = tudat_initialize_bodies(bodies_to_create)
    
    # Create a different bodies object for the purpose of calculating sun position
    # Need to set frame origin for the Sun's position to Earth
    bodies_to_create = ['Sun', 'Earth']
    
    # Use "Earth"/"J2000" as global frame origin and orientation.
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"

    # Create default body settings, usually from `spice`.
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        global_frame_origin,
        global_frame_orientation)
    
    body_settings.get( 'Sun' ).ephemeris_settings = environment_setup.ephemeris.direct_spice(
    frame_origin = 'Earth',
    frame_orientation = 'J2000' )

    # Create system of selected celestial bodies
    bodies_sun_state = environment_setup.create_system_of_bodies(body_settings)
    
    # Propagate orbit
    tf = t0 + 86400.
    tvec = np.array([t0, tf])
    tout, Xout = propagate_orbit(Xo, tvec, rso_params, int_params, bodies)
    
    # Loop over times, check constraints, store measurements at visible times
    tk_list = []
    Yk_list = []
    thrs_all = []
    rg_all = []
    az_all = []
    el_all = []
    ra_all = []
    dec_all = []
    mag_all = []
    phase_all = []
    tmeas_hrs = []
    rg_meas = []
    az_meas = []
    el_meas = []
    ra_meas = []
    dec_meas = []
    mag_meas = []
    phase_meas = []
    for kk in range(len(tout)):
        
        # Retrieve current time and state
        tk = tout[kk]
        Xk = Xout[kk,:].reshape(6,1)
        rso_eci = Xk[0:3].reshape(3,1)
        u_sat = rso_eci.flatten()/np.linalg.norm(rso_eci)
        
        # Compute geometric measurements
        meas = compute_measurement(tk, Xk, sensor_params, bodies=bodies)
        rg = meas[0,0]
        az = meas[1,0]
        el = meas[2,0]
        ra = meas[3,0]
        dec = meas[4,0]
        
        # Current sun position
        sun_ephemeris = bodies_sun_state.get('Sun').ephemeris
        sun_eci = sun_ephemeris.cartesian_state(tk)[0:3].reshape(3,1)
        sat2sun = sun_eci - rso_eci
        u_sun = sat2sun.flatten()/np.linalg.norm(sat2sun)        
        
        # Current sensor position
        sensor_ecef = sensor_params['sensor_ecef']
        earth_rotation_model = bodies.get("Earth").rotation_model
        ecef2eci = earth_rotation_model.body_fixed_to_inertial_rotation(tk)
        sensor_eci = np.dot(ecef2eci, sensor_ecef)
        sat2obs = sensor_eci - rso_eci
        u_obs = sat2obs/np.linalg.norm(sat2obs)
        
        # Compute brightness (Cognion Eq 1)
        phase_angle = math.acos(float(np.dot(u_sun.T, u_obs)))
        F_diff = (2./3.)*albedo*area/(np.pi**2.*rg**2.)*(np.sin(phase_angle) + (np.pi - phase_angle)*np.cos(phase_angle))
        mag = -26.74 - 2.5*math.log10(F_diff)
        
        # Store for plots
        thrs_all.append((tk-tout[0])/3600.)
        rg_all.append(rg/1000.)
        az_all.append(np.degrees(az))
        el_all.append(np.degrees(el))
        ra_all.append(ra)
        dec_all.append(dec)
        mag_all.append(mag)
        phase_all.append(phase_angle)
        
        # Check constraints as specified by input flags
        if check_elevation:
            if el < el_lim[0] or el > el_lim[1]:
                print(tk, 'elevation check failed')
                print('el', el)
                print('el_lim', el_lim)
                continue
            
        if check_station_dark:
            sun_meas = compute_measurement(tk, sun_eci, sensor_params, bodies=bodies)
            sun_el = sun_meas[2,0]
            if sun_el > sensor_params['sun_elmask']:
                print(tk, 'station not dark')
                continue
            
        if check_eclipse:
            r = np.linalg.norm(rso_eci)
            if r < 6378*1000.:
                continue               
            else:
                half_cone = math.asin((6378*1000.)/r)
                sun_angle = math.acos(np.dot(u_sun, -u_sat))
                if sun_angle < half_cone:                    
                    print(tk, 'eclipse')                    
                    continue
        
        # All constraints passed, store measurements
        tmeas_hrs.append((tk-tout[0])/3600.)
        rg_meas.append(rg/1000.)
        az_meas.append(np.degrees(az))
        el_meas.append(np.degrees(el))
        ra_meas.append(ra)
        dec_meas.append(dec)
        mag_meas.append(mag)
        phase_meas.append(phase_angle)
        
        
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(thrs_all, rg_all, 'k.')
    plt.plot(tmeas_hrs, rg_meas, 'r.')
    plt.ylabel('Range [km]')
    plt.subplot(3,1,2)
    plt.plot(thrs_all, az_all, 'k.')
    plt.plot(tmeas_hrs, az_meas, 'r.')
    plt.ylabel('Az [deg]')
    plt.subplot(3,1,3)
    plt.plot(thrs_all, el_all, 'k.')
    plt.plot(tmeas_hrs, el_meas, 'r.')
    plt.ylabel('El [deg]')
    plt.xlabel('Time [hours]')
    
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(thrs_all, ra_all, 'k.')
    plt.plot(tmeas_hrs, ra_meas, 'r.')
    plt.ylabel('RA [rad]')
    plt.subplot(2,1,2)
    plt.plot(thrs_all, dec_all, 'k.')
    plt.plot(tmeas_hrs, dec_meas, 'r.')
    plt.ylabel('DEC [rad]')
    plt.xlabel('Time [hours]')
    
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(thrs_all, phase_all, 'k.')
    plt.plot(tmeas_hrs, phase_meas, 'r.')
    plt.ylabel('Phase Angle [rad]')
    plt.subplot(2,1,2)
    plt.plot(thrs_all, mag_all, 'k.')
    plt.plot(tmeas_hrs, mag_meas, 'r.')
    plt.ylabel('App Mag')
    plt.xlabel('Time [hours]')
    
    plt.show()
    
    
    return


###############################################################################
# Utility functions
###############################################################################


def define_optical_sensor(latitude_rad, longitude_rad, height_m):
    '''
    This function will generate the sensor parameters dictionary for an optical
    sensor provided the location in latitude, longitude, height.
    
    It is pre-filled with constraint and noise parameters per assignment
    description.

    Parameters
    ----------
    latitude_rad : float
        geodetic latitude of sensor [rad]
    longitude_rad : float
        geodetic longitude of sensor [rad]
    height_m : float
        geodetic height of sensor [m]

    Returns
    -------
    sensor_params : dictionary
        location, constraint, noise parameters of sensor

    '''
    
    arcsec2rad = (1./3600.)*np.pi/180.
            
    # Compute sensor location in ECEF/ITRF
    sensor_ecef = latlonht2ecef(latitude_rad, longitude_rad, height_m)
    
    # Constraints/Limits
    az_lim = [0., 2.*np.pi]  # rad
    el_lim = [15.*np.pi/180., np.pi/2.]  # rad
    rg_lim = [0., np.inf]   # m
    sun_el_mask = -12.*np.pi/180.  # rad (Nautical twilight)
    
    # Measurement types and noise
    # meas_types = ['ra', 'dec']
    # sigma_dict = {}
    # sigma_dict['ra'] = arcsec2rad    # rad
    # sigma_dict['dec'] = arcsec2rad   # rad
    
    # meas_types = ['mag']
    # sigma_dict = {}
    # sigma_dict['mag'] = 0.01
    
    meas_types = []
    sigma_dict = {}
    
        
    # Location and constraints
    sensor_params = {}
    sensor_params['sensor_ecef'] = sensor_ecef
    sensor_params['el_lim'] = el_lim
    sensor_params['az_lim'] = az_lim
    sensor_params['rg_lim'] = rg_lim
    sensor_params['sun_elmask'] = sun_el_mask
    
    # Measurements and noise
    sensor_params['meas_types'] = meas_types
    sensor_params['sigma_dict'] = sigma_dict
    

    
    return sensor_params


def compute_measurement(tk, X, sensor_params, bodies=None):
    '''
    This function be used to compute a measurement given an input state vector
    and time.
    
    Parameters
    ------
    tk : float
        time in seconds since J2000
    X : nx1 numpy array
        Cartesian state vector [m, m/s]
    sensor_params : dictionary
        location, constraint, noise parameters of sensor
        
    Returns
    ------
    Y : px1 numpy array
        computed measurements for given state and sensor
    
    '''
    
    if bodies is None:
        body_settings = environment_setup.get_default_body_settings(
            ["Earth"],
            "Earth",
            "J2000")
        bodies = environment_setup.create_system_of_bodies(body_settings)
        
    # Rotation matrices
    earth_rotation_model = bodies.get("Earth").rotation_model
    eci2ecef = earth_rotation_model.inertial_to_body_fixed_rotation(tk)
    ecef2eci = earth_rotation_model.body_fixed_to_inertial_rotation(tk)
        
    # Retrieve measurement types
    meas_types = sensor_params['meas_types']
    
    # Compute station location in ECI    
    sensor_ecef = sensor_params['sensor_ecef']
    sensor_eci = np.dot(ecef2eci, sensor_ecef)    
    
    # Object location in ECI
    r_eci = X[0:3].reshape(3,1)
    
    # Compute range and line of sight vector
    rg = np.linalg.norm(r_eci - sensor_eci)
    rho_hat_eci = (r_eci - sensor_eci)/rg
    
    # Rotate to ENU frame
    rho_hat_ecef = np.dot(eci2ecef, rho_hat_eci)
    rho_hat_enu = ecef2enu(rho_hat_ecef, sensor_ecef)
    
    # Loop over measurement types
    Y = np.zeros((len(meas_types),1))
    ii = 0
    for mtype in meas_types:
        
        if mtype == 'rg':
            Y[ii] = rg  # m
            
        elif mtype == 'ra':
            Y[ii] = math.atan2(rho_hat_eci[1], rho_hat_eci[0]) # rad
            
        elif mtype == 'dec':
            Y[ii] = math.asin(rho_hat_eci[2])  # rad
    
        elif mtype == 'az':
            Y[ii] = math.atan2(rho_hat_enu[0], rho_hat_enu[1])  # rad  
            # if Y[ii] < 0.:
            #     Y[ii] += 2.*np.pi
            
        elif mtype == 'el':
            Y[ii] = math.asin(rho_hat_enu[2])  # rad
            
            
        ii += 1
            
            
    return Y


def tudat_initialize_bodies(bodies_to_create=[]):
    '''
    This function initializes the bodies object for use in the Tudat 
    propagator. For the cases considered, only Earth, Sun, and Moon are needed,
    with Earth as the frame origin.
    
    Parameters
    ------
    bodies_to_create : list, optional (default=[])
        list of bodies to create, if empty, will use default Earth, Sun, Moon
    
    Returns
    ------
    bodies : tudat object
    
    '''

    # Define string names for bodies to be created from default.
    if len(bodies_to_create) == 0:
        bodies_to_create = ["Sun", "Earth", "Moon"]

    # Use "Earth"/"J2000" as global frame origin and orientation.
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"

    # Create default body settings, usually from `spice`.
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        global_frame_origin,
        global_frame_orientation)

    # Create system of selected celestial bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)
    
    return bodies


def propagate_orbit(Xo, tvec, state_params, int_params, bodies=None):
    '''
    This function propagates an orbit using physical parameters provided in 
    state_params and integration parameters provided in int_params.
    
    Parameters
    ------
    Xo : 6x1 numpy array
        Cartesian state vector [m, m/s]
    tvec : list or numpy array
        propagator will only use first and last terms to set the initial and
        final time of the propagation, intermediate times are ignored
        
        [t0, ..., tf] given as time in seconds since J2000
        
    state_params : dictionary
        propagator parameters
        
        fields:
            Cd: float, drag coefficient
            Cr: float, reflectivity coefficient
            area: float [m^2]
            mass: float [kg]
            sph_deg: int, spherical harmonics expansion degree for Earth
            sph_ord: int, spherical harmonics expansion order for Earth
            central_bodies: list of central bodies for propagator ["Earth"]
            bodies_to_create: list of bodies to create ["Earth", "Sun", "Moon"]
            
    int_params : dictionary
        numerical integration parameters
        
    bodies : tudat object, optional
        contains parameters for the environment bodies used in propagation
        if None, will initialize with bodies given in state_params
        
    Returns
    ------
    tout : N element numpy array
        times of propagation output in seconds since J2000
    Xout : Nxn numpy array
        each row Xout[k,:] corresponds to Cartesian state at time tout[k]        
    
    '''
    
    # Initial state
    initial_state = Xo.flatten()
    
    # Retrieve input parameters
    central_bodies = state_params['central_bodies']
    bodies_to_create = state_params['bodies_to_create']
    mass = state_params['mass']
    Cd = state_params['Cd']
    Cr = state_params['Cr']
    area = state_params['area']
    sph_deg = state_params['sph_deg']
    sph_ord = state_params['sph_ord']
    
    # Simulation start and end
    simulation_start_epoch = tvec[0]
    simulation_end_epoch = tvec[-1]
    
    # Setup bodies
    if bodies is None:
        bodies = tudat_initialize_bodies(bodies_to_create)
    
    
    # Create the bodies to propagate
    # TUDAT always uses 6 element state vector
    N = int(len(Xo)/6)
    central_bodies = central_bodies*N
    bodies_to_propagate = []
    for jj in range(N):
        jj_str = str(jj)
        bodies.create_empty_body(jj_str)
        bodies.get(jj_str).mass = mass
        bodies_to_propagate.append(jj_str)
        
        if Cd > 0.:
            aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
                area, [Cd, 0, 0]
            )
            environment_setup.add_aerodynamic_coefficient_interface(
                bodies, jj_str, aero_coefficient_settings)
            
        if Cr > 0.:
            # occulting_bodies = ['Earth']
            # radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
            #     'Sun', srp_area_m2, Cr, occulting_bodies
            # )
            # environment_setup.add_radiation_pressure_interface(
            #     bodies, jj_str, radiation_pressure_settings)
            
            occulting_bodies_dict = dict()
            occulting_bodies_dict[ "Sun" ] = [ "Earth" ]
            
            radiation_pressure_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
                area, Cr, occulting_bodies_dict )
            
            environment_setup.add_radiation_pressure_target_model(
                bodies, jj_str, radiation_pressure_settings)
            

    acceleration_settings_setup = {}        
    if 'Earth' in bodies_to_create:
        
        # Gravity
        if sph_deg == 0 and sph_ord == 0:
            acceleration_settings_setup['Earth'] = [propagation_setup.acceleration.point_mass_gravity()]
        else:
            acceleration_settings_setup['Earth'] = [propagation_setup.acceleration.spherical_harmonic_gravity(sph_deg, sph_ord)]
        
        # Aerodynamic Drag
        if Cd > 0.:                
            acceleration_settings_setup['Earth'].append(propagation_setup.acceleration.aerodynamic())
        
    if 'Sun' in bodies_to_create:
        
        # Gravity
        acceleration_settings_setup['Sun'] = [propagation_setup.acceleration.point_mass_gravity()]
        
        # Solar Radiation Pressure
        if Cr > 0.:                
            #acceleration_settings_setup['Sun'].append(propagation_setup.acceleration.cannonball_radiation_pressure())
            acceleration_settings_setup['Sun'].append(propagation_setup.acceleration.radiation_pressure())
    
    if 'Moon' in bodies_to_create:
        
        # Gravity
        acceleration_settings_setup['Moon'] = [propagation_setup.acceleration.point_mass_gravity()]
    

    acceleration_settings = {}
    for jj in range(N):
        acceleration_settings[str(jj)] = acceleration_settings_setup
        
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )
    

    # Create termination settings
    termination_condition = propagation_setup.propagator.time_termination(
        simulation_end_epoch, terminate_exactly_on_final_condition=True
    )


    # Create numerical integrator settings
    if int_params['tudat_integrator'] == 'rk4':
        fixed_step_size = int_params['step']
        integrator_settings = propagation_setup.integrator.runge_kutta_4(
            fixed_step_size
        )
        
    elif int_params['tudat_integrator'] == 'rkf78':
        initial_step_size = int_params['step']
        maximum_step_size = int_params['max_step']
        minimum_step_size = int_params['min_step']
        rtol = int_params['rtol']
        atol = int_params['atol']
        integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
            initial_step_size,
            propagation_setup.integrator.CoefficientSets.rkf_78,
            minimum_step_size,
            maximum_step_size,
            rtol,
            atol)
    
        
        
    # Create propagation settings
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        simulation_start_epoch,
        integrator_settings,
        termination_condition
    )
    
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings )

    # Extract the resulting state history and convert it to an ndarray
    states = dynamics_simulator.state_history
    states_array = result2array(states)        
    
    
    tout = states_array[:,0]
    Xout = states_array[:,1:6*N+1]
    
    
    return tout, Xout


###############################################################################
# Coordinate Frames
###############################################################################


def ecef2enu(r_ecef, r_site):
    '''
    This function converts the coordinates of a position vector from
    the ECEF to ENU frame.

    Parameters
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF [m]
    r_site : 3x1 numpy array
      station position vector in ECEF [m]

    Returns
    ------
    r_enu : 3x1 numpy array
      position vector in ENU [m]
    '''

    # Compute lat,lon,ht of ground station
    lat, lon, ht = ecef2latlonht(r_site)

    # Compute rotation matrix
    lat1 = math.pi/2 - lat
    lon1 = math.pi/2 + lon

    R1 = np.array([[1.,               0.,             0.],
                   [0.,   math.cos(lat1), math.sin(lat1)],
                   [0.,  -math.sin(lat1), math.cos(lat1)]])

    R3 = np.array([[math.cos(lon1),  math.sin(lon1), 0.],
                   [-math.sin(lon1), math.cos(lon1), 0.],
                   [0.,              0.,             1.]])

    R = np.dot(R1, R3)

    r_enu = np.dot(R, r_ecef)

    return r_enu


def enu2ecef(r_enu, r_site):
    '''
    This function converts the coordinates of a position vector from
    the ENU to ECEF frame.

    Parameters
    ------
    r_enu : 3x1 numpy array
      position vector in ENU [m]
    r_site : 3x1 numpy array
      station position vector in ECEF [m]

    Returns
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF
    '''

    # Compute lat,lon,ht of ground station
    lat, lon, ht = ecef2latlonht(r_site)

    # Compute rotation matrix
    lat1 = math.pi/2 - lat
    lon1 = math.pi/2 + lon

    R1 = np.array([[1.,               0.,             0.],
                   [0.,   math.cos(lat1), math.sin(lat1)],
                   [0.,  -math.sin(lat1), math.cos(lat1)]])

    R3 = np.array([[math.cos(lon1),   math.sin(lon1), 0.],
                   [-math.sin(lon1),  math.cos(lon1), 0.],
                   [0.,                           0., 1.]])

    R = np.dot(R1, R3)

    R2 = R.T

    r_ecef = np.dot(R2, r_enu)

    return r_ecef


def ecef2latlonht(r_ecef):
    '''
    This function converts the coordinates of a position vector from
    the ECEF frame to geodetic latitude, longitude, and height.

    Parameters
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF [m]

    Returns
    ------
    lat : float
      latitude [rad] [-pi/2,pi/2]
    lon : float
      longitude [rad] [-pi,pi]
    ht : float
      height [m]
    '''

    # WGS84 Data (Pratap and Misra P. 103)
    a = 6378137.0   # m
    rec_f = 298.257223563

    # Get components from position vector
    x = float(r_ecef[0])
    y = float(r_ecef[1])
    z = float(r_ecef[2])

    # Compute longitude
    f = 1./rec_f
    e = np.sqrt(2.*f - f**2.)
    lon = math.atan2(y, x)

    # Iterate to find height and latitude
    p = np.sqrt(x**2. + y**2.)  # m
    lat = 0.
    lat_diff = 1.
    tol = 1e-12

    while abs(lat_diff) > tol:
        lat0 = float(lat)  # rad
        N = a/np.sqrt(1 - e**2*(math.sin(lat0)**2))  # km
        ht = p/math.cos(lat0) - N
        lat = math.atan((z/p)/(1 - e**2*(N/(N + ht))))
        lat_diff = lat - lat0


    return lat, lon, ht


def latlonht2ecef(lat, lon, ht):
    '''
    This function converts geodetic latitude, longitude and height
    to a position vector in ECEF.

    Parameters
    ------
    lat : float
      geodetic latitude [rad]
    lon : float
      geodetic longitude [rad]
    ht : float
      geodetic height [m]

    Returns
    ------
    r_ecef = 3x1 numpy array
      position vector in ECEF [m]
    '''
    
    # WGS84 Data (Pratap and Misra P. 103)
    Re = 6378137.0   # m
    rec_f = 298.257223563

    # Compute flattening and eccentricity
    f = 1/rec_f
    e = np.sqrt(2*f - f**2)

    # Compute ecliptic plane and out of plane components
    C = Re/np.sqrt(1 - e**2*math.sin(lat)**2)
    S = Re*(1 - e**2)/np.sqrt(1 - e**2*math.sin(lat)**2)

    rd = (C + ht)*math.cos(lat)
    rk = (S + ht)*math.sin(lat)

    # Compute ECEF position vector
    r_ecef = np.array([[rd*math.cos(lon)], [rd*math.sin(lon)], [rk]])

    return r_ecef



if __name__ == '__main__':
    
    
    # Set flags to all true to do visibility checks, or false to turn off
    plot_measurements(True, True, True)
