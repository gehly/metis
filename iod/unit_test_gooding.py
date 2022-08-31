import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta
import pickle
import csv
import os
import inspect
import time
from astroquery.jplhorizons import Horizons

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

import dynamics.dynamics_functions as dyn
import estimation.analysis_functions as analysis
import estimation.estimation_functions as est
import iod.iod_functions as iod
import sensors.measurement_functions as mfunc
import sensors.sensors as sens
import sensors.visibility_functions as visfunc
import utilities.astrodynamics as astro
import utilities.coordinate_systems as coord
import utilities.eop_functions as eop
import utilities.time_systems as timesys

from utilities.constants import GME, Re, arcsec2rad




def single_rev_geo():
    
    # Generic GEO Orbit
    elem = [42164.1, 0.01, 0.1, 90., 1., 1.]
    Xo = np.reshape(astro.kep2cart(elem), (6,1))
    print('Xo true', Xo)
    
    # Time vector
    UTC0 = datetime(2021, 6, 21, 0, 0, 0)
    UTC1 = datetime(2021, 6, 21, 4, 0, 0)
    UTC2 = datetime(2021, 6, 21, 6, 0, 0)
    UTC_list = [UTC0, UTC1, UTC2]
    
    # Sensor data
    sensor_id = 'UNSW Falcon'
    sensor_params = sens.define_sensors([sensor_id])
    sensor_params[sensor_id]['meas_types'] = ['ra', 'dec']
    sensor = sensor_params[sensor_id]
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = eop.get_celestrak_eop_alldata()
        
    # Retrieve polar motion data from file
    XYs_df = eop.get_XYs2006_alldata()
    
    # Compute measurements
    Yk_list = []
    sensor_id_list = []
    rho_list = []
    for UTC in UTC_list:
        
        dt_sec = (UTC - UTC0).total_seconds()
        EOP_data = eop.get_eop_data(eop_alldata, UTC)
        Xk = astro.element_conversion(Xo, 1, 1, dt=dt_sec)
        all_meas = mfunc.compute_measurement(Xk, {}, sensor, UTC, EOP_data,
                                             XYs_df, meas_types=['ra', 'dec', 'rg', 'az', 'el'])
        
        
        print(all_meas)
        
        Yk = all_meas[0:2].reshape(2,1)
        Yk_list.append(Yk)
        sensor_id_list.append(sensor_id)
        rho_list.append(float(all_meas[2]))
        
    
    
    print(Yk_list)
    print(sensor_id_list)
    print(rho_list)
    

    # Execute function
    X_list, M_list = iod.gooding_angles_iod(UTC_list, Yk_list, sensor_id_list,
                                            sensor_params)
    
    
    print('Final Answers')
    print('X_list', X_list)
    print('M_list', M_list)
    
    for ii in range(len(M_list)):
        
        X_err = X_list[ii] - Xo
        print('')
        print('ii', ii)
        print('M', M_list[ii])
        print('X err', np.linalg.norm(X_err))
    
    
    return



if __name__ == '__main__':
    
    
    single_rev_geo()





























