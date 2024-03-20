import numpy as np
import math

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import pickle

metis_dir = r'C:\Users\Steve\Documents\code\metis'
sys.path.append(metis_dir)


import data_processing.data_processing as proc
import dynamics.dynamics_functions as dyn
import estimation.analysis_functions as analysis
import estimation.estimation_functions as est
import sensors.measurement_functions as mfunc
import sensors.sensors as sens
import utilities.coordinate_systems as coord
import utilities.eop_functions as eop
import utilities.numerical_methods as num
import utilities.time_systems as timesys
import utilities.tle_functions as tle

from utilities.constants import arcsec2rad, GME, J2E, wE, Re






def lead_follow_tle(UTC):
    
    M2A_norad = 47967
    M2B_norad = 47973
    
    # Retreive TLE data for this time and ECI position vectors
    UTC_list = [UTC]
    obj_id_list = [M2A_norad, M2B_norad]
    output_state = tle.propagate_TLE(obj_id_list, UTC_list)
    
    # Use M2A as chief, compute relative position
    M2A_r_eci = output_state[M2A_norad]['r_GCRF'][0]
    M2A_v_eci = output_state[M2A_norad]['v_GCRF'][0]
    M2B_r_eci = output_state[M2B_norad]['r_GCRF'][0]
    M2B_v_eci = output_state[M2B_norad]['v_GCRF'][0]
    
    rho_eci = M2B_r_eci - M2A_r_eci
    rho_ric = coord.eci2ric(M2A_r_eci, M2A_v_eci, rho_eci)
    
    print('rho_ric', rho_ric)
    print('rho', np.linalg.norm(rho_ric))
    
    
    return




if __name__ == '__main__':
    
    UTC = datetime(2022, 8, 3, 9, 28, 14)
    lead_follow_tle(UTC)

