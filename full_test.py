import numpy as np
from datetime import datetime, timedelta

import utilities.astrodynamics as astro
import utilities.coordinate_systems as coord
import utilities.eop_functions as eop
import utilities.tle_functions as tle


###############################################################################
#
# This script runs a full set of unit tests for individual functions in the 
# repository.
#
###############################################################################



if __name__ == '__main__':
    
    # Orbital element conversions
    osc_elem = [23000., 0.7, 3., 10., 20., 30.]
    mean_elem = astro.osc2mean(osc_elem)
    check = astro.mean2osc(mean_elem)
    
    print(mean_elem)
    print(osc_elem)
    print(check)
    
    ###########################################################################
    # TLE to GCRF back to TLE
    ###########################################################################
    obj_id = 40940
    obj_id_list = [obj_id]
    UTC_list = [datetime(2018, 1, 16, 12, 43, 20)]
    
    # Get TLE and compute TEME and GCRF
    tle_dict = tle.get_spacetrack_tle_data(obj_id_list, UTC_list)
    UTC_list = tle_dict[obj_id]['UTC_list']
    output_state = tle.propagate_TLE(obj_id_list, UTC_list, tle_dict)
    UTC = output_state[obj_id]['UTC'][0]
    r_GCRF = output_state[obj_id]['r_GCRF'][0]
    v_GCRF = output_state[obj_id]['v_GCRF'][0]
    r_TEME = output_state[obj_id]['r_TEME'][0]
    v_TEME = output_state[obj_id]['v_TEME'][0]
    print(tle_dict)
    
    # Recompute TEME
    eop_alldata = eop.get_celestrak_eop_alldata()
    EOP_data = eop.get_eop_data(eop_alldata, UTC)
    IAU1980nut = eop.get_nutation_data()
    r_TEME2, v_TEME2 = coord.gcrf2teme(r_GCRF, v_GCRF, UTC, IAU1980nut, EOP_data)
    print('r_TEME check', r_TEME - r_TEME2)
    print('v_TEME check', v_TEME - v_TEME2)
    
    # Get osculating orbital elements, mean orbit elements, and TLE
    x_in = np.concatenate((r_TEME, v_TEME), axis=0)
    osc_elem = astro.element_conversion(x_in, 1, 0)
    mean_elem = astro.osc2mean(osc_elem)
    kep_dict = {}
    kep_dict[obj_id] = {}
    kep_dict[obj_id]['a'] = mean_elem[0]
    kep_dict[obj_id]['e'] = mean_elem[1]
    kep_dict[obj_id]['i'] = mean_elem[2]
    kep_dict[obj_id]['RAAN'] = mean_elem[3]               
    kep_dict[obj_id]['w'] = mean_elem[4]
    kep_dict[obj_id]['M'] = mean_elem[5]
    kep_dict[obj_id]['UTC'] = UTC
    tle_dict = tle.kep2tle(obj_id_list, kep_dict)
    
    print(tle_dict)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    