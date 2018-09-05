import numpy as np
from datetime import datetime, timedelta

import utilities.astrodynamics as astro


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
    
    obj_id_list = [40940]
    UTC_list = [datetime(2018, 1, 16, 12, 43, 20)]
    
    