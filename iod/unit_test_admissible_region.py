###############################################################################
# This file contains code to test Admissible Region IOD functions.
#
# References:
#
#  [1] DeMars, K.J. and Jah, M.K., "Probabilistic Initial Orbit Determination
#      Using Gaussian Mixture Models," JGCD, 2013.
#
###############################################################################

import numpy as np


import admissible_region as ar



def unit_test_optical_car_gmm_demars():
    
    # True orbit (DeMars Section V.E.)
    a = 43000.
    e = 0.03
    i = np.radians(3.)
    RAAN = 0.
    w = 0.
    theta = 0.
    
    # Vector of range values
    rho_vect = np.arange(0., 50000., 5.)
    
    
    
    # Set up parameters for CAR GMM function
    params = {}
    params['rho_vect'] = rho_vect
    
    
    return



if __name__ == '__main__':
    
    
    