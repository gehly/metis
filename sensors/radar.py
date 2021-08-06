import numpy as np
from math import pi


def compute_antenna_gain(A, lam):
    '''
    This function computes the antenna gain, the factor increase in power as
    compared to an isotropic antenna.
    
    Parameters
    ------
    A : float
        antenna aperture area [m^2]
    lam : float
        wavelength [m]
        
    Returns
    ------
    G : float
        antenna gain [unitless]
    
    '''
    
    G = 4.*pi*A/lam    
    
    return G

