import numpy as np
from math import pi, log10


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
    
    G = 4.*pi*A/(lam**2.)
    
    return G


def decimal2dB(value):
    '''
    This function converts a decimal value (e.g power) to decibels.
    
    Parameters
    ------
    value : float
        input value to convert to dB, s.g. power
    
    Returns
    ------
    dB : float
        value in decibels
    
    '''
    
    dB = 10. * log10(value)    
    
    return dB


def dB2decimal(dB):
    '''
    This function convert a decibel value to decimal.
    
    Parameters
    ------
    dB : float
        input value in decibels
        
    Returns
    ------
    value : float
        value in decimal 
        
    '''
    
    value = 10.**(dB/10.)
    
    return value