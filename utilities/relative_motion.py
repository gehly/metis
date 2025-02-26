import numpy as np
import math
import os
import sys
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from utilities import time_systems as timesys
from utilities import ephemeris as eph
from utilities.constants import GME, J2E, Re, wE


###############################################################################
# Relative Motion Parameterization
###############################################################################

def inertial2relative_ei(rc_vect, vc_vect, rd_vect, vd_vect, GM=GME):
    '''
    This function converts inertial Cartesian position and velocity vectors of 
    two space objects into relative eccentricity and inclination vectors, also
    expressed in the inertial coordinate frame.
    
    '''
    
    # Reshape inputs if needed
    rc_vect = np.reshape(rc_vect, (3,1))
    vc_vect = np.reshape(vc_vect, (3,1))
    rd_vect = np.reshape(rd_vect, (3,1))
    vd_vect = np.reshape(vd_vect, (3,1))
    
    # Compute angular momentum unit vectors
    hc_vect = np.cross(rc_vect, vc_vect, axis=0)
    hd_vect = np.cross(rd_vect, vd_vect, axis=0)
    ih_c = hc_vect/np.linalg.norm(hc_vect)
    ih_d = hd_vect/np.linalg.norm(hd_vect)
    
    # Compute eccentricity vectors
    ec_vect = np.cross(vc_vect, hc_vect, axis=0)/GM - rc_vect/np.linalg.norm(rc_vect)
    ed_vect = np.cross(vd_vect, hd_vect, axis=0)/GM - rd_vect/np.linalg.norm(rd_vect)
    
    # Compute relative ecc/inc vectors in inertial frame
    di_vect = np.cross(ih_c, ih_d, axis=0)
    de_vect = ed_vect - ec_vect    
    
    return de_vect, di_vect


