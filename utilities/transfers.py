import numpy as np
from math import pi, sin, cos, tan, asin, acos, atan2
import sys

sys.path.append('../')

from utilities.constants import GME, J2E, Re, wE
from utilities.astrodynamics import compute_vc


def compute_launch_velocity(lat_rad, R=Re, w=wE):
    '''
    This function computes the launch velocity component contributed by the 
    planet's rotation for a given latitude.
    
    Parameters
    ------
    lat_rad : float
        geodetic latitude [radians]
    R : float, optional
        planet radius (default=Re) [km]
    w : float, optional
        planet rotational velocity (default=wE) [rad/s]
        
    Returns
    ------
    v0 : float
        velocity magnitude [km/s]
    
    '''
    
    v0 = R*w*cos(lat_rad)
    
    return v0



if __name__ == '__main__':
    
    r = Re + 500.
    lat_rad = 5.2*pi/180.
    i = lat_rad
    
    v0 = compute_launch_velocity(lat_rad)
    vc = compute_vc(r)
    
    dV1 = vc - v0
    
    v1_vect = np.reshape([vc*cos(i), vc*sin(i)], (2,1))
    v2_vect = np.reshape([vc, 0.], (2,1))
    
    print(v1_vect)
    print(v2_vect)
    
    dV2 = np.linalg.norm(v2_vect - v1_vect)
    
    dV = dV1 + dV2
    
    check = vc*np.sqrt(2)*np.sqrt(1-cos(i))
    
    print(v0)
    print(vc)
    print(dV1)
    print(dV2)
    print(dV)
    print(check)

