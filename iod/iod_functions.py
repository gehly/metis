import numpy as np
import math
from scipy.integrate import odeint, ode
import sys
import os
import inspect
from datetime import datetime, timedelta

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

import dynamics.numerical_integration as numint
import utilities.astrodynamics as astro


###############################################################################
# Classical (Deterministic) Methods
###############################################################################


def lambert_iod():
    
    
    return


def multirev_lambert(r0_vect, rf_vect, tof, m, GM, transfer_type, branch):
    '''
    This function implements two methods to solve Lambert's Problem, a fast 
    method developed by Dr Izzo of ESA and a robust method based on work by
    Lancaster and Blanchard [2], and Gooding [3]. This code is written 
    following a MATLAB version written by Rody Oldenhuis, copyright below.
    
    Parameters
    ------
    r0_vect : 3x1 numpy array
        position vector at t0 [km]
    rf_vect : 3x1 numpy array
        position vector at tf [km]
    tof : float
        time of flight [sec]
    m : int
        number of complete orbit revolutions
    GM : float
        graviational parameter of central body [km^3/s^2]
        
    Returns
    ------
    v0_vect : 3x1 numpy array
        velocity vector at t0 [km/s]
    vf_vect : 3x1 numpy array
        velocity vector at tf [km/s]
    extremal_distances : list
        min and max distance from central body during orbit [km]
    exit_flag : int
        +1 : success
        -1 : problem has no solution
        -2 : both algorithms failed (should not occur)
        
    References
    ------
    [1] Izzo, D. ESA Advanced Concepts team. Code used available in MGA.M, on
         http://www.esa.int/gsp/ACT/inf/op/globopt.htm. Last retreived Nov, 2009.
         (broken link)
     
    [2] Lancaster, E.R. and Blanchard, R.C. "A unified form of Lambert's theorem."
         NASA technical note TN D-5368,1969.
     
    [3] Gooding, R.H. "A procedure for the solution of Lambert's orbital boundary-value
         problem. Celestial Mechanics and Dynamical Astronomy, 48:145�165,1990.
    
    
    Copyright
    ------
    Copyright (c) 2018, Rody Oldenhuis
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    
    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    
    The views and conclusions contained in the software and documentation are those
    of the authors and should not be interpreted as representing official policies,
    either expressed or implied, of this project.
    
    '''
    
    
    
    
    
    
    return v0_vect, vf_vect, extremal_distances, exit_flag


def fast_lambert(r0_vect, rf_vect, tof, m, GM, transfer_type, branch):
    '''
    This function implements the Izzo Lambert solver (fast but less robust).
    
    
    Parameters
    ------
    r0_vect : 3x1 numpy array
        position vector at t0 [km]
    rf_vect : 3x1 numpy array
        position vector at tf [km]
    tof : float
        time of flight [sec]
    m : int
        number of complete orbit revolutions
    GM : float
        graviational parameter of central body [km^3/s^2]
        
    Returns
    ------
    v0_vect : 3x1 numpy array
        velocity vector at t0 [km/s]
    vf_vect : 3x1 numpy array
        velocity vector at tf [km/s]
    extremal_distances : list
        min and max distance from central body during orbit [km]
    exit_flag : int
        +1 : success
        -1 : problem has no solution
        -2 : both algorithms failed (should not occur)
    
    '''
    
    # Initialize
    tol = 1e-14
    r0_vect = np.reshape(r0_vect, (3,1))
    rf_vect = np.reshape(rf_vect, (3,1))
    
    # Normalize units
    r0 = np.linalg.norm(r0)
    v0 = np.sqrt(GM/r0)
    T = r0/v0
    
    r0_vect = r0_vect/r0
    rf_vect = rf_vect/r0
    tf = tf/T
    logt = math.log(tf)
    
    # Check non-dimensional geometry
    rf_norm = np.linalg.norm(rf_vect)
    dth = math.acos(max(-1, min(1, np.dot(r0_vect.T, rf_vect)/rf_norm)))
    
    # Check for Type II and adjust
    if transfer_type == 2:
        dth = 2.*math.pi - dth
        
    # Derived non-dimensional quantities
    c = np.sqrt(1. + rf_norm**2. - 2.*rf_norm*math.cos(dth))    # chord
    s = (1. + rf_norm + c)/2.                                   # semi-parameter
    a_min = s/2.                                                # min energy ellipse SMA
    Lambda = np.sqrt(rf_norm)*math.cos(dth/2.)/s                # Lambda parameter from Battin
    r_cross_vect = np.cross(r0_vect, rf_vect)
    r_cross = np.linalg.norm(r_cross_vect)
    r_cross_hat = r_cross_vect/r_cross                          # unit vector
    
    # Setup initial values
    if m == 0:
        
        # Single revolution (1 solution)
        inn1 = -0.5233              # first initial guess
        inn2 = 0.5233               # second initial guess
        x1 = math.log(1. + inn1)    # transformed first initial guess
        x2 = math.log(1. + inn2)    # transformed first second guess
        
    else:
        
        # Multirev case, select right or left branch
        if branch == 'left':
            inn1 = -0.5234          # first initial guess
            inn2 = -0.2234          # second initial guess
        
        if branch == 'right':
            inn1 = 0.7234           # first initial guess
            inn2 = 0.5234           # second initial guess
            
        x1 = math.tan(inn1 + math.pi/2.)    # transformed first initial guess
        x2 = math.tan(inn2 + math.pi/2.)    # transformed first second guess
            
        
    
        
    

    
    
    
    
    return v0_vect, vf_vect, extremal_distances, exit_flag





def gauss_iod(tk_list, Yk_list, sensor_params):
    
    
    return


def gooding_iod():
    
    
    return





