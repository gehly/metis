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
    
    Source code from https://github.com/rodyo/FEX-Lambert
    
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
    
    # Fast Lambert
    v0_vect, vf_vect, extremal_distances, exit_flag = \
        fast_lambert(r0_vect, rf_vect, tof, m, GM, transfer_type, branch)
        
    # If not successful, run robust solver
    if exit_flag < 0:
        v0_vect, vf_vect, extremal_distances, exit_flag = \
            robust_lambert(r0_vect, rf_vect, tof, m, GM, transfer_type, branch)
    
            




    
    
    return v0_vect, vf_vect, extremal_distances, exit_flag


def fast_lambert(r0_vect, rf_vect, tof, m, GM, transfer_type, branch):
    '''
    This function implements a computationally efficient but less robust 
    approach to solve Lambert's Problem developed by Dario Izzo (ESA).
    
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
        -1 : fail
    '''
    
        
    # Initialize
    tol = 1e-14
    max_iters = 15
    exit_flag = -1
    r0_vect = np.reshape(r0_vect, (3,1))
    rf_vect = np.reshape(rf_vect, (3,1))
    
    # Normalize units
    r0 = np.linalg.norm(r0_vect)
    v0 = np.sqrt(GM/r0)
    T = r0/v0
    
    r0_vect = r0_vect/r0
    rf_vect = rf_vect/r0
    tf = tof/T
    logt = math.log(tf)
    
    # Check non-dimensional geometry
    rf_norm = np.linalg.norm(rf_vect)
    dtheta = math.acos(max(-1, min(1, np.dot(r0_vect.T, rf_vect)/rf_norm)))
    
    
    # Check for Type I (short) or II (long) transfer and adjust
    type_factor = 1.
    if transfer_type == 2:
        dtheta = 2.*math.pi - dtheta
        type_factor = -1.
        
    # Derived non-dimensional quantities
    c = np.sqrt(1. + rf_norm**2. - 2.*rf_norm*math.cos(dtheta)) # chord
    s = (1. + rf_norm + c)/2.                                   # semi-parameter
    a_min = s/2.                                                # min energy ellipse SMA
    Lambda = np.sqrt(rf_norm)*math.cos(dtheta/2.)/s             # Lambda parameter from Battin

    r_cross_vect = np.array([[float(r0_vect[1]*rf_vect[2] - r0_vect[2]*rf_vect[1])],
                             [float(r0_vect[2]*rf_vect[0] - r0_vect[0]*rf_vect[2])],
                             [float(r0_vect[0]*rf_vect[1] - r0_vect[1]*rf_vect[0])]])

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
            
        
    # Initial guess
    xx = np.array([inn1, inn2])
    aa = a_min/(1. - np.multiply(xx, xx))
    bbeta = np.asarray([type_factor * 2. * math.asin(np.sqrt((s-c)/2./ai)) for ai in aa])
    aalpha = np.asarray([2.*math.acos(xi) for xi in xx])
    
    # Evaluate the time of flight via Lagrange expression
    alpha_term = aalpha - np.asarray([math.sin(ai) for ai in aalpha])
    beta_term = bbeta - np.asarray([math.sin(bi) for bi in bbeta])
    y_term = alpha_term - beta_term + 2.*math.pi*m
    
    y12 = np.multiply(aa, np.multiply(np.sqrt(aa), y_term))
    
    # Initial estimate for y
    if m == 0:
        y1 = math.log(y12[0]) - logt
        y2 = math.log(y12[1]) - logt
    else:
        y1 = float(y12[0]) - tf
        y2 = float(y12[0]) - tf
        
    # Solve for x
    # Newton-Raphson iteration
    err = 1e6
    iters = 0
    xnew = 0.
    while err > tol:
        
        # Increment iterations
        iters += 1
        
        # Compute xnew
        xnew = (x1*y2 - y1*x2)/(y2 - y1)
        
        if m == 0:
            x = math.exp(xnew) - 1.
        else:
            x = math.atan(xnew)*2./math.pi
        
        a = a_min/(1. - x**2.)
        
        # Ellipse
        if x < 1.:
            beta = type_factor * 2.*math.asin(np.sqrt((s-c)/2./a))
            alpha = 2.*math.acos(max(-1., min(1., x)))
        
        # Hyperbola
        else:
            beta = type_factor * 2.*math.asinh(np.sqrt((s-c)/(-2.*a)))
            alpha = 2.*math.acosh(x)
            
            
        # Evaluate time of flight via Lagrange expression
        if a > 0.:
            tof_new = a*np.sqrt(a)*((alpha - math.sin(alpha) - (beta - math.sin(beta)) + 2.*math.pi*m))
        else:
            tof_new = -a*np.sqrt(-a)*((math.sinh(alpha) - alpha) - (math.sinh(beta) - beta))
            
        # New value of y
        if m == 0:
            ynew = math.log(tof_new) - logt
        else:
            ynew = tof_new - tf
            
        # Save previous and current values for next iteration
        x1 = x2
        x2 = xnew
        y1 = y2
        y2 = ynew
        err = abs(x1 - xnew)
        
        # Exit condition
        if iters > max_iters:
            exit_flag = -1
            break
        
    # Convert converged value of x
    if m == 0:
        x = math.exp(xnew) - 1.
    else:
        x = math.atan(xnew)*2./math.pi
        
    # The solution has been evaluated in terms of log(x+1) or tan(x*pi/2), we
    # now need the conic. As for transfer angles near to pi the Lagrange-
    # coefficients technique goes singular (dg approaches a zero/zero that is
    # numerically bad) we here use a different technique for those cases. When
    # the transfer angle is exactly equal to pi, then the ih unit vector is not
    # determined. The remaining equations, though, are still valid.
    
    # Solution for semi-major axis
    a = a_min/(1. - x**2.)
    
    # Calculate psi
    # Ellipse
    if x < 1.:
        beta = type_factor * 2.*math.asin(np.sqrt((s-c)/2./a))
        alpha = 2.*math.acos(max(-1., min(1., x)))
        psi = (alpha - beta)/2.
        eta2 = 2.*a*math.sin(psi)**2./s
        eta = np.sqrt(eta2)
        
    # Hyperbola
    else:
        beta = type_factor * 2.*math.asinh(np.sqrt((s-c)/(-2.*a)))
        alpha = 2.*math.acosh(x)
        psi = (alpha - beta)/2.
        eta2 = -2.*a*math.sinh(psi)**2./s
        eta = np.sqrt(eta2)
        
    # Unit of normalized unit vector
    ih = type_factor*r_cross_hat
    
    # Unit vector for rf_vect
    r0_vect_hat = r0_vect/np.linalg.norm(r0_vect)
    rf_vect_hat = rf_vect/rf_norm
    
    # Cross products    
    cross1 = np.array([[float(ih[1]*r0_vect_hat[2] - ih[2]*r0_vect_hat[1])],
                       [float(ih[2]*r0_vect_hat[0] - ih[0]*r0_vect_hat[2])],
                       [float(ih[0]*r0_vect_hat[1] - ih[1]*r0_vect_hat[0])]])
    
    cross2 = np.array([[float(ih[1]*rf_vect_hat[2] - ih[2]*rf_vect_hat[1])],
                       [float(ih[2]*rf_vect_hat[0] - ih[0]*rf_vect_hat[2])],
                       [float(ih[0]*rf_vect_hat[1] - ih[1]*rf_vect_hat[0])]])
    
    
    
    # Radial and tangential components for initial velocity
    Vr1 = 1./eta/np.sqrt(a_min) * (2.*Lambda*a_min - Lambda - x*eta)
    Vt1 = np.sqrt(rf_norm/a_min/eta2 * math.sin(dtheta/2.)**2.)
    
    # Radial and tangential components for final velocity
    Vt2 = Vt1/rf_norm
    Vr2 = (Vt1 - Vt2)/math.tan(dtheta/2.) - Vr1
    
    # Velocity vectors
    v0_vect = (Vr1*r0_vect + Vt1*cross1)*v0
    vf_vect = (Vr2*rf_vect_hat + Vt2*cross2)*v0
    
    # Exit flag - success
    exit_flag = 1

    # Compute min/max distance to central body
    extremal_distances = \
        compute_extremal_dist(r0_vect*r0, rf_vect*r0, v0_vect, vf_vect, dtheta,
                              a*r0, m, GM, transfer_type)
       
    return v0_vect, vf_vect, extremal_distances, exit_flag


#def robust_lambert(r0_vect, rf_vect, tof, m, GM, transfer_type, branch):
#    '''
#    This function implements a robust method to solve Lambert's Problem based 
#    on work by Lancaster and Blanchard, and Gooding. 
#    
#    Parameters
#    ------
#    r0_vect : 3x1 numpy array
#        position vector at t0 [km]
#    rf_vect : 3x1 numpy array
#        position vector at tf [km]
#    tof : float
#        time of flight [sec]
#    m : int
#        number of complete orbit revolutions
#    GM : float
#        graviational parameter of central body [km^3/s^2]
#        
#    Returns
#    ------
#    v0_vect : 3x1 numpy array
#        velocity vector at t0 [km/s]
#    vf_vect : 3x1 numpy array
#        velocity vector at tf [km/s]
#    extremal_distances : list
#        min and max distance from central body during orbit [km]
#    exit_flag : int
#        +1 : success
#        -1 : fail
#    '''
#    
#    
#    
#    # Initialize and normalize values
#    r0_vect = np.reshape(r0_vect, (3,1))
#    rf_vect = np.reshape(rf_vect, (3,1))
#    tol     = 1e-12                            % optimum for numerical noise v.s. actual precision
#    r0      = np.linalg.norm(r0_vect)              % magnitude of r1vec
#    rf      = np.linalg.norm(rf_vect)              % magnitude of r2vec
#    r0_hat  = r0_vect/r0                        % unit vector of r1vec
#    rf_hat  = rf_vect/rf                         % unit vector of r2vec
#    crsprod = np.cross(r0_vect, rf_vect);           % cross product of r1vec and r2vec
#    mcrsprd = sqrt(crsprod*crsprod.');          % magnitude of that cross product
#    th1unit = cross(crsprod/mcrsprd, r1unit);   % unit vectors in the tangential-directions
#    th2unit = cross(crsprod/mcrsprd, r2unit);
#    % make 100.4% sure it's in (-1 <= x <= +1)
#    dth = acos( max(-1, min(1, (r1vec*r2vec.')/r1/r2)) ); % turn angle
#
#    % if the long way was selected, the turn-angle must be negative
#    % to take care of the direction of final velocity
#    longway = sign(tf); tf = abs(tf);
#    if (longway < 0), dth = dth-2*pi; end
#
#    % left-branch
#    leftbranch = sign(m); m = abs(m);
#
#    % define constants
#    c  = sqrt(r1^2 + r2^2 - 2*r1*r2*cos(dth));
#    s  = (r1 + r2 + c) / 2;
#    T  = sqrt(8*muC/s^3) * tf;
#    q  = sqrt(r1*r2)/s * cos(dth/2);
#    
#    
#    
#    
#    
#    return v0_vect, vf_vect, extremal_distances, exit_flag
    

def compute_extremal_dist(r0_vect, rf_vect, v0_vect, vf_vect, dtheta, a, m, GM,
                          transfer_type):
    '''
    
    '''
    
    # Default, min/max of r0, rf
    r0 = np.linalg.norm(r0_vect)
    rf = np.linalg.norm(rf_vect)
    r0_vect_hat = r0_vect/r0
    rf_vect_hat = rf_vect/rf
    minimum_distance = min(r0, rf)
    maximum_distance = max(r0, rf)

    # Eccentricity vector 
    h0_vect = np.array([[float(r0_vect[1]*v0_vect[2] - r0_vect[2]*v0_vect[1])],
                        [float(r0_vect[2]*v0_vect[0] - r0_vect[0]*v0_vect[2])],
                        [float(r0_vect[0]*v0_vect[1] - r0_vect[1]*v0_vect[0])]])
    
    cross1 =  np.array([[float(v0_vect[1]*h0_vect[2] - v0_vect[2]*h0_vect[1])],
                        [float(v0_vect[2]*h0_vect[0] - v0_vect[0]*h0_vect[2])],
                        [float(v0_vect[0]*h0_vect[1] - v0_vect[1]*h0_vect[0])]])
    
    e0_vect = cross1/GM - r0_vect/r0
    e = np.linalg.norm(e0_vect)
    e0_vect_hat = e0_vect/e
    
    # Apses
    periapsis = a*(1. - e)
    apoapsis = np.inf
    if e < 1.:
        apoapsis = a*(1. + e)
        
    # Check if the trajectory goes through periapsis
    if m > 0:
        
        # Multirev case, must be elliptical and pass through both periapsis and
        # apoapsis
        minimum_distance = periapsis
        maximum_distance = apoapsis
        
    else:
        
        # Compute true anomaly at t0 and tf
        pm0 = np.sign(r0*r0*np.dot(e0_vect.T, v0_vect) - np.dot(r0_vect.T, e0_vect)*np.dot(r0_vect.T, v0_vect))
        pmf = np.sign(rf*rf*np.dot(e0_vect.T, vf_vect) - np.dot(rf_vect.T, e0_vect)*np.dot(rf_vect.T, vf_vect))

        theta0 = pm0 * math.acos(max(-1, min(1, np.dot(r0_vect_hat.T, e0_vect_hat))))
        thetaf = pmf * math.acos(max(-1, min(1, np.dot(rf_vect_hat.T, e0_vect_hat))))
        
        if theta0*thetaf < 0.:
            
            # Initial and final positions are on opposite sides of symmetry axis
            # Minimum and maximum distance depends on dtheta and true anomalies
            if abs(abs(theta0) + abs(thetaf) - dtheta) < 5.*np.finfo(float).eps:
                minimum_distance = periapsis
            
            # This condition can only be false for elliptic cases, and if it is
            # false, the orbit has passed through apoapsis
            else:
                maximum_distance = apoapsis
                
        else:
            
            # Initial and final positions are on the same side of symmetry axis
            # If it is a Type II transfer (longway) then the object must
            # pass through both periapsis and apoapsis
            if transfer_type == 2:
                minimum_distance = periapsis
                if e < 1.:
                    maximum_distance = apoapsis
                    
                    
    extremal_distances = [minimum_distance, maximum_distance]

    return extremal_distances


def gauss_iod(tk_list, Yk_list, sensor_params):
    
    
    return


def gooding_iod():
    
    
    return





