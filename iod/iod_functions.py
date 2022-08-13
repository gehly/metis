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


def robust_lambert(r0_vect, rf_vect, tof, m, GM, transfer_type, branch):
    '''
    This function implements a robust method to solve Lambert's Problem based 
    on work by Lancaster and Blanchard, and Gooding. 
    
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
    
    
    
    # Initialize and normalize values
    r0_vect = np.reshape(r0_vect, (3,1))
    rf_vect = np.reshape(rf_vect, (3,1))
    tol = 1e-12                            # optimum for numerical noise v.s. actual precision
    r0 = np.linalg.norm(r0_vect)              
    rf = np.linalg.norm(rf_vect)              
    r0_hat = r0_vect/r0                        
    rf_hat = rf_vect/rf                         
    cross_r0rf = np.reshape(np.cross(r0_vect.flatten(), rf_vect.flatten()), (3,1))
    cross_mag = np.linalg.norm(cross_r0rf)
    cross_r0rf_hat = cross_r0rf/cross_mag
    
    # Compute unit vectors in tangential direction
    t0_hat = np.reshape(np.cross(cross_r0rf_hat.flatten(), r0_hat.flatten()), (3,1))
    tf_hat = np.reshape(np.cross(cross_r0rf_hat.flatten(), rf_hat.flatten()), (3,1))
    
    # Compute turn angle
    dtheta = math.acos(max(-1, min(1, np.dot(r0_hat.T, rf_hat))))
    
    # Check for Type I (short) or II (long) transfer and adjust
    if transfer_type == 2:
        dtheta = dtheta - 2.*math.pi
    
    
    

#    % left-branch
#    leftbranch = sign(m); m = abs(m);

    
    # Define constants
    c = np.sqrt(r0**2. + rf**2. - 2.*r0*rf*math.cos(dtheta))
    s = (r0 + rf + c)/2.
    T = np.sqrt(8.*GM/s**3.) * tof
    q = np.sqrt(r0*rf)/s * math.cos(dtheta/2.)
    
    # General formulae for initial values (Gooding)
    T0, dT0, ddT0, dddT0 = LancasterBlanchard(0., q, m)
    Td = T0 - T
    phr = math.fmod(2.*math.atan2(1. - q**2., 2.*q), 2.*math.pi)
    
    # Initial output
    v0_vect = np.reshape([np.nan]*3, (3,1))
    vf_vect = np.reshape([np.nan]*3, (3,1))
    extremal_distances = [np.nan]*2
    
    # Single revolution case
    if m == 0:
        x01 = T0*Td/4./T
        if Td > 0.:
            x0 = x01
        else:
            x01 = Td/(4. - Td)
            x02 = -np.sqrt(-Td/(T + T0/2.))
            W = x01 + 1.7*np.sqrt(2. - phr/math.pi)
            if W >= 0.:
                x03 = x01
            else:
                x03 = x01 + (-W)**(1./16.)*(x02 - x01)
            
            Lambda = 1. + x03*(1. + x01)/2. - 0.03*x03**2.*np.sqrt(1. + x01)
            x0 = Lambda*x03
            
        # This estimate might not give a solution
        if x0 < -1.:
            exit_flag = -1
            return v0_vect, vf_vect, extremal_distances, exit_flag
        
    # Multi-revolution case
    else:
        
        # Determine minimum Tp(x)
        xMpi = 4./(3.*math.pi*(2.*m + 1.))
        if phr < math.pi:
            xM0 = xMpi*(phr/pi)**(1./8.)
        elif phr > math.pi:
            xM0 = xMpi*(2. - (2. - phr/math.pi)**(1./8.))
        else:
            xM0 = 0.
            
        # Use Halley's method
        xM = xM0
        Tp = np.inf
        iters = 0
        while abs(Tp) < tol:
            
            # Increment counter
            iters += 1
            
            # Compute first three derivatives
            dum, Tp, Tpp, Tppp = LancasterBlanchard(xM, q, m)
            
            # New value of xM
            xMp = float(xM)
            xM = xM - 2.*Tp*Tpp / (2.*Tpp**2. - Tp*Tppp)
            
            # Escape clause
            if math.fmod(iters, 7):
                xM = (xMp + xM)/2.
            
            # The method might fail
            if iters > 25:
                exit_flag = -2
                return v0_vect, vf_vect, extremal_distances, exit_flag
            
        # xM should be elliptic (-1 < x < 1)
        # This should be impossible to go wrong
        if xM < -1. or xM > 1.:
            exit_flag = -1
            return v0_vect, vf_vect, extremal_distances, exit_flag
        
        # Corresponding time
        TM, dum1, dum2, dum3 = LancasterBlanchard(xM, q, m)
        
        # T should lie above the minimum T
        if TM > T:
            exit_flag = -1
            return v0_vect, vf_vect, extremal_distances, exit_flag
        
        
        # Find two initial values for second solution (again with lambda-type patch)
        
        # Initial values
        TmTM = T - TM
        T0mTM = T0 - TM
        dum1, Tp, Tpp, dum2 = LancasterBlanchard(xM, q, m)
        
        # First estimate (only if left branch)
        if branch == 'left':
            x = np.sqrt(TmTM/(Tpp/2. + TmTM/(1.-xM)**2.))
            W = xM + x
            W = 4.*W/(4. + TmTM) + (1. - W)**2.
            x0 = x*(1. - (1. + m + (dtheta - 1./2.)) / (1. + 0.15*m)*x*(W/2. + 0.03*x*np.sqrt(W))) + xM
            
            # First estimate might not be able to yield possible solution
            if x0 > 1.:
                exit_flag = -1
                return v0_vect, vf_vect, extremal_distances, exit_flag
        
        # Second estimate
        else:
            if Td > 0.:
                x0 = xM - np.sqrt(TM/(Tpp/2. - TmTM*(Tpp/2./T0mTM - 1./xM**2.)))
            else:
                x00 = Td/(4. - Td)
                W = x00 + 1.7*np.sqrt(2.*(1. - phr))
                if W >= 0.:
                    x03 = x00
                else:
                    x03 = x00 - np.sqrt((-W)**(1./8.))*(x00 + np.sqrt(-Td/(1.5*T0 - Td)))
                W = 4./(4. - Td)
                Lambda = (1. + (1. + m + 0.24*(dtheta - 1./2.)) / (1. + 0.15*m)*x03*(W/2. - 0.03*x03*np.sqrt(W)))
                x0 = x03*Lambda
                
            # Estimate might not give solution
            if x0 < -1.:
                exit_flag = -1
                return v0_vect, vf_vect, extremal_distances, exit_flag
        
    
    
    
    # Find root of Lancaster and Blanchard's function
    # (Halley's method)
    x = x0
    Tx = np.inf
    iters = 0
    
    
    
    
    % find root of Lancaster & Blancard's function
    % --------------------------------------------

    % (Halley's method)
    x = x0; Tx = inf; iterations = 0;
    while abs(Tx) > tol
        % iterations
        iterations = iterations + 1;
        % compute function value, and first two derivatives
        [Tx, Tp, Tpp] = LancasterBlanchard(x, q, m);
        % find the root of the *difference* between the
        % function value [T_x] and the required time [T]
        Tx = Tx - T;
        % new value of x
        xp = x;
        x  = x - 2*Tx*Tp ./ (2*Tp^2 - Tx*Tpp);
        % escape clause
        if mod(iterations, 7), x = (xp+x)/2; end
        % Halley's method might fail
        if iterations > 25, exitflag = -2; return; end
    end

    % calculate terminal velocities
    % -----------------------------

    % constants required for this calculation
    gamma = sqrt(muC*s/2);
    if (c == 0)
        sigma = 1;
        rho   = 0;
        z     = abs(x);
    else
        sigma = 2*sqrt(r1*r2/(c^2)) * sin(dth/2);
        rho   = (r1 - r2)/c;
        z     = sqrt(1 + q^2*(x^2 - 1));
    end

    % radial component
    Vr1    = +gamma*((q*z - x) - rho*(q*z + x)) / r1;
    Vr1vec = Vr1*r1unit;
    Vr2    = -gamma*((q*z - x) + rho*(q*z + x)) / r2;
    Vr2vec = Vr2*r2unit;

    % tangential component
    Vtan1    = sigma * gamma * (z + q*x) / r1;
    Vtan1vec = Vtan1 * th1unit;
    Vtan2    = sigma * gamma * (z + q*x) / r2;
    Vtan2vec = Vtan2 * th2unit;

    % Cartesian velocity
    V1 = Vtan1vec + Vr1vec;
    V2 = Vtan2vec + Vr2vec;

    % exitflag
    exitflag = 1; % (success)

    % also determine minimum/maximum distance
    a = s/2/(1 - x^2); % semi-major axis
    extremal_distances = minmax_distances(r1vec, r1, r1vec, r2, dth, a, V1, V2, m, muC);

    
    
    
    return v0_vect, vf_vect, extremal_distances, exit_flag


def LancasterBlanchard(x, q, m):
    
    # Verify input
    if x < -1.:
        x = abs(x) - 2.
    elif x == -1.:
        x += np.finfo(float).eps
        
    # Compute parameter E
    E = x*x - 1.
    
    # Compute T(x) and derivatives
    if x == 1:
        
        # Parabolic, solutions known exactly
        T = (4./3.)*(1. - q**3.)
        Tp = (4./5.)*(q**5. - 1.)
        Tpp = Tp + (120./70.)*(1. - q**7.)
        Tppp = 3.*(Tpp - Tp) + (2400./1080.)*(q**9. - 1.)
        
    elif abs(x-1) < 1e-2:
        
        # Near-parabolic, compute with series
        sig1, dsigdx1, d2sigdx21, d3sigdx31 = compute_sigmax(-E)
        sig2, dsigdx2, d2sigdx22, d3sigdx32 = compute_sigmax(-E*q*q)
        
        T = sig1 - q**3.*sig2
        Tp = 2.*x*(q**5.*dsigdx2 - dsigdx1)
        Tpp = Tp/x + 4.*x**2.*(d2sigdx21 - q**7.*d2sigdx22)
        Tppp = 3.*(Tpp-Tp/x)/x + 8.*x*x*(q**9.*d3sigdx32 - d3sigdx31)
        
    else:
        
        # All other cases
        y = np.sqrt(abs(E))
        z = np.sqrt(1. + q**2.*E)
        f = y*(z - q*x)
        g = x*z - q*E
        
        if E < 0.:
            d = math.atan2(f, g) + math.pi*m
        elif E == 0.:
            d = 0.
        else:
            d = math.log(max(0., (f+g)))
            

        T = 2.*(x - q*z - d/y)/E
        Tp = (4. - 4.*q**3.*x/z - 3.*x*T)/E
        Tpp = (-4.*q**3./z * (1. - q**2.*x**2./z**2.) - 3.*T - 3.*x*Tp)/E
        Tppp = (4.*q**3./z**2.*((1. - q**2.*x**2./z**2.) + 2.*q**2.*x/z**2.*(z - x)) - 8.*Tp - 7.*x*Tpp)/E

    return T, Tp, Tpp, Tppp
    

def compute_sigmax(y):
    '''
    
    '''
    
    # Twenty-five factors more than enough for 16-digit precision
    an = [4.000000000000000e-001, 2.142857142857143e-001, 4.629629629629630e-002,
          6.628787878787879e-003, 7.211538461538461e-004, 6.365740740740740e-005,
          4.741479925303455e-006, 3.059406328320802e-007, 1.742836409255060e-008,
          8.892477331109578e-010, 4.110111531986532e-011, 1.736709384841458e-012,
          6.759767240041426e-014, 2.439123386614026e-015, 8.203411614538007e-017,
          2.583771576869575e-018, 7.652331327976716e-020, 2.138860629743989e-021,
          5.659959451165552e-023, 1.422104833817366e-024, 3.401398483272306e-026,
          7.762544304774155e-028, 1.693916882090479e-029, 3.541295006766860e-031,
          7.105336187804402e-033]
    

    # powers of y
    powers = [y**exponent for exponent in range(1, 26)]
    
    # Vectorize
    powers = np.reshape(powers, (25, 1))
    an = np.reshape(an, (25, 1))
    deriv_factors = np.reshape(range(1,26), (25,1))
    deriv2_factors = np.reshape(range(0,25), (25,1))
    deriv3_factors = np.reshape(range(-1, 24), (25,1))

    # sigma itself
    sig = float(4./3. + np.dot(powers.T, an))

    # dsigma / dx (derivative)
    first_der = np.reshape(np.insert(powers, 0, 1), (26, 1))
    dsigdx = float(np.dot(np.multiply(deriv_factors, first_der[0:25]).T, an))

    # d2sigma / dx2 (second derivative)
    second_der = np.reshape(np.insert(powers, 0, np.array([1./y, 1.])), (27, 1))
    d2sigdx2 = float(np.dot(np.multiply(np.multiply(deriv_factors, deriv2_factors), second_der[0:25]).T, an))
    
    # d3sigma / dx3 (third derivative)
    third_der = np.reshape(np.insert(powers, 0, np.array([1./y/y, 1./y, 1.])), (28, 1))
    d3sigdx3 = float(np.dot(np.multiply(np.multiply(np.multiply(deriv_factors, deriv2_factors), deriv3_factors), third_der[0:25]).T, an))

    
    return  sig, dsigdx, d2sigdx2, d3sigdx3
    

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





