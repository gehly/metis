import numpy as np
from math import exp, pi, sin, cos, asin, atan2
import os
import sys
from pathlib import Path

sys.path.append('../')

import utilities.conversions as conv
from utilities.constants import GM, Re, wE

###############################################################################
# This file contains functions to perform numerical integration using odeint
# Functions:
#  int_twobody
#  int_twobody_stm
#  int_twobody_ukf
#  int_twobody_diff_entropy
#  int_twobody_srp
#  compute_srp
###############################################################################

###############################################################################
# Orbit Propagation Routines
###############################################################################

def int_twobody(X, t, spacecraftConfig, forcesCoeff, surfaces):
    '''
    This function works with odeint to propagate object assuming
    simple two-body dynamics.  No perturbations included.

    Parameters
    ------
    X : 6 element list
      cartesian state vector (Inertial Frame)
    t : m element list
      vector of times when output is desired
    args : tuple
        additional arguments

    Returns
    ------
    dX : 6 element list
      state derivative vector
    '''
    

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])

    # Compute radius
    r = np.linalg.norm([x, y, z])

    # Derivative vector
    dX = [0.]*6

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3
    dX[4] = -GM*y/r**3
    dX[5] = -GM*z/r**3

    return dX


def int_twobody_ukf(X, t, spacecraftConfig, forcesCoeff, surfaces):
    '''
    This function works with odeint to propagate object assuming
    simple two-body dynamics.  No perturbations included.

    Parameters
    ------
    X : (n*(2n+1)) element list
      initial condition vector of cartesian state and sigma points
    t : m element list
      vector of times when output is desired
    inputs : dictionary
     input parameters

    Returns
    ------
    dX : (n*(2n+1)) element list
      derivative vector

    '''

    # Initialize
    dX = [0]*len(X)
    n = int((-1 + np.sqrt(1. + 8.*len(X)))/4.)

    for ind in range(0, 2*n+1):

        # Pull out relevant values from X
        x = float(X[ind*n])
        y = float(X[ind*n + 1])
        z = float(X[ind*n + 2])
        dx = float(X[ind*n + 3])
        dy = float(X[ind*n + 4])
        dz = float(X[ind*n + 5])

        # Compute radius
        r = np.linalg.norm([x, y, z])

        # Solve for components of dX
        dX[ind*n] = dx
        dX[ind*n + 1] = dy
        dX[ind*n + 2] = dz

        dX[ind*n + 3] = -GM*x/r**3
        dX[ind*n + 4] = -GM*y/r**3
        dX[ind*n + 5] = -GM*z/r**3

    return dX


def int_twobody_diff_entropy(X, t, inputs):
    '''
    This function works with odeint to propagate object assuming
    simple two-body dynamics.  No perturbations included.
    Partials for the STM dynamics are included. Differential Entropy is
    included in the state vector and STM.

    Parameters
    ------
    X : (n+n^2) element list
      initial condition vector of entropy and cartesian state and
      STM (Inertial Frame)
    t : m element list
      vector of times when output is desired
    inputs : dictionary
      input parameters

    Returns
    ------
    dX : (n+n^2) element list
      derivative vector
    '''

    # Input data
    GM = inputs['GM']

    # State Vector
    n = 7
    ej = X[0]
    x = X[1]
    y = X[2]
    z = X[3]
    dx = X[4]
    dy = X[5]
    dz = X[6]

    # Compute radius
    r = np.linalg.norm([x, y, z])

    # Find elements of A matrix
    xx_cf = -GM/r**3 + 3.*GM*x**2/r**5
    xy_cf = 3.*GM*x*y/r**5
    xz_cf = 3.*GM*x*z/r**5
    yy_cf = -GM/r**3 + 3.*GM*y**2/r**5
    yx_cf = xy_cf
    yz_cf = 3.*GM*y*z/r**5
    zz_cf = -GM/r**3 + 3.*GM*z**2/r**5
    zx_cf = xz_cf
    zy_cf = yz_cf

    # Generate A matrix
    A = np.zeros((n, n))

    A[0,3] = 1.
    A[1,4] = 1.
    A[2,5] = 1.

    A[3,0] = xx_cf
    A[3,1] = xy_cf
    A[3,2] = xz_cf

    A[4,0] = yx_cf
    A[4,1] = yy_cf
    A[4,2] = yz_cf

    A[5,0] = zx_cf
    A[5,1] = zy_cf
    A[5,2] = zz_cf

    # Derivative vector
    dX = np.zeros((7, 1))

    dX[0] = np.trace(A)

    dX[1] = dx
    dX[2] = dy
    dX[3] = dz

    dX[4] = -GM*x/r**3
    dX[5] = -GM*y/r**3
    dX[6] = -GM*z/r**3

    dX = dX.flatten()

    return dX




#def int_twobody_stm(X, t, inputs):
#    '''
#    This function works with odeint to propagate object assuming
#    simple two-body dynamics.  No perturbations included.
#    Partials for the STM dynamics are included.
#
#    Parameters
#    ------
#    X : (n+n^2) element list
#      initial condition vector of cartesian state and STM (Inertial Frame)
#    t : m element list
#      vector of times when output is desired
#    inputs : dictionary
#      input parameters
#
#    Returns
#    ------
#    dX : (n+n^2) element list
#      derivative vector
#    '''
#
#    # Input data
#    GM = inputs['GM']
#
#    # Compute number of states
#    n = (-1 + np.sqrt(1 + 4*len(X)))/2
#
#    # State Vector
#    x = X[0]
#    y = X[1]
#    z = X[2]
#    dx = X[3]
#    dy = X[4]
#    dz = X[5]
#
#    # Compute radius
#    r = np.linalg.norm([x, y, z])
#
#    # Find elements of A matrix
#    xx_cf = -GM/r**3 + 3.*GM*x**2/r**5
#    xy_cf = 3.*GM*x*y/r**5
#    xz_cf = 3.*GM*x*z/r**5
#    yy_cf = -GM/r**3 + 3.*GM*y**2/r**5
#    yx_cf = xy_cf
#    yz_cf = 3.*GM*y*z/r**5
#    zz_cf = -GM/r**3 + 3.*GM*z**2/r**5
#    zx_cf = xz_cf
#    zy_cf = yz_cf
#
#    # Generate A matrix
#    A = np.zeros((n, n))
#
#    A[0,3] = 1.
#    A[1,4] = 1.
#    A[2,5] = 1.
#
#    A[3,0] = xx_cf
#    A[3,1] = xy_cf
#    A[3,2] = xz_cf
#
#    A[4,0] = yx_cf
#    A[4,1] = yy_cf
#    A[4,2] = yz_cf
#
#    A[5,0] = zx_cf
#    A[5,1] = zy_cf
#    A[5,2] = zz_cf
#
#    # Compute STM components dphi = A*phi
#    phi_mat = np.reshape(X[n:], (n, n))
#    dphi_mat = np.dot(A, phi_mat)
#    dphi_v = np.reshape(dphi_mat, (n**2, 1))
#
#    # Derivative vector
#    dX = np.zeros((n+n**2, 1))
#
#    dX[0] = dx
#    dX[1] = dy
#    dX[2] = dz
#
#    dX[3] = -GM*x/r**3
#    dX[4] = -GM*y/r**3
#    dX[5] = -GM*z/r**3
#
#    dX[n:] = dphi_v
#
#    dX = dX.flatten()
#
#    return dX



def int_twobody_j2(X, t, inputs):
    '''
    This function works with odeint to propagate object assuming
    simple two-body dynamics and perturbations due to J2.

    Parameters
    ------
    X : 6 element list
      cartesian state vector (Inertial Frame)
    t : m element list
      vector of times when output is desired
    inputs : dictionary
      input parameters

    Returns
    ------
    dX : 6 element list
      state derivative vector
    '''

    # Input data
    GM = inputs['GM']
    J2 = inputs['J2']
    Re = inputs['Re']

    # State Vector
    n = 6
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])

    # Compute radius
    r_vect = np.array([[x], [y], [z]])
    r = np.linalg.norm(r_vect)

    # Derivative vector
    dX = np.zeros((n, 1))

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3. - 1.5*J2*Re**2.*GM*((x/r**5.) - (5.*x*z**2./r**7.))
    dX[4] = -GM*y/r**3. - 1.5*J2*Re**2.*GM*((y/r**5.) - (5.*y*z**2./r**7.))
    dX[5] = -GM*z/r**3. - 1.5*J2*Re**2.*GM*((3.*z/r**5.) - (5.*z**3./r**7.))

    dX = dX.flatten()

    return dX


def int_twobody_j2_ukf(X, t, inputs):
    '''
    This function works with odeint to propagate object assuming
    simple two-body dynamics and perturbations due to J2. Outputs sigma
    points for use in UKF covariance propagation.

    Parameters
    ------
    X : (n*(2n+1)) element list
      initial condition vector of cartesian state and sigma points
    t : m element list
      vector of times when output is desired
    inputs : dictionary
     input parameters

    Returns
    ------
    dX : (n*(2n+1)) element list
      derivative vector

    '''

    # Break out inputs
    GM = inputs['GM']
    J2 = inputs['J2']
    Re = inputs['Re']

    # Initialize
    n = int((-1 + np.sqrt(1. + 8.*len(X)))/4.)    
    dX = np.zeros(((2*n+1)*n, 1))

    for ind in range(0, 2*n+1):

        # Pull out relevant values from X
        x = float(X[ind*n])
        y = float(X[ind*n + 1])
        z = float(X[ind*n + 2])
        dx = float(X[ind*n + 3])
        dy = float(X[ind*n + 4])
        dz = float(X[ind*n + 5])

        # Compute radius
        r = np.linalg.norm([x, y, z])

        # Solve for components of dX
        dX[ind*n] = dx
        dX[ind*n + 1] = dy
        dX[ind*n + 2] = dz

        dX[ind*n + 3] = -GM*x/r**3. - 1.5*J2*Re**2.*GM*((x/r**5.) -
                                                        (5.*x*z**2./r**7.))
        dX[ind*n + 4] = -GM*y/r**3. - 1.5*J2*Re**2.*GM*((y/r**5.) -
                                                        (5.*y*z**2./r**7.))
        dX[ind*n + 5] = -GM*z/r**3. - 1.5*J2*Re**2.*GM*((3.*z/r**5.) -
                                                        (5.*z**3./r**7.))

    dX = dX.flatten()

    return dX


def int_twobody_j2_drag(X, t, inputs):
    '''
    This function works with odeint to propagate object assuming
    simple two-body dynamics and perturbations due to J2 and drag.

    Parameters
    ------
    X : 6 element list
      cartesian state vector (Inertial Frame)
    t : m element list
      vector of times when output is desired
    inputs : dictionary
      input parameters

    Returns
    ------
    dX : 6 element list
      state derivative vector
    '''

    # Input data
    GM = inputs['GM']
    J2 = inputs['J2']
    Cd = inputs['Cd']
    Re = inputs['Re']
    dtheta = inputs['dtheta']
    rho0 = inputs['rho0']
    ro = inputs['ro']
    H = inputs['H']
    A_m = inputs['A_m']
    
    

    # State Vector
    n = 6
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])

    # Compute radius
    r_vect = np.array([[x], [y], [z]])
    r = np.linalg.norm(r_vect)
    
    # Compute drag component
    # Find vector va of spacecraft relative to atmosphere
    v_vect = np.array([[dx], [dy], [dz]])
    w_vect = np.array([[0.], [0.], [dtheta]])
    va_vect = v_vect - np.cross(w_vect, r_vect, axis=0)
    va = np.linalg.norm(va_vect)
    va_x = float(va_vect[0])
    va_y = float(va_vect[1])
    va_z = float(va_vect[2])
    
    drag = -0.5*A_m*Cd*rho0*exp(-(r-ro)/H)
    
#    print r_vect
#    print v_vect
#    print va_vect
#    print va
#    print drag
#    mistake
    
    x_drag = drag*va*va_x
    y_drag = drag*va*va_y
    z_drag = drag*va*va_z
    
#    print x_drag
#    print y_drag
#    print z_drag
#    mistake

    # Derivative vector
    dX = np.zeros((n, 1))

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3. + x_drag - 1.5*J2*Re**2.*GM*((x/r**5.) - (5.*x*z**2./r**7.))
    dX[4] = -GM*y/r**3. + y_drag - 1.5*J2*Re**2.*GM*((y/r**5.) - (5.*y*z**2./r**7.))
    dX[5] = -GM*z/r**3. + z_drag - 1.5*J2*Re**2.*GM*((3.*z/r**5.) - (5.*z**3./r**7.))
    
    dX = dX.flatten()
    
#    print dX
#    mistake

    return dX


def int_twobody_j2_drag_stm(X, t, inputs):
    '''
    This function works with odeint to propagate object assuming
    simple two-body dynamics and perturbations due to J2 and drag.
    Includes components of the state transition matrix.

    Parameters
    ------
    X : 42 element list
      cartesian state vector (Inertial Frame)
    t : m element list
      vector of times when output is desired
    inputs : dictionary
      input parameters

    Returns
    ------
    dX : 42 element list
      state derivative vector
    '''

    # Input data
    GM = inputs['GM']
    J2 = inputs['J2']
    Cd = inputs['Cd']
    Re = inputs['Re']
    dtheta = inputs['dtheta']
    rho0 = inputs['rho0']
    ro = inputs['ro']
    H = inputs['H']
    A_m = inputs['A_m']

    # State Vector
    n = 6
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])

    # Compute radius
    r_vect = np.array([[x], [y], [z]])
    r = np.linalg.norm(r_vect)
    
    # Compute drag component
    # Find vector va of spacecraft relative to atmosphere
    v_vect = np.array([[dx], [dy], [dz]])
    w_vect = np.array([[0.], [0.], [dtheta]])
    va_vect = v_vect - np.cross(w_vect, r_vect, axis=0)
    va = np.linalg.norm(va_vect)
    va_x = float(va_vect[0])
    va_y = float(va_vect[1])
    va_z = float(va_vect[2])
    
    drag = -0.5*A_m*Cd*rho0*exp(-(r-ro)/H)
    
    x_drag = drag*va*va_x
    y_drag = drag*va*va_y
    z_drag = drag*va*va_z
    
    # Find elements of A matrix
    xx_cf = -GM/r**3. + 3.*GM*x**2./r**5.
    xx_drag = drag*((-x*va*va_x/(H*r)) - dtheta*va_y*va_x/va)
    xx_J2 = -1.5*J2*GM*Re**2./r**5. - 7.5*J2*GM*Re**2./r**7.*(-x**2. - z**2. + 7.*x**2.*z**2./r**2.)

    xy_cf = 3.*GM*x*y/r**5.
    xy_drag = drag*((-y*va*va_x/(H*r)) + dtheta*va_x**2./va + va*dtheta)
    xy_J2 = -7.5*x*y/r**7. * J2*Re**2.*GM*(-1. + 7.*z**2./r**2.)

    xz_cf = 3.*GM*x*z/r**5.
    xz_drag = drag*(-z*va*va_x/(H*r))
    xz_J2 = -7.5*x*z/r**7. * J2*Re**2.*GM*(-3. + 7.*z**2./r**2.)

    yy_cf = -GM/r**3. + 3.*GM*y**2./r**5.
    yy_drag = drag*((-y*va*va_y/(H*r)) + dtheta*va_x*va_y/va)
    yy_J2 = -1.5*J2*GM*Re**2./r**5. - 7.5*J2*GM*Re**2./r**7.*(-y**2. - z**2. + 7.*y**2.*z**2./r**2.)

    yx_cf = xy_cf
    yx_drag = drag*((-x*va*va_y/(H*r)) - dtheta*va_y**2./va - va*dtheta)
    yx_J2 = xy_J2

    yz_cf = 3.*GM*y*z/r**5.
    yz_drag = drag*(-z*va*va_y/(H*r))
    yz_J2 = -7.5*y*z/r**7. * J2*Re**2.*GM*(-3. + 7.*z**2./r**2.)

    zz_cf = -GM/r**3. + 3.*GM*z**2./r**5.
    zz_drag = drag*(-z*va*va_z/(H*r))
    zz_J2 = -4.5*J2*Re**2.*GM/r**5. - 7.5*J2*Re**2.*GM/r**7.*(-6.*z**2. + 7.*z**4./r**2.)

    zx_cf = xz_cf
    zx_drag = drag*((-x*va*va_z/(H*r)) - dtheta*va_y*va_z/va)
    zx_J2 = xz_J2

    zy_cf = yz_cf
    zy_drag = drag*((-y*va*va_z/(H*r)) + dtheta*va_x*va_z/va)
    zy_J2 = yz_J2
    
    # Generate A matrix using partials from above
    A = np.zeros((n,n))

    A[0,3] = 1. 
    A[1,4] = 1. 
    A[2,5] = 1.

    A[3,0] = xx_cf + xx_drag + xx_J2
    A[3,1] = xy_cf + xy_drag + xy_J2
    A[3,2] = xz_cf + xz_drag + xz_J2
    A[3,3] = drag*(va_x**2./va + va)
    A[3,4] = drag*(va_y*va_x/va)
    A[3,5] = drag*(va_z*va_x/va)      # Note, va_z = dz

    A[4,0] = yx_cf + yx_drag + yx_J2
    A[4,1] = yy_cf + yy_drag + yy_J2
    A[4,2] = yz_cf + yz_drag + yz_J2
    A[4,3] = drag*(va_y*va_x/va)
    A[4,4] = drag*(va_y**2./va + va)
    A[4,5] = drag*(va_y*va_z/va)       # Note, va_z = dz

    A[5,0] = zx_cf + zx_drag + zx_J2
    A[5,1] = zy_cf + zy_drag + zy_J2
    A[5,2] = zz_cf + zz_drag + zz_J2
    A[5,3] = drag*(va_x*va_z/va)
    A[5,4] = drag*(va_y*va_z/va)
    A[5,5] = drag*(va_z**2./va + va)
    
    # Compute STM components dphi = A*phi
    phi_mat = np.reshape(X[n:], (n, n))
    dphi_mat = np.dot(A, phi_mat)
    dphi_v = np.reshape(dphi_mat, (n**2, 1))

    # Derivative vector
    dX = np.zeros((n+n**2, 1))

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3 + x_drag - 1.5*J2*Re**2.*GM*((x/r**5.) - (5.*x*z**2./r**7.))
    dX[4] = -GM*y/r**3 + y_drag - 1.5*J2*Re**2.*GM*((y/r**5.) - (5.*y*z**2./r**7.))
    dX[5] = -GM*z/r**3 + z_drag - 1.5*J2*Re**2.*GM*((3.*z/r**5.) - (5.*z**3./r**7.))
    
    dX[n:] = dphi_v

    dX = dX.flatten()
    

    return dX


def int_twobody_srp(X, t, inputs):
    '''
    This function works with odeint to propagate object assuming
    simple two-body dynamics and SRP perturbation.  C_r and A_m are
    given parameters, not included in the state vector.

    Parameters
    ------
    X : n element list
      initial condition vector of cartesian state (Inertial Frame)
    t : m element list
      vector of times when output is desired
    inputs : dictionary
      input parameters

    Returns
    ------
    dX : n element list
      derivative vector
    '''

    # Input data
    GM = inputs['GM']
    Cr = inputs['Cr']
    A_m = inputs['A_m']

    # State Vector
    n = 6
    x = X[0]
    y = X[1]
    z = X[2]
    dx = X[3]
    dy = X[4]
    dz = X[5]

    # Compute radius
    r_vect = np.reshape([x, y, z], (3, 1))
    r = np.linalg.norm(r_vect)

    # Compute perturbation due to srp
    beta = Cr * A_m
    a_srp = compute_srp(r_vect, beta, inputs, t)

    # Derivative vector
    dX = np.zeros((n, 1))

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3 + a_srp[0]
    dX[4] = -GM*y/r**3 + a_srp[1]
    dX[5] = -GM*z/r**3 + a_srp[2]

    dX = dX.flatten()

    return dX


def compute_srp(r_vect, beta, inputs, t):
    '''
    This function computes the acceleration due to solar radiation pressure.

    Parameters
    ------
    r_vect : 3x1 numpy array
      position vector in ECI
    beta : float
      SRP parameter (Cr*A/m) [km^2/kg]
    inputs : dictionary
      input parameters
    t : float
      current time of day [sec]
    Returns
    ------
    a_srp : 3x1 numpy array
      acceleration vector from SRP perturbing force
    '''

    # Break out inputs
    d0 = inputs['d0']
    phi_sun = inputs['phi_sun']
    c = inputs['c']
    start_tdays = inputs['start_tdays']

    # Compute current solar position in ECI
    # Time in days since J2000
    tdays = start_tdays + t/86400.

    # Convert to centuries and compute mean longitude of the sun
    tcent = tdays/(365.25*100.)
    lam = 280.460 + 36000.771*tcent
    lam = (lam % 360.) * pi/180.  # rad

    # Compute mean anomaly of sun
    M = 357.5277233 + 35999.05034*tcent
    M = (M % 360.) * pi/180.  # rad

    # Compute ecliptic long/lat
    lam_ec = lam + (1.914666471*sin(M) + 0.019994643*sin(2.*M))*pi/180.  # rad

    # Compute distance to sun in AU and obliquity of ecliptic plane
    r = 1.000140612 - 0.016708617*cos(M) - 0.000139589*cos(2.*M)  # AU
    ep = (23.439291 - 0.0130042*tcent) * pi/180.

    # Compute sun position vector in Mean Equator of Date (MOD) Frame
    r_sun = r*d0*np.array([[cos(lam_ec)], [cos(ep)*sin(lam_ec)],
                           [sin(ep)*sin(lam_ec)]])
    rg = np.norm(r_sun)

    # Compute ra/dec of sun
    dec = asin(sin(ep)*sin(lam_ec))
    ra = atan2((cos(ep)*sin(lam_ec)/cos(dec)), (cos(lam_ec)/cos(dec)))

    # Convert to ECI
    sun_eci = conv.radec2eci(ra, dec, rg)

    # Compute current position vector from sun
    r_fromsun = r_vect - sun_eci
    d = np.norm(r_fromsun)
    u_sun = -r_fromsum/d

    # SRP acceleration
    a_srp = -(phi_sun/(c*(d/d0)**2.))*beta*(1./1000.)*u_sun  # km/s^2

    return a_srp
