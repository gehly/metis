import numpy as np
from math import pi, sin, cos, tan, asin, exp, fmod
import matplotlib.pyplot as plt

from numerical_integration import rk4


###############################################################################
# This file contains code to implement the Semi-Analytic Liu Theory (SALT)
# general perturbations orbit propagator.  The code models the secular and
# long period effects of J2, J3, and atmospheric drag (using the standard
# atmosphere model).
#
# References
#  1. Liu and Alford, "Semi-Analytic Theory for a Close-Earth Artificial 
#     Satellite," Guidance and Control, 1980.
#
#  2. Vallado, "Fundamentals of Astrodynamics and Applications," 4th ed., 2013.
#
#  3. Chao, "Applied Orbit Perturbations and Maintenance," 2nd ed., 2005.
#
###############################################################################



def lgwt(N,a,b):
    '''
    This function returns the locations and weights of nodes to use for
    Gauss-Legendre Quadrature for numerical integration.
    
    Adapted from MATLAB code by Greg von Winckel
    
    Parameters
    ------
    N : int
        number of nodes
    a : float
        lower limit of integral
    b : float
        upper limit of integral
    
    Returns
    ------
    x_vect : 1D numpy array
        node locations
    w_vect : 1D numpy array
        node weights
    
    '''
    
    xu = np.linspace(-1, 1, N)
    
    # Initial Guess
    y=np.cos((2*np.arange(0,N)+1)*pi/(2*(N-1)+2))+(0.27/N)*np.sin(pi*xu*(N-1)/(N+1))
    y=y.reshape(len(y),1)
    
    # Legendre-Gauss Vandermonde Matrix
    L=np.zeros((N,N+1))
    
    # Derivative of LGVM
    
    # Compute the zeros of the N+1 Legendre Polynomial
    # using the recursion relation and the Newton-Raphson method
    y0=2.
    eps = np.finfo(float).eps
    
    # Iterate until new points are uniformly within epsilon of old points
    while max(abs(y-y0)) > eps:
        
        L[:,0] = 1.
        
        L[:,1] = y.flatten()
        
        for k in range(1,N):
            
            L1 = (2*(k+1)-1)*y.flatten()
            L2 = L[:,k].flatten()
            L3 = L[:,k-1].flatten()
            
            L[:,k+1] = (np.multiply(L1, L2) - k*L3)/(k+1)

        y2 = np.multiply(y.flatten(), y.flatten())
        Lp1=(N+1)*( L[:,N-1]- np.multiply(y.flatten(), L[:,N].flatten() ))  
        Lp = np.multiply(Lp1, 1./(1-y2))

        y0 = y.copy()
        y = y0 - np.reshape(np.multiply(L[:,N].flatten(), 1./Lp), (len(y0), 1))
        
    # Linear map from[-1,1] to [a,b]
    x_vect = (a*(1-y)+b*(1+y))/2
    x_vect = x_vect.flatten()
    
    # Compute the weights
    y2 = np.multiply(y, y)
    Lp2 = np.multiply(Lp, Lp)
    w_vect = (b-a)/(np.multiply((1-y2.flatten()), Lp2.flatten()))*((N+1)/N)**2.
    
    return x_vect, w_vect


def atmosphere_lookup(h):
    '''
    This function acts as a lookup table for atmospheric density reference
    values, reference heights, and scale heights for a range of different 
    altitudes from 100 - 1000+ km.  Values from Vallado 4th ed. Table 8-4.
    
    Parameters
    ------
    h : float
        altitude [km]
    
    Returns
    ------
    rho0 : float
        reference density [kg/km^3]
    h0 : float
        reference altitude [km]
    H : float
        scale height [km]

    '''
    
    if h <= 100:
        # Assume at this height we have re-entered atmosphere
        rho0 = 0
        h0 = 1
        H = 1
    elif h < 110:
        rho0 = 5.297e-7 * 1e9  # kg/km^3
        h0 = 100.    # km
        H = 5.877    # km    
    elif h < 120:
        rho0 = 9.661e-8 * 1e9  # kg/km^3
        h0 = 110.    # km
        H = 7.263    # km   
    elif h < 130:
        rho0 = 2.438e-8 * 1e9  # kg/km^3
        h0 = 120.    # km
        H = 9.473    # km   
    elif h < 140: 
        rho0 = 8.484e-9 * 1e9  # kg/km^3
        h0 = 130.    # km
        H = 12.636   # km       
    elif h < 150:
        rho0 = 3.845e-9 * 1e9  # kg/km^3
        h0 = 140.    # km
        H = 16.149   # km       
    elif h < 180:
        rho0 = 2.070e-9 * 1e9  # kg/km^3
        h0 = 150.    # km
        H = 22.523   # km       
    elif h < 200:
        rho0 = 5.464e-10 * 1e9  # kg/km^3
        h0 = 180.    # km
        H = 29.740   # km     
    elif h < 250:
        rho0 = 2.789e-10 * 1e9  # kg/km^3
        h0 = 200.    # km
        H = 37.105   # km   
    elif h < 300:
        rho0 = 7.248e-11 * 1e9  # kg/km^3
        h0 = 250.    # km
        H = 45.546   # km       
    elif h < 350:
        rho0 = 2.418e-11 * 1e9  # kg/km^3
        h0 = 300.    # km
        H = 53.628   # km       
    elif h < 400:
        rho0 = 9.518e-12 * 1e9  # kg/km^3
        h0 = 350.    # km
        H = 53.298   # km       
    elif h < 450:
        rho0 = 3.725e-12 * 1e9   # kg/km^3
        h0 = 400.    # km
        H = 58.515   # km     
    elif h < 500:
        rho0 = 1.585e-12 * 1e9   # kg/km^3
        h0 = 450.    # km
        H = 60.828   # km   
    elif h < 600:
        rho0 = 6.967e-13 * 1e9   # kg/km^3
        h0 = 500.    # km
        H = 63.822   # km
    elif h < 700:
        rho0 = 1.454e-13 * 1e9   # kg/km^3
        h0 = 600.    # km
        H = 71.835   # km
    elif h < 800:
        rho0 = 3.614e-14 * 1e9   # kg/km^3
        h0 = 700.    # km
        H = 88.667   # km       
    elif h < 900:
        rho0 = 1.17e-14 * 1e9    # kg/km^3
        h0 = 800.    # km
        H = 124.64   # km       
    elif h < 1000:
        rho0 = 5.245e-15 * 1e9   # kg/km^3
        h0 = 900.    # km
        H = 181.05   # km       
    else:
        rho0 = 3.019e-15 * 1e9   # kg/km^3
        h0 = 1000.   # km
        H = 268.00   # km
    
    
    return rho0, h0, H


def int_salt_grav_drag(t, X, params):
    '''
    This function computes the derivatives for the Semi-Analytic Liu Theory
    orbit propagator, to be used with a numerical integrator such as RK4.

    Parameters
    ------
    t : float
        current time
    X : nx1 numpy array
        state vector
    params : dictionary
        extra parameters for integrator

    Returns
    -------
    dX : nx1 numpy array
        derivative vector
    
    '''
    
    # Retrieve parameters
    J2 = params['J2']
    J3 = params['J3']
    Re = params['Re']
    GM = params['GM']
        
    # Retrieve states
    a = float(X[0])
    e = float(X[1])
    i = float(X[2])
    RAAN = float(X[3])
    w = float(X[4])
    
    # Check re-entry condition
    if a - Re < 100:
        print('Re-entry at t = ',  t/(86400*365.25))
        dX = np.zeros(5,)
        return dX
    
    # Compute orbit params
    n = np.sqrt(GM/a**3.)
    p = a*(1.-e**2.)

    # Compute dadt and dedt for drag using Gauss Quadrature
    dadt_drag, dedt_drag = compute_gauss_quad_drag(X, params)

    
    # Compute dadt and dedt for gravity perturbations
    dadt_grav = 0.

    dedt_grav = -(3./8.)*n*J3*(Re/p)**3. * \
        (4. - 5.*sin(i)**2.)*(1.-e**2.)*sin(i)*cos(w)

    didt_grav = (3./8.)*n*J3*(Re/p)**3. * e * \
        (4. - 5.*sin(i)**2.)*cos(i)*cos(w)
    
    dRAANdt_grav = -(3./2.)*n*(Re/p)**2. * (J2*cos(i) + (J3/4.)*(Re/p) * 
                     (15.*sin(i)**2. - 4.)*(e*(1./tan(i))*sin(w)))
    
    dwdt_grav = (3./4.)*n*J2*(Re/p)**2.*(4. - 5.*sin(i)**2.) + \
        (3./8.)*n*J3*(Re/p)**3.*sin(w) * ((4. - 5.*sin(i)**2.) * 
         ((sin(i)**2. - e**2.*cos(i)**2.)/(e*sin(i))) + 
         2.*sin(i)*(13. - 15.*sin(i)**2.)*e)


    # Set up final derivative vector
    dX = np.zeros(5,)
    
    dX[0] = dadt_drag + dadt_grav    
    dX[1] = dedt_drag + dedt_grav    
    dX[2] = didt_grav    
    dX[3] = dRAANdt_grav    
    dX[4] = dwdt_grav
   
    
    return dX


def salt_setup():
    
    params = {}
    Xo = np.array([7000., 0.01, 28.3, 80.5, 173.])
    Xo[2] = asin(2./np.sqrt(5))*180/pi
    params['GM'] = 3.986e5
    params['J2'] = 1.0826e-3
    params['J3'] = -2.5327e-6*0
    params['Re'] = 6378.1363
    params['wE'] = 7.2921158553e-5
    params['rho0'] = 3.019e-4
    params['r0'] = 7378.1363
    params['H'] = 268
    params['Cd'] = 2.2*0
    params['A_m'] = 1e-8
    params['N'] = 20   
    
    return Xo, params


def run_salt_propagator(Xo, params):
    
    # Setup RK4 integrator
    tin = np.array([0, 1*365.25*86400])
    Xo[2] = Xo[2]*pi/180
    Xo[3] = Xo[3]*pi/180
    Xo[4] = Xo[4]*pi/180
    intfcn = int_salt_grav_drag
    
    a = Xo[0]
    P = 2.*pi*np.sqrt(a**3./params['GM'])
    params['step'] = P
    
    tout, yout = rk4(intfcn, tin, Xo, params)
    
    # Analytic result on Omega
    omega0 = Xo[3]
    dt = tin[1] - tin[0]
    n = np.sqrt(params['GM']/a**3.)
    e = Xo[1]
    p = a*(1.-e**2.)
    J2 = params['J2']
    Re = params['Re']
    dRAANdt = -3.*n*J2*Re**2./(2.*p**2.)*cos(Xo[2])
    RAANf_analytic = omega0 + dRAANdt*dt
    RAANf_analytic = fmod(RAANf_analytic, 2*pi)*180/pi
    
    RAANf_numeric = fmod(yout[-1,3], 2*pi)*180/pi

    return tout, yout


def plot_salt_propagator(tout, yout):
    
    plt.close('all')
    
    t_yrs = tout/(365.25*86400)

    for ii in range(len(tout)):
        if yout[ii,2] < 0:
            yout[ii,2] = fmod(yout[ii,2], pi) + pi
        
        if yout[ii,2] > pi:
            yout[ii,2] = fmod(yout[ii,2], pi)
        
        if yout[ii,3] < 0:
            yout[ii,3] = fmod(yout[ii,3], 2*pi) + 2*pi
        
        if yout[ii,3] > 2*pi:
            yout[ii,3] = fmod(yout[ii,3], 2*pi)
        
        if yout[ii,4] < 0:
            yout[ii,4] = fmod(yout[ii,4], 2*pi) + 2*pi
        
        if yout[ii,4] > 2*pi:
            yout[ii,4] = fmod(yout[ii,4], 2*pi)
            
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t_yrs, yout[:,0], 'k')
    plt.ylabel('SMA [km]')
    plt.subplot(2,1,2)
    plt.plot(t_yrs, yout[:,1], 'k')
    plt.ylabel('Eccentricity')
    plt.xlabel('Time [years]')
    
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_yrs, yout[:,2]*180/pi, 'k')
    plt.ylabel('Inc [deg]')
    plt.subplot(3,1,2)
    plt.plot(t_yrs, yout[:,3]*180/pi, 'k')
    plt.ylabel('RAAN [deg]')
    plt.subplot(3,1,3)
    plt.plot(t_yrs, yout[:,4]*180/pi, 'k')
    plt.ylabel('AoP [deg]')
    plt.xlabel('Time [years]')

    plt.show()
    
    
    return


def compute_gauss_quad_drag(X, params):
    '''
    This function computes the derivatives of semi-major axis and eccentricity
    resulting from drag forces as modeled by SALT.  The derivative requires
    numerical evaluation of an integral, which is done using Gauss-Legendre
    quadrature.
    
    Parameters
    ------
    X : 5x1 numpy array
        state vector [a, e, i, RAAN, w]
        units of distance in km, angles in radians
    params : dictionary
        additional input parameters including number of nodes to use in
        quadrature and physical parameters such as Cd, A/m ratio, etc.
    
    Returns
    ------
    dadt_drag : float
        instantaneous change in SMA wrt time [km/s]
    dedt_drag : float
        instantaneous change in eccentricity wrt time [1/s]
        
    '''
    
#    print('compute quad')
    
    
    # Retrieve values from state vector and params
    a = X[0]
    e = X[1]
    i = X[2]
    
    GM = params['GM']
    Re = params['Re']
    wE = params['wE']
    Cd = params['Cd']
    A_m = params['A_m']
    N = params['N']
    
    # Compute orbit params
    n = np.sqrt(GM/a**3.)
    p = a*(1.-e**2.)
    B = Cd*A_m
    
    # Compute locations and weights of nodes
    theta_vect, w_vect = lgwt(N, 0., 2.*pi)


    # Compute function value at node locations
    dadt_vect = np.array([])
    dedt_vect = np.array([])
    for theta_i in theta_vect:
        
        # Compute orbit and density parameters for this node
        r = p/(1. + e*cos(theta_i))  # orbit radius
        h = r - Re
        rho0, h0, H = atmosphere_lookup(h)
        rho = rho0*exp(-(h-h0)/H)
        eterm = 1. + e**2. + 2.*e*cos(theta_i)
        V = ((GM/p)*eterm)**(1./2.) * \
            (1.-((1.-e**2.)**(3./2.)/eterm)*(wE/n)*cos(i))
        
        # Compute function values at current node location
        dadt_i = -(B/(2.*pi))*rho*V*(r**2./(a*(1.-e**2)**(3./2.))) * \
            (eterm - wE*np.sqrt(a**3.*(1.-e**2.)/GM)*cos(i))
            
        dedt_i = -(B/(2.*pi))*rho*V*(e + cos(theta_i) - 
                   r**2.*wE*cos(i)/(2.*np.sqrt(GM*a*(1.-e)**2.)) * 
                   (2.*(e + cos(theta_i)) - e*sin(theta_i)**2.)*(r/a)**2. * 
                   (1.-e**2.)**(-1./2.))
    
        
        # Store in vector
        dadt_vect = np.append(dadt_vect,dadt_i)
        dedt_vect = np.append(dedt_vect,dedt_i)
        
    # Compute weighted output
    dadt_drag = np.dot(dadt_vect, w_vect)
    dedt_drag = np.dot(dedt_vect, w_vect)

    
    return dadt_drag, dedt_drag



if __name__ == '__main__':
    
#    N = 10
#    a = 0
#    b = 2*pi
#    x_vect, w_vect = lgwt(N,a,b)
    
    Xo, params = salt_setup()
    
    tout, yout = run_salt_propagator(Xo, params)
    
    
    plot_salt_propagator(tout, yout)
    
    
    
    
    
#    # Test drag quadrature
#    fdir = 'D:\documents\\teaching\\unsw_orbital_mechanics\code\salt_working2'
#    fname = os.path.join(fdir, 'salt_leo_10yr.pkl')
#    pklFile = open(fname, 'rb')
#    data = pickle.load(pklFile)
#    Xo = data[0]
#    params = data[1]
#    pklFile.close()
#    
#    Xo[2] = Xo[2]*pi/180
#    Xo[3] = Xo[3]*pi/180
#    Xo[4] = Xo[4]*pi/180
#    params['N'] = 2
#    dadt_drag, dedt_drag = compute_gauss_quad_drag(Xo, params)
#    
#    
#    B = params['Cd'] * params['A_m']
#    r = Xo[0]
#    h = r - params['Re']
#    rho0, h0, H = atmosphere_lookup(h)
#    rho = rho0*exp(-(h-h0)/H)
#    n = np.sqrt(params['GM']/Xo[0]**3.)
#    a = Xo[0]
#    wE = params['wE']
#    i = Xo[2]
#    
#    print(B)
#    print(rho)
#    print(a)
#    print(i)
#    
#    dadt_drag_analytic = -B*rho*n*a**2*(1-wE/n*cos(i))**2
#    
#    rel_error = (dadt_drag - dadt_drag_analytic)/dadt_drag_analytic
#    
#    print(dadt_drag)
#    print(dadt_drag_analytic)
#    print(rel_error)
        


