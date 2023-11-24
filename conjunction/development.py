import numpy as np
import math
import matplotlib.pyplot as plt
import os
import sys
import inspect


filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from utilities import astrodynamics as astro
from utilities import coordinate_systems as coord
from utilities.constants import Re, GME

def compute_initial_conditions(elem_chief, dt_vect, GM=GME):
    
    # Compute mean motion of chief
    a = float(elem_chief[0])
    n = astro.sma2meanmot(a, GM)
    
    # Set up initial conditions in RIC frame
    x0 = 0.
    y0 = -1.
    z0 = 0.
    dx0 = -2.0
    dy0 = 0.500
    dz0 = 1.0
    rho_ric = np.reshape([x0, y0, z0], (3,1))
    drho_ric = np.reshape([dx0, dy0, dz0], (3,1))
    
    x_off = 2*dy0/n
    d = dy0 + 2*n*x0
    rho0 = np.sqrt(x0**2 + y0**2 + z0**2)
    
    print('d', d)
    print('x_off', x_off)
    print('rho0', rho0)
    
    # Convert to ECI and compute deputy orbit    
    Xo_chief = astro.kep2cart(elem_chief, GM)
    rc_vect = Xo_chief[0:3].reshape(3,1)
    vc_vect = Xo_chief[3:6].reshape(3,1)
    rho_eci = coord.ric2eci(rc_vect, vc_vect, rho_ric)
    drho_eci = coord.ric2eci_vel(rc_vect, vc_vect, rho_ric, drho_ric)
    
    
    print(rho_eci)
    print(drho_eci)
    print(np.linalg.norm(rho_eci))
    print(np.linalg.norm(drho_eci))
    
    rd_vect = rc_vect + rho_eci
    vd_vect = vc_vect + drho_eci
    
    Xo_deputy = np.concatenate((rd_vect, vd_vect), axis=0)
    elem_deputy = astro.cart2kep(Xo_deputy, GM)
    
    print(Xo_deputy)
    print(elem_deputy)
    
    # Back and forward propagate, compute differences, and plot
    rho_plot = np.zeros(dt_vect.shape)
    r_plot = np.zeros(dt_vect.shape)
    i_plot = np.zeros(dt_vect.shape)
    c_plot = np.zeros(dt_vect.shape)
    ii = 0
    for dt in dt_vect:
        Xt_chief = astro.element_conversion(Xo_chief, 1, 1, GM, dt)
        Xt_deputy = astro.element_conversion(Xo_deputy, 1, 1, GM, dt)
        
        if ii == 0:
            Xc_output = Xt_chief.copy()
            Xd_output = Xt_deputy.copy()
        
        rc_t = Xt_chief[0:3].reshape(3,1)
        vc_t = Xt_chief[3:6].reshape(3,1)
        rd_t = Xt_deputy[0:3].reshape(3,1)
        
        rho_eci = rd_t - rc_t
        rho_ric = coord.eci2ric(rc_t, vc_t, rho_eci)
        
        rho_plot[ii] = np.linalg.norm(rho_eci)
        r_plot[ii] = float(rho_ric[0])
        i_plot[ii] = float(rho_ric[1])
        c_plot[ii] = float(rho_ric[2])
        
        ii += 1
        
    rho_min = min(rho_plot)
    ind = list(rho_plot).index(rho_min)
    tmin = dt_vect[ind]
    
    print(rho_min)
    print(tmin)
    print(rho_plot[ind-3:ind+4])
        
        
    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(dt_vect, rho_plot, 'k.')
    plt.plot(tmin, rho_min, 'ro')
    plt.ylabel('rho [km]')
    plt.subplot(4,1,2)
    plt.plot(dt_vect, r_plot, 'k.')
    plt.ylabel('r [km]')
    plt.subplot(4,1,3)
    plt.plot(dt_vect, i_plot, 'k.')
    plt.ylabel('i [km]')
    plt.subplot(4,1,4)
    plt.plot(dt_vect, c_plot, 'k.')
    plt.ylabel('c [km]')
    plt.xlabel('Seconds since epoch')
    
    plt.show()
    
    
    
    return Xc_output, Xd_output, tmin, rho_min


def TCA_chebyshev(X1, X2, a, b, N=16, GM=GME):
    '''
    
    References
    ------
    [1] Denenberg, E., "Satellite Closest Approach Calculation Through 
    Chebyshev Proxy Polynomials," Acta Astronautica, 2020.
    
    
    '''
    
    # Determine Chebyshev-Gauss-Lobato node locations (Eq 10)
    jvec = np.arange(0,N+1)
    kvec = jvec.copy()
    tvec = ((b-a)/2.)*(np.cos(np.pi*jvec/N)) + ((b+a)/2.)
    
    # print(jvec)
    # print(tvec)
    
    # Compute function values to find roots of
    # In order to minimize rho, we seek zeros of first derivative
    # f(t) = dot(rho_vect, rho_vect)
    # g(t) = df/dt = 2*dot(drho_vect, rho_vect)
    gvec = np.zeros(tvec.shape)
    rvec = np.zeros(tvec.shape)
    ivec = np.zeros(tvec.shape)
    cvec = np.zeros(tvec.shape)
    jj = 0
    for t in tvec:
        X1_t = astro.element_conversion(X1, 1, 1, GM, t)
        X2_t = astro.element_conversion(X2, 1, 1, GM, t)
        rc_vect = X1_t[0:3].reshape(3,1)
        vc_vect = X1_t[3:6].reshape(3,1)
        rd_vect = X2_t[0:3].reshape(3,1)
        vd_vect = X2_t[3:6].reshape(3,1)
        
        rho_eci = rd_vect - rc_vect
        drho_eci = vd_vect - vc_vect
        rho_ric = coord.eci2ric(rc_vect, vc_vect, rho_eci)
        drho_ric = coord.eci2ric_vel(rc_vect, vc_vect, rho_ric, drho_eci)
        
        gvec[jj] = float(2*np.dot(rho_ric.T, drho_ric))
        rvec[jj] = float(rho_ric[0])
        ivec[jj] = float(rho_ric[1])
        cvec[jj] = float(rho_ric[2])
        jj += 1
        
    # print(gvec)
    
    # Compute interpolation matrix Fmat (Eq 12-13)
    pvec = np.ones(N+1,)
    pvec[0] = 2.
    pvec[N] = 2.
    jkmat = np.dot(jvec.reshape(N+1,1),kvec.reshape(1,N+1))
    Cmat = np.cos(np.pi/N*jkmat)
    Fmat = np.zeros((N+1, N+1))
    for jj in range(N+1):
        Fmat[jj,:] = (2./(pvec[jj]*pvec*N))
        
    # print(Fmat)
    # print(Cmat)
    Fmat = np.multiply(Fmat, Cmat)
    
    # print(Fmat)
    
    # Compute aj coefficients (Eq 14)
    aj_vec = np.dot(Fmat, gvec.reshape(N+1,1))
    
    
    # Test approximation
    gn_test = np.zeros(tvec.shape)
    tt = 0
    for t in tvec:
        
        # TODO Check use of (b+a) or (b-a) Eq 9 and Eq 24 seem to disagree
        # From testing, this seems to be correct, should test more cases
        x = (2*t - (b+a))/(b-a)   
        
        gn = 0.
        for jj in range(N+1):
            Tj = np.cos(jj * math.acos(x))
            gn += aj_vec[jj]*Tj
            
        gn_test[tt] = float(gn)
        tt += 1
        
    # print('\n\n')
    # print(gvec)
    # print(gn_test)
    
    # print(gvec - gn_test)
    
        
    # Compute the companion matrix (Eq 18)
    Amat = np.zeros((N,N))
    Amat[0,1] = 1.
    Amat[-1,:] = -aj_vec[0:N].flatten()/(2*aj_vec[N])
    Amat[-1,-2] += 0.5
    for jj in range(1,N-1):
        Amat[jj,jj-1] = 0.5
        Amat[jj,jj+1] = 0.5
    
    # print(Amat)
    # print(aj_vec)
    
    # Compute eigenvalues
    eig, dum = np.linalg.eig(Amat)
    eig_real = np.asarray([np.real(ee) for ee in eig if (np.isreal(ee) and ee >= -1. and ee <= 1.)])
    roots = (b+a)/2. + eig_real*(b-a)/2.
    
    print(eig)
    print(roots)
    
    # For each value in roots, compute Chebyshev proxy polynomial for RIC
    # Compute aj coefficients (Eq 14)
    ar_vec = np.dot(Fmat, rvec.reshape(N+1,1))
    ai_vec = np.dot(Fmat, ivec.reshape(N+1,1))
    ac_vec = np.dot(Fmat, cvec.reshape(N+1,1))
    rho_min = np.inf
    for t in roots:
        
        
        #TODO Check use of (b+a) or (b-a) Eq 9 and Eq 24 seem to disagree
        # From testing, (b+a) as in Eq 9 seems to be correct but should check more cases
        x = (2*t - (b+a))/(b-a) 
        
        r_n = 0.
        i_n = 0.
        c_n = 0.
        for jj in range(N+1):            
            Tj = np.cos(jj * math.acos(x))
            r_n += ar_vec[jj]*Tj
            i_n += ai_vec[jj]*Tj
            c_n += ac_vec[jj]*Tj
            
        rho_n = float(np.sqrt(r_n**2 + i_n**2 + c_n**2))
        if rho_n < rho_min:
            rho_min = rho_n
            tmin = t
            
            
    print('tmin', tmin)
    print('rho_min', rho_min)
        

    
    return tmin, rho_min


def TCA_test():
    
    # Setup initial states and compute truth (Two-Body Dynamics)
    elem_chief = np.array([Re+550., 1e-4, 98.6, 30., 40., 50.])
    dt_vect = np.arange(-100., 100.1, 0.1)
    
    Xc, Xd, tmin, rho_min = compute_initial_conditions(elem_chief, dt_vect)
    
    a = 0.
    b = dt_vect[-1] - dt_vect[0]
    # a = dt_vect[0]
    # b = dt_vect[-1]
    N = 16
    
    
    # # Adjust times to run with different a,b
    # a = dt_vect[0]
    # b = dt_vect[-1]
    # Xc = astro.element_conversion(Xc, 1, 1, GME, 100.)
    # Xd = astro.element_conversion(Xd, 1, 1, GME, 100.)
    
    
    tmin, rho_min = TCA_chebyshev(Xc, Xd, a, b, N)
    
    # Test if this is actual min
    ttest = np.arange(tmin-0.1, tmin+0.1, 0.001)
    rho_vec = np.zeros(ttest.shape)
    ii = 0
    for tt in ttest:
        Xc_test = astro.element_conversion(Xc, 1, 1, GME, tt)
        Xd_test = astro.element_conversion(Xd, 1, 1, GME, tt)
        rho_vec[ii] = np.linalg.norm(Xc_test[0:3]-Xd_test[0:3])
        ii += 1
        
    rho_min2 = min(rho_vec)
    ind = list(rho_vec).index(rho_min2)
    tmin2 = ttest[ind]
    
    print('rho_min2', rho_min2)
    print('tmin', tmin)
    
    
    return


if __name__ == '__main__':
    
    plt.close('all')
    
    TCA_test()

