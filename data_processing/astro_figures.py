import numpy as np
from math import pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

sys.path.append('../')

from utilities.astrodynamics import mean2ecc, ecc2true, element_conversion
from utilities.astrodynamics import cart2kep, kep2cart



def plot_mean2ecc():
    
    e_list = [0., 0.2, 0.4, 0.6, 0.8, 0.99]
    M_list = list(np.arange(-pi+0.001, pi+0.001, 0.01))
    
    E_array = np.zeros((len(e_list), len(M_list)))
    
    for e in e_list:
        for M in M_list:
            ii = e_list.index(e)
            jj = M_list.index(M)
            
            E = mean2ecc(M, e)
            if E > pi:
                E -= 2*pi
            
            E_array[ii,jj] = E
            
    color=iter(plt.cm.rainbow(np.linspace(0,1,len(e_list))))
    plt.figure()
    for e in e_list:
        plt.plot(M_list, E_array[e_list.index(e),:], c=next(color), label='e = ' + str(e))
        
    plt.legend()
    plt.xlabel('Mean Anomaly [rad]')
    plt.ylabel('Eccentric Anomaly [rad]')
    
    plt.grid()
            
    plt.show()
    
    
    
    return


def plot_orbit_energy():
    
    mu = 3.986e5
    v = np.linspace(0, 12, 1000)
    r = np.linspace(6378, 50000, 1000)
    E = np.zeros((len(v), len(r)))
    
    for ii in range(len(v)):
        vi = v[ii]
        for jj in range(len(r)):
            rj = r[jj]
            
            E[ii,jj] = (vi**2)/2. - mu/rj
    
    print(E.shape)
    print(len(v))
    print(len(r))
    
    fig=plt.figure()
    ax = fig.gca(projection='3d')
    ax.contour3D(r, v, E, 50)
    
    plt.show()
    
    
    return


def plot_twobody_prop():
    
    # Set up elements for class
    mu = 3.986e5
    Re = 6378
    
    P = 86164.1/2
    a = ((P/(2*pi))**2*mu)**(1./3.)
    n = np.sqrt(mu/a**3.)
    rp = Re+300.
    e = 1. - (rp/a)
    i = 63.4
    RAAN = 75.
    w = -90
    
    # Anomaly angles in rad
    M0 = 0.*pi/180
    E0 = mean2ecc(M0,e)
    theta0 = ecc2true(E0,e)
    
    # Convert to deg
    theta0 *= 180/pi
    
    # Compute initial Cartesian pos/vel
    elem = [a,e,i,RAAN,w,theta0]
    Xo = kep2cart(elem)
    
    print(Xo)
    
    # Set up propagation
    tvec = np.arange(0.,2.*86401.,60.)
    print(tvec)
    
    tdays = tvec/86400.
    xvec = np.zeros(tdays.shape)
    yvec = np.zeros(tdays.shape)
    zvec = np.zeros(tdays.shape)
    dxvec = np.zeros(tdays.shape)
    dyvec = np.zeros(tdays.shape)
    dzvec = np.zeros(tdays.shape)
    avec = np.zeros(tdays.shape)
    evec = np.zeros(tdays.shape)
    ivec = np.zeros(tdays.shape)
    Ovec = np.zeros(tdays.shape)
    wvec = np.zeros(tdays.shape)
    theta_vec = np.zeros(tdays.shape)
    Evec = np.zeros(tdays.shape)
    Mvec = np.zeros(tdays.shape)
    energy_vec = np.zeros(tdays.shape)
    
    ii = 0
    for t in tvec:
        
        # Compute new mean anomaly [rad]
        M = M0 + n*(t-tvec[0])
        while M > 2*pi:
            M -= 2*pi
        
        # Convert to true anomaly [rad]
        E = mean2ecc(M,e)
        theta = ecc2true(E,e)  
        
        # Convert anomaly angles to deg
        M *= 180/pi
        E *= 180/pi
        theta *= 180/pi
        
        elem = [a,e,i,RAAN,w,theta]
        cart = kep2cart(elem)
        
        xvec[ii] = cart[0]
        yvec[ii] = cart[1]
        zvec[ii] = cart[2]
        dxvec[ii] = cart[3]
        dyvec[ii] = cart[4]
        dzvec[ii] = cart[5]
        
        avec[ii] = elem[0]
        evec[ii] = elem[1]
        ivec[ii] = elem[2]
        Ovec[ii] = elem[3]
        wvec[ii] = elem[4]
        Mvec[ii] = M
        Evec[ii] = E
        theta_vec[ii] = theta
        if theta_vec[ii] < 0:
            theta_vec[ii] += 360.
        
        energy_vec[ii] = -mu/(2.*avec[ii])
        
        
        
        
        ii += 1
    
    
    plt.figure()    
    plt.subplot(3,1,1)
    plt.title('Size and Shape Parameters')
    plt.plot(tdays, energy_vec, 'k.')
    plt.ylabel('Energy [$km^2$/$s^2$]')
    plt.subplot(3,1,2)
    plt.plot(tdays, avec, 'k.')
    plt.ylabel('SMA [km]')
    plt.subplot(3,1,3)
    plt.plot(tdays, evec, 'k.')
    plt.ylabel('Eccentricity')
    plt.xlabel('Time [days]')
    
    plt.figure()    
    plt.subplot(3,1,1)
    plt.title('Orientation Parameters')
    plt.plot(tdays, ivec, 'k.')
    plt.ylabel('Inclination [deg]')
    plt.subplot(3,1,2)
    plt.plot(tdays, Ovec, 'k.')
    plt.ylabel('RAAN [deg]')
    plt.subplot(3,1,3)
    plt.plot(tdays, wvec, 'k.')
    plt.ylabel('AoP [deg]')
    plt.xlabel('Time [days]')
    
    plt.figure()    
    plt.subplot(3,1,1)
    plt.title('Anomaly Angles')
    plt.plot(tdays, Mvec, 'k.')
    plt.ylabel('Mean[deg]')
    plt.subplot(3,1,2)
    plt.plot(tdays, Evec, 'k.')
    plt.ylabel('Eccentric[deg]')
    plt.subplot(3,1,3)
    plt.plot(tdays, theta_vec, 'k.')
    plt.ylabel('True [deg]')
    plt.xlabel('Time [days]')
    
    plt.figure()    
    plt.subplot(3,1,1)
    plt.title('Cartesian Position [ECI]')
    plt.plot(tdays, xvec, 'k.')
    plt.ylabel('X [km]')
    plt.subplot(3,1,2)
    plt.plot(tdays, yvec, 'k.')
    plt.ylabel('Y [km]')
    plt.subplot(3,1,3)
    plt.plot(tdays, zvec, 'k.')
    plt.ylabel('Z [km]')
    plt.xlabel('Time [days]')
    
    plt.figure()    
    plt.subplot(3,1,1)
    plt.title('Cartesian Velocity [ECI]')
    plt.plot(tdays, dxvec, 'k.')
    plt.ylabel('dX [km/s]')
    plt.subplot(3,1,2)
    plt.plot(tdays, dyvec, 'k.')
    plt.ylabel('dY [km/s]')
    plt.subplot(3,1,3)
    plt.plot(tdays, dzvec, 'k.')
    plt.ylabel('dZ [km/s]')
    plt.xlabel('Time [days]')
    
    
    plt.show()
    
        
        
    
    return
    


if __name__ == '__main__':
    
    plt.close('all')
    
    plot_twobody_prop()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    