import numpy as np
from scipy.integrate import odeint
from scipy.integrate import ode
import matplotlib.pyplot as plt
import sys

sys.path.append('../')

from utilities.astrodynamics import element_conversion
from utilities.constants import GME
from numerical_integration import rk4, rkf78, dopri87
from integration_functions import ode_twobody


def test_twobody():
    
    # Set up initial conditions and integration parameters
#    x_in = np.reshape([7000., 0.1, 10., 100., 40., 95.], (6,1))
#    x_in = [-1371.563, 2837.540, -7043.244, -6.711557, -2.544285, 0.282073]
    x_in = [26311.2, 0.7489727, 63.4, 0., 270., 30.]
    Xo = element_conversion(x_in, 0, 1)
    SMA = element_conversion(Xo, 1, 0)[0]
    t_in = [0., 24.*3600.]
    
    intfcn = ode_twobody
    solver_integrator = 'dop853'
    int_tol = 1e-12
    
    params = {}
    params['step'] = 5.
    params['GM'] = GME
    params['rtol'] = int_tol
    params['atol'] = int_tol
    params['local_extrap'] = False
    
    # Times to integrate to
#    tvec = np.arange(t_in[0], t_in[1]+0.1, params['step'])
    

        
#    # Generate RK4 trajectory
#    t_rk4, X_rk4 = rk4(intfcn, t_in, Xo, params)
    
#    print(t_rk4[1])
#    print(X_rk4[1])
    
    # Generate RKF78 trajectory
#    X_rkf78 = np.zeros((len(tvec), len(Xo)))
#    X_rkf78[0] = Xo.flatten()
#    for ii in range(1,len(tvec)):
#        y0 = X_rkf78[ii-1].flatten()
#        t_in = [tvec[ii-1], tvec[ii]]
#        t_out, x_out = rkf78(intfcn, t_in, y0, params)
#        X_rkf78[ii] = x_out[-1,:].flatten()
#        
#        print(t_out)
#        print(X_rkf78)
#        mistake
    
    tvec, X_rkf78, fcalls = rkf78(intfcn, t_in, Xo, params)
#    tvec, X_rkf78, fcalls = dopri87(intfcn, t_in, Xo, params)
    
    print(fcalls)
    
    # Generate true trajectory
    X_true = np.zeros((len(tvec), len(Xo)))
    for ii in range(len(tvec)):
        ti = tvec[ii]
        x_out = element_conversion(Xo, 1, 1, dt=(ti-tvec[0]))
        X_true[ii,:] = x_out.flatten()
    
    # Generate ode dopri853 trajectory
    solver = ode(intfcn)
    solver.set_integrator(solver_integrator, atol=int_tol, rtol=int_tol)
    solver.set_f_params(params)
    
    solver.set_initial_value(Xo.flatten(), tvec[0])
    X_dop853 = np.zeros((len(tvec), len(Xo)))
    X_dop853[0] = Xo.flatten()
    
    k = 1
    while solver.successful() and solver.t < tvec[-1]:
        solver.integrate(tvec[k])
        X_dop853[k] = solver.y
        k += 1
        
        

        
    
    
    # Compute and plot errors
    err_rk4 = np.zeros((len(tvec)))
    err_rkf78 = np.zeros((len(tvec)))
    err_dop853 = np.zeros((len(tvec)))
    rk4_dop853 = np.zeros((len(tvec)))
    rkf78_dop853 = np.zeros((len(tvec)))
    sma_rk4 = np.zeros((len(tvec)))
    sma_rkf78 = np.zeros((len(tvec)))
    sma_dop853 = np.zeros((len(tvec)))
    for ii in range(len(tvec)):
#        err_rk4[ii] = np.linalg.norm(X_rk4[ii,:] - X_true[ii,:])
#        err_rkf78[ii] = np.linalg.norm(X_rkf78[ii,:] - X_true[ii,:])
#        err_dop853[ii] = np.linalg.norm(X_dop853[ii,:] - X_true[ii,:])
#        rk4_dop853[ii] = np.linalg.norm(X_dop853[ii,:] - X_rk4[ii,:])
#        rkf78_dop853[ii] = np.linalg.norm(X_dop853[ii,:] - X_rkf78[ii,:])
        
#        err_rk4[ii] = X_rk4[ii,0] - X_true[ii,0]
        err_rkf78[ii] = X_rkf78[ii,0] - X_true[ii,0]
        err_dop853[ii] = X_dop853[ii,0] - X_true[ii,0]
#        rk4_dop853[ii] = X_dop853[ii,0] - X_rk4[ii,0]
        rkf78_dop853[ii] = X_dop853[ii,0] - X_rkf78[ii,0]
        
#        sma_rk4[ii] = element_conversion(X_rk4[ii,:], 1, 0)[0] - SMA
        sma_rkf78[ii] = element_conversion(X_rkf78[ii,:], 1, 0)[0] - SMA
        sma_dop853[ii] = element_conversion(X_dop853[ii,:], 1, 0)[0] - SMA
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tvec/3600, err_rk4, 'k.')
    plt.ylabel('RK4 Error')
    plt.subplot(3,1,2)
    plt.plot(tvec/3600, err_dop853, 'k.')
    plt.ylabel('DOPRI853 Error')
    plt.subplot(3,1,3)
    plt.plot(tvec/3600, rk4_dop853)
    plt.ylabel('RK4 - DOPRI853')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tvec/3600, err_rkf78, 'k.')
    plt.ylabel('RKF78 Error')
    plt.subplot(3,1,2)
    plt.plot(tvec/3600, err_dop853, 'k.')
    plt.ylabel('DOPRI853 Error')
    plt.subplot(3,1,3)
    plt.plot(tvec/3600, rkf78_dop853)
    plt.ylabel('RKF78 - DOPRI853')
    plt.xlabel('Time [hours]')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tvec/3600, sma_rk4, 'k.')
    plt.ylabel('RK4 SMA Error')
    plt.subplot(3,1,2)
    plt.plot(tvec/3600, sma_rkf78, 'k.')
    plt.ylabel('RKF78 SMA Error')
    plt.subplot(3,1,3)
    plt.plot(tvec/3600, sma_dop853)
    plt.ylabel('DOPRI853 SMA Error')
    plt.xlabel('Time [hours]')
    
    plt.show()
    
    
    
    return




if __name__ == '__main__':
    
    plt.close('all')
    
    test_twobody()



















