import numpy as np
from scipy.integrate import odeint
from scipy.integrate import ode
import matplotlib.pyplot as plt
import sys

sys.path.append('../')

from utilities.astrodynamics import element_conversion
from utilities.constants import GME
from numerical_integration import rk4
from integration_functions import ode_twobody


def test_twobody():
    
    # Set up initial conditions and integration parameters
    x_in = np.reshape([7000., 0.1, 10., 100., 40., 95.], (6,1))
    Xo = element_conversion(x_in, 0, 1)    
    t_in = [0., 6.*3600.]
    
    params = {}
    params['step'] = 5.
    params['GM'] = GME
    
    intfcn = ode_twobody
    integrator = 'dop853'
    int_tol = 1e-12
    tvec = np.arange(t_in[0], t_in[1]+0.1, params['step'])
    
    # Generate true trajectory
    X_true = np.zeros((len(tvec), len(Xo)))
    for ii in range(len(tvec)):
        ti = tvec[ii]
        x_out = element_conversion(x_in, 0, 1, dt=(ti-tvec[0]))
        X_true[ii,:] = x_out.flatten()
        
        
    # Generate RK4 trajectory
    t_rk4, X_rk4 = rk4(intfcn, t_in, Xo, params)    
    
    # Generate ode dopri853 trajectory
    solver = ode(intfcn)
    solver.set_integrator(integrator, atol=int_tol, rtol=int_tol)
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
    err_dop853 = np.zeros((len(tvec)))
    rk4_dop853 = np.zeros((len(tvec)))
    for ii in range(len(tvec)):
        err_rk4[ii] = np.linalg.norm(X_rk4[ii,:] - X_true[ii,:])
        err_dop853[ii] = np.linalg.norm(X_dop853[ii,:] - X_true[ii,:])
        rk4_dop853[ii] = np.linalg.norm(X_dop853[ii,:] - X_rk4[ii,:])
    
    
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
    
    plt.show()
    
    
    
    return




if __name__ == '__main__':
    
    plt.close('all')
    
    test_twobody()



















