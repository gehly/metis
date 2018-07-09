import numpy as np
from scipy.integrate import odeint
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

#sys.path.append('../')

#from propagation.integration_functions import int_twobody



def propagate_orbit(spacecraftConfig, forcesCoeff, surfaces, ephemeris,
                    ndays, dt):

    # Integrator 
    int_tol = 1e-12
    intfcn = spacecraftConfig['intfcn']
    
    if 1: #spacecraftConfig['type'] == '3DoF':
    
        start = time.time()
        
        # Setup propagator
        tvec = np.arange(0., 86400.*ndays+(0.1*dt), dt)
        args = (spacecraftConfig, forcesCoeff, surfaces)
        int0 = spacecraftConfig['X'].flatten()
        
        print(int0)

        state = odeint(intfcn,int0,tvec,args,rtol=int_tol,atol=int_tol)
 
        print('Propagation Time:', time.time() - start)                    
                                   
                                   
#    elif spacecraftConfig['type'] == '6DoF':
#        
#        # Setup propagator
#        tvec = np.arange(0., 86400.*ndays+0.1, dt)
        
        
        
    else:
        print('Error: invalid config type, choose 3DoF or 6DoF')



    # Output time and state
    UTC0 = spacecraftConfig['time']
    UTC_times = [UTC0 + timedelta(seconds=ti) for ti in tvec]
    

    return UTC_times, state


