import numpy as np
from scipy.integrate import odeint
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path

from integration_functions import int_twobody

def propagate_orbit(params_file, intfcn, ndays, dt):
    
     # Load parameters
    pklFile = open(params_file, 'rb')
    data = pickle.load(pklFile)
    spacecraftConfig = data[0]
    forcesCoeff = data[1]
    brdfCoeff = data[2]
    pklFile.close() 
    
    # Integrator tolerance
    int_tol = 1e-12
    
    if spacecraftConfig['type'] == '3DoF':
    
        start = time.time()
        
        # Setup propagator
        tvec = np.arange(0., 86400.*ndays+(0.1*dt), dt)
        args = (spacecraftConfig, forcesCoeff, brdfCoeff)
        int0 = spacecraftConfig['X'].flatten()
        
        state = odeint(intfcn,int0,tvec,args,rtol=int_tol,atol=int_tol)
 
        print('Propagation Time:', time.time() - start)                    
                                   
                                   
    elif spacecraftConfig['type'] == '6DoF':
        
        # Setup propagator
        tvec = np.arange(0., 86400.*ndays+0.1, dt)
        
        
        
    else:
        print('Error: invalid config type, choose 3DoF or 6DoF')



    # Output time and state
    UTC0 = spacecraftConfig['time']
    UTC_times = [UTC0 + timedelta(seconds=ti) for ti in tvec]
    

    return UTC_times, state


if __name__ == '__main__':
    
    # Data directory
    datadir = Path('C:/Users/Steve/Documents/data/multiple_model/'
                   '2018_07_07_leo')
    
    object_type = 'sphere_lamr'
    
    fname = 'leo_' + object_type + '_2018_07_05_true_params.pkl'
    true_params_file = datadir / fname
    
    
    # Generate truth trajectory and measurements file
    ndays = 3.
    dt = 10.
    
    intfcn = int_twobody
    
    UTC_times, state = propagate_orbit(true_params_file, intfcn, ndays, dt)
    
    
    
    
    