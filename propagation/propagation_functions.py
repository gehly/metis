import numpy as np
from scipy.integrate import odeint
from scipy.integrate import ode
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

#sys.path.append('../')

#from propagation.integration_functions import int_twobody



def propagate_orbit(spacecraftConfig, forcesCoeff, surfaces, ephemeris,
                    ndays, dt, ode_flag=True):

    # Integrator 
    int_tol = 1e-12
    intfcn = spacecraftConfig['intfcn']
    integrator = spacecraftConfig['integrator']
    
    # If flag is set use ode
    if ode_flag:
        
        start = time.time()
        
        y0 = spacecraftConfig['X'].flatten()
        tvec = np.arange(0., 86400.*ndays+(0.1*dt), dt)

        solver = ode(intfcn)
        solver.set_integrator(integrator)
#        solver.set_integrator('vode', method='bdf', nsteps=1000000)
        solver.set_f_params([spacecraftConfig, forcesCoeff, surfaces])
        
        solver.set_initial_value(y0, tvec[0])
        state = np.zeros((len(tvec), len(y0)))
        state[0] = y0
        
        k = 1
        while solver.successful() and solver.t < tvec[-1]:
            solver.integrate(tvec[k])
            state[k] = solver.y
            k += 1
        
#        print(k)
#        print(len(tvec))
        
#        solver = ode(self.FuncDyn)
#        solver.set_integrator(self.integrator)
#        self.t0 = 0.0
#        if (self.state.type=='3DoF'):
#            solver.set_f_params(self.forces)
#            self.lenForces = len(self.forces)
#            y0 =  [self.state.scStateOrekit.getPVCoordinates().getPosition().getX(),
#                   self.state.scStateOrekit.getPVCoordinates().getPosition().getY(),
#                   self.state.scStateOrekit.getPVCoordinates().getPosition().getZ(),
#                   self.state.scStateOrekit.getPVCoordinates().getVelocity().getX(), 
#                   self.state.scStateOrekit.getPVCoordinates().getVelocity().getY(),  
#                   self.state.scStateOrekit.getPVCoordinates().getVelocity().getZ()]
#            lenY0 = 6
#        elif (self.state.type=='6DoF'):
#            solver.set_f_params([self.forces,self.torques])
#            self.lenForces = len(self.forces)
#            self.lenTorques = len(self.torques)
#            # Below states should be defined in some inertial frame - e.g. J2000 
#            y0 =  [self.state.scStateOrekit.getPVCoordinates().getPosition().getX(),
#                   self.state.scStateOrekit.getPVCoordinates().getPosition().getY(),
#                   self.state.scStateOrekit.getPVCoordinates().getPosition().getZ(),
#                   self.state.scStateOrekit.getPVCoordinates().getVelocity().getX(), 
#                   self.state.scStateOrekit.getPVCoordinates().getVelocity().getY(),  
#                   self.state.scStateOrekit.getPVCoordinates().getVelocity().getZ(),
#                   self.state.scStateOrientationIne2Body[0],
#                   self.state.scStateOrientationIne2Body[1],
#                   self.state.scStateOrientationIne2Body[2],
#                   self.state.scStateOrientationIne2Body[3],
#                   self.state.scStateOrientationRateBodywrtIne[0],
#                   self.state.scStateOrientationRateBodywrtIne[1],
#                   self.state.scStateOrientationRateBodywrtIne[2]]
#            lenY0 = 13 
#        else:
#            print('Please select a 3DoF or 6DoF dynamics for the type')
#        solver.set_initial_value(y0, self.t0)
#        N = int(self.tShift/self.tStep) 
#        t = np.linspace(self.t0, self.tShift, N)
#        self.sol = {}
#        self.sol['time'] = []
#        self.sol['time'].append(str(self.state.scStateOrekit.getDate()))
#        self.sol['state'] = np.empty((N, lenY0))
#        self.sol['state'][0] = y0
#        if (self.visibility!=None):
#            self.sol['visibility'] = np.zeros((N, 6))
#            visibilityState = self.visibility.ComputeVisibilityState()
#            if visibilityState[0]!=0.0:
#                appMagnitude = self.visibility.ComputeAppMagnitude()
#                self.sol['visibility'][0] = [appMagnitude,self.visibility.sumFobs\
#                        ,visibilityState[0],visibilityState[1],visibilityState[2]\
#                        ,visibilityState[3]]
#        k = 1
#        while solver.successful() and solver.t < self.tShift:
#            solver.integrate(t[k])
#            self.sol['state'][k] = solver.y
#            self.sol['time'].append(str(self.state.scStateOrekit.getDate()))
#            if (self.visibility!=None):
#                visibilityState = self.visibility.ComputeVisibilityState()
#                if visibilityState[0]!=0.0:
#                    appMagnitude = self.visibility.ComputeAppMagnitude()
#                    self.sol['visibility'][k] = [appMagnitude,self.visibility.sumFobs\
#                            ,visibilityState[0],visibilityState[1],visibilityState[2]\
#                            ,visibilityState[3]]
#            k += 1
    
    # Otherwise use odeint
    else:
    
        start = time.time()
        
        # Setup propagator
        tvec = np.arange(0., 86400.*ndays+(0.1*dt), dt)
        args = (spacecraftConfig, forcesCoeff, surfaces)
        int0 = spacecraftConfig['X'].flatten()
        
        print(int0)

        state = odeint(intfcn,int0,tvec,args,rtol=int_tol,atol=int_tol)
 
    
    
    # Print time
    print('Propagation Time:', time.time() - start)

    # Output time
    UTC0 = spacecraftConfig['time']
    UTC_times = [UTC0 + timedelta(seconds=ti) for ti in tvec]
    

    return UTC_times, state


