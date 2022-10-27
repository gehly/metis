import numpy as np
from numba import jit
from numba.typed import Dict


@jit(nopython=True)
def rk4(intfcn, tin, y0, params):
    '''
    This function implements the fixed-step, single-step, 4th order Runge-Kutta
    integrator.
    
    Parameters
    ------
    intfcn : function handle
        handle for function to integrate
    tin : 1D numpy array
        times to integrate over, [t0, tf] or [t0, t1, t2, ... , tf]
    y0 : numpy array
        initial state vector
    params : dictionary
        parameters for integration including step size and any additional
        variables needed for the integration function
    
    Returns
    ------
    tvec : 1D numpy array
        times corresponding to output states from t0 to tf
    yvec : (N+1)xn numpy array
        output state vectors at each time, each row is 1xn vector of state
        at corresponding time
    
    '''
    
    # Retrieve inputs
    h = params['step']
    
    # Start and end times
    t0 = tin[0]
    tf = tin[-1]

    # Initial setup    
    yn = y0.flatten()
    tn = t0
    yvec = y0.reshape(1, len(y0))
    tvec = np.array([t0])
    
    # Loop to end
    while tn < tf:
                
        # Compute k values
        k1 = h * intfcn(tn,yn,params)
        k2 = h * intfcn(tn+h/2.,yn+k1/2.,params)
        k3 = h * intfcn(tn+h/2.,yn+k2/2.,params)
        k4 = h * intfcn(tn+h,yn+k3,params)
        
        # Compute solution
        yn += (1./6.)*(k1 + 2.*k2 + 2.*k3 + k4)

        # Store output
        yvec = np.concatenate((yvec, yn.reshape(1,len(y0))), axis=0)
        
        # Increment time
        tn += h
        tvec = np.append(tvec, tn)

    return tvec, yvec


@jit(nopython=True)
def dopri87(intfcn, tin, y0, params):
    '''
    This function implements the variable step-size, single-step,
    DOPRI87 integrator.
    
    DOPRI uses a similar structure to RKF but has a different Butcher table.
    Both the 7th and 8th order solutions are 13-stage. The algorithm outputs
    the 8th order solution.
    
    Generally DOPRI is expected to require fewer function calls to achieve a
    similar level of accuracy as RKF.
    
    The additional parameter m is used to adjust step size, currently
    hard-coded but could be moved to params if needed. 
    
    Parameters
    ------
    intfcn : function handle
        handle for function to integrate
    tin : 1D numpy array
        times to integrate over, [t0, tf]
    y0 : numpy array
        initial state vector
    params : dictionary
        parameters for integration including step size and any additional
        variables needed for the integration function
    
    Returns
    ------
    tvec : 1D numpy array
        times corresponding to output states from t0 to tf
    yvec : (N+1)xn numpy array
        output state vectors at each time, each row is 1xn vector of state
        at corresponding time
        
    Reference
    ------
    [1] Prince, P.J. and J.R. Dormand, "High Order Embedded Runge-Kutta 
         Formulae," Journal of Computational and Applied Mathematics,
         Vol. 7, 1981.

    [2] Montenbruck, O. and E. Gill, "Satellite Orbits: Models, Methods,
         Applications," Corrected Third Printing, 2005, Springer-Verlag
         Berlin Heidelberg, 2000.
    
    '''
    
    # Input parameters
    h = params['step']
    rtol = params['rtol']
    atol = params['atol']    
    m = 0.9
    
    # Start and end times
    t0 = tin[0]
    tf = tin[-1]
    
    
    # Initial setup
    yn = y0.flatten()
    tn = t0
    yvec = y0.reshape(1, len(y0))
    tvec = np.array([t0])
    
    # Set up weights
    # 7th order (b)
    b_7  = np.array([13451932./455176623., 0., 0., 0., 0., 
                    -808719846./976000145., 1757004468./5645159321., 
                     656045339./265891186., -3867574721./1518517206., 
                     465885868./322736535., 53011238./667516719., 2./45., 0.])
    
    # 8th order (bhat)
    b_8  = np.array([14005451./335480064., 0., 0., 0., 0., 
                    -59238493./1068277825., 181606767./758867731., 
                     561292985./797845732., -1041891430./1371343529., 
                     760417239./1151165299., 118820643./751138087., 
                    -528747749./2220607170., 1./4.])
 
    # Set up nodes
    c = np.array([0., 1./18., 1./12., 1./8., 5./16., 3./8., 59./400., 93./200.,
                  5490023248./9719169821., 13./20., 1201146811./1299019798.,
                  1., 1.])

    # Set up Runge-Kutta matrix
    A = np.zeros((13,13))
    A[1,0] = 1./18.
    A[2,0:2] = [1./48., 1./16.]
    A[3,0:3] = [1./32., 0., 3./32.]
    A[4,0:4] = [5./16., 0., -75./64., 75./64.]
    A[5,0:5] = [3./80., 0., 0., 3./16., 3./20.]
    A[6,0:6] = [29443841./614563906., 0., 0., 77736538./692538347., 
               -28693883./1125000000., 23124283./1800000000.]
    A[7,0:7] = [16016141./946692911., 0., 0., 61564180./158732637., 
                22789713./633445777., 545815736./2771057229., 
               -180193667./1043307555.]
    A[8,0:8] = [39632708./573591083., 0., 0., -433636366./683701615., 
               -421739975./2616292301., 100302831./723423059., 
                790204164./839813087., 800635310./3783071287.]
    A[9,0:9] = [246121993./1340847787., 0., 0., -37695042795./15268766246., 
               -309121744./1061227803., -12992083./490766935., 
                6005943493./2108947869., 393006217./1396673457., 
                123872331./1001029789.]
    A[10,0:10] = [-1028468189./846180014., 0., 0., 8478235783./508512852., 
                   1311729495./1432422823., -10304129995./1701304382., 
                  -48777925059./3047939560., 15336726248./1032824649., 
                  -45442868181./3398467696., 3065993473./597172653.]
    A[11,0:11] = [185892177./718116043., 0., 0., -3185094517./667107341., 
                 -477755414./1098053517., -703635378./230739211., 
                  5731566787./1027545527., 5232866602./850066563., 
                 -4093664535./808688257., 3962137247./1805957418., 
                  65686358./487910083.]
    A[12,0:12] = [403863854./491063109., 0., 0., -5068492393./434740067., 
                 -411421997./543043805., 652783627./914296604., 
                  11173962825./925320556., -13158990841./6184727034., 
                  3936647629./1978049680., -160528059./685178525., 
                  248638103./1413531060., 0.]
    
    
    
    # Loop to end
    k8 = np.zeros((len(yn),13))
    while tn < tf:
        
        # Ensure final time step is exactly to the end
        if tn + h > tf:
            h = tf - tn

        k8[:,0] = h * intfcn(tn,yn,params)
        k8[:,1] = h * intfcn(tn+c[1]*h,yn+(k8[:,0]*A[1,0]).flatten(),params)
        k8[:,2] = h * intfcn(tn+c[2]*h,yn+np.dot(k8[:,0:2],A[2,0:2].T).flatten(),params)
        k8[:,3] = h * intfcn(tn+c[3]*h,yn+np.dot(k8[:,0:3],A[3,0:3].T).flatten(),params)
        k8[:,4] = h * intfcn(tn+c[4]*h,yn+np.dot(k8[:,0:4],A[4,0:4].T).flatten(),params)
        k8[:,5] = h * intfcn(tn+c[5]*h,yn+np.dot(k8[:,0:5],A[5,0:5].T).flatten(),params)
        k8[:,6] = h * intfcn(tn+c[6]*h,yn+np.dot(k8[:,0:6],A[6,0:6].T).flatten(),params)
        k8[:,7] = h * intfcn(tn+c[7]*h,yn+np.dot(k8[:,0:7],A[7,0:7].T).flatten(),params)
        k8[:,8] = h * intfcn(tn+c[8]*h,yn+np.dot(k8[:,0:8],A[8,0:8].T).flatten(),params)
        k8[:,9] = h * intfcn(tn+c[9]*h,yn+np.dot(k8[:,0:9],A[9,0:9].T).flatten(),params)
        k8[:,10] = h * intfcn(tn+c[10]*h,yn+np.dot(k8[:,0:10],A[10,0:10].T).flatten(),params)
        k8[:,11] = h * intfcn(tn+c[11]*h,yn+np.dot(k8[:,0:11],A[11,0:11].T).flatten(),params)
        k8[:,12] = h * intfcn(tn+c[12]*h,yn+np.dot(k8[:,0:12],A[12,0:12].T).flatten(),params)
        
        # Compute updated solution and embedded solution for step size control
        # 7th Order Solution (q)
        y_7 = yn + np.dot(k8,b_7.T).flatten()
    
        # 8th Order Solution (p)
        y_8 = yn + np.dot(k8,b_8.T).flatten()
    
        # Error Checks
        delta = np.max(np.abs(y_7 - y_8))
        if delta < 1e-15:
            delta = 5e-15
        
        epsilon = np.max(np.abs(yn))*rtol + atol
    
        # Updated step size
        hnew = h*m*(epsilon/delta)**(1./8.)
        
        # Check condition for appropriate step size
        # If condition met, solution is good, proceed 
        if delta <= epsilon:
                  
            # Standard 8th Order Solution
            yn = y_8

            # Increment current time
            tn = tn + h            
            
            # Use new time step
            h = hnew
        
            # Store Output
            yvec = np.concatenate((yvec, yn.reshape(1,len(yn))), axis=0)
            tvec = np.append(tvec, tn)

        # Otherwise, solution is bad, repeat at current time with new h
        else:            
            h = hnew

    return tvec, yvec







###############################################################################
# AEGIS Functions
###############################################################################


@jit(nopython=True)
def dopri87_aegis(intfcn, tin, y0, params):
    '''
    This function implements the variable step-size, single-step,
    DOPRI87 integrator for the AEGIS Gaussian Mixture splitting method.
    
    DOPRI uses a similar structure to RKF but has a different Butcher table.
    Both the 7th and 8th order solutions are 13-stage. The algorithm outputs
    the 8th order solution.
    
    Generally DOPRI is expected to require fewer function calls to achieve a
    similar level of accuracy as RKF.
    
    The additional parameter m is used to adjust step size, currently
    hard-coded but could be moved to params if needed. 
    
    Parameters
    ------
    intfcn : function handle
        handle for function to integrate
    tin : 1D numpy array
        times to integrate over, [t0, tf]
    y0 : numpy array
        initial state vector
    params : dictionary
        parameters for integration including step size and any additional
        variables needed for the integration function
    
    Returns
    ------
    tvec : 1D numpy array
        times corresponding to output states from t0 to tf
    yvec : (N+1)xn numpy array
        output state vectors at each time, each row is 1xn vector of state
        at corresponding time
        
    Reference
    ------
    [1] Prince, P.J. and J.R. Dormand, "High Order Embedded Runge-Kutta 
         Formulae," Journal of Computational and Applied Mathematics,
         Vol. 7, 1981.

    [2] Montenbruck, O. and E. Gill, "Satellite Orbits: Models, Methods,
         Applications," Corrected Third Printing, 2005, Springer-Verlag
         Berlin Heidelberg, 2000.
         
    [3] DeMars, K.J., "Entropy-based Approach for Uncertainty Propagation of
        Nonlinear Dynamical Systems," JGCD 2013.
    
    '''
    
    # Input parameters
    h = params['step']
    rtol = params['rtol']
    atol = params['atol']
    m = 0.9
        
    # Retrieve number of states, sigma points, entropies
    nstates = int(params['nstates'])
    npoints = int(params['npoints'])
    split_T = params['split_T']
    
    # Unscented Transform Parameters
    kurt = 3.  # Gaussian
    beta = kurt - 1.
    kappa = kurt - nstates
    alpha = params['alpha']
    lam = alpha**2.*(nstates + kappa) - nstates
    Wm = 1./(2.*(nstates + lam)) * np.ones(2*nstates,)
    Wc = Wm.copy()
    Wm = np.insert(Wm, 0, lam/(nstates + lam))
    Wc = np.insert(Wc, 0, lam/(nstates + lam) + (1 - alpha**2 + beta))
    diagWc = np.diag(Wc)    
    
    # Start and end times
    t0 = tin[0]
    tf = tin[-1]

    # Initial setup
    yn = y0.flatten()
    tn = t0
    yvec = y0.reshape(1, len(y0))
    tvec = np.array([t0])
    ej_initial = float(y0[0])
    
    # Set up weights
    # 7th order (b)
    b_7  = np.array([13451932./455176623., 0., 0., 0., 0., 
                    -808719846./976000145., 1757004468./5645159321., 
                     656045339./265891186., -3867574721./1518517206., 
                     465885868./322736535., 53011238./667516719., 2./45., 0.])
    
    # 8th order (bhat)
    b_8  = np.array([14005451./335480064., 0., 0., 0., 0., 
                    -59238493./1068277825., 181606767./758867731., 
                     561292985./797845732., -1041891430./1371343529., 
                     760417239./1151165299., 118820643./751138087., 
                    -528747749./2220607170., 1./4.])
 
    # Set up nodes
    c = np.array([0., 1./18., 1./12., 1./8., 5./16., 3./8., 59./400., 93./200.,
                  5490023248./9719169821., 13./20., 1201146811./1299019798.,
                  1., 1.])

    # Set up Runge-Kutta matrix
    A = np.zeros((13,13))
    A[1,0] = 1./18.
    A[2,0:2] = [1./48., 1./16.]
    A[3,0:3] = [1./32., 0., 3./32.]
    A[4,0:4] = [5./16., 0., -75./64., 75./64.]
    A[5,0:5] = [3./80., 0., 0., 3./16., 3./20.]
    A[6,0:6] = [29443841./614563906., 0., 0., 77736538./692538347., 
               -28693883./1125000000., 23124283./1800000000.]
    A[7,0:7] = [16016141./946692911., 0., 0., 61564180./158732637., 
                22789713./633445777., 545815736./2771057229., 
               -180193667./1043307555.]
    A[8,0:8] = [39632708./573591083., 0., 0., -433636366./683701615., 
               -421739975./2616292301., 100302831./723423059., 
                790204164./839813087., 800635310./3783071287.]
    A[9,0:9] = [246121993./1340847787., 0., 0., -37695042795./15268766246., 
               -309121744./1061227803., -12992083./490766935., 
                6005943493./2108947869., 393006217./1396673457., 
                123872331./1001029789.]
    A[10,0:10] = [-1028468189./846180014., 0., 0., 8478235783./508512852., 
                   1311729495./1432422823., -10304129995./1701304382., 
                  -48777925059./3047939560., 15336726248./1032824649., 
                  -45442868181./3398467696., 3065993473./597172653.]
    A[11,0:11] = [185892177./718116043., 0., 0., -3185094517./667107341., 
                 -477755414./1098053517., -703635378./230739211., 
                  5731566787./1027545527., 5232866602./850066563., 
                 -4093664535./808688257., 3962137247./1805957418., 
                  65686358./487910083.]
    A[12,0:12] = [403863854./491063109., 0., 0., -5068492393./434740067., 
                 -411421997./543043805., 652783627./914296604., 
                  11173962825./925320556., -13158990841./6184727034., 
                  3936647629./1978049680., -160528059./685178525., 
                  248638103./1413531060., 0.]
    
    
    
    # Loop to end
    k8 = np.zeros((len(yn),13))
    split_flag = False
    while tn < tf:
        
        # Ensure final time step is exactly to the end
        if tn + h > tf:
            h = tf - tn

        k8[:,0] = h * intfcn(tn,yn,params)
        k8[:,1] = h * intfcn(tn+c[1]*h,yn+(k8[:,0]*A[1,0]).flatten(),params)
        k8[:,2] = h * intfcn(tn+c[2]*h,yn+np.dot(k8[:,0:2],A[2,0:2].T).flatten(),params)
        k8[:,3] = h * intfcn(tn+c[3]*h,yn+np.dot(k8[:,0:3],A[3,0:3].T).flatten(),params)
        k8[:,4] = h * intfcn(tn+c[4]*h,yn+np.dot(k8[:,0:4],A[4,0:4].T).flatten(),params)
        k8[:,5] = h * intfcn(tn+c[5]*h,yn+np.dot(k8[:,0:5],A[5,0:5].T).flatten(),params)
        k8[:,6] = h * intfcn(tn+c[6]*h,yn+np.dot(k8[:,0:6],A[6,0:6].T).flatten(),params)
        k8[:,7] = h * intfcn(tn+c[7]*h,yn+np.dot(k8[:,0:7],A[7,0:7].T).flatten(),params)
        k8[:,8] = h * intfcn(tn+c[8]*h,yn+np.dot(k8[:,0:8],A[8,0:8].T).flatten(),params)
        k8[:,9] = h * intfcn(tn+c[9]*h,yn+np.dot(k8[:,0:9],A[9,0:9].T).flatten(),params)
        k8[:,10] = h * intfcn(tn+c[10]*h,yn+np.dot(k8[:,0:10],A[10,0:10].T).flatten(),params)
        k8[:,11] = h * intfcn(tn+c[11]*h,yn+np.dot(k8[:,0:11],A[11,0:11].T).flatten(),params)
        k8[:,12] = h * intfcn(tn+c[12]*h,yn+np.dot(k8[:,0:12],A[12,0:12].T).flatten(),params)
        
        # Compute updated solution and embedded solution for step size control
        # 7th Order Solution (q)
        y_7 = yn + np.dot(k8,b_7.T).flatten()

        # 8th Order Solution (p)
        y_8 = yn + np.dot(k8,b_8.T).flatten()
    
        # Error Checks
        delta = np.max(np.abs(y_7 - y_8))
        if delta < 1e-15:
            delta = 5e-15
        
        epsilon = np.max(np.abs(yn))*rtol + atol
    
        # Updated step size
        hnew = h*m*(epsilon/delta)**(1./8.)
        
        # Check condition for appropriate step size
        # If condition met, solution is good, proceed 
        if delta <= epsilon:
                  
            # Standard 8th Order Solution
            yn = y_8

            # Increment current time
            tn = tn + h            
            
            # Use new time step
            h = hnew
        
            # Store Output
            yvec = np.concatenate((yvec, yn.reshape(1,len(yn))), axis=0)
            tvec = np.append(tvec, tn)
            
            # Check split condition
            # Retrieve linear entropy
            ej_linear = yn[0]
            
            # Nonlinear entropy
            chi_v = yn[1:1+(nstates*npoints)]
            chi = np.reshape(chi_v, (npoints, nstates)).T

            Xbar = np.dot(chi, Wm.T)
            Xbar = np.reshape(Xbar, (nstates, 1))
            chi_diff = chi - np.dot(Xbar, np.ones((1, npoints)))
            Pbar = np.dot(chi_diff, np.dot(diagWc, chi_diff.T))
            
            ej_nonlin = gaussian_entropy(Pbar)
            
            if np.abs(ej_linear - ej_nonlin) > np.abs(split_T*ej_initial):
                split_flag = True
                return tvec, yvec, split_flag

        # Otherwise, solution is bad, repeat at current time with new h
        else:            
            h = hnew

    return tvec, yvec, split_flag


@jit(nopython=True)
def jit_aegis_twobody(t, X, params):
    '''
    This function propagates the sigma points and entropy of a Gaussian Mixture
    Model per the dynamics model specificied in the input params.
    
    Parameters
    ------
    X : numpy array
        initial condition vector of entropies and cartesian state vectors 
        corresponding to sigma points
    t : float 
        current time in seconds
    params : dictionary
        additional arguments

    Returns
    ------
    dX : numpy array
        derivative vector
      
    Reference
    ------
    DeMars, K.J., Bishop, R.H., Jah, M.K., "Entropy-Based Approach for 
        Uncertainty Propagation of Nonlinear Dynamical Systems," JGCD 2013.
        
    '''
    
    # Retrieve inputs
    GM = params['GM']
    nstates = int(params['nstates'])
    npoints = int(params['npoints'])
    ncomp = int(params['ncomp'])
    
    # For each GMM component, there should be 1 entropy, n states, and 2n+1
    # sigma points. Loop over components to compute derivative values
    dX = np.zeros(len(X),)
    for jj in range(ncomp):
        
        # Indices
        nn = npoints*nstates 
        entropy_ind = jj*(nn + 1)
        mean_ind = entropy_ind + 1
        
        # Compute derivative of entropy (DeMars Eq. 13)
        # For twobody case, trace(A) = 0
        dX[entropy_ind] = 0.
        
        # Compute derivatives of states
        Xj = X[mean_ind:mean_ind+nn]
        dXj = np.zeros(len(Xj),)
        for ind in range(npoints):
            
            # Pull out relevant values from Xj
            x = Xj[ind*nstates]
            y = Xj[ind*nstates + 1]
            z = Xj[ind*nstates + 2]
            dx = Xj[ind*nstates + 3]
            dy = Xj[ind*nstates + 4]
            dz = Xj[ind*nstates + 5]
            
            # Compute radius
            r = np.sqrt(x**2. + y**2. + z**2.)
            
            # Solve for components of dXj
            dXj[ind*nstates] = dx
            dXj[ind*nstates + 1] = dy
            dXj[ind*nstates + 2] = dz
    
            dXj[ind*nstates + 3] = -GM*x/r**3
            dXj[ind*nstates + 4] = -GM*y/r**3
            dXj[ind*nstates + 5] = -GM*z/r**3
            
        # Store in output
        dX[mean_ind:mean_ind+nn] = dXj.flatten()    
    
    return dX


@jit(nopython=True)
def jit_twobody_ukf(t, X, params):
    '''
    This function works with ode to propagate object assuming
    simple two-body dynamics.  No perturbations included.  States for UKF
    sigma points included.

    Parameters
    ------
    X : (n*(2n+1)) element list
      initial condition vector of cartesian state and sigma points
    t : float 
      current time in seconds
    params : dictionary
        additional arguments

    Returns
    ------
    dX : (n*(2n+1)) element list
      derivative vector

    '''
    
    # Additional arguments
    GM = params['GM']
    
    # Initialize
    dX = np.zeros(len(X),)
    n = int((-1 + np.sqrt(1. + 8.*len(X)))/4.)

    for ind in range(0, 2*n+1):

        # Pull out relevant values from X
        x = X[ind*n]
        y = X[ind*n + 1]
        z = X[ind*n + 2]
        dx = X[ind*n + 3]
        dy = X[ind*n + 4]
        dz = X[ind*n + 5]

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



@jit(nopython=True)
def test_jit():
    
    test = np.array([1,2,3,4,5,6,7,8,9,10])
    test2 = np.reshape(test, (2,5))
#    test3 = np.reshape(test, (2,5), order='F')
    test3 = np.reshape(test, (5,2)).T
    
    test4 = np.array([1.,2.,3.])
    test4 = np.reshape(test4, (3,1))
#    test5 = test4.repeat(7).reshape((-1,7))
    test5 = np.dot(test4, np.ones((1,7)))
    
    for ii in range(10):
        x = 1.
        print(x)
        
    
    
    return test, test2, test3, test4, test5



@jit(nopython=True)
def gaussian_entropy(P) :
    '''
    This function computes the entropy of a Gaussian PDF given the covariance.
    
    Parameters
    ------
    P : nxn numpy array
        covariance matrix
    
    Returns
    ------
    H : float
        differential entropy
        
    Reference
    ------
    DeMars, K.J., Bishop, R.H., Jah, M.K., "Entropy-Based Approach for 
        Uncertainty Propagation of Nonlinear Dynamical Systems," JGCD 2013.
    '''

    # Differential Entropy (Eq. 5)
    H = 0.5 * np.log(np.linalg.det(2.*np.pi*np.e*P))

    return H


###############################################################################
# Dynamics Functions
###############################################################################




@jit(nopython=True)
def jit_twobody(t, X, params):
    '''
    This function works with ode to propagate object assuming
    simple two-body dynamics.  No perturbations included.

    Parameters
    ------
    X : 6 element array
      cartesian state vector (Inertial Frame)
    t : float 
      current time in seconds
    params : dictionary
        additional arguments

    Returns
    ------
    dX : 6 element array array
      state derivative vector
    
    '''
    
    GM = params['GM']

    # State Vector
    x = X[0]
    y = X[1]
    z = X[2]
    dx = X[3]
    dy = X[4]
    dz = X[5]

    # Compute radius
    r = np.sqrt(x**2. + y**2. + z**2.)

    # Derivative vector
    dX = np.zeros(6,)

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3
    dX[4] = -GM*y/r**3
    dX[5] = -GM*z/r**3
    
    return dX