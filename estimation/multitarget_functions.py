import numpy as np
import math
import sys
import os
import inspect
import copy


filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from dynamics import dynamics_functions as dyn
from estimation import estimation_functions as est
from sensors import measurement_functions as mfunc
from utilities import time_systems as timesys

###############################################################################
# This file contains a number of basic functions useful for data association
# and multitarget estimation problems.
#
#
# References:
#  [1] Blackman and Popoli, "Desing and Analysis of Modern Tracking Systems,"
#     1999.
#
#  [2] Cox and Hingorani, "An Efficient Implementation of Reid's Multiple 
#     Hypothesis Tracking Algorithm and Its Evaluation for the Purpose of 
#     Visual Tracking," IEEE TPAMI 1996.
#
###############################################################################



###############################################################################
# PHD Filter
###############################################################################

def phd_filter(state_dict, truth_dict, meas_dict, meas_fcn, params_dict):
    
    # Break out inputs
    state_params = params_dict['state_params']
    filter_params = params_dict['filter_params']
    
    # State information
    state_tk = sorted(state_dict.keys())[-1]
    weights = state_dict[state_tk]['weights']
    means = state_dict[state_tk]['means']
    covars = state_dict[state_tk]['covars']
    GMM_dict = {}
    GMM_dict['weights'] = weights
    GMM_dict['means'] = means
    GMM_dict['covars'] = covars
    nstates = len(means[0])    
    
    # Prior information about the distribution
    pnorm = 2.
    kurt = math.gamma(5./pnorm)*math.gamma(1./pnorm)/(math.gamma(3./pnorm)**2.)
    beta = kurt - 1.
    kappa = kurt - float(nstates)

    # Compute sigma point weights
    alpha = filter_params['alpha']
    lam = alpha**2.*(nstates + kappa) - nstates
    gam = np.sqrt(nstates + lam)
    Wm = 1./(2.*(nstates + lam)) * np.ones(2*nstates,)
    Wc = Wm.copy()
    Wm = np.insert(Wm, 0, lam/(nstates + lam))
    Wc = np.insert(Wc, 0, lam/(nstates + lam) + (1 - alpha**2 + beta))
    diagWc = np.diag(Wc)
    filter_params['gam'] = gam
    filter_params['Wm'] = Wm
    filter_params['diagWc'] = diagWc

    # Initialize output
    filter_output = {}

    # Measurement times
    tk_list = sorted(meas_dict.keys())
    
    # Number of epochs
    N = len(tk_list)
  
    # Loop over times
    for kk in range(N):
    
        # Current and previous time
        if kk == 0:
            tk_prior = state_tk
        else:
            tk_prior = tk_list[kk-1]

        tk = tk_list[kk]
        
        print('')
        print(tk)
        # print('ncomps', len(GMM_dict['weights']))
        # print('Nk est', sum(GMM_dict['weights']))

        # Predictor Step
        tin = [tk_prior, tk]
        GMM_bar = phd_predictor(GMM_dict, tin, params_dict)
        
        # print('predictor')
        # print('ncomps', len(GMM_bar['weights']))
        # print('Nk est', sum(GMM_bar['weights']))
        
        # Corrector Step
        Zk = meas_dict[tk]['Zk_list']
        sensor_id_list = meas_dict[tk]['sensor_id_list']
        GMM_dict = phd_corrector(GMM_bar, tk, Zk, sensor_id_list, meas_fcn,
                                 params_dict)
        
        # print('corrector')
        # print('ncomps', len(GMM_dict['weights']))
        # print('Nk est', sum(GMM_dict['weights']))
        
        # Prune/Merge Step
        GMM_dict = est.merge_GMM(GMM_dict, filter_params)
        
        print('merge')
        print('ncomps', len(GMM_dict['weights']))
        print('Nk est', sum(GMM_dict['weights']))
        
        
        # State extraction and residuals calculation
        wk_list, Xk_list, Pk_list, resids_k = \
            phd_state_extraction(GMM_dict, tk, Zk, sensor_id_list, meas_fcn,
                                 params_dict)
            
            
        
        # print('wk_list', wk_list)
        
        
        # Store output
        filter_output[tk] = {}
        filter_output[tk]['weights'] = GMM_dict['weights']
        filter_output[tk]['means'] = GMM_dict['means']
        filter_output[tk]['covars'] = GMM_dict['covars']
        filter_output[tk]['wk_list'] = wk_list
        filter_output[tk]['Xk_list'] = Xk_list
        filter_output[tk]['Pk_list'] = Pk_list
        filter_output[tk]['resids'] = resids_k
        
        
    # TODO Generation of full_state_output not working correctly
    # Use filter_output for error analysis
    full_state_output = {}
    
    return filter_output, full_state_output
    



def phd_predictor(GMM_dict, tin, params_dict):
    '''
    
    
    '''
    
    
    # Break out inputs
    filter_params = params_dict['filter_params']
    state_params = params_dict['state_params']
    int_params = params_dict['int_params']
    
    # Copy input to ensure pass by value
    GMM_dict = copy.deepcopy(GMM_dict)
    filter_params = copy.deepcopy(filter_params)
    state_params = copy.deepcopy(state_params)
    int_params = copy.deepcopy(int_params)
    
    # Retrieve parameters
    p_surv = filter_params['p_surv']
    Q = filter_params['Q']
    gam = filter_params['gam']
    Wm = filter_params['Wm']
    diagWc = filter_params['diagWc']    
    gap_seconds = filter_params['gap_seconds']        
    time_format = int_params['time_format']     
    
    # Fudge to work with general_dynamics
    state_params['alpha'] = filter_params['alpha']
    
    q = int(Q.shape[0])
    
    tk_prior = tin[0]
    tk = tin[1]
    
    if time_format == 'seconds':
        delta_t = tk - tk_prior
    elif time_format == 'JD':
        delta_t = (tk - tk_prior)*86400.
    elif time_format == 'datetime':
        delta_t = (tk - tk_prior).total_seconds()
    
    # Check if propagation is needed
    if delta_t == 0.:
        return GMM_dict
    
    # Initialize for integrator
    # Retrieve current GMM
    weights = GMM_dict['weights']
    means = GMM_dict['means']
    covars = GMM_dict['covars']    
    ncomp = len(weights)
    nstates = len(means[0])
    npoints = 2*nstates + 1

    # Loop over components
    for jj in range(ncomp):
        
#        print('\nstart loop')
#        print('jj', jj)
#        print('ncomp', len(weights))
#        print('t0', t0_list[jj])
        
        # Retrieve component values
        wj = weights[jj]
        mj = means[jj]
        Pj = covars[jj]
        tin = [tk_prior, tk]
            
        # Compute sigma points
        sqP = np.linalg.cholesky(Pj)
        Xrep = np.tile(mj, (1, nstates))
        chi = np.concatenate((mj, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
        chi_v = np.reshape(chi, (nstates*npoints, 1), order='F')
        
        # Integrate sigma points
        int0 = chi_v.flatten()
        tout, intout = \
            dyn.general_dynamics(int0, tin, state_params, int_params)

        # Retrieve output state        
        chi_v = intout[-1,:]
        chi = np.reshape(chi_v, (nstates, npoints), order='F')

        # State Noise Compensation
        # Zero out SNC for long time gaps
        Gamma = np.zeros((nstates,q))
        if delta_t < gap_seconds:   
            Gamma[0:q,:] = (delta_t**2./2) * np.eye(q)
            Gamma[q:2*q,:] = delta_t * np.eye(q)

        Xbar = np.dot(chi, Wm.T)
        Xbar = np.reshape(Xbar, (nstates, 1))
        chi_diff = chi - np.dot(Xbar, np.ones((1, npoints)))
        Pbar = np.dot(chi_diff, np.dot(diagWc, chi_diff.T)) + np.dot(Gamma, np.dot(Q, Gamma.T))

        # Store output
        weights[jj] *= p_surv
        means[jj] = Xbar
        covars[jj] = Pbar

    # Form output
    GMM_bar = {}
    GMM_bar['weights'] = weights
    GMM_bar['means'] = means
    GMM_bar['covars'] = covars   


    return GMM_bar



def phd_corrector(GMM_bar, tk, Zk, sensor_id_list, meas_fcn, params_dict):
    '''
    
    
    '''
    
    
    # Retrieve inputs
    filter_params = params_dict['filter_params']
    state_params = params_dict['state_params']
    sensor_params = params_dict['sensor_params']
    gam = filter_params['gam']
    Wm = filter_params['Wm']
    diagWc = filter_params['diagWc']
    p_det = filter_params['p_det']
    
    # Break out GMM
    weights0 = GMM_bar['weights']
    means0 = GMM_bar['means']
    covars0 = GMM_bar['covars']
    nstates = len(means0[0])
    npoints = 2*nstates + 1
    ncomp = len(weights0)
    nmeas = len(Zk)
    
    # Components for missed detection case
    weights = [(1. - p_det)*wj for wj in weights0]
    means = copy.copy(means0)
    covars = copy.copy(covars0)
    
    # Loop over each measurement and compute updates
    for ii in range(nmeas):
        
        # Retrieve measurement
        zi = Zk[ii]
        sensor_id = sensor_id_list[ii]
    
        # Loop over components   
        qk_list = []
        for jj in range(ncomp):        
            
            mj = means0[jj]
            Pj = covars0[jj]
            
            # Compute sigma points
            sqP = np.linalg.cholesky(Pj)
            Xrep = np.tile(mj, (1, nstates))
            chi = np.concatenate((mj, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
            chi_diff = chi - np.dot(mj, np.ones((1, npoints)))
    
            # Computed measurements and covariance
            gamma_til_k, Rk = meas_fcn(tk, chi, state_params, sensor_params, sensor_id)
            zbar = np.dot(gamma_til_k, Wm.T)
            zbar = np.reshape(zbar, (len(zbar), 1))
            z_diff = gamma_til_k - np.dot(zbar, np.ones((1, npoints)))
            Pyy = np.dot(z_diff, np.dot(diagWc, z_diff.T)) + Rk
            Pxy = np.dot(chi_diff,  np.dot(diagWc, z_diff.T))
            
            print('zi', zi)
            print('zbar', zbar)
            
            # Kalman gain and measurement update
            Kk = np.dot(Pxy, np.linalg.inv(Pyy))
            mf = mj + np.dot(Kk, zi-zbar)
            
            # Joseph form
            cholPbar = np.linalg.inv(np.linalg.cholesky(Pj))
            invPbar = np.dot(cholPbar.T, cholPbar)
            P1 = (np.eye(nstates) - np.dot(np.dot(Kk, np.dot(Pyy, Kk.T)), invPbar))
            P2 = np.dot(Kk, np.dot(Rk, Kk.T))
            Pf = np.dot(P1, np.dot(Pj, P1.T)) + P2
            
            # Compute Gaussian likelihood
            qk_j = est.gaussian_likelihood(zi, zbar, Pyy)
    
            # Store output
            means.append(mf)
            covars.append(Pf)
            qk_list.append(qk_j)
        
        # Normalize updated weights
        denom = p_det*np.dot(qk_list, weights0) + clutter_intensity(zi, sensor_id, sensor_params)
        wf = [p_det*a1*a2/denom for a1,a2 in zip(weights0, qk_list)]
        weights.extend(wf)
        
    # Form output  
    GMM_dict = {}
    GMM_dict['weights'] = weights
    GMM_dict['means'] = means
    GMM_dict['covars'] = covars
    
    
    return GMM_dict



def phd_state_extraction(GMM_dict, tk, Zk, sensor_id_list, meas_fcn, 
                         params_dict):
    '''
    
    
    '''
    
    # Retrieve inputs
    filter_params = params_dict['filter_params']
    state_params = params_dict['state_params']
    sensor_params = params_dict['sensor_params']
    int_params = params_dict['int_params']
    time_format = int_params['time_format']
    
    # Compute UTC
    if time_format == 'JD':
        UTC = timesys.jd2dt(tk)
    elif time_format == 'datetime':
        UTC = tk
    
    # Retrieve current GMM componets
    weights = GMM_dict['weights']
    means = GMM_dict['means']
    covars = GMM_dict['covars']

    
    # Compute cardinality
    Nk = int(round(sum(weights)))
    if Nk > len(weights):
        Nk = len(weights)
    
    # Select the Nk highest weighted components as the state estimate at 
    # current time
    sorted_inds = sorted(range(len(weights)), key=lambda k: weights[k], reverse=True)
    max_inds = sorted_inds[0:Nk]
    
    # Values for output
    wk_list = [weights[ii] for ii in max_inds]
    Xk_list = [means[ii] for ii in max_inds]
    Pk_list = [covars[ii] for ii in max_inds]
    
    # Calculate residuals
    resids_out = []
    for ii in range(len(Zk)):
        zi = Zk[ii]
        sensor_id = sensor_id_list[ii]
        
        resids_list = []
        for jj in range(len(Xk_list)):
            Xj = Xk_list[jj]            
            zbar = mfunc.compute_measurement(Xj, state_params, sensor_params,
                                             sensor_id, UTC)
            resids = zi - zbar
            resids_list.append(resids)
            
        # Take smallest magnitude as residual for this measurement
        min_list = [np.linalg.norm(resid) for resid in resids_list]
        resids_k = resids_list[min_list.index(min(min_list))]
        resids_out.append(resids_k)
    
    
    return wk_list, Xk_list, Pk_list, resids_out























###############################################################################
# Utility Functions
###############################################################################


def clutter_intensity(zi, sensor_id, sensor_params):
    
    # Assume clutter is poisson-distributed in number and uniform in spatial
    # distribution
    sensor = sensor_params[sensor_id]
    lam_clutter = sensor['lam_clutter']
    V_sensor = sensor['V_sensor']
    
    kappa = lam_clutter/V_sensor    
    
    return kappa




def auction(A) :
    '''
    This function computes a column order of assignments to maximize the score
    for the 2D assignment matrix A.

    Parameters
    ------
    A : NxM numpy array
        score table

    Returns
    ------
    row_indices : list
        each entry in list is assigned row index for the corresponding column
        e.g. row_index[0] = assigned row index for column index 0
    score : float
        total score of assignment
    eps: float
        parameter to increment prices to avoid repeated swapping of same
        assignment bids
    
    References
    ------
    [1] Blackman and Popoli, Section 6.5.1

    '''

    N = int(A.shape[0])
    M = int(A.shape[1])
    eps = 1./(2.*N)
    flag = 0

    #Check if A still has assignments possible
    Acheck = np.zeros((N,M))
    for ii in range(N):
        for jj in range(M):
            if A[ii,jj] > 0.:
                Acheck[ii,jj] = 1.
    sumA = sum(Acheck)

    for ii in range(len(sumA)):
        if sumA[ii] == 0.:
#            print('No more assignments available')
            flag = 1
                       
    if not flag:
        #Step 1: Initialize assignment matrix and track prices
        assign_mat = np.zeros((N,M))
        price = np.zeros((N,1))
        real_price = np.zeros((N,1))
        

#        eps = 0.5

        loop_count = 0

        #Repeat until all columns have been assigned a row
        while np.sum(assign_mat) < M:
            for jj in range(M):

                #print 'j',j
                
                #Step 2: Check if column j is unassigned
                if np.sum(assign_mat[:,jj]) == 0:

                    #Set cost for unallowed assignments
                    for row in range(N):
                        if A[row,jj] <= 0 and price[row] == 0:                            
                            price[row] = 1e15

                            #if row == 0 :
                            #    print 'unallowed cost set'

                    #Step 3: Find the best row i for column j                
                    jvec = np.reshape(A[:,jj],(N,1)) - price
                    ii = np.argmax(jvec)

                    #print 'best i',i

                    #Check if [i,j] is a valid assignment
                    if A[ii,jj] <= 0:
                        flag = 1
                        break

                    #Step 4: Assign row i to column j
                    assign_mat[ii,:] = np.zeros((1,M))
                    assign_mat[ii,jj] = 1.

                    #Step 5: Compute new price
                    jvec2 = np.sort(list(np.reshape(jvec,(1,N))))
                    yj = jvec2[0][-1] - jvec2[0][-2]                
                    real_price[ii] = real_price[ii] + yj + eps
                    price = copy.copy(real_price)

##                    print 'yj',yj
##                    print 'eps',eps
##                    print 'price',price[i]

##                    #Reset price for unallowed assignments
##                    for row in xrange(0,N) :
##                        if A[row,j] <= 0 :
##                            price[row] = 0.

                    #print 'assign_mat',assign_mat
                    #print 'price',price


##            for kk in xrange(0,M) :
##                x = np.nonzero(assign_mat[:,kk])
##                print kk
##                print x[0]

            loop_count += 1
#            print('loop', loop_count)
            if loop_count > 3*M:
                eps *= 2.
                loop_count = 0
                assign_mat = np.zeros((N,M))
                price = np.zeros((N,1))
                real_price = np.zeros((N,1))

            #mistake

            if flag :
                break            

    #Set the row indices to achieve assignment
    row_indices = []
    score = 0.
    #print 'eps',eps
    if not flag :
        for jj in range(M):
            x = np.nonzero(assign_mat[:,jj])       
            row_indices.append(int(x[0]))
            score += A[int(x[0]),jj]

    return row_indices, score, eps



def murty(A0, kbest=1):
    '''
    This function computes the k-best solutions to the 2D assignment problem
    by repeatedly running auction on reduced forms of the input score matrix.

    Parameters
    ------
    A0 : NxM numpy array
        score table
    kbest : int
        number of solutions to return (k highest scoring assignments)
    
    Returns
    ------
    final_list : list of lists
        each entry in list is a row_index list
        each entry in row_indices is assigned row index for the corresponding
        column, e.g. row_indices[0] = assigned row index for column index 0
    
    
    References
    ------
    [2] Cox and Hingorani

    '''

    #Form association table
    N = int(A0.shape[0])
    M = int(A0.shape[1])
    
    #Step 1: Solve for the best solution
    row_indices, score, eps = auction(A0)

    #print 'A',A
#    print(row_indices)
#    print(score)

    #Step 2: Initialize List of Problem/Solution Pairs
    candidate_A_list = [A0]
    candidate_solution_list = [row_indices]
    score_list = [score]

    #Step 3: Clear the list of solutions to be returned
    solution_list = []

    #Step 4: Loop to find kbest possible solutions
    for ind in range(kbest):

#        print('ind',ind)
#        print('A_list',candidate_A_list)
#        print('cand_list', candidate_solution_list)
#        print('scores',score_list)

        if not candidate_solution_list :
            # print('No more solutions available')
            break

        #Step 4.1: Find the best solution in PS_list
        best_ind = np.argmax(score_list)
        A1 = candidate_A_list[best_ind]
        S = candidate_solution_list[best_ind]

        #Step 4.2: Remove this entry from PS_list and score list
        del candidate_A_list[best_ind]
        del candidate_solution_list[best_ind]
        del score_list[best_ind]

        #Step 4.3: Add this solution to the final list
        solution_list.append(S)
#        print('solution_list', solution_list)

        #Step 4.4: Loop through all solution pairs in S
        for j in range(0,len(S)):
            
#            print('\n\n', j)

            #Step 4.4.1: Set A2 = A1
            A2 = copy.copy(A1)

            #Step 4.4.2: Remove solution [i,j] from A2
            i = S[j]
            A2[i,j] = 0.

#            print(A1)
#            print(A2)

            #Step 4.4.3: Solve for best remaining solution
            row_indices, score, eps = auction(A2)


            #Step 4.4.4: If solution exists, add to PS_list
            if row_indices:
                candidate_A_list.append(A2)
                candidate_solution_list.append(row_indices)
                score_list.append(score)
            else:
                continue

            #Step 4.4.5: Remove row/col from A1 except [i,j]           
            for i1 in range(N):
                if i1 != i:
                    A1[i1,j] = 0.
            
            for j1 in range(M):
                if j1 != j:
                    A1[i,j1] = 0.

#            print(row_indices)
#            print(score)
#            print('A1',A1)
                               

    #Remove duplicate solutions
    final_list = []    
    for i in solution_list:
        flag = 0
        for j in final_list:
            if i == j:
                flag = 1
        if not flag:
            final_list.append(i)
            
    return final_list








