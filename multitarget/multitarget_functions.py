import numpy as np
from math import *
import sys
import copy


sys.path.append('../')



############################################################################
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
############################################################################



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
            print('No more assignments available')
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
            print('loop', loop_count)
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
    print(row_indices)
    print(score)

    #Step 2: Initialize List of Problem/Solution Pairs
    PS_list = [row_indices]
    score_list = [score]

    #Step 3: Clear the list of solutions to be returned
    row_indices_matrix = []

    #Step 4: Loop to find kbest possible solutions
    for ind in range(kbest):

        #Reset A
        A1 = copy.copy(A0)
        print('ind',ind)
        print('PS_list',PS_list)
        print('scores',score_list)

        if not PS_list :
            #print 'No more solutions available'
            break

        #Step 4.1: Find the best solution in PS_list
        best_ind = np.argmax(score_list)
        S = PS_list[best_ind]

        #Step 4.2: Remove this entry from PS_list and score list
        del PS_list[best_ind]
        del score_list[best_ind]

        #Step 4.3: Add this solution to the final list
        row_indices_matrix.append(S)
        print('row_indices_matrix', row_indices_matrix)

        #Step 4.4: Loop through all solution pairs in S
        for j in range(0,len(S)):
            
            print(j)

            #Step 4.4.1: Set A2 = A1
            A2 = copy.copy(A1)

            #Step 4.4.2: Remove solution [i,j] from A2
            i = S[j]
            A2[i,j] = 0.

            #print 'i',i
            #print 'j',j
            #print 'A2',A2

            #Step 4.4.3: Solve for best remaining solution
            row_indices, score, eps = auction(A2)


            #Step 4.4.4: If solution exists, add to PS_list
            if row_indices:
                PS_list.append(row_indices)
                score_list.append(score)
            else:
                continue

            #Step 4.4.5: Remove row/col from A1 except [i,j]

            #SHOULD BE A2????
            
            for i1 in range(N):
                if i1 != i:
                    A1[i1,j] = 0.
            
            for j1 in range(M):
                if j1 != j:
                    A1[i,j1] = 0.

            print(row_indices)
            print(score)
            print('A1',A1)
                               

    #Remove duplicate solutions
    final_list = []    
    for i in row_indices_matrix:
        flag = 0
        for j in final_list:
            if i == j:
                flag = 1
        if not flag:
            final_list.append(i)
            
    return final_list


def unit_test_auction():
    '''
    Example assignment problem from [1] Blackman and Popoli 
    
    '''
    
    # C is cost matrix to minimize
    C = np.array([[10.,    5.,   8.,   9.],
                  [7.,   100.,  20., 100.],
                  [100.,  21., 100., 100.],
                  [100.,  15.,  17., 100.],
                  [100., 100.,  16.,  22.]])
    
    # A is score matrix to maximize
    A = 100.*np.ones((5,4)) - C
    
    # Compute assignment
    row_index, score, eps = auction(A)
    
    print(row_index, score, eps)
    
    truth = [7., 15., 16., 9.]
    test_sum = 0.
    for ii in range(4):
        print(C[row_index[ii],ii])
        test_sum += C[row_index[ii],ii] - truth[ii]
        
    if test_sum == 0.:
        print('pass')
    
    
    
    return


def unit_test_murty():
    '''
    Example assignment problem from [1] Blackman and Popoli
    '''
    
    # C is cost matrix to minimize
    C = np.array([[10.,    5.,   8.,   9.],
                  [7.,   100.,  20., 100.],
                  [100.,  21., 100., 100.],
                  [100.,  15.,  17., 100.],
                  [100., 100.,  16.,  22.]])
    
    # A is score matrix to maximize
    A = 100.*np.ones((5,4)) - C
    
    # Compute assignment
    kbest = 3
    final_list = murty(A, kbest)
    
    print(final_list)
    
    for row_index in final_list:
        for ii in range(4):
            print(C[row_index[ii], ii])
    
    
    
    return




if __name__ == '__main__':
    
    
#    unit_test_auction()
    
    
    unit_test_murty()
    
    
    
    

#    # A is score matrix to maximize
#    A = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                  [2, 10, 3, 6, 2, 12, 6, 9, 6, 10],
#                  [3, 11, 1, 9, 4, 15, 5, 4, 9, 12],
#                  [4, 6, 5, 4, 0, 3, 4, 6, 10, 11],
#                  [5, 0, 6, 8, 1, 10, 3, 7, 8, 13],
#                  [6, 11, 0, 6, 5, 9, 2, 5, 3, 8],
#                  [7, 9, 2, 5, 6, 5, 1, 3, 6, 6],
#                  [8, 8, 6, 9, 4, 0, 8, 2, 1, 5],
#                  [10, 12, 11, 6, 5, 10, 9, 1, 6, 7],
#                  [9, 10, 4, 8, 0, 9, 1, 0, 5, 9]])
#
#    
#    # Compute assignment
#    row_index, score, eps = auction(A)
#    
#    print(row_index, score, eps)
#
#    for ii in range(10):
#        print(A[row_index[ii],ii])
#    



