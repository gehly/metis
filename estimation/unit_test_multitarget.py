# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 18:28:16 2022

@author: Steve
"""



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
    kbest = 4
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
