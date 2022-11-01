import numpy as np
import math
from numba import jit
from numba import types
from numba.typed import Dict


@jit(nopython=True)
def test_jit(input_flag, num, myarray):
    
    print(myarray)
    
    x = np.array([1., 2., 3.]).reshape(3,1)
    y = np.array([4., 5., 6.]).reshape(3,1)
    print(x)
    print(y)
    
    z = np.concatenate((x,y),axis=0)
    print(z)
    
    test = np.array([[1.], [2.], [3.]])
    
    test = np.reshape(y, (1,3))
    
    test = np.cos(0.5)
    test = math.acos(0.5)
    
    test = np.empty(0, dtype=np.float64)
    test = np.append(test, [1.2])
    
    
    print(test)
    
    print(np.sign(1.2))
    print(np.sign(-2.3))
    
    if input_flag == 'sometext':
        print('yep')
        
    else:
        print('hmm')
        
    if input_flag == 'notext':
        print('nope')
        
    if np.isnan(num):
        print('nan')
        
    if not np.isnan(num):
        print('num')
    
    return

@jit(nopython=True)
def cross(x, y):

    x1 = x[0][0]
    x2 = x[1][0]
    x3 = x[2][0]
    y1 = y[0][0]
    y2 = y[1][0]
    y3 = y[2][0]
    z1 = x2*y3 - x3*y2
    z2 = -(x1*y3 - x3*y1)
    z3 = x1*y2 - x2*y1
    z = np.array([[z1], [z2], [z3]])
    
    return z


@jit(nopython=True)
def norm(avec):
    
    a = np.sqrt(avec[0][0]**2. + avec[1][0]**2. + avec[2][0]**2.)
    
    return a






if __name__ == '__main__':
    
    test_jit('notext', np.nan, np.random.rand(3,3))
    
    x = np.array([1., 2., 3.]).reshape(3,1)
    y = np.array([4., 5., 6.]).reshape(3,1)
    
    z1 = np.cross(x, y, axis=0)
    z2 = cross(x, y)
    
    print(z1)
    print(z2)
    
    
    print(np.linalg.norm(x), norm(x))
    print(np.linalg.norm(y), norm(y))
    print(np.linalg.norm(z1), norm(z1))
    
    
    
    
    
    
    