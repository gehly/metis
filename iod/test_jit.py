import numpy as np
import math
from numba import jit
from numba import types
from numba.typed import Dict
import time

import iod_functions2 as iod2
from utilities import astrodynamics as astro


@jit(nopython=True)
def test_jit(input_flag, num, myarray):
    
    print(myarray)
    
    x = np.array([1., 2., 3.]).reshape(3,1)
    y = np.array([4., 5., 6.]).reshape(3,1)
    print(x)
    print(y)
    
    z = np.concatenate((x,y),axis=0)
    print(z)
    
    z2 = np.dot(x.T, y)
    print(z2)
    print(z2[0])
    print(z2[0][0])
    
    print(np.max(x))
    print(np.max(np.array([1., 4., 7.])))
    
    test = np.array([[1.], [2.], [3.]])
    
    test = np.reshape(y, (1,3))
    
    test = np.cos(0.5)
    test = math.acos(0.5)
    
    test = np.empty(0, dtype=np.float64)
    test = np.append(test, [1.2])
    
    test = 10. % (2.*np.pi)
    test = -10. % (2.*np.pi)
    
    test = math.atan(0.5)
    test = math.atan2(0.5, 1.)
    
    
    print('test', test)
    
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



def compare_propagation():
    
#    elem0 = [7000., 0.001, 10., 200., 320., 175.]
    rp = 7000.
    vinf = 2.
    e = 1 + rp*vinf**2./3.986e5
    a = -3.986e5/vinf**2.
    
    elem0 = [a, e, 100., 345., 290., -40.]
    
    cart0 = astro.kep2cart(elem0)
    
    dt = 100000.
    
#    print(cart0)
    
    start = time.time()
    for ii in range(1000):
        X1 = astro.element_conversion(cart0, 1, 1, dt=dt)
    basic_time = time.time() - start
    
    X2 = iod2.twobody_propagate(cart0, dt)
    
    start = time.time()
    for ii in range(1000):
        X2 = iod2.twobody_propagate(cart0, dt)
    jit_time = time.time() - start
    
    print(X1)
    print(X2)
    
    print('err', X1[0:3] - X2[0:3])
    
    print(basic_time)
    print(jit_time)
    
    
    
    return



if __name__ == '__main__':
    
    test_jit('notext', np.nan, np.random.rand(3,3))
#    
#    x = np.array([1., 2., 3.]).reshape(3,1)
#    y = np.array([4., 5., 6.]).reshape(3,1)
#    
#    z1 = np.cross(x, y, axis=0)
#    z2 = cross(x, y)
#    
#    print(z1)
#    print(z2)
#    
#    
#    print(np.linalg.norm(x), norm(x))
#    print(np.linalg.norm(y), norm(y))
#    print(np.linalg.norm(z1), norm(z1))
    
#    compare_propagation()
    
    
    
    
    
    
    