import numpy as np
from math import pi
import matplotlib.pyplot as plt
import sys

sys.path.append('../')

from utilities.astrodynamics import mean2ecc



def plot_mean2ecc():
    
    e_list = [0., 0.2, 0.4, 0.6, 0.8, 0.99]
    M_list = list(np.arange(0., pi+0.001, 0.01))
    
    E_array = np.zeros((len(e_list), len(M_list)))
    
    for e in e_list:
        for M in M_list:
            ii = e_list.index(e)
            jj = M_list.index(M)
            
            E = mean2ecc(M, e)
            if E > pi:
                E -= 2*pi
            
            E_array[ii,jj] = E
            
    color=iter(plt.cm.rainbow(np.linspace(0,1,len(e_list))))
    plt.figure()
    for e in e_list:
        plt.plot(M_list, E_array[e_list.index(e),:], c=next(color), label='e = ' + str(e))
        
    plt.legend()
    plt.xlabel('Mean Anomaly [rad]')
    plt.ylabel('Eccentric Anomaly [rad]')
    
    plt.grid()
            
    plt.show()
    
    
    
    return


if __name__ == '__main__':
    
    plt.close('all')
    
    plot_mean2ecc()