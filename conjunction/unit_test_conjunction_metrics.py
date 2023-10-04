import numpy as np
import os
import sys
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
current_dir = os.path.dirname(os.path.abspath(filename))

ind = current_dir.find('metis')
metis_dir = current_dir[0:ind+5]
sys.path.append(metis_dir)

from conjunction import conjunction_analysis as ca
from utilities import coordinate_systems as coord
from utilities import eop_functions as eop


###############################################################################
#
# 2D Probability of Collision Tests
#
###############################################################################


# NASA CARA MATLAB Basic Examples/Validation Cases
def unit_test_Pc2D_Foster_basic():
    
#   Examples/Validation Cases:
#
#    Case 1:
#    r1      = [378.39559 4305.721887 5752.767554];
#    v1      = [2.360800244 5.580331936 -4.322349039];
#    r2      = [374.5180598 4307.560983 5751.130418];
#    v2      = [-5.388125081 -3.946827739 3.322820358];
#    cov1    = [44.5757544811362  81.6751751052616  -67.8687662707124;
#               81.6751751052616  158.453402956163  -128.616921644857;
#               -67.8687662707124 -128.616921644858 105.490542562701];
#    cov2    = [2.31067077720423  1.69905293875632  -1.4170164577661;
#               1.69905293875632  1.24957388457206  -1.04174164279599;
#               -1.4170164577661  -1.04174164279599 0.869260558223714];
#    HBR     = 0.020;
#    Tol     = 1e-09;
#    HBRType = 'circle';
#    [Pc]    = Pc2D_Foster(r1,v1,cov1,r2,v2,cov2,HBR,Tol,HBRType)
#    Pc      = 2.7060234765697e-05
  
#    Case 2:
#    r1      = [378.39559 4305.721887 5752.767554];
#    v1      = [2.360800244 5.580331936 -4.322349039];
#    r2      = [374.5180598 4307.560983 5751.130418];
#    v2      = [-5.388125081 -3.946827739 3.322820358];
#    cov1    = [44.5757544811362  81.6751751052616  -67.8687662707124;
#               81.6751751052616  158.453402956163  -128.616921644857;
#               -67.8687662707124 -128.616921644858 105.490542562701];
#    cov2    = [2.31067077720423  1.69905293875632  -1.4170164577661;
#               1.69905293875632  1.24957388457206  -1.04174164279599;
#               -1.4170164577661  -1.04174164279599 0.869260558223714];
#    HBR     = 0.020;
#    Tol     = 1e-09;
#    HBRType = 'square';
#    [Pc]    = Pc2D_Foster(r1,v1,cov1,r2,v2,cov2,HBR,Tol,HBRType)
#    Pc      = 3.4453464970356e-05
 
#    Case 3:
#    r1      = [378.39559 4305.721887 5752.767554];
#    v1      = [2.360800244 5.580331936 -4.322349039];
#    r2      = [374.5180598 4307.560983 5751.130418];
#    v2      = [-5.388125081 -3.946827739 3.322820358];
#    cov1    = [44.5757544811362  81.6751751052616  -67.8687662707124;
#               81.6751751052616  158.453402956163  -128.616921644857;
#               -67.8687662707124 -128.616921644858 105.490542562701];
#    cov2    = [2.31067077720423  1.69905293875632  -1.4170164577661;
#               1.69905293875632  1.24957388457206  -1.04174164279599;
#               -1.4170164577661  -1.04174164279599 0.869260558223714];
#    HBR     = 0.020;
#    Tol     = 1e-09;
#    HBRType = 'squareEquArea';
#    [Pc]    = Pc2D_Foster(r1,v1,cov1,r2,v2,cov2,HBR,Tol,HBRType)
#    Pc      = 2.70601573490093e-05
    
    # 
    
    
    X1 = np.reshape([378.39559, 4305.721887, 5752.767554, 2.360800244, 5.580331936, -4.322349039], (6,1))
    X2 = np.reshape([374.5180598, 4307.560983, 5751.130418, -5.388125081, -3.946827739, 3.322820358], (6,1))
    P1 = np.zeros((6,6))
    P2 = np.zeros((6,6))
    P1[0:3,0:3] = np.array([[44.5757544811362,  81.6751751052616,  -67.8687662707124],
                            [81.6751751052616,  158.453402956163,  -128.616921644857],
                            [-67.8687662707124, -128.616921644858, 105.490542562701]])
    
    P2[0:3,0:3] = np.array([[2.31067077720423,  1.69905293875632,  -1.4170164577661],
                            [1.69905293875632,  1.24957388457206,  -1.04174164279599],
                            [-1.4170164577661,  -1.04174164279599, 0.869260558223714]])
    
    HBR = 0.020
    tol = 1e-9
    
    
    # Circle
    HBR_type = 'circle'    
    Pc = ca.Pc2D_Foster(X1, P1, X2, P2, HBR, rtol=tol, HBR_type=HBR_type)    
    print(HBR_type, Pc, 2.7060234765697e-05)
    
    # Square
    HBR_type = 'square'    
    Pc = ca.Pc2D_Foster(X1, P1, X2, P2, HBR, rtol=tol, HBR_type=HBR_type)    
    print(HBR_type, Pc, 3.4453464970356e-05)
    
    # Square equivalent to the area of the circle
    HBR_type = 'squareEqArea'    
    Pc = ca.Pc2D_Foster(X1, P1, X2, P2, HBR, rtol=tol, HBR_type=HBR_type)    
    print(HBR_type, Pc, 2.70601573490093e-05)
    
    
    
    
    
    return


def unit_test_Pc2D_Foster_full():
    
    Pc_true_list = [0.146749549, 0.006222267, 0.100351176, 0.049323406, 
                    0.044487386, 0.004335455, 0.000158147, 0.036948008,
                    0.290146291, 0.290146291, 0.002672026]
    
    accuracy = 1e-3
    for ii in range(1, len(Pc_true_list)+1):
        Pc_true = Pc_true_list[ii-1]
        fname = 'AlfanoTestCase' + str(ii).zfill(2) + '.cdm'
        cdm_file = os.path.join('unit_test', fname)
        test = run_unit_test(cdm_file, accuracy, Pc_true)
        print(fname, test)
    
    
    
    return


def run_unit_test(cdm_file, accuracy, Pc_true, tol=1e-8, HBR_type='circle'):
    
    # Load CDM data
    TCA_UTC, miss_params, obj_params = ca.read_cdm_file(cdm_file)
    
    # Retrieve data
    HBR = miss_params['COMMENT HBR']            # meters
    
    obj1_frame = obj_params[1]['REF_FRAME']
    obj1_state = obj_params[1]['mean']*1000.    # convert to meters
    obj1_covar = obj_params[1]['covar']         # m^2, m^2/s^2
    
    obj2_frame = obj_params[2]['REF_FRAME']
    obj2_state = obj_params[2]['mean']*1000.    # convert to meters
    obj2_covar = obj_params[2]['covar']         # m^2, m^2/s^2
    
    # Convert covariance matrices to inertial frame
    if obj1_frame == 'EME2000':
        r1_eci = obj1_state[0:3].reshape(3,1)
        v1_eci = obj1_state[3:6].reshape(3,1)
        
    elif obj1_frame == 'ITRF':
        r1_ecef = obj1_state[0:3].reshape(3,1)
        v1_ecef = obj1_state[3:6].reshape(3,1)
        eop_alldata = eop.get_celestrak_eop_alldata()
        EOP_data = eop.get_eop_data(eop_alldata, TCA_UTC)
        r1_eci, v1_eci = coord.itrf2gcrf(r1_ecef, v1_ecef, TCA_UTC, EOP_data)
        
    else:
        print('Error: Object 1 coordinate frame is not known')
        print(obj1_frame)
        return
    
    # print('obj1')
    # print('r1_eci', r1_eci)
    # print('v1_eci', v1_eci)
    
    X1_eci = np.concatenate((r1_eci, v1_eci), axis=0)
    P1_ric = obj1_covar[0:3,0:3]
    P1_eci = coord.ric2eci(r1_eci, v1_eci, P1_ric)
    
    # print('P1_ric', P1_ric)
    # print('P1_eci', P1_eci)
    
    if obj2_frame == 'EME2000':
        r2_eci = obj2_state[0:3].reshape(3,1)
        v2_eci = obj2_state[3:6].reshape(3,1)
        
    elif obj2_frame == 'ITRF':
        r2_ecef = obj2_state[0:3].reshape(3,1)
        v2_ecef = obj2_state[3:6].reshape(3,1)
        eop_alldata = eop.get_celestrak_eop_alldata()
        EOP_data = eop.get_eop_data(eop_alldata, TCA_UTC)
        r2_eci, v2_eci = coord.itrf2gcrf(r2_ecef, v2_ecef, TCA_UTC, EOP_data)
        
    else:
        print('Error: Object 2 coordinate frame is not known')
        print(obj2_frame)
        return
    
    X2_eci = np.concatenate((r2_eci, v2_eci), axis=0)
    P2_ric = obj2_covar[0:3,0:3]
    P2_eci = coord.ric2eci(r2_eci, v2_eci, P2_ric)
    
    # print('obj2')
    # print('r2_eci', r2_eci)
    # print('v2_eci', v2_eci)
    # print('P2_ric', P2_ric)
    # print('P2_eci', P2_eci)
    
    # Calculate 2D Pc
    Pc = ca.Pc2D_Foster(X1_eci, P1_eci, X2_eci, P2_eci, HBR, tol, HBR_type)  
    
    # print('Pc', Pc)
    # print('Pc_true', Pc_true)
    
    
    # Error check
    error = abs(Pc - Pc_true)/Pc_true
    
    # print('rel error', error)
    
    return (error < accuracy)





if __name__ == '__main__':
    
    # unit_test_Pc2D_Foster_basic()
    
    unit_test_Pc2D_Foster_full()
    



