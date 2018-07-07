import numpy as np
from datetime import datetime

from time_systems import utcdt2ttjd
from time_systems import utcdt2ut1jd
from time_systems import jd2cent

from eop_functions import get_celestrak_eop_alldata
from eop_functions import get_eop_data
from eop_functions import compute_precession_IAU1976
from eop_functions import compute_nutation_IAU1980
from eop_functions import eqnequinox_IAU1982_simple
from eop_functions import compute_polarmotion
from eop_functions import compute_ERA
from eop_functions import init_XYs2006
from eop_functions import get_XYs
from eop_functions import compute_BPN



###############################################################################
#
# This file contains function to perform transformations between different
# coordinate systems.
#
# References:
#
#  [1] Petit, Gerard and Brian Luzum, "IERS Conventions (2010)," 
#        International Earth Rotation and Reference Systems Service (IERS),
#        IERS Conventions Center, IERS Technical Note No. 36, December 2010.
# 
#  [2] International Astronomical Union, "Standards of Fundamental 
#         Astronomy: SOFA Tools for Earth Attitude," December 21, 2009.
#
#  [3] Wallace, P.T. and N. Capitaine, "Precession-Nutation Precedures
#         Consistent with IAU 2006 Resolutions," Astronomy & Astrophysics,
#         Volume 459, pages 981-985, December 2006.
#
#  [4] Coppola, V., J.H. Seago, and D.A. Vallado, "The IAU 2000A and IAU
#         2006 Precession-Nutation Theories and their Implementation," 
#         AAS 09-159, 2009.
#
#  [5] Bradley, B.K., Sibois, A., and Axelrad, P., "Influence of ITRS/GCRS
#        implementation for astrodynamics: Coordinate transformations,"
#        Advances in Space Research, Vol 57, 2016, pp 850-866.
#
#  [6] Vallado, D.A. "Fundamentals of Astrodynamics and Applications".
#         Third Edition. Microcosm Press. 2007.
#
###############################################################################


def teme2gcrf(r_TEME, v_TEME, UTC, IAU1980nut, EOP_data):
    '''
    This function converts position and velocity vectors from the True
    Equator Mean Equinox (TEME) frame used for TLEs to the GCRF inertial
    frame.
    
    Parameters
    ------
    r_TEME : 3x1 numpy array
        position vector in TEME frame
    v_TEME : 3x1 numpy array
        velocity vector in TEME frame
    UTC : datetime object
        time in UTC
    IAU1980nut : 2D numpy array
        nutation coefficients
    EOP_data : dictionary
        EOP data for the given time including pole coordinates and offsets,
        time offsets, and length of day  
    
    Returns
    ------
    r_GCRF : 3x1 numpy array
        position vector in GCRF frame
    v_GCRF : 3x1 numpy array
        velocity vector in GCRF frame    
    '''
    
    # Compute TT in JD format
    TT_JD = utcdt2ttjd(UTC, EOP_data['TAI_UTC'])
    
    # Compute TT in centuries since J2000 epoch
    TT_cent = jd2cent(TT_JD)
    
    # IAU 1976 Precession
    P = compute_precession_IAU1976(TT_cent)
    
    # IAU 1980 Nutation
    N, FA, Eps_A, Eps_true, dPsi, dEps = \
        compute_nutation_IAU1980(IAU1980nut, TT_cent, EOP_data['ddPsi'],
                                 EOP_data['ddEps'])

    # Equation of the Equinonx 1982
    R = eqnequinox_IAU1982_simple(dPsi, Eps_A)
    
    # Compute transformation matrix and output
    GCRF_TEME = np.dot(P, np.dot(N, R))
    
    r_GCRF = np.dot(GCRF_TEME, r_TEME)
    v_GCRF = np.dot(GCRF_TEME, v_TEME)
    
    return r_GCRF, v_GCRF


def gcrf2itrf(r_GCRF, v_GCRF, UTC, EOP_data):
    '''
    This function converts a position and velocity vector in the GCRF(ECI)
    frame to the ITRF(ECEF) frame using the IAU 2006 precession and 
    IAU 2000A_R06 nutation theories. This routine employs a hybrid of the 
    "Full Theory" using Fukushima-Williams angles and the CIO-based method.  

    Specifically, this routine interpolates a table of X,Y,s values and then
    uses them to construct the BPN matrix directly.  The X,Y,s values in the 
    data table were generated using Fukushima-Williams angles and the 
    IAU 2000A_R06 nutation theory.  This general scheme is outlined in [3]
    and [4].
    
    Parameters
    ------
    r_GCRF : 3x1 numpy array
        position vector in GCRF
    v_GCRF : 3x1 numpy array
        velocity vector in GCRF
    UTC : datetime object
        time in UTC
    EOP_data : dictionary
        EOP data for the given time including pole coordinates and offsets,
        time offsets, and length of day  
    
    Returns
    ------
    r_ITRF : 3x1 numpy array
        position vector in ITRF
    v_ITRF : 3x1 numpy array
        velocity vector in ITRF
    
    '''
    
    # Compute UT1 in JD format
    UT1_JD = utcdt2ut1jd(UTC, EOP_data['UT1_UTC'])
    
    # Compute TT in JD format
    TT_JD = utcdt2ttjd(UTC, EOP_data['TAI_UTC'])
    
    # Compute TT in centuries since J2000 epoch
    TT_cent = jd2cent(TT_JD)
    
    # Construct polar motion matrix (ITRS to TIRS)
    W = compute_polarmotion(EOP_data['xp'], EOP_data['yp'], TT_cent)
    
    # Contruct Earth rotaion angle matrix (TIRS to CIRS)
    R = compute_ERA(UT1_JD)
    
    # Construct Bias-Precessino-Nutation matrix (CIRS to GCRS/ICRS)
    XYs_data = init_XYs2006(UTC, UTC)
    X, Y, s = get_XYs(XYs_data, TT_JD)
    
    # Add in Free Core Nutation (FCN) correction
    X = EOP_data['dX'] + X  # rad
    Y = EOP_data['dY'] + Y  # rad
    
    # Compute Bias-Precssion-Nutation (BPN) matrix
    BPN = compute_BPN(X, Y, s)
    
    # Transform position vector
    eci2ecef = np.dot(W.T, np.dot(R.T, BPN.T))
    r_ITRF = np.dot(eci2ecef, r_GCRF)

    # Transform velocity vector
    # Calculate Earth rotation rate, rad/s (Vallado p227)
    wE = 7.29211514670639e-5*(1 - EOP_data['LOD']/86400)                    
    r_TIRS = np.dot(W, r_ITRF)
        
    v_ITRF = np.dot(W.T, (np.dot(R.T, np.dot(BPN.T, v_GCRF)) - 
                          np.cross(np.array([[0.],[0.],[wE]]), r_TIRS, axis=0)))
    
    
    
    return r_ITRF, v_ITRF


def itrf2gcrf(r_ITRF, v_ITRF, UTC, EOP_data):
    '''
    This function converts a position and velocity vector in the ITRF(ECEF)
    frame to the GCRF(ECI) frame using the IAU 2006 precession and 
    IAU 2000A_R06 nutation theories. This routine employs a hybrid of the 
    "Full Theory" using Fukushima-Williams angles and the CIO-based method.  

    Specifically, this routine interpolates a table of X,Y,s values and then
    uses them to construct the BPN matrix directly.  The X,Y,s values in the 
    data table were generated using Fukushima-Williams angles and the 
    IAU 2000A_R06 nutation theory.  This general scheme is outlined in [3]
    and [4].
    
    Parameters
    ------
    r_ITRF : 3x1 numpy array
        position vector in ITRF
    v_ITRF : 3x1 numpy array
        velocity vector in ITRF
    UTC : datetime object
        time in UTC
    EOP_data : dictionary
        EOP data for the given time including pole coordinates and offsets,
        time offsets, and length of day  
    
    Returns
    ------
    r_GCRF : 3x1 numpy array
        position vector in GCRF
    v_GCRF : 3x1 numpy array
        velocity vector in GCRF
    
    '''
    
    # Compute UT1 in JD format
    UT1_JD = utcdt2ut1jd(UTC, EOP_data['UT1_UTC'])
    
    # Compute TT in JD format
    TT_JD = utcdt2ttjd(UTC, EOP_data['TAI_UTC'])
    
    # Compute TT in centuries since J2000 epoch
    TT_cent = jd2cent(TT_JD)
    
    # Construct polar motion matrix (ITRS to TIRS)
    W = compute_polarmotion(EOP_data['xp'], EOP_data['yp'], TT_cent)
    
    # Contruct Earth rotaion angle matrix (TIRS to CIRS)
    R = compute_ERA(UT1_JD)
    
    # Construct Bias-Precessino-Nutation matrix (CIRS to GCRS/ICRS)
    XYs_data = init_XYs2006(UTC, UTC)
    X, Y, s = get_XYs(XYs_data, TT_JD)
    
    # Add in Free Core Nutation (FCN) correction
    X = EOP_data['dX'] + X  # rad
    Y = EOP_data['dY'] + Y  # rad
    
    # Compute Bias-Precssion-Nutation (BPN) matrix
    BPN = compute_BPN(X, Y, s)
    
    # Transform position vector
    ecef2eci = np.dot(BPN, np.dot(R, W))
    r_GCRF = np.dot(ecef2eci, r_ITRF)
    
    # Transform velocity vector
    # Calculate Earth rotation rate, rad/s (Vallado p227)
    wE = 7.29211514670639e-5*(1 - EOP_data['LOD']/86400)                    
    r_TIRS = np.dot(W, r_ITRF)
    
    v_GCRF = np.dot(BPN, np.dot(R, (np.dot(W, v_ITRF) + 
                    np.cross(np.array([[0.],[0.],[wE]]), r_TIRS, axis=0))))  
    
    return r_GCRF, v_GCRF


###############################################################################
# Unit Test
###############################################################################


if __name__ == '__main__':

    obj_id_list = [43014]
    UTC = datetime(2018, 6, 23, 0, 0, 0)   
    
    r_GCRF = np.array([[ 1970.55034496],
                       [-5808.61407296],
                       [ 3477.84103265]])
    v_GCRF = np.array([[ 0.01631311],
                       [-3.68975479],
                       [-6.53041229]])
    
    eop_alldata = get_celestrak_eop_alldata()    
    EOP_data = get_eop_data(eop_alldata, UTC)
    
    print(EOP_data)
    
    r_ITRF, v_ITRF = gcrf2itrf(r_GCRF, v_GCRF, UTC, EOP_data)
    
    print(r_ITRF)
    print(v_ITRF)
    
    UTC = datetime(2018, 6, 23, 2, 13, 21)
    r_GCRF = np.array([[-7.47620899e+02],
                       [ 3.91726014e+03],
                       [-5.60901124e+03]])
    
    v_GCRF = np.array([[ 2.71877455e-03],
                       [-6.23954021e+00],
                       [-4.34951830e+00]])
    
    r_ITRF = np.array([[-3651.380321],
                       [ 1598.487431],
                       [-5610.448359]])
    
    v_ITRF = np.array([[ 5.276523548],
                       [-3.242081015],
                       [-4.349310553]])
    
    EOP_data = get_eop_data(eop_alldata, UTC)
    r_check, v_check = gcrf2itrf(r_GCRF, v_GCRF, UTC, EOP_data)
    
    print(r_check)
    print(v_check)
    
    print(r_check - r_ITRF)
    print(v_check - v_ITRF)
    
    r_check2, v_check2 = itrf2gcrf(r_check, v_check, UTC, EOP_data)
    
    print(r_check2)
    print(v_check2)
    
    print(r_check2 - r_GCRF)
    print(v_check2 - v_GCRF)
    
    
    r_GCRF = np.array([[10000.], [5000.], [3000.]])
    v_GCRF = np.array([[-2.], [-3.], [4.]])
    
    UTC = datetime(2018, 6, 25, 6, 30, 10)
    EOP_data = get_eop_data(eop_alldata, UTC)
    print(EOP_data)
    
    r_ITRF, v_ITRF = gcrf2itrf(r_GCRF, v_GCRF, UTC, EOP_data)
    r_check, v_check = itrf2gcrf(r_ITRF, v_ITRF, UTC, EOP_data)
    
    print(r_ITRF)
    print(v_ITRF)
    print(r_GCRF - r_check)
    print(v_GCRF - v_check)
    
    
    
