import numpy as np

from time_systems import utcdt2ttjd
from time_systems import utcdt2ut1jd
from time_systems import jd2cent
from eop_functions import compute_precession_IAU1976
from eop_functions import compute_nutation_IAU1980
from eop_functions import eqnequinox_IAU1982_simple


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


def gcrf2itrf(UTC, EOP_data):
    
    
    # Compute UT1 in JD format
    UT1_JD = utcdt2ut1jd(UTC, EOP_data['UT1_UTC'])
    
    # Compute TT in JD format
    TT_JD = utcdt2ttjd(UTC, EOP_data['TAI_UTC'])
    
    # Compute TT in centuries since J2000 epoch
    TT_cent = jd2cent(TT_JD)
    
    
    
    
    return r_ITRF, v_ITRF


def itrf2gcrf():
    
    
    
    return r_GCRF, v_GCRF






