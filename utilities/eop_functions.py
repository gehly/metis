import numpy as np
from math import pi, sin, cos, modf
import requests
import pandas as pd
import os
import sys
import pickle
import copy
from datetime import datetime

sys.path.append('../')

cwd = os.getcwd()
ind = cwd.find('metis')
metis_dir = cwd[0:ind+5]
input_data_dir = os.path.join(metis_dir, 'input_data')

from utilities.time_systems import dt2mjd, mjd2dt, utcdt2ut1jd, utcdt2ttjd, jd2cent
from utilities.numerical_methods import interp_lagrange

###############################################################################
#
# This file contains functions to retrieve Earth Orientation Parameters (EOPs)
# and compute rotation matrices.
#
# References:
# 
#  [1] Vallado, D.A. "Fundamentals of Astrodynamics and Applications".
#         Third Edition. Microcosm Press. 2007.
#
#  [2] Montenbruck, O. and Gill, E., "Satellite Orbits: Models, Methods,
#         Applications," Corrected Third Printing, 2005, Springer-Verlag
#         Berlin Heidelberg, 2000.
#
#  [3] McCarthy, Dennis D., "IERS Conventions (1996)," International Earth 
#        Rotation and Reference Systems Service (IERS), U.S. Naval 
#        Observatory, IERS Technical Note No. 21, July 1996.
#
#  [4] Bradley, B.K., Sibois, A., and Axelrad, P., "Influence of ITRS/GCRS
#        implementation for astrodynamics: Coordinate transformations,"
#        Advances in Space Research, Vol 57, 2016, pp 850-866.
#
#  [5] Petit, Gerard and Brian Luzum, "IERS Conventions (2010)," 
#        International Earth Rotation and Reference Systems Service (IERS),
#        IERS Conventions Center, IERS Technical Note No. 36, December
#        2010.
#
#  [6] International Astronomical Union, "Standards of Fundamental 
#        Astronomy: SOFA Tools for Earth Attitude," September 5, 2010.
#
#  [7] Coppola, V., J.H. Seago, and D.A. Vallado, "The IAU 2000A and IAU
#         2006 Precession-Nutation Theories and their Implementation," 
#         AAS 09-159, 2009.
#
#  [8] Wallace, P.T. and N. Capitaine, "Precession-Nutation Precedures
#         Consistent with IAU 2006 Resolutions," Astronomy & Astrophysics,
#         Volume 459, pages 981-985, December 2006.
#
#
###############################################################################


def get_celestrak_eop_alldata(offline_flag=False):
    '''
    This function retrieves the full EOP data file from celestrak.com.
    
    Format
    ------
    http://celestrak.com/SpaceData/EOP-format.asp
    
    Parameters
    ------
    offline_flag : boolean, optional
        flag to indicate internet access, if True will load data from local
        files (default=False)
    
    Returns
    ------
    data_text : string
        string containing observed and predicted EOP data, no header
        information
    '''
    
    if offline_flag:
        
        # Load data from file
        fname = os.path.join('../input_data', 'eop_alldata.pkl')        
        pklFile = open(fname, 'rb')
        data = pickle.load(pklFile)
        data_text = data[0]
        pklFile.close() 
        
    else:
        
        # Retrieve data from internet
        pageData = 'https://celestrak.com/SpaceData/eop19620101.txt'
#        pageData = 'http://www.celestrak.com/SpaceData/EOP-Last5Years.txt'
    
        r = requests.get(pageData)
        if r.status_code != requests.codes.ok:
            print("Error: Page data request failed.")            
        
        ind_BEGIN_OBSERVED = r.text.find('BEGIN OBSERVED')
        ind_END_OBSERVED = r.text.find('END OBSERVED')
        ind_BEGIN_PREDICTED = r.text.find('BEGIN PREDICTED')
        ind_END_PREDICTED = r.text.find('END PREDICTED')
    
        # Reduce to data
        data_text = r.text[ind_BEGIN_OBSERVED+16:ind_END_OBSERVED] \
            + r.text[ind_BEGIN_PREDICTED+17:ind_END_PREDICTED]

    return data_text


def save_celestrak_eop_alldata():
    '''
    This function saves EOP data in a pickle file for use when offline or
    for reducing computational demand.
    
    '''
    
    data_text = get_celestrak_eop_alldata()
    
    fname = os.path.join(input_data_dir, 'eop_alldata.pkl')
    
    # Save data    
    pklFile = open( fname, 'wb' )
    pickle.dump( [data_text], pklFile, -1 )
    pklFile.close()
    
    return


def get_eop_data(data_text, UTC):
    '''
    This function retrieves the EOP data for a specific time by computing
    a linear interpolation of parameters from the two closest times.
    
    Parameters
    ------
    data_text : string
        string containing observed and predicted EOP data, no header
        information
    UTC : datetime object
        time in UTC
    
    Returns
    ------
    EOP_data : dictionary
        EOP data for the given time including pole coordinates and offsets,
        time offsets, and length of day
    '''    
        
    # Compute MJD for desired time
    MJD = dt2mjd(UTC)
    MJD_int = int(MJD)
    
    # Find EOP data lines around time of interest
    nchar = 102
    nskip = 1
    nlines = 0
    for ii in range(len(data_text)):
        start = ii + nlines*(nchar+nskip)
        stop = ii + nlines*(nchar+nskip) + nchar
        line = data_text[start:stop]
        nlines += 1
        
        MJD_line = int(line[11:16])
        
        if MJD_line == MJD_int:
            line0 = line
        if MJD_line == MJD_int+1:
            line1 = line
            break
    
    # Compute EOP data at desired time by interpolating
    EOP_data = eop_linear_interpolate(line0, line1, MJD)
    
    return EOP_data


def eop_linear_interpolate(line0, line1, MJD):
    '''
    This function computes the linear interpolation of EOP parameters between
    two lines of the EOP data file.
    
    Parameters
    ------
    line0 : string
        EOP data line from time before desired MJD
    line1 : string
        EOP data line from time after desired MJD
    MJD : float
        fractional days since 1858-11-17 in UTC
    
    Returns
    ------
    EOP_data : dictionary
        EOP data for the given time including pole coordinates and offsets,
        time offsets, and length of day        
    '''    
    
    # Initialize output
    EOP_data = {}
    
    # Leap seconds do not interpolate
    EOP_data['TAI_UTC'] = int(line0[99:102])
    
    # Retrieve values
    line0_array = eop_read_line(line0)
    line1_array = eop_read_line(line1)
    
    # Adjust UT1-UTC column in case leap second occurs between lines
    line0_array[3] -= line0_array[9]
    line1_array[3] -= line1_array[9]
    
    # Linear interpolation
    dt = MJD - line0_array[0]
    interp = (line1_array[1:] - line0_array[1:])/ \
        (line1_array[0] - line0_array[0]) * dt + line0_array[1:]

    # Convert final output
    arcsec2rad = (1./3600.) * pi/180.
    EOP_data['xp'] = interp[0]*arcsec2rad
    EOP_data['yp'] = interp[1]*arcsec2rad
    EOP_data['UT1_UTC'] = interp[2] + EOP_data['TAI_UTC']
    EOP_data['LOD'] = interp[3]
    EOP_data['ddPsi'] = interp[4]*arcsec2rad
    EOP_data['ddEps'] = interp[5]*arcsec2rad
    EOP_data['dX'] = interp[6]*arcsec2rad
    EOP_data['dY'] = interp[7]*arcsec2rad
    

    return EOP_data


def eop_read_line(line):
    '''
    This function reads a single line of the EOP data file and returns the
    floating point values of each parameter per the format given below.
    
    http://celestrak.com/SpaceData/EOP-format.asp
    
    Columns   Description
    001-004	Year
    006-007	Month (01-12)
    009-010	Day
    012-016	Modified Julian Date (Julian Date at 0h UT minus 2400000.5)
    018-026	x (arc seconds)
    028-036	y (arc seconds)
    038-047	UT1-UTC (seconds)
    049-058	Length of Day (seconds)
    060-068	delta-Delta-psi (arc seconds)
    070-078	delta-Delta-epsilon (arc seconds)
    080-088	delta-X (arc seconds)
    090-098	delta-Y (arc seconds)
    100-102	Delta Atomic Time, TAI-UTC (seconds)
    
    Parameters
    ------
    line : string
        single line from EOP data file (format as specified)
    
    Returns
    ------
    line_array : 1D numpy array
        EOP parameters in 1D array    
    '''
    
    MJD = float(line[11:16])
    xp = float(line[17:26])
    yp = float(line[27:36])
    UT1_UTC = float(line[37:47])
    LOD = float(line[48:58])
    ddPsi = float(line[59:68])
    ddEps = float(line[69:78])
    dX = float(line[79:88])
    dY = float(line[89:98])
    TAI_UTC = float(line[99:102])
    
    line_array = np.array([MJD, xp, yp, UT1_UTC, LOD, ddPsi, ddEps, dX, dY,
                           TAI_UTC])
    
    return line_array

        
def get_nutation_data(TEME_flag=True):
    '''
    This function retrieves nutation data from the IAU 1980 CSV file included
    in this distribution, compiled from Reference [2].  For the conversion
    from TEME to GCRF, it is recommended to reduce the coefficients to the
    four largest terms, which can be done using the optional input flag.
    
    Parameters
    ------
    TEME_flag : boolean, optional
        flag to determine whether to reduce coefficient array to four largest
        rows (default=True)
    
    Returns
    ------
    IAU1980_nutation : 2D numpy array
        array of nutation coefficients    
    '''
    
    df = pd.read_csv(os.path.join(input_data_dir, 'IAU1980_nutation.csv'))
    
    # For TEME-GCRF conversion, reduce to 4 largest entries      
    if TEME_flag:          
        df = df.loc[np.abs(df['dPsi']) > 2000.]
        
    IAU1980_nutation = df.values
    
    return IAU1980_nutation


def get_XYs2006_alldata():
    
    # Load data
    XYs_df = pd.read_csv(os.path.join(input_data_dir, 'IAU2006_XYs.csv'))
    
    return XYs_df


def init_XYs2006(TT1, TT2, XYs_df=[]):
    '''
    This loads the data file containing CIP coordinates, X and Y, as well as 
    the CIO locator, s. The data file is named IAU2006_XYs.csv.
    X, Y, and s are tabulated from 1980 to 2050 every day at 0h 
    Terrestrial Time (TT). 

    The data is loaded into a single matrix and then trimmed. The resulting 
    XYsdata matrix contains data from 8 days before TT1 to 8 days after TT2 
    for interpolation purposes.

    NOTE: although TT is used for input, UTC can also be used without any
        issues.  The difference between TT and UTC is about 60 seconds. 
        Since data is trimmed for +/- 8 days on either side of the input
        times, UTC is fine.  The resulting data matrix will still contain
        X,Y,s data for 0h of TT though.
        
    Parameters
    ------
    TT1 : datetime object
        start time in TT
    TT2 : datetime object
        final time in TT
        
    Returns
    ------
    XYs_data : nx7 numpy array
        each row contains data for 0h TT for consecutive days
        [yr, mo, day, MJD, X, Y, s]
    
    '''
    
    # Load data if needed
    if len(XYs_df) == 0:        
        XYs_df = pd.read_csv(os.path.join(input_data_dir, 'IAU2006_XYs.csv'))        
        
    XYs_alldata = XYs_df.values
    
    # Compute MJD and round to nearest whole day
    MJD1 = int(round(dt2mjd(TT1)))
    MJD2 = int(round(dt2mjd(TT2)))
    
    # Number of additional data points to include on either side
    num = 10
    
    # Find rows
    MJD_data = XYs_df['MJD (0h TT)'].tolist()
    
    if MJD1 < MJD_data[0]:
        print('Error: init_XYs2006 start date before first XYs time')
    elif MJD1 <= MJD_data[0] + num:
        row1 = 0
    elif MJD1 > MJD_data[-1]:
        print('Error: init_XYs2006 start date after last XYs time')
    else:
        row1 = MJD_data.index(MJD1) - num
    
    if MJD2 < MJD_data[0]:
        print('Error: init_XYs2006 final date before first XYs time')
    elif MJD2 >= MJD_data[0] - num:
        row2 = -1
    elif MJD2 > MJD_data[-1]:
        print('Error: init_XYs2006 final date after last XYs time')
    else:
        row2 = MJD_data.index(MJD2) + num
        
    if row2 == -1:
        XYs_data = XYs_alldata[row1:, :]
    else:
        XYs_data = XYs_alldata[row1:row2, :]
    
    return XYs_data


def get_XYs(XYs_data, TT_JD):
    '''
    Interpolates X,Y, and s loaded by init_XYs2006.m using an 11th-order 
    Lagrange interpolation method. The init_XYsdata function must be 
    called before get_XYs is used.  This function uses the XYs data set that 
    has been loaded as a matrix. Each of the three values listed below are 
    tabulated at 0h TT of each day.
    
    Parameters
    ------
    XYs_data : nx7 numpy array
        each row contains data for 0h TT for consecutive days
        [yr, mo, day, MJD, X, Y, s]
    TT_JD : float
        fractional days since 12:00:00 Jan 1 4713 BC TT
 
    
    Returns
    ------
    X : float
        x-coordinate of the Celestial Intermediate Pole (CIP) [rad]
    Y : float
        y-coordinate of the Celestial Intermediate Pole (CIP) [rad]
    s : float
        Celestial Intermediate Origin (CIO) locator [rad]
    
    '''
    
    # Conversion
    arcsec2rad  = (1./3600.) * (pi/180.)
    
    # Compute MJD
    TT_MJD = TT_JD - 2400000.5
    
    # Compute interpolation
    XYs = interp_lagrange(XYs_data[:,3], XYs_data[:,4:], TT_MJD, 11)
    
    X = float(XYs[0,0])*arcsec2rad
    Y = float(XYs[0,1])*arcsec2rad
    s = float(XYs[0,2])*arcsec2rad
    
    return X, Y, s


def compute_precession_IAU1976(TT_cent):
    '''    
    This function computes the IAU1976 precession matrix required for the 
    frame transformation between GCRF and Mean of Date (MOD).
    
    r_GCRF = P76 * r_MOD
    
    Parameters
    ------
    TT_cent : float
        Terrestrial Time (TT) since J2000 in Julian centuries
    
    Returns
    ------
    P76 : 3x3 numpy array
        precession matrix to compute frame rotation    
    '''
    
    # Conversion
    arcsec2rad  = (1./3600.) * (pi/180.)

    # Table values in arcseconds
    Pcoef = np.array([[2306.2181,   0.30188,   0.017998], 
                      [2004.3109,  -0.42665,  -0.041833],
                      [2306.2181,   1.09468,   0.018203]])

    
 
    # Multiply by [TT, TT**2, TT**3]^T  (creates column vector)
    # M[0] = zeta, M[1] = theta, M[2] = z
    vec = np.array([[TT_cent], [TT_cent**2.], [TT_cent**3.]])
    M = np.dot(Pcoef, vec) * arcsec2rad;

    # Construct IAU 1976 Precession Matrix
    # P76 = ROT3(zeta) * ROT2(-theta) * ROT3(z);    
    czet = cos(M[0])
    szet = sin(M[0])  
    cth  = cos(M[1])
    sth  = sin(M[1]) 
    cz   = cos(M[2])
    sz   = sin(M[2])
    
    
    P76 = np.array([[cth*cz*czet-sz*szet,   sz*cth*czet+szet*cz,  sth*czet],
                    [-szet*cth*cz-sz*czet, -sz*szet*cth+cz*czet, -sth*szet],
                    [-sth*cz,              -sth*sz,                    cth]])
    
    
    return P76


def compute_nutation_IAU1980(IAU1980nut, TT_cent, ddPsi, ddEps):
    '''
    This function computes the IAU1976 precession matrix required for the 
    frame transformation between Mean of Date (MOD) and True of Date (TOD).
    
    r_MOD = N80 * r_TOD
    
    Parameters
    ------
    IAU1980nut : 2D numpy array
        array of nutation coefficients  
    TT_cent : float
        Terrestrial Time (TT) since J2000 in Julian centuries
    ddPsi : float
        EOP parameter for correction to nutation in longitude [rad]
    ddEps : float
        EOP parameter for correction to nutation in obliquity [rad]
    
    Returns
    ------  
    N80 : 3x3 numpy array
        nutation matrix to compute frame rotation
    FA : 5x1 numpy array
        fundamental arguments of nutation (Delauney arguments)
    Eps0 : float
        mean obliquity of the ecliptic [rad]
    Eps_true : float
        true obliquity of the ecliptic [rad]
    dPsi : float
        nutation in longitude [rad]
    dEps : float
        nutation in obliquity [rad]    
    '''
    
    # Conversion
    arcsec2rad  = (1./3600.) * (pi/180.)
    
    # Compute fundamental arguments of nutation
    FA = compute_fundarg_IAU1980(TT_cent)
    
    # Compute Nutation in longitude and obliquity  
    phi = np.dot(IAU1980nut[:,0:5], FA)  # column vector
    sphi = np.sin(phi)
    cphi = np.cos(phi)

    # Calculate Nutation in Longitude, rad    
    dPsi_vec = IAU1980nut[:,5] + IAU1980nut[:,6]*TT_cent
    dPsi_sum = float(np.dot(dPsi_vec.T, sphi))
    dPsi = ddPsi + dPsi_sum*0.0001*arcsec2rad

    # Calculate Nutation in Obliquity, rad
    dEps_vec = IAU1980nut[:,7] + IAU1980nut[:,8]*TT_cent
    dEps_sum = float(np.dot(dEps_vec.T, cphi))
    dEps = ddEps + dEps_sum*0.0001*arcsec2rad
        
    # Mean Obliquity of the Ecliptic, rad
    Eps0 = (((0.001813*TT_cent - 0.00059)*TT_cent - 46.8150)*TT_cent + 
             84381.448)*arcsec2rad
    
    # True Obliquity of the Ecliptic, rad
    Eps_true = Eps0 + dEps
    
    # Construct Nutation matrix
    # N = ROT1(-Eps_0 * ROT3(dPsi) * ROT1(Eps_true)
    cep  = cos(Eps0)
    sep  = sin(Eps0)
    cPsi = cos(dPsi)
    sPsi = sin(dPsi)
    cept = cos(Eps_true)
    sept = sin(Eps_true)    
    
    N80 = \
        np.array([[ cPsi,     sPsi*cept,              sept*sPsi             ],
                  [-sPsi*cep, cept*cPsi*cep+sept*sep, sept*cPsi*cep-sep*cept],
                  [-sPsi*sep, sep*cept*cPsi-sept*cep, sept*sep*cPsi+cept*cep]])
    
    return N80, FA, Eps0, Eps_true, dPsi, dEps


def compute_fundarg_IAU1980(TT_cent):
    '''
    This function computes the fundamental arguments (Delauney arguments) due
    to luni-solar forces.
    
    Parameters
    ------
    TT_cent : float
        Terrestrial Time (TT) since J2000 in Julian centuries
    
    Returns
    ------
    DA_vec : 5x1 numpy array
        fundamental arguments of nutation (Delauney arguments) [rad]
    
    '''
    
    # Conversion
    arcsec2rad  = (1./3600.) * (pi/180.)
    arcsec360 = 3600.*360.
    
    # Construct table for fundamental arguments of nutation
    #  Units: col 1,   degrees
    #         col 2-5, arcseconds
    #  Note: These values come from page 23 of [3].
    
    # Delauney Arguments
    DA = np.array([[134.96340251,  1717915923.2178,  31.8792,  0.051635, -0.00024470], # M_moon (l)
                   [357.52910918,  129596581.04810, -0.55320,  0.000136, -0.00001149], # M_sun (l')
                   [93.27209062,   1739527262.8478, -12.7512, -0.001037,  0.00000417], # u_Mmoon (F)
                   [297.85019547,  1602961601.2090, -6.37060,  0.006593, -0.00003169], # D_sun (D)
                   [125.04455501, -6962890.2665000,  7.47220,  0.007702, -0.00005939]]) # Om_moon (Omega)

    # Mulitply by [3600., TT, TT**2, TT**3, TT**4]^T to get column vector 
    # in arcseconds
    vec = np.array([[3600.], [TT_cent], [TT_cent**2.], [TT_cent**3],
                    [TT_cent**4]])
    DA_vec = np.dot(DA, vec)
    
    # Get fractional part of circle and convert to radians
    DA_vec = np.mod(DA_vec, arcsec360) * arcsec2rad
    
    return DA_vec


def eqnequinox_IAU1982_simple(dPsi, Eps0):
    '''
    This function computes the IAU1982 equation of the equinoxes matrix 
    required for the frame transformation between True of Date (TOD) and
    True Equator Mean Equinox (TEME).    
    
    r_TOD = R * r_TEME
    
    Parameters
    ------
    dPsi : float
        nutation in longitude [rad]
    Eps0 : float
        mean obliquity of the ecliptic [rad]
    
    Returns
    ------
    R : 3x3 numpy array
        matrix to compute frame rotation    
    '''
    
    # Equation of the Equinoxes (simple form for use with TEME) (see [1])
    Eq1982 = dPsi*cos(Eps0) # rad
    
    # Construct Rotation matrix
    # R  = ROT3(-Eq1982) (Eq. 3-80 in [1])
    cEq = cos(Eq1982) 
    sEq = sin(Eq1982)

    R = np.array([[cEq,    -sEq,    0.],
                  [sEq,     cEq,    0.],
                  [0.,      0.,     1.]])
    

    return R


def compute_polarmotion(xp, yp, TT_cent):
    '''
    This function computes the polar motion transformation matrix required
    for the frame transformation between TIRS and ITRF.    
    
    r_TIRS = W * r_ITRF
    
    Parameters
    ------
    xp : float
        x-coordinate of the CIP unit vector [rad]
    yp : float
        y-coordinate of the CIP unit vector [rad]
    TT_cent : float
        Julian centuries since J2000 TT
    
    Returns
    ------
    W : 3x3 numpy array
        matrix to compute frame rotation    
    '''
    
    # Conversion
    arcsec2rad  = (1./3600.) * (pi/180.)
    
    # Calcuate the Terrestrial Intermediate Origin (TIO) locator 
    # Eq 5.13 in [5]
    sp = -0.000047 * TT_cent * arcsec2rad
    
    # Construct rotation matrix
    # W = ROT3(-sp)*ROT2(xp)*ROT1(yp) (Eq. 5.3 in [5])
    cx = cos(xp)
    sx = sin(xp)
    cy = cos(yp)
    sy = sin(yp)
    cs = cos(sp)
    ss = sin(sp)
    
    W = np.array([[cx*cs,  -cy*ss + sy*sx*cs,  -sy*ss - cy*sx*cs],
                  [cx*ss,   cy*cs + sy*sx*ss,   sy*cs - cy*sx*ss],
                  [   sx,             -sy*cx,              cy*cx]])
    
    
    return W


def compute_ERA(UT1_JD):
    '''
    This function computes the Earth Rotation Angle (ERA) and the ERA rotation
    matrix based on UT1 time. The ERA is modulated to lie within [0,2*pi] and 
    is computed using the precise equation given by Eq. 5.15 in [5].

    The ERA is the angle between the Celestial Intermediate Origin, CIO, and 
    Terrestrial Intermediate Origin, TIO (a reference meridian 100m offset 
    from Greenwich meridian).
    
    r_CIRS = R * r_TIRS
   
    Parameters
    ------
    UT1_JD : float
        fractional days since 12:00:00 Jan 1 4713 BC UT1
        
    Returns
    ------
    R : 3x3 numpy array
        matrix to compute frame rotation
    
    '''
    
    # Compute ERA based on Eq. 5.15 of [5]
    d,i = modf(UT1_JD)
    ERA = 2.*pi*(d + 0.7790572732640 + 0.00273781191135448*(UT1_JD - 2451545.))
    
    # Compute ERA between [0, 2*pi]
    ERA = ERA % (2.*pi)
    if ERA < 0.:
        ERA += 2*pi
        
#    print(ERA)
    
    # Construct rotation matrix
    # R = ROT3(-ERA)
    ct = cos(ERA)
    st = sin(ERA)
    R = np.array([[ct, -st, 0.],
                  [st,  ct, 0.],
                  [0.,  0., 1.]])

    return R


def compute_BPN(X, Y, s):
    '''
    This function computes the Bias-Precession-Nutation matrix required for the 
    CIO-based transformation between the GCRF/ITRF frames.
    
    r_GCRS = BPN * r_CIRS
    
    Parameters
    ------
    X : float
        x-coordinate of the Celestial Intermediate Pole (CIP) [rad]
    Y : float
        y-coordinate of the Celestial Intermediate Pole (CIP) [rad]
    s : float
        Celestial Intermediate Origin (CIO) locator [rad]
    
    Returns
    ------
    BPN : 3x3 numpy array
        matrix to compute frame rotation
    
    '''
    
    # Compute z-coordinate of CIP
    Z  = np.sqrt(1 - X*X - Y*Y)
    aa = 1./(1. + Z)
    
    # Construct BPN (Bias-Precession-Nutation Matrix) 
    # Eq. 5.1 in [5]:  BPN = [f(X,Y)]*ROT3(s)
    cs = cos(s)
    ss = sin(s)
    
    f = np.array([[1-aa*X*X,    -aa*X*Y,                X],
                  [ -aa*X*Y,   1-aa*Y*Y,                Y], 
                  [      -X,         -Y,   1-aa*(X*X+Y*Y)]])
    
    R3 = np.array([[ cs,  ss,  0.],
                   [-ss,  cs,  0.],
                   [ 0.,  0.,  1.]])
    
    BPN = np.dot(f, R3)
    
    return BPN


def batch_eop_rotation_matrices(UTC_start, UTC_stop, eop_alldata_text,
                                increment=10., eop_flag='linear',
                                GMST_only_flag=False):
    '''
    This function generates a list of rotation matrices between TEME/GCRF and
    GCRF/ITRF for use in coordinate transformations.  The function can process
    a large array of times and has flags to control the level of fidelity of
    the transformation, in order to facilitate good computational performance
    for a large number of transforms.
    
    Parameters
    ------
    UTC_start : datetime object
        start time in UTC
    UTC_stop : datetime object
        stop time in UTC
    eop_alldata_text : string
        string containing observed and predicted EOP data, no header
        information
    increment : float, optional
        time increment between desired frame rotations [sec] (default=10.)
    eop_flag : string, optional
        flag to determine how to determine EOP parameters from input text data
        'zeros' = set all EOP values to zero (~2km error)
        'nearest' = set EOP values to nearest whole day (~3m error)
        'linear' = linearly interpolate EOP values between 2 nearest days (~20cm error)
        (default = 'linear')
    GMST_only_flag : boolean, optional
        flag to determine whether to apply only the Earth rotation angle for
        the GCRF/ITRF transformation (i.e., no precession, nutation, polar motion)
        True = apply GMST Earth rotation angle only (no P, N, W)
        False = apply full transformation including P, N, W
        (default = False)
    
    Returns
    ------
    GCRF_TEME_list : list of 3x3 numpy arrays
        transformation matrix at each time for GCRF/TEME
        r_GCRF = GCRF_TEME * r_TEME
    ITRF_GCRF_list : list of 3x3 numpy arrays
        transformation matrix at each time for GCRF/TEME
        r_ITRF = ITRF_GCRF * r_GCRF
    
    '''    
    
    # Initialize Output
    GCRF_TEME_list = []
    ITRF_GCRF_list = []
    
    # Constants    
    arcsec2rad = (1./3600.) * pi/180.
    
    # Retrieve IAU Nutation data from file
    IAU1980_nut = get_nutation_data()
    
    # Retrieve polar motion data from file
    XYs_df = get_XYs2006_alldata()
    
    # MJD Array
    MJD0 = dt2mjd(UTC_start)
    MJDf = dt2mjd(UTC_stop)
    MJD_array = np.arange(MJD0, MJDf+(increment/(86400.*2.)), increment/86400.)
    
    # Loop over times
    for kk in range(len(MJD_array)):
        
        MJD_kk = MJD_array[kk]
        UTC_kk = mjd2dt(MJD_kk)
        
        # Compute the appropriate EOP values for this time using the input
        # text data.  Per Reference 4 Table 2-3, the following error levels are
        # expected for different levels of fidelity in the approximation
        
        # Set all EOPs to zero
        # Expected error level ~2km in GEO
        if eop_flag == 'zeros':
            
            xp = 0.
            yp = 0.
            UT1_UTC = 0.
            LOD = 0.
            ddPsi = 0.
            ddEps = 0.
            dX = 0.
            dY = 0.
            TAI_UTC = 0.
        
        # Read EOP text data for higher level approximations
        else:
            nchar = 102
            nskip = 1
            nlines = 0
            MJD_int = int(MJD_kk)
            
            # Get closest lines from EOP text data
            # First index have to search text data
            if kk == 0:
                
                MJD_prior = copy.copy(MJD_int)
                
                # Find EOP data lines around time of interest                
                for ii in range(len(eop_alldata_text)):
                    start = ii + nlines*(nchar+nskip)
                    stop = ii + nlines*(nchar+nskip) + nchar
                    line = eop_alldata_text[start:stop]
                    nlines += 1
                    
                    MJD_line = int(line[11:16])
                    
                    if MJD_line == MJD_int:
                        line0 = line
                    if MJD_line == MJD_int+1:
                        line1 = line
                        break
                
            # Otherwise, we can use previous knowledge
            else:
                
                # If it's the same day as last time we can just reuse line0
                # and line1.  Otherwise, we have to increment nlines to get
                # the new line1                
                if MJD_int == MJD_prior + 1:
                    line0 = copy.copy(line1)
                    nlines += 1
                    start = ii + nlines*(nchar+nskip)
                    stop = ii + nlines*(nchar+nskip) + nchar
                    line1 = eop_alldata_text[start:stop]
                    
                    MJD_line = int(line[11:16])
                    if MJD_line != MJD_int:
                        print(MJD_line)
                        print(MJD_int)
                        sys.exit('Wrong MJD value encountered!')
                    
                    MJD_prior = MJD_int
                
                # Read the EOP values for each line
                line0_array = eop_read_line(line0)
                line1_array = eop_read_line(line1)
            
            # Set all EOPs to values for nearest day
            # Expected error level ~3.6m max (0.8m RMS) in GEO
            if eop_flag == 'nearest':
                if (MJD_kk - MJD_int) < 0.5:
                    xp = line0_array[1]*arcsec2rad
                    yp = line0_array[2]*arcsec2rad
                    UT1_UTC = line0_array[3]
                    LOD = line0_array[4]
                    ddPsi = line0_array[5]*arcsec2rad
                    ddEps = line0_array[6]*arcsec2rad
                    dX = line0_array[7]*arcsec2rad
                    dY = line0_array[8]*arcsec2rad
                    TAI_UTC = line0_array[9]
                    
                else:
                    xp = line1_array[1]*arcsec2rad
                    yp = line1_array[2]*arcsec2rad
                    UT1_UTC = line1_array[3]
                    LOD = line1_array[4]
                    ddPsi = line1_array[5]*arcsec2rad
                    ddEps = line1_array[6]*arcsec2rad
                    dX = line1_array[7]*arcsec2rad
                    dY = line1_array[8]*arcsec2rad
                    TAI_UTC = line1_array[9]
            
            # Linear interpolation of values between two nearest days
            # Expected error level ~22cm max (4cm RMS) in GEO
            if eop_flag == 'linear':
                                
                # Adjust UT1-UTC column in case leap second occurs between lines
                line0_array[3] -= line0_array[9]
                line1_array[3] -= line1_array[9]
                
                # Leap seconds do not interpolate
                TAI_UTC = line0_array[9]
            
                # Linear interpolation
                dt = MJD_kk - line0_array[0]
                interp = (line1_array[1:] - line0_array[1:])/ \
                    (line1_array[0] - line0_array[0]) * dt + line0_array[1:]
            
                # Convert final output
                xp = interp[0]*arcsec2rad
                yp = interp[1]*arcsec2rad
                UT1_UTC = interp[2] + TAI_UTC
                LOD = interp[3]
                ddPsi = interp[4]*arcsec2rad
                ddEps = interp[5]*arcsec2rad
                dX = interp[6]*arcsec2rad
                dY = interp[7]*arcsec2rad
                
        
        
        
        # Compute rotation matrices for transform
        # Compute current times
        UT1_JD = utcdt2ut1jd(UTC_kk, UT1_UTC)
        TT_JD = utcdt2ttjd(UTC_kk, TAI_UTC)
        TT_cent = jd2cent(TT_JD)
        
        # GCRF/ITRF Transformation
        # Contruct Earth rotaion angle matrix (TIRS to CIRS)
        R_CIRS = compute_ERA(UT1_JD)
        
        if GMST_only_flag:
            ITRF_GCRF = R_CIRS.T
            
        else:            
            # Construct polar motion matrix (ITRS to TIRS)
            W = compute_polarmotion(xp, yp, TT_cent)
            
            # Construct Bias-Precessino-Nutation matrix (CIRS to GCRS/ICRS)
            XYs_data = init_XYs2006(UTC_kk, UTC_kk, XYs_df)
            
            X, Y, s = get_XYs(XYs_data, TT_JD)
            
            # Add in Free Core Nutation (FCN) correction
            X = dX + X  # rad
            Y = dY + Y  # rad
            
            # Compute Bias-Precssion-Nutation (BPN) matrix
            BPN = compute_BPN(X, Y, s)
            
            # Transform position vector
            ITRF_GCRF = np.dot(W.T, np.dot(R_CIRS.T, BPN.T))
        
        # TEME/GCRF Transformation
        # IAU 1976 Precession
        P = compute_precession_IAU1976(TT_cent)
        
        # IAU 1980 Nutation
        N, FA, Eps0, Eps_true, dPsi, dEps = \
            compute_nutation_IAU1980(IAU1980_nut, TT_cent, ddPsi, ddEps)
    
        # Equation of the Equinonx 1982
        R_1982 = eqnequinox_IAU1982_simple(dPsi, Eps0)
        
        # Compute transformation matrix and output
        GCRF_TEME = np.dot(P, np.dot(N, R_1982))
        
        # Store output
        GCRF_TEME_list.append(GCRF_TEME)
        ITRF_GCRF_list.append(ITRF_GCRF)
    
    return GCRF_TEME_list, ITRF_GCRF_list


if __name__ == '__main__':
    
#    save_celestrak_eop_alldata()
    
    UTC = datetime(2020, 8, 10, 0, 0, 0)
    
    eop_alldata = get_celestrak_eop_alldata()
    EOP_data = get_eop_data(eop_alldata, UTC)
    
#    print(EOP_data)
    
    UT1_JD = utcdt2ut1jd(UTC, EOP_data['UT1_UTC'])
    
    R = compute_ERA(UT1_JD)
    
    print(R)
    
    
