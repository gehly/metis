from datetime import datetime, timedelta
from math import pi, sin


def dt2mjd(dt):
    '''
    This function converts a calendar time to Modified Julian Date (MJD)
    fractional days since 1858-11-17 00:00:00.  No conversion between time
    systems is performed.
    
    Parameters
    ------
    dt : datetime object
        time in calendar format
    
    Returns
    ------
    MJD : float
        fractional days since 1858-11-17 00:00:00
    '''
    
    MJD_datetime = datetime(1858, 11, 17, 0, 0, 0)
    delta = dt - MJD_datetime
    MJD = delta/timedelta(days=1)
    
    return MJD


def mjd2dt(MJD):
    '''
    This function converts a Modified Julian Date (MJD) to calendar datetime
    object.  MJD is fractional days since 1858-11-17 00:00:00.  No conversion 
    between time systems is performed.
    
    Parameters
    ------
    MJD : float
        fractional days since 1858-11-17 00:00:00
    
    Returns
    ------
    dt : datetime object
        time in calendar format
        
    '''
    
    MJD_datetime = datetime(1858, 11, 17, 0, 0, 0)
    dt = MJD_datetime + timedelta(days=MJD)
    
    return dt


def dt2jd(dt):
    '''
    This function converts a calendar time to Julian Date (JD) fractional days
    since 12:00:00 Jan 1 4713 BC.  No conversion between time systems is 
    performed.
    
    Parameters
    ------
    dt : datetime object
        time in calendar format
    
    Returns
    ------
    JD : float
        fractional days since 12:00:00 Jan 1 4713 BC
    
    '''
    
    MJD = dt2mjd(dt)
    JD = MJD + 2400000.5
    
    return JD


def jd2dt(JD):
    '''
    This function converts from Julian Date (JD) fractional days
    since 12:00:00 Jan 1 4713 BC to a calendar datetime object.  No conversion 
    between time systems is performed.
    
    Parameters
    ------
    JD : float
        fractional days since 12:00:00 Jan 1 4713 BC    
    
    Returns
    ------
    dt : datetime object
        time in calendar format
    
    '''
    
    MJD = JD - 2400000.5
    dt = mjd2dt(MJD)
    
    return dt


def utcdt2maisec(UTC):
    '''
    This function converts a UTC time to MAI seconds since 1980-01-06 00:00:00,
    the initial epoch of GPS time.  No conversion between time systems is made,
    the output is seconds in UTC since the initial GPS epoch, including leap
    seconds.
    
    Parameters
    ------
    UTC : datetime object
        time in UTC
    
    Returns
    MAI : float
        seconds in UTC since GPS epoch 1980-01-06 00:00:00
        
    '''
    
    MAI_datetime = datetime(1980, 1, 6, 0, 0, 0)
    delta = UTC - MAI_datetime
    MAI = delta.total_seconds()
    
    return MAI


def utcdt2ut1jd(UTC, UT1_UTC):
    '''
    This function converts a UTC time to UT1 in Julian Date (JD) format.
    
    UT1_UTC = UT1 - UTC
    
    Parameters
    ------
    UTC : datetime object
        time in UTC
    UT1_UTC : float
        EOP parameter, time offset between UT1 and UTC 
    
    Returns
    ------
    UT1_JD : float
        fractional days since 12:00:00 Jan 1 4713 BC UT1
    
    '''
    
    UTC_JD = dt2jd(UTC)
    UT1_JD = UTC_JD + (UT1_UTC/86400.)
    
    return UT1_JD


def utcdt2ttjd(UTC, TAI_UTC):
    '''
    This function converts a UTC time to Terrestrial Time (TT) in Julian Date
    (JD) format.
    
    UTC = TAI - TAI_UTC
    TT = TAI + 32.184
    
    Parameters
    ------
    UTC : datetime object
        time in UTC
    TAI_UTC : float
        EOP parameter, time offset between atomic time (TAI) and UTC 
        (10 + leap seconds)        
    
    Returns
    ------
    TT_JD : float
        fractional days since 12:00:00 Jan 1 4713 BC TT
    
    '''
    
    UTC_JD = dt2jd(UTC)
    TT_JD = UTC_JD + (TAI_UTC + 32.184)/86400.
    
    return TT_JD


def utcdt2taijd(UTC, TAI_UTC):
    '''
    This function converts a UTC time to Terrestrial Time (TT) in Julian Date
    (JD) format.
    
    UTC = TAI - TAI_UTC
    
    Parameters
    ------
    UTC : datetime object
        time in UTC
    TAI_UTC : float
        EOP parameter, time offset between atomic time (TAI) and UTC 
        (10 + leap seconds)        
    
    Returns
    ------
    TAI_JD : float
        fractional days since 12:00:00 Jan 1 4713 BC TAI
    
    '''
    
    UTC_JD = dt2jd(UTC)
    TAI_JD = UTC_JD + TAI_UTC/86400.
    
    return TAI_JD


def utcdt2gpsjd(UTC, TAI_UTC):
    '''
    This function converts a UTC time to Terrestrial Time (TT) in Julian Date
    (JD) format.
    
    UTC = TAI - TAI_UTC
    GPS = TAI - 19 sec
    
    Parameters
    ------
    UTC : datetime object
        time in UTC
    TAI_UTC : float
        EOP parameter, time offset between atomic time (TAI) and UTC 
        (10 + leap seconds)        
    
    Returns
    ------
    GPS_JD : float
        fractional days since 12:00:00 Jan 1 4713 BC GPS
    
    '''
    
    UTC_JD = dt2jd(UTC)
    GPS_JD = UTC_JD + (TAI_UTC - 19.)/86400.
    
    return GPS_JD


def utcdt2jedjd(UTC, TAI_UTC):
    '''
    This function converts a UTC time to Julian Ephemeris Date (JED) in Julian
    Date (JD) format.
    
    Parameters
    ------
    UTC : datetime object
        time in UTC
    TAI_UTC : float
        EOP parameter, time offset between atomic time (TAI) and UTC 
        (10 + leap seconds)        
    
    Returns
    ------
    JED_JD : float
        fractional days since 12:00:00 Jan 1 4713 BC JED
    
    '''    
    
    TT_JD = utcdt2ttjd(UTC, TAI_UTC)
    JED_JD = ttjd2jedjd(TT_JD)
    
    return JED_JD


def jd2cent(JD):
    '''
    This function computes Julian centuries since J2000. No conversion between
    time systems is performed.
    
    Parameters
    ------
    JD : float
        fractional days since 12:00:00 Jan 1 4713 BC
    
    Returns
    ------
    cent : float
        fractional centuries since 12:00:00 Jan 1 2000
    '''
    
    cent = (JD - 2451545.)/36525.
    
    return cent


def ttjd2jedjd(TT_JD):
    '''
    This function converts Terrestrial Time (TT) to Julian Ephemeris Date (JED)
    in Julian Date (JD) format.
    
    Parameters
    ------
    TT_JD : float
        fractional days since 12:00:00 Jan 1 4713 BC TT       
    
    Returns
    ------
    JED_JD : float
        fractional days since 12:00:00 Jan 1 4713 BC JED
    
    '''   
    
    # Find approximate mean anomaly of Earth
    TT_cent = jd2cent(TT_JD)
    M_Earth = (357.5277233 + 35999.05034*TT_cent)*pi/180.
    
    # Add in relativity correction
    JED_JD = TT_JD + (0.001658*sin(M_Earth) + 1.385e-5*sin(2*M_Earth))/86400.
    
    return JED_JD


def jedjd2ttjd(JED_JD):
    '''
    This function converts Julian Ephemeris Date (JED) to Terrestrial Time (TT) 
    in Julian Date (JD) format.
    
    Parameters
    ------
    JED_JD : float
        fractional days since 12:00:00 Jan 1 4713 BC JED     
    
    Returns
    ------
    TT_JD : float
        fractional days since 12:00:00 Jan 1 4713 BC TT  
        
    '''   
    
    # Find approximate mean anomaly of Earth
    TT_cent = jd2cent(JED_JD)
    M_Earth = (357.5277233 + 35999.05034*TT_cent)*pi/180.
    
    # Subtract relativity correction
    TT_JD = JED_JD - (0.001658*sin(M_Earth) + 1.385e-5*sin(2*M_Earth))/86400.
    
    return TT_JD


###############################################################################
# Unit Test
###############################################################################

if __name__ == '__main__':
    
    from eop_functions import get_celestrak_eop_alldata
    from eop_functions import get_eop_data

    
    UTC = datetime(2018, 1, 27, 0, 0, 0)
    
    eop_alldata = get_celestrak_eop_alldata()
    EOP_data = get_eop_data(eop_alldata, UTC)
    
    print(EOP_data)
    
    MJD = dt2mjd(UTC)
    JD = dt2jd(UTC)
    UT1_JD = utcdt2ut1jd(UTC, EOP_data['UT1_UTC'])
    TT_JD = utcdt2ttjd(UTC, EOP_data['TAI_UTC'])
    JED_JD = utcdt2jedjd(UTC, EOP_data['TAI_UTC'])
    TT_check = jedjd2ttjd(JED_JD)
    TAI_JD = utcdt2taijd(UTC, EOP_data['TAI_UTC'])
    GPS_JD = utcdt2gpsjd(UTC, EOP_data['TAI_UTC'])
    
    print('UTC', UTC)
    print('MJD', MJD)
    print('JD', JD)
    print('UT1_JD', UT1_JD)
    print('TT_JD', TT_JD)
    print('TT_check', TT_check)
    print('JED_JD', JED_JD)
    print('TAI_JD', TAI_JD)
    print('GPS_JD', GPS_JD)
    





