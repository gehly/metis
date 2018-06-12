from datetime import datetime, timedelta

def utcdt2utcmjd(UTC):
    '''
    This function converts a UTC time to Modified Julian Date (MJD) fractional
    days since 1858-11-17 00:00:00 UTC.  No conversion between time systems is
    performed.
    
    Parameters
    ------
    UTC : datetime object
        time in UTC
    
    Returns
    ------
    MJD : float
        fractional days since 1858-11-17 00:00:00 UTC
    '''
    
    MJD_datetime = datetime(1858, 11, 17, 0, 0, 0)
    delta = UTC - MJD_datetime
    MJD = delta/timedelta(days=1)
    
    return MJD


def utcdt2utcjd(UTC):
    '''
    This function converts a UTC time to Julian Date (JD) fractional days since
    12:00:00 Jan 1 4713 BC UTC.  No conversion between time systems is 
    performed.
    
    Parameters
    ------
    UTC : datetime object
        time in UTC
    
    Returns
    ------
    JD : float
        fractional days since 12:00:00 Jan 1 4713 BC UTC
    
    '''
    
    MJD = utcdt2utcmjd(UTC)
    JD = MJD + 2400000.5    
    
    return JD


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
    
    UTC_JD = utcdt2utcjd(UTC)
    TT_JD = UTC_JD + (TAI_UTC + 32.184)/86400.
    
    return TT_JD






