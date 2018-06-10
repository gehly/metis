import numpy as np
from math import pi
import requests
import getpass
from datetime import datetime, timedelta
import time

from conversions import utcdt2utcmjd, utcdt2ttjd

from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs84


def get_spacetrack_tle_data(obj_id_list, username='', password=''):
    '''
    This function retrieves the latest two-line element (TLE) data for objects
    in the input list from space-track.org.
    
    Parameters
    ------
    obj_id_list : list
        object NORAD IDs (int)
    username : string, optional
        space-track.org username (code will prompt for input if not supplied)
    password : string, optional
        space-track.org password (code will prompt for input in not supplied)
    
    Returns
    ------
    r : str
        string returned from requests containing TLE data
    '''
    
    if len(username) == 0:    
        username = input('space-track username: ')
    if len(password) == 0:
        password = getpass.getpass('space-track password: ')    
    
    myString = ",".join(map(str, obj_id_list))
    
    pageData = ('//www.space-track.org/basicspacedata/query/class/tle_latest/'
                'ORDINAL/1/NORAD_CAT_ID/' + myString + '/orderby/'
                'TLE_LINE1 ASC/format/tle')
    payload = {'identity':username, 'password':password, 'submit':'Login'}
    
    with requests.Session() as s:
        s.post("https://www.space-track.org/auth/login", data=payload)
        r = s.get('https:' + pageData)
        if r.status_code != requests.codes.ok:
            print("Error: Page data request failed.")
            
    # Parse response and form output
    tle_dict = {}
    nchar = 69
    nskip = 2
    ii = 0  
    for obj_id in obj_id_list:
        line1_start = ii*2*(nchar+nskip)
        line1_stop = ii*2*(nchar+nskip) + nchar
        line2_start = ii*2*(nchar+nskip) + nchar + nskip
        line2_stop = ii*2*(nchar+nskip) + 2*nchar + nskip
        line1 = r.text[line1_start:line1_stop]
        line2 = r.text[line2_start:line2_stop]

        tle_dict[obj_id] = {}
        tle_dict[obj_id]['line1'] = line1
        tle_dict[obj_id]['line2'] = line2
        
        ii += 1
    
    return tle_dict


def get_celestrak_eop_alldata():
    
    start = time.time()
    
    pageData = 'https://celestrak.com/SpaceData/eop19620101.txt'

    r = requests.get(pageData)
    if r.status_code != requests.codes.ok:
        print("Error: Page data request failed.")
            
    print(r.text[0:200])
    
    print('Time: ', time.time()-start)
    
    ind_NUM_OBSERVED_POINTS = r.text.find('NUM_OBSERVED_POINTS')
    ind_BEGIN_OBSERVED = r.text.find('BEGIN OBSERVED')
    ind_END_OBSERVED = r.text.find('END OBSERVED')
    ind_NUM_PREDICTED_POINTS = r.text.find('NUM_PREDICTED_POINTS')
    ind_BEGIN_PREDICTED = r.text.find('BEGIN PREDICTED')
    ind_END_PREDICTED = r.text.find('END PREDICTED')
    
    print(ind_NUM_OBSERVED_POINTS)
    print(ind_BEGIN_OBSERVED)
    print(ind_END_OBSERVED)
    print(ind_NUM_PREDICTED_POINTS)
    print(ind_BEGIN_PREDICTED)
    print(ind_END_PREDICTED)
    
    
    test = r.text[ind_NUM_OBSERVED_POINTS:ind_BEGIN_OBSERVED]
    print(test)
    print(len(test))
    
    # Reduce to data
    data_text = r.text[ind_BEGIN_OBSERVED+16:ind_END_OBSERVED] \
        + r.text[ind_BEGIN_PREDICTED+17:ind_END_PREDICTED]
    
    
    print(data_text[-19200:-18000])
    

    
    return data_text


def get_eop_data(data_text, UTC):
    
        
    # Compute MJD for desired time
    MJD = utcdt2utcmjd(UTC)
    MJD_int = int(MJD)
    
    print('\n\n Parse Lines')
    
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
        

    print(line0)
    print(line1)
    
    # Compute EOP data at desired time by interpolating
    EOP_data = eop_linear_interpolate(line0, line1, MJD)
    
    
    return EOP_data


def eop_linear_interpolate(line0, line1, MJD):
    
    
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
    
    print(MJD)
    print(dt)
    print(line0_array)
    print(line1_array)
    print(interp)
    
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
    
    print(EOP_data)

    
    
    
    
    return EOP_data


def eop_read_line(line):
    '''
    http://celestrak.com/SpaceData/EOP-format.asp
    
    012-016	Modified Julian Date (Julian Date at 0h UT minus 2400000.5)
    018-026	x (arc seconds)
    028-036	y (arc seconds)
    038-047	UT1-UTC (seconds)
    049-058	Length of Day (seconds)
    060-068	δΔψ (arc seconds)
    070-078	δΔε (arc seconds)
    080-088	δX (arc seconds)
    090-098	δY (arc seconds)
    100-102	Delta Atomic Time, TAI-UTC (seconds)
    '''
    
    MJD = int(line[11:16])
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

        


def propagate_TLE(obj_id_list, UTC_list, username='', password=''):
    
    
    
    # Retrieve latest TLE data from space-track.org
    tle_dict = get_spacetrack_tle_data(obj_id_list, username, password)
    
    # Retrieve latest EOP data from celestrak.com
    eop_alldata = get_celestrak_eop_alldata()
    
    # Loop over objects
    for obj_id in obj_id_list:
        
        line1 = tle_dict[obj_id]['line1']
        line2 = tle_dict[obj_id]['line2']
        
        # Loop over times
        for UTC in UTC_list:
            
            # Propagate TLE using SGP4
            satellite = twoline2rv(line1, line2, wgs84)
            tleTime = satellite.epoch
            r_TEME, v_TEME = satellite.propagate(UTC.year, UTC.month, UTC.day,
                                                 UTC.hour, UTC.minute,
                                                 UTC.second + UTC.microsecond)
            
            print(obj_id)
            print(tleTime)
            print(UTC)
            print(r_TEME)
            print(v_TEME)
            
            # Get EOP data for this time
            EOP_data = get_eop_data(eop_alldata, UTC)            
            
            # Compute TT in JD format
            TT_JD = utcdt2ttjd(UTC, EOP_data['TAI_UTC'])
            
            
            # Convert from TEME to GCRF (ECI)
                
            
            # 
            
            # Compute MAI Time (UTC seconds since _____ epoch)
            
            
            
    
    
    
    
    return


def teme2gcrf(r_TEME, v_TEME, UTC):
    
    
    return r_GCRF, v_GCRF



###############################################################################
# Stand-alone execution
###############################################################################

if __name__ == '__main__' :
    
    
    username = 'steve.gehly@gmail.com'
    password = 'SpaceTrackPword!'
    
    
    obj_id_list = [43014]
    UTC_list = [datetime(2018, 6, 10, 12, 0, 0)]
    
    
    propagate_TLE(obj_id_list, UTC_list, username, password)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
