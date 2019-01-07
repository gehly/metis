import numpy as np
from math import pi, asin, atan2
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.append('../')

from utilities.tle_functions import get_spacetrack_tle_data
from utilities.tle_functions import propagate_TLE
from utilities.coordinate_systems import itrf2gcrf
from utilities.coordinate_systems import latlonht2ecef
from utilities.eop_functions import get_celestrak_eop_alldata
from utilities.eop_functions import get_eop_data
from sensors.sensors import define_sensors


def read_csv_angles_meas(fname):
    
    df = pd.read_csv(fname)
    
#    print(df)
    
    t_list = df['time'].tolist()
    Rab = df['Rab'].tolist()
    Decb = df['Decb'].tolist()
    Rae = df['Rae'].tolist()
    Dece = df['Dece'].tolist()
    dt_list = []
    
    
    
#    print(t_list)
    
    for ti in t_list:
        
        start = ti.find("'2")
#        print(start)
#        stop = ti.find("[0-9]'")
#        print(stop)
#        print(ti)
#        print(str(ti[start+1:start+24]))
#        print(iso2dt(ti[start+1:start+24]))
        
        dt_list.append(iso2dt(ti[start+1:start+24]))
        
    for ii in range(len(dt_list)):
        dt_list.append(dt_list[ii] + timedelta(seconds=5.))


    ra_list = []
    for ra in Rab:
        ra_list.append(hms2deg(ra))
        
    for ra in Rae:
        ra_list.append(hms2deg(ra))
        
    dec_list = []
    for dec in Decb:        
        dec = dec.strip()
        dec_list.append(damas2deg(dec))
        
    for dec in Dece:
        dec = dec.strip()
        dec_list.append(damas2deg(dec))
        
        
    sort_inds = [i[0] for i in sorted(enumerate(dt_list), key=lambda x:x[1])]
    
    dt_list = [dt_list[ii] for ii in sort_inds]
    ra_list = [ra_list[ii] for ii in sort_inds]
    dec_list = [dec_list[ii] for ii in sort_inds]
    
    return dt_list, ra_list, dec_list


def iso2dt(iso):
    
    dt, _, micro = iso.partition('.')
    dt = datetime.strptime(dt, '%Y-%m-%dT%H:%M:%S')
    micro = int(micro.rstrip('Z'), 10) * 1000
    
    return dt + timedelta(microseconds = micro)


def hms2deg(hms):
    
    deg = float(hms[0:2])*15. + float(hms[3:5])*(15./60.) + float(hms[6:])*(15./3600.)
    
    return deg


def damas2deg(damas):
    
    deg = float(damas[0:2]) + float(damas[3:5])*(1./60.) + float(damas[6:])*(1./3600.)
    
    return deg



if __name__ == '__main__':
    
    plt.close('all')
    
    obj_id_list = [37158]
    sensor_id_list = ['RMIT ROO']
    get_tle_UTC = [datetime(2018, 10, 29, 0, 0, 0)]
    tle_dict, tle_df = get_spacetrack_tle_data(obj_id_list, get_tle_UTC)
    print(tle_dict)
    
    fdir = Path('D:/documents/research/sensor_management/reports/2018_asrc_ROO/data')
    fname = fdir / 'QZSS_OD_angles.csv'
    
    UTC_list, ra_list, dec_list = read_csv_angles_meas(fname)
    
    

    output_state = propagate_TLE(obj_id_list, UTC_list, tle_dict, offline_flag=True)
    
    sensor_dict = define_sensors(sensor_id_list)
    latlonht = sensor_dict[sensor_id_list[0]]['geodetic_latlonht']
    lat = latlonht[0]
    lon = latlonht[1]
    ht = latlonht[2]
    stat_ecef = latlonht2ecef(lat, lon, ht)
    eop_alldata = get_celestrak_eop_alldata(offline_flag=True)
    
    ra_geo = []
    dec_geo = []
    ra_topo = []
    dec_topo = []    
    for ii in range(len(UTC_list)):
        
        # Retrieve object position vector
        UTC = UTC_list[ii]
        r_GCRF = output_state[obj_id_list[0]]['r_GCRF'][ii]
        x = float(r_GCRF[0])
        y = float(r_GCRF[1])
        z = float(r_GCRF[2])
        r = np.linalg.norm(r_GCRF)
        
        # Retrieve sensor location in ECI        
        EOP_data = get_eop_data(eop_alldata, UTC)
        stat_eci, dum = itrf2gcrf(stat_ecef, np.zeros((3,1)), UTC, EOP_data)
        xs = float(stat_eci[0])
        ys = float(stat_eci[1])
        zs = float(stat_eci[2])
        rho = np.linalg.norm(r_GCRF - stat_eci)
        
        print(UTC)
        print(stat_eci)
        
        
        
        # Compute measurements
        ra_geo.append(atan2(y,x) * 180./pi)
        dec_geo.append(asin(z/r) * 180./pi)
        ra_topo.append(atan2((y-ys), (x-xs)) * 180./pi)
        dec_topo.append(asin((z-zs)/rho) * 180./pi)
        

    mistake
        
    print(ra_list)
    print(ra_geo)
    print(ra_topo)
    
    print(dec_list)
    print(dec_geo)
    print(dec_topo)
    
    for ii in range(len(ra_geo)):
        if ra_geo[ii] < 0.:
            ra_geo[ii] += 360.
        if ra_topo[ii] < 0.:
            ra_topo[ii] += 360.
        
        
    ra_geo_resids = [(ra_list[ii] - ra_geo[ii])*3600. for ii in range(len(ra_list))]
    dec_geo_resids = [(dec_list[ii] - dec_geo[ii])*3600. for ii in range(len(dec_list))]
    ra_topo_resids = [(ra_list[ii] - ra_topo[ii])*3600. for ii in range(len(ra_list))]
    dec_topo_resids = [(dec_list[ii] - dec_topo[ii])*3600. for ii in range(len(dec_list))]
    
    
    t_sec = [(UTC_list[ii] - UTC_list[0]).total_seconds() for ii in range(len(UTC_list))]
    
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t_sec, ra_geo_resids, 'k.')
    plt.ylabel('Geo RA [arcsec]')
    plt.subplot(2,1,2)
    plt.plot(t_sec, dec_geo_resids, 'k.')
    plt.ylabel('Geo DEC [arcsec]')
    plt.xlabel('Time [sec]')
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t_sec, ra_topo_resids, 'k.')
    plt.ylabel('Topo RA [arcsec]')
    plt.subplot(2,1,2)
    plt.plot(t_sec, dec_topo_resids, 'k.')
    plt.ylabel('Topo DEC [arcsec]')
    plt.xlabel('Time [sec]')
    
    
    
#    for obj_id in obj_id_list:
#        r_GCRF = output_state[obj_id]['r_GCRF'][0]
#        v_GCRF = output_state[obj_id]['v_GCRF'][0]
#        x_in = np.concatenate((r_GCRF, v_GCRF), axis=0)
#        print(obj_id)
#        print(x_in)
#        elem = element_conversion(x_in, 1, 0)
#        print(elem)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    