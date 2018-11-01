import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from datetime import datetime, timedelta
from math import pi

from mpl_toolkits.basemap import Basemap

sys.path.append('../')

from sensors.sensors import define_sensors


def multiple_model_plot_measurements(measdir):
    
    plt.close('all')
    
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]
    
    
    # Load truth data file
    truth_file = measdir / 'leo_sphere_lamr_big_2018_07_12_truth.pkl'
    pklFile = open(truth_file, 'rb')
    data = pickle.load(pklFile)
    truth_time = data[0]
    state = data[1]
    visibility = data[2]
    pklFile.close()
    
    
    t0 = truth_time[0]
    
    legend = ['Sphere LAMRB', 'Sphere MAMRB', 'Sphere LAMRS', 'Sphere MAMRS',
              'Cube Nadir', 'Cube Spin', 'Cube Tumble', 'BW Nadir', 'BW Spin',
              'BW Tumble']
    
    plt.figure()
    
    # Load measurement data files
    meas_file = measdir / 'leo_sphere_lamr_big_2018_07_12_meas.pkl'
    pklFile = open(meas_file, 'rb')
    data = pickle.load(pklFile)
    meas_times = data[0]
    meas = data[1]
    meas_true = data[2]
    pklFile.close()    
    
#    t_sec = [(ti - t0).total_seconds() for ti in meas_times] 
    t_hrs = [(ti - t0).total_seconds()/3600. for ti in meas_times]    
    
    plt.subplot(3,1,1)
    plt.xlim([0., 40.])
    plt.ylim([0., 20.])
    plt.plot(t_hrs, meas_true[:,2], 'ko')
    plt.ylabel('Apparent Mag')
    plt.subplot(3,1,2)
    plt.xlim([0., 40.])
    plt.ylim([-200, 200.])
    plt.plot(t_hrs, meas_true[:,0]*180/pi, 'ko')
    plt.ylabel('RA [deg]')
    plt.subplot(3,1,3)
    plt.plot(t_hrs, meas_true[:,1]*180/pi, 'ko')
    plt.ylabel('DEC [deg]')
    plt.xlabel('Time [hours]')
    plt.xlim([0., 40.])
    plt.ylim([-90, 90.])
    
    
    t_hrs = []
    mag = []
    for ti in meas_times:
        if ti > (t0+timedelta(days=23./24)) and ti < (t0+timedelta(days=25./24)):
            t_hrs.append((ti-t0).total_seconds()/3600.)
            ind = meas_times.index(ti)
            mag.append(meas_true[ind,2])
        
    
    
    plt.figure()
    
    
    plt.plot(t_hrs, mag, color=colors[0], marker='o', linestyle=None)
    
    
    # Load measurement data files
    meas_file = measdir / 'leo_sphere_mamr_big_2018_07_12_meas.pkl'
    pklFile = open(meas_file, 'rb')
    data = pickle.load(pklFile)
    meas_times = data[0]
    meas = data[1]
    meas_true = data[2]
    pklFile.close()    
    
    t_hrs = []
    mag = []
    for ti in meas_times:
        if ti > (t0+timedelta(days=23./24)) and ti < (t0+timedelta(days=25./24)):
            t_hrs.append((ti-t0).total_seconds()/3600.)
            ind = meas_times.index(ti)
            mag.append(meas_true[ind,2])

    plt.plot(t_hrs, mag, color=colors[1], marker='o', linestyle=None)
    
    
    # Load measurement data files
    meas_file = measdir / 'leo_sphere_lamr_small_2018_07_12_meas.pkl'
    pklFile = open(meas_file, 'rb')
    data = pickle.load(pklFile)
    meas_times = data[0]
    meas = data[1]
    meas_true = data[2]
    pklFile.close()    
    
    t_hrs = []
    mag = []
    for ti in meas_times:
        if ti > (t0+timedelta(days=23./24)) and ti < (t0+timedelta(days=25./24)):
            t_hrs.append((ti-t0).total_seconds()/3600.)
            ind = meas_times.index(ti)
            mag.append(meas_true[ind,2])
    
    plt.plot(t_hrs, mag, color=colors[2], marker='o', linestyle=None)
    
    # Load measurement data files
    meas_file = measdir / 'leo_sphere_mamr_small_2018_07_12_meas.pkl'
    pklFile = open(meas_file, 'rb')
    data = pickle.load(pklFile)
    meas_times = data[0]
    meas = data[1]
    meas_true = data[2]
    pklFile.close()    
    
    t_hrs = []
    mag = []
    for ti in meas_times:
        if ti > (t0+timedelta(days=23./24)) and ti < (t0+timedelta(days=25./24)):
            t_hrs.append((ti-t0).total_seconds()/3600.)
            ind = meas_times.index(ti)
            mag.append(meas_true[ind,2])   
    
    plt.plot(t_hrs, mag, color=colors[3], marker='o', linestyle=None)
    
    # Load measurement data files
    meas_file = measdir / 'leo_cubesat_nadir_2018_07_12_meas.pkl'
    pklFile = open(meas_file, 'rb')
    data = pickle.load(pklFile)
    meas_times = data[0]
    meas = data[1]
    meas_true = data[2]
    pklFile.close()    
    
    t_hrs = []
    mag = []
    for ti in meas_times:
        if ti > (t0+timedelta(days=23./24)) and ti < (t0+timedelta(days=25./24)):
            t_hrs.append((ti-t0).total_seconds()/3600.)
            ind = meas_times.index(ti)
            mag.append(meas_true[ind,2]) 
    
    plt.plot(t_hrs, mag, color=colors[4], marker='d', linestyle=None)
    
    
    # Load measurement data files
    meas_file = measdir / 'leo_cubesat_spin_2018_07_12_meas.pkl'
    pklFile = open(meas_file, 'rb')
    data = pickle.load(pklFile)
    meas_times = data[0]
    meas = data[1]
    meas_true = data[2]
    pklFile.close()    
    
    t_hrs = []
    mag = []
    for ti in meas_times:
        if ti > (t0+timedelta(days=23./24)) and ti < (t0+timedelta(days=25./24)):
            t_hrs.append((ti-t0).total_seconds()/3600.)
            ind = meas_times.index(ti)
            mag.append(meas_true[ind,2])  
    
    plt.plot(t_hrs, mag, color='g', marker='d', linestyle='dashed')
    
    # Load measurement data files
    meas_file = measdir / 'leo_cubesat_tumble_2018_07_12_meas.pkl'
    pklFile = open(meas_file, 'rb')
    data = pickle.load(pklFile)
    meas_times = data[0]
    meas = data[1]
    meas_true = data[2]
    pklFile.close()    
    
    t_hrs = []
    mag = []
    for ti in meas_times:
        if ti > (t0+timedelta(days=23./24)) and ti < (t0+timedelta(days=25./24)):
            t_hrs.append((ti-t0).total_seconds()/3600.)
            ind = meas_times.index(ti)
            mag.append(meas_true[ind,2])  
    
    plt.plot(t_hrs, mag, color='b', marker='d', linestyle='dashed')
    
    # Load measurement data files
    meas_file = measdir / 'leo_boxwing_nadir_2018_07_12_meas.pkl'
    pklFile = open(meas_file, 'rb')
    data = pickle.load(pklFile)
    meas_times = data[0]
    meas = data[1]
    meas_true = data[2]
    pklFile.close()    
    
    t_hrs = []
    mag = []
    for ti in meas_times:
        if ti > (t0+timedelta(days=23./24)) and ti < (t0+timedelta(days=25./24)):
            t_hrs.append((ti-t0).total_seconds()/3600.)
            ind = meas_times.index(ti)
            mag.append(meas_true[ind,2]) 
    
    plt.plot(t_hrs, mag, color=colors[7], marker='s', linestyle=None)
    
    
    # Load measurement data files
    meas_file = measdir / 'leo_boxwing_spin_2018_07_12_meas.pkl'
    pklFile = open(meas_file, 'rb')
    data = pickle.load(pklFile)
    meas_times = data[0]
    meas = data[1]
    meas_true = data[2]
    pklFile.close()    
    
    t_hrs = []
    mag = []
    for ti in meas_times:
        if ti > (t0+timedelta(days=23./24)) and ti < (t0+timedelta(days=25./24)):
            t_hrs.append((ti-t0).total_seconds()/3600.)
            ind = meas_times.index(ti)
            mag.append(meas_true[ind,2]) 
    
    plt.plot(t_hrs, mag, color=colors[8], marker='s', linestyle='dashed')
    
    # Load measurement data files
    meas_file = measdir / 'leo_boxwing_tumble_2018_07_12_meas.pkl'
    pklFile = open(meas_file, 'rb')
    data = pickle.load(pklFile)
    meas_times = data[0]
    meas = data[1]
    meas_true = data[2]
    pklFile.close()    
    
    t_hrs = []
    mag = []
    for ti in meas_times:
        if ti > (t0+timedelta(days=23./24)) and ti < (t0+timedelta(days=25./24)):
            t_hrs.append((ti-t0).total_seconds()/3600.)
            ind = meas_times.index(ti)
            mag.append(meas_true[ind,2])  
    
    plt.plot(t_hrs, mag, color='r', marker='s', linestyle='dashed')
    
    
    plt.xlim([23.3, 23.7])
    plt.ylim([5., 18.])
    
    plt.ylabel('Apparent Magnitude')
    plt.xlabel('Time [hours]')
    plt.legend(legend)
    
    
    
    plt.show()
    
    
    return


def plot_sensor_map(sensor_id_list):
    
    sensor_dict = define_sensors(sensor_id_list)
    
    plt.figure()
    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')
    m.drawcoastlines()
    
    for sensor_id in sensor_dict:
        latlonht = sensor_dict[sensor_id]['geodetic_latlonht']
        
        plt.plot(latlonht[1], latlonht[0], 'bo', ms=8)
    
    
    
    return


if __name__ == '__main__':
    
    # Data directory
    measdir = Path('C:/Users/Steve/Documents/data/multiple_model/'
                   '2018_07_12_leo/measurements')
    
    
    
    multiple_model_plot_measurements(measdir)
    
    
    
    
    
    
    
    
    
    
    
    
    
    