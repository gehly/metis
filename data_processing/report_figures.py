import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import pickle
from datetime import datetime, timedelta
from math import pi
import sys
import scipy.stats as ss

# from mpl_toolkits.basemap import Basemap

# sys.path.append('../')

# from sensors.sensors import define_sensors


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


def plot_gaussian():
    
    x = np.arange(0, 10, 0.01)
    y = ss.norm(5,1).pdf(x)
    
    plt.figure()
    plt.plot(x, y, 'b-')
    plt.plot([-1, 11], [0,0], 'k-')
    plt.plot([0,0], [-0.1, 0.5], 'k-')
    plt.plot([5,5], [-0.02, 0.02], 'k')
    plt.plot([2,2], [-0.02, 0.02], 'k')
    plt.plot([6,6], [-0.02, 0.02], 'k')
    plt.axis('off')
    
    plt.show()
    
    
    return



def plot_multitarget_gaussian():
    
    
    x, y = np.mgrid[-4:4:0.1, -4:4:0.1]
    pos = np.dstack((x, y))

    # m_list = [np.array([0., 0.]), 
    #           np.array([3., -4.]),
    #           np.array([-6, 6.]),
    #           np.array([-7., -7.])]
    
    # P_list = [np.diag([1., 2.]),
    #           np.diag([4., 4.]),
    #           np.diag([4., 1.]),
    #           np.diag([1., 0.4])]
    
    # w_list = [0.5, 0.6, 0.4, 0.1]
    
    c_list = ['b', 'b', 'r', 'r']
    
    w_list = [1.]
    m_list = [np.array([0., 0.])]
    P_list = [np.diag([1., 1.])]
    
    
    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')
    
    for ii in range(len(w_list)):
        c = c_list[ii]
        w = w_list[ii]
        m = m_list[ii]
        P = P_list[ii]
        z = ss.multivariate_normal(m, P)
    
        ax1.plot_surface(x, y, w*z.pdf(pos), cmap=cm.jet, antialiased=False)
        # ax1.plot_wireframe(x, y, w*z.pdf(pos), color='k')
    
    ax1.set_axis_off()
    
    plt.savefig('demo.png', transparent=True)
    
    plt.show()
    
    
    
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    
    # # Make data
    # u = np.linspace(0, 2 * np.pi, 100)
    # v = np.linspace(0, np.pi, 100)
    # x = 10 * np.outer(np.cos(u), np.sin(v))
    # y = 10 * np.outer(np.sin(u), np.sin(v))
    # z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # # Plot the surface
    # ax.plot_surface(x, y, z)
    
    # # Set an equal aspect ratio
    # ax.set_aspect('equal')
    
    # plt.show()
    
    return


if __name__ == '__main__':
    
    
    plt.close('all')
    
    # # Data directory
    # measdir = Path('C:/Users/Steve/Documents/data/multiple_model/'
    #                '2018_07_12_leo/measurements')
    
    
    
    # multiple_model_plot_measurements(measdir)
    
    plot_gaussian()
    
    # plot_multitarget_gaussian()
    
    
    
    
    
    
    
    
    
    
    
    