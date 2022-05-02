import numpy as np
from math import pi, sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, asinh, acosh, atanh, fmod, fabs
from datetime import datetime, timedelta
import copy
import sys
import matplotlib.pyplot as plt
import pandas as pd

metis_dir = r'C:\Users\Steve\Documents\code\metis'
sys.path.append(metis_dir)

from utilities.tle_functions import get_spacetrack_tle_data, find_closest_tle_epoch
from utilities.astrodynamics import cart2kep, kep2cart, element_conversion, osc2mean
from utilities.astrodynamics import meanmot2sma, RAAN_to_LTAN, LTAN_to_RAAN, sunsynch_inclination
from utilities.constants import Re, GME, J2E, wE
from utilities.eop_functions import get_celestrak_eop_alldata, get_eop_data
from utilities.eop_functions import get_XYs2006_alldata, batch_eop_rotation_matrices, get_nutation_data
from utilities.coordinate_systems import ecef2latlonht, latlonht2ecef, ric2eci, gcrf2itrf, gcrf2teme, enu2ecef, itrf2gcrf
from utilities.coordinate_systems import inc2az, dist2latlon, latlon2dist
from utilities.time_systems import mjd2dt, dt2mjd
from utilities.tle_functions import propagate_TLE as prop_TLE_full


from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs84



###############################################################################
# Separation Distance Calculation
###############################################################################



def compute_separation_distance():
    
    
    # Sensor attributes
#    xpix_m = 6.45e-6
#    ypix_m = 6.45e-6
#    focal_length_m = 690e-3
#    xpix_rad = xpix_m/focal_length_m
#    ypix_rad = ypix_m/focal_length_m
    
    
    xpix_rad = 9.180822244391437e-06
    ypix_rad = 9.180822244391437e-06
    
    # Orbit parameters
    r = Re + 550.
    
    
    # Load and extract data
    data = cmu_falcon_data2()
    
    frame_list = data['frame_list']
    dt_vec = data['dt_vec']
    center_el_deg_list = data['center_el_deg_list']
    lead_xpix = data['lead_xpix']
    lead_ypix = data['lead_ypix']
    trail_xpix = data['trail_xpix']
    trail_ypix = data['trail_ypix']
    
    # Loop over frames
    alongtrack_list = []
    dV_list = []
    for ii in range(len(frame_list)):
        dt = dt_vec[ii]
        
        # Compute angular separation in Sensor Frame
        xrad = abs(lead_xpix[ii] - trail_xpix[ii])*xpix_rad
        yrad = abs(lead_ypix[ii] - trail_ypix[ii])*ypix_rad
        observer_angle_rad = np.sqrt(xrad**2. + yrad**2.)
        
        # Compute range and beta at this time
        el_rad = center_el_deg_list[ii] * pi/180.
        Recosel = Re*cos(el_rad+pi/2.)
        
        rho = Recosel + np.sqrt(Recosel**2. - (Re**2. - r**2.))
        beta_rad = acos((Re**2. - r**2. - rho**2.)/(-2.*r*rho))
                
        # Compute separation distance in sensor and orbit frame
        sensor_frame_dist = observer_angle_rad*rho
        alongtrack_dist_m = sensor_frame_dist/cos(beta_rad) * 1000.
        alongtrack_list.append(alongtrack_dist_m)
        
        # Compute initial separation delta-V
        dV = alongtrack_dist_m/(3.*dt)
        dV_list.append(dV)
        
    print(alongtrack_list)
    print(dV_list)
    
    alongtrack_mean = np.mean(alongtrack_list)
    alongtrack_std = np.std(alongtrack_list)
    dV_mean = np.mean(dV_list)
    dV_std = np.std(dV_list)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(frame_list, alongtrack_list, 'bo')
    plt.plot(frame_list, [alongtrack_mean]*len(frame_list), 'k')
    plt.plot(frame_list, [alongtrack_mean + alongtrack_std]*len(frame_list), 'k--')
    plt.plot(frame_list, [alongtrack_mean - alongtrack_std]*len(frame_list), 'k--')
    plt.xticks(frame_list)
    plt.ylabel('Along-Track [m]')
    plt.title('CMU Falcon Separation Estimates 2021-09-11 10:27 UTC')
    
    plt.subplot(2,1,2)
    plt.plot(frame_list, dV_list, 'bo')
    plt.plot(frame_list, [dV_mean]*len(frame_list), 'k')
    plt.plot(frame_list, [dV_mean + dV_std]*len(frame_list), 'k--')
    plt.plot(frame_list, [dV_mean - dV_std]*len(frame_list), 'k--')
    plt.xticks(frame_list)
    plt.xlabel('Frame Number')
    plt.ylabel('dV [m/s]')
    
    
    plt.show()
    
    print('\n\n')
    print('Along-Track Mean and STD [m]: ', alongtrack_mean, alongtrack_std)
    print('Initial delta-V Mean and STD [m/s]: ', dV_mean, dV_std)
    
    
    return


def demo_distance_calc():
    
    plt.close('all')
    
    # Setup ground station
    lat = 0.
    lon = 0.
    ht = 0.
    
    stat_ecef = latlonht2ecef(lat, lon, ht)
    
    # Initial orbit elements
    a = Re + 550.
    elem0 = [a, 0.0001, 0., 0., 0., -20.]
    deputy_alongtrack = 1.    # km
    
    
    # Propagate for 30 minutes
    ti_vect = np.arange(0, 1000., 30.)   
    ti_plot = []
    el_plot = []
    rg_plot = []
    beta_plot = []
    fpdist_plot = []
    dist_plot = []
    error_plot = []
    for ii in range(len(ti_vect)):
        
        # Propagate chief orbit to current time
        ti = ti_vect[ii]
        chief_cart = element_conversion(elem0, 0, 1, GME, ti)
        rc_vect = chief_cart[0:3].reshape(3,1)
        vc_vect = chief_cart[3:6].reshape(3,1)
        rc = np.linalg.norm(rc_vect)
        
        # Compute deputy position in ECI
        rho_ric = np.array([[0.], [deputy_alongtrack], [0.]])
        rho_eci = ric2eci(rc_vect, vc_vect, rho_ric)
        deputy_eci = rc_vect + rho_eci
        
        # Compute unit pointing vector of deputy and chief
        # Assume non-rotating earth so ECI = ECEF
        chief_rho_ecef = rc_vect - stat_ecef
        deputy_rho_ecef = deputy_eci - stat_ecef
        
        # Compute az/el/range for both
        chief_rho_enu = ecef2enu(chief_rho_ecef, stat_ecef)
        deputy_rho_enu = ecef2enu(deputy_rho_ecef, stat_ecef)
        
        
        chief_rg = np.linalg.norm(chief_rho_enu)
        chief_az = atan2(chief_rho_enu[0], chief_rho_enu[1])
        chief_el = asin(chief_rho_enu[2]/chief_rg)
        
        deputy_rg = np.linalg.norm(deputy_rho_enu)
        deputy_az = atan2(deputy_rho_enu[0], deputy_rho_enu[1])
        deputy_el = asin(deputy_rho_enu[2]/deputy_rg)
        
        # Compute observer angle
        chief_rho_hat = chief_rho_enu/chief_rg
        deputy_rho_hat = deputy_rho_enu/deputy_rg
        observer_angle_diff = acos(np.dot(chief_rho_hat.T, deputy_rho_hat))
        
#        print(np.dot(chief_rho_hat.T, deputy_rho_hat))
#        
#        print(rho_ric)
#        print(rho_eci)
#        print(rc_vect)
#        print(deputy_eci)
#        print(chief_rg, chief_az*180/pi, chief_el*180/pi)
#        print(deputy_rg, deputy_az*180/pi, deputy_el*180/pi)
#        print(observer_angle_diff)
#        
#        mistake
        
        if chief_el < 5.*pi/180.:
            continue
        
        print(ti)
        print(chief_el*180/pi)
        
        # Compute beta angle
        beta = acos((Re**2. - rc**2. - chief_rg**2)/(-2.*rc*chief_rg))
        
#        print( (Re**2. - a**2. - chief_rg**2)/(-2.*a*chief_rg) )
        
        # Compute distances
        focalplane_dist = chief_rg*observer_angle_diff
        computed_dist = focalplane_dist/cos(beta)
        
        print(chief_el*180/pi)
        print(beta)
        print(focalplane_dist)
        print(computed_dist)
        
        # Store data for plot
        ti_plot.append(ti)
        el_plot.append(chief_el*180/pi)
        rg_plot.append(chief_rg)
        beta_plot.append(beta*180/pi)
        fpdist_plot.append(focalplane_dist*1000.)
        dist_plot.append(computed_dist*1000.)
        error_plot.append( (computed_dist-deputy_alongtrack) * 1000.)
        
    
    # Generate plots
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(ti_plot, rg_plot, 'k.')
    plt.ylabel('Range [km]')
    plt.subplot(3,1,2)
    plt.plot(ti_plot, el_plot, 'k.')
    plt.ylabel('El [deg]')
    plt.subplot(3,1,3)
    plt.plot(ti_plot, beta_plot, 'k.')
    plt.ylabel('Beta [deg]')
    plt.xlabel('Time [sec]')
    
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(ti_plot, fpdist_plot, 'k.')
    plt.ylabel('FP Dist [m]')
    plt.subplot(3,1,2)
    plt.plot(ti_plot, dist_plot, 'k.')
    plt.ylabel('Along-Track [m]')
    plt.ylim([900, 1100])
    plt.subplot(3,1,3)
    plt.plot(ti_plot, error_plot, 'k.')
    plt.ylabel('A-T Error [m]')
    plt.xlabel('Time [sec]')
    

    
    
    plt.show()
    
    return


###############################################################################
# Measurement Data
###############################################################################


def sensor_data():
    
    # Sensor location - from TheSkyX for CMU Falcon
    lat = 38. + 57./60. + 48.12/3600.       # deg
    lon = -(108. + 14./60. + 15.84/3600.)   # deg
    ht = 1.86116    # km
    
#    # From Chun document
#    lon = 251.76
#    lat = 39.96
#    ht = 1.380
    
    
    sensor_ecef = latlonht2ecef(lat, lon, ht)

    return sensor_ecef


def cmu_falcon_data():
    
    
    frame_list = [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 
                  54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
    
    t0 = '2021-09-10T04:55:00.000'
    t0_dt = datetime.strptime(t0, '%Y-%m-%dT%H:%M:%S.%f')
    
    time_list = ['2021-09-10T10:30:06.160', '2021-09-10T10:30:11.490',
                 '2021-09-10T10:30:16.812', '2021-09-10T10:30:22.178',
                 '2021-09-10T10:30:27.492', '2021-09-10T10:30:32.805',
                 '2021-09-10T10:30:38.130', '2021-09-10T10:30:43.478',
                 '2021-09-10T10:30:48.804', '2021-09-10T10:30:54.119',
                 '2021-09-10T10:30:59.435', '2021-09-10T10:31:04.745',
                 '2021-09-10T10:31:10.048', '2021-09-10T10:31:15.362',
                 '2021-09-10T10:31:20.727', '2021-09-10T10:31:26.070',
                 '2021-09-10T10:31:31.393', '2021-09-10T10:31:36.735',
                 '2021-09-10T10:31:42.057', '2021-09-10T10:31:47.377',
                 '2021-09-10T10:31:52.697', '2021-09-10T10:31:58.005',
                 '2021-09-10T10:32:03.351', '2021-09-10T10:32:08.653',
                 '2021-09-10T10:32:13.973', '2021-09-10T10:32:19.335',
                 '2021-09-10T10:32:24.707', '2021-09-10T10:32:30.028']
    
    center_az_deg_list = [87.04141089, 88.63960805, 90.06643832, 91.34687358,
                          92.48599834, 93.52078875, 94.45512814, 95.30465751,
                          96.08839456, 96.80372237, 97.46327446, 98.06804191,
                          98.63130861, 99.15498081, 99.64552512, 100.1032925,
                          100.5279028, 100.9319102, 101.3111703, 
                          101.666846737521, 102.003055740061, 102.322364210451,
                          102.623183735174, 102.90777676212, 103.177919660113,
                          103.438378041991, 103.685090009724, 103.922803623345]
    
    center_el_deg_list = [43.58582747, 41.6855151,  39.85831155, 38.10789431,
                          36.45314825, 34.8791562,  33.38001593, 31.95174273,
                          30.59138964, 29.30180592, 28.07026441, 26.90681166,
                          25.78714707, 24.72563935, 23.69652596, 22.72295432,
                          21.78581985, 20.88836471, 20.03030062, 
                          19.2098835042126, 18.4175735424508, 17.6515778454327,
                          16.915516684556, 16.2114320090356, 15.5264171909172,
                          14.8610265511143, 14.2138043772949, 13.5924667976961]
    
    center_ra_list = ['06 13 14.625', '06 18 08.354', '06 22 39.816',
                      '06 26 52.031', '06 30 43.478', '06 34 18.976',
                      '06 37 40.386', '06 40 48.988', '06 43 47.006',
                      '06 46 34.006', '06 49 12.172', '06 51 40.777',
                      '06 54 03.025', '06 56 17.819', '06 58 27.890',
                      '07 00 31.335', '07 02 29.591', '07 04 23.593',
                      '07 06 12.873', '07 07 57.654', '07 09 39.019',
                      '07 11 17.379', '07 12 52.292', '07 14 23.590',
                      '07 15 52.637', '07 17 19.754', '07 18 44.677',
                      '07 20 07.107']
    
    center_dec_list = ['+27 33 42.28', '+25 36 06.99', '+23 44 01.80',
                       '+21 57 23.17', '+20 17 16.35', '+18 42 16.55',
                       '+17 12 27.47', '+15 47 22.93', '+14 26 26.12',
                       '+13 10 02.88', '+11 57 25.02', '+10 49 01.19',
                       '+09 43 31.19', '+08 41 28.15', '+07 41 42.18',
                       '+06 45 08.91', '+05 51 09.23', '+04 59 21.14',
                       '+04 09 57.01', '+03 22 51.90', '+02 37 34.70',
                       '+01 53 55.94', '+01 12 09.40', '+00 32 15.09',
                       '-00 06 21.00', '-00 43 49.35', '-01 20 02.12',
                       '-01 54 53.05']
    
    
    # Pixel location measurements (approx center of smear)
    # Note object smear is about 20 pixels across

    lead_xpix = [1350, 1339, 1328, 1323, 1313, 1308, 1298, 1294, 1292, 1287,
                 1280, 1271, 1268, 1266, 1264, 1259, 1262, 1258, 1256, 1256,
                 1257, 1269, 1262, 1263, 1266, 1270, 1273, 1282]
    
    lead_ypix = [295, 293, 293, 288, 280, 277, 270, 267, 264, 256, 253, 248,
                 246, 246, 245, 237, 236, 232, 223, 221, 206, 199, 192, 190,
                 188, 186, 185, 187]

    trail_xpix = [1240, 1238, 1232, 1233, 1226, 1228, 1221, 1221, 1223, 1220,
                  1217, 1212, 1210, 1211, 1211, 1208, 1214, 1212, 1211, 0, 0,
                  0, 0, 0, 0, 0, 0, 0]
    
    trail_ypix = [384, 379, 374, 364, 353, 347, 336, 330, 322, 311, 307, 300,
                  296, 292, 290, 279, 276, 272, 260, 0, 0, 0, 0, 0, 0, 0, 0, 0]



    
    print(len(frame_list))
    print(len(time_list))
    print(len(center_az_deg_list))
    print(len(center_el_deg_list))
    print(len(center_ra_list))
    print(len(center_dec_list))
    print(len(lead_xpix))
    print(len(lead_ypix))
    print(len(trail_xpix))
    print(len(trail_ypix))
            
    dt_vec = []
    center_ra_deg_list = []
    center_dec_deg_list = []
    for ii in range(len(frame_list)):
        
        ti = time_list[ii]
        ti_dt = datetime.strptime(ti, '%Y-%m-%dT%H:%M:%S.%f')
        dt_sec = (ti_dt - t0_dt).total_seconds()
        dt_vec.append(dt_sec)
        
        ra = center_ra_list[ii]
        dec = center_dec_list[ii]
        ra_deg = hms2deg(ra)
        dec_deg = degstr2deg(dec)
        
        center_ra_deg_list.append(ra_deg)
        center_dec_deg_list.append(dec_deg)
        
    print(dt_vec)
    print(center_ra_deg_list)
    print(center_dec_deg_list)
    
    data = {}
    data['frame_list'] = frame_list
    data['dt_vec'] = dt_vec
    data['center_az_deg_list'] = center_az_deg_list
    data['center_el_deg_list'] = center_el_deg_list
    data['center_ra_deg_list'] = center_ra_deg_list
    data['center_dec_deg_list'] = center_dec_deg_list
    data['lead_xpix'] = lead_xpix
    data['lead_ypix'] = lead_ypix
    data['trail_xpix'] = trail_xpix
    data['trail_ypix'] = trail_ypix
    
    

    
    return data


def cmu_falcon_data2():
    
    
    frame_list = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                  26, 27, 28]
    
    t0 = '2021-09-10T04:55:00.000'
    t0_dt = datetime.strptime(t0, '%Y-%m-%dT%H:%M:%S.%f')
    
    time_list = ['2021-09-11T10:25:39.709', '2021-09-11T10:25:47.024',
                 '2021-09-11T10:25:54.332', '2021-09-11T10:26:01.647',
                 '2021-09-11T10:26:08.990', '2021-09-11T10:26:16.339',
                 '2021-09-11T10:26:23.647', '2021-09-11T10:26:30.961',
                 '2021-09-11T10:26:38.276', '2021-09-11T10:26:45.603',
                 '2021-09-11T10:26:52.899', '2021-09-11T10:27:00.220',
                 '2021-09-11T10:27:07.522', '2021-09-11T10:27:14.833',
                 '2021-09-11T10:27:22.158', '2021-09-11T10:27:29.460',
                 '2021-09-11T10:27:36.771', '2021-09-11T10:27:44.076']
        
    center_az_deg_list = [111.758086776514, 111.965624975093, 112.172505123979,
                          112.360237259098, 112.537835494133, 112.696649362228,
                          112.820451472316, 112.977266451537, 113.079079400546,
                          113.202201611083, 113.319518357399, 113.40095155707,
                          113.5229184, 113.6037592, 113.6928918, 113.7877283, 
                          113.8518337, 113.937073263596]
    
    center_el_deg_list = [27.9860206658238, 26.329233551579, 24.8066310006462,
                          23.3739782725777, 22.0416937106087, 20.7774004894584,
                          19.5878122667381, 18.4661172088583, 17.3885125781707,
                          16.3857719431395, 15.4388421956785, 14.50338729439,
                          13.66015568, 12.82901599, 12.02026761, 11.26936596, 
                          10.51510151, 9.82372231697601]
    
    center_ra_list = ['06 14 53.503', '06 19 12.980', '06 23 14.669', 
                      '06 27 02.269', '06 30 35.537', '06 33 57.809',
                      '06 37 05.741', '06 40 09.677', '06 43 00.644',
                      '06 45 44.897', '06 48 21.186', '06 50 51.578',
                      '06 53 15.372', '06 55 32.058', '06 57 47.623',
                      '06 59 56.256', '07 02 01.593', '07 04 01.258']
    
    center_dec_list = ['+02 19 44.77', '+01 03 05.49', '-00 08 11.95', 
                       '-01 15 10.39', '-02 17 41.66', '-03 16 43.32',
                       '-04 11 16.74', '-05 04 22.88', '-05 53 25.16',
                       '-06 40 13.07', '-07 24 27.25', '-08 06 42.36',
                       '-08 46 48.36', '-09 24 39.57', '-10 01 54.43',
                       '-10 36 58.66', '-11 10 50.96', '-11 42 59.77']

    lead_xpix = [904, 925, 942, 958, 974, 990, 1003, 1018, 1028, 1039, 1050, 
                 1060, 1073, 1085, 1095, 1108, 1122, 0]
    
    lead_ypix = [371, 358, 344, 329, 308, 298, 289, 278, 268, 259, 252, 244,
                 236, 229, 220, 211, 207, 0]

    trail_xpix = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 896, 916, 935, 953, 971, 987]
    
    trail_ypix = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 328, 315, 303, 291, 280, 271]
    
    print(len(frame_list))
    print(len(time_list))
    print(len(center_az_deg_list))
    print(len(center_el_deg_list))
    print(len(center_ra_list))
    print(len(center_dec_list))
    print(len(lead_xpix))
    print(len(lead_ypix))
    print(len(trail_xpix))
    print(len(trail_ypix))
   
            
    UTC_list = []
    dt_seconds = []
    center_ra_deg_list = []
    center_dec_deg_list = []
    for ii in range(len(frame_list)):
        
        ti = time_list[ii]
        ti_dt = datetime.strptime(ti, '%Y-%m-%dT%H:%M:%S.%f')
        UTC_list.append(ti_dt)
        dt_seconds.append((ti_dt - t0_dt).total_seconds())
        
        ra = center_ra_list[ii]
        dec = center_dec_list[ii]
        ra_deg = hms2deg(ra)
        dec_deg = degstr2deg(dec)
        
        center_ra_deg_list.append(ra_deg)
        center_dec_deg_list.append(dec_deg)
        
    print(dt_seconds)
    print(center_ra_deg_list)
    print(center_dec_deg_list)
    
    data = {}
    data['frame_list'] = frame_list
    data['UTC_list'] = UTC_list
    data['dt_seconds'] = dt_seconds
    data['center_az_deg_list'] = center_az_deg_list
    data['center_el_deg_list'] = center_el_deg_list
    data['center_ra_deg_list'] = center_ra_deg_list
    data['center_dec_deg_list'] = center_dec_deg_list
    data['lead_xpix'] = lead_xpix
    data['lead_ypix'] = lead_ypix
    data['trail_xpix'] = trail_xpix
    data['trail_ypix'] = trail_ypix
    
    
    df = generate_measurement_dataframe(data)

    
    return data


def generate_measurement_dataframe(data_dict):
    
    # Get sensor location
    sensor_ecef = sensor_data()
    
    # Get calibration data
    pixel_mean, pixel_std, ra_mb, dec_mb = generate_calibration_data()
    
    # Setup data for coordinate frame rotations
    eop_alldata = get_celestrak_eop_alldata()
    XYs_df = get_XYs2006_alldata()
    
    
    # Loop over data
    for ii in range(len(data_dict['UTC_list'])):
        
        # Retrieve data
        UTC = data_dict['UTC_list'][ii]
        center_az_deg = data_dict['center_az_deg_list'][ii]
        center_el_deg = data_dict['center_el_deg_list'][ii]
        center_ra_deg = data_dict['center_ra_deg_list'][ii]
        center_dec_deg = data_dict['center_dec_deg_list'][ii]
        lead_xpix = data_dict['lead_xpix'][ii]
        lead_ypix = data_dict['lead_ypix'][ii]
        trail_xpix = data_dict['trail_xpix'][ii]
        trail_ypix = data_dict['trail_ypix'][ii]        
        
        # Compute RA/DEC offsets for this elevation
        ra_offset_deg = center_el_deg*float(ra_mb[0]) + float(ra_mb[1])
        dec_offset_deg = center_el_deg*float(dec_mb[0]) + float(dec_mb[1])
        
        # Compute corrected RA/DEC for center
        center_ra_deg_corrected = center_ra_deg + ra_offset_deg
        center_dec_deg_corrected = center_dec_deg + dec_offset_deg
        center_ra_rad_corrected = center_ra_deg_corrected * pi/180.
        center_dec_rad_corrected = center_dec_deg_corrected * pi/180.
        
        # Check az/el values by converting to RA/DEC
        EOP_data = get_eop_data(eop_alldata, UTC)
        center_az_rad = center_az_deg * pi/180.
        center_el_rad = center_el_deg * pi/180.
        center_ra_rad = center_ra_deg * pi/180.
        center_dec_rad = center_dec_deg * pi/180.
        rho_hat_enu = azel2losvec(center_az_rad, center_el_rad)
        rho_hat_ecef = enu2ecef(rho_hat_enu, sensor_ecef)
        rho_hat_eci_azel, dum = itrf2gcrf(rho_hat_ecef, np.zeros((3,1)), UTC, EOP_data, XYs_df)
        rho_hat_eci_radec = radec2losvec(center_ra_rad, center_dec_rad)
        delta_azel_radec = compute_delta(rho_hat_eci_radec, rho_hat_eci_azel)
        
        # Compute RA/DEC of pixel locations for objects
        
        
        
        
        
        
        print(UTC)
        print('Mount Model RA [rad]: ', center_ra_rad)
        print('Mount Model DEC [rad]: ', center_dec_rad)
        print('Corrected RA [rad]: ', center_ra_rad_corrected)
        print('Corrected DEC [rad]: ', center_dec_rad_corrected)
        print('Delta Az/El and RA/DEC [arcsec]: ', delta_azel_radec * 180/pi * 3600.)
        

        
        mistake
        
    
    
    
    
    
    
    return 0


def validate_radec_calculation():
    
    
    CD = [0.000498599177456, 0.000168285345455, -0.000168149028289, 0.00049861086459]
    theta = asin(CD[1])*180/pi
    
    print(theta)
    
    
    
    return


###############################################################################
# Calibration
###############################################################################



def compute_pixel_width(fov_data):
    
    ra1 = fov_data[1]['ra']
    dec1 = fov_data[1]['dec']
    xpix1 = fov_data[1]['xpix']
    ypix1 = fov_data[1]['ypix']
    
    ra2 = fov_data[2]['ra']
    dec2 = fov_data[2]['dec']
    xpix2 = fov_data[2]['xpix']
    ypix2 = fov_data[2]['ypix']
    
    # Compute ra and dec in radians
    ra1_deg = hms2deg(ra1)
    dec1_deg = degstr2deg(dec1)    
    ra2_deg = hms2deg(ra2)
    dec2_deg = degstr2deg(dec2)

    ra1_rad = ra1_deg * pi/180.
    ra2_rad = ra2_deg * pi/180.
    dec1_rad = dec1_deg * pi/180.
    dec2_rad = dec2_deg * pi/180.
    
    # Compute unit vectors
    rho_hat_eci1 = radec2losvec(ra1_rad, dec1_rad)
    rho_hat_eci2 = radec2losvec(ra2_rad, dec2_rad)
    
    # Compute angle difference
    delta = compute_delta(rho_hat_eci1, rho_hat_eci2)
    
    # Compute pixel width in radians
    npix = np.sqrt((xpix1 - xpix2)**2. + (ypix1 - ypix2)**2.)
    pix_rad = delta/npix    
    
    return pix_rad


def compute_center_offsets(mount_ra, mount_dec, fits_ra, fits_dec):
    '''
    All outputs in degrees.
    
    '''
    
    
    mount_ra_deg = hms2deg(mount_ra)
    mount_dec_deg = degstr2deg(mount_dec)
    fits_ra_deg = hms2deg(fits_ra)
    fits_dec_deg = degstr2deg(fits_dec)
    
    ra_offset = fits_ra_deg - mount_ra_deg
    dec_offset = fits_dec_deg - mount_dec_deg
    
    return [ra_offset, dec_offset]


def compute_pixel_width_grid(corners_ra, corners_dec, corners_xpix, corners_ypix):
    
    
    fov_data = {}
    fov_data[1] = {}
    fov_data[1]['ra'] = corners_ra[0]
    fov_data[1]['dec'] = corners_dec[0]
    fov_data[1]['xpix'] = corners_xpix[0]
    fov_data[1]['ypix'] = corners_ypix[0]
    
    fov_data[2] = {}
    fov_data[2]['ra'] = corners_ra[1]
    fov_data[2]['dec'] = corners_dec[1]
    fov_data[2]['xpix'] = corners_xpix[1]
    fov_data[2]['ypix'] = corners_ypix[1]
    
    pix_rad1 = compute_pixel_width(fov_data)
    
    fov_data = {}
    fov_data[1] = {}
    fov_data[1]['ra'] = corners_ra[2]
    fov_data[1]['dec'] = corners_dec[2]
    fov_data[1]['xpix'] = corners_xpix[2]
    fov_data[1]['ypix'] = corners_ypix[2]
    
    fov_data[2] = {}
    fov_data[2]['ra'] = corners_ra[3]
    fov_data[2]['dec'] = corners_dec[3]
    fov_data[2]['xpix'] = corners_xpix[3]
    fov_data[2]['ypix'] = corners_ypix[3]
    
    pix_rad2 = compute_pixel_width(fov_data)
    
    fov_data = {}
    fov_data[1] = {}
    fov_data[1]['ra'] = corners_ra[0]
    fov_data[1]['dec'] = corners_dec[0]
    fov_data[1]['xpix'] = corners_xpix[0]
    fov_data[1]['ypix'] = corners_ypix[0]
    
    fov_data[2] = {}
    fov_data[2]['ra'] = corners_ra[3]
    fov_data[2]['dec'] = corners_dec[3]
    fov_data[2]['xpix'] = corners_xpix[3]
    fov_data[2]['ypix'] = corners_ypix[3]
    
    pix_rad3 = compute_pixel_width(fov_data)
    
    
    fov_data = {}
    fov_data[1] = {}
    fov_data[1]['ra'] = corners_ra[1]
    fov_data[1]['dec'] = corners_dec[1]
    fov_data[1]['xpix'] = corners_xpix[1]
    fov_data[1]['ypix'] = corners_ypix[1]
    
    fov_data[2] = {}
    fov_data[2]['ra'] = corners_ra[2]
    fov_data[2]['dec'] = corners_dec[2]
    fov_data[2]['xpix'] = corners_xpix[2]
    fov_data[2]['ypix'] = corners_ypix[2]
    
    pix_rad4 = compute_pixel_width(fov_data)
    
    pix_rad_list = [pix_rad1, pix_rad2, pix_rad3, pix_rad4]
        
    return pix_rad_list 


def generate_calibration_data():
    
    # corners = [upper left, upper right, lower right, lower left]
    
    ra_list = []
    dec_list = []
    az_list = []
    el_list = []
    ra_offset_list = []
    dec_offset_list = []
    pixel_list = []
    
    ###########################################################################
    # 2021-09-10 Frame 007 Data
    ###########################################################################
    mount_ra = '04 36 58.141'
    mount_dec = '+28 44 34.53'
    mount_az = 96.2650676929706
    mount_el = 57.0511175076831
    fits_ra = '04 36 20.587'
    fits_dec = '+28 49 16.70'
    
    corners_ra = ['04:35:08.303', '04:38:19.332', '04:37:32.392', '04:34:22.337']
    corners_dec = ['+29:11:40.33', '+28:58:00.18', '+28:26:52.02', '+28:40:27.16']
    corners_xpix = [0.1667, 1391.1667, 1391.1667, 0.5]
    corners_ypix = [1039.5, 1039.5, 0.5, 0.5]
    
    print('\nCMU 2021-09-10 Frame 007')
    offset = compute_center_offsets(mount_ra, mount_dec, fits_ra, fits_dec)
    pixels = compute_pixel_width_grid(corners_ra, corners_dec, corners_xpix, corners_ypix)
    print('offset', offset)
    print('pixels', pixels)
    
    ra_list.append(hms2deg(mount_ra))
    dec_list.append(degstr2deg(mount_dec))
    az_list.append(mount_az)
    el_list.append(mount_el)
    ra_offset_list.append(offset[0])
    dec_offset_list.append(offset[1])
    pixel_list.extend(pixels)

    
    
    ###########################################################################
    # 2021-09-10 Frame 078 Data
    ###########################################################################
    mount_ra = '05 44 35.713'
    mount_dec = '+38 47 45.62'
    mount_az = 75.2513765902723
    mount_el = 54.4343726881934
    fits_ra = '05 43 50.583'
    fits_dec = '+38 52 31.69'
    
    corners_ra = ['05:42:28.698', '05:46:04.049', '05:45:11.180', '05:41:37.469']
    corners_dec = ['+39:14:55.60', '+39:01:20.22', '+38:30:12.34', '+38:43:42.46']
    corners_xpix = [0.5, 1391.5, 1391.5, 0.5]
    corners_ypix = [1039.5, 1039.5, 1.0, 0.5]
    
    print('\nCMU 2021-09-10 Frame 078')
    offset = compute_center_offsets(mount_ra, mount_dec, fits_ra, fits_dec)
    pixels = compute_pixel_width_grid(corners_ra, corners_dec, corners_xpix, corners_ypix)
    print('offset', offset)
    print('pixels', pixels)
    
    ra_list.append(hms2deg(mount_ra))
    dec_list.append(degstr2deg(mount_dec))
    az_list.append(mount_az)
    el_list.append(mount_el)
    ra_offset_list.append(offset[0])
    dec_offset_list.append(offset[1])
    pixel_list.extend(pixels)
    
    
    ###########################################################################
    # 2021-09-16 Frame 035 Data
    ###########################################################################
    mount_ra = '17 32 18.495'
    mount_dec = '-17 47 33.15'
    mount_az = 217.529203971693
    mount_el = 24.1205816864921
    fits_ra = '17 32 58.433'
    fits_dec = '-17 42 27.12'
    
    corners_ra = ['17:34:02.805', '17:31:08.538', '17:31:54.158', '17:34:47.938']
    corners_dec = ['-18:05:10.65', '-17:50:41.01', '-17:19:43.16', '-17:34:08.71']
    corners_xpix = [0.5, 1391.5, 1391.5, 0.5]
    corners_ypix = [1039.5, 1039.5, 0.5, 0.5]
    
    print('\nCMU 2021-09-16 Frame 035')
    offset = compute_center_offsets(mount_ra, mount_dec, fits_ra, fits_dec)
    pixels = compute_pixel_width_grid(corners_ra, corners_dec, corners_xpix, corners_ypix)
    print('offset', offset)
    print('pixels', pixels)
    
    ra_list.append(hms2deg(mount_ra))
    dec_list.append(degstr2deg(mount_dec))
    az_list.append(mount_az)
    el_list.append(mount_el)
    ra_offset_list.append(offset[0])
    dec_offset_list.append(offset[1])
    pixel_list.extend(pixels)
    
    
    ###########################################################################
    # 2021-09-16 Frame 036 Data
    ###########################################################################
    mount_ra = '15 39 11.115'
    mount_dec = '+43 46 03.28'
    mount_az = 297.63600622219
    mount_el = 42.5465410327429
    fits_ra = '15:40:04.477'
    fits_dec = '+43:47:02.10'
    
    corners_ra = ['15:41:28.340', '15:37:39.750', '15:38:39.641', '15:42:29.476']
    corners_dec = ['+43:24:13.86', '+43:38:47.79', '+44:09:45.51', '+43:55:06.47']
    corners_xpix = [0.5, 1391., 1391., 0.5]
    corners_ypix = [1039.5, 1039.5, 0.5, 0.5]
    
    print('\nCMU 2021-09-16 Frame 036')
    offset = compute_center_offsets(mount_ra, mount_dec, fits_ra, fits_dec)
    pixels = compute_pixel_width_grid(corners_ra, corners_dec, corners_xpix, corners_ypix)
    print('offset', offset)
    print('pixels', pixels)
    
    ra_list.append(hms2deg(mount_ra))
    dec_list.append(degstr2deg(mount_dec))
    az_list.append(mount_az)
    el_list.append(mount_el)
    ra_offset_list.append(offset[0])
    dec_offset_list.append(offset[1])
    pixel_list.extend(pixels)
    
    
    ###########################################################################
    # 2021-09-16 Frame 037 Data
    ###########################################################################
    mount_ra = '17 06 41.793'
    mount_dec = '+81 16 20.91'
    mount_az = 351.437722816466
    mount_el = 44.9819069974655
    fits_ra = '17:10:12.504'
    fits_dec = '+81:08:56.03'
    
    corners_ra = ['17:16:07.923', '16:58:53.676', '17:03:54.204', '17:21:48.906']
    corners_dec = ['+80:45:19.12', '+81:00:54.86', '+81:32:07.18', '+81:15:14.25']
    corners_xpix = [0.5, 1391.5, 1391.5, 0.5]
    corners_ypix = [1039.5, 1039.5, 0.5, 0.5]
    
    print('\nCMU 2021-09-16 Frame 037')
    offset = compute_center_offsets(mount_ra, mount_dec, fits_ra, fits_dec)
    pixels = compute_pixel_width_grid(corners_ra, corners_dec, corners_xpix, corners_ypix)
    print('offset', offset)
    print('pixels', pixels)
    
    ra_list.append(hms2deg(mount_ra))
    dec_list.append(degstr2deg(mount_dec))
    az_list.append(mount_az)
    el_list.append(mount_el)
    ra_offset_list.append(offset[0])
    dec_offset_list.append(offset[1])
    pixel_list.extend(pixels)
    
    ###########################################################################
    # 2021-09-16 Frame 038 Data
    ###########################################################################
    mount_ra = '00 12 23.323'
    mount_dec = '+61 37 56.41'
    mount_az = 37.4854657199066
    mount_el = 46.202955708531
    fits_ra = '00:10:59.154'
    fits_dec = '+61:39:41.33'
    
    corners_ra = ['00:08:44.469', '00:14:38.916', '00:13:10.405', '00:07:21.575']
    corners_dec = ['+62:02:01.31', '+61:48:14.65', '+61:17:12.70', '+61:30:45.43']
    corners_xpix = [0.5, 1391.5, 1391.5, 0.5]
    corners_ypix = [1039.5, 1039.5, 0.5, 0.5]
    
    print('\nCMU 2021-09-16 Frame 038')
    offset = compute_center_offsets(mount_ra, mount_dec, fits_ra, fits_dec)
    pixels = compute_pixel_width_grid(corners_ra, corners_dec, corners_xpix, corners_ypix)
    print('offset', offset)
    print('pixels', pixels)
    
    ra_list.append(hms2deg(mount_ra))
    dec_list.append(degstr2deg(mount_dec))
    az_list.append(mount_az)
    el_list.append(mount_el)
    ra_offset_list.append(offset[0])
    dec_offset_list.append(offset[1])
    pixel_list.extend(pixels)
    
    ###########################################################################
    # 2021-09-16 Frame 039 Data
    ###########################################################################
    mount_ra = '23 13 46.356'
    mount_dec = '-07 07 21.03'
    mount_az = 124.993323976242
    mount_el = 26.2400296735649
    fits_ra = '23:13:15.352'
    fits_dec = '-06:55:13.61'
    
    corners_ra = ['23:12:12.291', '23:15:00.334', '23:14:18.628', '23:11:30.772']
    corners_dec = ['-06:32:41.39', '-06:46:34.43', '-07:17:43.16', '-07:03:47.85']
    corners_xpix = [0.5, 1391.5, 1391.5, 0.5]
    corners_ypix = [1039.5, 1039.5, 0.5, 0.5]
    
    print('\nCMU 2021-09-16 Frame 039')
    offset = compute_center_offsets(mount_ra, mount_dec, fits_ra, fits_dec)
    pixels = compute_pixel_width_grid(corners_ra, corners_dec, corners_xpix, corners_ypix)
    print('offset', offset)
    print('pixels', pixels)
    
    ra_list.append(hms2deg(mount_ra))
    dec_list.append(degstr2deg(mount_dec))
    az_list.append(mount_az)
    el_list.append(mount_el)
    ra_offset_list.append(offset[0])
    dec_offset_list.append(offset[1])
    pixel_list.extend(pixels)
    
    
    ###########################################################################
    # 2021-09-16 Frame 040 Data
    ###########################################################################
    mount_ra = '02 12 08.156'
    mount_dec = '+32 27 16.35'
    mount_az = 62.6629538059287
    mount_el = 18.4957877882498
    fits_ra = '02:11:21.472'
    fits_dec = '+32:37:23.43'
    
    corners_ra = ['02:10:06.366', '02:13:25.317', '02:12:36.147', '02:09:18.454']
    corners_dec = ['+32:59:46.44', '+32:46:01.41', '+32:14:56.16', '+32:28:37.03']
    corners_xpix = [0.5, 1391.5, 1391.5, 0.5]
    corners_ypix = [1039.5, 1039.5, 0.5, 0.5]
    
    print('\nCMU 2021-09-16 Frame 040')
    offset = compute_center_offsets(mount_ra, mount_dec, fits_ra, fits_dec)
    pixels = compute_pixel_width_grid(corners_ra, corners_dec, corners_xpix, corners_ypix)
    print('offset', offset)
    print('pixels', pixels)
    
    ra_list.append(hms2deg(mount_ra))
    dec_list.append(degstr2deg(mount_dec))
    az_list.append(mount_az)
    el_list.append(mount_el)
    ra_offset_list.append(offset[0])
    dec_offset_list.append(offset[1])
    pixel_list.extend(pixels)
    
    
    ###########################################################################
    # 2021-09-22 Frame 001 Data
    ###########################################################################
    mount_ra = '22 47 54.594'
    mount_dec = '+25 17 08.70'
    mount_az = 88.5561588299199
    mount_el = 41.2023415847586
    fits_ra = '22:47:09.317'
    fits_dec = '+25:21:18.85'
    
    corners_ra = ['22:46:00.320', '22:49:04.769', '22:48:17.926', '22:45:14.106']
    corners_dec = ['+25:43:51.82', '+25:29:47.15', '+24:58:43.12', '+25:12:46.16']
    corners_xpix = [0.5, 1391.5, 1391.5, 0.5]
    corners_ypix = [1039.5, 1039.5, 0.5, 0.5]
    
    print('\nCMU 2021-09-22 Frame 001')
    offset = compute_center_offsets(mount_ra, mount_dec, fits_ra, fits_dec)
    pixels = compute_pixel_width_grid(corners_ra, corners_dec, corners_xpix, corners_ypix)
    print('offset', offset)
    print('pixels', pixels)
    
    ra_list.append(hms2deg(mount_ra))
    dec_list.append(degstr2deg(mount_dec))
    az_list.append(mount_az)
    el_list.append(mount_el)
    ra_offset_list.append(offset[0])
    dec_offset_list.append(offset[1])
    pixel_list.extend(pixels)
    
    
    ###########################################################################
    # 2021-09-22 Frame 002 Data
    ###########################################################################
    mount_ra = '23 00 27.087'
    mount_dec = '+18 36 20.55'
    mount_az = 94.1677340861094
    mount_el = 35.6940151922101
    fits_ra = '22:59:44.208'
    fits_dec = '+18:41:54.95'
    
    corners_ra = ['22:58:38.300', '23:01:34.455', '23:00:49.624', '22:57:54.363']
    corners_dec = ['+19:04:27.72', '+18:50:22.81', '+18:19:18.73', '+18:33:21.12']
    corners_xpix = [0.5, 1391.5, 1391.5, 0.5]
    corners_ypix = [1039.5, 1039.5, 0.5, 0.5]
    
    print('\nCMU 2021-09-22 Frame 002')
    offset = compute_center_offsets(mount_ra, mount_dec, fits_ra, fits_dec)
    pixels = compute_pixel_width_grid(corners_ra, corners_dec, corners_xpix, corners_ypix)
    print('offset', offset)
    print('pixels', pixels)
    
    ra_list.append(hms2deg(mount_ra))
    dec_list.append(degstr2deg(mount_dec))
    az_list.append(mount_az)
    el_list.append(mount_el)
    ra_offset_list.append(offset[0])
    dec_offset_list.append(offset[1])
    pixel_list.extend(pixels)
    
    
    ###########################################################################
    # 2021-09-22 Frame 003 Data
    ###########################################################################
    mount_ra = '23 09 14.310'
    mount_dec = '+12 06 42.28'
    mount_az = 99.0566161931329
    mount_el = 30.3243604243906
    fits_ra = '23:08:33.152'
    fits_dec = '+12:13:41.47'
    
    corners_ra = ['23:07:29.581', '23:10:19.940', '23:09:37.134', '23:06:46.537']
    corners_dec = ['+12:36:18.75', '+12:22:10.84', '+11:51:17.70', '+12:05:09.89']
    corners_xpix = [0.5, 1391.5, 1391.5, 0.5]
    corners_ypix = [1039.5, 1039.5, 0.5, 0.5]
    
    print('\nCMU 2021-09-22 Frame 003')
    offset = compute_center_offsets(mount_ra, mount_dec, fits_ra, fits_dec)
    pixels = compute_pixel_width_grid(corners_ra, corners_dec, corners_xpix, corners_ypix)
    print('offset', offset)
    print('pixels', pixels)
    
    ra_list.append(hms2deg(mount_ra))
    dec_list.append(degstr2deg(mount_dec))
    az_list.append(mount_az)
    el_list.append(mount_el)
    ra_offset_list.append(offset[0])
    dec_offset_list.append(offset[1])
    pixel_list.extend(pixels)
    
    
    ###########################################################################
    # 2021-09-22 Frame 005 Data
    ###########################################################################
    mount_ra = '23 23 12.169'
    mount_dec = '+08 19 35.05'
    mount_az = 100.534657123327
    mount_el = 25.9083438403895
    fits_ra = '23:22:33.226'
    fits_dec = '+08:27:24.60' 
    
    corners_ra = ['23:21:30.247', '23:24:18.741', '23:23:36.240', '23:20:48.038']
    corners_dec = ['+08:50:01.92', '+08:36:00.04', '+08:04:52.47', '+08:18:57.44']
    corners_xpix = [0.5, 1391.5, 1391.5, 0.5]
    corners_ypix = [1039.5, 1039.5, 0.5, 0.5]
    
    print('\nCMU 2021-09-22 Frame 005')
    offset = compute_center_offsets(mount_ra, mount_dec, fits_ra, fits_dec)
    pixels = compute_pixel_width_grid(corners_ra, corners_dec, corners_xpix, corners_ypix)
    print('offset', offset)
    print('pixels', pixels)
    
    ra_list.append(hms2deg(mount_ra))
    dec_list.append(degstr2deg(mount_dec))
    az_list.append(mount_az)
    el_list.append(mount_el)
    ra_offset_list.append(offset[0])
    dec_offset_list.append(offset[1])
    pixel_list.extend(pixels)
    
    
    ###########################################################################
    # 2021-09-22 Frame 007 Data
    ###########################################################################
    mount_ra = '23 39 36.318'
    mount_dec = '+05 14 31.09'
    mount_az = 100.696243322839
    mount_el = 21.2550955811482
    fits_ra = '23:38:59.621'
    fits_dec = '+05:23:12.92'
    
    corners_ra = ['23:37:56.841', '23:40:44.111', '23:40:02.111', '23:37:14.931']
    corners_dec = ['+05:45:48.12', '+05:31:46.67', '+05:00:40.85', '+05:14:43.36']
    corners_xpix = [0.5, 1391.5, 1391.5, 0.5]
    corners_ypix = [1039.5, 1039.5, 0.5, 0.5]
    
    print('\nCMU 2021-09-22 Frame 007')
    offset = compute_center_offsets(mount_ra, mount_dec, fits_ra, fits_dec)
    pixels = compute_pixel_width_grid(corners_ra, corners_dec, corners_xpix, corners_ypix)
    print('offset', offset)
    print('pixels', pixels)
    
    ra_list.append(hms2deg(mount_ra))
    dec_list.append(degstr2deg(mount_dec))
    az_list.append(mount_az)
    el_list.append(mount_el)
    ra_offset_list.append(offset[0])
    dec_offset_list.append(offset[1])
    pixel_list.extend(pixels)
    
    
    ###########################################################################
    # 2021-09-22 Frame 008 Data
    ###########################################################################
    mount_ra = '23 30 46.232'
    mount_dec = '-07 45 57.49'
    mount_az = 112.737683274018
    mount_el = 14.5391737814157
    fits_ra = '23:30:11.863'
    fits_dec = '-07:34:57.77'
    
    corners_ra = ['23:29:08.827', '23:31:57.313', '23:31:15.286', '23:28:26.719']
    corners_dec = ['-07:12:22.56', '-07:26:31.52', '-07:57:36.79', '-07:43:35.04']
    corners_xpix = [0.5, 1391.5, 1391.5, 0.5]
    corners_ypix = [1039.5, 1039.5, 0.5, 0.5]
    
    print('\nCMU 2021-09-22 Frame 008')
    offset = compute_center_offsets(mount_ra, mount_dec, fits_ra, fits_dec)
    pixels = compute_pixel_width_grid(corners_ra, corners_dec, corners_xpix, corners_ypix)
    print('offset', offset)
    print('pixels', pixels)
    
    ra_list.append(hms2deg(mount_ra))
    dec_list.append(degstr2deg(mount_dec))
    az_list.append(mount_az)
    el_list.append(mount_el)
    ra_offset_list.append(offset[0])
    dec_offset_list.append(offset[1])
    pixel_list.extend(pixels)
    
    
    
    ###########################################################################
    # Final Outputs
    ###########################################################################
    
    print('\n\nPixel Width Mean, STD [rad]:', np.mean(pixel_list), np.std(pixel_list))
    print('Mean RA offset [deg]: ', np.mean(np.abs(ra_offset_list)))
    print('Mean DEC offset [deg]: ', np.mean(np.abs(dec_offset_list)))
    
    plot_inds = [8, 9, 10, 11, 12, 13]
    el_plot = [el_list[ind] for ind in plot_inds]
    az_plot = [az_list[ind] for ind in plot_inds]
    ra_offset_plot = [ra_offset_list[ind] for ind in plot_inds]
    dec_offset_plot = [dec_offset_list[ind] for ind in plot_inds]
    
    H = np.ones((len(plot_inds), 2))
    H[:,0] = el_plot
    print(H)
    
    HtHinv = np.linalg.inv(np.dot(H.T, H))
    ra_offset_vec = np.reshape(ra_offset_plot, (len(plot_inds),1))
    dec_offset_vec = np.reshape(dec_offset_plot, (len(plot_inds),1))
    
    ra_mb = np.dot(HtHinv, np.dot(H.T, ra_offset_vec))
    dec_mb = np.dot(HtHinv, np.dot(H.T, dec_offset_vec))
    
    print(ra_mb)
    print(dec_mb)
    
    ra_line = [el_plot[0]*ra_mb[0] + ra_mb[1], el_plot[-1]*ra_mb[0] + ra_mb[1]]
    dec_line = [el_plot[0]*dec_mb[0] + dec_mb[1], el_plot[-1]*dec_mb[0] + dec_mb[1]]
    
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(el_plot, ra_offset_plot, 'bo')
    plt.plot([el_plot[0], el_plot[-1]], ra_line, 'b--')
    plt.ylabel('RA Offset [deg]')
    plt.subplot(2,1,2)
    plt.plot(el_plot, dec_offset_plot, 'bo')
    plt.plot([el_plot[0], el_plot[-1]], dec_line, 'b--')
    plt.ylabel('DEC Offset [deg]')
    plt.xlabel('Elevation [deg]')
    
    plt.show()
    
    
    
    
    return np.mean(pixel_list), np.std(pixel_list), ra_mb, dec_mb





###############################################################################
# Utility Functions
###############################################################################
    

def hms2deg(hms_string):
    
    deg = (float(hms_string[0:2]) + float(hms_string[3:5])/60. + float(hms_string[6:12])/3600.)*15.
    
    return deg


def degstr2deg(deg_string):
    
    sign = deg_string[0]
    deg = float(deg_string[1:3]) + float(deg_string[4:6])/60. + float(deg_string[7:12])/3600.
    if sign == '-':
        deg *= -1.
    
    return deg
    

def radec2losvec(ra, dec):
    '''
    This function computes the LOS unit vector in ECI frame given topocentric
    RA and DEC in radians.
    
    '''
    
    rho_hat_eci = np.array([[cos(ra)*cos(dec)],
                            [sin(ra)*cos(dec)],
                            [sin(dec)]])
    
    return rho_hat_eci


def azel2losvec(az, el):
    '''
    This function computes the LOS unit vector in ENU frame given 
    az/el in radians.
    
    '''
    
    rho_hat_enu = np.array([[sin(az)*cos(el)],
                            [cos(az)*cos(el)],
                            [sin(el)]])
    
    return rho_hat_enu


def losvec2radec(rho_hat_eci):
    '''
    This function computes topocentric RA/DEC in radians given LOS unit vector
    in ECI.
    
    '''
    
    ra = atan2(rho_hat_eci[1], rho_hat_eci[0])  # rad
    dec = asin(rho_hat_eci[2])                 # rad
    
    return ra, dec


def compute_delta(rho_hat1, rho_hat2):
    '''
    This function computes the angle in radians between two unit vectors
    '''
    
    delta = acos(np.dot(rho_hat1.T, rho_hat2))
    
    return delta


def compute_dec(dec1, delta):
    '''
    All angles in radians.
    '''
    
    #rho_hat1 = radec2losvec(ra, dec1)
    
    dec2 = dec1 + delta    
    
    return dec2


def compute_ra(ra1, dec, delta):
    '''
    All angles in radians.
    '''
    
    rhs = (cos(delta) - sin(dec)**2.)/(cos(dec)**2)
    diff = acos(rhs)
    
    if delta > 0.:
        ra2 = ra1 + diff
    else:
        ra2 = ra1 - diff
    
    
    return ra2


def compute_radec_pixels(fits_x, fits_y, theta):
    '''
    theta in radians
    
    '''
    
    R = np.array([[ cos(theta), sin(theta)],
                  [-sin(theta), cos(theta)]])
        
    pix_vec = np.dot(R, np.array([[fits_x],[fits_y]])) - np.array([[695.5],[519.5]])
    
    ra_pix = float(pix_vec[0])
    dec_pix = float(pix_vec[1])
    
    return ra_pix, dec_pix



if __name__ == '__main__':
    
    plt.close('all')
    
#    demo_distance_calc()
    
#    cmu_falcon_data2()
    
#    generate_calibration_data_table()
    
    validate_radec_calculation()
    
    