# -*- coding: utf-8 -*-
"""
Created on Tue Nov 9 16:09:00 2021

This program crudely extracts the orbital data from SP3 files for a specific satellite, converts to RA and Dec for a given location on Earth, and plots the output. 
(a modified version of "SP3_plot_clean.py")

This program is meant to help us determine how to best interpolate the QZSS data from the 15 min cadence to the high time resolution of the ROO data, 
as a way of figuring out how accurate the angles are from ROO (broader analysis)

Yep, pulling in ROO data now and focusing on those observations compared to the interpolated SP3 data for QZSS...

@author: Dr Brett Carter, RMIT University
"""

import astropy
from astropy.time import Time, TimeDelta
import numpy as np
from math import *
import matplotlib.pyplot as plt
from astropy.coordinates import (SkyCoord, 
                                GCRS, ITRS, EarthLocation, AltAz, TETE, CIRS)
from astropy import units as u
from astropy.coordinates.attributes import (TimeAttribute,
                                            CartesianRepresentationAttribute)
from scipy.interpolate import (interp1d, lagrange)
import sys
import csv
from datetime import datetime, timedelta


#adding Steve's Metis project directory to the path to access functions and conversions
sys.path.append('./metis/')

#import utilities.astrodynamics as astro
import utilities.coordinate_systems as coord
import utilities.eop_functions as eop



#############################################################
#############################################################
#First, we get ROO's data (now in csv format)


ROO_dir = "ROO_data/"
#a = "PNGs_angles.csv"
a = "PNGs_angles_cleaned.csv"
#a = "PNGs_angles_test_cleaned.csv"
#a = "PNGs_angles_uncorrected_time_cleaned.csv"
#a = "PNGs_angles_corrected_time_cleaned.csv"
files = [a]


for i in files:
    
    RA = []
    Dec = []
    jd_time = []
    utc_time = []
    fitname = []
    
    
    iline = 0
    
    g = open(ROO_dir+i,'r')
    
    header1 = g.readline()
    labels = header1.split()
    
    for line in g:
        
        columns = line.split(",")
        
        RA.append(float(columns[3]))            
        Dec.append(float(columns[4]))
        t = Time(columns[0],format='isot',scale='utc')# + timedelta(seconds = 0.81) + timedelta(seconds = 0.5)
        utc_time.append(t)
        jd_time.append(t.jd)
        
    
    g.close()
    
    RA_array_a = np.array(RA)
    Dec_array_a = np.array(Dec)
    JD_time_a = np.array(jd_time)
    
    #+/- 15 mins...
    start_time_JD = min(JD_time_a) - (15/(24*60))
    end_time_JD = max(JD_time_a) + (15/(24*60))
    
#############################################################
#next we'll get the QZSS "truth data"

data_dir = 'qzss.go.jp/'

#November 8th 2021
filename = "qzu21831_06.sp3"   #ultra rapid
filename = "qzf21831.sp3"   #ultra rapid

#November 10th 2021
#filename = "qzr21833.sp3"   #rapid


satellite = "PJ03"
#satellite = "PJ01"
head_len = 23

#what we've been using
#melb = [-37.840935*u.deg, 144.946457*u.deg, 25*u.m]
#Yang's values
#melb = [-37.680563*u.deg, 145.061638*u.deg, 172.4*u.m]

#Daniel's measured values (Easting, Northing, height)
#EarthLocation.from_geocentric(-329072.866*u.m, 5827855.595*u.m, 155.083*u.m)
#GRS80
melb = [-37.680589141*u.deg, 145.061634327*u.deg, 155.083*u.m]

#inserting observing location (specifying GRS80 ellipsoid)
melbourne = EarthLocation.from_geodetic(lat=melb[0], lon=melb[1], height=melb[2], ellipsoid = 'GRS80')


#for Steve's calcs
eop_alldata = eop.get_celestrak_eop_alldata()    


f = open(data_dir+filename,'r')

#initialising lists that will be filled with the data
rx = []
ry = []
rz = []
time_stamp = []


#skipping header
for i in range(1,head_len):
    header1 = f.readline()


for line in f:
    
    columns = line.split()
    
    #read in time
    if len(columns) == 7:
        #obs_time_format = "2021-03-31 09:15:00"
        year = columns[1]
        month = columns[2]
        day = columns[3]
        hour = columns[4]
        minute = columns[5]
        second = str(int(float(columns[6])))        
        
        str_time = year+'-'+month.zfill(2)+'-'+day.zfill(2)+"T"+hour.zfill(2)+":"+minute.zfill(2)+":"+second.zfill(2)+"Z"
        
        t = Time(str_time, format='isot',scale='utc') - TimeDelta(18 * u.s)
        
        if t.jd > start_time_JD and t.jd < end_time_JD:  
            time_stamp.append(str_time)
        
        
    else:    
        #when it's not the date, there's more values
        #only get data for our "satellite"
        if columns[0] == satellite and t.jd > start_time_JD and t.jd < end_time_JD:
            rx.append(float(columns[1]))
            ry.append(float(columns[2]))
            rz.append(float(columns[3]))
    
f.close()

rx = np.array(rx)
ry = np.array(ry)
rz = np.array(rz)

#######################################################################################
#Now to do the conversion from position vector to RA and Dec, as observed from Melbourne
qzss_ra = []
qzss_dec = []

qzss_ra_2 = []
qzss_dec_2 = []

qzss_ra_steve = []
qzss_dec_steve = []

plot_time = []
index = 0

for i in time_stamp:

    #creating time object
    t2 = Time(i, format='isot',scale='utc') - (18 * u.s) #+ (0.81 * u.s)
    
    #if t2.jd > start_time_JD and t2.jd < end_time_JD:
    #setting up SkyCoord object (astropy)
    pos = SkyCoord(rx[index]*u.km,ry[index]*u.km,rz[index]*u.km,frame = 'itrs', obstime = t2)
    
    ###################################
    #An experimental method
    
    r = [rx[index]*u.km,ry[index]*u.km,rz[index]*u.km]
    
    #for satellite
    itrs_sat = ITRS((r[0],r[1],r[2]), obstime = t2,representation_type="cartesian")
    gcrs_sat = itrs_sat.transform_to(GCRS(obstime = t2))

    #for observation location
    itrs_location = ITRS((melbourne.x,melbourne.y,melbourne.z),obstime = t2,representation_type="cartesian")
    gcrs_location = itrs_location.transform_to(GCRS(obstime = t2))
    
    #now to work out the observation vector
    rho_eci_2x = (gcrs_sat.cartesian.x - gcrs_location.cartesian.x)/u.m
    rho_eci_2y = (gcrs_sat.cartesian.y - gcrs_location.cartesian.y)/u.m
    rho_eci_2z = (gcrs_sat.cartesian.z - gcrs_location.cartesian.z)/u.m
    
    rho_eci_2 = np.array([rho_eci_2x,rho_eci_2y,rho_eci_2z])

    rho_2 = np.linalg.norm(rho_eci_2)
    
    ra_2 = atan2(rho_eci_2[1], rho_eci_2[0])*180/pi
    dec_2 = asin(rho_eci_2[2]/rho_2)*180/pi
    
    if ra_2 < 0:
        ra_2 += 360
    
    qzss_ra_2.append(ra_2)
    qzss_dec_2.append(dec_2)
    
    ##############################
    
    
    #performing conversion of ITRS values to TETE
    ee = TETE(location = melbourne, obstime = t2, representation_type="cartesian")
    #ee = GCRS(obstime = t2, representation_type="cartesian", obsgeoloc = [melbourne.x,melbourne.y,melbourne.z]*u.m)
    bb = pos.transform_to(ee)

    #saving them for plotting later
    qzss_ra.append(bb.ra/u.deg)
    qzss_dec.append(bb.dec/u.deg)
    
    
    #convert plot_time to JD time (including a shift of 18 seconds due to GPS leap seconds (GPS time appears to be ahead by 18 seconds... so it needs to be taken away))
    #check out http://leapsecond.com/java/gpsclock.htm
    t = Time(i, format='isot',scale='utc') - TimeDelta(18 * u.s) #+ (0.81 * u.s)
    
    #converting time to JD and saving it
    plot_time.append(t.jd)
    
    #####################################################################################
    #doing Steve's one now...
    time_split = i.split("T")
    time_split_2 = time_split[0].split("-")
    time_split_3 = time_split[1].split(":")
    
    year = int(time_split_2[0])
    month = int(time_split_2[1])
    day = int(time_split_2[2])
    hr = int(time_split_3[0])
    mn = int(time_split_3[1])
    sec = int(time_split_3[2].split("Z")[0])
    
    #is the mid-point time being recorded correctly by the pipeline? Yes, it is, even if it makes the Dec more correct...
    UTC = datetime(year, month, day, hr, mn, sec) - timedelta(seconds = 18)# + timedelta(seconds = 0.81)+ timedelta(seconds = 0.5)
    
    EOP_data = eop.get_eop_data(eop_alldata, UTC)
    
    v_ITRF = np.array([[ 0.],
                       [0.],
                       [0.]])
    
    r_ITRF = np.array([[rx[index]],[ry[index]],[rz[index]]])
    
    #satellite location in GCRF
    r_GCRF, v_GCRF = coord.itrf2gcrf(r_ITRF, v_ITRF, UTC, EOP_data)
    
    
    #Site location in ECEF
    melb_efec = coord.latlonht2ecef(melb[0]/u.deg,melb[1]/u.deg,melb[2]/(1e3*u.m))

    #Site location in GCRF
    melb_gcrf, melb_v = coord.itrf2gcrf(melb_efec, [0.,0.,0.], UTC, EOP_data)

    rho_eci = r_GCRF - melb_gcrf
    rho = np.linalg.norm(rho_eci)
    
    #print(rho)
    
    steve_ra = atan2(rho_eci[1], rho_eci[0])*180/pi
    steve_dec = asin(rho_eci[2]/rho)*180/pi
    
    if steve_ra < 0:
        steve_ra +=360

    qzss_ra_steve.append(steve_ra)
    qzss_dec_steve.append(steve_dec)
    

    index+=1
    
    #####################################################################################

qzss_ra = np.array(qzss_ra)
qzss_dec = np.array(qzss_dec)

qzss_ra_2 = np.array(qzss_ra_2)
qzss_dec_2 = np.array(qzss_dec_2)

qzss_ra_steve = np.array(qzss_ra_steve)
qzss_dec_steve = np.array(qzss_dec_steve)



#QZS data is 15-min cadence, so we must interpolate (trying out both linear and cubic interpolations)

#high-res x axis (every minute between 0UT and 23:55 UT)
plot_time_new = np.linspace(min(plot_time),max(plot_time),num=1435,endpoint=True)

#interpolation RA
f_ra = interp1d(plot_time, qzss_ra)
#f3_ra = interp1d(plot_time, qzss_ra, kind='cubic')


#interpolation RA Steve
f_ra_steve = interp1d(plot_time, qzss_ra_steve)
#f3_ra_steve = interp1d(plot_time, qzss_ra_steve,kind='cubic')

f_ra_lagr = lagrange(plot_time, qzss_ra_steve)

#interpolation - astropy (fixed)
f_ra_2 = interp1d(plot_time, qzss_ra_2)
#f3_ra_2 = interp1d(plot_time, qzss_ra_2,kind='cubic')

#interpolation Dec
f_dec = interp1d(plot_time, qzss_dec)
#f3_dec = interp1d(plot_time, qzss_dec, kind='cubic')


#interpolation Dec Steve
f_dec_steve = interp1d(plot_time, qzss_dec_steve)
#f3_dec_steve = interp1d(plot_time, qzss_dec_steve,kind='cubic')

f_dec_lagr = lagrange(plot_time, qzss_dec_steve)


#interpolation - astropy (fixed)
f_dec_2 = interp1d(plot_time, qzss_dec_2)
#f3_dec_2 = interp1d(plot_time, qzss_dec_2,kind='cubic')

    
#############################################################
#############################################################
#plotting all now

plt.figure(1)
ax = plt.subplot(211)
plt.xlabel("Time")
plt.ylabel("Declination")

xmin = min(JD_time_a) - (max(JD_time_a) - min(JD_time_a))*0.1
xmax= max(JD_time_a) + (max(JD_time_a) - min(JD_time_a))*0.1


ymin = -60
ymax = 60

ymin = min(Dec_array_a) - (max(Dec_array_a) - min(Dec_array_a))*0.1
ymax = max(Dec_array_a) + (max(Dec_array_a) - min(Dec_array_a))*0.1


plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)

#Truth
plt.plot(plot_time,qzss_dec,'ro')

#Truth, interpolated
plt.plot(plot_time_new,f_dec(plot_time_new),'b-',plot_time_new)#,f3_dec(plot_time_new),'g--')
#plt.plot(plot_time_new,f_dec2(plot_time_new),'b-',plot_time_new,f3_dec2(plot_time_new),'g--')
plt.plot(plot_time_new,f_dec_steve(plot_time_new),'g-',plot_time, qzss_dec_steve,'go')
#plt.plot(plot_time_new,f_dec_steve(plot_time_new),'g-',plot_time_new,f3_dec_steve(plot_time_new),'g--')
plt.plot(plot_time_new,f_dec_2(plot_time_new),'r-', plot_time, qzss_dec_2,'b+')

#plt.plot(plot_time_new,f_dec_lagr(plot_time_new),'y')

#print(f_dec_lagr(plot_time_new))

#Observed
###
plt.plot(JD_time_a,Dec_array_a,'rx')

ax2 = plt.subplot(212,sharex=ax)
plt.xlabel("Time")
plt.ylabel("Right Ascension")

ymin = min(RA_array_a) - (max(RA_array_a) - min(RA_array_a))*0.1
ymax = max(RA_array_a) + (max(RA_array_a) - min(RA_array_a))*0.1


plt.ylim(ymin,ymax)

#Truth
plt.plot(plot_time,qzss_ra,'ro')
#plt.plot(plot_time,qzss_ra2,'r+')

#plt.setp(ax2.get_xticklabels(),fontsize=6)

#Truth, interpolated
plt.plot(plot_time_new,f_ra(plot_time_new),'b-')#,plot_time_new,f3_ra(plot_time_new),'g--')
#plt.plot(plot_time_new,f_ra2(plot_time_new),'b-',plot_time_new,f3_ra2(plot_time_new),'g--')

#plt.plot(plot_time_new,f_ra_steve(plot_time_new),'g-',plot_time_new,f3_ra_steve(plot_time_new),'g--')
plt.plot(plot_time_new,f_ra_steve(plot_time_new),'g-', plot_time, qzss_ra_steve, 'go', )
plt.plot(plot_time_new,f_ra_2(plot_time_new),'g-', plot_time, qzss_ra_2,'b+' )


#Observed
###
plt.plot(JD_time_a,RA_array_a,'rx')


plt.plot(plot_time_new,f_ra_lagr(plot_time_new),'y')

plt.show()



###########################################################
#Doing difference plots now...

#cycling through measured data, looking for the corresponding interpolated truth data point, and calculating the differences

diff_dec = []
diff_ra = []

diff_dec_2 = []
diff_ra_2 = []

diff_ra_3 = []
diff_dec_3 = []

index = 0

for i in JD_time_a:
    
    #RA
    diff_ra.append(RA_array_a[index] - f_ra_steve(i))
    
    diff_ra_2.append(RA_array_a[index] - f_ra_2(i))
    
    diff_ra_3.append(RA_array_a[index] - f_ra_lagr(i))
    
    #DEC
    diff_dec.append(Dec_array_a[index] - f_dec_steve(i))
    
    diff_dec_2.append(Dec_array_a[index] - f_dec_2(i))
    
    diff_dec_3.append(Dec_array_a[index] - f_dec_lagr(i))
    
    index += 1
    


diff_ra = np.array(diff_ra)
diff_dec = np.array(diff_dec)

mean_diff_dec = np.nanmean(diff_dec)
mean_diff_ra = np.nanmean(diff_ra)


diff_ra_2 = np.array(diff_ra_2)
diff_dec_2 = np.array(diff_dec_2)

mean_diff_dec_2 = np.nanmean(diff_dec_2)
mean_diff_ra_2 = np.nanmean(diff_ra_2)


diff_ra_3 = np.array(diff_ra_3)
diff_dec_3 = np.array(diff_dec_3)


mean_diff_dec_3 = np.nanmean(diff_dec_3)
mean_diff_ra_3 = np.nanmean(diff_ra_3)

#plotting the differences
plt.figure(2)

ax3 = plt.subplot(211)
plt.xlabel("Time")
plt.ylabel("Declination (deg)")


ymin = min(diff_dec) - (max(diff_dec) - min(diff_dec))*0.1
ymax = max(diff_dec) + (max(diff_dec) - min(diff_dec))*0.1

#ymax = -41
#ymin = -43

plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)

plt.text(0.5*(xmax-xmin)+xmin,0.7*(ymax-ymin)+ymin,"Mean: "+str(mean_diff_dec*3600)+" arcseconds")
plt.text(0.5*(xmax-xmin)+xmin,0.3*(ymax-ymin)+ymin,"Mean: "+str(mean_diff_dec_2*3600)+" arcseconds")

#Declination
plt.plot(JD_time_a,diff_dec,'ro',JD_time_a,diff_dec_2,'bx',JD_time_a,diff_dec_3,'r+')



ax4 = plt.subplot(212,sharex=ax3)
plt.xlabel("Time")
plt.ylabel("Right Ascension (deg)")


ymin = min(diff_ra) - (max(diff_ra) - min(diff_ra))*0.1
ymax = max(diff_ra) + (max(diff_ra) - min(diff_ra))*0.1

#ymax = -41
#ymin = -43

plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)

plt.text(0.5*(xmax-xmin)+xmin,0.7*(ymax-ymin)+ymin,"Mean: "+str(mean_diff_ra*3600)+" arcseconds")
plt.text(0.5*(xmax-xmin)+xmin,0.3*(ymax-ymin)+ymin,"Mean: "+str(mean_diff_ra_2*3600)+" arcseconds")

#Declination
plt.plot(JD_time_a,diff_ra,'ro',JD_time_a,diff_ra_2,'bx',JD_time_a,diff_ra_3,'r+')



plt.show()

###########################################################




