import numpy as np
from datetime import datetime, timedelta
import os
import pandas as pd



def read_contact_file(fname):
    '''
    This function reads a GMAT contact textfile and outputs a dictionary
    containing start/stop times and durations for each observer
    
    Parameters
    ------
    fname : string
        file path and name of GMAT contact data in .txt
        
    Returns
    ------
    observer_dict : dictionary
        observer data organized by name of contact, contains lists of 
        start/stop times as datetime objects and durations in seconds
    
    '''
    
    # Initialize output
    observer_dict = {}    
    
    # Open file and retrieve all lines
    file_object = open(fname, 'r')
    data = file_object.readlines()    
    file_object.close()
    
    # Parse lines one at a time
    for ii in range(len(data)):
        
        line = data[ii]
        
        # If a new observer is present, generate new dictionary entry
        if line[0:8] == 'Observer':

            # Retrieve observer name, strip new line character
            observer = line[10:-1]
            
            # Add to dictionary
            observer_dict[observer] = {}
            observer_dict[observer]['start_list'] = []
            observer_dict[observer]['stop_list'] = []
            observer_dict[observer]['duration_list'] = []
                
            # Skip next line, then add all times to lists
            ii += 2
            while True:                
                line2 = data[ii]
                try:
                    start_dt = datetime.strptime(line2[0:24], '%d %b %Y %H:%M:%S.%f')
                    stop_dt = datetime.strptime(line2[28:52], '%d %b %Y %H:%M:%S.%f')
                    duration = float(line2[58:])
                    
                    observer_dict[observer]['start_list'].append(start_dt)
                    observer_dict[observer]['stop_list'].append(stop_dt)
                    observer_dict[observer]['duration_list'].append(duration)
                    
                    ii += 1
                    
                # If this line is not a start/stop time except loop and 
                # continue to next observer
                except:
                    break

    
    return observer_dict


def process_radio_contact(observer_dict, start_date, stop_date):
    '''
    This function processes the observer dictionary to generate data for report
    
    Parameters
    ------
    observer_dict : dictionary
        observer data organized by name of contact, contains lists of 
        start/stop times as datetime objects and durations in seconds
    start_date : datetime object
        initial date and time for radio contact evaluation
    stop_date : datetime object
        final date and time for radio contact evaluation
        
    Returns
    ------
    
    
    '''
    
    # Calculate total contact time each day
    ndays = int(np.ceil((stop_date - start_date).total_seconds()/86400.))
    
    # Loop over all days and compute total duration in contact
    comms_per_day = [0.]*ndays
    day_list = []
    for ii in range(ndays):
        
        current_day = start_date + timedelta(days=ii)
        next_day = current_day + timedelta(days=1)
        day_list.append(current_day)
        
        for observer in observer_dict:
            start_list = observer_dict[observer]['start_list']
            stop_list = observer_dict[observer]['stop_list']
            duration_list = observer_dict[observer]['duration_list']
            
            for jj in range(len(start_list)):
                

                # Comms pass completely contained in the day of interest
                if start_list[jj] > current_day and start_list[jj] < next_day and stop_list[jj] < next_day:
                    comms_per_day[ii] += duration_list[jj]
                    
                # Comms pass continues into the next day
                elif start_list[jj] > current_day and start_list[jj] < next_day and stop_list[jj] > next_day:
                    comms_per_day[ii] += (next_day - start_list[jj]).total_seconds()
                    
                # Comms pass continues from previous day
                elif ii > 0 and start_list[jj] < current_day and stop_list[jj] > current_day and stop_list[jj] < next_day:
                    comms_per_day[ii] += (stop_list[jj] - current_day).total_seconds()
                    
                    
#    print(day_list)
#    print(comms_per_day)
    
    # Generate output
    comms_per_day_min = [ti/60. for ti in comms_per_day]
    comms_per_day_flag = [i for i,v in enumerate(comms_per_day_min) if v < 20.]
    
    if len(comms_per_day_flag) == 0:
        print('\n\nCommunications Requirement Passed!!')
    else:
        print('\n\nCommunications Requirement Failed!!')
        for ii in comms_per_day_flag:
            print('Date: ', day_list[ii], ' Comms Time [min]: ', comms_per_day_min[ii])
    
    
    df = pd.DataFrame(list(zip(day_list, comms_per_day_min)), columns=['UTC Date', 'Comms Time [min]']) 
    
    
    
    return df


def process_ground_obs(obs_dict):
    
    UTC_offset_dict = define_utc_offsets()
    
    output_list = []
    summary_output = []
    total_points = 0.
    for observer in obs_dict:
        UTC_offset = UTC_offset_dict[observer]
        start_list = obs_dict[observer]['start_list']
        stop_list = obs_dict[observer]['stop_list']
        
        night_obs = 0.
        day_obs= 0.
        bonus_obs = 0.
        points_sum = 0.
        for ii in range(len(start_list)):
            local_start = start_list[ii] + timedelta(hours=UTC_offset)
            local_stop = stop_list[ii] + timedelta(hours=UTC_offset)
            
            points = 0.
            night_obs += 1.
            if local_stop.hour >= 6 and local_start.hour < 18.:
                points += 1.
                day_obs += 1.
                night_obs -= 1.
            if local_stop.hour >= 10 and local_start.hour < 12.:
                points += 1.
                bonus_obs += 1.
                
            points_sum += points
            total_points += points
            output_list.append([observer, local_start, local_stop, points])
            
        summary_output.append([observer, night_obs, day_obs, bonus_obs, points_sum])

    
    full_df = pd.DataFrame(output_list, columns=['Observer', 'Local Start', 'Local Stop', 'Points'])
    summary_df = pd.DataFrame(summary_output, columns=['Observer', 'Night Obs', 'Day Obs', 'Bonus Obs', 'Points'])
    
    print('\n\nTotal Points Scored: ', total_points)
            
    
    return full_df, summary_df


def define_utc_offsets():
    
    UTC_offset_dict = {}
    UTC_offset_dict['FGISjokulla'] = 3.
    UTC_offset_dict['Mali'] = 0.
    UTC_offset_dict['NamibDesert'] = 2.
    UTC_offset_dict['Shadnagar'] = 5.5
    UTC_offset_dict['TingaTingana'] = 9.5
    UTC_offset_dict['UyuniSaltFlats'] = -4.
    UTC_offset_dict['WhiteSands'] = -6.
    
    
    return UTC_offset_dict


def large_satellite_analysis():
    
    
    datadir = r'D:\documents\teaching\unsw_orbital_mechanics\2022\lab\large_satellite\final'
    fname = os.path.join(datadir, 'CanberraComms.txt')
    
    radio_dict = read_contact_file(fname)
    start_date = datetime(2022, 3, 1)
    stop_date = datetime(2022, 4, 1)
    radio_df = process_radio_contact(radio_dict, start_date, stop_date)
    
    fname = os.path.join(datadir, 'Radio_Comms.csv')
    radio_df.to_csv(fname)
    
    
    fname = os.path.join(datadir, 'GroundObs.txt')
    obs_dict = read_contact_file(fname)    
    full_obs_df, summary_obs_df = process_ground_obs(obs_dict)
    
    fname = os.path.join(datadir, 'Ground_Obs_Points_Breakdown.csv')
    full_obs_df.to_csv(fname)
    
    fname = os.path.join(datadir, 'Ground_Obs_Summary.csv')
    summary_obs_df.to_csv(fname)
    
    
    return


def small_satellite_analysis():
    
    datadir = r'D:\documents\teaching\unsw_orbital_mechanics\2022\lab\small_satellite'
    start_date = datetime(2022, 3, 1)
    stop_date = datetime(2022, 4, 1)
    
    fname = os.path.join(datadir, 'CanberraComms1.txt')    
    radio_dict1 = read_contact_file(fname)
    radio_df1 = process_radio_contact(radio_dict1, start_date, stop_date)
    fname = os.path.join(datadir, 'Radio_Comms1.csv')
    radio_df1.to_csv(fname)
    
    fname = os.path.join(datadir, 'CanberraComms2.txt')    
    radio_dict2 = read_contact_file(fname)
    radio_df2 = process_radio_contact(radio_dict2, start_date, stop_date)
    fname = os.path.join(datadir, 'Radio_Comms2.csv')
    radio_df2.to_csv(fname)
    
    fname = os.path.join(datadir, 'CanberraComms3.txt')    
    radio_dict3 = read_contact_file(fname)
    radio_df3 = process_radio_contact(radio_dict3, start_date, stop_date)
    fname = os.path.join(datadir, 'Radio_Comms3.csv')
    radio_df3.to_csv(fname)
    
    fname = os.path.join(datadir, 'CanberraComms4.txt')    
    radio_dict4 = read_contact_file(fname)
    radio_df4 = process_radio_contact(radio_dict4, start_date, stop_date)
    fname = os.path.join(datadir, 'Radio_Comms4.csv')
    radio_df4.to_csv(fname)
    

    
    
    total_points = 0.
    
    # GroundObs1
    fname = os.path.join(datadir, 'GroundObs1.txt')
    obs_dict = read_contact_file(fname)    
    full_obs_df, summary_obs_df = process_ground_obs(obs_dict)
    
    fname = os.path.join(datadir, 'GroundObs1_Points_Breakdown.csv')
    full_obs_df.to_csv(fname)
    
    fname = os.path.join(datadir, 'GroundObs1_Summary.csv')
    summary_obs_df.to_csv(fname)
    
    total_points += sum(summary_obs_df['Points'].tolist())
    

    
    
    # GroundObs2
    fname = os.path.join(datadir, 'GroundObs2.txt')
    obs_dict = read_contact_file(fname)    
    full_obs_df, summary_obs_df = process_ground_obs(obs_dict)
    
    fname = os.path.join(datadir, 'GroundObs2_Points_Breakdown.csv')
    full_obs_df.to_csv(fname)
    
    fname = os.path.join(datadir, 'GroundObs2_Summary.csv')
    summary_obs_df.to_csv(fname)
    
    total_points += sum(summary_obs_df['Points'].tolist())



    # GroundObs3
    fname = os.path.join(datadir, 'GroundObs3.txt')
    obs_dict = read_contact_file(fname)    
    full_obs_df, summary_obs_df = process_ground_obs(obs_dict)
    
    fname = os.path.join(datadir, 'GroundObs3_Points_Breakdown.csv')
    full_obs_df.to_csv(fname)
    
    fname = os.path.join(datadir, 'GroundObs3_Summary.csv')
    summary_obs_df.to_csv(fname)
    
    total_points += sum(summary_obs_df['Points'].tolist())
    
    # GroundObs4
    fname = os.path.join(datadir, 'GroundObs4.txt')
    obs_dict = read_contact_file(fname)    
    full_obs_df, summary_obs_df = process_ground_obs(obs_dict)
    
    fname = os.path.join(datadir, 'GroundObs4_Points_Breakdown.csv')
    full_obs_df.to_csv(fname)
    
    fname = os.path.join(datadir, 'GroundObs4_Summary.csv')
    summary_obs_df.to_csv(fname)
    
    total_points += sum(summary_obs_df['Points'].tolist())
    
    print('Total Points for Constellation: ', total_points)
    
    
    return


if __name__ == '__main__':
    
#    large_satellite_analysis()
    
    small_satellite_analysis()
    
    
    
    
    
    