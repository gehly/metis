
from datetime import datetime, timedelta
import sys
import os

sys.path.append(r'C:\Users\Steve\Documents\code\metis')
os.chdir(r'C:\Users\Steve\Documents\code\metis\utilities')

from utilities.tle_functions import get_tle_range

tle_df = get_tle_range(username='',password='', start_norad='43689', stop_norad='81000')

os.chdir(r'D:\documents\research\launch_identification\data\tle_archive')

filename = datetime.now().strftime("%Y%m%d-%H%M%S")+'_tle_latest.csv'
tle_df.to_csv(filename)
