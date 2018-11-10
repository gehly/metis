'''
To use with Windows Task Scheduler:
    path for metis library must be input to sys.path.append() (line 10)
    path for metis\utilities must be input to os.chdir() (line 11)
    spacetrack username and password must be input to get_tle_range() (line 12)
'''
from datetime import datetime, timedelta
import sys
import os

sys.path.append(r'C:\Users\s3435051\metis')
os.chdir(r'C://Users\s3435051\metis\utilities')

from utilities.tle_functions import get_tle_range

tle_df = get_tle_range(username='',password='', start_norad='43689', stop_norad='81000')

filename = datetime.now().strftime("%Y%m%d-%H%M%S")+'_tle_latest.csv'
tle_df.to_csv(filename)
