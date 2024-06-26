import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os



def csv2dataframe(csv_file):
    
    df = pd.read_csv(csv_file)
    
    return df


def plot_lightcurve(df_list, label_list=[], color=[], title_text='',
                    tspace_flag=False, twinx_flag=False):
    
    UTC_list = []
    SNR_list = []
    JD0 = 1e12
    for df in df_list:
        UTC_JD = df['J.D.-2400000'].tolist()
        SNR = df['Source_SNR_T1'].tolist()
        
        if UTC_JD[0] < JD0:
            JD0 = UTC_JD[0]
            
        if tspace_flag:
            t_diff = np.diff(UTC_JD)
            nonzeros = np.where(t_diff > 0.)[0]
            nonzeros_diff = np.diff(nonzeros)
            count_inds = np.where(nonzeros_diff > 1)[0]
            
            if nonzeros[0] > 0:
                dt = t_diff[nonzeros[0]]/(nonzeros[0]+1)
                for ii in range(nonzeros[0]+1):
                    t_diff[ii] = dt
            
            for ind in count_inds:
                count = nonzeros_diff[ind]
                
                tind0 = nonzeros[ind]
                tind1 = nonzeros[ind+1]
                dt = t_diff[tind1]/count
                for ii in range(tind0+1, tind1+1):
                    t_diff[ii] = dt
            
            UTC_JD1 = np.zeros(len(UTC_JD),)
            UTC_JD1[0] = UTC_JD[0]
            for ii in range(len(t_diff)):
                UTC_JD1[ii+1] = UTC_JD1[ii] + t_diff[ii]
            
            print(UTC_JD1)
            UTC_JD = UTC_JD1            
        
        UTC_list.append(UTC_JD)
        SNR_list.append(SNR)
        
#    color=iter(plt.cm.rainbow(np.linspace(0,1,len(UTC_list))))
#    if len(color) == 0:
#        color = iter(['b', 'r'])
    plt.figure()
    for ii in range(len(UTC_list)):
        
        UTC_JD = UTC_list[ii]
        SNR = SNR_list[ii]
        t_sec = [(UTC_JD[jj] - JD0)*86400. for jj in range(len(UTC_JD))]
        c = next(color)
        if len(label_list) > 0:
            plt.plot(t_sec, SNR, 'o--', c=c, label=label_list[ii])
        else:
            plt.plot(t_sec, SNR, 'o--', c=c)
            
    plt.xlabel('Time [sec]')
    plt.ylabel('SNR')
    plt.title(title_text)
    if len(label_list) > 0:
        plt.legend()
   
    if len(label_list) == 2 and twinx_flag:
        color = iter(['b', 'r'])
        fig, ax1 = plt.subplots()

        UTC_JD = UTC_list[0]
        SNR = SNR_list[0]
        t_sec = [(UTC_JD[jj] - JD0)*86400. for jj in range(len(UTC_JD))]
        c = next(color)

        ax1.set_xlabel('Time [sec]')
        ax1.set_ylabel(label_list[0] + ' SNR', color=c)
        ax1.plot(t_sec, SNR, 'o--', c=c)
        ax1.tick_params(axis='y', labelcolor=c)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        UTC_JD = UTC_list[1]
        SNR = SNR_list[1]
        t_sec = [(UTC_JD[jj] - JD0)*86400. for jj in range(len(UTC_JD))]
        c = next(color)
        ax2.set_ylabel(label_list[1] + ' SNR', color=c)  # we already handled the x-label with ax1
        ax2.plot(t_sec, SNR, 'o--', c=c)
        ax2.tick_params(axis='y', labelcolor=c)
        
#        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        plt.title(title_text)

    
    return t_sec, SNR


def generate_plots():
    
#    # Load all data
#    data_dir = Path('C:/Users/z3523941/Documents/research/launch_ident/data/'
#                    'processed_data')
#    
#    
#  
#    data_file = data_dir / '43690_2018_11_11_viewfinder.csv'
#    RL2_43690_20181111 = csv2dataframe(data_file)
#    
#    data_file = data_dir / '43691_2018_11_25_viewfinder.csv'
#    RL2_43691_20181125 = csv2dataframe(data_file)
#    
#    data_file = data_dir / '43692_2018_11_24_main.csv'
#    NABEO_43692_20181124m = csv2dataframe(data_file)
#    
#    data_file = data_dir / '43692_2018_11_24_viewfinder.csv'
#    NABEO_43692_20181124v = csv2dataframe(data_file)
#    
#    data_file = data_dir / '43692_2018_11_25_main.csv'
#    NABEO_43692_20181125m = csv2dataframe(data_file)
#    
#    data_file = data_dir / '43692_2018_11_25_viewfinder.csv'
#    NABEO_43692_20181125v = csv2dataframe(data_file)
#    
#    data_file = data_dir / '43693_2018_11_24_main.csv'
#    IRVINE_43693_20181124 = csv2dataframe(data_file)
#    
#    data_file = data_dir / '43694_2018_11_24_main.csv'
#    PROXIMA1_43694_20181124m = csv2dataframe(data_file)
#    
#    data_file = data_dir / '43694_2018_11_24_viewfinder.csv'
#    PROXIMA1_43694_20181124v = csv2dataframe(data_file)
#    
#    data_file = data_dir / '43697_2018_11_25_main.csv'
#    LEMUR_43697_20181125 = csv2dataframe(data_file)
#    
#    
#    
#    # Generate Plots
#    
#    df_list = [RL2_43690_20181111]
#    label_list = ['View']
#    color = iter(['r'])
#    plot_lightcurve(df_list, label_list, color, 'NORAD 43690 2018-11-11 RL Stage2')
#    
#    df_list = [RL2_43691_20181125]
#    label_list = ['View']
#    color = iter(['r'])
#    plot_lightcurve(df_list, label_list, color, 'NORAD 43691 2018-11-25 RL Stage2')
#    
#    df_list = [NABEO_43692_20181124m, NABEO_43692_20181124v]
#    label_list = ['Main', 'View']
#    color = iter(['b','r'])
#    plot_lightcurve(df_list, label_list, color, 'NORAD 43692 2018-11-24 Kickstage/NABEO')
#    
#    df_list = [NABEO_43692_20181125m, NABEO_43692_20181125v]
#    label_list = ['Main', 'View']
#    color = iter(['b','r'])
#    plot_lightcurve(df_list, label_list, color, 'NORAD 43692 2018-11-25 Kickstage/NABEO')
#    
#    df_list = [IRVINE_43693_20181124]
#    label_list = ['Main']
#    color = iter(['b'])
#    plot_lightcurve(df_list, label_list, color, 'NORAD 43693 2018-11-24 IRVINE01')
#    
#    df_list = [PROXIMA1_43694_20181124m, PROXIMA1_43694_20181124v]
#    label_list = ['Main', 'View']
#    color = iter(['b','r'])
#    plot_lightcurve(df_list, label_list, color, 'NORAD 43694 2018-11-24 PROXIMA1')
#    
#    df_list = [LEMUR_43697_20181125]
#    label_list = ['Main']
#    color = iter(['b'])
#    plot_lightcurve(df_list, label_list, color, 'NORAD 43697 2018-11-25 LEMUR')
    

    
    
#    data_dir = Path('D:/documents/research/sensor_management/reports/2018_asrc_ROO/data')
#    
#    data_file = data_dir / 'Sloshsat_ROO_pass1_processed.csv'
#    ROO_sloshsat_pass1 = csv2dataframe(data_file)
#    
#    data_file = data_dir / 'FalconSloshsatPass1.csv'
#    Falcon_sloshsat_pass1 = csv2dataframe(data_file)
#    
#    df_list = [Falcon_sloshsat_pass1, ROO_sloshsat_pass1]
#    label_list = ['Falcon', 'ROO']
#    color = iter(['b', 'r'])
#    plot_lightcurve(df_list, label_list, color, twinx_flag=True)
    
    
#    data_dir = Path('D:\documents\\research\launch_identification\\reports\\2019_iac_NABEO\data')
#    
#    data_file = data_dir / '43164_2018_11_30_main.csv'
#    st_main = csv2dataframe(data_file)
#    
#    data_file = data_dir / '43164_2018_11_30_viewfinder.csv'
#    st_view = csv2dataframe(data_file)
#    
#    df_list = [st_view]
#    label_list = []
#    color = iter(['k'])
#    st_sec, st_SNR = plot_lightcurve(df_list, label_list, color)
#    
#    
#    data_file = data_dir / '43692_2018_11_25_main.csv'
#    nabeo_main = csv2dataframe(data_file)
#    
#    data_file = data_dir / '43692_2018_11_25_viewfinder.csv'
#    nabeo_view = csv2dataframe(data_file)
#    
#    df_list = [nabeo_view]
#    label_list = []
#    color = iter(['k'])
#    nabeo_sec, nabeo_SNR = plot_lightcurve(df_list, label_list, color)
#    
#    
#    fig = plt.figure()
#    ax1 = fig.add_subplot(2,1,1)
#    ax1.plot(nabeo_sec, nabeo_SNR, 'ko--')
#    ax1.set_ylim([0, 700])
#    ax1.set_xlim([0, 225])
#    ax1.set_ylabel('NABEO SNR')
#    
#    ax2 = fig.add_subplot(2,1,2)
#    ax2.plot(st_sec, st_SNR, 'ko--')
#    ax2.set_ylim([0, 700])
#    ax2.set_xlim([0, 225])
#    ax2.set_ylabel('ST SNR')
#    ax2.set_xlabel('Time [sec]')
#    
#    plt.setp(ax1.get_xticklabels(), visible=False)
#    
#    
#    data_file = data_dir / '43692_2019_09_23_main.csv'
#    nabeo_main = csv2dataframe(data_file)
#    
#    data_file = data_dir / '43692_2019_09_23_view.csv'
#    nabeo_view = csv2dataframe(data_file)
#    
#    df_list = [nabeo_view]
#    label_list = []
#    color = iter(['k'])
#    nabeo_sec, nabeo_SNR = plot_lightcurve(df_list, label_list, color)
#    
#    
#    data_file = data_dir / '44372_2019_09_23_pass1_main.csv'
#    mir_main = csv2dataframe(data_file)
#    
#    data_file = data_dir / '44372_2019_09_23_pass1_view.csv'
#    mir_view = csv2dataframe(data_file)
#    
#    df_list = [mir_view]
#    label_list = []
#    color = iter(['k'])
#    mir_sec, mir_SNR = plot_lightcurve(df_list, label_list, color)
#    
#    
#    
#    fig = plt.figure()
#    ax1 = fig.add_subplot(2,1,1)
#    ax1.plot(np.asarray(nabeo_sec[45:-14])-nabeo_sec[45], nabeo_SNR[45:-14], 'ko--')
##    ax1.set_ylim([0, 1000])
#    ax1.set_xlim([0, 160])
#    ax1.set_ylabel('NABEO SNR')
#    
#    ax2 = fig.add_subplot(2,1,2)
#    ax2.plot(mir_sec, mir_SNR, 'ko--')
##    ax2.set_ylim([0, 1000])
#    ax2.set_xlim([0, 160])
#    ax2.set_ylabel('MIR SNR')
#    ax2.set_xlabel('Time [sec]')
#    
#    plt.setp(ax1.get_xticklabels(), visible=False)
#    
#    
#    data_file = data_dir / '44372_2019_09_23_pass2_main.csv'
#    mir_main = csv2dataframe(data_file)
#    
#    data_file = data_dir / '44372_2019_09_23_pass2_view.csv'
#    mir_view = csv2dataframe(data_file)
#    
#    df_list = [mir_view]
#    label_list = []
#    color = iter(['k'])
#    mir_sec, mir_SNR = plot_lightcurve(df_list, label_list, color)
    

    # Load all data
    data_dir = 'D:\documents\\research\cubesats\M2\data'
  
#    data_file = os.path.join(data_dir, 'CBR_2021_09_03_main.csv')
#    CBR_main = csv2dataframe(data_file)
    
    data_file = os.path.join(data_dir, 'CMU_2021_09_07_view.csv')
    CBR_view = csv2dataframe(data_file)
    
#    df_list = [CBR_main, CBR_view]
#    label_list = ['Main', 'View']
#    color = iter(['b', 'r'])
    
    df_list = [CBR_view]
    label_list = ['View']
    color = iter(['r'])
    plot_lightcurve(df_list, label_list, color, 'NORAD 47967 2021-09-07 CMU-Falcon')
    
    return


if __name__ == '__main__':
    
    plt.close('all')
    
    generate_plots()
    
    plt.show()
    