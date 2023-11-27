import numpy as np
import pandas as pd
import heartpy as hp
import matplotlib.pyplot as plt
from scipy import signal
import sys
from datetime import datetime
# Automatic Gain Control=Checked,IR PA (mA)=10,Red PA (mA)=10,IR LED Range (mA)=51,Red LED Range (mA)=51,ALC + FDM=Checked,
# Sample Rate (Hz)=100,Pulse Width (usec)=400,ADC Range (nA)=32768,FIFO Rolls on Full=Checked,FIFO Almost Full=17,Sample 
# Averaging=1,IA Gain=5,ECG Gain=8,Sample Rate=200,Adaptive Filter=Checked,Notch Freq=60,Cutoff Freq=50,

### Peak Finding Parameters ###
MIN_WIDTH = 10
MAX_WIDTH = 500
MIN_WIDTH_ECG_VAL = 0
MAX_WIDTH_ECG_VAL = 25
MIN_WIDTH_ECG_PEAK = 0
MAX_WIDTH_ECG_PEAK = 50
PROMINENCE_IR = 1000
PROMINENCE_RED = 500
PROMINENCE_ECG = 800
SAMPLERATE_ECG = 200
HEIGHT = None
THRESHOLD = None

### SpO2 Parameters ###
a = -16.666666 # SpO2 = aR^2 + bR + c
b = 8.333333
c = 100

# READ IN FILE
# CSV # 1
# df = pd.read_csv("C:\data\honors project ppg data\ECPPG_2023-10-21_21-29-57.csv") #Karston
#df = pd.read_csv("C:\\Users\pazul\Documents\BMEN 207\Honors Project\Coding\data\ECPPG_2023-10-21_21-29-57.csv") #Pablo

# CSV # 2
# df = pd.read_csv("C:\data\honors project ppg data\ECPPG_2023-10-21_23-45-37.csv") #Karston
#df = pd.read_csv("C:\\Users\pazul\Documents\BMEN 207\Honors Project\Coding\data\ECPPG_2023-10-21_23-45-37.csv") # Pablo

# CSV # 3
# df = pd.read_csv("C:\data\honors project ppg data\ECPPG_2023-10-21.csv") #Karston
#df = pd.read_csv("C:\\Users\pazul\Documents\BMEN 207\Honors Project\Coding\data\ECPPG_2023-10-21.csv") # Pablo

# CSV # 4
# df = pd.read_csv("C:\data\honors project ppg data\ECPPG_2023-11-07_12-17-55.csv") #Karston
# #df = pd.read_csv("C:\\Users\pazul\Documents\BMEN 207\Honors Project\Coding\data\ECPPG_2023-11-07_12-17-55.csv") # Pablo
# df.drop(df.head(1000).index, axis=0, inplace=True)
# df.drop(df.tail(1000).index, axis=0, inplace=True)
# df.reset_index(inplace=True)
# df.head

# CSV 5
#df = pd.read_csv("C:\data\honors project ppg data\ECPPG_2023-11-10_13-32-13.csv") #Karston
df = pd.read_csv("/home/pablo/projects/ECG-PPG/Coding/data/ECPPG_2023-11-10_13-32-13.csv") # Pablo - Linux

# print(df.head)


# =============================================================================
# def smooth_data(dataset, analysis_type):
#     """Takes in a dataset and smoothes the data for
#     analysis.
# 
#     Args:
#         dataset (DataFrame): a dataframe containing the necessary data
#         analysis_type (string): what data to smooth down
# 
#     Returns:
#         Array: the filtered signal
#     """
#     #read in data
#     if analysis_type.lower() == 'ir':
#         signal = dataset[' IR Count']
#     elif analysis_type.lower() == 'red':
#         signal = dataset[' Red Count']
#     else:
#         sys.exit("Wrong input for analysis_type. Accepted values are 'ir' or 'red'.")
#     
#     
#     time = dataset['Time']
# 
#     #get sample rate
#     sample_rate = hp.get_samplerate_datetime(time, timeformat = '%Y-%m-%d %H:%M:%S.%f')
#     #print('sampling rate is: %.3f Hz' %sample_rate)
# 
#     #find seconds elapsed (ALL OF THIS TIME STUFF IS FOR THE PLOT SO WE DON'T REALLY NEED IT! It might be a faster way to do convert_to_sec though!)
#     start = datetime.strptime(time[0], '%Y-%m-%d %H:%M:%S.%f')
#     start_timedelta = start - datetime(1900, 1, 1)
#     start_seconds = start_timedelta.total_seconds()
# 
#     end = datetime.strptime(time[time.index[-1]], '%Y-%m-%d %H:%M:%S.%f')
#     end_timedelta = end - datetime(1900, 1, 1)
#     end_seconds = end_timedelta.total_seconds()
# 
#     total_time = end_seconds - start_seconds
#     print(f'Time elapsed: {total_time} s')
# 
#     #filter signal using bandpass
#     filtered = hp.filter_signal(signal, [0.5, 4], sample_rate=sample_rate, order=3, filtertype='bandpass')
#     
#     #demonstrate effectiveness
#     plt.figure(figsize=(12,8))
#     plt.subplot(211)
#     plt.plot(signal[0:int(total_time * sample_rate)])
#     plt.title(f'Original {analysis_type} data')
#     plt.subplot(212)
#     plt.plot(filtered[0:int(total_time * sample_rate)])
#     plt.title(f'Filtered {analysis_type} data')
#     plt.show()
# 
#     print(f'Results from turning original {analysis_type} count signal of length {len(signal)} into filtered IR count signal of {len(filtered)}')
#     return filtered
# =============================================================================

def read_file(df,drop):
    time = np.array(df['Time'][drop:],dtype='datetime64')
    samplecount = np.array(df[' Sample Count'][drop:])
    ir_count = np.array(df[' IR Count'][drop:])
    red_count = np.array(df[' Red Count'][drop:])
    ecg_raw = np.array(df[' Raw ECG'][drop:])
    ecg_raw_mv = np.array(df[' Raw ECG (mV)'][drop:])
    ecg_filtered = np.array(df[' Filtered ECG'][drop:])
    ecg_filtered_mv = np.array([' Filtered ECG (mV)'][drop:])
    
    return time, samplecount, ir_count, red_count, ecg_raw, ecg_raw_mv, ecg_filtered, ecg_filtered_mv

def convert_to_sec(time):
    timelist_ms = np.array(time-time[0],dtype='float')
    timelist_s = timelist_ms/1000
    return timelist_ms, timelist_s

def ecg_peak_removal(ecg_peak_loc, ecg_val_loc):
    """ Checks for indices where there are either 0 or >1 peaks in between valleys, then returns those indices in a list.

    Parameters
    ----------
    ir_peak_loc : ARRAY
        DESCRIPTION.
    ir_val_loc : ARRAY
        DESCRIPTION.

    Returns
    -------
    ARRAY
        Array of indices where the IR valley difference calculation should be thrown out.

    """
    cut_indices = []
    for i in range(len(ecg_val_loc)-1):
        #return array of indices where ir_peak_loc has a value that is between ir_val_loc[i] and ir_val_loc[i+1]
        checker = np.where(np.logical_and(ecg_peak_loc >= ecg_val_loc[i], ecg_peak_loc <= ecg_val_loc[i+1]))
        if len(checker[0]) > 1 or len(checker[0]) == 0:
            cut_indices += [int(i)]
    # print(cut_indices)
    # print(ir_val_loc)
    return np.array(cut_indices)

def ecg_peak_finder(ecgdata):
    """

    Parameters
    ----------
    ecgdata : ARRAY
        ECG Data for peak finding.

    Returns
    -------
    ecg_peak_loc : ARRAY
        Indices of ECG peaks.
    ecg_val_loc : ARRAY
        Indices of ECG valleys.

    """
    
    # data = pd.read_csv(f'C:\\data\PPG_data-1\Data_{file_num}.csv')[8:] #read in file but drop the first 8 rows
    # red_count = np.asfarray(data[' IR (L)'])
    # ir_count = np.asfarray(data[' Red (L)'])
    
    ecg_peak_loc, _ = signal.find_peaks(ecgdata,HEIGHT,THRESHOLD,prominence=PROMINENCE_ECG,width=(MIN_WIDTH_ECG_PEAK, MAX_WIDTH_ECG_PEAK))
    ecg_val_loc, _ = signal.find_peaks(-ecgdata,HEIGHT,THRESHOLD,prominence=PROMINENCE_ECG,width=(MIN_WIDTH_ECG_VAL, MAX_WIDTH_ECG_VAL))
    
    # Plot ECG Signal
    fig1, (ax1) = plt.subplots(1)
    fig1.tight_layout(pad=3)
    ax1.plot(ecg_filtered,'-',ms=5,label='ECG Signal')
    ax1.plot(ecg_peak_loc,ecgdata[ecg_peak_loc],'.',ms=10,label='ECG Peaks') #x,y = index, y-value at index, do the same w trough
    ax1.plot(ecg_val_loc,ecgdata[ecg_val_loc],'.',ms=10,label='ECG Valleys')
    
    # Plot where bad data is cut
    cut_indices = np.asfarray(ecg_peak_removal(ecg_peak_loc,ecg_val_loc))
    if len(cut_indices) > 0:
        xvals = []
        yvals = []
        for i in cut_indices:
            xvals += [ecg_val_loc[int(i)]]
            yvals += [ecg_filtered[ecg_val_loc[int(i)]]]
        ax1.plot(xvals,yvals,'x',ms=10,label='Cut Valleys',color='red')
    
    #Pretty graph
    ax1.legend(loc='upper right')
    ax1.title.set_text(f'ECG count')
    ax1.set_xlabel(f'Sample - {SAMPLERATE_ECG} samples/sec')
    ax1.set_ylabel('ECG')

    
    return ecg_peak_loc, ecg_val_loc

def heartrate(ecg_peak_loc,ecg_val_loc,SAMPLERATE_ECG):
    #calculate IR bpm and stdev
    ecg_diffs = np.diff(ecg_val_loc)
    ecg_timediff = ecg_diffs/SAMPLERATE_ECG
    ecg_bpms = 60*(1/ecg_timediff)
    ecg_avg_bpm = np.mean(ecg_bpms)
    ecg_stdev_bpm = np.std(ecg_bpms)
    
    #calculate bpm2
    ecg_diffs_cut = ecg_peak_removal(ecg_peak_loc,ecg_val_loc)
    if ecg_diffs_cut.size > 0:
        ecg_diffs_2 = np.delete(ecg_diffs,ecg_diffs_cut) #delete invalid time calculations by invalid trough positions
        ecg_timediff_2 = ecg_diffs_2/SAMPLERATE_ECG
        ecg_bpms_2 = 60*(1/ecg_timediff_2)
        ecg_avg_bpm_2 = np.mean(ecg_bpms_2)
        ecg_stdev_bpm_2 = np.std(ecg_bpms_2)
    else: #no indices thrown out
        ecg_avg_bpm_2 = ecg_avg_bpm
        ecg_stdev_bpm_2 = ecg_stdev_bpm

    return ecg_avg_bpm_2, ecg_stdev_bpm_2, ecg_avg_bpm, ecg_stdev_bpm, ecg_diffs

time, samplecount, ir_count, red_count, ecg_raw, ecg_raw_mv, ecg_filtered, ecg_filtered_mv = read_file(df,100)
ecg_peak_loc, ecg_val_loc = ecg_peak_finder(ecg_filtered)
ecg_avg_bpm_2, ecg_stdev_bpm_2, ecg_avg_bpm, ecg_stdev_bpm, ecg_diffs = heartrate(ecg_peak_loc,ecg_val_loc,SAMPLERATE_ECG)

print(f'average ecg bpm: {ecg_avg_bpm}')
print(f'ecg bpm stdev: {ecg_stdev_bpm}')
print(f'average ecg bpm2: {ecg_avg_bpm_2}')
print(f'ecg bpm2 stdev: {ecg_stdev_bpm_2}')