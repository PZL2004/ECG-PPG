import numpy as np
import heartpy as hp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys
import time
# Automatic Gain Control=Checked,IR PA (mA)=10,Red PA (mA)=10,IR LED Range (mA)=51,Red LED Range (mA)=51,ALC + FDM=Checked,
# Sample Rate (Hz)=100,Pulse Width (usec)=400,ADC Range (nA)=32768,FIFO Rolls on Full=Checked,FIFO Almost Full=17,Sample 
# Averaging=1,IA Gain=5,ECG Gain=8,Sample Rate=200,Adaptive Filter=Checked,Notch Freq=60,Cutoff Freq=50,

#start compute time
start = time.time()

##### INITIAL PARAMETERS #####
### Peak Finding ###
MIN_WIDTH = 20
MAX_WIDTH = 500
MIN_WIDTH_ECG_VAL = 0
MAX_WIDTH_ECG_VAL = 25
MIN_WIDTH_ECG_PEAK = 0
MAX_WIDTH_ECG_PEAK = 50
PROMINENCE_IR_PEAK = 500
PROMINENCE_RED_PEAK = 350
PROMINENCE_IR_VAL = 500
PROMINENCE_RED_VAL = 250
PROMINENCE_IR_SMOOTHED = 500
PROMINENCE_RED_SMOOTHED = 200
PROMINENCE_ECG = 800
SAMPLERATE_PPG = 100
SAMPLERATE_ECG = 200
THRESHOLD_SMOOTHED = 0
HEIGHT = None
THRESHOLD = None

### SpO2 Parameters ###
a = -16.666666 # SpO2 = aR^2 + bR + c
b = 8.333333
c = 100 # goofy values from Maxim

#DIAG
compute_moving_average = 0
plot = 0
important_plot = 1

##### FUNCTIONS #####

def convert_to_sec(t):
    """converts the datetime format to seconds using numpy

    Args:
        t (array): array of datetime format

    Returns:
        array: time points in seconds since start
    """
    timelist_ms = np.array(t - t[0], dtype=float)
    timelist_s = timelist_ms/1000
    return timelist_s

def read_file(df,drop):
    """reads in the df and outputs numpy arrays

    Args:
        df (DataFrame): dataframe from csv file
        drop (int): amount of rows to be dropped at start

    Returns:
        tuple: arrays of valuable data
    """
    time_ = convert_to_sec(np.array(df['Time'][drop:], dtype='datetime64'))
    samplecount = np.array(df[' Sample Count'][drop:])
    ir_count = np.array(df[' IR Count'][drop:])
    red_count = np.array(df[' Red Count'][drop:])
    ecg_raw = np.array(df[' Raw ECG'][drop:])
    ecg_raw_mv = np.array(df[' Raw ECG (mV)'][drop:])
    ecg_filtered = np.array(df[' Filtered ECG'][drop:])
    ecg_filtered_mv = np.array([' Filtered ECG (mV)'][drop:])
    
    return time_, samplecount, ir_count, red_count, ecg_raw, ecg_raw_mv, ecg_filtered, ecg_filtered_mv

def smooth_data(signal, analysis_type):
    """Takes in a dataset and smoothes the data for
    analysis.

    Args:
        dataset (DataFrame): a dataframe containing the necessary data
        analysis_type (string): what data to smooth down

    Returns:
        Array: the filtered signal
    """
    

    #filter signal using bandpass
    filtered = hp.filter_signal(signal, [0.5, 4], sample_rate=100, order=3, filtertype='bandpass')
    
    #demonstrate effectiveness
    if plot:
        plt.figure(figsize=(12,10))
        plt.subplot(211)
        plt.plot(signal)
        plt.xlabel("Samples (100 Hz)")
        plt.ylabel("Signal Count")
        plt.title(f'Original {analysis_type} data')
        plt.subplot(212)
        plt.plot(filtered)
        plt.xlabel("Samples (100 Hz)")
        plt.ylabel("Filtered Signal Count")
        plt.title(f'Filtered {analysis_type} data')
        plt.show()

    return filtered

def midpoint_finder(peaks, valleys):
    """finds the midpoints of a PPG peak

    Args:
        peaks (array): array of peak indices
        valleys (array): array of valley indices

    Returns:
        array: array of indices of midpoints
    """
    lcd = min([len(peaks),len(valleys)])

    if valleys[0] < peaks[0]: #if it starts with a valley
        if len(peaks[1:]) == len(valleys): #good / ending point is a peak
            midpoints = np.round(np.array(peaks[0:lcd])+np.array(valleys[0:lcd+1]))/2
        elif len(peaks[1:]) < len(valleys): #+1 valley / ending point is a valley
            midpoints = np.round(np.array(peaks)+np.array(valleys[:len(peaks[1:])+1]))/2
    elif valleys[0] > peaks[0]: #but if it starts with a peak we need to remove the first peak
        if len(peaks[1:]) == len(valleys): #good / ending point is a peak
            midpoints = np.round(np.array(peaks[1:])+np.array(valleys))/2
        elif len(peaks[1:]) < len(valleys): #+1 valley / ending point is a valley
            midpoints = np.round(np.array(peaks[1:])+np.array(valleys[:len(peaks[1:])]))/2
        elif len(peaks[1:]) > len(valleys): #+1 valley / ending point is a valley
            midpoints = np.round(np.array(peaks[1:lcd])+np.array(valleys[:len(peaks[1:lcd])]))/2

    # plt.plot(peaks,label='p')
    # plt.plot(valleys,label='v')
    # plt.plot(midpoints,label='mp')
    # plt.legend()
    # plt.show()
    return midpoints.astype(int)

def peak_valleys_PPG(data, time_, datatype, smoothed):
    """finds the peaks of a dataset and plots the data

    Args:
        data (list/list-like): the y-axis values of the dataset
        time_ (list/list-like): the x-axis time values of the dataset (should be same shape as data) 
        datatype (string): what dataset is being used
        smoothed (bool): if smoothed signal or not

    Returns:
        list: list containing the indexes of all of the peaks in the dataset (no minimum peaks, just maxes)
    """
    if datatype == 'IR':
        peaks = find_peaks(data, height=HEIGHT, threshold=THRESHOLD, prominence=PROMINENCE_IR_PEAK, width=(MIN_WIDTH, MAX_WIDTH))[0]
        valleys = find_peaks(-data, height=HEIGHT, threshold=THRESHOLD, prominence=PROMINENCE_IR_VAL, width=(MIN_WIDTH, MAX_WIDTH))[0]
    elif datatype == "Red":
        peaks = find_peaks(data, height=HEIGHT, threshold=THRESHOLD, prominence=PROMINENCE_RED_PEAK, width=(MIN_WIDTH, MAX_WIDTH))[0]
        valleys = find_peaks(-data, height=HEIGHT, threshold=THRESHOLD, prominence=PROMINENCE_RED_VAL, width=(MIN_WIDTH, MAX_WIDTH))[0]
    elif datatype == "Smoothed IR":
        peaks = find_peaks(data, height=HEIGHT, threshold=THRESHOLD_SMOOTHED, prominence=PROMINENCE_IR_SMOOTHED, width=(MIN_WIDTH, MAX_WIDTH))[0]
        valleys = find_peaks(-data, height=HEIGHT, threshold=THRESHOLD_SMOOTHED, prominence=PROMINENCE_IR_SMOOTHED, width=(MIN_WIDTH, MAX_WIDTH))[0]
    elif datatype == "Smoothed Red":
        peaks = find_peaks(data, height=HEIGHT, threshold=THRESHOLD_SMOOTHED, prominence=PROMINENCE_RED_SMOOTHED, width=(MIN_WIDTH, MAX_WIDTH))[0]
        valleys = find_peaks(-data, height=HEIGHT, threshold=THRESHOLD_SMOOTHED, prominence=PROMINENCE_RED_SMOOTHED, width=(MIN_WIDTH, MAX_WIDTH))[0]
    
    # fixes edge case where max of a peak/valley is found twice in same peak.
    usedValues_peaks = set()
    newList = []
    for v in peaks:
        if v not in usedValues_peaks:
            newList.append(v)
            for lv in range(v - MIN_WIDTH, v + MIN_WIDTH + 1):
                usedValues_peaks.add(lv)
    peaks = list(newList)
    
    usedValues_valleys = set()
    newList2 = []
    for v in valleys:
        if v not in usedValues_valleys:
            newList2.append(v)
            for lv in range(v - MIN_WIDTH, v + MIN_WIDTH + 1):
                usedValues_valleys.add(lv)
    valleys = list(newList2)
    
    if smoothed:
        midpoints = midpoint_finder(peaks, valleys)

    #plotting
    if important_plot:
        plt.figure(figsize=(12,4))
        plt.plot(time_, data, label = "Signal")
        plt.plot(time_[peaks], data[peaks], "x", label = "peaks")
        plt.plot(time_[valleys], data[valleys], "x", label="valleys")
        if smoothed:
            plt.plot(time_[midpoints], data[midpoints], "x", label="midpoints")
        plt.xlabel("Time (s)")
        plt.ylabel("Signal Count")
        plt.title(f"{datatype} Count vs Time")
        plt.legend()
        plt.show()
    return peaks, valleys

def SpO2(red_count, ir_count, red_peak_loc, red_val_loc, ir_peak_loc, ir_val_loc):
    """This function finds the average ratio of ratios and the average SpO2.

    Args:
        red_count (series or array): a series/array of values given for the red count
        ir_count (series or array): a series/array of values given for the IR count
        red_peak_loc (list/list-like): a list of the red count peak locations
        red_val_loc (list/list-like): a list of the red count valley/trough locations
        ir_peak_loc (list/list-like): a list of the IR count peak locations
        ir_val_loc (list/list-like): a list of the IR count valley/trough locations

    Returns:
        tuple: contains the average red and IR peak values, the average 
            red and IR valley values, the average ratio of ratios, and the average SpO2
    """
    red_peak_vals = red_count[red_peak_loc]
    red_val_vals = red_count[red_val_loc]
    ir_peak_vals = ir_count[ir_peak_loc]
    ir_val_vals = ir_count[ir_val_loc]
    
    avg_red_peak = np.mean(red_peak_vals)
    avg_red_val = np.mean(red_val_vals)
    avg_ir_peak = np.mean(ir_peak_vals)
    avg_ir_val = np.mean(ir_val_vals)
    
    avg_AC_red = np.mean(red_peak_vals) - np.mean(red_val_vals)
    avg_DC_red = np.mean(red_val_vals)
    
    avg_AC_ir = np.mean(ir_peak_vals) - np.mean(ir_val_vals)
    avg_DC_ir = np.mean(ir_val_vals) 
    
    avg_R = (avg_AC_red/avg_DC_red)/(avg_AC_ir/avg_DC_ir)
    
    avg_SpO2 = a*avg_R**2 + b*avg_R + c
    
    return avg_red_peak, avg_red_val, avg_ir_peak, avg_ir_val, avg_R, avg_SpO2

def BPM1(time_, peak_positions, trough_positions, compute_moving_average):
    """calculates the BPM for both the peaks and the troughs of the data in question.

    Args:
        time_ (ndarray): an array of time in seconds for each point in the data in question
        peak_positions (list/list-like): a list of the peak indices for the data in question
        trough_positions (list/list-like): a list of the trough indices for the data in question
        compute_moving average (bool): whether or not to compute a moving average (0 or 1)

    Returns:
        tuple: contains the average bpm for the peaks and troughs, as well as a moving average for both if wanted.
    """
    peak_times = list(set(np.round(time_[peak_positions], 3)))

    trough_times = list(set(np.round(time_[trough_positions], 3)))

    diffs_peaks = np.diff(peak_times)
    diffs_troughs = np.diff(trough_times)

    #average over the whole timeframe
    avg_bpm_peaks = 60/np.mean(diffs_peaks)
    avg_bpm_troughs = 60/np.mean(diffs_troughs)

    #moving average every 5 heartbeats
    if compute_moving_average:
        try:
            bpms_peaks = [(60/np.mean(diffs_peaks[i:i+5])) for i in range(len(diffs_peaks) - 5)]
            bpms_troughs = [(60/np.mean(diffs_troughs[i:i+5])) for i in range(len(diffs_troughs) - 5)]
        except IndexError:
            sys.exit("Use more than 5 heartbeats for your data. Preferably way more!")
    
        return avg_bpm_peaks, avg_bpm_troughs, bpms_peaks, bpms_troughs
    return avg_bpm_peaks, avg_bpm_troughs

def BPM2(time_, peak_positions, trough_positions, compute_moving_average):
    """calculates the BPM for the data points in which a single peak in in between
    two troughs. If there is not 1 peak in between, the data is thrown out for
    the heartrate calculation.

    Args:
        time_ (ndarray): an array of time in seconds for each point in the data in question
        peak_positions (list/list-like): a list of the peak indices for the data in question
        trough_positions (list/list-like): a list of the peak indices for the data in question
        compute_moving average (bool): whether or not to compute a moving average (0 or 1)

    Returns:
        tuple: contains the average bpm for the peaks and trough, as well as a moving average for both if wanted.
    """
    peak_positions = np.array(peak_positions)
    trough_positions = np.array(trough_positions)
    # initializing list for valid peak indices
    valid_peaks = []
    valid_troughs = []
    cut_indexes = []
    #looping through the trough positions
    for i in range(len(trough_positions) - 1):
        # Using np.where to find peaks between adjacent troughs by comparing
        # the peak position to the start and end trough for this iteration
        peaks_between = peak_positions[np.where((peak_positions > trough_positions[i]) & (peak_positions < trough_positions[i+1]))[0]]
        if len(peaks_between) == 1:
            # turning the list with length 1 into an integer and then appending
            valid_peaks.extend(peaks_between)
            valid_troughs.extend([trough_positions[i], trough_positions[i+1]])
        else:
            cut_indexes.append(i)
    
    valid_peaks = np.array(valid_peaks)
    valid_troughs = np.array(sorted(list(set(valid_troughs))))
    cut_indexes = np.array(cut_indexes)
    
    # if no indices are cut. FOR REALLY CLEAN DATA!!!
    if len(cut_indexes) == 0:
        # print("BPM2 same as BPM1") # diagnostic
        return BPM1(time_, peak_positions, trough_positions, compute_moving_average)
    
    all_peak_times = np.unique(np.round(time_[peak_positions], 3))
    all_trough_times = np.unique(np.round(time_[trough_positions], 3))
    
    all_diffs_peaks = np.diff(all_peak_times)
    all_diffs_troughs = np.diff(all_trough_times)
    
    # if value in cut_indexes is not a possible index in all_diffs, then delete them
    try:
        valid_diffs_peaks = np.delete(all_diffs_peaks, cut_indexes)
        valid_diffs_troughs = np.delete(all_diffs_troughs, cut_indexes)
    except:
        for i in reversed(range(len(cut_indexes))):
            if cut_indexes[i] not in range(len(all_diffs_peaks)) or cut_indexes[i] not in range(len(all_diffs_troughs)):
                cut_indexes = np.delete(cut_indexes, i)
        
        valid_diffs_peaks = np.delete(all_diffs_peaks, cut_indexes)
        valid_diffs_troughs = np.delete(all_diffs_troughs, cut_indexes)


    #average over the whole timeframe
    avg_bpm_peaks = 60/np.mean(valid_diffs_peaks)
    avg_bpm_troughs = 60/np.mean(valid_diffs_troughs)

    #moving average every 5 heartbeats
    
    if compute_moving_average:
        try:
            bpms_peaks = [(60/np.mean(valid_diffs_peaks[i:i+5])) for i in range(len(valid_diffs_peaks) - 5)]
            bpms_troughs = [(60/np.mean(valid_diffs_troughs[i:i+5])) for i in range(len(valid_diffs_troughs) - 5)]
        except IndexError:
            sys.exit("Use more than 5 heartbeats for your data. Preferably way more!")
    
        return avg_bpm_peaks, avg_bpm_troughs, bpms_peaks, bpms_troughs
    return avg_bpm_peaks, avg_bpm_troughs

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

def ecg_peak_finder(ecgdata, time_):
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
    
    ecg_peak_loc, _ = find_peaks(ecgdata,HEIGHT,THRESHOLD,prominence=PROMINENCE_ECG,width=(MIN_WIDTH_ECG_PEAK, MAX_WIDTH_ECG_PEAK))
    ecg_val_loc, _ = find_peaks(-ecgdata,HEIGHT,THRESHOLD,prominence=PROMINENCE_ECG,width=(MIN_WIDTH_ECG_VAL, MAX_WIDTH_ECG_VAL))
    
    # Plot ECG Signal
    if important_plot:
        fig1, (ax1) = plt.subplots(1)
        fig1.tight_layout(pad=3)
        ax1.plot(time_,ecgdata,'-',ms=5,label='ECG Signal')
        ax1.plot(time_[ecg_peak_loc],ecgdata[ecg_peak_loc],'.',ms=10,label='ECG Peaks') #x,y = index, y-value at index, do the same w trough
        ax1.plot(time_[ecg_val_loc],ecgdata[ecg_val_loc],'.',ms=10,label='ECG Valleys')
        
        # Plot where bad data is cut
        cut_indices = np.asfarray(ecg_peak_removal(ecg_peak_loc,ecg_val_loc))
        if len(cut_indices) > 0:
            xvals = []
            yvals = []
            for i in cut_indices:
                xvals += [ecg_val_loc[int(i)]]
                yvals += [ecg_filtered[ecg_val_loc[int(i)]]]
            ax1.plot(time_[xvals],yvals,'x',ms=10,label='Cut Valleys',color='red')
        
        #Pretty graph
        ax1.legend(loc='upper right')
        ax1.set_title('ECG count')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('ECG')
        plt.show()
    
    return ecg_peak_loc, ecg_val_loc

def ecg_heartrate(ecg_peak_loc,ecg_val_loc,SAMPLERATE_ECG):
    """finds the heart rate from an ECG signal

    Args:
        ecg_peak_loc (array): peak locations of ECG
        ecg_val_loc (array): valley locations of ECG
        SAMPLERATE_ECG (int): sample rate of ECG in Hz

    Returns:
        tuple: ECG heart rate and other important info
    """
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

def pulse_transit_time(ecg_peak_loc,midpoints_ir_smoothed):
    """Finds the pulse transit time

    Args:
        ecg_peak_loc (array): array of ECG peak indices
        midpoints_ir_smoothed (array): array of IR count peak midpoints

    Returns:
        float: calculated pulse transit time
    """

    offset = 0
    for i in reversed(range(1,len(midpoints_ir_smoothed))):
        distance = len(midpoints_ir_smoothed) - i
        # print(f'checking index {i-1} at {midpoints_ir_smoothed[i-1]} > {ecg_peak_loc[len(ecg_peak_loc)-distance+offset]}')
        if midpoints_ir_smoothed[i-1] > ecg_peak_loc[len(ecg_peak_loc)-distance+offset]:
            # print(f'index {i} delete')
            midpoints_ir_smoothed = np.delete(midpoints_ir_smoothed,i)
            offset = -1
            
    for i in reversed(range(1,len(ecg_peak_loc))):
        distance = len(ecg_peak_loc) - i
        # print(f'checking index {i-1} at {ecg_peak_loc[i-1]} > {midpoints_ir_smoothed[len(midpoints_ir_smoothed)-distance+offset]}')
        if ecg_peak_loc[i-1] > midpoints_ir_smoothed[len(midpoints_ir_smoothed)-distance+offset]:
            # print(f'index {i} delete')
            ecg_peak_loc = np.delete(ecg_peak_loc,i)
            offset = -1
            
    for i in reversed(range(1,len(midpoints_ir_smoothed))):
        distance = len(midpoints_ir_smoothed) - i
        # print(f'checking index {i-1} at {midpoints_ir_smoothed[i-1]} > {ecg_peak_loc[len(ecg_peak_loc)-distance+offset]}')
        if midpoints_ir_smoothed[i-1] > ecg_peak_loc[len(ecg_peak_loc)-distance+offset]:
            # print(f'index {i} delete')
            midpoints_ir_smoothed = np.delete(midpoints_ir_smoothed,i)
            offset = -1


    while len(ecg_peak_loc) > len(midpoints_ir_smoothed):
        ecg_peak_loc = np.delete(ecg_peak_loc,0)
    while len(midpoints_ir_smoothed) > len(ecg_peak_loc):
        midpoints_ir_smoothed = np.delete(midpoints_ir_smoothed,0)

    ptt = np.abs(time_[midpoints_ir_smoothed] - time_[ecg_peak_loc])

    return ptt

def signaltonoise(a, axis=0, ddof=0):
    """calculates the signal to noise ratio

    Args:
        a (array): signal
        axis (int, optional): what axis. Defaults to 0.
        ddof (int, optional): delta degrees of freedom. Defaults to 0.

    Returns:
        float: signal to noise ratio
    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))

##### MAIN PROGRAM #####

# df = pd.read_csv("/home/pablo/projects/ECG-PPG/Coding/data/ECPPG_2023-11-10_13-32-13.csv") # Pablo - Linux
df = pd.read_csv("C:\\Users\pazul\Documents\BMEN 207\Honors Project\ECG-PPG\Coding\data\ECPPG_2023-11-10_13-32-13.csv") # Pablo - Windows
# df = pd.read_csv("C:\data\honors project ppg data\ECPPG_2023-11-10_13-32-13.csv") # Karston
# df = pd.read_csv("C:\data\honors project ppg data\ECPPG_2023-11-30_17-54-16.csv") # Karston
# df = pd.read_csv("C:\data\honors project ppg data\ECPPG_2023-11-30_18-13-24.csv") # Karston
# df = pd.read_csv("C:\\Users\pazul\Documents\BMEN 207\Honors Project\ECG-PPG\Coding\data\ECPPG_2023-11-30_18-13-24.csv") # Pablo - Windows
# df = pd.read_csv("C:\data\honors project ppg data\ECPPG_2023-11-30_18-10-45.csv") # Karston
# df = pd.read_csv("C:\\Users\pazul\Documents\BMEN 207\Honors Project\ECG-PPG\Coding\data\ECPPG_2023-11-30_18-10-45.csv") # Pablo - Windows
# df = pd.read_csv("C:\\Users\pazul\Documents\BMEN 207\Honors Project\ECG-PPG\Coding\data\ECPPG_2023-11-08_16-33-32.csv")

time_, samplecount, IR_Count, Red_Count, ecg_raw, ecg_raw_mv, ecg_filtered, ecg_filtered_mv = read_file(df,100)
smoothed_IR = smooth_data(IR_Count, 'IR')
smoothed_Red = smooth_data(Red_Count, 'Red')

peaks_ir, valleys_ir = peak_valleys_PPG(IR_Count, time_, "IR",False)
peaks_red, valleys_red = peak_valleys_PPG(Red_Count, time_, "Red",False)

peaks_ir_smoothed, valleys_ir_smoothed = peak_valleys_PPG(smoothed_IR, time_, "Smoothed IR", smoothed=True)
peaks_red_smoothed, valleys_red_smoothed = peak_valleys_PPG(smoothed_Red, time_, "Smoothed Red", smoothed=True)

midpoints_ir_smoothed = midpoint_finder(peaks_ir_smoothed, valleys_ir_smoothed)
midpoints_red_smoothed = midpoint_finder(peaks_red_smoothed, valleys_red_smoothed)

ecg_peak_loc, ecg_val_loc = ecg_peak_finder(ecg_filtered, time_)
ecg_avg_bpm_2, ecg_stdev_bpm_2, ecg_avg_bpm, ecg_stdev_bpm, ecg_diffs = ecg_heartrate(ecg_peak_loc,ecg_val_loc,SAMPLERATE_ECG)

ptt_array = pulse_transit_time(ecg_peak_loc, midpoints_ir_smoothed)
avg_ptt = np.mean(ptt_array)

##### RESULTS ##### (CODE DIAGNOSTIC OUTPUTS COMMENTED OUT)

### USING THE IR COUNT ###
BPM1_vals_ir = BPM1(time_, peaks_ir, valleys_ir, compute_moving_average)
BPM2_vals_ir = BPM2(time_, peaks_ir, valleys_ir, compute_moving_average)
    
# print("------- Based on IR Count -------")
# print(f'Average BPM over time interval using peaks: {BPM1_vals_ir[0]:.2f}')
# print(f'Average BPM over time interval using valleys: {BPM1_vals_ir[1]:.2f}')
# if compute_moving_average:
#     print("BPM Moving Average with window length of 5 Heartbeats:")
#     print("Based on peaks: ",np.round(BPM1_vals_ir[2], 2))
#     print("Based on valleys: ", np.round(BPM1_vals_ir[4], 2))

# print("\n------- Based on IR Count -------")
# print(f'Average BPM2 over time interval using peaks: {BPM2_vals_ir[0]:.2f}')
# print(f'Average BPM2 over time interval using valleys: {BPM2_vals_ir[1]:.2f}')
# if compute_moving_average:
#     print("BPM2 Moving Average with window length of 5 Heartbeats:")
#     print("Based on peaks: ",np.round(BPM2_vals_ir[2], 2))
#     print("Based on valleys: ", np.round(BPM2_vals_ir[3], 2))

### USING THE SMOOTHED IR COUNT ###
BPM1_vals_smoothed_ir = BPM1(time_, peaks_ir_smoothed, valleys_ir_smoothed, compute_moving_average)
BPM2_vals_smoothed_ir = BPM2(time_, peaks_ir_smoothed, valleys_ir_smoothed, compute_moving_average)

# print("\n------- Based on Smoothed IR Count -------")
# print(f'Average BPM over time interval using peaks: {BPM1_vals_smoothed_ir[0]:.2f}')
# print(f'Average BPM over time interval using valleys: {BPM1_vals_smoothed_ir[1]:.2f}')
# if compute_moving_average:
#     print("BPM Moving Average with window length of 5 Heartbeats:")
#     print("Based on peaks: ",np.round(BPM1_vals_smoothed_ir[2], 2))
#     print("Based on valleys: ", np.round(BPM1_vals_smoothed_ir[3], 2))

# print("\n------- Based on Smoothed IR Count -------")
# print(f'Average BPM2 over time interval using peaks: {BPM2_vals_smoothed_ir[0]:.2f}')
# print(f'Average BPM2 over time interval using valleys: {BPM2_vals_smoothed_ir[1]:.2f}')
# if compute_moving_average:
#     print("BPM2 Moving Average with window length of 5 Heartbeats:")
#     print("Based on peaks: ",np.round(BPM2_vals_smoothed_ir[2], 2))
#     print("Based on valleys: ", np.round(BPM2_vals_smoothed_ir[3], 2))

### USING THE RED COUNT ###
BPM1_vals_red = BPM1(time_, peaks_red, valleys_red, compute_moving_average)
BPM2_vals_red = BPM2(time_, peaks_red, valleys_red, compute_moving_average)

# print("\n------- Based on Red Count -------")
# print(f'Average BPM over time interval using peaks: {BPM1_vals_red[0]:.2f}')
# print(f'Average BPM over time interval using valleys: {BPM1_vals_red[1]:.2f}')
# if compute_moving_average:
#     print("BPM Moving Average with window length of 5 Heartbeats:")
#     print("Based on peaks: ",np.round(BPM1_vals_red[2], 2))
#     print("Based on valleys: ", np.round(BPM1_vals_red[4], 2))

# print("\n------- Based on Red Count -------")
# print(f'Average BPM2 over time interval using peaks: {BPM2_vals_red[0]:.2f}')
# print(f'Average BPM2 over time interval using valleys: {BPM2_vals_red[1]:.2f}')
# if compute_moving_average:
#     print("BPM2 Moving Average with window length of 5 Heartbeats:")
#     print("Based on peaks: ",np.round(BPM2_vals_red[2], 2))
#     print("Based on valleys: ", np.round(BPM2_vals_red[3], 2))

### USING THE SMOOTHED RED COUNT ###
BPM1_vals_smoothed_red = BPM1(time_, peaks_red_smoothed, valleys_red_smoothed, compute_moving_average)
BPM2_vals_smoothed_red = BPM2(time_, peaks_red_smoothed, valleys_red_smoothed, compute_moving_average)

# print("\n------- Based on Smoothed Red Count -------")
# print(f'Average BPM over time interval using peaks: {BPM1_vals_smoothed_red[0]:.2f}')
# print(f'Average BPM over time interval using valleys: {BPM1_vals_smoothed_red[1]:.2f}')
# if compute_moving_average:
#     print("BPM Moving Average with window length of 5 Heartbeats:")
#     print("Based on peaks: ",np.round(BPM1_vals_smoothed_red[2], 2))
#     print("Based on valleys: ", np.round(BPM1_vals_smoothed_red[3], 2))

# print("\n------- Based on Smoothed Red Count -------")
# print(f'Average BPM2 over time interval using peaks: {BPM2_vals_smoothed_red[0]:.2f}')
# print(f'Average BPM2 over time interval using valleys: {BPM2_vals_smoothed_red[1]:.2f}')
# if compute_moving_average:
#     print("BPM2 Moving Average with window length of 5 Heartbeats:")
#     print("Based on peaks: ",np.round(BPM2_vals_smoothed_red[2], 2))
#     print("Based on valleys: ", np.round(BPM2_vals_smoothed_red[3], 2))

### ECG Values ###
# print("\n------- Based on ECG -------")
# print(f'Average ECG BPM: {ecg_avg_bpm}')
# #print(f'ECG BPM stdev: {ecg_stdev_bpm}')
# print(f'Average ECG BPM2: {ecg_avg_bpm_2}')
# #print(f'ECG BPM2 stdev: {ecg_stdev_bpm_2}')



### BPM REAL USING MOST ACCURATE IR_SMOOTH BPM2 AND ECG BPM2 DATA ###
bpm_mean = (ecg_avg_bpm_2 + BPM2_vals_smoothed_ir[1])/2
print('\n------- BPM -------')
print(f'Average Smoothed IR BPM using valleys: {BPM2_vals_smoothed_ir[1]:.2f}')
print(f'Average ECG BPM using valleys: {ecg_avg_bpm_2:.2f}')
print(f'Average BPM from ECG and IR: {bpm_mean:.2f}')

### SpO2 ###
avg_red_peak, avg_red_val, avg_ir_peak, avg_ir_val, avg_R, avg_SpO2 = SpO2(Red_Count, IR_Count, peaks_red, valleys_red, peaks_ir, valleys_ir)
print("\n------- SpO2 -------")
# print(f"Average Peak Value for Red Channel: {avg_red_peak:.2f} [counts]")
# print(f"Average Peak Value for IR Channel: {avg_ir_peak:.2f} [counts]")
# print(f"Average Trough Value for Red Channel: {avg_red_val:.2f} [counts]")
# print(f"Average Trough Value for IR Channel: {avg_ir_val:.2f} [counts]")
print(f"Average Ratio of Ratios: {avg_R:.2f}")
print(f"Average SpO2: {avg_SpO2:.2f}%") # probably >100% for us because we are young, older people would get a normal SpO2 percentage

### PULSE TRANSIT TIME ### 
print("\n------- Pulse Transit Time -------")
if avg_ptt > 0.5:
    print("Probably bad data! Unless patient is very old!")
print(f'Average Pulse Transit Time: {avg_ptt:.3f} seconds')

### FIGURES OF MERIT ### 
FOM_ecg_filtered = signaltonoise(ecg_filtered, axis = 0, ddof = 0)
FOM_ppg_ir = signaltonoise(IR_Count, axis = 0, ddof = 0)
FOM_ppg_red = signaltonoise(Red_Count, axis = 0, ddof = 0)

print("\n------- Figures of Merit -------")
print(f'Signal to noise ratio for IR Signal: {FOM_ppg_ir:.2f}')
print(f'Signal to noise ratio for Red Signal: {FOM_ppg_red:.2f}')
print(f'Signal to noise ratio for ECG Signal: {FOM_ecg_filtered:.2f}')
if 5 > FOM_ecg_filtered or FOM_ecg_filtered > 18:
    print("Bad reading! Values may not be accurate. By taking another reading and staying still, values may be more accurate.")

# end of computation
end = time.time()
print(f"\nCompute Time: {end-start:.4f} seconds") #includes time spent on plots if used in VS Code