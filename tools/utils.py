import numpy as np
import pandas as pd
import scipy.io
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import argrelextrema, find_peaks, find_peaks_cwt
from .constants import OERSTED_2_TESLA

flatten = lambda l: [item for sublist in l for item in sublist]

def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')

def load_matrix(filepath):
    extension = filepath.split('.')[-1]
    if str(extension) == 'csv':
        return np.genfromtxt(filepath,delimiter = ',')
    elif str(extension) == 'npy':
        return np.load(filepath)
    elif str(extension) == 'mat':
        return scipy.io.loadmat(filepath)
    elif str(extension) == 'npz':
        return np.load(filepath)

def file_exists(filename):
    import os
    exists = os.path.isfile(filename)
    if exists:
        return True
    else:
        return False

def makedir(path):
    import os
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return False

def save_file(data,path,name):

    if hasattr(data, '__len__') and (not isinstance(data, str)):
        data = np.asarray(data)


    default_delimiter = ','
    format = '%1.5f'

    if len(data.shape) <= 2:
        file = path + str(name) + '.csv'

        if not file_exists(file):
            np.savetxt(file, data, delimiter=default_delimiter, fmt=format)
    else:
        file = path + str(name) + '.npy'
        if not file_exists(file):
            np.save(file,[data])

def oersted_to_tesla(arr):
    return arr/OERSTED_2_TESLA

def remove_irrelevant_columns(df, threshold=0.75):
    lst = [tup[0] for tup in df.isna().sum().items() if tup[1]>int(round(threshold* df.shape[0]))]
    return df.drop(columns=lst)

def remove_constant_column(df, std_threshold=0.007, drop_eto=True):
    lst = [tup[0] for tup in df.std().items() if tup[1] <= std_threshold]
    if drop_eto: lst.append('eto_code')
    return df.drop(columns=lst)

def power_law(x,a,b):
    return a * np.power(x,b)


def extract_changing_field(df, col_name, new_col_name, root_flag_marker, threshold=0.001, round_val=0):
    relevant_parameter = np.array(df[col_name])

    flag = []

    time_deriv = np.diff(relevant_parameter)

    for ind, val in enumerate(time_deriv):
        if np.abs(val) < threshold:
            flag.append('constant_' + root_flag_marker + '_' + str(relevant_parameter[ind].round(round_val)))
        elif val < 0:
            flag.append('decreasing')
        else:
            flag.append('increasing')

    flag.append(flag[-1])

    df[new_col_name] = flag

    return df

def extract_stepwise_peaks(df, col_name, new_col_name, root_flag_marker, threshold=0.9, round_num=3):

    relevant_parameter = np.array(df[col_name])
    time_deriv = np.diff(relevant_parameter)
    peaks = np.where(time_deriv>=threshold)[0]
    peaks+=1
    locs = flatten([[0],peaks.tolist(),[len(relevant_parameter)]])

    lst = []

    #peaks = find_peaks(time_deriv,distance=1000)

    new_arr = np.zeros(len(relevant_parameter),dtype='S32')

    for idx_pair in zip(locs[:-1],locs[1:]):
        for j in range(idx_pair[0],idx_pair[1]):
            indexes = np.arange(idx_pair[0],idx_pair[1])
            lst.append(root_flag_marker + str(np.mean(relevant_parameter[indexes]).round(round_num)))


    df[new_col_name] = lst

    return df, locs

def drop_nans(df):
    df_copy = df.copy()
    #nan_locs=np.argwhere(np.isnan(df.voltage_amp_ch2))
    df_no_nan1 = df_copy.dropna(how='all')
    #df_no_nan1.reset_index(drop=False)
    df_no_nan2 = df_copy.apply(lambda x: pd.Series(x.dropna().values))
    df_no_nan2.reset_index(drop=False)
    return df_copy.apply(lambda x: pd.Series(x.dropna().values)).dropna()

def extract_sweep_peaks(df, col_name, new_col_name, root_flag_marker, distance_between_peaks=50):

    relevant_parameter = np.array(df[col_name])
    time_deriv = np.diff(relevant_parameter)

    peaks, properties = find_peaks(time_deriv, distance=distance_between_peaks)

    peaks+=1

    locs = flatten([[0],peaks.tolist(),[len(relevant_parameter)]])

    lst = []


    c=0
    for idx_pair in zip(locs[:-1],locs[1:]):
        for j in range(idx_pair[0],idx_pair[1]):
            indexes = np.arange(idx_pair[0],idx_pair[1])
            lst.append(root_flag_marker + str(round(relevant_parameter[idx_pair[0]+1]*1000,2)))
        c+=1


    df[new_col_name] = lst

    return df, locs






