import numpy as np
import pandas as pd
import scipy.io
import os
import matplotlib.pyplot as plt
import matplotlib
from .MakePlot import MakePlot
from matplotlib.animation import FuncAnimation
from scipy.signal import argrelextrema, find_peaks, find_peaks_cwt
from .constants import OERSTED_2_TESLA
from .DataFile import *

flatten = lambda l: [item for sublist in l for item in sublist]

default_parameters_dat_file = [',',
                               True,
                               True,
                               None,
                               False,
                               None]

def deriv_skip_one(array):
    lst=[]
    for ind, val in enumerate(array):
        len_arr = len(array)
        if ind%2 == 0 and ind<len_arr: lst.append(array[ind+2]-val)
        else: lst.append(np.nan)
    return np.array(lst)

def first_derivative(array, ind=1):
    p = []
    for i in range(array.shape[0]):
        if i+ind > (array.shape[0] - 1):
            p_i = array[i]
            p.append(p_i)
        else:
            p_i = array[i+ind] - array[i]
            p.append(p_i)
    return np.asarray(p)

def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')

def load_matrix(filepath, delimeter=',',dat_type = 'mpms',params=default_parameters_dat_file):
    extension = filepath.split('.')[-1]
    if str(extension) == 'csv':
        return np.genfromtxt(filepath,delimiter = delimeter)
    elif str(extension) == 'txt':
        return np.genfromtxt(filepath,delimiter = delimeter, names=True)
    elif str(extension) == 'npy':
        return np.load(filepath)
    elif str(extension) == 'mat':
        return scipy.io.loadmat(filepath)
    elif str(extension) == 'npz':
        return np.load(filepath)
    elif str(extension) == 'dat':
        return DataFile(filepath,parameters=params).open()

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

def find_b_extrma(df, col_name='b_flag',threshold=0.001):
    # This program uses my convention of before for labelling the b_flag
    # We know that if it goes from increasing to decreasing, it's  maximum value!
    # Conversely, if it goes from decreasing to increasing, its a minimum!
    # This will still run into the problem with cSMB6 data where it is constant...

    # Pseudocode
    # given b_flag list
    # for flag1, flag2 in zip(b_flag[:-1],b_flag[1:]
        # if flag1 == 'increasing' and flag2 =='decreasing':
            #save maximum index which is flag1 position
        # elif flag1 == 'decreasing

    b_flag = df[col_name].tolist()

    total_inds = np.arange(len(b_flag))

    increasing_inds = [i for i, n in enumerate(b_flag) if n == 'increasing']
    decreasing_inds = [i for i, n in enumerate(b_flag) if n == 'decreasing']

    max_locs, min_locs = [],[]
    c=0

    for flag1, flag2 in zip(b_flag[:-1],b_flag[1:]):
        if flag1 == 'increasing':
            if flag2 == 'decreasing':
                max_locs.append(c)

            #elif flag2 != 'increasing':
             #   max_locs.append(next(x for x, val in enumerate(decreasing_inds) if val > c))

        if flag1 == 'decreasing':
            if flag2 == 'increasing':
                min_locs.append(c)
            #elif flag2 != 'decreasing':
             #   min_locs.append(next(x for x, val in enumerate(increasing_inds) if val > c))
        c+=1

    return max_locs, min_locs

    relevant_parameter = np.array(df[col_name])

    flag = []
    c=0
    for ind, val in enumerate(second_deriv):
        if np.abs(val) < threshold:
            flag.append(c)
        c+=1

    return flag

def extract_stepwise_peaks(df, col_name, new_col_name, root_flag_marker, threshold=1, round_num=1):

    relevant_parameter = np.abs(np.array(df[col_name]))
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

def drop_nans(df,nan_on_two=True):
    # If the position of the na is on two (the case of ch2, use this) otherwise, we have the nan in position 1 (ch1)
    df_copy = df.copy()
    df_copy = drop_double_nan(df,'resistance_ch2')
    if not nan_on_two:
        df_copy = df_copy.drop(df_copy.index[0])
        return df_copy.iloc[::2]
    else: return df_copy.iloc[::2]

def extract_sweep_peaks(df, col_name, new_col_name, root_flag_marker, distance_between_peaks=50):

    relevant_parameter = np.array(df[col_name])
    time_deriv = np.diff(relevant_parameter)

    peaks, properties = find_peaks(time_deriv, distance=distance_between_peaks)
    #fig, ax = MakePlot().create()
    #plt.plot(time_deriv)
    #plt.show()
    peaks+=1
    locs = flatten([[0],peaks.tolist(),[len(relevant_parameter)]])
    lst = []

    c=0
    for idx_pair in zip(locs[:-1],locs[1:]):
        for j in range(idx_pair[0],idx_pair[1]):
            indexes = np.arange(idx_pair[0],idx_pair[1])
            lst.append(root_flag_marker + str(round(relevant_parameter[idx_pair[0]+1]*1000,1)))
        c+=1


    df[new_col_name] = lst

    return df, locs

def drop_double_nan(df, col_name='resistance_ch1'):

    nans_to_drop = np.diff(np.isnan(np.array(df[col_name])))
    locs_to_drop = []
    for ind, el in enumerate(nans_to_drop):
        if el == False:
            locs_to_drop.append(ind)

    return df.drop(locs_to_drop)

def load_r_and_h(temp_path, current):
    current_path = temp_path + str(current) + '/'

    data = np.squeeze(load_matrix(current_path + 'data.csv'))

    resistance = data[0]
    field = data[1]
    return resistance, field

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]





