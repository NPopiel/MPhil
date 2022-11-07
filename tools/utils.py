from .constants import *
from .DataFile import *
import numpy as np
import pandas as pd
import scipy.io
import os
import matplotlib.pyplot as plt
import matplotlib
from .MakePlot import MakePlot
from matplotlib.animation import FuncAnimation
from scipy.signal import argrelextrema, find_peaks, find_peaks_cwt

from scipy.signal import savgol_filter, argrelmin, argrelmax
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit



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

def find_nearest(array, value,return_idx=False):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if not return_idx:
        return array[idx]
    else:
        return array[idx], idx

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

def load_matrix(filepath, delimeter=',',dat_type = 'mpms',skiprows=0,params=default_parameters_dat_file):
    extension = filepath.split('.')[-1]
    if str(extension) == 'csv':
        return np.genfromtxt(filepath,delimiter = delimeter)
    elif str(extension) == 'txt':
        return pd.read_csv(filepath,delimiter=delimeter,skiprows=skiprows).to_numpy()
    elif str(extension) == 'npy':
        return np.load(filepath)
    elif str(extension) == 'mat':
        return scipy.io.loadmat(filepath)
    elif str(extension) == 'npz':
        return np.load(filepath)
    elif str(extension) == 'dat':
        if dat_type.lower() == 'mpms':
            return DataFile(filepath,parameters=params).open()
        elif dat_type.lower() == 'ppms':
            return DataFile(filepath,parameters=params).open()
        else:
            return pd.read_csv(filepath, delimiter=delimeter, skiprows=skiprows).to_numpy()


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

def remove_noise(data,window_size, eps):
    series = pd.Series(data)
    avg = np.array(series.rolling(window_size, min_periods=1).mean())
    stddev = np.array(series.rolling(window_size, min_periods=1).std())
    lst_to_drop = []
    for ind, val in enumerate(series):
        if not (avg[ind] - eps*stddev[ind] < val < avg[ind] + eps*stddev[ind]):
            lst_to_drop.append(ind)

    return np.array(series.reset_index().drop(lst_to_drop))[:,1], lst_to_drop

def save_file(data,path,name,file_check=True):

    if hasattr(data, '__len__') and (not isinstance(data, str)):
        data = np.asarray(data)


    default_delimiter = ','
    format = '%1.5f'

    if len(data.shape) <= 2:
        file = path + str(name) + '.csv'
        if file_check:
            if not file_exists(file):
                np.savetxt(file, data, delimiter=default_delimiter, fmt=format)
        else:
            np.savetxt(file, data, delimiter=default_delimiter, fmt=format)
    else:
        file = path + str(name) + '.npy'
        if file_check:
            if not file_exists(file):
                np.save(file,[data])
        else:
            np.save(file, [data])



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

def extract_stepwise_peaks(df, col_name, new_col_name, root_flag_marker, plot=False,threshold=1, round_num=1, last_point=True):

    relevant_parameter = np.abs(np.array(df[col_name]))
    time_deriv = np.diff(relevant_parameter)
    peaks = np.where(time_deriv>=threshold)[0]
    peaks+=1

    if last_point: locs = flatten([[0],peaks.tolist(),[len(relevant_parameter)]])
    else: locs = flatten([[0],peaks.tolist()[:-1], [len(relevant_parameter)]])

    lst = []

    #peaks = find_peaks(time_deriv,distance=1000)

    new_arr = np.zeros(len(relevant_parameter),dtype='S32')

    for idx_pair in zip(locs[:-1],locs[1:]):
        for j in range(idx_pair[0],idx_pair[1]):
            indexes = np.arange(idx_pair[0],idx_pair[1])
            lst.append(root_flag_marker + str(np.mean(relevant_parameter[indexes]).round(round_num)))


    df[new_col_name] = lst

    if plot:
        fig, ax = MakePlot().create()
        plt.plot(relevant_parameter)
        for loc in locs: plt.axvline(loc)
        plt.show()

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

def kelvin_2_mev(delta):
    delta = np.abs(delta)
    delta_joules = delta*kb
    delta_ev = delta_joules/eV

    return delta_ev * 1000

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

#def remove_outliers(data):
    # for each point, take average of the three points ahead and behind, if the po


def publication_plot(ax, xlab, ylab, label_fontsize=24, tick_fontsize=22, x_ax_log=False, y_ax_log=False,
                     x_ax_sci=False, y_ax_sci=False,
                     fontname='arial', num_minor_ticks=2, title=None):
    from matplotlib.ticker import AutoMinorLocator

    plt.setp(ax.get_xticklabels(), fontsize=tick_fontsize, fontname=fontname)
    plt.setp(ax.get_yticklabels(), fontsize=tick_fontsize, fontname=fontname)
    ax.set_ylabel(ylab, fontsize=label_fontsize, fontname=fontname)
    ax.set_xlabel(xlab, fontsize=label_fontsize, fontname=fontname)

    ax.minorticks_on()

    ax.tick_params('both', which='major', direction='in', length=5, width=1.2,
                   bottom=True, top=True, left=True, right=True)

    ax.tick_params('both', which='minor', direction='in', length=3, width=1,
                   bottom=True, top=True, left=True, right=True)

    if x_ax_log:
        ax.set_xscale('log')
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator(num_minor_ticks))

    if y_ax_log:
        ax.set_yscale('log')
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator(num_minor_ticks))

    if x_ax_sci:
        ax.ticklabel_format(axis='x', style="sci", useMathText=True, scilimits=(0, 0))
        ax.xaxis.offsetText.set_fontsize(label_fontsize-2)
    # else:
    #     ax.ticklabel_format(style='plain')

    if y_ax_sci:
        ax.ticklabel_format(axis='y', style="sci", useMathText=True, scilimits=(0, 0))
        ax.yaxis.offsetText.set_fontsize(label_fontsize-2)

    if title is not None:
        ax.set_title(title, fontsize=label_fontsize + 6, fontname=fontname)

    return ax


def publication_plot_broken(ax, xlab, ylab, label_fontsize=24, tick_fontsize=22, x_ax_log=False, y_ax_log=False,
                     x_ax_sci=False, y_ax_sci=False,
                     fontname='arial', num_minor_ticks=2, title=None):
    from matplotlib.ticker import AutoMinorLocator

    plt.setp(ax.get_xticklabels(), fontsize=tick_fontsize, fontname=fontname)
    plt.setp(ax.get_yticklabels(), fontsize=tick_fontsize, fontname=fontname)
    ax.set_ylabel(ylab, fontsize=label_fontsize, fontname=fontname)
    ax.set_xlabel(xlab, fontsize=label_fontsize, fontname=fontname)

    ax.minorticks_on()

    ax.tick_params('both', which='major', direction='in', length=5, width=1.2,
                   bottom=True, top=True, left=True, right=True)

    ax.tick_params('both', which='minor', direction='in', length=3, width=1,
                   bottom=True, top=True, left=True, right=True)

    return ax

def fill_legend_by_rows(ax, ncols):

    handles, labels = ax.get_legend_handles_labels()

    inds = np.arange(len(handles))

    new_inds = []

    i = 0

    while i < ncols:

        rel_inds = inds[i:len(handles):ncols]

        for j in rel_inds:
            new_inds.append(j)

        i+=1

    return [handles[k] for k in new_inds], [labels[k] for k in new_inds]

def get_bsqr_deviation_analytic(field, volts, N, plot=False,threshold=3.,std=False):

    func = lambda x, alpha, beta, gamma, delta : alpha * x ** 2 + beta * np.exp(x * gamma) + delta

    # func = lambda x, alpha, beta, gamma : alpha * x ** 2 + beta / (x - gamma)

    popt, pcov = curve_fit(func, field, volts)

    alpha = popt[0]
    beta = popt[1]
    gamma = popt[2]
    delta = popt[3]

    err_in_fit = np.sqrt(np.diag(pcov))

    err_in_deriv = 2*err_in_fit[0] + np.sqrt((err_in_fit[1]/beta)**2 + 2*(err_in_fit[2]/gamma)**2)

    second_deriv = 2 * alpha + beta * gamma ** 2 * np.exp(gamma * field)

    mean_2nd_deriv_og = np.mean(second_deriv[:N])
    std_2nd_deriv_og = np.std(second_deriv[:N])

    locs_of_mean = np.arange(N)
    locs = np.setdiff1d(np.arange(len(field)), locs_of_mean)



    if not std:
        dev_locs = np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og)  # + 25 * std_2nd_deriv_og
    else:
        dev_locs = np.abs(second_deriv)[locs] > threshold * std_2nd_deriv_og

    min_err =  np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og) - err_in_deriv
    max_err =  np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og) + err_in_deriv

    dev_loc2nd = np.argmax(dev_locs > 0) + N

    min_err_loc = np.argmax(min_err > 0) + N
    max_err_loc = np.argmax(max_err > 0) + N

    if plot:
        fig, axs = plt.subplots(ncols=2,figsize=(16,9))
        axs[0].plot(field, volts,linewidth=2,c='indianred',label='Data')
        axs[0].plot(field, func(field,alpha, beta, gamma, delta),linewidth=2,c='midnightblue',label='Fit',linestyle='dashed')

        axs[1].plot(field, second_deriv,linewidth=2,c='darkgray')
        axs[1].axvline(field[dev_loc2nd])
        axs[0].axvline(field[dev_loc2nd])

        axs[0].legend(framealpha=0, ncol=1, loc='best',
                            prop={'size': 24, 'family': 'arial'})
        publication_plot(axs[0],'Magnetic Field (T)', 'Torque (arb.)')
        publication_plot(axs[1], 'Magnetic Field (T)', r'$\frac{\partial^2 \tau}{\partial B^2}$')
        plt.tight_layout(pad=1)
        plt.show()



    return field[dev_loc2nd], field[min_err_loc], field[max_err_loc]
