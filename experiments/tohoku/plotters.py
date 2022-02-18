import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
import pandas as pd
import numpy.linalg
from tools.DataFile import DataFile
from tools.MakePlot import *
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from tools.utils import *

# Functions from QuickLook Jupyter
def remove_outliers(an_array, max_deviations):
    mean = np.mean(an_array)
    standard_deviation = np.std(an_array)
    distance_from_mean = abs(an_array - mean)
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    return not_outlier


def remove_outliers2(an_array, filter_num):
    return median_filter(an_array, filter_num)


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:

    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def check_arrays(arr1, arr2, diff_factor=3):
    d1 = np.diff(arr1)
    d2 = np.diff(arr2)

    locs = []
    for i in range(len(d1)):
        if median_filter(d1[i], 15) - d2[i] < diff_factor:
            locs.append(False)
        else:
            locs.append(True)

    return locs


def get_loess_window(data_set, tesla_window):
    spacing = data_set[4] - data_set[3]
    return 2 * int(round(0.5 * tesla_window / spacing)) + 1


def remove_outliers2(an_array, filter_num):
    return median_filter(an_array, filter_num)


def process_data(main_path, filename_open, field_window=2.2):
    dat = load_matrix(main_path + filename_open, delimeter='\t')

    field = dat[:, field_ind]
    capacitance = dat[:, cap_ind]
    Vx = dat[:, Vx_ind]
    Vy = dat[:, Vy_ind]
    I = dat[:, I_ind]
    R = Vx / I

    field_sorted_inds = np.argsort(field)

    field = field[field_sorted_inds]
    capacitance = capacitance[field_sorted_inds]
    R = R[field_sorted_inds]

    locs = np.where(field > 12)

    if filename_open == '032.txt':
        locs = locs[0][:-16]

    R = R[locs]

    field = field[locs]

    capacitance = capacitance[locs]
    raw_cap = capacitance

    cap_no_outliers = remove_outliers2(capacitance, 2)

    cap_interpolated = np.copy(cap_no_outliers)

    nans, x = nan_helper(cap_interpolated)
    cap_interpolated[nans] = np.interp(x(nans), x(~nans), cap_interpolated[~nans])

    if filename_open != '042.txt':
        cap_interpolated = savgol_filter(cap_interpolated, get_loess_window(field, field_window), 2)

    field = field[15:]
    cap_interpolated = cap_interpolated[15:]
    R = R[15:]

    delta = ref_C0 - cap_interpolated[0]

    cap_interpolated = cap_interpolated

    deriv = np.diff(cap_interpolated)
    idx2 = np.where(np.sign(deriv[:-1]) != np.sign(deriv[1:]))[0] + 1

    return field, cap_interpolated, R, idx2, raw_cap


# Snippet of code to read in stuff and get the values

main_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2021_11_Tohoku/'

filenames = ['016.txt', '033.txt', '017.txt', '032.txt', '031.txt', '018.txt', '029.txt', '028.txt', '019.txt',
             '027.txt', '026.txt',
             '020.txt', '022.txt',
             '023.txt', '024.txt',
             '025.txt', '040.txt', '044.txt']

angles = [0, 10, 20, 25, 27.5, 30, 30, 32.5, 35, 35, 37.5, 40, 50, 60, 70, 80, 33, 25]

filenames_2ndsweep = ['033.txt', '034.txt', '035.txt', '040.txt', '036.txt', '037.txt', '038.txt', '039.txt']
angles_2 = [0, 10, 20, 26, 30, 40, 50, 60]

dat = load_matrix(main_path + '033.txt', delimeter='\t')

time_ind = 0
field_ind = 1
temp_ind = 2
Vx_ind = 3
Vy_ind = 4
cap_ind = 5
loss_ind = 6
Vext_ind = 7
I_ind = 8
hall_ind = 9

field = dat[:, field_ind]
capacitance = dat[:, cap_ind]
Vx = dat[:, Vx_ind]
Vy = dat[:, Vy_ind]
I = dat[:, I_ind]
R = Vx / I

field_sorted_inds = np.argsort(field)

field = field[field_sorted_inds]
capacitance = capacitance[field_sorted_inds]
R = R[field_sorted_inds]

locs = np.where(field > 12)

R = R[locs]

field = field[locs]

capacitance = capacitance[locs]

cap_no_outliers = remove_outliers2(capacitance, 2)

cap_interpolated = np.copy(cap_no_outliers)

nans, x = nan_helper(cap_interpolated)
cap_interpolated[nans] = np.interp(x(nans), x(~nans), cap_interpolated[~nans])

cap_interpolated = savgol_filter(cap_interpolated, get_loess_window(field, 2.2), 2)

field = field[15:]
cap_interpolated = cap_interpolated[15:]
R = R[15:]

ref_C0 = cap_interpolated[0]

fig, axs = MakePlot(nrows=1, ncols=2).create()

fig2, ax2 = MakePlot().create()
fig3, ax3 = MakePlot().create()

fig_together, axs_together = MakePlot(nrows=1, ncols=2).create()

for ind, filename_open in enumerate(filenames):

    d = {}

    field, cap_interpolated, R, idx2, raw_cap = process_data(main_path, filename_open)

    if idx2.shape[0] != 0:

        b_sign_change_idx = np.argmin(cap_interpolated)
        ax2.scatter(angles[ind], field[b_sign_change_idx], s=250, c=plt.cm.jet(ind / len(filenames)), label=angles[ind])
    elif filename_open == '032.txt':
        b_sign_change_idx = np.argmin(cap_interpolated)
        ax2.scatter(angles[ind], field[b_sign_change_idx], s=250, c=plt.cm.jet(ind / len(filenames)), label=angles[ind])

    ylabel = 'Magnetic Field at Minima (T)'
    xlabel = 'Angle (degrees)'
    publication_plot(ax2, xlabel, ylabel, tick_fontsize=16)

    ax = axs[0]
    ax.plot(field, cap_interpolated, lw=2.5, c=plt.cm.jet(ind / len(filenames)))
    # ax.plot(field, raw_cap[15:],lw=2.5, c=plt.cm.jet(ind/len(filenames)))
    xlabel = 'Magnetic Field (T)'
    ylabel = 'Capacitance (pF)'
    publication_plot(ax, xlabel, ylabel, tick_fontsize=16)

    ax = axs_together[0]
    ax.plot(field, cap_interpolated, lw=2.5, c=plt.cm.jet(ind / len(filenames)), label=angles[ind])
    xlabel = 'Magnetic Field (T)'
    ylabel = 'Capacitance (pF)'
    publication_plot(ax, xlabel, ylabel, tick_fontsize=16)
    ax.set_title('Rotating c to a', fontname='arial', fontsize=26)

    ax = axs[1]

    ax.plot(field ** 2, cap_interpolated, lw=2.5, c=plt.cm.jet(ind / len(filenames)), label=angles[ind])
    # ax.axvline(divergence_1, c=plt.cm.jet(ind/len(filenames)))
    xlabel = 'B$^2$ (T$^2$)'
    ylabel = 'Capacitance (pF)'
    publication_plot(ax, xlabel, ylabel, tick_fontsize=16)

    ax = ax3
    ax.plot(field, R, lw=2.5, c=plt.cm.jet(ind / len(filenames)), label=angles[ind])
    xlabel = 'Magnetic Field (T)'
    ylabel = 'Resistance ($\Omega$)'
    publication_plot(ax, xlabel, ylabel, y_ax_sci=True, tick_fontsize=16)
    ax.set_ylim(0, np.max(R) + 50)
    ax.set_title('Magnetoresistance', fontname='arial', fontsize=26)

l = axs[1].legend(framealpha=0, ncol=len(filenames) // 8, loc='best',
                  title='Angle (Degrees)', prop={'size': 16})
plt.setp(l.get_title(), fontsize=18)

l1 = ax.legend(framealpha=0, ncol=len(filenames) // 8, loc='best',
               title='Angle (Degrees)', prop={'size': 16})

plt.setp(l1.get_title(), fontsize=18)

fig.tight_layout(pad=3)

# fig.show()

l2 = ax2.legend(framealpha=0, ncol=len(filenames) // 8, loc='lower right',
                title='Angle (Degrees)', prop={'size': 12})
plt.setp(l2.get_title(), fontsize=18)
ax3.axhline(0)

l1 = axs_together[0].legend(framealpha=0, ncol=len(filenames) // 8, loc='best',
                            title='Angle (Degrees)', prop={'size': 16})

# fig2.show()

plt.show()

for ind