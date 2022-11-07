import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter, get_window
import pandas as pd
import numpy.linalg
from tools.DataFile import DataFile
from tools.MakePlot import *
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from tools.ColorMaps import select_discrete_cmap
from tools.utils import *

def sort_func(filepath):
  name = filepath.split('/')[-1]
  sweep_num = name.split('p')[1].split('.')[0]
  return float(sweep_num)

'''
/Volumes/GoogleDrive/Shared drives/Quantum Materials/Popiel/FeSi/FeSi_Feb2022_torque006.txt
'''

main_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/Popiel/FeSi/SdH-Bad/'

filenames = ['FeSi_Feb2022_SdH001.txt',
                 'FeSi_Feb2022_SdH002.txt'
                 ]

def split_by_field(field, volts, min_field=0.001, max_field=13.99, threshold=15):
    t_zero = np.abs(field) < min_field
    t_high = field > max_field
    t_low = field < -1*max_field

    t_any = t_zero + t_high + t_low

    idxs = np.cumsum(t_any)

    split_fields, split_volts = {}, {}
    for i in range(idxs[-1]):
        if np.sum(idxs == i) > threshold:
            split_fields[i] = field[idxs == i]
            split_volts[i] = volts[idxs == i]

    return split_fields, split_volts


data_dict = {}

I = 1e-6 #uA

Vs, Bs = [], []

fig, ax = MakePlot().create()

i = 0

field_dict_list, volts_dict_list = [], []

for file in filenames:

    data = np.genfromtxt(main_path + file, delimiter='\t', skip_header=31, usecols=(0, 8, 9, 17))
    # time,temp,field,data
    temps = data[:360000, 1]
    field = data[:360000, 2]
    volts = data[:360000, 3]


    split_fields, split_volts = split_by_field(field, volts, max_field=13.99, min_field=0.001, threshold=15)

    field_dict_list.append(split_fields)
    volts_dict_list.append(split_volts)


    for key, dat in split_fields.items():

        ax.plot(split_fields[key], np.abs(split_volts[key]/I), linewidth=2, c=select_discrete_cmap('bulbasaur')[i], label='Sweep '+str(key))
        i+=1

ax.legend()

publication_plot(ax, 'Magnetic Field (T)', 'Resistance ($\Omega$)')
plt.show()


volts_dict = volts_dict_list[0]
volts_dict.update(volts_dict_list[1])

field_dict = field_dict_list[0]
field_dict.update(field_dict_list[1])

field_494 = field_dict[494]
volts_494 = np.abs(volts_dict[494])

tesla_window = 1.5

spacing = np.abs(field_494[1] - field_494[0])

polyorder = 3

filtered_volts = savgol_filter(volts_494,
                               2 * int(round(0.5 * tesla_window / spacing)) + 1,
                               polyorder=polyorder
                               )

fitted_volts = np.poly1d(np.polyfit(field_494, volts_494, 3))

subtracted_volts = filtered_volts - fitted_volts(field_494)

fft_vals = np.abs(np.fft.rfft(subtracted_volts * get_window('hanning',
                                                      len(subtracted_volts)),
                              n=6553600))
fft_freqs = np.fft.rfftfreq(6553600, d=spacing)

fig, axs = MakePlot(ncols=2).create()

axs[0].plot(field_494, subtracted_volts,linewidth=2, c=select_discrete_cmap('bulbasaur')[i])
publication_plot(axs[0], 'Magnetic Field (T)', 'BG Subtracted Resistance ($\Omega$)')


axs[1].plot(fft_freqs, fft_vals,linewidth=2, c=select_discrete_cmap('bulbasaur')[i])
publication_plot(axs[1], 'Frequency (T)', 'FFT Amplitude (arb.)')

plt.show()

# Load in all of Angle 8 and plot as a function of time to examine noise
