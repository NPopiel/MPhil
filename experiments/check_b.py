import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from tools.utils import *
from scipy.signal import find_peaks
from tools.MakePlot import MakePlot
from matplotlib.offsetbox import AnchoredText

main_path = '/Users/npopiel/Documents/MPhil/Data/csvs_by_temp/'

folder_exts = ['FeSb2_data1',
               'FeSb2_data2']
# ,'SmB6_data1']

base_file_name = 'constant_temp_'
file_ext = '.csv'

temp_ranges = [(2, 30),
               (2, 10)]
# ,(2,23)]

headers_ch1 = ['b_field', 'chamber_pressure', 'sigma_resistance_ch1',
               'phase_angle_ch1', 'ac_current_ch1', 'voltage_amp_ch1', 'in_phase_voltage_ch1',
               'quad_voltage_ch1', 'gain_ch1', 'second_harmonic_ch1', 'third_harmonic_ch1']
headers_ch2 = ['b_field', 'chamber_pressure', 'sigma_resistance_ch2', 'phase_angle_ch2', 'ac_current_ch2',
               'voltage_amp_ch2',
               'in_phase_voltage_ch2', 'quad_voltage_ch2', 'gain_ch2', 'second_harmonic_ch2',
               'third_harmonic_ch2']

for idx, folder in enumerate(folder_exts):
    open_path = main_path + folder + '/'
    t1 = temp_ranges[idx][0]
    t2 = temp_ranges[idx][1] + 1
    save_path = open_path + 'figures/'
    makedir(save_path)
    save_path = save_path +'ch1/'
    makedir(save_path)
    for temp in range(t1, t2):
        file_name = open_path + base_file_name + str(temp) + file_ext
        df = drop_nans(pd.read_csv(file_name))

        df, peaks_voltage = extract_sweep_peaks(df, 'voltage_amp_ch1', 'voltage_sweep_ch1', 'sweep_')

        df, peaks_current = extract_sweep_peaks(df, 'ac_current_ch1', 'current_sweep_ch1', 'I = ')

        fig, axs = MakePlot(ncols=2, nrows=1, figsize=(16, 9)).create()
        ax1 = axs[0]
        ax2 = axs[1]


        sns.scatterplot(y='resistance_ch1', x='time', hue='voltage_sweep_ch1', data=df, ax=ax1, legend=False)
        # ax1.set_title('Resistance by Time')
        ax1.set_ylabel(r'$R  $', usetex=True, rotation=0)
        ax1.set_xlabel(r'$t $', usetex=True)

        sns.scatterplot(x='b_field', y='resistance_ch1', hue='voltage_sweep_ch1', data=df, ax=ax2, legend=True)
        # ax2.set_title('Resistance by Time')
        ax2.set_ylabel(r'$R  $', usetex=True, rotation=0)
        ax2.set_xlabel(r'$B  $', usetex=True)

        fig.suptitle('Resistance (T = ' + str(temp) + ' K)', fontsize=22)

        name = save_path + 'resistance_t_' + str(temp) + '.pdf'
        plt.show()
        #plt.savefig(name, dpi=200)
        print('Done!', name)
        plt.close()


