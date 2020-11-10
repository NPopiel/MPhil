import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from tools.utils import *
from scipy.signal import find_peaks
from tools.MakePlot import MakePlot
from matplotlib.offsetbox import AnchoredText


# FUCK YA I NEED TO FIX THE drop_nan function! This causes an issue with selecting out the subsets of currents

main_path = '/Users/npopiel/Documents/MPhil/Data/csvs_by_temp/'

folder_exts = ['FeSb2_data1',
               'FeSb2_data2','SmB6_data1']

base_file_name = 'constant_temp_'
file_ext = '.csv'

temp_ranges = [(2, 30),
               (2, 10),(2,23)]

headers_ch1 = ['b_field', 'chamber_pressure', 'sigma_resistance_ch1',
               'phase_angle_ch1', 'ac_current_ch1', 'voltage_amp_ch1', 'in_phase_voltage_ch1',
               'quad_voltage_ch1', 'gain_ch1', 'second_harmonic_ch1', 'third_harmonic_ch1']
headers_ch2 = ['b_field', 'chamber_pressure', 'sigma_resistance_ch2', 'phase_angle_ch2', 'ac_current_ch2',
               'voltage_amp_ch2',
               'in_phase_voltage_ch2', 'quad_voltage_ch2', 'gain_ch2', 'second_harmonic_ch2',
               'third_harmonic_ch2']

channel='ch1'

for idx, folder in enumerate(folder_exts):
    if channel == 'ch2': nan_on_two = False
    else: nan_on_two = True

    open_path = main_path + folder + '/'
    t1 = temp_ranges[idx][0]
    t2 = temp_ranges[idx][1] + 1
    save_path = open_path + 'figures/'
    makedir(save_path)
    save_path = save_path +channel+'/'
    makedir(save_path)
    save_path+='smaller/'
    makedir(save_path)
    for temp in range(t1, t2):
        file_name = open_path + base_file_name + str(temp) + file_ext
        df_og = pd.read_csv(file_name)

        df = drop_nans(df_og, nan_on_two=nan_on_two)

        df, peaks_current = extract_sweep_peaks(df, 'ac_current_'+channel, 'current_sweep_'+channel, 'I = ')

        fig, axs = MakePlot(ncols=3, nrows=1, figsize=(16, 9)).create()
        ax1 = axs[0]
        ax2 = axs[1]
        ax3 = axs[2]

        sns.scatterplot(y='resistance_'+channel, x='time', hue='current_sweep_'+channel, data=df, ax=ax1, legend=False)
        # ax1.set_title('Resistance by Time')
        ax1.set_ylabel(r'$R  $', usetex=True, rotation=0,fontsize=16)
        ax1.set_xlabel(r'$t  $', usetex=True,fontsize=16)

        sns.scatterplot(x='voltage_amp_'+channel, y='resistance_'+channel, hue='current_sweep_'+channel, data=df, ax=ax2, legend=False)
        # ax2.set_title('Resistance by Time')
        ax2.set_ylabel(r'', usetex=True, rotation=0,fontsize=16)
        ax2.set_xlabel(r'$V  $', usetex=True,fontsize=16)

        sns.scatterplot(y='resistance_'+channel, x='b_field', hue='current_sweep_'+channel, data=df, ax=ax3, legend=True)
        # ax3.set_title('Voltage by Time')
        ax3.set_ylabel(r'', usetex=True, rotation=0,fontsize=16)
        ax3.set_xlabel(r'$\mu_o H_o  $', usetex=True,fontsize=16)

        legend = ax3.legend()
        ax3.get_legend().remove()

        fig.suptitle('Resistance (T = ' + str(temp) + ' K)', fontsize=22)
        plt.figlegend(frameon=False,
                      loc='center right',
                      title='Current (mA)',)  # ,labels=np.unique(df.current_sweep_ch2.values), frameon=True)
        name = save_path + 'resistance_t_' + str(temp) + '.pdf'
        #plt.show()
        plt.savefig(name, dpi=200)
        print('Done!', name)
        plt.close()


