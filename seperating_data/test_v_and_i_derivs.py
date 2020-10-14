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
              'FeSb2_data2',
              'SmB6_data1']

base_file_name = 'constant_temp_'
file_ext = '.csv'

temp_ranges = [(2,30),
               (2,10),
               (2,23)]

headers_ch1 = ['b_field', 'chamber_pressure', 'sigma_resistance_ch1',
               'phase_angle_ch1', 'ac_current_ch1', 'voltage_amp_ch1', 'in_phase_voltage_ch1',
               'quad_voltage_ch1', 'gain_ch1', 'second_harmonic_ch1', 'third_harmonic_ch1']
headers_ch2 = ['b_field', 'chamber_pressure', 'sigma_resistance_ch2', 'phase_angle_ch2', 'ac_current_ch2',
               'voltage_amp_ch2',
               'in_phase_voltage_ch2', 'quad_voltage_ch2', 'gain_ch2', 'second_harmonic_ch2',
               'third_harmonic_ch2']

for idx, folder in enumerate(folder_exts):
    open_path = main_path+folder+'/'
    t1 = temp_ranges[idx][0]
    t2 = temp_ranges[idx][1]+1
    for temp in range(t1,t2):
        file_name = open_path+base_file_name+str(temp)+file_ext
        df = drop_nans(pd.read_csv(file_name))

        df, peaks_voltage = extract_sweep_peaks(df,'voltage_amp_ch2','voltage_sweep_ch2','sweep_')

        df, peaks_current = extract_sweep_peaks(df, 'ac_current_ch2','current_sweep_ch2','I = ')



        fig, axs = MakePlot().create()
        sns.scatterplot(y='resistance_ch2', x='voltage_amp_ch2',hue='current_sweep_ch2', data=df, ax=axs)
        plt.title('Resistance by Voltage')
        axs.set_ylabel('R')
        axs.set_xlabel('V')
        anchored_text = AnchoredText('T = ' + str(temp) + 'K', loc='upper left')
        axs.add_artist(anchored_text)
        legend = plt.legend()
        plt.show()
        print('')

        fig, axs = MakePlot().create()
        sns.scatterplot(y='resistance_ch2', x='ac_current_ch2', hue='current_sweep_ch2', data=df, ax=axs)
        plt.title('Resistance by Current')
        axs.set_ylabel('R')
        axs.set_xlabel('I')
        anchored_text = AnchoredText('T = ' + str(temp) + 'K', loc='upper right')
        axs.add_artist(anchored_text)
        plt.show()
        print('')

        fig, axs = MakePlot().create()
        sns.scatterplot(y='resistance_ch2', x='b_field', hue='current_sweep_ch2', data=df, ax=axs)
        plt.title('Resistance by Magnetic Field')
        axs.set_ylabel('R')
        axs.set_xlabel('B')
        anchored_text = AnchoredText('T = ' + str(temp) + 'K', loc='upper right')
        axs.add_artist(anchored_text)
        plt.show()
        print('')

        fig, axs = MakePlot().create()
        sns.scatterplot(y='resistance_ch2', x='b_field', hue='voltage_sweep_ch2', data=df, ax=axs)
        plt.title('Resistance by Magnetic Field')
        axs.set_ylabel('R')
        axs.set_xlabel('B')
        anchored_text = AnchoredText('T = ' + str(temp) + 'K', loc='upper right')
        axs.add_artist(anchored_text)
        plt.show()
        print('')

        fig, axs = MakePlot().create()
        sns.scatterplot(y='ac_current_ch2', x='b_field', hue='voltage_sweep_ch2', data=df, ax=axs)
        plt.title('Current by Magnetic Field')
        axs.set_ylabel('I')
        axs.set_xlabel('B')
        anchored_text = AnchoredText('T = ' + str(temp) + 'K', loc='upper right')
        axs.add_artist(anchored_text)
        plt.show()
        print('')

        fig, axs = MakePlot().create()
        sns.scatterplot(y='voltage_amp_ch2', x='b_field', hue='current_sweep_ch2', data=df, ax=axs)#, legend_out=True)
        plt.title('Current by Magnetic Field')
        axs.set_ylabel('V')
        axs.set_xlabel('B')
        anchored_text = AnchoredText('T = ' + str(temp) + 'K', loc='upper right')
        axs.add_artist(anchored_text)
        plt.show()
        print('')

        '''

        fig, axs = MakePlot().create()
        sns.scatterplot(y='ac_current_ch2', x='time',hue='current_sweep_ch2',data=df, ax=axs)
        #sns.scatterplot(y='ac_current_ch2', x='time',data=df, ax=axs)
        #plt.plot(np.diff(df.ac_current_ch2))
        #for peak in peaks_current: plt.axvline(peak)
        plt.title('Current in Time')
        axs.set_ylabel('Current')
        axs.set_xlabel('Time')
        anchored_text = AnchoredText('T = ' + str(temp) + 'K', loc='upper left')
        axs.add_artist(anchored_text)
        plt.show()
        print('')


        fig, axs = MakePlot().create()
        sns.scatterplot(y='voltage_amp_ch2', x='time',hue='current_sweep_ch2',data=df, ax=axs)
        #plt.plot(np.diff(df.voltage_amp_ch2))
        #for peak in peaks: plt.axvline(peak)
        plt.title('Voltage in Time')
        axs.set_ylabel('Voltage')
        axs.set_xlabel('Time')
        anchored_text = AnchoredText('T = ' + str(temp) + 'K', loc='upper left')
        axs.add_artist(anchored_text)
        plt.show()
        print('')
        '''