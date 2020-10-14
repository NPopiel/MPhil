import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tools.utils import *
from tools.DataFile import DataFile
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
    save_path = open_path + 'figures/'
    for temp in range(t1,t2):
        file_name = open_path+base_file_name+str(temp)+file_ext
        df = pd.read_csv(file_name)

        fig, axs = MakePlot().create()
        sns.scatterplot(y=df.resistance_ch2, x=df.voltage_amp_ch2, ax=axs)
        plt.title('Resistance (CH2) as a Function of Voltage')
        axs.set_xlabel('Voltage')
        axs.set_ylabel('Resistance')
        anchored_text = AnchoredText('T = ' + str(temp) + 'K', loc='upper left')
        axs.add_artist(anchored_text)
        plt.show()
        print('')


        fig, axs = MakePlot().create()
        sns.scatterplot(y=df.resistance_ch2, x=df.time, ax=axs)
        plt.title('Resistance (CH1) as a Function of Time')
        axs.set_xlabel('Time')
        axs.set_ylabel('Resistance')
        anchored_text = AnchoredText('T = ' + str(temp) + 'K', loc='upper left')
        axs.add_artist(anchored_text)
        plt.show()
        print('')

        fig, axs = MakePlot().create()
        sns.scatterplot(y=df.resistance_ch2, x=df.b_field, ax=axs)
        plt.title('Resistance (CH2) as a Function of Magnetic Field')
        axs.set_xlabel('Magnetic Field')
        axs.set_ylabel('Resistance')
        anchored_text = AnchoredText('T = ' + str(temp) + 'K', loc='upper left')
        axs.add_artist(anchored_text)
        plt.show()
        print('')

        fig, axs = MakePlot().create()
        sns.scatterplot(y=df.resistance_ch2, x=df.ac_current_ch2, ax=axs)
        plt.title('Resistance (CH2) as a Function of Current')
        axs.set_xlabel('Current')
        axs.set_ylabel('Resistance')
        anchored_text = AnchoredText('T = ' + str(temp) + 'K', loc='upper left')
        axs.add_artist(anchored_text)
        plt.show()
        print('')

        fig, axs = MakePlot().create()
        sns.scatterplot(y=df.voltage_amp_ch2, x=df.ac_current_ch2, ax=axs)
        plt.title('Voltage (CH2) as a Function of Current')
        axs.set_xlabel('Current')
        axs.set_ylabel('Voltage')
        anchored_text = AnchoredText('T = ' + str(temp) + 'K', loc='upper left')
        axs.add_artist(anchored_text)
        plt.show()
        print('')

        #df = extract_changing_field(df, col_name='b_field', new_col_name='b_flag', root_flag_marker='b')


        new_headers = df.columns



