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
               'FeSb2_data2']
# ,'SmB6_data1']

base_file_name = 'constant_temp_'
file_ext = '.csv'

temp_ranges = [(2, 30),
               (2, 10),
               (2,23)]

headers_ch1 = ['b_field', 'chamber_pressure', 'sigma_resistance_ch1',
               'phase_angle_ch1', 'ac_current_ch1', 'voltage_amp_ch1', 'in_phase_voltage_ch1',
               'quad_voltage_ch1', 'gain_ch1', 'second_harmonic_ch1', 'third_harmonic_ch1']
headers_ch2 = ['b_field', 'chamber_pressure', 'sigma_resistance_ch2', 'phase_angle_ch2', 'ac_current_ch2',
               'voltage_amp_ch2',
               'in_phase_voltage_ch2', 'quad_voltage_ch2', 'gain_ch2', 'second_harmonic_ch2',
               'third_harmonic_ch2']

channel = 'ch2'

for idx, folder in enumerate(folder_exts):
    open_path = main_path + folder + '/'
    t1 = temp_ranges[idx][0]
    t2 = temp_ranges[idx][1] + 1
    save_path = open_path + 'data/'
    makedir(save_path)
    save_path = save_path + channel + '/'
    makedir(save_path)
    for temp in range(t1, t2):
        file_name = open_path + base_file_name + str(temp) + file_ext
        df_og = pd.read_csv(file_name)

        # Here it is important to not what happens. The PPMS measures on two channels, oscillating from each channel. While a measurement is being
        # made on channel two, channel one reports a nan and vice versa. The drop_nans function essentially just returns everyother row of the dataframe
        # and to ensure that it makes sense i.e. pick out whether channel one or two has a nan first you need to use the boolean nan_on_two parameter
        # it should be True for ch1 and False for ch2

        if channel == 'ch2': nan_on_two = False
        else: nan_on_two = True

        df = drop_nans(df_og,nan_on_two=nan_on_two)

        df, peaks_current = extract_sweep_peaks(df, 'ac_current_'+channel, 'current_sweep_'+channel, 'I = ')

        groupers = df.groupby('current_sweep_'+channel)

        for current, inds in groupers.groups.items():
            subsection = df[df['current_sweep_'+channel] == current]
            current = round(float(current.split(' ')[-1]))
            save_name = save_path + 'current_'+str(current)+ 'temp_' + str(temp) + '.csv'
            subsection.to_csv(save_name, index=False)


