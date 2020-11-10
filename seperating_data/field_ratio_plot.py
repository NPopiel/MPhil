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

# titles1 = ['Resistance by Temperature (VT26)', 'Resistance by Temperature (VT49)', 'Resistance by Temperature (SmB6)']
# titles2 = ['Resistance by Current (VT26)', 'Resistance by Current (VT49)', 'Resistance by Current (SmB6)']
# titles3 = ['Resistance by Voltage (VT26)', 'Resistance by Voltage (VT49)', 'Resistance by Voltage (SmB6)']
# samples = ['VT26','VT49','SmB6']

titles1 = ['Resistance by Temperature (VT1)', 'Resistance by Temperature (VT51)', 'Resistance by Temperature (VT11)']
titles2 = ['Resistance by Current (VT1)', 'Resistance by Current (VT51)', 'Resistance by Current (VT11)']
titles3 = ['Resistance by Voltage (VT1)', 'Resistance by Voltage (VT51)', 'Resistance by Voltage (VT11)']
samples = ['VT1','VT51','VT11']

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
    temps = []
    resistance_ratios, voltage_ratios = [], []
    current_for_ratio = []

    for temp in range(t1, t2):
        file_name = open_path + base_file_name + str(temp) + file_ext
        df_og = pd.read_csv(file_name)

        df = drop_nans(df_og, nan_on_two=nan_on_two)

        df=extract_changing_field(df,'b_field','b_flag','constant_b_')

        df, peaks_current = extract_sweep_peaks(df, 'ac_current_'+channel, 'current_sweep_'+channel, 'I = ')

        max_locs, min_locs = find_b_extrma(df)

        resistance_max_b = np.array(df['resistance_'+channel])[max_locs]
        resistance_min_b = np.array(df['resistance_'+channel])[min_locs]

        current_max_b = np.array(df['ac_current_'+channel])[max_locs]
        current_min_b = np.array(df['ac_current_'+channel])[min_locs]

        voltage_max_b = np.array(df['voltage_amp_'+channel])[max_locs]
        voltage_min_b = np.array(df['voltage_amp_'+channel])[min_locs]

        resistance_ratio, voltage_ratio, current_ratio = [], [], []

        if len(current_max_b)>len(current_min_b):
            relevant_indices = np.arange(len(current_min_b))
        else:
            relevant_indices = np.arange(len(current_max_b))

        for ind in relevant_indices:
            if current_min_b[ind] == current_max_b[ind]:
                resistance_ratio.append(resistance_max_b[ind]/resistance_min_b[ind])
                voltage_ratio.append(voltage_max_b[ind]/voltage_min_b[ind])
                current_ratio.append(current_min_b[ind])

        resistance_ratios.append(resistance_ratio)
        voltage_ratios.append(voltage_ratio)
        temps.append([temp]*len(voltage_ratio))
        current_for_ratio.append(current_ratio)


    diction = {'Resistance':np.array(flatten(resistance_ratios))*100,
               'Current':np.array(flatten(current_for_ratio))*1000,
               'Voltage':flatten(voltage_ratios),
               'Temperature':flatten(temps)}

    data = pd.DataFrame(diction)


    fig, axs = MakePlot(figsize=(16, 9)).create()
    sns.set_palette('bright')
    sns.lineplot(y='Resistance', x='Temperature',hue='Current',palette='bright',data=data, ax=axs, linestyle='-', legend=True)
    axs.set_ylabel(r'R',rotation=0, fontsize=16)
    axs.set_xlabel(r'T',fontsize=16)
    axs.set_title(titles1[idx],fontsize=22)
    name1 = main_path+'r_v_temp_'+samples[idx]+'.pdf'
    plt.savefig(name1,dpi=200)
    #plt.show()
    plt.close()


    fig, axs = MakePlot(figsize=(16, 9)).create()
    sns.set_palette('bright')
    sns.scatterplot(y='Resistance', x='Current',hue='Temperature',palette='bright',data=data, ax=axs, legend=True)
    axs.set_ylabel(r'R', rotation=0, fontsize=16)
    axs.set_xlabel(r'I', fontsize=16)
    axs.set_title(titles2[idx],fontsize=22)
    name2 = main_path+'r_v_curr_'+samples[idx]+'.pdf'
    plt.savefig(name2,dpi=200)
    plt.close()

    #plt.show()

    fig, axs = MakePlot(figsize=(16, 9)).create()
    sns.set_palette('bright')
    sns.scatterplot(y='Resistance', x='Voltage',hue='Temperature',palette='bright',data=data, ax=axs, legend=True)
    axs.set_ylabel(r'R', rotation=0, fontsize=16)
    axs.set_xlabel(r'V', fontsize=16)
    axs.set_title(titles3[idx],fontsize=22)
    name3 = main_path+'r_v_volt_'+samples[idx]+'.pdf'
    plt.savefig(name3,dpi=200)
    plt.close()

    #plt.show()

    print()

