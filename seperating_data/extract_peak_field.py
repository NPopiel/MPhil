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
    max_r_vals, min_r_vals = [], []
    max_i_vals, min_i_vals = [], []
    max_v_vals, min_v_vals = [], []
    temps_max, temps_min = [], []
    temps, max_flag, min_flag, flag = [], [], [], []
    resistances, currents, voltages = [], [], []
    ratio = []
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

        max_r_vals.append(resistance_max_b)
        min_r_vals.append(resistance_min_b)
        max_i_vals.append(current_max_b)
        min_i_vals.append(current_min_b)
        max_v_vals.append(voltage_max_b)
        min_v_vals.append(voltage_min_b)
        temps_max.append([temp]*len(voltage_max_b))
        temps_min.append([temp]*len(voltage_min_b))
        max_flag.append(['Max']*len(resistance_max_b))
        min_flag.append(['Min']*len(resistance_min_b))

    resistances.append(flatten(max_r_vals))
    resistances.append(flatten(min_r_vals))
    currents.append(flatten(max_i_vals))
    currents.append(flatten(min_i_vals))
    voltages.append(flatten(max_v_vals))
    voltages.append(flatten(min_v_vals))
    temps.append(flatten(temps_max))
    temps.append(flatten(temps_min))
    flag.append(flatten(max_flag))
    flag.append(flatten(min_flag))
    #ratio.append(flatten())

    diction = {'Resistance':flatten(resistances),
               'Current':np.array(flatten(currents))*1000,
               'Voltage':flatten(voltages),
               'Temperature':flatten(temps),
               'Field':flatten(flag)}

    data = pd.DataFrame(diction)


    fig, axs = MakePlot(figsize=(16, 9)).create()
    sns.set_palette('bright')
    sns.scatterplot(y='Resistance', x='Temperature',style='Field',hue='Current',palette='bright',data=data, ax=axs, legend=True)
    axs.set_ylabel(r'R',rotation=0, fontsize=16)
    axs.set_xlabel(r'T',fontsize=16)
    axs.set_title(titles1[idx],fontsize=22)
    name1 = main_path+'r_v_temp_'+samples[idx]+'.pdf'
    plt.savefig(name1,dpi=200)
    #plt.show()
    plt.close()


    fig, axs = MakePlot(figsize=(16, 9)).create()
    sns.set_palette('bright')
    sns.scatterplot(y='Resistance', x='Current',style='Field',hue='Temperature',palette='bright',data=data, ax=axs, legend=True)
    axs.set_ylabel(r'R', rotation=0, fontsize=16)
    axs.set_xlabel(r'I', fontsize=16)
    axs.set_title(titles2[idx],fontsize=22)
    name2 = main_path+'r_v_curr_'+samples[idx]+'.pdf'
    plt.savefig(name2,dpi=200)
    plt.close()

    #plt.show()

    fig, axs = MakePlot(figsize=(16, 9)).create()
    sns.set_palette('bright')
    sns.scatterplot(y='Resistance', x='Voltage',style='Field',hue='Temperature',palette='bright',data=data, ax=axs, legend=True)
    axs.set_ylabel(r'R', rotation=0, fontsize=16)
    axs.set_xlabel(r'V', fontsize=16)
    axs.set_title(titles3[idx],fontsize=22)
    name3 = main_path+'r_v_volt_'+samples[idx]+'.pdf'
    plt.savefig(name3,dpi=200)
    plt.close()

    #plt.show()

    print()

