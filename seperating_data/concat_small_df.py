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
              'FeSb2_data2'
              ,'SmB6_data1']

base_file_name = 'constant_temp_'
file_ext = '.csv'

temp_ranges = [(2,30),
               (2,10)
               ,(2,23)]

headers_ch1 = ['b_field', 'chamber_pressure', 'sigma_resistance_ch1',
               'phase_angle_ch1', 'ac_current_ch1', 'voltage_amp_ch1', 'in_phase_voltage_ch1',
               'quad_voltage_ch1', 'gain_ch1', 'second_harmonic_ch1', 'third_harmonic_ch1']
headers_ch2 = ['b_field', 'chamber_pressure', 'sigma_resistance_ch2', 'phase_angle_ch2', 'ac_current_ch2',
               'voltage_amp_ch2',
               'in_phase_voltage_ch2', 'quad_voltage_ch2', 'gain_ch2', 'second_harmonic_ch2',
               'third_harmonic_ch2']

current_names = ['I = 1.0', 'I = 5.0', 'I = 10.0', 'I = 20.0','I = 50.0','I = 100.0' , 'I = 200.0' ,  'I = 500.0','I = 1000.0', 'I = 1500.0' ]
currents = [1, 5, 10,20,50,100,200 ,500,1000, 1500 ]

channel = 'ch2'

def load_w_exceptions(open_path,file_name,nan_on_two):
    if open_path + file_name == '/Users/npopiel/Documents/MPhil/Data/csvs_by_temp/FeSb2_data1/data/ch2/current_1500temp_4.csv':
        return
    if open_path + file_name == '/Users/npopiel/Documents/MPhil/Data/csvs_by_temp/FeSb2_data2/data/ch2/current_10temp_10.csv':
        return
    if open_path+file_name == '/Users/npopiel/Documents/MPhil/Data/csvs_by_temp/FeSb2_data2/data/ch2/current_20temp_10.csv':
        return
    if open_path+file_name == '/Users/npopiel/Documents/MPhil/Data/csvs_by_temp/FeSb2_data2/data/ch2/current_50temp_10.csv':
        return
    if open_path+file_name == '/Users/npopiel/Documents/MPhil/Data/csvs_by_temp/FeSb2_data2/data/ch2/current_100temp_10.csv':
        return
    if open_path+file_name == '/Users/npopiel/Documents/MPhil/Data/csvs_by_temp/FeSb2_data2/data/ch2/current_200temp_10.csv':
        return
    if open_path+file_name == '/Users/npopiel/Documents/MPhil/Data/csvs_by_temp/FeSb2_data2/data/ch2/current_500temp_10.csv':
        return
    if open_path+file_name == '/Users/npopiel/Documents/MPhil/Data/csvs_by_temp/FeSb2_data2/data/ch2/current_1000temp_10.csv':
        return
    if open_path+file_name == '/Users/npopiel/Documents/MPhil/Data/csvs_by_temp/FeSb2_data2/data/ch2/current_1500temp_10.csv':
        return

    return drop_nans(pd.read_csv(open_path + file_name), nan_on_two=nan_on_two)


for idx, folder in enumerate(folder_exts):
    open_path = main_path+folder+'/data/'+ channel+'/'
    t1 = temp_ranges[idx][0]
    t2 = temp_ranges[idx][1]+1

    df_currents = []

    if channel == 'ch2': nan_on_two = False
    else: nan_on_two = True

    for current in currents:

        df_temps = []

        for temp in range(t1,t2):
            file_name = 'current_' + str(current) + 'temp_' + str(temp) + '.csv'

            df = load_w_exceptions(open_path,file_name,nan_on_two)

            df_temps.append(df)

        big_df_change_temp = pd.concat(df_temps)
        df_currents.append(big_df_change_temp)


        print()

    proper_big_df = pd.concat(df_currents)

    proper_big_df = remove_irrelevant_columns(proper_big_df)
    save_name = main_path+folder+'/big_df_'+channel+'.csv'
    proper_big_df.to_csv(save_name,index=False)

    print('/Volumes/GoogleDrive/Shared drives/Quantum Materials/Popiel')

