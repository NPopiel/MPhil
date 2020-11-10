from tools.utils import *
from tools.MakePlot import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import find_peaks

main_path = '/Users/npopiel/Documents/MPhil/Data/'

file_names = ['FeSb2_data1.csv',
              'FeSb2_data2.csv',
              'SmB6_data1.csv']

for ind, file_name in enumerate(file_names):

    df = pd.read_csv(main_path + 'data_csvs_cleaned/' + file_name)

    fig, ax = MakePlot().create()

    plt.scatter(y=df.resistance_ch2, x=df.b_field)
    plt.show()

    df, peaks_voltage = extract_sweep_peaks(df, 'voltage_amp_ch2', 'voltage_sweep_ch2', 'sweep_')
    df, peaks_current = extract_sweep_peaks(df, 'ac_current_ch2', 'current_sweep_ch2', 'I = ')

    fig, ax = MakePlot().create()
    sns.scatterplot(y='resistance_ch2', x='b_field', data=df, hue='current_sweep_ch2')
    plt.show()

    fig, ax = MakePlot().create()
    sns.scatterplot(y='resistance_ch2', x='temp', data=df, hue='current_sweep_ch2')
    plt.show()

    fig, ax = MakePlot().create()
    sns.scatterplot(y='b_field', x='temp', data=df, hue='current_sweep_ch2')
    plt.show()

    fig, ax = MakePlot().create()
    sns.scatterplot(y=df.b_field[5000:10000], x=df.time[5000:10000])
    plt.show()

    df = extract_changing_field(df,'b_field','b_sweep','sweep')

    peaks, properties = find_peaks(np.array(df.b_field))

    # CHeck if peaks are constant differennce
    # for peak1, peak2 in zip(peaks[:-1],peaks[1:]):
    #     print(peak2-peak1)
    fig, ax = MakePlot().create()
    sns.scatterplot(y=df.b_field, x=np.arange(len(df.b_field)))
    plt.axvline(peaks.any())
    plt.show()

    fig, ax = MakePlot().create()
    sns.scatterplot(y='b_field', x='time', data=df, hue='b_sweep')
    plt.show()


    df, locs = extract_stepwise_peaks(df, 'temp', 'temp_flag', 'const_temp_')

    groupers = df.groupby('temp_flag')

    new_headers = df.columns

    makedir(main_path + 'csvs_by_temp/')
    base_save_path = main_path + 'csvs_by_temp/' + file_names[ind].split('.')[0] + '/'
    makedir(base_save_path)

    for constant_temp, inds in groupers.groups.items():
        subsection = df[df.temp_flag == constant_temp]
        temp = round(float(constant_temp.split('_')[-1]))
        save_name = base_save_path + 'constant_temp_' + str(temp) + '.csv'
        subsection.to_csv(save_name, index=False)

