import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tools.utils import *
from tools.DataFile import DataFile
from tools.MakePlot import MakePlot
from matplotlib.offsetbox import AnchoredText

main_path = '/Users/npopiel/Documents/MPhil/Data/data_csvs_cleaned/'

file_names = ['FeSb2_data1.csv',
              'FeSb2_data2.csv']

df = pd.read_csv(main_path+file_names[0])

df = extract_changing_field(df, col_name='b_field', new_col_name='b_flag',root_flag_marker='b')

df, locs=extract_stepwise_peaks(df,'temp','temp_flag','const_temp_')



w = df.groupby('temp_flag')

new_headers = df.columns

headers_ch1 = ['b_field', 'chamber_pressure','sigma_resistance_ch1',
           'phase_angle_ch1', 'ac_current_ch1','voltage_amp_ch1', 'in_phase_voltage_ch1',
           'quad_voltage_ch1','gain_ch1', 'second_harmonic_ch1', 'third_harmonic_ch1']
headers_ch2 = ['b_field', 'chamber_pressure','sigma_resistance_ch2', 'phase_angle_ch2','ac_current_ch2', 'voltage_amp_ch2',
           'in_phase_voltage_ch2','quad_voltage_ch2', 'gain_ch2', 'second_harmonic_ch2',
           'third_harmonic_ch2']
nice_names_ch1 = ['Magnetic Field', 'Chamber Pressure','Standard Deviation of Resistance (CH1)',
           'Phase Angle (CH1)', 'AC Current (CH1)','Voltage Amplitude (CH1)', 'In Phase Voltage (CH1)',
           'Quadrature Voltage (CH1)','Gain (CH1)', 'Second Harmonic (CH1)', 'Third Harmonic (CH1)']
nice_names_ch2 = ['Magnetic Field', 'Chamber Pressure','Standard Deviation of Resistance (CH2)', 'Phase Angle (CH2)','AC Current (CH2)', 'Voltage Amplitude (CH2)',
           'In Phase Voltage (CH2)','Quadrature Voltage (CH2)', 'Gain (CH2)', 'Second Harmonic (CH2)',
           'Third Harmonic (CH2)']

headers_ch1_short = ['voltage_amp_ch1','ac_current_ch1','sigma_resistance_ch1',
           'phase_angle_ch1',  'in_phase_voltage_ch1',
           'quad_voltage_ch1','gain_ch1', 'second_harmonic_ch1', 'third_harmonic_ch1']
nice_names_ch1_short = ['Voltage Amplitude (CH1)','AC Current (CH1)','Standard Deviation of Resistance (CH1)',
           'Phase Angle (CH1)',  'In Phase Voltage (CH1)',
           'Quadrature Voltage (CH1)','Gain (CH1)', 'Second Harmonic (CH1)', 'Third Harmonic (CH1)']

for ind, header in enumerate(headers_ch1_short):

    for constant_temp, inds in w.groups.items():
        fig, axs = MakePlot().create()
        sns.scatterplot(y=df[header][inds],x=inds, ax=axs)
        #plt.plot(np.diff(df.temp[inds]))
        #for peak in locs: plt.axvline(peak)
        plt.title(nice_names_ch1_short[ind])
        axs.set_xlabel('Time')
        axs.set_ylabel(nice_names_ch1_short[ind])
        temp = round(float(constant_temp.split('_')[-1]))
        anchored_text = AnchoredText('T = ' + str(temp) + 'K', loc='upper left')
        axs.add_artist(anchored_text)
        plt.show()
        print('')
'''
    for constant_temp, inds in w.groups.items():
        fig, axs = MakePlot().create()
        sns.scatterplot(y=df.resistance_ch1[inds], x=df[header][inds])
        plt.title('Resistance (CH1) as a Function of ' + nice_names_ch1[ind])
        axs.set_xlabel(nice_names_ch1[ind])
        axs.set_ylabel('Resistance')
        temp = round(float(constant_temp.split('_')[-1]))
        anchored_text = AnchoredText('T = '+str(temp) + 'K', loc='upper left')
        axs.add_artist(anchored_text)
        #plt.show()
        #print('')
        makedir(main_path+'plots/')
        makedir(main_path+'plots/'+nice_names_ch1[ind]+'/')
        plt.savefig(main_path+'plots/'+nice_names_ch1[ind]+'/'+str(temp)+'.png', dpi=200)
        plt.close()


for ind, header in enumerate(headers_ch1):
    for constant_temp, inds in w.groups.items():
        fig, axs = MakePlot().create()
        sns.scatterplot(y=df.resistance_ch1[inds], x=df[header][inds])
        plt.title('Resistance (CH1) as a Function of ' + nice_names_ch1[ind])
        axs.set_xlabel(nice_names_ch1[ind])
        axs.set_ylabel('Resistance')
        temp = round(float(constant_temp.split('_')[-1]))
        anchored_text = AnchoredText('T = '+str(temp) + 'K', loc='upper left')
        axs.add_artist(anchored_text)
        #plt.show()
        #print('')
        makedir(main_path+'plots/')
        makedir(main_path+'plots/'+nice_names_ch1[ind]+'/')
        plt.savefig(main_path+'plots/'+nice_names_ch1[ind]+'/'+str(temp)+'.png', dpi=200)
        plt.close()

for ind, header in enumerate(headers_ch2):
    for constant_temp, inds in w.groups.items():
        fig, axs = MakePlot().create()
        sns.scatterplot(y=df.resistance_ch2[inds], x=df[header][inds])
        plt.title('Resistance (CH2) as a unction of ' + header)
        plt.title(header)
        plt.show()
        print('')
        plt.close(fig)

'''

'''
for constant_temp, inds in w.groups.items():
    fig, axs = MakePlot().create()
    sns.scatterplot(y=df.resistance_ch2[inds], x=df.b_field[inds])
    # plt.plot(np.diff(df.temp))
    # for peak in peaks: plt.axvline(peak)
    plt.title(constant_temp)
    plt.show()
    print('')
    
'''

print(w)


#new_df = extract_stepwise_peaks(df,'temp','temp_flag','constant_temp_',)







#fig, axs = MakePlot().create()




print(df)

