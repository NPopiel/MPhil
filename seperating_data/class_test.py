import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tools.utils import *
from tools.DataFile import DataFile
from tools.MakePlot import MakePlot




filename = '/Users/npopiel/Documents/MPhil/Data/data/3Kup_FeSb2_VT11&SmB6_sbf25-19_EVERYTHING-GrapheneCentre.dat'


# here are the original column names for reference
og_col_names = ['Time Stamp (s)', 'Temperature (K)', 'Field (Oe)', 'Sample Position (deg)', 'Chamber Pressure (Torr)',
                'Resistance Ch1 (Ohms)', 'Resistance Std. Dev. Ch1 (Ohms)', 'Phase Angle Ch1 (deg)',
                'I-V Current Ch1 (mA)', 'I-V Voltage Ch1 (V)', 'Frequency Ch1 (Hz)', 'Averaging Time Ch1 (s)',
                'AC Current Ch1 (mA)', 'DC Current Ch1 (mA)', 'Voltage Ampl Ch1 (V)', 'In Phase Voltage Ampl Ch1 (V)',
                'Quadrature Voltage Ch1 (V)', 'AC Voltage Ch1 (V)', 'DC Voltage Ch1 (V)', 'Current Ampl Ch1 (mA)',
                'In Phase Current Ch1 (mA)', 'Quadrature Current Ch1 (mA)', 'Gain Ch1', '2nd Harmonic Ch1 (dB)',
                '3rd Harmonic Ch1 (dB)', 'Resistance Ch2 (Ohms)', 'Resistance Std. Dev. Ch2 (Ohms)',
                'Phase Angle Ch2 (deg)', 'I-V Current Ch2 (mA)', 'I-V Voltage Ch2 (V)', 'Frequency Ch2 (Hz)',
                'Averaging Time Ch2 (s)', 'AC Current Ch2 (mA)', 'DC Current Ch2 (mA)', 'Voltage Ampl Ch2 (V)',
                'In Phase Voltage Ampl Ch2 (V)', 'Quadrature Voltage Ch2 (V)', 'AC Voltage Ch2 (V)',
                'DC Voltage Ch2 (V)', 'Current Ampl Ch2 (mA)', 'In Phase Current Ch2 (mA)',
                'Quadrature Current Ch2 (mA)', 'Gain Ch2', '2nd Harmonic Ch2 (dB)', '3rd Harmonic Ch2 (dB)',
                'ETO Status Code', 'ETO Measurement Mode', 'Temperature Status (code)', 'Field Status (code)',
                'Chamber Status (code)', 'ETO Channel 1', 'ETO Channel 2', 'ETO Channel 3', 'ETO Channel 4',
                'ETO Channel 5', 'ETO Channel 6', 'ETO Channel 7', 'ETO Channel 8', 'ETO Channel 9', 'ETO Channel 10',
                'ETO Channel 11', 'ETO Channel 12', 'ETO Channel 13', 'ETO Channel 14', 'ETO Channel 15',
                'ETO Channel 16\n']

# the new column names I created
new_col_names = ['time', 'temp', 'b_field', 'samp_degrees', 'chamber_pressure',
                'resistance_ch1', 'sigma_resistance_ch1', 'phase_angle_ch1',
                'iv_current_ch1', 'iv_voltage_ch1', 'frequency_ch1', 'avging_time_ch1',
                'ac_current_ch1', 'dc_current_ch1', 'voltage_amp_ch1', 'in_phase_voltage_ch1',
                'quad_voltage_ch1', 'ac_voltage_ch1', 'dc_voltage_ch1', 'current_amp_ch1',
                'in_phase_current_ch1', 'quad_current_ch1', 'gain_ch1', 'second_harmonic_ch1',
                'third_harmonic_ch1', 'resistance_ch2', 'sigma_resistance_ch2', 'phase_angle_ch2',
                'iv_current_ch2', 'iv_voltage_ch2', 'frequency_ch2', 'avging_time_ch2',
                'ac_current_ch2', 'dc_current_ch2', 'voltage_amp_ch2', 'in_phase_voltage_ch2',
                'quad_voltage_ch2', 'ac_voltage_ch2', 'dc_voltage_ch2', 'current_amp_ch2',
                'in_phase_current_ch2', 'quad_current_ch2', 'gain_ch2', 'second_harmonic_ch2',
                'third_harmonic_ch2','eto_code', 'eto_mode', 'temp_stat', 'field_stat',
                'chamber_stat', 'eto_ch1', 'eto_ch2', 'eto_ch3', 'eto_ch4',
                'eto_ch5', 'eto_ch6', 'eto_ch7', 'eto_ch8', 'eto_ch9', 'eto_ch10',
                'eto_ch11', 'eto_ch12', 'eto_ch13', 'eto_ch14', 'eto_ch15',
                'eto_ch16']



lines_to_skip = 18
delimeter = ','
row_after_header_useless = True
delete_comment_flag = True
new_headers = new_col_names
convert_b_flag = True
cols_to_remove = None

parameters = [lines_to_skip,
              delimeter,
              row_after_header_useless,
              delete_comment_flag,
              new_headers,
              convert_b_flag,
              cols_to_remove]

file = DataFile(filename, parameters)

df = file.open()

df = remove_irrelevant_columns(df)

df= remove_constant_column(df)

# I am just going to assume the ETO Code is not relevant even though it changes therefore I'll put it in as a parameter
 # in remove constant column



df = extract_changing_field(df, col_name='b_field', new_col_name='b_flag',root_flag_marker='b')

df=extract_stepwise_peaks(df,'temp','temp_flag','const_temp_')

w = df.groupby('temp_flag')

for constant_temp, inds in w.groups.items():
    fig, axs = MakePlot().create()
    sns.scatterplot(y=df.resistance_ch2[inds], x=df.b_field[inds])
    # plt.plot(np.diff(df.temp))
    # for peak in peaks: plt.axvline(peak)
    plt.title(constant_temp)
    plt.show()
    print(w)


print(w)


#new_df = extract_stepwise_peaks(df,'temp','temp_flag','constant_temp_',)






'''

new_headers = df.columns

for header in new_headers:
    fig, axs = MakePlot().create()
    plt.plot(df[header])
    #sns.scatterplot(x=df[header][:1000], y=df.resistance_ch2[:1000])
    #plt.title('Resistance as a function of ' + header)
    plt.title(header)
    plt.show()
    print('')
    plt.close(fig)
'''

#fig, axs = MakePlot().create()




print(df)

