import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


flatten = lambda l: [item for sublist in l for item in sublist]

filename = '/Users/npopiel/Documents/MPhil/Data/data/3Kup_FeSb2_VT11&SmB6_sbf25-19_EVERYTHING-GrapheneCentre.txt'
'''
with open(filename, 'r') as the_file:
    for line in the_file.readlines():
        all_data = line.split(',')

    height_line = all_data[3]
    data = all_data[8:]
'''

with open(filename, 'r') as the_file:
    all_data = [line.split(',') for line in the_file.readlines()]
    dat_arr = np.array(flatten(all_data)).reshape((len(all_data), len(all_data[0])))
the_file.close()

# Delete the column called comment
dat = np.delete(dat_arr,0,1)

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

# loop over the columns and update the names
for ind, name in enumerate(new_col_names):
    dat[0,ind] = name

# convert to pandas dataframe
df = pd.DataFrame(dat)

# Change new labels to the header
df.columns = df.iloc[0]
df = df[1:]

# ensure data is a float, converting all empty and string values to nan
df = df.apply(pd.to_numeric, errors='coerce')


# convert oersted to tesla
df['b_field'] = df['b_field']/ 10000

# Pseudo code to detect where b_field changes
# Either I could consider the mean of a rolling window
# or loop over values to determine when it changes

b_vals = df['b_field']

b_deriv = np.diff(b_vals)

# ~ 65 vals at 14 T, ~65 vals @ 0T, 65 vals @ 14T
# Therefore I can probably count ~ 70 values once it gets within 0.1  T of target

print(dat)