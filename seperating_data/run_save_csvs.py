import pandas as pd
from tools.utils import *
from tools.DataFile import DataFile

main_path = '/Users/npopiel/Documents/MPhil/Data/'

filenames = ['data_original/FeSb2_VT11&SmB6_sbf25-19_EVERYTHING-GrapheneCentre.dat',
             'data_original/3Kup_FeSb2_VT11&SmB6_sbf25-19_EVERYTHING-GrapheneCentre.dat',
             'data_original/FeSb2_VT1_VT26_EVERYTHING-GrapheneCentre.dat',
             'data_original/4Kup_FeSb2_VT1_VT26_EVERYTHING-GrapheneCentre.dat',
             'data_original/FeSb2_VT51_VT49_EVERYTHING-GrapheneCentre.dat',
             'data_original/7Kup_FeSb2_VT51_VT49_EVERYTHING-GrapheneCentre.dat',
             'data_original/10Kup_FeSb2_VT51_VT49_EVERYTHING-GrapheneCentre.dat']


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
                 'third_harmonic_ch2', 'eto_code', 'eto_mode', 'temp_stat', 'field_stat',
                 'chamber_stat', 'eto_ch1', 'eto_ch2', 'eto_ch3', 'eto_ch4',
                 'eto_ch5', 'eto_ch6', 'eto_ch7', 'eto_ch8', 'eto_ch9', 'eto_ch10',
                 'eto_ch11', 'eto_ch12', 'eto_ch13', 'eto_ch14', 'eto_ch15',
                 'eto_ch16']

lines_to_skip = 18
delimeter = ','
row_after_header_useless = True
row_before_header_useless = False
delete_comment_flag = True
new_headers = new_col_names
convert_b_flag = True
cols_to_remove = None

test_lst = []

lines_per_file = [18,18,16,16,19,19,19]
rows_after_header_useless = True
#rows_before_header_useless = [False, False, False, False, True, True, True]

for ind, name in enumerate(filenames):

    filename = main_path + name

    parameters = [lines_per_file[ind],
                  delimeter,
                  rows_after_header_useless,
                  delete_comment_flag,
                  new_headers,
                  convert_b_flag,
                  cols_to_remove]

    file = DataFile(filename, parameters)

    df = file.open()

    test_lst.append(df)

sm_b6_data = pd.concat([test_lst[0],test_lst[1]])
fe_sb2_data1 = pd.concat([test_lst[2],test_lst[3]])
fe_sb2_data2 = pd.concat([test_lst[4],test_lst[5], test_lst[6]])


sm_b6_data = remove_irrelevant_columns(sm_b6_data)
sm_b6_data = remove_constant_column(sm_b6_data)

fe_sb2_data1 = remove_irrelevant_columns(fe_sb2_data1)
fe_sb2_data1 = remove_constant_column(fe_sb2_data1)

fe_sb2_data2 = remove_irrelevant_columns(fe_sb2_data2)
fe_sb2_data2 = remove_constant_column(fe_sb2_data2)

print('SmB6 Headers:')
print(sm_b6_data.columns)
print('\n')
print('FeSb2 - data 1 - Headers:')
print(fe_sb2_data1.columns)
print('\n')
print('FeSb2 - data 2 - Headers:')
print(fe_sb2_data2.columns)
print('\n')

sm_b6_data.to_csv(main_path+'data_csvs_cleaned/SmB6_data1.csv',index=False)
fe_sb2_data1.to_csv(main_path+'data_csvs_cleaned/FeSb2_data1.csv',index=False)
fe_sb2_data2.to_csv(main_path+'data_csvs_cleaned/FeSb2_data2.csv',index=False)
# I am just going to assume the ETO Code is not relevant even though it changes therefore I'll put it in as a parameter
# in remove constant column
