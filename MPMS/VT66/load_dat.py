import pandas as pd
from tools.utils import *
from tools.DataFile import DataFile
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tools.MakePlot import MakePlot

main_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/Popiel/FeSb2/VT66/'

folders = ['sample_placement1_131020/',
           'sample_placement2_281020/',
           'sample_placement3_291020/',
           'sample_placement4_021120/',
           'sample_placement5_031120/',
           'sample_placement6_051120/']

m_v_h_names = ['VT66-0deg_from_010-1p8K-FS_00001.dat',
               'VT66-placement_2-1p8K-FS.dat',
               'VT66-placement_3-1p8K-FS.dat',
               'VT66-placement_4-1p8K-FS.dat',
               'VT66-placement_5-1p8K-FS.dat',
               'VT66-placement_6-1p8K-FS.dat']

m_v_t_names = ['VT66-0deg_from_010-7T-tempsweep.dat',
               'VT66-placement_2-7T-tempsweep.dat',
               'VT66-placement_3-7T-tempsweep.dat',
               'VT66-placement_4-7T-tempsweep.dat',
               'VT66-placement_5-7T-tempsweep.dat',
               'VT66-placement_6-7T-tempsweep.dat']

placement_7_folds = ['sample_placement7a_091120/',
                     'sample_placement7b_101120/',
                     'sample_placement7c_111120/',
                     'sample_placement7d_111120/',
                     'sample_placement7e_121120/']

placement_7_field = ['VT66-placement_7-1p8K-FS.dat',
                     'VT66-placement_7b-1p8K-FS.dat',
                     'VT66-placement_7c-1p8K-FS.dat',
                     'VT66-placement_7d-1p8K-FS.dat',
                     'VT66-placement_7e-1p8K-FS.dat']

placement_7_temps = ['VT66-placement_7-7T-tempsweep.dat',
                     'VT66-placement_7b-7T-tempsweep.dat',
                     'VT66-placement_7c-7T-tempsweep.dat',
                     'VT66-placement_7d-7T-tempsweep.dat',
                     'VT66-placement_7e-7T-tempsweep.dat']

relevant_cols = ['Temperature (K)', 'Magnetic Field (Oe)', 'DC Moment Fixed Ctr (emu)', 'DC Moment Free Ctr (emu)']

# here are the original column names for reference
# the new column names I created


lines_to_skip = 18
delimeter = ','
row_after_header_useless = False
delete_comment_flag = True
new_headers = None
convert_b_flag = False
cols_to_keep = None
field_lst, temp_lst = [], []

# Temp sweep has for num2 39 same for field sweep
# 26 on royce

# First loop over all of the non-placement 7 stuff
#get two lists, one for f(T), one f(H)

for ind, folder in enumerate(folders):

    angle_num = ind+1

    filename_field = main_path + folder + m_v_h_names[ind]
    filename_temps = main_path + folder + m_v_t_names[ind]

    parameters = [delimeter,
                  row_after_header_useless,
                  delete_comment_flag,
                  new_headers,
                  convert_b_flag,
                  cols_to_keep]

    file_field = DataFile(filename_field, parameters)
    field_df = file_field.open()
    field_df = field_df[['Temperature (K)', 'Magnetic Field (Oe)','DC Moment Fixed Ctr (emu)', 'DC Moment Free Ctr (emu)']]
    field_df['Angle'] = 'Angle '+ str(angle_num)
    field_lst.append(field_df)

    file_temps = DataFile(filename_temps, parameters)
    temps_df = file_temps.open()
    temps_df = temps_df[['Temperature (K)', 'Magnetic Field (Oe)', 'DC Moment Fixed Ctr (emu)', 'DC Moment Free Ctr (emu)']]
    temps_df['Angle'] = 'Angle '+ str(angle_num)
    temp_lst.append(temps_df)


big_field_df = pd.concat(field_lst)
big_temp_df = pd.concat(temp_lst)

big_field_df['Magnetic Field (T)'] = big_field_df['Magnetic Field (Oe)']/10000

big_temp_df['Magnetic Field (T)'] = big_temp_df['Magnetic Field (Oe)']/10000

big_field_df['DC Moment Fixed Ctr (emu)'] = moving_average(big_field_df['DC Moment Fixed Ctr (emu)'],15)

big_temp_df['DC Moment Fixed Ctr (emu)'] = moving_average(big_temp_df['DC Moment Fixed Ctr (emu)'],15)

sns.set_palette('Paired')

fig, axs = MakePlot(ncols=1, nrows=2, figsize=(16, 9)).create()
ax1 = axs[0]
ax2 = axs[1]


sns.lineplot(y='DC Moment Fixed Ctr (emu)', x='Magnetic Field (T)', data=big_field_df, hue='Angle', ax=ax1, legend=False)
# ax1.set_title('Field Sweep')
# ax1.set_ylabel(r'Magnetic Moment $(emu)$', usetex=True, rotation=90, fontsize=16)
# ax1.set_xlabel(r'Magnetic Field $(T)$', usetex=True, fontsize=16)

sns.lineplot(y='DC Moment Fixed Ctr (emu)', x='Temperature (K)',  hue='Angle', data=big_temp_df, ax=ax2)
# ax2.set_title('Temperature Sweep')
# ax2.set_ylabel(r'Magnetic Moment $(emu)$', usetex=True, rotation=90, fontsize=16)
# ax2.set_xlabel(r'Temperature $(K)$', usetex=True, fontsize=16)

legend = ax2.legend()
ax2.get_legend().remove()

fig.suptitle('Angular Dependence of Magnetization', fontsize=22)
plt.figlegend(frameon=False,
                loc='center right',
                title='Angle')  # ,labels=np.unique(df.current_sweep_ch2.values), frameon=True)
#
#plt.show()
plt.savefig('/Users/npopiel/Desktop/fig_smoothed_15.png')

