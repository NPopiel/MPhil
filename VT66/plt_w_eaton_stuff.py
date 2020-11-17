import pandas as pd
from tools.utils import *
from tools.DataFile import DataFile
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tools.MakePlot import MakePlot
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.signal import savgol_filter

main_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/Popiel/FeSb2/VT66/'

folders = ['sample_placement1_131020/',
           'sample_placement2_281020/',
           'sample_placement3_291020/',
           'sample_placement4_021120/',
           'sample_placement5_031120/',
           'sample_placement6_051120/',
           'a/',
           'ab/',
           'b/',
           'c/',
           'c2/']

m_v_h_names = ['VT66-0deg_from_010-1p8K-FS_00001.dat',
               'VT66-placement_2-1p8K-FS.dat',
               'VT66-placement_3-1p8K-FS.dat',
               'VT66-placement_4-1p8K-FS.dat',
               'VT66-placement_5-1p8K-FS.dat',
               'VT66-placement_6-1p8K-FS.dat',
               'VT66-a-1.8-FS.dat',
               'VT66-ab-1.8-FS.dat',
               'VT66-b-1.8-FS.dat',
               'VT66-c-1.8-FS.dat',
               'VT66-c2-1.8-FS.dat']



relevant_cols = ['Temperature (K)', 'Magnetic Field (Oe)', 'DC Moment Fixed Ctr (emu)', 'DC Moment Free Ctr (emu)']

# here are the original column names for reference
# the new column names I created


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

labels = ['Angle 1',
          'Angle 2',
          'Angle 3',
          'Angle 4',
          'Angle 5',
          'Angle 6',
          'a',
          'ab',
          'b',
          'c',
          'c2']

for ind, folder in enumerate(folders):

    filename_field = main_path + folder + m_v_h_names[ind]

    parameters = [delimeter,
                  row_after_header_useless,
                  delete_comment_flag,
                  new_headers,
                  convert_b_flag,
                  cols_to_keep]

    file_field = DataFile(filename_field, parameters)
    field_df = file_field.open()
    field_df = field_df[['Temperature (K)', 'Magnetic Field (Oe)','DC Moment Fixed Ctr (emu)', 'DC Moment Free Ctr (emu)']]
    field_df['Placement'] = labels[ind]
    field_lst.append(field_df)


big_field_df = pd.concat(field_lst)


big_field_df['Magnetic Field (T)'] = big_field_df['Magnetic Field (Oe)']/10000


sns.set_palette('Paired')

fig, axs = MakePlot(ncols=1, nrows=1, figsize=(16, 9)).create()
ax1 = axs

sns.lineplot(y='DC Moment Fixed Ctr (emu)', x='Magnetic Field (T)', data=big_field_df, hue='Placement', ax=ax1)
# ax1.set_title('Field Sweep')
# ax1.set_ylabel(r'Magnetic Moment $(emu)$', usetex=True, rotation=90, fontsize=16)
# ax1.set_xlabel(r'Magnetic Field $(T)$', usetex=True, fontsize=16)

fig.suptitle('Angular Dependence of Magnetization', fontsize=22)
# plt.figlegend(frameon=False,
#                 loc='center right',
#                 title='Angle')  # ,labels=np.unique(df.current_sweep_ch2.values), frameon=True)
#
#plt.show()
plt.savefig('/Users/npopiel/Desktop/fig_w_eaton_smooth.png', dpi=600)

