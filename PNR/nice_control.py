import pandas as pd
from tools.utils import *
from tools.DataFile import DataFile
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.interpolate
import scipy.optimize
import numpy as np
from tools.MakePlot import MakePlot
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.signal import savgol_filter
from tools.constants import *


def fit_line(mag,field,abs_val_h=5):

    linear_h_top = np.where(field>abs_val_h)
    linear_h_bot = np.where(field<-1*abs_val_h)

    upper_fit = np.polyfit(field[linear_h_top],mag[linear_h_top],deg=1)
    lower_fit = np.polyfit(field[linear_h_bot],mag[linear_h_bot],deg=1)

    upper_slope = upper_fit[0]
    lower_slope = lower_fit[0]

    upper_const = upper_fit[1]
    lower_const = lower_fit[1]

    return (upper_slope+lower_slope)/2, (upper_const+lower_const)/2

def langevin(field,mu_eff,c_imp):

    return c_imp*mu_eff*(1/np.tanh(np.array(mu_eff*field/1.8/kb)) - 1/(np.array(mu_eff*field/1.8/kb)))


main_path = '/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/'
# '/Users/npopiel/Documents/MPhil/Data/PNR/311020_PNR_1p8K_MvsH_FC.dat',
zero_field_cools = ['PNR_80K_MvsH_ZFC_recentred.dat',
                    'PNR_110K_MvsH_ZFC.dat',
                    'PNR_190K_MvsH_ZFC.dat',
                    'PNR_260K_MvsH_ZFC.dat']
control = '/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/1p8K_FS_control.dat'

paths = [
'/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_1p8K_MvsH_FC.dat',
'/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_1p8K_MvsH_ZFC.dat',
'/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_80K_MvsH_FC.dat',
'/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_80K_MvsH_ZFC.dat',
'/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_110K_MvsH_FC.dat',
'/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_110K_MvsH_ZFC.dat',
'/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_190K_MvsH_FC.dat',
'/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_190K_MvsH_ZFC.dat',
'/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_260K_MvsH_FC.dat',
'/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_260K_MvsH_ZFC.dat',
'/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_300K_MvsH.dat',
'/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_tempsweep_FC_again.dat',
'/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_tempsweep_FC.dat',
'/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_tempsweep_ZFC_again.dat',
'/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_tempsweep_ZFC.dat'
]
field_cools = ['PNR_1p8K_MvsH_FC.dat',
               'PNR_80K_MvsH_FC.dat',
               'PNR_110K_MvsH_FC.dat',
               'PNR_190K_MvsH_FC.dat',
               'PNR_260K_MvsH_FC.dat']
zero_field_cools = ['PNR_1p8K_MvsH_ZFC.dat',
               'PNR_80K_MvsH_ZFC.dat',
               'PNR_110K_MvsH_ZFC.dat',
               'PNR_190K_MvsH_ZFC.dat',
               'PNR_260K_MvsH_ZFC.dat']


temp_sweeps = ['PNR_tempsweep_FC_again.dat',
'PNR_tempsweep_FC.dat',
'PNR_tempsweep_ZFC_again.dat',
'PNR_tempsweep_ZFC.dat']

control_fs = ['1p8K_FS_control.dat',
             '300K_FS_control.dat']
control_tempsweep = ['300K_tempsweep_control.dat']





relevant_cols = ['Temperature (K)', 'Magnetic Field (Oe)', 'DC Moment Fixed Ctr (emu)', 'DC Moment Free Ctr (emu)']

# here are the original column names for reference
# the new column names I created


delimeter = ','
row_after_header_useless = False
delete_comment_flag = True
new_headers = None
convert_b_flag = False
cols_to_keep = None
field_lst, zf_lst, temp_lst = [], [], []

# Temp sweep has for num2 39 same for field sweep
# 26 on royce

# First loop over all of the non-placement 7 stuff
#get two lists, one for f(T), one f(H)

labels = [r'$1.8 K$',r'$80 K$',r'$110 K$',r'$190 K$',r'$260 K$',
          ]

x_linspace = np.linspace(-7,7,10000)

tempsweep_fc1_df = load_matrix('/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/1p8K_FS_control.dat')
field_sweep_fc1 = np.array(tempsweep_fc1_df['Magnetic Field (Oe)']) / 10000
magnetisation_1p8 = np.array(tempsweep_fc1_df['DC Moment Fixed Ctr (emu)'])

slope_fc1, const1 = fit_line(magnetisation_1p8, field_sweep_fc1)


mag_wo_line_1p8 = magnetisation_1p8 - slope_fc1 * field_sweep_fc1





tempsweep_fc2_df = load_matrix('/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/300K_FS_control.dat')
field_sweep_fc2 = np.array(tempsweep_fc2_df['Magnetic Field (Oe)'])/10000
magnetisation_300 = np.array(tempsweep_fc2_df['DC Moment Fixed Ctr (emu)'])

slope_fc2, const2 = fit_line(magnetisation_1p8, field_sweep_fc1)


mag_wo_line_300 = magnetisation_300 - slope_fc2 * field_sweep_fc2



tempsweep_zfc1_df = load_matrix('/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/300K_tempsweep_control2.dat')
temps_sweep = np.array(tempsweep_zfc1_df['Temperature (K)'])
magnetisation_tempsweep = np.array(tempsweep_zfc1_df['DC Moment Fixed Ctr (emu)'])


sns.set_palette('husl')


fig, ax = MakePlot(nrows=1, ncols=3).create()
# Plot original
ax1 = ax[0]
ax2 = ax[1]
ax3 = ax[2]

ax2.plot(field_sweep_fc1, magnetisation_1p8, linewidth=2.5, label='Raw')
ax2.plot(x_linspace,slope_fc1*x_linspace+const1,linewidth=2.5, label='Diamagnetic Response')
ax2.plot(field_sweep_fc1,mag_wo_line_1p8,  linewidth=2.5,label='Subtracted')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax2.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='Times')
ax2.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='Times')
ax2.set_title(r'$1.8 K$', fontsize=14,fontname='Times')
ax2.set_xlim()
ax2.set_ylim()
ax2.minorticks_on()
ax2.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)

ax3.plot(field_sweep_fc2, magnetisation_300, linewidth=2.5, label='Raw')
ax3.plot(x_linspace,slope_fc2*x_linspace+const2,linewidth=2.5, label='Diamagnetic Response')
ax3.plot(field_sweep_fc2,mag_wo_line_300,  linewidth=2.5,label='Subtracted')
ax3.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax3.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='Times')
ax3.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='Times')
ax3.set_title(r'$300 K$', fontsize=14,fontname='Times')
ax3.set_xlim()
ax3.set_ylim()
ax3.minorticks_on()
ax3.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)

ax1.plot(temps_sweep, scipy.ndimage.filters.median_filter(magnetisation_tempsweep,size=5), linewidth=2.5,c='r')
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax1.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='Times')
ax1.set_xlabel(r'Temperature $(K)$', fontsize=12,fontname='Times')
ax1.set_title(r'Temperature Sweep', fontsize=14,fontname='Times')
ax1.set_xlim()
ax1.set_ylim()
ax1.minorticks_on()
ax1.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)


plt.legend(framealpha=0,
    bbox_to_anchor=(1, 1), loc=2,
    title='Temperature')

plt.suptitle('Control Magnetisation', fontsize=18,fontname='Times')
plt.tight_layout(pad=3.0)
plt.show()


