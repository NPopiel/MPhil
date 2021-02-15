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

main_path = '/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/'
# '/Users/npopiel/Documents/MPhil/Data/PNR/311020_PNR_1p8K_MvsH_FC.dat',
zero_field_cools = ['PNR_80K_MvsH_ZFC_recentred.dat',
                    'PNR_110K_MvsH_ZFC.dat',
                    'PNR_190K_MvsH_ZFC.dat',
                    'PNR_260K_MvsH_ZFC.dat']

field_cools = ['PNR_1p8K_MvsH_FC.dat',
'1p8K_FS_control.dat']
               #'PNR_300K_MvsH.dat']

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
field_lst, temp_lst = [], []

# Temp sweep has for num2 39 same for field sweep
# 26 on royce

# First loop over all of the non-placement 7 stuff
#get two lists, one for f(T), one f(H)

labels = [r'$1.8 K$',
          r'$1.8 K$ Control']

for ind, file in enumerate(field_cools):

    filename_field = main_path + file

    field_df = load_matrix(filename_field)
    field_df = field_df[['Temperature (K)', 'Magnetic Field (Oe)','DC Moment Fixed Ctr (emu)', 'DC Moment Free Ctr (emu)']]
    #arr = scipy.ndimage.filters.median_filter(np.array(field_df['DC Moment Fixed Ctr (emu)']), size=3)
    #interpd_arr = np.interp(np.array(field_df['Magnetic Field (Oe)']), np.array(field_df['Magnetic Field (Oe)']), arr)
    #smoothed_arr = moving_average(arr,5)
    #field_df['DC Moment Fixed Ctr (emu)'] = arr
    #field_df['DC Moment Fixed Ctr (emu)'] = field_df['DC Moment Fixed Ctr (emu)'].rolling(3).mean()
    field_df['Temperature'] = labels[ind]
    field_lst.append(field_df)


big_field_df = pd.concat(field_lst)




big_field_df['Magnetic Field (T)'] = big_field_df['Magnetic Field (Oe)']/10000


sns.set_palette('husl')

#fig, axs = MakePlot(ncols=1, nrows=1, figsize=(16, 9)).create()
#ax1 = axs

#sns.scatterplot(y='DC Moment Fixed Ctr (emu)', x='Magnetic Field (T)', data=big_field_df, hue='Temperature',style='Temperature',  alpha=0.7,  ax=ax1)
# ax1.set_title('Field Sweep')
# ax1.set_ylabel(r'Magnetic Moment $(emu)$', usetex=True, rotation=90, fontsize=16)
# ax1.set_xlabel(r'Magnetic Field $(T)$', usetex=True, fontsize=16)

#fig.suptitle('Temperature Dependence of PNR Magnetization', fontsize=22)
# plt.figlegend(frameon=False,
#                 loc='center right',
#                 title='Angle')  # ,labels=np.unique(df.current_sweep_ch2.values), frameon=True)
#
#plt.show()
#plt.savefig('/Users/npopiel/Desktop/fig_w_eaton_smooth.png', dpi=600)

# The function for plotting fit should be something like

def fit_line(mag,field,abs_val_h=4):

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

#def sinusoid(angle,)

x_linspace = np.linspace(-7,7,10000)

lines, langevins, fields, mags, subtracted_lines, slopes = [], [], [], [], [], []
for ind, file in enumerate(field_cools):

    filename_field = main_path + file


    field_df = load_matrix(filename_field)
    field_df = field_df[['Temperature (K)', 'Magnetic Field (Oe)','DC Moment Fixed Ctr (emu)', 'DC Moment Free Ctr (emu)']]
    magnetisation = np.array(field_df['DC Moment Fixed Ctr (emu)'])#scipy.ndimage.filters.median_filter(np.array(field_df['DC Moment Fixed Ctr (emu)']), size=3)/1000
    field = np.array(field_df['Magnetic Field (Oe)'])/10000

    fields.append(field)
    mags.append(magnetisation)

    slope, const = fit_line(magnetisation,field)
    lines.append(slope*x_linspace)
    slopes.append(slope)

    mag_wo_line = magnetisation - slope*field

    subtracted_lines.append(mag_wo_line)

    popt, pcov = scipy.optimize.curve_fit(langevin, field, mag_wo_line)

    mu_eff = popt[0]
    c_imp = popt[1]

    langevins.append(langevin(x_linspace,mu_eff,c_imp))



fig, (ax1, ax2, ax3) = MakePlot(nrows=1, ncols=3).create()
# Plot original
for ind, arr in enumerate(mags):
    ax1.plot(fields[ind], arr, linewidth=2.5)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax1.set_ylabel('Magnetisation (emu)', fontsize=12)
ax1.set_xlabel('Magnetic Field (T)', fontsize=12)
ax1.set_title('Raw Data', fontsize=14)
ax1.set_xlim()
ax1.set_ylim()
# ax2.legend(framealpha=0,
#     bbox_to_anchor=(1, 1), loc=2,
#     title='Sweeps')
#ax1.grid()
ax1.minorticks_on()
ax1.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)

# Linear Response
for ind, line in enumerate(lines):
    ax2.plot(x_linspace, line, linewidth=2.5)
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax2.set_ylabel('Magnetisation (emu)', fontsize=12)
ax2.set_xlabel('Magnetic Field (T)', fontsize=12)
ax2.set_title('Linear Response', fontsize=14)
ax2.set_xlim()
ax2.set_ylim()
# ax2.legend(framealpha=0,
#     bbox_to_anchor=(1, 1), loc=2,
#     title='Sweeps')
#ax2.grid()

ax2.minorticks_on()
ax2.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)

# Langevin Impurities
for ind, subtracted_line in enumerate(subtracted_lines):
    ax3.plot(fields[ind], subtracted_line, label=labels[ind], linewidth=2.5)
ax3.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax3.set_ylabel('Magnetisation (emu)', fontsize=12)
ax3.set_xlabel('Magnetic Field (T)', fontsize=12)
ax3.set_title('Subtracted Lines', fontsize=14)
ax3.set_xlim()
ax3.set_ylim()
# ax2.legend(framealpha=0,
#     bbox_to_anchor=(1, 1), loc=2,
#     title='Sweeps')
#ax3.grid()

ax3.minorticks_on()
ax3.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)
'''
# Langevin Impurities
for langevin in langevins:
    ax4.scatter(x=x_linspace, y=langevin)
ax4.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax4.set_ylabel('Magnetisation (J/T)', fontsize=14)
ax4.set_xlabel('Magnetic Field (T)', fontsize=14)
ax4.set_title('Langevin Response')
'''

plt.suptitle('Decomposition of Magnetisation (Field Cool)', fontsize=18)
plt.legend(framealpha=0,
    bbox_to_anchor=(1, 1), loc=2,
    title='Temperature')
plt.tight_layout(pad=3.0)
plt.show()
