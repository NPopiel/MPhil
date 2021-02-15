import pandas as pd
from tools.utils import *
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

tempsweep_fc1_df = load_matrix('/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_tempsweep_FC.dat')
temps_sweep_fc1 = np.array(tempsweep_fc1_df['Temperature (K)'])
magnetisation_tempsweep_fc1 = np.array(tempsweep_fc1_df['DC Moment Fixed Ctr (emu)'])

tempsweep_fc2_df = load_matrix('/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_tempsweep_FC_again.dat')
temps_sweep_fc2 = np.array(tempsweep_fc2_df['Temperature (K)'])
magnetisation_tempsweep_fc2 = np.array(tempsweep_fc2_df['DC Moment Fixed Ctr (emu)'])

tempsweep_zfc1_df = load_matrix('/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_tempsweep_ZFC.dat')
temps_sweep_zfc1 = np.array(tempsweep_zfc1_df['Temperature (K)'])
magnetisation_tempsweep_zfc1 = np.array(tempsweep_zfc1_df['DC Moment Fixed Ctr (emu)'])

tempsweep_zfc2_df = load_matrix('/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_tempsweep_ZFC_again.dat')
temps_sweep_zfc2 = np.array(tempsweep_zfc2_df['Temperature (K)'])
magnetisation_tempsweep_zfc2 = np.array(tempsweep_zfc2_df['DC Moment Fixed Ctr (emu)'])


for ind, file in enumerate(temp_sweeps):

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




lines_fc,  fields_fc, mags_fc, subtracted_lines_fc, slopes_fc = [], [], [], [], []
lines_zfc,  fields_zfc, mags_zfc, subtracted_lines_zfc, slopes_zfc = [], [], [], [], []

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

    filename_z_field = main_path + zero_field_cools[ind]

    zero_field_df = load_matrix(filename_z_field)
    zero_field_df = zero_field_df[['Temperature (K)', 'Magnetic Field (Oe)','DC Moment Fixed Ctr (emu)', 'DC Moment Free Ctr (emu)']]
    #arr = scipy.ndimage.filters.median_filter(np.array(field_df['DC Moment Fixed Ctr (emu)']), size=3)
    #interpd_arr = np.interp(np.array(field_df['Magnetic Field (Oe)']), np.array(field_df['Magnetic Field (Oe)']), arr)
    #smoothed_arr = moving_average(arr,5)
    #field_df['DC Moment Fixed Ctr (emu)'] = arr
    #field_df['DC Moment Fixed Ctr (emu)'] = field_df['DC Moment Fixed Ctr (emu)'].rolling(3).mean()
    zero_field_df['Temperature'] = labels[ind]
    zf_lst.append(zero_field_df)

    magnetisation_fc = np.array(field_df['DC Moment Fixed Ctr (emu)'])#scipy.ndimage.filters.median_filter(np.array(field_df['DC Moment Fixed Ctr (emu)']), size=3)/1000
    field_fc = np.array(field_df['Magnetic Field (Oe)'])/10000

    fields_fc.append(field_fc)
    mags_fc.append(magnetisation_fc)

    slope_fc, const = fit_line(magnetisation_fc,field_fc)
    lines_fc.append(slope_fc*x_linspace)
    slopes_fc.append(slope_fc)

    mag_wo_line_fc = magnetisation_fc - slope_fc*field_fc

    subtracted_lines_fc.append(mag_wo_line_fc)

    magnetisation_zfc = np.array(zero_field_df['DC Moment Fixed Ctr (emu)'])#scipy.ndimage.filters.median_filter(np.array(field_df['DC Moment Fixed Ctr (emu)']), size=3)/1000
    field_zfc = np.array(zero_field_df['Magnetic Field (Oe)'])/10000

    fields_zfc.append(field_zfc)
    mags_zfc.append(magnetisation_zfc)

    slope_zfc, const = fit_line(magnetisation_zfc,field_zfc)
    lines_zfc.append(slope_zfc*x_linspace)
    slopes_zfc.append(slope_zfc)

    mag_wo_line_zfc = magnetisation_fc - slope_zfc*field_zfc

    subtracted_lines_zfc.append(mag_wo_line_zfc)


big_field_df = pd.concat(field_lst)
big_zf_df = pd.concat((zf_lst))


big_field_df['Magnetic Field (T)'] = big_field_df['Magnetic Field (Oe)']/10000

big_zf_df['Magnetic Field (T)'] = big_zf_df['Magnetic Field (Oe)']/10000


sns.set_palette('husl')


fig, ax = MakePlot(nrows=2, ncols=3).create()
# Plot original
ax1 = ax[0,0]
ax2 = ax[0,1]
ax3 = ax[0,2]
ax4 = ax[1,0]
ax5 = ax[1,1]
ax6 = ax[1,2]

ax1.plot(fields_fc[0],subtracted_lines_fc[0],  linewidth=2.5, label='Field Cool')
ax1.plot( fields_zfc[0],subtracted_lines_zfc[0], linewidth=2.5, label='Zero Field Cool')
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax1.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='Times')
ax1.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='Times')
ax1.set_title(r'$1.8 K$', fontsize=14,fontname='Times')
ax1.set_xlim()
ax1.set_ylim()
ax1.minorticks_on()
ax1.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)

ax2.plot(fields_fc[1],subtracted_lines_fc[1],  linewidth=2.5, label='Field Cool')
ax2.plot(fields_zfc[1],subtracted_lines_zfc[1],  linewidth=2.5, label='Zero Field Cool')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax2.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='Times')
ax2.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='Times')
ax2.set_title(r'$80 K$', fontsize=14,fontname='Times')
ax2.set_xlim()
ax2.set_ylim()
ax2.minorticks_on()
ax2.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)

ax3.plot( fields_fc[2], subtracted_lines_fc[2],linewidth=2.5, label='Field Cool')
ax3.plot(fields_zfc[2], subtracted_lines_zfc[2], linewidth=2.5, label='Zero Field Cool')
ax3.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax3.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='Times')
ax3.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='Times')
ax3.set_title(r'$110 K$', fontsize=14,fontname='Times')
ax3.set_xlim()
ax3.set_ylim()
ax3.minorticks_on()
ax3.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)

ax4.plot( fields_fc[3],subtracted_lines_fc[3], linewidth=2.5, label='Field Cool')
ax4.plot(fields_zfc[3], subtracted_lines_zfc[3], linewidth=2.5, label='Zero Field Cool')
ax4.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax4.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='Times')
ax4.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='Times')
ax4.set_title(r'$190 K$', fontsize=14,fontname='Times')
ax4.set_xlim()
ax4.set_ylim()
ax4.minorticks_on()
ax4.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)

ax5.plot( fields_fc[4], subtracted_lines_fc[4],linewidth=2.5, label='Field Cool')
ax5.plot(fields_zfc[4],subtracted_lines_zfc[4],  linewidth=2.5, label='Zero Field Cool')
ax5.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax5.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='Times')
ax5.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='Times')
ax5.set_title(r'$260 K$', fontsize=14,fontname='Times')
ax5.set_xlim()
ax5.set_ylim()
ax5.minorticks_on()
ax5.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)

# ax6.plot( temps_sweep_fc1, savgol_filter(savgol_filter(magnetisation_tempsweep_fc1,5,3),5,3),linewidth=2.5, label='Field Cool')
# ax6.plot( temps_sweep_zfc1,savgol_filter(savgol_filter(magnetisation_tempsweep_zfc1,5,3),5,3), linewidth=2.5, label='Zero Field Cool')
ax6.plot( temps_sweep_fc1, scipy.ndimage.filters.median_filter(magnetisation_tempsweep_fc1,size=5),linewidth=2.5, label='Field Cool')
ax6.plot( temps_sweep_zfc1,scipy.ndimage.filters.median_filter(magnetisation_tempsweep_zfc1,size=5), linewidth=2.5, label='Zero Field Cool')

ax6.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax6.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='Times')
ax6.set_xlabel(r'Temperature $(K)$', fontsize=12,fontname='Times')
ax6.set_title(r'Temperature Sweep', fontsize=14,fontname='Times')
ax6.set_xlim()
ax6.set_ylim()
ax6.minorticks_on()
ax6.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)




plt.suptitle('PNR Magnetisation', fontsize=18,fontname='Times')
plt.legend(framealpha=0,
    bbox_to_anchor=(1, 1), loc=2,
    title='Cooling')
plt.tight_layout(pad=3.0)
plt.show()


