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

def subtract_background(df):

    mag = scipy.ndimage.median_filter(np.array(df['DC Moment Fixed Ctr (emu)']),size=5)#scipy.ndimage.filters.median_filter(np.array(field_df['DC Moment Fixed Ctr (emu)']), size=3)/1000
    field = np.array(df['Magnetic Field (Oe)'])/10000

    slope, const = fit_line(mag,field)

    return mag, field, slope

def filter_tempsweeps(df,size=5):
    mag = np.array(df['DC Moment Fixed Ctr (emu)'])
    temp = np.array(df['Temperature (K)'])
    mag = scipy.ndimage.median_filter(scipy.ndimage.median_filter(mag,size=size),size=5)

    return temp, mag


main_path = '/Volumes/GoogleDrive/My Drive/PNR/Royce/'


# Real data first
# Field Sweeps
base_fc_fs = load_matrix(main_path+'PNR_1p8K_MvsH_FC.dat')
base_zfc_fs = load_matrix(main_path+'PNR_1p8K_MvsH_ZFC.dat')
room_fs = load_matrix(main_path+'PNR_300K_MvsH.dat')

# Need to get the diamagnetic contribution and subtract it from the field sweeps

mag_base_fc, field_base_fc, slope_base_fc = subtract_background(base_fc_fs)
mag_base_zfc, field_base_zfc, slope_base_zfc = subtract_background(base_zfc_fs)
mag_room, field_room, slope_room = subtract_background(room_fs)

# Temp sweeps
fc_ts1 = load_matrix(main_path+'PNR_tempsweep_FC.dat')
fc_ts2 = load_matrix(main_path+'PNR_tempsweep_FC_again.dat')
zfc_ts1 = load_matrix(main_path+'PNR_tempsweep_ZFC.dat')
zfc_ts2 = load_matrix(main_path+'PNR_tempsweep_ZFC_again.dat')

# All that needs to be done to tempsweeps is filtering!
temp_fcts1, mag_fcts1 = filter_tempsweeps(fc_ts1)
temp_fcts2, mag_fcts2 = filter_tempsweeps(fc_ts2)
temp_zfcts1, mag_zfcts1 = filter_tempsweeps(zfc_ts1)
temp_zfcts2, mag_zfcts2 = filter_tempsweeps(zfc_ts2)


# Now i need to subtract diamagnetic background off of the field sweeps

relevant_cols = ['Temperature (K)', 'Magnetic Field (Oe)', 'DC Moment Fixed Ctr (emu)', 'DC Moment Free Ctr (emu)']



x_linspace = np.linspace(-7,7,10000)



lines_fc,  fields_fc, mags_fc, subtracted_lines_fc, slopes_fc = [], [], [], [], []
lines_zfc,  fields_zfc, mags_zfc, subtracted_lines_zfc, slopes_zfc = [], [], [], [], []


fig, ax = MakePlot(nrows=1, ncols=3).create()
# Plot original
ax1 = ax[0]
ax2 = ax[1]
ax3 = ax[2]


ax1.plot(field_base_fc,scipy.ndimage.median_filter(mag_base_fc-slope_base_fc*field_base_fc, size=5),  c='purple',linewidth=2.5, label='0.05 T Cool')
ax1.plot(field_base_zfc, scipy.ndimage.median_filter(mag_base_zfc-slope_base_zfc*field_base_zfc,size=5), c='orange', label='0 T Cool')
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0),useMathText=True)
ax1.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='Times')
ax1.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='Times')
ax1.set_title(r'$1.8 K$', fontsize=14,fontname='Times')
ax1.set_xlim()
ax1.set_ylim()
ax1.minorticks_on()
ax1.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)
ax1.legend(framealpha=0,
    title='Cooling')

ax2.plot(field_room,scipy.ndimage.median_filter(mag_room-slope_room*field_room,size=5), color='k', linewidth=2.5)
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax2.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='Times')
ax2.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='Times')
ax2.set_title(r'$300 K$', fontsize=14,fontname='Times')
ax2.set_xlim()
ax2.set_ylim()
ax2.minorticks_on()
ax2.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)

ax3.plot(temp_fcts1, mag_fcts1,linewidth=2.5, label='0.05 T Cool')
ax3.plot(temp_fcts2, mag_fcts2,linewidth=2.5, label='0.05 T Cool')
ax3.plot(temp_zfcts1, mag_zfcts1,linewidth=2.5, label='0 T Cool')
ax3.plot(temp_zfcts2, mag_zfcts2,linewidth=2.5, label='0 T Cool')
ax3.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax3.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='Times')
ax3.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='Times')
ax3.set_title(r'Temperature Sweeps', fontsize=14,fontname='Times')
ax3.set_xlim()
ax3.set_ylim()
ax3.minorticks_on()
ax3.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)

ax3.legend(framealpha=0,
    bbox_to_anchor=(1, 1), loc=2,
    title='Cooling')



plt.suptitle('PNR Magnetisation Royce', fontsize=18,fontname='Times')

plt.tight_layout(pad=3.0)
plt.show()


