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

def increasing_v_decreasing(field, abs_val_h):

    idxs = np.arange(field.shape[0])
    increasing_pos_field, decreasing_pos_field = [], []
    increasing_neg_field, decreasing_neg_field = [], []

    # loop over top bit first

    for b1, b2 in zip(field[:-1],field[1:]):

        if np.abs(b1)<np.abs(b2):
            if b1 < 0:
                increasing_pos_field.append(False)
                decreasing_pos_field.append(False)
                increasing_neg_field.append(True)
                decreasing_neg_field.append(False)
            else:
                increasing_pos_field.append(True)
                decreasing_pos_field.append(False)
                increasing_neg_field.append(False)
                decreasing_neg_field.append(False)
        else:
            if b1 < 0:
                increasing_pos_field.append(False)
                decreasing_pos_field.append(False)
                increasing_neg_field.append(False)
                decreasing_neg_field.append(True)
            else:
                increasing_pos_field.append(False)
                decreasing_pos_field.append(True)
                increasing_neg_field.append(False)
                decreasing_neg_field.append(False)


    increasing_pos_field.append(False)
    decreasing_pos_field.append(False)
    increasing_neg_field.append(False)
    decreasing_neg_field.append(True)

    idx_pos_cutoff =  (np.abs(field - abs_val_h)).argmin()
    idx_neg_cutoff = (np.abs(field + abs_val_h)).argmin()

    increasing_pos_field = np.array(increasing_pos_field)
    decreasing_pos_field = np.array(decreasing_pos_field)
    increasing_neg_field = np.array(increasing_neg_field)
    decreasing_neg_field = np.array(decreasing_neg_field)


    increasing_pos_field[np.where(np.abs(field)<abs_val_h)] = False
    decreasing_pos_field[np.where(np.abs(field)<abs_val_h)] = False
    increasing_neg_field[np.where(np.abs(field)<abs_val_h)] = False
    decreasing_neg_field[np.where(np.abs(field)<abs_val_h)] = False


    return idxs[increasing_pos_field], idxs[decreasing_pos_field], idxs[increasing_neg_field], idxs[decreasing_neg_field]

def fit_line2(mag,field,abs_val_h=4):

    linear_h_top = np.where(field>abs_val_h)
    linear_h_bot = np.where(field<-1*abs_val_h)




    increasing_pos_field, decreasing_pos_field, increasing_neg_field, decreasing_neg_field = increasing_v_decreasing(field, abs_val_h)


    upper_fit_inc = np.polyfit(field[increasing_pos_field],mag[increasing_pos_field],deg=1)
    upper_fit_dec = np.polyfit(field[decreasing_pos_field], mag[decreasing_pos_field], deg=1)
    lower_fit_inc = np.polyfit(field[increasing_neg_field],mag[increasing_neg_field],deg=1)
    lower_fit_dec = np.polyfit(field[decreasing_neg_field], mag[decreasing_neg_field], deg=1)

    upper_slope = np.average([upper_fit_inc[0], upper_fit_dec[0]])
    lower_slope = np.average([lower_fit_inc[0], upper_fit_dec[0]])

    upper_const = np.average([upper_fit_inc[1], upper_fit_dec[1]])
    lower_const = np.average([lower_fit_inc[1], lower_fit_dec[1]])

    return (upper_slope+lower_slope)/2, (upper_const+lower_const)/2

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


labels = [r'$1.8 K$',r'$80 K$',r'$110 K$',r'$190 K$',r'$260 K$',
          ]

x_linspace = np.linspace(-7,7,10000)



# Let's consider only the 1p8 K sweeps to begin. we want to plot original data, and the four possible fitted lines
# as well as the subtracted data with each line. We can also have a seperate panel subtracting th eGE control


field_df = load_matrix('/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_1p8K_MvsH_FC.dat')
field_df = field_df[['Temperature (K)', 'Magnetic Field (Oe)', 'DC Moment Fixed Ctr (emu)', 'DC Moment Free Ctr (emu)']]
field_df['Temperature'] = labels[0]


zero_field_df = load_matrix('/Users/npopiel/Documents/MPhil/Data/PNR/Attempt 3/PNR_1p8K_MvsH_ZFC.dat')
zero_field_df = zero_field_df[
    ['Temperature (K)', 'Magnetic Field (Oe)', 'DC Moment Fixed Ctr (emu)', 'DC Moment Free Ctr (emu)']]
zero_field_df['Temperature'] = labels[0]


magnetisation_fc = np.array(field_df[
                                'DC Moment Fixed Ctr (emu)'])  # scipy.ndimage.filters.median_filter(np.array(field_df['DC Moment Fixed Ctr (emu)']), size=3)/1000
field_fc = np.array(field_df['Magnetic Field (Oe)']) / 10000


idxs_pos_up, idxs_pos_dn, idxs_neg_up, idxs_neg_dn = increasing_v_decreasing(field_fc,4)

upper_fit_inc_fc = np.polyfit(field_fc[idxs_pos_up], magnetisation_fc[idxs_pos_up], deg=1)
upper_fit_dec_fc = np.polyfit(field_fc[idxs_pos_dn], magnetisation_fc[idxs_pos_dn], deg=1)
lower_fit_inc_fc = np.polyfit(field_fc[idxs_neg_up], magnetisation_fc[idxs_neg_up], deg=1)
lower_fit_dec_fc = np.polyfit(field_fc[idxs_neg_dn], magnetisation_fc[idxs_neg_dn], deg=1)

upper_slope_fc = np.average([upper_fit_inc_fc[0], upper_fit_dec_fc[0]])
lower_slope_fc = np.average([lower_fit_inc_fc[0], upper_fit_dec_fc[0]])

upper_const_fc = np.average([upper_fit_inc_fc[1], upper_fit_dec_fc[1]])
lower_const_fc = np.average([lower_fit_inc_fc[1], lower_fit_dec_fc[1]])

average_slope_fc = np.average([upper_fit_inc_fc[0], upper_fit_dec_fc[0],lower_fit_inc_fc[0], upper_fit_dec_fc[0]])
average_const_fc = np.average([upper_fit_inc_fc[1], upper_fit_dec_fc[1],lower_fit_inc_fc[1], upper_fit_dec_fc[1]])

# FOr the FC, get each data without the line!

wo_line_1fc = magnetisation_fc - upper_fit_inc_fc[0]*(field_fc)
wo_line_2fc = magnetisation_fc - upper_fit_dec_fc[0]*(field_fc)
wo_line_3fc = magnetisation_fc - lower_fit_inc_fc[0]*(field_fc)
wo_line_4fc = magnetisation_fc - lower_fit_dec_fc[0]*(field_fc)



magnetisation_zfc = np.array(zero_field_df[
                                 'DC Moment Fixed Ctr (emu)'])  # scipy.ndimage.filters.median_filter(np.array(field_df['DC Moment Fixed Ctr (emu)']), size=3)/1000
field_zfc = np.array(zero_field_df['Magnetic Field (Oe)']) / 10000

idxs_pos_up, idxs_pos_dn, idxs_neg_up, idxs_neg_dn = increasing_v_decreasing(field_zfc,4)

upper_fit_inc_zfc = np.polyfit(field_zfc[idxs_pos_up], magnetisation_zfc[idxs_pos_up], deg=1)
upper_fit_dec_zfc = np.polyfit(field_zfc[idxs_pos_dn], magnetisation_zfc[idxs_pos_dn], deg=1)
lower_fit_inc_zfc = np.polyfit(field_zfc[idxs_neg_up], magnetisation_zfc[idxs_neg_up], deg=1)
lower_fit_dec_zfc = np.polyfit(field_zfc[idxs_neg_dn], magnetisation_zfc[idxs_neg_dn], deg=1)

upper_slope_zfc = np.average([upper_fit_inc_zfc[0], upper_fit_dec_zfc[0]])
lower_slope_zfc = np.average([lower_fit_inc_zfc[0], upper_fit_dec_zfc[0]])

upper_const_zfc = np.average([upper_fit_inc_zfc[1], upper_fit_dec_zfc[1]])
lower_const_zfc = np.average([lower_fit_inc_zfc[1], lower_fit_dec_zfc[1]])

average_slope_zfc = np.average([upper_fit_inc_zfc[0], upper_fit_dec_zfc[0],lower_fit_inc_zfc[0], upper_fit_dec_zfc[0]])
average_const_zfc = np.average([upper_fit_inc_zfc[1], upper_fit_dec_zfc[1],lower_fit_inc_zfc[1], upper_fit_dec_zfc[1]])

# FOr the FC, get each data without the line!

wo_line_1zfc = magnetisation_zfc - upper_fit_inc_zfc[0]*(field_zfc)
wo_line_2zfc = magnetisation_zfc - upper_fit_dec_zfc[0]*(field_zfc)
wo_line_3zfc = magnetisation_zfc - lower_fit_inc_zfc[0]*(field_zfc)
wo_line_4zfc = magnetisation_zfc - lower_fit_dec_zfc[0]*(field_zfc)

sns.set_palette('husl')

# Okay, in essence I want to plot raw data in first panel, second panel, all diamagnetic lines,
# next four the raw minus each fitted line,
# maybe a langevin fit also?

# then repeat for ZFC

fig, ax = MakePlot(nrows=2, ncols=3).create()
# Plot original
ax1 = ax[0,0]
ax2 = ax[0,1]
ax3 = ax[0,2]
ax4 = ax[1,0]
ax5 = ax[1,1]
ax6 = ax[1,2]

ax1.plot(field_fc,magnetisation_fc,  linewidth=2.5)
#ax1.plot(fields_zfc[0],subtracted_lines_zfc[0], linewidth=2.5, label='Zero Field Cool')
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax1.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='arial')
ax1.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='arial')
ax1.set_title(r'Raw Data', fontsize=14,fontname='arial')
ax1.set_xlim()
ax1.set_ylim()
ax1.minorticks_on()
ax1.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)

ax2.plot(x_linspace,upper_fit_inc_fc[0]*x_linspace,  linewidth=2.5, label='Positive Field Increasing')
ax2.plot(x_linspace,upper_fit_dec_fc[0]*x_linspace,  linewidth=2.5, label='Positive Field Decreasing')
ax2.plot(x_linspace,lower_fit_inc_fc[0]*x_linspace,  linewidth=2.5, label='Negative Field Increasing')
ax2.plot(x_linspace,lower_fit_dec_fc[0]*x_linspace,  linewidth=2.5, label='Negative Field Decreasing')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax2.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='arial')
ax2.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='arial')
ax2.set_title('Diamagnetic Contribution', fontsize=14,fontname='arial')
ax2.legend(framealpha=0, loc=3,
    title='Fitting Regime', prop={"size":8})
ax2.set_xlim()
ax2.set_ylim()
ax2.minorticks_on()
ax2.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)
#

# Add inset zoom to show deviation!inset axes....
axins = ax2.inset_axes([0.5, 0.5, 0.47, 0.47])
axins.plot(x_linspace,upper_fit_inc_fc[0]*x_linspace,  linewidth=2.5, label='Positive Field Increasing')
axins.plot(x_linspace,upper_fit_dec_fc[0]*x_linspace,  linewidth=2.5, label='Positive Field Decreasing')
axins.plot(x_linspace,lower_fit_inc_fc[0]*x_linspace,  linewidth=2.5, label='Negative Field Increasing')
axins.plot(x_linspace,lower_fit_dec_fc[0]*x_linspace,  linewidth=2.5, label='Negative Field Decreasing')
# sub region of the original image
x1, x2, y1, y2 = 6.5, 7, -0.0007, -0.00086
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')


ax2.indicate_inset_zoom(axins)
 # that is dummy code  rewrite!

ax3.plot(field_fc, wo_line_1fc,linewidth=2.5)
ax3.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax3.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='Times')
ax3.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='Times')
ax3.set_title('Positive Increasing Fit', fontsize=14,fontname='Times')
ax3.set_xlim()
ax3.set_ylim()
ax3.minorticks_on()
ax3.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)

ax4.plot(field_fc, wo_line_2fc,linewidth=2.5)
ax4.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax4.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='arial')
ax4.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='arial')
ax4.set_title('Positive Decreasing Fit', fontsize=14,fontname='arial')
ax4.set_xlim()
ax4.set_ylim()
ax4.minorticks_on()
ax4.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)

ax5.plot(field_fc, wo_line_3fc,linewidth=2.5)
ax5.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax5.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='arial')
ax5.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='arial')
ax5.set_title('Negative Increasing Fit', fontsize=14,fontname='arial')
ax5.set_xlim()
ax5.set_ylim()
ax5.minorticks_on()
ax5.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)



ax6.plot(field_fc, wo_line_4fc,linewidth=2.5)
ax6.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax6.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='arial')
ax6.set_xlabel(r'Temperature $(K)$', fontsize=12,fontname='arial')
ax6.set_title(r'Negative Decreasing Fit', fontsize=14,fontname='arial')
ax6.set_xlim()
ax6.set_ylim()
ax6.minorticks_on()
ax6.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)


plt.suptitle('PNR Magnetisation Different Subtractions 1.8 K FC', fontsize=18,fontname='Times')
plt.tight_layout(pad=3.0)
plt.show()


fig, ax2 = MakePlot().create()

ax2.plot(field_fc, wo_line_4fc,linewidth=2.5, label='Positive Field Increasing')
ax2.plot(field_fc, wo_line_4fc,linewidth=2.5, label='Positive Field Decreasing')
ax2.plot(field_fc, wo_line_4fc,linewidth=2.5, label='Negative Field Increasing')
ax2.plot(field_fc, wo_line_4fc,linewidth=2.5, label='Negative Field Decreasing')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax2.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='arial')
ax2.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='arial')
ax2.set_title('Diamagnetic Contribution', fontsize=14,fontname='arial')
ax2.legend(framealpha=0, loc=3,
    title='Fitting Regime', prop={"size":8})
ax2.set_xlim()
ax2.set_ylim()
ax2.minorticks_on()
ax2.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)
#

# Add inset zoom to show deviation!inset axes....
axins = ax2.inset_axes([0.5, 0.5, 0.47, 0.47])
axins.plot(x_linspace,upper_fit_inc_fc[0]*x_linspace,  linewidth=2.5, label='Positive Field Increasing')
axins.plot(x_linspace,upper_fit_dec_fc[0]*x_linspace,  linewidth=2.5, label='Positive Field Decreasing')
axins.plot(x_linspace,lower_fit_inc_fc[0]*x_linspace,  linewidth=2.5, label='Negative Field Increasing')
axins.plot(x_linspace,lower_fit_dec_fc[0]*x_linspace,  linewidth=2.5, label='Negative Field Decreasing')
# sub region of the original image
x1, x2, y1, y2 = 6.5, 7, -0.0007, -0.00086
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')


ax2.indicate_inset_zoom(axins)
 # that is dummy code  re



fig, ax = MakePlot(nrows=2, ncols=3).create()
# Plot original
ax1 = ax[0,0]
ax2 = ax[0,1]
ax3 = ax[0,2]
ax4 = ax[1,0]
ax5 = ax[1,1]
ax6 = ax[1,2]

ax1.plot(field_zfc,magnetisation_zfc,  linewidth=2.5)
#ax1.plot(fields_zfc[0],subtracted_lines_zfc[0], linewidth=2.5, label='Zero Field Cool')
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax1.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='arial')
ax1.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='arial')
ax1.set_title(r'Raw Data', fontsize=14,fontname='arial')
ax1.set_xlim()
ax1.set_ylim()
ax1.minorticks_on()
ax1.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)

ax2.plot(x_linspace,upper_fit_inc_zfc[0]*x_linspace,  linewidth=2.5, label='Positive Field Increasing')
ax2.plot(x_linspace,upper_fit_dec_zfc[0]*x_linspace,  linewidth=2.5, label='Positive Field Decreasing')
ax2.plot(x_linspace,lower_fit_inc_zfc[0]*x_linspace,  linewidth=2.5, label='Negative Field Increasing')
ax2.plot(x_linspace,lower_fit_dec_zfc[0]*x_linspace,  linewidth=2.5, label='Negative Field Decreasing')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax2.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='arial')
ax2.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='arial')
ax2.set_title('Diamagnetic Contribution', fontsize=14,fontname='arial')
ax2.legend(framealpha=0, loc=3,
    title='Fitting Regime', prop={"size":8})
ax2.set_xlim()
ax2.set_ylim()
ax2.minorticks_on()
ax2.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)

ax3.plot(field_zfc, wo_line_1zfc,linewidth=2.5)
ax3.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax3.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='Times')
ax3.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='Times')
ax3.set_title('Positive Increasing Fit', fontsize=14,fontname='Times')
ax3.set_xlim()
ax3.set_ylim()
ax3.minorticks_on()
ax3.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)

ax4.plot(field_zfc, wo_line_2zfc,linewidth=2.5)
ax4.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax4.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='arial')
ax4.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='arial')
ax4.set_title('Positive Decreasing Fit', fontsize=14,fontname='arial')
ax4.set_xlim()
ax4.set_ylim()
ax4.minorticks_on()
ax4.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)

ax5.plot(field_zfc, wo_line_3zfc,linewidth=2.5)
ax5.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax5.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='arial')
ax5.set_xlabel('Magnetic Field (T)', fontsize=12,fontname='arial')
ax5.set_title('Negative Increasing Fit', fontsize=14,fontname='arial')
ax5.set_xlim()
ax5.set_ylim()
ax5.minorticks_on()
ax5.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)



ax6.plot(field_zfc, wo_line_4zfc,linewidth=2.5)
ax6.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax6.set_ylabel('Magnetisation (emu)', fontsize=12,fontname='arial')
ax6.set_xlabel(r'Temperature $(K)$', fontsize=12,fontname='arial')
ax6.set_title(r'Negative Decreasing Fit', fontsize=14,fontname='arial')
ax6.set_xlim()
ax6.set_ylim()
ax6.minorticks_on()
ax6.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)


plt.suptitle('PNR Magnetisation Different Subtractions 1.8 K ZFC', fontsize=18,fontname='Times')
plt.tight_layout(pad=3.0)
plt.show()


