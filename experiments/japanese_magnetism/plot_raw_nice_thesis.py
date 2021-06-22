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

plt.rcParams['font.size'] = 12
# Say, "the default sans-serif font is COMIC SANS"
plt.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
plt.rcParams['font.family'] = "sans-serif"

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
    #linear_h_bot = np.where(field<-1*abs_val_h)

    upper_fit = np.polyfit(field[linear_h_top],mag[linear_h_top],deg=1)
    #lower_fit = np.polyfit(field[linear_h_bot],mag[linear_h_bot],deg=1)

    upper_slope = upper_fit[0]
    #lower_slope = lower_fit[0]

    upper_const = upper_fit[1]
    #lower_const = lower_fit[1]

    return (upper_slope), (upper_const)


def fit_line(mag,field,abs_val_h=4):

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

def langevin(field,mu_eff,c_imp):

    return c_imp*mu_eff*(1/np.tanh(np.array(mu_eff*field/1.8/kb)) - 1/(np.array(mu_eff*field/1.8/kb)))

def mols(mass, molar_mass):
    return mass/molar_mass



main_path = '/Volumes/GoogleDrive/My Drive/Data/isotherms/'

filenames = ['VT66-c2-1.8-FS.dat',
             'VT66-c2-25-FS.dat',
             'VT66-c2-50-FS.dat',
             'VT66-c2-75-FS.dat',
             'VT66-c2-100-FS.dat',
             'VT66-c2-125-FS.dat',
             'VT66-c2-150-FS.dat',
             'VT66-c2-175-FS.dat',
             'VT66-c2-200-FS.dat',
             'VT66-c2-250-FS.dat',
             'VT66-c2-275-FS.dat',
             'VT66-c2-300-FS.dat']

tempsweeps = ['/Volumes/GoogleDrive/My Drive/Data/isotherms/VT66-c2-500-FC_00001.dat',
            '/Volumes/GoogleDrive/My Drive/Data/isotherms/VT66-c2-500-ZFC.dat',
            '/Volumes/GoogleDrive/My Drive/Data/isotherms/VT66-c2-20000-ZFC.dat',
            '/Volumes/GoogleDrive/My Drive/Data/isotherms/VT66-c2-70000-ZFC.dat',
            '/Volumes/GoogleDrive/My Drive/Data/isotherms/VT66-c2-70000-ZFC.rw.dat']

labels = [r'$1.8 (K)$',
          r'$25 (K)$',
          r'$50 (K)$',
          r'$75 (K)$',
          r'$100 (K)$',
          r'$125 (K)$',
          r'$150 (K)$',
          r'$175 (K)$',
          r'$200 (K)$',
          r'$250 (K)$',
          r'$275 (K)$',
          r'$300 (K)$']

temps = [1.8, 25,50,75,100,125,150,175,200,250,275, 300]
lines_fc,  fields_fc, mags_fc, subtracted_lines_fc, slopes_fc = [], [], [], [], []
lst = []
x_linspace = np.linspace(-7,7,10000)

for ind, file in enumerate(filenames):

    filename = main_path + file

    df = load_matrix(filename)
    df = df[['Temperature (K)', 'Magnetic Field (Oe)','DC Moment Fixed Ctr (emu)', 'DC Moment Free Ctr (emu)']]
    # arr = scipy.ndimage.filters.median_filter(np.array(df['DC Moment Fixed Ctr (emu)']), size=3)
    # interpd_arr = np.interp(np.array(df['Magnetic Field (Oe)']), np.array(df['Magnetic Field (Oe)']), arr)
    # smoothed_arr = moving_average(arr,5)
    # df['DC Moment Fixed Ctr (emu)'] = arr
    # df['DC Moment Fixed Ctr (emu)'] = df['DC Moment Fixed Ctr (emu)'].rolling(3).mean()
    df['Temperature'] = labels[ind]
    lst.append(df)

    moles = mols(5.1,299.36)
    magnetisation_fc = scipy.ndimage.filters.median_filter(np.array(df['DC Moment Fixed Ctr (emu)']), size=3)/moles
    #magnetisation_fc = np.array(df['DC Moment Free Ctr (emu)'])/moles
    field_fc = np.array(df['Magnetic Field (Oe)'])/10000

    increase_pos, decrease_pos, increase_neg, decrease_neg = increasing_v_decreasing(field_fc,0)

    #magnetisation_fc = scipy.ndimage.median_filter(scipy.ndimage.median_filter(magnetisation_fc,25),3)
    field_fc = field_fc

    fields_fc.append(field_fc)
    mags_fc.append(magnetisation_fc)

    slope_fc, const = fit_line2(magnetisation_fc,field_fc)
    lines_fc.append(slope_fc*x_linspace)
    slopes_fc.append(slope_fc)

    mag_wo_line_fc = magnetisation_fc - (slope_fc*field_fc + const)

    subtracted_lines_fc.append(mag_wo_line_fc)


big_field_df = pd.concat(lst)


big_field_df['Magnetic Field (T)'] = big_field_df['Magnetic Field (Oe)']/10000




# Figure 2 of Koyama et al FeSi

T = np.array(temps)
chi = np.array(slopes_fc)
#Ms = np.array(subtracted_lines_fc)
Ms = np.array(mags_fc)
Bs = np.array(fields_fc)

fig, ax = MakePlot(nrows=2,ncols=6,figsize=(32,18)).create()
ax_lab_size = 36
title_size = 69
plt.style.use('seaborn-paper')

ax1 = ax[0,0]
ax2 = ax[0,1]
ax3 = ax[0,2]
ax4 = ax[0,3]
ax5 = ax[0,4]
ax6 = ax[0,5]
ax7 = ax[1,0]
ax8 = ax[1,1]
ax9 = ax[1,2]
ax10 = ax[1,3]
ax11 = ax[1,4]
ax12 = ax[1,5]

axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]

for i, t in enumerate(T):
    axs[i].plot(Bs[i],Ms[i],marker='o',label=str(t),c=plt.cm.gnuplot(i/len(T)))

    axs[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    # axs[i].set_ylabel(r'Magnetization $(\frac{emu}{mol})$', fontsize=16,fontname='arial')
    # axs[i].set_xlabel(r'Magnetic Field $(T)$', fontsize=16,fontname='arial')
    axs[i].set_xlim()
    axs[i].set_ylim()
    axs[i].minorticks_on()
    axs[i].tick_params('both', which='both', direction='in',
        bottom=True, top=True, left=True, right=True)
    axs[i].yaxis.offsetText.set_fontsize(24)
    axs[i].set_title(str(t) + ' K',fontname='arial', fontsize=36)


    plt.setp(axs[i].get_xticklabels(), fontsize=28, fontname='arial')
    plt.setp(axs[i].get_yticklabels(), fontsize=28, fontname='arial')



# Get inset axis with temperature dependence

fig.text(0.5, 0.05, 'Magnetic Field (T)', ha='center', va='center', fontname='arial', fontsize='48')
fig.text(0.08, 0.5, r'Magnetisation $(\frac{\mathrm{emu}}{\mathrm{mol}})$', ha='center', va='center', rotation='vertical', fontname='arial', fontsize='48')


#plt.tight_layout(pad=6)
plt.savefig(main_path+'raw_data.png',dpi=400)
#plt.show()