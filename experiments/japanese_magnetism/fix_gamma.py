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
    #arr = scipy.ndimage.filters.median_filter(np.array(field_df['DC Moment Fixed Ctr (emu)']), size=3)
    #interpd_arr = np.interp(np.array(field_df['Magnetic Field (Oe)']), np.array(field_df['Magnetic Field (Oe)']), arr)
    #smoothed_arr = moving_average(arr,5)
    #field_df['DC Moment Fixed Ctr (emu)'] = arr
    #field_df['DC Moment Fixed Ctr (emu)'] = field_df['DC Moment Fixed Ctr (emu)'].rolling(3).mean()
    df['Temperature'] = labels[ind]
    lst.append(df)
    print(mols(5.1e-3,299.36))
    magnetisation_fc = scipy.ndimage.filters.median_filter(np.array(df['DC Moment Free Ctr (emu)']), size=5)/mols(5.1e-3,299.36)
    field_fc = np.array(df['Magnetic Field (Oe)'])/10000

    increase_pos, decrease_pos, increase_neg, decrease_neg = increasing_v_decreasing(field_fc,0)

    magnetisation_fc = savgol_filter(savgol_filter(magnetisation_fc[increase_pos],3,1),3,1)
    field_fc = field_fc[increase_pos]

    if file == 'VT66-c2-300-FS.dat':
        magnetisation_fc = magnetisation_fc[:-3]
        field_fc = field_fc[:-3]

    fields_fc.append(field_fc)
    mags_fc.append(magnetisation_fc)

    slope_fc, const = fit_line2(magnetisation_fc,field_fc)
    lines_fc.append(slope_fc*x_linspace)
    slopes_fc.append(slope_fc)

    mag_wo_line_fc = magnetisation_fc - (slope_fc*field_fc + const)

    subtracted_lines_fc.append(mag_wo_line_fc)

#
# big_field_df = pd.concat(lst)
#
#
# big_field_df['Magnetic Field (T)'] = big_field_df['Magnetic Field (Oe)']/10000
#
#
# # Need to interpolate the temps and the slopes(susc)
#
# # Figure 1 of Koyama et al FeSi
#
# T = np.array(temps)
# chi = np.array(slopes_fc)
# Ms = np.array(mag_wo_line_fc)
#
# oversamled_temps = np.linspace(np.min(T),np.max(T),300)
#
# interpd_chi = scipy.interpolate.griddata(T,chi,oversamled_temps,'cubic')
# interpd_inv_chi = scipy.interpolate.griddata(T,1/chi,oversamled_temps,'linear')
#
# plt.suptitle('VT66 Magnetization', fontsize=18,fontname='Times')
# plt.tight_layout(pad=3.0)
# plt.show()
#
# fig, ax = MakePlot(nrows=1, ncols=1).create()
# # Plot original
#
# ax.plot(T,1/chi,  linewidth=2.5, c='r', marker='o', label=r'$\frac{1}{\chi}$')
# ax.plot(oversamled_temps,interpd_inv_chi,  linewidth=2.5, linestyle='--',c='r', label=r'$\frac{1}{\chi}$ Interpolated')
# ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax.set_ylabel(r'$\frac{1}{\chi}$', fontsize=12,fontname='Times', color='r')
# ax.set_xlabel(r'Temperature $(K)$', fontsize=12,fontname='Times')
# ax.set_xlim()
# ax.set_ylim()
# ax.minorticks_on()
# ax.tick_params('both', which='both', direction='in',
#     bottom=True, top=True, left=True, right=True)
#
# ax2 = ax.twinx()
# ax2.plot(T,chi,linewidth=2.5, c='k', marker='o', label=r'$\chi$')
# ax2.plot(oversamled_temps,interpd_chi,linewidth=2.5, linestyle='--', c='k', label=r'$\chi$ Intrepolated')
# ax2.set_ylabel(r'$\chi$', fontsize=12,fontname='Times', color='k')
# ax2.set_xlim()
# ax2.set_ylim()
# ax2.minorticks_on()
# ax2.tick_params('both', which='both', direction='in',
#     bottom=True, top=True, left=True, right=True)
#
# plt.suptitle('VT66 Magnetization', fontsize=18,fontname='Times')
# plt.tight_layout(pad=3.0)
# #plt.legend()
# plt.show()


# Figure 2 of Koyama et al FeSi

T = np.array(temps)
chi = np.array(slopes_fc)
#Ms = np.array(subtracted_lines_fc)
Ms = np.array(mags_fc)
Bs = np.array(fields_fc)

# fig, ax = MakePlot().create()
#
# for i, t in enumerate(T):
#     ax.plot(Bs[i][3:],Ms[i][3:],marker='o',label=str(t))
#
# ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax.set_ylabel(r'Magnetization (emu)', fontsize=12,fontname='arial')
# ax.set_xlabel(r'Magnetic Field $(T)$', fontsize=12,fontname='arial')
# ax.set_xlim()
# ax.set_ylim()
# ax.minorticks_on()
# ax.tick_params('both', which='both', direction='in',
#     bottom=True, top=True, left=True, right=True)
#
# plt.legend(title=r'Temperature $(K)$', loc='best',frameon=True, fancybox=False, edgecolor='k', framealpha=1, borderpad=1)
# plt.title('Magnetization in Positive Field Region',fontsize=18,fontname='arial')
#
# plt.show()

# Figure 3 of Koyama et al FeSi

M_sqr = Ms**2
H_over_M = Bs/Ms

string_temps = labels

temperatures = T

# fig, ax = MakePlot().create()
#
# for i, t in enumerate(string_temps):
#     ax.plot(H_over_M[i][3:],M_sqr[i][3:],marker='o',label=str(t))
#
# ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax.set_ylabel(r'$M^2$ $(emu^2)$', fontsize=12,fontname='arial')
# ax.set_xlabel(r'$\frac{H}{M}$ $(\frac{T}{emu})$', fontsize=12,fontname='arial')
# ax.set_xlim()
# ax.set_ylim()
# ax.minorticks_on()
# ax.tick_params('both', which='both', direction='in',
#     bottom=True, top=True, left=True, right=True)
#
# plt.legend(title=r'Temperature $(K)$', loc='best',frameon=True, fancybox=False, edgecolor='k', framealpha=1, borderpad=1)
# plt.title('Arrot Plot in Positive Field Region',fontsize=18,fontname='arial')
#
# plt.show()
#
# # Fiig 3b
# fig, ax = MakePlot().create()
#
#
# for i, t in enumerate(string_temps):
#     ax.plot(H_over_M[i]-1/chi[i],M_sqr[i],marker='o',label=str(t))
#
# ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax.set_ylabel(r'$M^2$ $(emu^2)$', fontsize=12,fontname='arial')
# ax.set_xlabel(r'$\frac{H}{M} - \frac{1}{\chi}$ $(\frac{T}{emu})$', fontsize=12,fontname='arial')
# ax.set_xlim()
# ax.set_ylim()
# ax.minorticks_on()
# ax.tick_params('both', which='both', direction='in',
#     bottom=True, top=True, left=True, right=True)
#
# plt.legend(title=r'Temperature $(K)$', loc='best',frameon=True, fancybox=False, edgecolor='k', framealpha=1, borderpad=1)
# plt.title('Arrot Plot in Positive Field Region',fontsize=18,fontname='arial')
#
# plt.show()

# 3B 150 K zoom


# Now ill attempt to fit a line to the arrot plot after removing the first 8 points!
lines_to_fit = []
inverse_gammas = []
err = []
for q in range(len(H_over_M)):
    # fig, ax = MakePlot().create()
    x = H_over_M[q]- 1 / chi[q]
    y = M_sqr[q]
    l = len(y)
    start = int(round(l/3))

    y = scipy.ndimage.median_filter(y,size=10)
    fit,cov = np.polyfit(x[start:],y[start:],1, cov=True)
    # error is 1/m^2 * err
    err.append(((np.sqrt(np.diag(cov))[0])*(1/fit[0]**2)))
    inverse_gammas.append(fit[0])

    print('len of fit region', len(x[15:]))

    # ax.plot(H_over_M[q]- 1 / chi[q] , M_sqr[q], marker='o')#
    # ax.plot(x[start:],np.poly1d(fit)(x[start:]) )
    #
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax.set_ylabel(r'$M^2$ $(emu^2)$', fontsize=12, fontname='arial')
    # ax.set_xlabel(r'$\frac{H}{M} - \frac{1}{\chi}$ $(\frac{T}{emu})$', fontsize=12, fontname='arial')
    # ax.set_xlim()
    # ax.set_ylim()
    # ax.minorticks_on()
    # ax.tick_params('both', which='both', direction='in',
    #                bottom=True, top=True, left=True, right=True)
    #
    # plt.legend(title=r'Temperature $(K)$', loc='best', frameon=True, fancybox=False, edgecolor='k', framealpha=1,
    #            borderpad=1)
    # plt.title('Arrot Plot in Positive Field Region for' + string_temps[q], fontsize=18, fontname='arial')
    #
    # plt.show()


fig, ax = MakePlot().create()

print(1/np.array(inverse_gammas))

ax.scatter(temperatures[4:], 1/np.array(inverse_gammas)[4:],s=150,c='red')
ax.errorbar(temperatures[4:], 1/np.array(inverse_gammas)[4:],yerr=err[4:],fmt='none',c='k',linewidth=2.1)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0),useMathText=True)
ax.set_ylabel(r'$\gamma$', fontsize=28, fontname='arial')
ax.set_xlabel(r'Temperature (K)', fontsize=28, fontname='arial')
# ax.set_xlim()
# ax.set_ylim()
ax.set_yscale('log')
ax.minorticks_on()
ax.tick_params('both', which='major', direction='in', length=6, width=2,
               bottom=True, top=True, left=True, right=True)

ax.tick_params('both', which='minor', direction='in', length=4, width=2,
               bottom=True, top=True, left=True, right=True)

plt.setp(ax.get_xticklabels(), fontsize=24, fontname='arial')
plt.setp(ax.get_yticklabels(), fontsize=24, fontname='arial')

plt.savefig(main_path+'gamma-T.png', dpi=400)

