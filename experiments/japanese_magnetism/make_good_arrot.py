import pandas as pd
from tools.utils import *
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
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

filenames = [
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

labels = [
          r'$125 (\mathrm{K})$',
          r'$150 (\mathrm{K})$',
          r'$175 (\mathrm{K})$',
          r'$200 (\mathrm{K})$',
          r'$250 (\mathrm{K})$',
          r'$275 (\mathrm{K})$',
          r'$300 (\mathrm{K})$']

temps = [125,150,175,200,250,275, 300]
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


T = np.array(temps)
chi = np.array(slopes_fc)
#Ms = np.array(subtracted_lines_fc)
Ms = np.array(mags_fc)
Bs = np.array(fields_fc)


# Figure 3 of Koyama et al FeSi

M_sqr = Ms**2
H_over_M = Bs/Ms

string_temps = labels

temperatures = T

fig, ax = MakePlot().create()

offset_val = 0.5

plt.xticks([-0,0.5,1.0,1.5,2.0,2.5,3.0,3.5],[0,0,0,0,0,0,0,0])

for i, t in enumerate(string_temps):

    # This plots everything
    ax.plot((H_over_M[i]-1/chi[i])+i*offset_val,M_sqr[i],marker='.',label=str(t),color=plt.cm.cool(i/len(string_temps)))
    # This plots only the last 15 points
    #ax.plot((H_over_M[i]-1/chi[i])[-15:]+i*offset_val,M_sqr[i][-15:],marker='o',label=str(t))
    # Here is code to change the colors of tick labels. I need to write a loop to assign the same value as data to the label 0.
    plt.setp(ax.get_xticklabels()[i], color=plt.cm.cool(i/len(string_temps)))

ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0),useMathText=True)
ax.set_ylabel(r'$M^2$ $(\frac{\mathrm{emu}^2}{\mathrm{mol}^2})$', fontsize=28,fontname='arial')
ax.set_xlabel(r'$\frac{H}{M} - \frac{1}{\chi}$ $(\frac{\mathrm{T}}{\mathrm{emu}})$', fontsize=28,fontname='arial')

ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
# ax.tick_params('y', which='both', direction='in',
#     left=True, right=True)
# ax.tick_params('x', which='major', direction='in',
#     bottom=True, top=True)

ax.tick_params('both', which='major', direction='in', length=6, width=2,
               bottom=True, top=True, left=True, right=True)

ax.tick_params('y', which='minor', direction='in', length=4, width=2,
               bottom=True, top=True, left=True, right=True)
ax.yaxis.offsetText.set_fontsize(18)
# plt.legend(title=r'Temperature $(K)$', loc='best',frameon=False, fancybox=False, framealpha=0, borderpad=1)
#ax.set_title('Magnetization in Positive Field Region',fontsize=title_size,fontname='arial', pad=30)
#
# ax.annotate(r'FeSb$_2$',xy=(4,3.2e-2),fontname='arial',fontsize=24,va='center',ha='center')
# ax.annotate(r'VT66',xy=(4,2.9e-2),fontname='arial',fontsize=24,va='center',ha='center')
# ax.annotate(r'$ \vec H \parallel c $',xy=(4,2.6e-2), fontname='arial',fontsize=24,va='center',ha='center')

handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles, labels, framealpha=0, ncol=1,  # len(dset)//12+
                    title='Temperature (K)', prop={"size": 22})

plt.setp(ax.get_xticklabels(), fontsize=24, fontname='arial')
plt.setp(ax.get_yticklabels(), fontsize=24, fontname='arial')
plt.setp(legend.get_title(), fontsize=24, fontname='arial')

for l in legend.get_lines():
    l.set_linewidth(4)


#Here i need to get the xtick labels, and loop over them setting them all to 0




ax.set_ylim()
#ax.minorticks_on()
# ax.tick_params('both', which='both', direction='in',
#     bottom=True, top=True, left=True, right=True)

#plt.legend(title=r'Temperature $(K)$', loc='best',frameon=True, fancybox=False, edgecolor='k', framealpha=1, borderpad=1)

fig.savefig(main_path+'arrot_no_fit.png',dpi=400)


#plt.show()

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