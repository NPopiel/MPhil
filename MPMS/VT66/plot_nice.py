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

main_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/Popiel/FeSb2/VT66/'

folders = ['neg2/',
                   'bonus/',
                   'neg1/',
                   '0degb/',
                   'placement_0/',
                   'b/',
                   'sample_placement1_131020/',
                   #'sample_placement2_281020/',
                   'sample_placement3_291020/',
                  #'ab/',
                   'sample_placement5_031120/',
                   'sample_placement4_021120/',
                   #'90ish/',
                   'a/',
                   'beyond_90/',
                   'morebeyond90/',
                   'more+beyond90/']



m_v_h_names = ['VT66-placement_neg2-1p8K-FS.dat',
               'VT66-placement_bonus-1p8K-FS.dat',
               'VT66-placement_neg1-1p8K-FS.dat',
               'VT66-placement_0degb-1p8K-FS.dat',
               'VT66-placement_0-1p8K-FS.dat',
               'VT66-b-1.8-FS.dat',
               'VT66-0deg_from_010-1p8K-FS_00001.dat',
               #'VT66-placement_2-1p8K-FS.dat',
               'VT66-placement_3-1p8K-FS.dat',
               #'VT66-ab-1.8-FS.dat',
               'VT66-placement_5-1p8K-FS.dat',
               'VT66-placement_4-1p8K-FS.dat',
               #'VT66-placement_90ish-1p8K-FS.dat',
               'VT66-a-1.8-FS.dat',
               'VT66-placement_beyond90-1p8K-FS.dat',
               'VT66-placement_morebeyond90-1p8K-FS.dat',
               'VT66-placement_more+beyond90-1p8K-FS.dat']



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

labels = [r'$-94^o$',
          r'$-18^o$',
          r'$-10^o$',
          r'$0^o$',
          r'$3^o$',
          'b',
          r'$21^o$',
          #r'$30^o$',
          r'$36^o$',
          #'ab',
          r'$59^o$',
          r'$64^o$',
          #r'$80^o$',
          'a',
          r'$117^o$',
          r'$127^o$',
          r'$129^o$',


]

angles = np.array([-94,
          -18,
          -10,
          0,
          3,
          19,
          21,
          #30,
          36,
          #47,
          59,
          64,
          #80,
          95,
          117,
          127,
          129]) - 21


sns.set_palette('muted')



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
    magnetisation = scipy.ndimage.filters.median_filter(np.array(field_df['DC Moment Fixed Ctr (emu)']), size=5)#p.array(field_df['DC Moment Fixed Ctr (emu)'])#
    field = np.array(field_df['Magnetic Field (Oe)'])/10000

    fields.append(field)
    mags.append(magnetisation)

    slope, const = fit_line(magnetisation,field)
    lines.append(slope*x_linspace)
    slopes.append(slope)

    mag_wo_line = savgol_filter(savgol_filter(magnetisation - slope*field, 5, 3),5,3)

    subtracted_lines.append(mag_wo_line)

    popt, pcov = scipy.optimize.curve_fit(langevin, field, mag_wo_line)

    mu_eff = popt[0]
    c_imp = popt[1]

    langevins.append(langevin(x_linspace,mu_eff,c_imp))

fig, ax = MakePlot().create()

amp = (np.max(slopes) - np.min(slopes))/2
linspace = np.linspace(-120,200,800)
sine = -1*amp*np.cos((linspace)*np.pi/90) + np.min(slopes) + amp
sine_plus = -1.0*amp*np.cos((linspace-5)*np.pi/90) + np.min(slopes) + amp #+ amp*0.025
sine_minus = -1*amp*np.cos((linspace+5)*np.pi/90) + np.min(slopes) + amp #- amp*0.025

deg_err = 0.05

min_angles = angles[np.argmin(slopes)]
print('Min Angle', min_angles)

ax.scatter(angles, slopes,c='IndianRed')
ax.plot(linspace,sine,c='k')
ax.fill_between(linspace,sine+deg_err*sine,sine-sine*deg_err,alpha=0.2,color='red')
ax.set_xlabel('Angle Away from [010]')
ax.set_ylabel(r'$\nabla M$')
ax.set_xlim()
ax.set_ylim()
# ax.annotate('Min Angle',
#             xy=(1, 0), xycoords='axes fraction',
#             xytext=(-20, 20), textcoords='offset pixels',
#             horizontalalignment='right',
#             verticalalignment='bottom')
# ax2.legend(framealpha=0,
#     bbox_to_anchor=(1, 1), loc=2,
#     title='Sweeps')
#ax1.grid()
ax.minorticks_on()
ax.tick_params('both', which='both', direction='in',
    bottom=True, top=True, left=True, right=True)
plt.title('Variation in Magnetisation by Angle')
plt.show()





fig, (ax1, ax2, ax3) = MakePlot(nrows=1, ncols=3).create()
# Plot original
for ind, arr in enumerate(mags):
    ax1.plot(fields[ind], arr, linewidth=2.5)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax1.set_ylabel('Magnetisation (emu)', fontsize=14)
ax1.set_xlabel('Magnetic Field (T)', fontsize=14)
ax1.set_title('Raw Data')
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
ax2.set_ylabel('Magnetisation (emu)', fontsize=14)
ax2.set_xlabel('Magnetic Field (T)', fontsize=14)
ax2.set_title('Linear Response')
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
ax3.set_ylabel('Magnetisation (emu)', fontsize=14)
ax3.set_xlabel('Magnetic Field (T)', fontsize=14)
ax3.set_title('Subtracted Lines')
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

plt.suptitle('Decomposition of Magnetisation by Angle', fontsize=22)
plt.legend(framealpha=0,
    bbox_to_anchor=(1, 1), loc=2,
    title='Temperature')
plt.tight_layout(pad=3.0)
plt.show()




