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

main_path = '/Users/npopiel/Documents/MPhil/Data/VT68/'




m_v_h_names = ['/Users/npopiel/Documents/MPhil/Data/VT68/VT68-placement_c1-1p8K-FS.dat',
               '/Users/npopiel/Documents/MPhil/Data/VT68/VT68-placement2-1p8K-FS_00001.dat',
               '/Users/npopiel/Documents/MPhil/Data/VT68/VT68-placement3-1p8K-FS.dat',
               '/Users/npopiel/Documents/MPhil/Data/VT68/VT68-placement4-1p8K-FS.dat',
               '/Users/npopiel/Documents/MPhil/Data/VT68/VT68-placement5-1p8K-FS.dat',
               '/Users/npopiel/Documents/MPhil/Data/VT68/VT68-placement6-1p8K-FS.dat',
               '/Users/npopiel/Documents/MPhil/Data/VT68/VT68-placement7-1p8K-FS.dat',
               '/Users/npopiel/Documents/MPhil/Data/VT68/VT68-placement8-1p8K-FS.dat',
               '/Users/npopiel/Documents/MPhil/Data/VT68/VT68-placement9-1p8K-FS.dat',
               '/Users/npopiel/Documents/MPhil/Data/VT68/VT68-placement10-1p8K-FS.dat',
               '/Users/npopiel/Documents/MPhil/Data/VT68/VT68-placement14-1p8K-FS.dat',
               '/Users/npopiel/Documents/MPhil/Data/VT68/VT68-placement17-1p8K-FS.dat',
               '/Users/npopiel/Documents/MPhil/Data/VT68/VT68-placement18-1p8K-FS.dat',
               '/Users/npopiel/Documents/MPhil/Data/VT68/VT68-placement19-1p8K-FS.dat',
               '/Users/npopiel/Documents/MPhil/Data/VT68/VT68-placement20-1p8K-FS.dat',
               '/Users/npopiel/Documents/MPhil/Data/VT68/VT68-placement21-1p8K-FS.dat',
               '/Users/npopiel/Documents/MPhil/Data/VT68/VT68-placement22-1p8K-FS.dat']






relevant_cols = ['Temperature (K)', 'Magnetic Field (Oe)', 'DC Moment Fixed Ctr (emu)', 'DC Moment Free Ctr (emu)']

# here are the original column names for reference
# the new column names I created



# Temp sweep has for num2 39 same for field sweep
# 26 on royce

# First loop over all of the non-placement 7 stuff
#get two lists, one for f(T), one f(H)



angles = [0,
          10,
          14,
          52,
          45,
          60,
          40,
          87,
          68,
          140,
          26,
          16,
          -65,
          -44,
          -70,
          -90,
          -130
          ]

field_lst = []
sns.set_palette('muted',n_colors=25)



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

def sinusoid(amp, freq, phase, mean):
    N = 1000  # number of data points
    t = np.linspace(-2*np.pi, 2 * np.pi, N)
    return amp * np.sin(freq*t + phase) + mean

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}



x_linspace = np.linspace(-7,7,10000)

lines, langevins, fields, mags, subtracted_lines, slopes = [], [], [], [], [], []
for ind, folder in enumerate(m_v_h_names):

    filename_field = m_v_h_names[ind]

    field_df = load_matrix(filename_field)
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

#optimize_func = lambda x: sinusoid(x[0],x[1],x[2], x[3]) - slopes
#est_amp, est_freq, est_phase, est_mean = scipy.optimize.leastsq(optimize_func, [-1*amp, 1/2, 0, np.min(slopes) + amp])[0]

#est_sine = sinusoid(est_amp,est_freq,est_phase,est_mean)

N, amp, omega, phase, offset, noise = 1000, 1., 2., .5, 4., 3

tt = np.linspace(-120, 150, N)



res = fit_sin(angles, slopes)
error = np.sqrt(res['maxcov'])

sine_plus = res["fitfunc"](tt)*(1+error)
sine_minus = res["fitfunc"](tt)*(1-error)

print( "Amplitude=%(amp)s, Angular freq.=%(omega)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s" % res )

plt.plot(angles, slopes, "or", label="Raw Data")
plt.plot(tt, res["fitfunc"](tt), "k-", label="Fitted Function", linewidth=2)
ax.set_xlabel('Angle Away from [001] (Degrees)',fontsize=12,fontname='Times')
ax.set_ylabel(r'Susceptibility $(\frac{emu}{T})$', fontsize=12,fontname='Times')
#ax.fill_between(tt,sine_plus, sine_minus,alpha=0.2,color='gray')
ax.axvline(0,linestyle="--",c='rosybrown',label='c')
ax.axvline(90,linestyle='--', c='firebrick',label='ab')
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
plt.title('Variation in Magnetisation by Angle', fontname='Times', fontsize=18)


plt.legend(loc="best")
plt.show()


sine = -1*amp*np.cos((linspace)*np.pi/90) + np.min(slopes) + amp
sine_plus = -1.0*amp*np.cos((linspace-5)*np.pi/90) + np.min(slopes) + amp #+ amp*0.025
sine_minus = -1*amp*np.cos((linspace+5)*np.pi/90) + np.min(slopes) + amp #- amp*0.025

deg_err = 0.05

min_angles = angles[np.argmin(slopes)]
print('Min Angle', min_angles)

# ax.scatter(angles, slopes,c='IndianRed')
# ax.plot(linspace,sine,c='k')
# ax.plot(linspace,est_sine,c='b')




sns.set_palette('husl')
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
    ax3.plot(fields[ind], subtracted_line, label=angles[ind], linewidth=2.5)
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





