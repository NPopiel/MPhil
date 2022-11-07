import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelmin, argrelmax
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from tools.utils import *
from tools.ColorMaps import select_discrete_cmap
from tools.utils import *
from tools.ColorMaps import *
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import sounddevice as sd
from matplotlib.ticker import AutoMinorLocator


def qo_fft(x,y,n=6553600, window='hanning', freq_cut=0):
    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter, get_window, find_peaks

    spacing = x[1] - x[0]
    if not np.isclose(spacing,
                      np.diff(x)).all():
        raise ValueError('The data needs to be evenly spaced to smooth')
    fft_vals = np.abs(np.fft.rfft(y*get_window(window,
                                                        len(x)),
                                  n=n))
    fft_freqs = np.fft.rfftfreq(n, d=spacing)
    freq_arg = None
    if freq_cut > 0:
        freq_arg = np.searchsorted(fft_freqs, freq_cut)
    return fft_freqs[0:freq_arg], fft_vals[0:freq_arg]

def invert_x(x, y):

    """
    This inverts the x data and then reinterpolates the y points so the x
    data is evenly spread.
    Args:
        data_set (Data): The data object to invert x
    Returns:
        A data object with inverted x and evenly spaced x
    """

    from scipy.interpolate import interp1d

    if not np.all(x[:-1] <= x[1:]):
        raise ValueError('Array to invert not sorted!')
    interp = interp1d(x,y, bounds_error=False,
                     fill_value=(y[0], y[1])
                     )  # Needs fill_value for floating point errors
    new_x = np.linspace(1./x.max(), 1./x.min(),
                       len(x))
    return new_x, interp(1/new_x)

def interp_range(x, y, min_x, max_x, step_size=0.0001, **kwargs):

        if np.min(x) > min_x:
            raise ValueError('min_x value to interpolate is below data')
        if np.max(x) < max_x:
            raise ValueError('max_x value to interpolate is above data')
        x_vals = np.arange(min_x, max_x, step_size)
        return x_vals, interp1d(x, y, **kwargs)(x_vals)




fig, a = MakePlot(figsize=(12, 8), gs=True).create()
gs = fig.add_gridspec(3, 2, hspace=0)
ax1 = fig.add_subplot(gs[:2, 0])
ax2 = fig.add_subplot(gs[2, 0])
gs = fig.add_gridspec(3, 2)
ax3 = fig.add_subplot(gs[:,1])






path = '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/'




filenames = [path+'0.035K_192deg_sweep306_up.csv',
             '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/0.035K_186deg_sweep226_up.csv',
             path+'0.035K_174deg_sweep304_up.csv',
            '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/0.035K_162deg_sweep232_down.csv',
            '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/0.035K_150deg_sweep239_up.csv',
            '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/0.035K_144deg_sweep250_down.csv',
             path+'0.035K_126deg_sweep299_up.csv',
             path+'0.035K_114deg_sweep296_up.csv',
            '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/0.035K_102deg_sweep289_down.csv',
            '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/0.035K_96deg_sweep283_down.csv'
]



field_ranges = (25.1, 27.9, 0.71)



def tesla_window(x, tesla_wind):
    # print(2 * int(round(0.5 * tesla_wind / x[1] - x[0])) + 1)
    return 2 * int(round(0.5 * tesla_wind / np.abs(x[1] - x[0]))) + 1





xs, inv_xs, ys, inv_ys = [], [], [], []

angles = [-12,-6, 6, 18, 30, 36, 54,66, 78,84]


frequencys = [(3.2, 4, 4.7, 5.2, 7.8),#-12
              (4.3, 5.6,6.2, None, None),#-6
              (3.9, 4.6, 5.1, None, None),#6
              (4,  4.9, 5.4, 6.1, None),#18 6.1 is maybe
              (3.5, 4.1, 4.5, 5.3,7.0),#30 7.0
              (3.3, 3.7, 4, 4.5, 7.9),#367.7
              (3.1, 3.6, 4.1, None, 7.1),#54 7.1
              (3.47,4,4.7,None,6.9),#66 ,9.9
              (3.5,4.6, None, None, 7.7),#78
              (3.8, 4.85,5.4, None, 7.45)] # 84

markers = ['o', '+', 'v', 'x','4']

# colours = ['#56CEBE',
#            '#60ADBF',
#            '#6A8DC0',
#            '#746CC1',
#            '#7E4BC2',
#            '#882BC3',
#            '#920AC4']

colours = ['#3bcfd4','#6CC0A1','#9CB16D','#CCA239','#E49B1F','#FC9305','#FA6F29','#F74A4D','#F52571','#F20094']

# Plot the frequencies in the right-most panel

fft_xs, fft_ys = [], []

grads = []

slopes = []
for i, filename in enumerate(filenames):

        # dat = load_matrix(filename,delimeter='\t',skiprows=2, dat_type='other')
        dat = load_matrix(filename)

        field = dat[:,0]
        tau = dat[:,1]

        inds1 = field > 25.1
        inds2 = field < 27.9

        inds = inds1 & inds2



        field = field[inds]
        tau = tau[inds]

        if field[0] > field[-1]:

            tau = np.flip(tau)
            field = np.flip(field)

        tau -= tau[0]

        tau = np.abs(tau)

        fit_inds1 = field > 25.1
        fit_inds2 = field < 27.9

        fit_inds = fit_inds1 & fit_inds2

        fit = np.polyfit(field[fit_inds], tau[fit_inds], 1)

        ax1.plot(field, 1e4*tau, linewidth=2, c=colours[i], label=angles[i])
        ax1.plot(field, 1e4*(fit[0]*field + fit[1]), linewidth=1, linestyle='dashed', c=colours[i], label=angles[i])

        new_field, new_tau = interp_range(field, tau, 25.11, 27.89)
        deriv_tau = savgol_filter(new_tau, tesla_window(new_field, 0.1), 3, 1)


        ax2.plot(new_field, 1e8*deriv_tau, linewidth=2, c=colours[i], label=angles[i])

        ax3.scatter(angles[i], 1e4*fit[0], s=250,marker='o',facecolors='none', edgecolor=colours[i])

        slopes.append(1e4*fit[0])



def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 0., guess_offset])

    def sinfunc(t, A, p, c):  return A * np.sin(4*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess, maxfev=250000)
    A, p, c = popt
    f = 2/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(4*t + p) + c
    return {"amp": A, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

res = fit_sin(np.deg2rad(angles), slopes)
print( "Amplitude=%(amp)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s" % res )

oversamples_angles = np.deg2rad(np.linspace(-10,90,250))

print(np.rad2deg(res['phase']))

ax3.plot(np.rad2deg(oversamples_angles), res["fitfunc"](oversamples_angles), c='k', linewidth=2)

print(np.rad2deg(oversamples_angles)[np.argmax(res["fitfunc"](oversamples_angles))])

ax1.set_xbound(25,28)
ax2.set_xbound(25,28)
ax1.set_xticks([25, 26,27,28])
ax1.set_xticklabels([])
ax2.set_xticks([25, 26,27,28])
ax2.set_xticklabels([25, 26,27,28])



publication_plot(ax1, '', r'$\tau$ (arb.)')
publication_plot(ax2, r'$(\mu_0H)$ (T)', r'$\frac{\partial\tau}{\partial \mu_0H}$ (arb.)')
publication_plot(ax3, r'$\theta$ ($\degree$)', r'$\tau^\prime$')


# fft_ax.set_xbound(0,25.05)

plt.tight_layout(pad=1.5)
# plt.savefig('/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/'+'derivs-s6.pdf', dpi=300)
plt.show()




