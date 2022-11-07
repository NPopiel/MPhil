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


def qo_fft(x, y, n=6553600, window='hanning', freq_cut=0):
    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter, get_window, find_peaks

    spacing = x[1] - x[0]
    if not np.isclose(spacing,
                      np.diff(x)).all():
        raise ValueError('The data needs to be evenly spaced to smooth')
    fft_vals = np.abs(np.fft.rfft(y * get_window(window,
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
    interp = interp1d(x, y, bounds_error=False,
                      fill_value=(y[0], y[1])
                      )  # Needs fill_value for floating point errors
    new_x = np.linspace(1. / x.max(), 1. / x.min(),
                        len(x))
    return new_x, interp(1 / new_x)

def interp_range(x, y, min_x, max_x, step_size=0.0001, **kwargs):
    if np.min(x) > min_x:
        raise ValueError('min_x value to interpolate is below data')
    if np.max(x) < max_x:
        raise ValueError('max_x value to interpolate is above data')
    x_vals = np.arange(min_x, max_x, step_size)
    return x_vals, interp1d(x, y, **kwargs)(x_vals)

def lifshitz_kosevich(temps, e_mass, amp, field=26.96657407):
    kb = 1.380649e-23
    me = 9.1093837015e-31
    hbar = 1.054571817e-34
    qe = 1.602176634e-19

    chi = 2 * np.pi * np.pi * kb * temps * me * e_mass / (hbar * qe * field)

    r_lk = amp * chi / np.sinh(chi)

    return r_lk

def lk_field_val(min_field, max_field):
    denom = 1 / min_field + 1 / max_field
    return 2 / denom



path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2022_10_TLH/week2/torque/VT16/day3/new/'



filenames = ['0.4K_74deg_sweep144_up.csv']

angle = 6

field_ranges = (36.05, 38.73, 0.71)


def tesla_window(x, tesla_wind):
    # print(2 * int(round(0.5 * tesla_wind / x[1] - x[0])) + 1)
    return 2 * int(round(0.5 * tesla_wind / np.abs(x[1] - x[0]))) + 1


fig, a = MakePlot(figsize=(12, 12), gs=True).create()
gs = fig.add_gridspec(1, 1)

ax1 = fig.add_subplot(gs[0,0])

# ax4 = fig.add_subplot(gs[1, 5:])


temperatures = [0.75, 1.45]  # These are definitely wrong and mass is much heavier as a consequence
freq1 = np.array([0.002242, 0.001818])
# freq2 = np.array([0.001248,])
freq2 = np.array([0.00124, 483.51e-6])

colours = ['#264653', '#2A9D8F', '#BABB74', '#E9C46A', '#EE8959', '#E76F51']



temperatures_K = np.squeeze(np.array(temperatures))

popt1, pcov1 = scipy.optimize.curve_fit(lifshitz_kosevich, temperatures_K, freq1, p0=(60, 2.3515721812642054e-05))
# popt2, pcov2 = scipy.optimize.curve_fit(lifshitz_kosevich, temperatures_K, freq2, p0=(60, 2.3515721812642054e-05))

perr1 = np.sqrt(np.diag(pcov1))
# perr2 = np.sqrt(np.diag(pcov2))

oversampled_temps = np.linspace(0.0001, 15, 10000)

lk1 = lifshitz_kosevich(oversampled_temps, *popt1)
# lk2 = lifshitz_kosevich(oversampled_temps, *popt2)

print('m1: ', popt1[0], '+/- ', perr1[0])
# print('m2: ', popt2[0], '+/- ', perr2[0])
#
# print('Amp1: ', popt1[1], '+/- ', perr1[1])
# print('amp2: ', popt2[1], '+/- ', perr2[1])

ax1.plot(oversampled_temps, lk1, c='darkslategray', label=r'$F_1$', linestyle='dashed',zorder=-1)
# ax1.plot(oversampled_temps, lk2, c='darkslategray', label=r'$F_2$',zorder=-1)

ax1.scatter(temperatures_K, freq1, c='r', s=200)
# ax1.scatter(temperatures_K, freq2,facecolor='none', edgecolor='r', s=200)


publication_plot(ax1, 'Temperature (K)', 'FFT Amplitude (arb.)')

plt.tight_layout(pad=0.99)
plt.show()



