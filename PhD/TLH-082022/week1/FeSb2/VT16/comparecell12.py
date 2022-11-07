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


colours = ['#001219',
           '#005F73',
           '#0A9396',
           '#94D2BD',
           '#E9D8A6',
           '#EE9B00',
           '#CA6702',
           '#BB3E03',
           '#AE2012',
           '#9B2226']


def tesla_window(x, tesla_wind):
    # print(2 * int(round(0.5 * tesla_wind / x[1] - x[0])) + 1)
    return 2 * int(round(0.5 * tesla_wind / np.abs(x[1] - x[0]))) + 1



torque_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeSb2-VT16/'

fig, a = MakePlot(figsize=(12, 16), gs=True).create()
gs = fig.add_gridspec(2, 3)

ax1 = fig.add_subplot(gs[:,0])
ax2 = fig.add_subplot(gs[:,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[1,2])

axin1 = ax1.inset_axes([0.2,0.4,.45,.35])
axin2 = ax2.inset_axes([0.2,0.4,.45,.35])



torque_dat = load_matrix('/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeSb2-VT16/0.29K_103.33deg_sweep135_up.csv')
field = torque_dat[:, 0]
tau = torque_dat[:, 1]

locs = field > 27

field = field[locs]
tau = tau[locs]

if field[0] > field[-1]:
    field = np.flip(field)
    tau = np.flip(tau)


tau -= tau[0]

ax1.plot(field, tau*1e5, linewidth=2, c='midnightblue', label='VT16')

new_field, new_tau = interp_range(field, tau, 33.15, 34.9)

fit_tau = savgol_filter(new_tau, tesla_window(new_field, .71), 3)

subtracted_tau = new_tau - fit_tau

ax3.plot(new_field, subtracted_tau,linewidth=2, c='midnightblue', label='VT16')

inv_x, inv_y = invert_x(new_field, subtracted_tau)

fft_x, fft_y = qo_fft(inv_x, inv_y, freq_cut=10000)
axin1.plot(fft_x/1e3, fft_y/np.max(fft_y), linewidth=1.1, c='midnightblue', label='VT16')


vt4_dat = load_matrix('/Users/npopiel/Desktop/Hybrid/VT4-cell12_14.dat', dat_type='tlh',skiprows=10, delimeter='\t')
field = vt4_dat[:, 0]
tau = vt4_dat[:, 1]

if field[0] > field[-1]:
    field = np.flip(field)
    tau = np.flip(tau)


tau -= tau[0]

ax2.plot(field, tau*1e5, linewidth=2, c='indianred', label='VT4')

new_field, new_tau = interp_range(field, tau, 28.07, 34.9)

fit_tau = savgol_filter(new_tau, tesla_window(new_field, .71), 3)

subtracted_tau = new_tau - fit_tau

ax4.plot(new_field, subtracted_tau,linewidth=2, c='indianred', label='VT4')

inv_x, inv_y = invert_x(new_field, subtracted_tau)

fft_x, fft_y = qo_fft(inv_x, inv_y, freq_cut=10000)
axin2.plot(fft_x/1e3, fft_y/np.max(fft_y), linewidth=1.1, c='indianred', label='VT4')




publication_plot(ax1, r'$\mu_0H$ (T)', r'$\tau$ (arb.)')
publication_plot(ax2, r'$\mu_0H$ (T)', r'$\tau$ (arb.)')
publication_plot(ax3, r'$\mu_0H$ (T)', r'$\Delta\tau$ (arb.)')
publication_plot(ax4, r'$\mu_0H$ (T)', r'$\Delta\tau$ (arb.)')

publication_plot(axin1, r'Frequency (kT)', r'FFT Amplitude (arb.)', label_fontsize=14, tick_fontsize=12)
publication_plot(axin2, r'Frequency (kT)', r'FFT Amplitude (arb.)', label_fontsize=14, tick_fontsize=12)


handles, labels = ax1.get_legend_handles_labels()

ax1.legend(framealpha=0, loc='best',
                    prop={'size': 18, 'family': 'arial'},
                    handlelength=0)


ax2.legend(framealpha=0, loc='best',
                    prop={'size': 18, 'family': 'arial'},
                    handlelength=0)

plt.tight_layout(pad=1)

plt.savefig('/Volumes/GoogleDrive/My Drive/Figures/TLH_08_2022/VT16/VT15-VT16-compare.pdf', dpi=300, bbox_inches='tight')


plt.show()




