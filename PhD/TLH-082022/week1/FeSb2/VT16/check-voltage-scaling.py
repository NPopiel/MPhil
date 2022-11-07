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
gs = fig.add_gridspec(3,1, hspace=0)

ax1 = fig.add_subplot(gs[:2,0])
ax2 = fig.add_subplot(gs[2,0])
# ax2 = fig.add_subplot(gs[0,1])
# ax3 = fig.add_subplot(gs[1,1])




torque_dat_100 = load_matrix('/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeSb2-VT16/0.29K_96.66deg_sweep140_up.csv')
field_100 = torque_dat_100[:, 0]
tau_100 = torque_dat_100[:, 1]

if field_100[0] > field_100[-1]:
    field_100 = np.flip(field_100)
    tau_100 = np.flip(tau_100)


tau_100 -= tau_100[0]

torque_dat_14 = load_matrix('/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeSb2-VT16/0.29K_96.66deg_sweep138_down.csv')
field_14 = torque_dat_14[:, 0]
tau_14 = torque_dat_14[:, 1]

if field_14[0] > field_14[-1]:
    field_14 = np.flip(field_14)
    tau_14 = np.flip(tau_14)


tau_14 -= tau_14[0]

torque_dat_50 = load_matrix('/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeSb2-VT16/0.29K_96.66deg_sweep141_up.csv')
field_50 = torque_dat_50[:, 0]
tau_50 = torque_dat_50[:, 1]

if field_50[0] > field_50[-1]:
    field_50 = np.flip(field_50)
    tau_50 = np.flip(tau_50)


tau_50 -= tau_50[0]

f = lambda x, a, b,c : a* x**2 + b*np.exp(c*x)

field_fit_locs_100 = field_100 < 25
field_fit_locs_50 = field_50 < 35
field_fit_locs_14 = field_14 < 34.5

popt_100, pcov_100 = curve_fit(f, field_100[field_fit_locs_100], -1*tau_100[field_fit_locs_100], p0 = (0.00001, .0001, 1))
popt_50, pcov_50 = curve_fit(f, field_50[field_fit_locs_50], -1*tau_50[field_fit_locs_50], p0 = (0.00001, .0001, 1))
popt_14, pcov_14 = curve_fit(f, field_14[field_fit_locs_14], -1*tau_14[field_fit_locs_14], p0 = (0.00001, .0001, 1))

scale_50 = popt_100[0]/popt_50[0]
scale_14 = popt_100[0]/popt_14[0]

print('50: ', scale_50)
print('14: ', scale_14)
ax1.plot(field_100, -1*tau_100, linewidth=2, c='midnightblue', label='100 V')
ax1.plot(field_50, -1*tau_50*scale_50, linewidth=2, c='indianred', label='50 V')
ax1.plot(field_14, -1*tau_14*scale_14, linewidth=2, c='darkslategray', label='14 V')

ax1.plot(field_100, f(field_100, *popt_100), linewidth=2, c='midnightblue', linestyle='dashed')
ax1.plot(field_50, f(field_50, *popt_50)*scale_50, linewidth=2, c='indianred', linestyle='dashed')
ax1.plot(field_14, f(field_14, *popt_14)*scale_14, linewidth=2, c='darkslategray', linestyle='dashed')



ax2.plot(field_100, savgol_filter(-1*tau_100, 501,3,1)*1e6, linewidth=2, c='midnightblue', label='100 V')
ax2.plot(field_50, savgol_filter(-1*tau_50*3, 501,3,1)*1e6, linewidth=2, c='indianred', label='50 V')
ax2.plot(field_14, savgol_filter(-1*tau_14*15, 501,3,1)*1e6, linewidth=2, c='darkslategray', label='14 V')



publication_plot(ax1, r'$\mu_0H$ (T)', r'$\tau$ (arb.)')

publication_plot(ax2, r'$\mu_0H$ (T)', r'$\tau^\prime$ (arb.)')

ax2.set_ybound(-0.051,0.2)


ax1.legend(framealpha=0, loc='best',
                    prop={'size': 18, 'family': 'arial'})



plt.tight_layout(pad=1)

plt.savefig('/Volumes/GoogleDrive/My Drive/Figures/TLH_08_2022/VT16/scaling-96.66.pdf', dpi=300, bbox_inches='tight')

plt.show()




