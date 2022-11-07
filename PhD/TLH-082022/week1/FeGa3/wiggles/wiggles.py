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


angle = 6

field_ranges = [(18.8, 21.8, 0.71),
                (26.067, 33.724, 1.7)]


def tesla_window(x, tesla_wind):
    # print(2 * int(round(0.5 * tesla_wind / x[1] - x[0])) + 1)
    return 2 * int(round(0.5 * tesla_wind / np.abs(x[1] - x[0]))) + 1





colours = ['#92140C',
           '#A3BFA8',
           '#7286A0',
           '#59594A',
           '#D7816A']

torque_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeGa3-VLS1/reload/32/'

fig, a = MakePlot(figsize=(12, 8), gs=True).create()
gs = fig.add_gridspec(1, 5)

ax1 = fig.add_subplot(gs[0,:3])
ax2 = fig.add_subplot(gs[0, 3:])


sweeps = ['0.29K_32deg_sweep118_up.csv',
          '0.29K_32deg_sweep119_up.csv']

lstyles = ['solid', 'dashed', ]

angles = [32, 32]

torque_dat = load_matrix(torque_path + '0.29K_32deg_sweep118_up.csv')
field = torque_dat[:, 0]
tau = torque_dat[:, 1]

if field[0] > field[-1]:
    field = np.flip(field)
    tau = np.flip(tau)

# if tau[-1] < 0:
#     tau *=-1

tau -= tau[0]

new_field, new_tau = interp_range(field, tau, 26.6, 34.9)

fit_tau = savgol_filter(new_tau, tesla_window(new_field, 0.71), 3)

subtracted_tau = new_tau - fit_tau

inv_x, inv_y = invert_x(new_field, subtracted_tau)

fft_x, fft_y = qo_fft(inv_x, inv_y, freq_cut=8000)

ax1.plot(new_field, subtracted_tau * 1e8, linewidth=2, c='#30C5D2', label='Up')
ax2.plot(fft_x / 1e3, fft_y / np.max(fft_y), linewidth=2, c='#30C5D2', label='Up')


torque_dat = load_matrix(torque_path + '0.29K_32deg_sweep119_up.csv')
field = torque_dat[:, 0]
tau = torque_dat[:, 1]

if field[0] > field[-1]:
    field = np.flip(field)
    tau = np.flip(tau)

# if tau[-1] < 0:
#     tau *=-1

tau -= tau[0]

new_field, new_tau = interp_range(field, tau, 26.6, 34.9)

fit_tau = savgol_filter(new_tau, tesla_window(new_field, 0.71), 3)

subtracted_tau = new_tau - fit_tau

inv_x, inv_y = invert_x(new_field, subtracted_tau)

fft_x, fft_y = qo_fft(inv_x, inv_y, freq_cut=8000)

ax1.plot(new_field, subtracted_tau * 1e8 + 3, linewidth=2, c='#471069', label='Down')
ax2.plot(fft_x / 1e3, fft_y / np.max(fft_y) + 0.5, linewidth=2, c='#471069', label='Down')


ax1.annotate('Up', xy=(0.15, 0.15), xycoords='axes fraction',
             fontname='arial', fontsize=24, color='#30C5D2')

ax1.annotate('Down', xy=(0.55, 0.85), xycoords='axes fraction',
             fontname='arial', fontsize=24, color='#471069')

ax2.annotate('Up', xy=(0.75, 0.03), xycoords='axes fraction', ha='left',
             fontname='arial', fontsize=24, color='#30C5D2')

ax2.annotate('Down', xy=(0.75, 0.4), xycoords='axes fraction',ha='left',
             fontname='arial', fontsize=24, color='#471069')

publication_plot(ax1, r'$\mu_0H$ (T)', r'$\Delta \tau$ (arb.)')
publication_plot(ax2, r'Frequency (kT)', r'FFT Amplitude (arb.)')


ax2.set_ybound(0,1.55)
ax2.set_yticks([0,0.5, 1, 1.5])


ax1.annotate(r'$\theta \approx 32 \degree$', xy=(0.95, 0.05), xycoords='axes fraction',
             ha="right", va="center", fontname='arial', fontsize=24)

plt.tight_layout(pad=1)

plt.savefig('/Volumes/GoogleDrive/My Drive/Figures/TLH_08_2022/FeGa3-VLS1/32deg-wiggles-26up.png',
            dpi=300, bbox_inches = 'tight')

plt.show()



