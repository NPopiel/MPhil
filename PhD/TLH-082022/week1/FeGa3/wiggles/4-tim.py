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

field_ranges = [(24.087, 34.9, 2.2),
                (26.067, 33.724, 1.7)]


def tesla_window(x, tesla_wind):
    # print(2 * int(round(0.5 * tesla_wind / x[1] - x[0])) + 1)
    return 2 * int(round(0.5 * tesla_wind / np.abs(x[1] - x[0]))) + 1





colours = ['#92140C',
           '#A3BFA8',
           '#7286A0',
           '#59594A',
           '#D7816A']

torque_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeGa3-VLS1/'

fig, a = MakePlot(figsize=(12, 12), gs=True).create()
gs = fig.add_gridspec(2, 5)

ax1 = fig.add_subplot(gs[0,:3])
ax2 = fig.add_subplot(gs[0, 3:])
ax3 = fig.add_subplot(gs[1,:3])
ax4 = fig.add_subplot(gs[1,3:])

ax1s = [ax1, ax3]
ax2s = [ax2, ax4]

sweeps = ['0.29K_26deg_sweep040_down.csv',
          '0.29K_33.33deg_sweep046_down.csv']

lstyles = ['solid', 'dashed', ]

angles = [26, 33]

for i, s_name in enumerate(sweeps):


    torque_dat = load_matrix(torque_path + s_name)
    field = torque_dat[:, 0]
    tau = torque_dat[:, 1]

    if field[0] > field[-1]:
        field = np.flip(field)
        tau = np.flip(tau)

    # if tau[-1] < 0:
    #     tau *=-1

    tau -= tau[0]

    new_field, new_tau = interp_range(field, tau, field_ranges[i][0], field_ranges[i][1])

    fit_tau = savgol_filter(new_tau, tesla_window(new_field, field_ranges[i][2]), 3)

    subtracted_tau = new_tau - fit_tau

    inv_x, inv_y = invert_x(new_field, subtracted_tau)

    fft_x, fft_y = qo_fft(inv_x, inv_y, freq_cut=8000)

    ax1s[i].plot(new_field, subtracted_tau*1e8, linewidth=2, c=colours[i], label=str(angles[i])+r'$\degree$')
    ax2s[i].plot(fft_x/1e3, fft_y/np.max(fft_y), linewidth=2, c=colours[i], label=str(angles[i]) + r'$\degree$')

    # ax1.annotate(str(angles[i]) + r'$\degree$', xy=(1.03, i * 0.1125), xycoords='axes fraction',
    #              fontname='arial', fontsize=16, color=colours[i])

publication_plot(ax1, r'$\mu_0H$ (T)', r'$\Delta \tau$ (arb.)')
publication_plot(ax2, r'', r'FFT Amplitude (arb.)')
publication_plot(ax3, r'$\mu_0H$ (T)', r'$\Delta \tau$ (arb.)')
publication_plot(ax4, r'Frequency (kT)', r'FFT Amplitude (arb.)')

ax2.set_ybound(0,1.05)
ax2.set_yticks([0,0.5, 1])
ax4.set_ybound(0,1.05)
ax4.set_yticks([0,0.5, 1])

ax1.annotate(r'$\theta \approx 26 \degree$', xy=(0.95, 0.9), xycoords='axes fraction',
             ha="right", va="center", fontname='arial', fontsize=24)
ax3.annotate(r'$\theta \approx 30 \degree$', xy=(0.05, 0.1), xycoords='axes fraction',
             ha="left", va="center", fontname='arial', fontsize=24)

plt.tight_layout(pad=1)

plt.savefig('/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeGa3-VLS1/good/15up/4-tim.png',
            dpi=300, bbox_inches = 'tight')



