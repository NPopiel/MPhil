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




path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeSb2-VT19/resistance/'




filenames = [path+'0.29K_-6.66deg_sweep010_up.csv',
             path+'0.29K_-6.66deg_sweep012_down.csv']

angles = ['1', '2']


def tesla_window(x, tesla_wind):
    # print(2 * int(round(0.5 * tesla_wind / x[1] - x[0])) + 1)
    return 2 * int(round(0.5 * tesla_wind / np.abs(x[1] - x[0]))) + 1



colours = [#'#001219',
           '#005F73',
           '#0A9396',
           '#94D2BD',
           '#E9D8A6',
           '#EE9B00',
           '#CA6702',
           '#BB3E03',
           '#AE2012',
           '#9B2226']

fig, a = MakePlot(figsize=(16, 12), gs=True).create()
gs = fig.add_gridspec(2, 8)
ax1 = fig.add_subplot(gs[:, :3])
ax2 = fig.add_subplot(gs[0, 3:6])
ax3 = fig.add_subplot(gs[0, 6:])
ax4 = fig.add_subplot(gs[1, 3:6])
ax5 = fig.add_subplot(gs[1, 6:])



field_ranges1 = (28.75, 34.37, .71)
field_ranges2 = (22.2, 27, .71)

for i, filename in enumerate(filenames):

        dat = load_matrix(filename)
        field = dat[:, 0]
        tau = dat[:, 1]

        if field[0] > field[-1]:
            field = np.flip(field)
            tau = np.flip(tau)

        ax1.plot(field, tau, c=colours[i], linewidth=2, label=str(angles[i]))#+r'$\degree$')

        new_field, new_tau = interp_range(field, tau, field_ranges1[0], field_ranges1[1])

        fit_tau = savgol_filter(new_tau, tesla_window(new_field, field_ranges1[2]), 3)

        subtracted_tau = new_tau - fit_tau

        inv_x, inv_y = invert_x(new_field, subtracted_tau)
        fft_x, fft_y = qo_fft(inv_x, inv_y, freq_cut=10000)


        ax2.plot(new_field, subtracted_tau, linewidth=1.4, c=colours[i])
        ax3.plot(fft_x / 1e3, fft_y / np.max(fft_y) , linewidth=1.8, c=colours[i])


        new_field, new_tau = interp_range(field, tau, field_ranges2[0], field_ranges2[1])

        fit_tau = savgol_filter(new_tau, tesla_window(new_field, field_ranges2[2]), 3)

        subtracted_tau = new_tau - fit_tau

        inv_x, inv_y = invert_x(new_field, subtracted_tau)
        fft_x, fft_y = qo_fft(inv_x, inv_y, freq_cut=10000)


        ax4.plot(new_field, subtracted_tau, linewidth=1.4, c=colours[i])
        ax5.plot(fft_x / 1e3, fft_y / np.max(fft_y) , linewidth=1.8, c=colours[i])







ax1.set_xbound(-0.1,35)


# ax2.set_ybound(-0.0,10.5)
ax2.set_xbound(28.7, 34.4)
ax2.set_xticks([29,30, 31, 32, 33, 34])

ax3.set_ybound(0,1.1)
ax3.set_yticks([0,0.5,1])

ax4.set_xbound(22.1, 27.1)
ax4.set_xticks([23, 24, 27, 25, 26])

ax5.set_ybound(0,1.1)
ax5.set_yticks([0,0.5,1])



# handles, labels = ax1.get_legend_handles_labels()
#
# legend = ax1.legend(handles[::-1], labels[::-1], framealpha=0, ncol=1, loc='upper right',
#                     prop={'size': 18, 'family': 'arial'},
#                     handlelength=0, labelspacing=0.9)  # , bbox_to_anchor=(27.5,12.5), bbox_transform=ax1.transData)
#
# for line, text in zip(legend.get_lines(), legend.get_texts()):
#     text.set_color(line.get_color())

# ax3.set_xbound(-15, 90)
#
publication_plot(ax1, r'$\mu_0H$ (T)', r'$\Delta\tau$ (arb.)')
publication_plot(ax2, r'$\mu_0H$ (T)', r'$\Delta\tau$ (arb.)')
publication_plot(ax3, r'Frequency (kT)', r'FFT amplitude (arb.)')
publication_plot(ax4, r'$\mu_0H$ (T)', r'$\Delta\tau$ (arb.)')
publication_plot(ax5, r'Frequency (kT)', r'FFT amplitude (arb.)')



ax1.annotate(r'$T = 300$ mK ', xy=(0.05, 0.96), xycoords='axes fraction',
             ha="left", va="center", fontname='arial', fontsize=20)

plt.tight_layout(pad=1)

plt.show()

