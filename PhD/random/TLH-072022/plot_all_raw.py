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


from glob import glob

path = '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S6sb/'

files = glob(path + "*.csv")




angles = [-6, 0, 6, 12, 15, 21, 27]

field_ranges = [(25.1, 27.9, 0.71),
                (27.158, 27.84, 0.15),
                (26.94, 27.85, 0.33),
                (25.986, 26.48, 0.15),
                (27.29, 27.642, 0.15),
                (26.06, 27.877, 0.71),
                (26.5764, 27.89, 0.71)]



def tesla_window(x, tesla_wind):
    # print(2 * int(round(0.5 * tesla_wind / x[1] - x[0])) + 1)
    return 2 * int(round(0.5 * tesla_wind / np.abs(x[1] - x[0]))) + 1

cs = ['#087F8C',
      '#F75C03',
      '#AD91A3',
      '#9E0031']

fig, ax1 = MakePlot(figsize=(8, 8), gs=False).create()


xs, inv_xs, ys, inv_ys = [], [], [], []

angle_lst = []

for i, filename in enumerate(files):

        angle = filename.split('_')[-3].split('d')[0]

        dat = load_matrix(filename)

        field = dat[:,0]
        tau = dat[:,1]

        if angle_lst.count(angle) < 1:

            ax1.plot(field, tau , lw=1.3, c=plt.cm.jet(i/len(files)), label=str(angle) + r'$\degree$')

            angle_lst.append(str(angle))



ax1.set_xbound(24.9,28.05)
#
# ax1.annotate(r'$\theta = 21 \degree \angle$ $c$', xy=(0.2, 0.9), xycoords='axes fraction',
#     ha="center", va="center", fontname='arial', fontsize=20)
# ax1.annotate(r'$T$ = 50 mK', xy=(0.85, 0.05), xycoords='axes fraction',
#     ha="center", va="center", fontname='arial', fontsize=20)
#
# ax2.annotate(r'$25.9$ T $< \mu_0H < 27.2$ T', xy=(0.7, 0.9), xycoords='axes fraction',
#     ha="center", va="center", fontname='arial', fontsize=14)
#
# ax2.annotate(r'$F \approx 10$ kT', xy=(0.7, 0.8), xycoords='axes fraction',
#     ha="center", va="center", fontname='arial', fontsize=14)

publication_plot(ax1, r'$\mu_0H$ (T)',r'$\tau$ (arb.)',label_fontsize=20,tick_fontsize=17)

legend = ax1.legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 10, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

plt.tight_layout(pad=.5)
# plt.savefig(path+'21deg-10kT.pdf', dpi=300)
plt.show()




