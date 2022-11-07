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




path = '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S6sb/good/'



filenames_186 = ['0.035K_186deg_sweep048_up.csv',
            '0.035K_186deg_sweep047_up.csv']
filenames_180 = ['0.035K_180deg_sweep054_up.csv',
            '0.035K_180deg_sweep053_up.csv']
filenames_174 = ['0.035K_174deg_sweep061_up.csv',
            '0.035K_174deg_sweep060_down.csv',
            '0.035K_174deg_sweep059_down.csv',
            '0.035K_174deg_sweep062_up.csv']

filenames = [filenames_186, filenames_180, filenames_174]

angles = [-6, 0, 6]

def tesla_window(x, tesla_wind):
    # print(2 * int(round(0.5 * tesla_wind / x[1] - x[0])) + 1)
    return 2 * int(round(0.5 * tesla_wind / np.abs(x[1] - x[0]))) + 1

cs = ['#087F8C',
      '#F75C03',
      '#AD91A3',
      '#9E0031']

fig, a = MakePlot(figsize=(12, 6), gs=True).create()
gs = fig.add_gridspec(1, 4)
ax1 = fig.add_subplot(gs[0, :3])
ax2 = fig.add_subplot(gs[0, 3:4])

for i, filename_list in enumerate(filenames):

    angle = angles[i]
    c = cs[i]

    xs, inv_xs, ys, inv_ys = [], [], [], []

    for filename in filename_list:


        dat = load_matrix(path + filename)

        field = dat[:,0]
        tau = dat[:,1]

        new_field, new_tau = interp_range(field, tau, 25.1, 27.9)

        fit_tau = savgol_filter(new_tau, tesla_window(new_field, 0.7), 3)

        subtracted_tau = new_tau - fit_tau

        xs.append(new_field)
        ys.append(subtracted_tau)

        inv_x, inv_y = invert_x(new_field, subtracted_tau)

        inv_xs.append(inv_x)
        inv_ys.append(inv_y)

    avg_tau = np.mean(ys, axis=0)
    avg_inv_tau = np.mean(inv_ys, axis=0)

    avg_field = np.mean(xs, axis=0)
    avg_inv_field = np.mean(inv_xs, axis=0)

    fft_x, fft_y = qo_fft(avg_inv_field, avg_inv_tau, freq_cut=10000)

    ax1.plot(avg_field,1e6*avg_tau + i*0.5,lw=1.5,c=cs[i])
    ax2.plot(fft_x,1e4*fft_y,lw=1.2,c=cs[i], label=str(angle) + r'$\degree$')


publication_plot(ax1, r'$\mu_0H$ (T)',r'$\Delta\tau$ (arb.)',label_fontsize=20,tick_fontsize=17)

    # ax2.set_xticks([0,6,12])
publication_plot(ax2, r'Frequency (kT)', r'FFT amplitude (arb.)',label_fontsize=20,tick_fontsize=17)

legend = ax2.legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 20, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

plt.tight_layout(pad=.5)
plt.show()




