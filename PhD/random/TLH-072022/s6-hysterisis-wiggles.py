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
import matplotlib.colors as mcolors


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




path = '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S6sb/good/actual_good/'



filenames_186 = ['0.035K_186deg_sweep047_up.csv',
                 '0.035K_186deg_sweep048_up.csv'
            ]
filenames_180 = ['0.035K_180deg_sweep054_up.csv',
            '0.035K_180deg_sweep053_up.csv']
filenames_174 = ['0.035K_174deg_sweep061_up.csv',
            '0.035K_174deg_sweep060_down.csv',
            '0.035K_174deg_sweep059_down.csv']

filenames = [filenames_186]

angles = [-6, 0, 6]

def tesla_window(x, tesla_wind):
    # print(2 * int(round(0.5 * tesla_wind / x[1] - x[0])) + 1)
    return 2 * int(round(0.5 * tesla_wind / np.abs(x[1] - x[0]))) + 1

cs = ['#d1495b',
      '#00798c']

labels = ['Up Sweep',
          'Down Sweep']

fig, a = MakePlot(figsize=(12, 6), gs=True).create()
gs = fig.add_gridspec(1, 5)
ax1 = fig.add_subplot(gs[0, :3])
ax2 = fig.add_subplot(gs[0, 3:5])

fft_xs, fft_ys = [], []

for i, filename in enumerate(filenames_186):


        dat = load_matrix(path + filename)

        field = dat[:,0]
        tau = dat[:,1]

        new_field, new_tau = interp_range(field, tau, 25.1, 27.9)

        fit_tau = savgol_filter(new_tau, tesla_window(new_field, 0.7), 3)

        subtracted_tau = new_tau - fit_tau

        inv_x, inv_y = invert_x(new_field, subtracted_tau)


        fft_x, fft_y = qo_fft(inv_x, inv_y, freq_cut=12000)

        ax1.plot(new_field, 1e7*subtracted_tau,lw=1.5,c=cs[i], label=labels[i])

        fft_xs.append(fft_x)
        fft_ys.append(fft_y)

avg_color = np.average([mcolors.hex2color('#d1495b'), mcolors.hex2color('#00798c')], axis=0)

print('c1 ,', mcolors.hex2color('#d1495b'))
print('c2 ,', mcolors.hex2color('#00798c'))

print('avg, ', avg_color)

ax2.plot(np.average(fft_xs, axis=0)/1e3,np.average(fft_ys, axis=0)/np.max(fft_ys),lw=1.6,c=avg_color, label=labels[i])

ax2.set_ybound(0,1.05)
ax2.set_xbound(0,12.505)

ax1.set_yticks([-4, -2, 0, 2, 4])
ax1.set_xticks([25, 26, 27, 28])

ax2.set_xticks([0, 5, 10])



publication_plot(ax1, r'$\mu_0H$ (T)',r'$\Delta\tau$ (arb.)')

    # ax2.set_xticks([0,6,12])
publication_plot(ax2, r'Frequency (kT)', r'FFT amplitude (arb.)')

ax1.annotate(r'$T$ = 50 mK', xy=(0.85, 0.08), xycoords='axes fraction',
    ha="center", va="center", fontname='arial', fontsize=20)
# ax1.annotate(r'$\theta = 6\degree \angle$ $c$', xy=(0.85, 0.03), xycoords='axes fraction',
#     ha="center", va="center", fontname='arial', fontsize=20)

ax2.annotate(r'$F \approx$ 3 kT', xy=(0.75, 0.90), xycoords='axes fraction',
    ha="center", va="center", fontname='arial', fontsize=20)

legend = ax1.legend(framealpha=0, ncol=2, loc='best',
              prop={'size': 20, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

plt.tight_layout(pad=.5)
plt.show()

# plt.savefig(path+'186deg_without_angle.png', dpi=300)



