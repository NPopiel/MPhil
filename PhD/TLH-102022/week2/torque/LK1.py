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
gs = fig.add_gridspec(4, 4)

ax1 = fig.add_subplot(gs[:2, :4])
ax2 = fig.add_subplot(gs[2:, :2])
ax3 = fig.add_subplot(gs[2:, 2:])
# ax4 = fig.add_subplot(gs[1, 5:])


temperatures = [0.46]  # These are definitely wrong and mass is much heavier as a consequence
freq1 = np.array([531.3965, 434.8956, 327.3881, 216.5465, 147.95, 111.4137])
freq2 = np.array([1357.081, 955.45, 635.6688, 338.9361, 217.54, 88.029])

colours = ['#264653', '#2A9D8F', '#BABB74', '#E9C46A', '#EE8959', '#E76F51']

xs, inv_xs, ys, inv_ys = [], [], [], []

dominant_freqs = []

fft_xs, fft_ys = [], []

# norm_const = np.max(fft_ys) # use tis at first, once you've LKd rescale everything
# norm_const = 1414.3502003343096
norm_const = 1 #PRINT THIS OUT

for i, filename in enumerate(filenames):
    dat = load_matrix(path + filename)

    field = dat[:, 0]
    tau = dat[:, 1]

    new_field, new_tau = interp_range(field, tau, field_ranges[0], field_ranges[1])

    fit_tau = savgol_filter(new_tau, tesla_window(new_field, field_ranges[2]), 3)

    subtracted_tau = new_tau - fit_tau

    xs.append(new_field)
    ys.append(subtracted_tau)

    inv_x, inv_y = invert_x(new_field, subtracted_tau)

    inv_xs.append(inv_x)
    inv_ys.append(inv_y)

    fft_x, fft_y = qo_fft(inv_x, inv_y, freq_cut=10000)

    # save_dat = np.array([new_field,subtracted_tau]).T
    #
    # np.savetxt('/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/data/LK/'+str(temperatures[i])+'.csv',
    #            save_dat, delimiter=',')

    ax1.plot(new_field, 1e7 * subtracted_tau - i * 4, lw=1, c=colours[i], label=' ' + str(temperatures[i]) + r' mK')

    fft_xs.append(fft_x)
    fft_ys.append(fft_y)



temperatures_K = np.squeeze(np.array(temperatures))

popt1, pcov1 = scipy.optimize.curve_fit(lifshitz_kosevich, temperatures_K, freq1, p0=(60, 2.3515721812642054e-05))
popt2, pcov2 = scipy.optimize.curve_fit(lifshitz_kosevich, temperatures_K, freq2, p0=(60, 2.3515721812642054e-05))

perr1 = np.sqrt(np.diag(pcov1))
perr2 = np.sqrt(np.diag(pcov2))

oversampled_temps = np.linspace(0.0001, 15, 10000)

lk1 = lifshitz_kosevich(oversampled_temps, *popt1)
lk2 = lifshitz_kosevich(oversampled_temps, *popt2)

print('m1: ', popt1[0], '+/- ', perr1[0])
print('m2: ', popt2[0], '+/- ', perr2[0])

print('Amp1: ', popt1[1], '+/- ', perr1[1])
print('amp2: ', popt2[1], '+/- ', perr2[1])

ax3.plot(oversampled_temps, lk1 / norm_const, c='darkslategray', label=r'$F_1$', linestyle='dashed',zorder=-1)
ax3.plot(oversampled_temps, lk2 / norm_const, c='darkslategray', label=r'$F_2$',zorder=-1)

for i in range(len(temperatures)):
    fft_x = fft_xs[i]
    fft_y = fft_ys[i]

    ax2.plot(fft_x / 1e3, fft_y / norm_const * 1e6, lw=2.4, c=colours[i])
    ax3.scatter(temperatures[i], freq1[i] / norm_const, s=250, marker='o', facecolors='none', edgecolors=colours[i],zorder=1)
    ax3.scatter(temperatures[i], freq2[i] / norm_const, s=250, marker='d', c=colours[i],zorder=1)




ax3.set_ybound(0, 1.05)
ax3.set_xbound(0, 200)

# ax3.annotate(r'$m^*_1 = 3(2)$ $m_e$', xy=(0.26, 0.61), xycoords='axes fraction',
#              ha="left", va="center", fontname='arial', fontsize=18)
#
# ax3.annotate(r'$m^*_2 = 3(3)$ $m_e$', xy=(30 / 400, 0.08), xycoords='axes fraction',
#              ha="left", va="center", fontname='arial', fontsize=18)

publication_plot(ax3, 'Temperature (K)', '')
ax3.set_yticklabels([])

ax1.set_xbound(26, 28.25)
ax1.set_xticks([26,27,28])

ax2.set_ybound(0, 1.05)
ax2.set_xbound(0, 10)

# ax2.annotate(r'$F_{2} \approx 4.4$ kT', xy=(0.25, 0.4), xycoords='axes fraction',
#              ha="center", va="center", fontname='arial', fontsize=24)
#
# ax2.annotate(r'$F_{1} \approx 5.4$ kT', xy=(0.7, 0.85), xycoords='axes fraction',
#              ha="center", va="center", fontname='arial', fontsize=24)

# ax8.scatter(angles, dominant_freqs)
publication_plot(ax2, r'Frequency (kT)', r'FFT amplitude (arb.)')
publication_plot(ax1, r'$\mu_0H$ (T)', r'$\Delta\tau$ (arb.)')

# annotate the FFT frequencys

ax1.annotate(r'a', xy=(0.03, 0.925), xycoords='axes fraction',
             ha="left", va="center", fontname='arial', fontweight='bold',fontsize=28)

ax2.annotate(r'b', xy=(0.03*(1.5), 0.925), xycoords='axes fraction',
             ha="left", va="center", fontname='arial', fontweight='bold',fontsize=28)

ax3.annotate(r'c', xy=(1-(0.03*1.5), 0.925), xycoords='axes fraction',
             ha="right", va="center", fontname='arial', fontweight='bold',fontsize=28)

# handles, labels = ax1.get_legend_handles_labels()
#
# legend = ax1.legend(handles, labels, framealpha=0, ncol=1, loc='upper right',
#                     prop={'size': 22, 'family': 'arial'},
#                     handlelength=0, labelspacing=1.75)  # , bbox_to_anchor=(27.5,12.5), bbox_transform=ax1.transData)
#
# k = 0
# spacing = 0.135
# for line, text in zip(legend.get_lines(), legend.get_texts()):
#     ax1.annotate(labels[k], xy=(0.93, 0.8-k*spacing), xycoords='axes fraction',
#                  ha="center", va="center", fontname='arial', fontsize=24, color=line.get_color())
#
#     k+=1
#
# legend.remove()


ax3.set_xticks([0,50,100,150,200])
ax2.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0])

plt.tight_layout(pad=0.99)
plt.savefig('/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/new_figs/'+'LK-v4b.pdf', dpi=300, bbox_inches='tight')
# plt.show()



