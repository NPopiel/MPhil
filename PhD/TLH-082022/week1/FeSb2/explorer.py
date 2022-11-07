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


path = '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/temps/'



filenames = ['0.035K_174deg_sweep261_down.csv',
             '0.06K_174deg_sweep264_down.csv',
             '0.1K_174deg_sweep266_up.csv',
             '0.15K_174deg_sweep268_up.csv',
             '0.21K_174deg_sweep270_up.csv',
             '0.4K_174deg_sweep272_down.csv']

angle = 6

field_ranges = (26.05, 27.95, 0.31)


def tesla_window(x, tesla_wind):
    # print(2 * int(round(0.5 * tesla_wind / x[1] - x[0])) + 1)
    return 2 * int(round(0.5 * tesla_wind / np.abs(x[1] - x[0]))) + 1


fig, a = MakePlot(figsize=(12, 12), gs=True).create()
gs = fig.add_gridspec(4, 4)

ax1 = fig.add_subplot(gs[:2, :4])
ax2 = fig.add_subplot(gs[2:, :2])
ax3 = fig.add_subplot(gs[2:, 2:])
# ax4 = fig.add_subplot(gs[1, 5:])


temperatures = [35, 60, 100, 150, 210, 400]  # These are definitely wrong and mass is much heavier as a consequence
freq1 = np.array([531.3965, 434.8956, 327.3881, 216.5465, 147.95, 111.4137])
freq2 = np.array([1357.081, 955.45, 635.6688, 338.9361, 217.54, 88.029])

colours = ['#577590', '#43aa8b', '#90be6d', '#f9c74f', '#f8961e', '#f94144']

xs, inv_xs, ys, inv_ys = [], [], [], []

dominant_freqs = []

fft_xs, fft_ys = [], []

# norm_const = np.max(fft_ys) # use tis at first, once you've LKd rescale everything
norm_const = 1414.3502003343096

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

    ax1.plot(new_field, 1e7 * subtracted_tau + i * 2.8, lw=1, c=colours[i], label=' ' + str(temperatures[i]) + r' mK')

    fft_xs.append(fft_x)
    fft_ys.append(fft_y)

    ax3.scatter(temperatures[i], freq1[i] / norm_const, s=175, marker='o', facecolors='none', edgecolors=colours[i])
    ax3.scatter(temperatures[i], freq2[i] / norm_const, s=175, marker='d', c=colours[i])

for i in range(len(temperatures)):
    fft_x = fft_xs[i]
    fft_y = fft_ys[i]

    ax2.plot(fft_x / 1e3, fft_y / norm_const * 1e6, lw=2.4, c=colours[i])


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


temperatures_K = np.squeeze(np.array(temperatures)) / 1e3

popt1, pcov1 = scipy.optimize.curve_fit(lifshitz_kosevich, temperatures_K, freq1, p0=(60, 2.3515721812642054e-05))
popt2, pcov2 = scipy.optimize.curve_fit(lifshitz_kosevich, temperatures_K, freq2, p0=(60, 2.3515721812642054e-05))

perr1 = np.sqrt(np.diag(pcov1))
perr2 = np.sqrt(np.diag(pcov2))

oversampled_temps = np.linspace(0.0001, temperatures_K[-1] + 0.01, 10000)

lk1 = lifshitz_kosevich(oversampled_temps, *popt1)
lk2 = lifshitz_kosevich(oversampled_temps, *popt2)

print('m1: ', popt1[0], '+/- ', perr1[0])
print('m2: ', popt2[0], '+/- ', perr2[0])

print('Amp1: ', popt1[1], '+/- ', perr1[1])
print('amp2: ', popt2[1], '+/- ', perr2[1])

ax3.plot(oversampled_temps * 1e3, lk1 / norm_const, c='darkslategray', label=r'$F_1$')
ax3.plot(oversampled_temps * 1e3, lk2 / norm_const, c='darkslategray', linestyle='dashed', label=r'$F_2$')

ax3.set_ybound(0, 1.05)
ax3.set_xbound(0, 400.5)

ax3.annotate(r'$m \approx 42(4)$ $m_e$', xy=(65 / 400, 0.91), xycoords='axes fraction',
             ha="left", va="center", fontname='arial', fontsize=18)

ax3.annotate(r'$m \approx 29(4)$ $m_e$', xy=(30 / 400, 0.08), xycoords='axes fraction',
             ha="left", va="center", fontname='arial', fontsize=18)

publication_plot(ax3, 'Temperature (mK)', '')
ax3.set_yticklabels([])

ax_raw = ax3.inset_axes([175, 0.4, 200, .4], transform=ax3.transData)

for i, filename in enumerate(filenames):
    dat = load_matrix(path + filename)

    field = dat[:, 0]
    tau = dat[:, 1]

    ax_raw.plot(field, tau * 1e3 + i * .8, lw=1, c=colours[i],  # i * 2.8
                label=' ' + str(temperatures[i]) + r' mK')

publication_plot(ax_raw, r'$\mu_0H$ (T)', r'$\tau$ (arb.)', label_fontsize=14, tick_fontsize=12)

ax1.set_xbound(26, 28.25)
import matplotlib as mpl

# read image file
with mpl.cbook.get_sample_data(ute2_crystal_path) as file:
    arr_image = plt.imread(file, format='png')

# Draw image
axin = ax2.inset_axes([-2, 0.12, 6.5, .85], transform=ax2.transData)  # create new inset axes in data coordinates
axin.imshow(arr_image)
axin.axis('off')

ax2.set_ybound(0, 1.05)
ax2.set_xbound(0, 10.05)

ax2.annotate(r'$F_1 \approx 4.4$ kT', xy=(0.8, 0.89), xycoords='axes fraction',
             ha="center", va="center", fontname='arial', fontsize=24)

ax2.annotate(r'$F_2 \approx 5.4$ kT', xy=(0.8, 0.71), xycoords='axes fraction',
             ha="center", va="center", fontname='arial', fontsize=24)

# ax8.scatter(angles, dominant_freqs)
publication_plot(ax2, r'Frequency (kT)', r'FFT amplitude (arb.)')
publication_plot(ax1, r'$\mu_0H$ (T)', r'$\Delta\tau$ (arb.)')

# annotate the FFT frequencys

handles, labels = ax1.get_legend_handles_labels()

legend = ax1.legend(handles[::-1], labels[::-1], framealpha=0, ncol=1, loc='upper right',
                    prop={'size': 22, 'family': 'arial'},
                    handlelength=0)  # , bbox_to_anchor=(27.5,12.5), bbox_transform=ax1.transData)

for line, text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

plt.tight_layout(pad=0.99)
# plt.savefig(path+'LK-draft-og_temps.png', dpi=300)
plt.show()

'''

code for different top axis


fig = plt.figure(figsize=(32,24))
ax1 = fig.add_subplot(111)
topax = ax1.twiny()
fig.subplots_adjust(top=0.85)

topax.set_xlim([max_field,min_field])
topax.set_xlabel('Magnetic Field (T)',fontsize=ax_lab_size , fontname='arial',labelpad=10)
plt.setp(topax.get_xticklabels(),fontsize=ax_lab_size , fontname='arial')


count = 0
for temp, loess_av in loess_avgs.items():

  ax1.plot(loess_av.invert.x, (loess_av.invert.y + count*waterfall_const)/1e-8, c=plt.cm.magma(count/len(loess_avgs)), label = temp)
  count+=1
ax1.set_xlabel(xlabel1, fontname='arial',fontsize=ax_lab_size)
ax1.set_ylabel(ylabel1, fontname='arial',fontsize=ax_lab_size)


# ax1.minorticks_on()
# ax1.tick_params('both', which='both', direction='in',
#         bottom=True, top=True, left=True, right=True)

    #ax2.set_xlim(left=0)
    #ax2.set_ylim(bottom=0)

#ax1.ticklabel_format(axis='y',style="sci",useMathText=True,scilimits=(0,0))

handles, labels = ax1.get_legend_handles_labels()
legend = ax1.legend(handles[::-1], labels[::-1], framealpha=0, ncol=1, # len(dset)//12+
         title='Temperature',prop={"size":36}, bbox_to_anchor=(1.05, 1))

plt.setp(ax1.get_xticklabels(),fontsize=ax_lab_size , fontname='arial')
plt.setp(ax1.get_yticklabels(),fontsize=ax_lab_size, fontname='arial' )
plt.setp(legend.get_title(),fontsize=ax_lab_size, fontname='arial')

for l in legend.get_lines():
  l.set_linewidth(6)


ax1.minorticks_on()
ax1.tick_params('both', which='major', direction='in',
      bottom=True, top=False, left=True, right=True,length=12, width=4)
ax1.tick_params('both', which='minor', direction='in',
      bottom=True, top=False, left=True, right=True,length=8, width=4)
plt.suptitle('dHvA Oscillations in FeGa$_3$', fontname='arial',fontsize=title_size)
topax.tick_params(top=False, labeltop=True, left=False, labelleft=False, right=False, labelright=False, bottom=False, labelbottom=False)


fig.savefig('waterfall_wiggles1.png', dpi=400)
'''



