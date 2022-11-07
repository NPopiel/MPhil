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

torque_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2022_10_TLH/week2/torque/VT16/day3/0.3K_74deg_sweep135_down.csv'

fig, a = MakePlot(figsize=(14, 8), gs=True).create()
gs = fig.add_gridspec(3, 8, hspace=0, wspace=2.7)
ax1 = fig.add_subplot(gs[:2,:3])
ax_deriv = fig.add_subplot(gs[2,:3])
gs = fig.add_gridspec(2, 8, wspace=2.7)
ax2 = fig.add_subplot(gs[0,3:6])
ax3 = fig.add_subplot(gs[1,3:6])
ax4 = fig.add_subplot(gs[0, 6:])
ax5 = fig.add_subplot(gs[1, 6:])



torque_dat = load_matrix(torque_path)
field = torque_dat[:, 0]
tau = torque_dat[:, 1]

tau *= 74

if field[0] > field[-1]:
    field = np.flip(field)
    tau = np.flip(tau)

tau -= tau[0]

new_field, new_tau = interp_range(field, tau, 17.34, 38.9)

fit_tau = savgol_filter(new_tau, tesla_window(new_field, 2), 3)

subtracted_tau = new_tau - fit_tau

ax1.plot(field, tau, linewidth=2, c='#592E83', label=str(14)+r'$\degree$')

d2_tau = savgol_filter(tau, 101, 2, deriv=2)

ax_deriv.plot(field, d2_tau/1e-6, linewidth=1, c='#592E83', label=str(14)+r'$\degree$')

l1 = new_field > 28.4
l2 = new_field < 24.5

locs = l1 | l2

ax2.plot(new_field[l1], subtracted_tau[l1]*1e4, c='#113537', linewidth=1.2)
ax2.plot(new_field[l2], subtracted_tau[l2]*1e4, c='#113537', linewidth=1.2)

dat_fft = np.loadtxt('/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2022_10_TLH/week2/torque/VT16/day3/mega/mega0020.3K_74deg_sweep133_down.csv', delimiter=',')

fft_x = dat_fft[:,0]
fft_y = dat_fft[:,1]

ax4.plot(fft_x/1e3, fft_y / np.max(fft_y), linewidth=1.2, c='#113537')

new_field, new_tau = interp_range(field, tau, 36.0572, 38.7298)

fit_tau = savgol_filter(new_tau, tesla_window(new_field, .5), 3)

subtracted_tau = new_tau - fit_tau
ax3.plot(new_field, subtracted_tau*1e4, c='#FF47DA', linewidth=1.2)

dat_fft = np.loadtxt('/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2022_10_TLH/week2/torque/VT16/day3/mega/mega0040.3K_74deg_sweep135_down.csv', skiprows=2,delimiter=',')

fft_x = dat_fft[:,0]
fft_y = dat_fft[:,1]

ax5.plot(fft_x/1e3, fft_y / np.max(fft_y), linewidth=1.2, c='#FF47DA')



import matplotlib as mpl

# read image file
with mpl.cbook.get_sample_data('/Users/npopiel/Desktop/78deg-removebg-preview.png') as file:
    arr_image = plt.imread(file, format='png')

# Draw image
axin = ax1.inset_axes([0.1, 0.55, 0.32, 0.32])  # create new inset axes in data coordinates
axin.imshow(arr_image)
axin.axis('off')

axin.annotate(r'$\phi \approx 75 \degree$', xy=(0.32, 0.4), xycoords='axes fraction',
              ha="left", va="center", fontname='arial', fontsize=20)

axin.annotate(r'$b$', xy=(0, 1.2), xycoords='axes fraction',
              ha="center", va="center", fontname='arial', fontsize=20)

axin.annotate(r'$c$', xy=(1.12, 0), xycoords='axes fraction',
              ha="center", va="center", fontname='arial', fontsize=20)


ax2.annotate(r'$T = 350$ mK', xy=(0.15, 0.8), xycoords='axes fraction',
              ha="left", va="center", fontname='arial', fontsize=22)

ax4.annotate(r'$30 < \mu_0H < 38.8$ T', xy=(0.11, 0.88), xycoords='axes fraction',
              ha="left", va="center", fontname='arial', fontsize=13.5)

ax5.annotate(r'$36 < \mu_0H < 38.8$ T', xy=(0.11, 0.88), xycoords='axes fraction',
              ha="left", va="center", fontname='arial', fontsize=13.5)


publication_plot(ax1, r'$\mu_0H$ (T)', r'$\tau$ (pF)')

publication_plot(ax_deriv, r'$\mu_0H$ (T)', r'$\frac{\partial^2 \tau}{\partial(\mu_0H)^2}$ (pF)')

publication_plot(ax2, r'', r'$\Delta \tau$ ($\times 10^{-4}$ pF)')
publication_plot(ax3, r'$\mu_0H$ (T)', r'$\Delta \tau$ ($\times 10^{-4}$ pF)')
publication_plot(ax4, r'', r'FFT Amplitude (arb.)')
publication_plot(ax5, r'Frequency (kT)', r'FFT Amplitude (arb.)')

ax1.set_xbound(0,42)
ax1.set_xticklabels([])
ax1.set_ybound(0,2.5)
ax_deriv.set_xbound(0,42)
ax_deriv.set_ybound(-2.5,2.5)

ax_deriv.set_yticks([-2.5, 0, 2.5])
ax_deriv.set_yticklabels([-2.5, 0, ''])


ax3.set_ybound(-6.75, 6.75)
ax3.set_yticks([-6,-3,0,3,6])
ax3.set_xbound(36, 38.9)
ax3.set_xticks([36,37,38])

ax4.set_ybound(0,1.05)
ax4.set_yticks([0,0.5, 1])
ax4.set_xbound(0, 8)
ax4.set_xticks([0,2,4,6,8])

ax5.set_ybound(0,1.05)
ax5.set_yticks([0,0.5, 1])

ax5.set_xbound(0,20)
ax5.set_xticks([0,5,10,15,20])


plt.tight_layout(pad=3)
fig.align_labels()


plt.savefig('/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2022_10_TLH/week2/draft1-wiggles.png',
            dpi=300, bbox_inches = 'tight')

# plt.show()


