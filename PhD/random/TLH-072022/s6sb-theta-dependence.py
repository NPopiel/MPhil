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




path = '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S6sb/properly_good/angles/'




filenames = ['0.035K_186deg_sweep048_up.csv',
             '0.035K_180deg_sweep054_up.csv',
             '0.035K_174deg_sweep062_up.csv',
             '0.035K_168deg_sweep071_down.csv',
             '0.035K_165deg_sweep078_down.csv',
             '0.035K_159deg_sweep087_down.csv',
             '0.035K_153deg_sweep097_up.csv',
             '0.035K_147deg_sweep103_up.csv']


angles = [-6, 0, 6, 12, 15, 21, 27, 33]

field_ranges = [(25.1, 27.9, 0.71),
                (27.158, 27.84, 0.5),
                (26.958, 27.85, 0.5),
                (25.986, 26.48, 0.3),
                (27.3, 27.5981, 0.2),
                (26.06, 27.877, 0.71),
                (27.2, 27.89, 0.3)]



def tesla_window(x, tesla_wind):
    # print(2 * int(round(0.5 * tesla_wind / x[1] - x[0])) + 1)
    return 2 * int(round(0.5 * tesla_wind / np.abs(x[1] - x[0]))) + 1

fig, a = MakePlot(figsize=(16, 14), gs=True).create()
gs = fig.add_gridspec(3, 6)
ax1 = fig.add_subplot(gs[:2, :4])
# ax2 = fig.add_subplot(gs[:2, 4])
ax3 = fig.add_subplot(gs[:2, 4:])
ax4 = fig.add_subplot(gs[2, :2])
ax5 = fig.add_subplot(gs[2, 2])
ax6 = fig.add_subplot(gs[2, 3:5])
ax7 = fig.add_subplot(gs[2, 5])



xs, inv_xs, ys, inv_ys = [], [], [], []

frequencys = [(3.2, 3.7,6.274),
              (2.96, 3.8, 6.25),
              (2.5, 3.7, 6.7),
              (3.5, None,6.9),
              (None, None, 8.5),
              (None,None,9.9),
              (None, 4.5, 13.2),
              (None, 5.7, 15.6)]

markers = ['o', '+', 'v']

# colours = ['#56CEBE',
#            '#60ADBF',
#            '#6A8DC0',
#            '#746CC1',
#            '#7E4BC2',
#            '#882BC3',
#            '#920AC4']

colours = ['#832388',
           '#912884',
           '#9E2C80',
           '#AC317C',
           '#BA3577',
           '#C83A73',
           '#D53E6F',
           '#E3436B']

for i, filename in enumerate(filenames):


        dat = load_matrix(path + filename)

        field = dat[:,0]
        tau = dat[:,1]

        if field[0] > field[-1]:
            field = np.flip(field)
            tau = np.flip(tau)

        if tau[0] > tau[-1]:
            tau *= -1

        tau -= tau[0]

        ax1.plot(field, tau*1e4, lw=3, c=colours[i],
                 label=' ' + str(angles[i]) + r'$\degree$')

        magnitude_torque = np.abs(np.max(tau) - np.min(tau))

        # ax2.scatter(angles[i],magnitude_torque, s=120, c=colours[i])

        freqs = frequencys[i]

        for j, f in enumerate(freqs):

            if f is not None:

                ax3.scatter(angles[i], f, marker=markers[j], c=colours[i], s=200)



# plot the lower frequency example

dat = load_matrix(path+'0.035K_174deg_sweep062_up.csv')
field = dat[:, 0]
tau = dat[:, 1]

new_field, new_tau = interp_range(field, tau, 26.9, 27.9)

fit_tau = savgol_filter(new_tau, tesla_window(new_field, 0.5), 3)

subtracted_tau = new_tau - fit_tau

inv_x, inv_y = invert_x(new_field, subtracted_tau)
fft_x, fft_y = qo_fft(inv_x, inv_y, freq_cut=12000)

ax4.plot(inv_x, inv_y*1e7, linewidth=1.4, c='#A32E7E')
ax5.plot(fft_x/1e3, fft_y/np.max(fft_y),linewidth=1.8, c='#A32E7E')


# plot the lower frequency example

dat = load_matrix(path+'0.035K_153deg_sweep097_up.csv')
field = dat[:, 0]
tau = dat[:, 1]

new_field, new_tau = interp_range(field, tau, 26.9, 27.9)

fit_tau = savgol_filter(new_tau, tesla_window(new_field, 0.3), 3)

subtracted_tau = new_tau - fit_tau

inv_x, inv_y = invert_x(new_field, subtracted_tau)
fft_x, fft_y = qo_fft(inv_x, inv_y, freq_cut=20000)

ax6.plot(inv_x, inv_y*1e7, linewidth=1.4, c='#D33E70',)
ax7.plot(fft_x/1e3, fft_y/np.max(fft_y),linewidth=1.8, c='#D33E70',)

ax3.scatter(33, 18.12, marker='x', c=colours[-1], s=200)
publication_plot(ax1, r'$\mu_0H$ (T)', r'$\tau$ (arb.)')
# publication_plot(ax2, r'$\theta$ ($\degree$)', r'$|\tau|$ (arb.)')
publication_plot(ax3, r'$\theta$ ($\degree$)', r'Frequency (kT)')
publication_plot(ax4, r'$(\mu_0H)^{-1}$ (T$^{-1}$)', r'$\Delta\tau$ (arb.)')
publication_plot(ax5, r'Frequency (kT)', r'FFT amplitude (arb.)')
publication_plot(ax6, r'$(\mu_0H)^{-1}$ (T$^{-1}$)', r'$\Delta\tau$ (arb.)')
publication_plot(ax7, r'Frequency (kT)', r'FFT amplitude (arb.)')


# fft_ax.set_xbound(0,25.05)

plt.tight_layout(pad=2)
# plt.savefig(path+'angles-draft1.png', dpi=300)
plt.show()




