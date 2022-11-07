import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelmin, argrelmax
from scipy.ndimage import median_filter
from tools.utils import *
from tools.ColorMaps import select_discrete_cmap
from tools.utils import *
from tools.ColorMaps import *
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import seaborn as sns
from scipy.interpolate import interp1d



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


def extract_moving_indices(field, window_size = 1.5, n=.1):

    min_field = np.min(field)
    max_field = np.max(field)

    inds_list = []

    last_field = min_field

    end_field = np.round(window_size + last_field,1)
    while end_field <= max_field:
        end_field = np.round(window_size + last_field,1)

        inds1 = field > last_field
        inds2 = field < end_field

        inds = inds1 & inds2

        inds_list.append(inds)

        last_field += n

    return inds_list

def power_law(x,a,b,c):
    return a * np.power(x,b) + c


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




def fit_poly_as_function(inds_list, field, volts, plot=False):
    lst = []
    eps = 5e-7
    popt=(1,1,1)

    for ind_set in inds_list:

        v = volts[ind_set]
        B = field[ind_set]

        # a0, b0, c0 = popt
        a0, b0, c0 = (.1,2,.1)
        a0 += eps
        b0+=.001
        c0+=eps

        (a,b,c) = np.polyfit(B, v, deg=2)



        # print(np.mean(B))
        if plot:
            fig, ax = MakePlot().create()

            ax.plot(B,v,lw=2, c=select_discrete_cmap('venasaur')[0])
            ax.plot(B, a*B**2 + b*B +c, lw=2, c=select_discrete_cmap('venasaur')[6], linestyle='dashed')
            print(b)
            plt.show()

        lst.append([np.mean(B), a, b, c])

    return np.array(lst)


def tesla_window(x, tesla_wind):
    # print(2 * int(round(0.5 * tesla_wind / x[1] - x[0])) + 1)
    return 2 * int(round(0.5 * tesla_wind / np.abs(x[1] - x[0]))) + 1

files = ['/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.3K_90deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_82.5deg_sweep3.csv',
# '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_82.5deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_79deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_75.5deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_72deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_67.5deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_64deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_60deg_sweep1.csv',

]

# 60 32.3 - 34
#64 33.3 - 35
# 67.5 35.3 - 37
# 72 - 39 - 40.2
# 75.5 41.1 - end
angles = [90, 82.5, 79, 75.5, 72, 67.5, 64, 60]

fit_ranges = [(32.3, 34),
              (33.3, 35),
              (35.3, 37)]
              # (37.1, 38.10)]
fit_ranges.reverse()
small_plot_fields = [32.5, 33.5, 35.3]

fig, a = MakePlot(figsize=(7,9),gs=True).create()

gs = fig.add_gridspec(4,1,hspace=0)
ax1 = fig.add_subplot(gs[:3, 0])
ax2 = fig.add_subplot(gs[3:4, 0])

fig, a = MakePlot(figsize=(7,6),gs=True).create()

gs = fig.add_gridspec(2,4,hspace=0)
ax3 = fig.add_subplot(gs[0, :2])
ax4 = fig.add_subplot(gs[1, :2])
ax5 = fig.add_subplot(gs[:, 2:])




# ax3 = fig.add_subplot(gs[4:, 0])
# gs = fig.add_gridspec(4,2)
# ax3 = fig.add_subplot(gs[:2, 1])
# ax4 = fig.add_subplot(gs[2:4, 1])


inds = [0,0,0,0,0,0,1,2]
# #
B_thresh = 10
ax = ax1
for i, f1 in enumerate(files):


    print(f1)

    field = np.genfromtxt(f1, delimiter=',')[5:, 0]
    volts = 1e3*np.abs(medfilt(np.genfromtxt(f1, delimiter=',')[5:, 1],31))

    volts -= volts[0]

    ax.plot(field, np.abs(volts), lw=2, c=plt.cm.jet_r(i / len(angles)), label=str(angles[i])+r'$\degree$')

    if i > 4:

        locs1 = field > fit_ranges[inds[i]][0]
        locs2 = field < fit_ranges[inds[i]][1]

        locs = locs1 & locs2

        field_fit = field[locs]
        volts_fit = volts[locs]
        field_fit = np.insert(field_fit, 0, 0, axis=0)
        volts_fit = np.insert(volts_fit, 0, 0, axis=0)

        m,b = np.polyfit(field_fit, volts_fit, 1)

        ax.plot(np.linspace(-0,41.5), m*np.linspace(-0,41.5), linestyle='-.',lw=1, c=plt.cm.jet_r(i / len(angles)), alpha=0.4)

        small_plot_locs = field > 32

        small_field = field[small_plot_locs]
        small_volts = volts[small_plot_locs]

        # ax3.plot(small_field, small_volts, lw=2, c=plt.cm.jet(i / len(angles)))

    deriv = 1e3*savgol_filter(volts, 225, 3, 1)

    ax2.plot(field, deriv - i*0.15, c=plt.cm.jet_r(i / len(angles)),lw=2)


file = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_67.5deg_sweep1.csv'

dat = load_matrix(file)

field = dat[:,0]
tau = dat[:,1]

locs = field > 34.9

field = field[locs]
tau = tau[locs]

new_field, new_tau = interp_range(field, tau, 38.1, 40.5)

fit_tau = savgol_filter(new_tau, tesla_window(new_field, 1), 3)

subtracted_tau = new_tau - fit_tau

inv_x, inv_y = invert_x(new_field, subtracted_tau)

fft_x, fft_y = qo_fft(inv_x, inv_y,freq_cut=15000)


ax3.plot(new_field, new_tau, c=plt.cm.jet_r(5/len(angles)))
ax4.plot(new_field, subtracted_tau, c=plt.cm.jet_r(5/len(angles)))

ax5.plot(fft_x, fft_y/np.max(fft_y))

ax4.set_ylim(-4e-7, 4e-7)

publication_plot(ax, '', r'$\tau$ (arb.)')
publication_plot(ax2, '$\mu_0H_0$ (T)', r'$\frac{\partial}{\partial (\mu_0H)}\tau$ (arb.)')

publication_plot(ax3, '', r'$\tau$ (arb.)')
publication_plot(ax4, '$\mu_0H_0$ (T)', r'$\Delta\tau$ (arb.)')
publication_plot(ax5, 'Frequency (T)', r'FFT Amplitude (arb.)')
# publication_plot(ax4, 'Frequency (kT)', r'Amplitude (arb.)')

handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles[::-1], labels[::-1], framealpha=0, ncol=1, loc='best',
              prop={'size': 20, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines()[::-1], legend.get_texts()[::-1]):
    text.set_color(line.get_color())

plt.tight_layout(pad=2)
# plt.savefig('/Volumes/GoogleDrive/My Drive/FirstYearReport/Figures/VT69-torque.png', dpi=300)
plt.show()