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



def tesla_window(x, tesla_wind):
    # print(2 * int(round(0.5 * tesla_wind / x[1] - x[0])) + 1)
    return 2 * int(round(0.5 * tesla_wind / np.abs(x[1] - x[0]))) + 1

fig, a = MakePlot(figsize=(12, 6), gs=True).create()
gs = fig.add_gridspec(1, 3)
ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[0, 2])



dat = load_matrix(path+'0.035K_159deg_sweep087_down.csv')

angle = 21

field = dat[:, 0]
tau = dat[:, 1]

if field[0] > field[-1]:
    field = np.flip(field)
    tau = np.flip(tau)

if tau[0] > tau[-1]:
    tau *= -1

tau -= tau[0]


new_field, new_tau = interp_range(field, tau, 25.681, 27.9)

fit_tau = savgol_filter(new_tau, tesla_window(new_field, 0.71), 3)

subtracted_tau = new_tau - fit_tau

inv_x, inv_y = invert_x(new_field, subtracted_tau)
fft_x, fft_y = qo_fft(inv_x, inv_y, freq_cut=12000)

ax1.plot(inv_x, inv_y, lw=3, c='indianred',
         label=' ' + str(angle) + r'$\degree$')

# Now i need to fount the number of times it crosses the axis, and where it is in inverse B


deriv = savgol_filter(inv_y, tesla_window(new_field, 0.71), 3, 1)

locs_where_neg = deriv < 0

diff_locs = np.diff(locs_where_neg)

inverse_fields = inv_x[:-1][diff_locs]

total_N = np.sum(diff_locs)

N = 5

for k, loc in enumerate(inverse_fields):

    ax2.scatter(loc, N+k)


# ax2.set_ybound(0, total_N+15)
ax2.set_xbound(0, inv_x[-1]+0.001*inv_x[-1])




topax = ax1.twiny()
fig.subplots_adjust(top=0.85)

topax.set_xlim([28,25.65])
topax.set_xlabel(r'$\mu_0H$ (T)',fontsize=22 , fontname='arial',labelpad=10)
plt.setp(topax.get_xticklabels(),fontsize=20 , fontname='arial')
plt.setp(ax1.get_xticklabels(),fontsize=22 , fontname='arial')
plt.setp(ax1.get_yticklabels(),fontsize=22, fontname='arial' )

ax1.minorticks_on()
ax1.tick_params('both', which='major', direction='in',
      bottom=True, top=False, left=True, right=True,length=5, width=1.2)
ax1.tick_params('both', which='minor', direction='in',
      bottom=True, top=False, left=True, right=True, length=3, width=1)
topax.tick_params(top=True, labeltop=True, which='major',length=5, width=1.2,left=False, labelleft=False,
                  right=False, labelright=False, bottom=False, labelbottom=False, direction='in',)
topax.tick_params(top=True, labeltop=True, which='minor',length=3, width=1,left=False, labelleft=False,
                  right=False, labelright=False, bottom=False, labelbottom=False, direction='in',)




publication_plot(ax2, r'$1/(\mu_0H)$ (T$^{-1}$)', r'N')



# fft_ax.set_xbound(0,25.05)

plt.tight_layout(pad=2)
# plt.savefig(path+'angles-draft1.png', dpi=300)
plt.show()




