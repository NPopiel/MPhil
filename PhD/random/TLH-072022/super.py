import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelmin, argrelmax
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
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



def tesla_window(x, tesla_wind):
    # print(2 * int(round(0.5 * tesla_wind / x[1] - x[0])) + 1)
    return 2 * int(round(0.5 * tesla_wind / np.abs(x[1] - x[0]))) + 1

fig, a = MakePlot(figsize=(16, 14), gs=True).create()
gs = fig.add_gridspec(1, 3)
ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[0, 2])


colours = ['#832388',
           '#912884',
           '#9E2C80',
           '#AC317C',
           '#BA3577',
           '#C83A73',
           '#D53E6F',
           '#E3436B']

#
# def filter_in_chunks(x, y,windowsize,filter_window):
#
#     n = 12800
#
#     ys = []
#
#     for i in range(0, len(y[:12800*28]), n):
#         small_y = y[i:i+n]
#
#         new_small_y = median_filter(small_y, 9088)
#
#         ys.append(new_small_y)
#
#     return x[:12800*28], np.hstack(ys)
#


path = '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/'

field = np.flip(load_matrix(path+'super/0.035K_102deg_sweep292_up.csv')[:,0])

tau_x = np.flip(load_matrix(path+'super/0.035K_102deg_sweep292_up.csv')[:,1])
print(len(tau_x))



# tau_x = filter_in_chunks(field, tau_x,int(round(len(tau_x)/28))-1,0.71)

field_diff = np.diff(field)

inds = field_diff > 0

field = field[:-1]
tau_x = tau_x[:-1]
field = field[inds]
tau_x = tau_x[inds]



# inds1 = field < 21.96354
# inds2 = field > 22.13925
#
# inds = inds1 + inds2
#
# field = field[inds]
# tau_x = tau_x[inds]

new_field, new_tau = interp_range(field, tau_x, 17.95, 27.9,0.0001)

# # cut first spurious point
#
# inds1 = new_field < 22.6621
# inds2 = new_field > 22.6978
#
# inds = inds1 + inds2
#
# inds2smooth = ~inds
#
# tau_small_smooth = medfilt(savgol_filter(new_tau[inds2smooth],301,0), 101)
#
# new_tau[inds2smooth] = np.NaN
#
# inds1 = new_field < 21.99354
# inds2 = new_field > 22.03925
#
# inds = inds1 + inds2
#
# inds2smooth = ~inds
#
# tau_small_smooth = medfilt(savgol_filter(new_tau[inds2smooth],301,0), 101)
#
# new_tau[inds2smooth] = np.NaN
#
# inds1 = new_field < 20.95
# inds2 = new_field > 21.08
#
# inds = inds1 + inds2
#
# inds2smooth = ~inds
#
# tau_small_smooth = medfilt(savgol_filter(new_tau[inds2smooth],301,0), 101)
#
# new_tau[inds2smooth] = np.NaN
#
# inds1 = new_field < 20.74
# inds2 = new_field > 20.86
#
# inds = inds1 + inds2
#
# inds2smooth = ~inds
#
# tau_small_smooth = medfilt(savgol_filter(new_tau[inds2smooth],301,0), 101)
#
# new_tau[inds2smooth] = np.NaN
#
# inds1 = new_field < 19.82
# inds2 = new_field > 20.1
#
# inds = inds1 + inds2
#
# inds2smooth = ~inds
#
# tau_small_smooth = medfilt(savgol_filter(new_tau[inds2smooth],301,0), 101)
#
# new_tau[inds2smooth] = np.NaN
#
# inds1 = new_field < 19.42
# inds2 = new_field > 19.53
#
# inds = inds1 + inds2
#
# inds2smooth = ~inds
#
# tau_small_smooth = medfilt(savgol_filter(new_tau[inds2smooth],301,0), 101)
#
# new_tau[inds2smooth] = np.NaN
# inds1 = new_field < 18.63
# inds2 = new_field > 18.83
#
# inds = inds1 + inds2
#
# inds2smooth = ~inds
#
# tau_small_smooth = medfilt(savgol_filter(new_tau[inds2smooth],301,0), 101)
#
# new_tau[inds2smooth] = np.NaN


def cut_to_value(y, value):
    y[np.abs(y)>value] = value

    return y


# new_tau = medfilt(new_tau, 1501)
# new_tau = medfilt(new_tau, 501)

fit_tau = savgol_filter(new_tau, tesla_window(new_field, .18), 0)
# fit_tau = savgol_filter(new_tau, tesla_window(new_field, .18), 0)
# fit_tau = medfilt(fit_tau, 5571)
# fit_tau = savgol_filter(fit_tau, tesla_window(new_field, .5), 3)
# fit_tau = savgol_filter(fit_tau, tesla_window(new_field, .2), 3)
# fit_tau = savgol_filter(fit_tau, tesla_window(new_field, .3), 3)

# fit_tau = medfilt(fit_tau, 501)


subtracted_tau = new_tau - fit_tau

# cut first spurious point

inds1 = new_field < 22.62
inds2 = new_field > 22.73

inds = inds1 + inds2

inds2smooth = ~inds

# tau_small_smooth = savgol_filter(subtracted_tau[inds2smooth],301,5)
tau_small_smooth = cut_to_value(subtracted_tau[inds2smooth], 0.96e-7)
subtracted_tau[inds2smooth] = tau_small_smooth

inds1 = new_field < 22.6621
inds2 = new_field > 22.6978

inds = inds1 + inds2

inds2smooth = ~inds

# tau_small_smooth = savgol_filter(subtracted_tau[inds2smooth],301,5)
tau_small_smooth = cut_to_value(subtracted_tau[inds2smooth], 0.98e-7)

subtracted_tau[inds2smooth] = tau_small_smooth

inds1 = new_field < 21.99354
inds2 = new_field > 22.03925

inds = inds1 + inds2

inds2smooth = ~inds

tau_small_smooth = cut_to_value(subtracted_tau[inds2smooth], 1.82e-7)

subtracted_tau[inds2smooth] = tau_small_smooth

inds1 = new_field < 20.95
inds2 = new_field > 21.08

inds = inds1 + inds2

inds2smooth = ~inds

tau_small_smooth = savgol_filter(subtracted_tau[inds2smooth],501,0)

subtracted_tau[inds2smooth] = tau_small_smooth

inds1 = new_field < 20.74
inds2 = new_field > 20.86

inds = inds1 + inds2

inds2smooth = ~inds

tau_small_smooth = medfilt(savgol_filter(subtracted_tau[inds2smooth],301,0), 101)

subtracted_tau[inds2smooth] = tau_small_smooth

inds1 = new_field < 19.82
inds2 = new_field > 20.1

inds = inds1 + inds2

inds2smooth = ~inds

tau_small_smooth = savgol_filter(subtracted_tau[inds2smooth],1001,0)

subtracted_tau[inds2smooth] = tau_small_smooth

inds1 = new_field < 19.42
inds2 = new_field > 19.53

inds = inds1 + inds2

inds2smooth = ~inds

tau_small_smooth = savgol_filter(subtracted_tau[inds2smooth],501,0)

subtracted_tau[inds2smooth] = tau_small_smooth

inds1 = new_field < 18.63
inds2 = new_field > 18.83

inds = inds1 + inds2

inds2smooth = ~inds

tau_small_smooth = savgol_filter(subtracted_tau[inds2smooth],501,0)

subtracted_tau[inds2smooth] =tau_small_smooth

# new_field, subtracted_tau = interp_range(new_field, subtracted_tau, 17.96, 27.89, 0.0001)

inds1 = new_field > 18.06
inds2 = new_field < 27.8

inds = inds1 & inds2

new_field = new_field[inds]
subtracted_tau = subtracted_tau[inds]

# subtracted_tau = savgol_filter(subtracted_tau, tesla_window(new_field, .3), 3)
# subtracted_tau = medfilt(subtracted_tau, 11)
# subtracted_tau = medfilt(subtracted_tau, 31)

interp_fft_field, interp_fft_tau = interp_range(field, tau_x, 24.05, 27.8)

fft_subtracted_tau = interp_fft_tau - savgol_filter(interp_fft_tau, tesla_window(interp_fft_field, .71), 3)

fft_inds1 = new_field > 24.05
fft_inds2 = new_field < 27.8

fft_inds = fft_inds1 & fft_inds2


inv_x, inv_y = invert_x(interp_fft_field,fft_subtracted_tau)

fft_x, fft_y = qo_fft(inv_x, inv_y, freq_cut=12000)


bgsub_cut = np.array([new_field, subtracted_tau]).T

inv_field, inv_tau = invert_x(new_field, subtracted_tau)
bgsub_inv_cut = np.array([inv_field, inv_tau]).T

ffts = np.array([fft_x, fft_y]).T

np.savetxt(path+'bgsub_super_cut.csv',bgsub_cut,delimiter=',')
np.savetxt(path+'bgsub_super_cut_inv.csv',bgsub_inv_cut,delimiter=',')
np.savetxt(path+'fft_super_cut.csv',ffts,delimiter=',')




ax1.plot(new_field, 1e7*subtracted_tau)

ax1.set_ybound(-10, 10)

ax2.plot(fft_x/1e3, fft_y/np.max(fft_y))

publication_plot(ax1, r'$\mu_0H$ (T)', r'$\tau$ (arb.)')

publication_plot(ax2, r'Frequency (kT)', r'FFT amplitude (arb.)')



# fft_ax.set_xbound(0,25.05)

plt.tight_layout(pad=2)
plt.savefig('/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/super.png', dpi=300)
plt.show()




