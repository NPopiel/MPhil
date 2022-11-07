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

dat = load_matrix('/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/VT15/muB/data/0.3K_16deg_sweep1.csv')

field = dat[:,0]

bgsub_torque = median_filter(dat[:,1],11)



fig, a = MakePlot(figsize=(16,10),gs=True).create()
gs = fig.add_gridspec(1,4)
ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[0, 2:4])

rang1 = field < 50
rang2 = field > 0

ranga = rang1 & rang2

x, y = invert_x(field[ranga], bgsub_torque[ranga])

fft_x, fft_y = qo_fft(x, y,freq_cut=8000)



ax1.plot(x,y,lw=1.5,c='#820263')
ax2.plot(fft_x,fft_y,lw=1.2,c='#820263')
#
#
# #variables
# samplfreq = 1000 #print(samplfreq)#the sampling frequency of your data (mine=100Hz, yours=44100)
# factor = 1     #incr./decr frequency (speed up / slow down by a factor) (normal speed = 1)
#
#
# #normalise the data to between -1 and 1. If your data wasn't/isn't normalised it will be very noisy when played here
# sd.play( y / np.max(np.abs(y)), samplfreq*factor)

publication_plot(ax1, r'($\mu_0H)^{-1}$ (T$^{-1}$)', r'$\Delta\tau$ ($\times 10^{-4}$ $\mu_B$T per f.u.)',label_fontsize=20,tick_fontsize=17)

# ax2.set_xticks([0,6,12])
publication_plot(ax2, r'Frequency (T)', r'Amplitude (arb.)',label_fontsize=20,tick_fontsize=17)

plt.tight_layout(pad=.5)
plt.show()




