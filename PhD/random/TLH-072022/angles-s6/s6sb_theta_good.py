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
from matplotlib.ticker import AutoMinorLocator


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


angles = np.abs(np.array([-6, 0, 6, 12, 15, 21, 27, 33])+9)



sorted_inds = np.argsort(angles)

filenames = np.array(filenames)[sorted_inds].tolist()
# field_ranges = np.array(field_ranges)[sorted_inds].tolist()
angles = angles[sorted_inds]



def tesla_window(x, tesla_wind):
    # print(2 * int(round(0.5 * tesla_wind / x[1] - x[0])) + 1)
    return 2 * int(round(0.5 * tesla_wind / np.abs(x[1] - x[0]))) + 1

fig, a = MakePlot(figsize=(16, 10), gs=True).create()
gs = fig.add_gridspec(2, 8, hspace=0)
ax1 = fig.add_subplot(gs[0, :3])
ax2 = fig.add_subplot(gs[1, :3])
gs = fig.add_gridspec(2, 8, wspace=2, hspace=0.1)
ax3 = fig.add_subplot(gs[0, 3:5])
ax4 = fig.add_subplot(gs[1, 3:5])
gs = fig.add_gridspec(2, 8)
ax5 = fig.add_subplot(gs[:, 5:])




xs, inv_xs, ys, inv_ys = [], [], [], []

frequencys = [(3.2, 3.7,6.274),
              (2.96, 3.8, 6.25),
              (2.5, 3.7, 6.7),
              (3.5, None,6.9),
              (None, None, 8.5),
              (None,None,9.9),
              (4.5, None,13.2),
              (5.7, None, 15.6)]

frequencys = np.array(frequencys)[sorted_inds].tolist()

print(angles)
print(frequencys)


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

# Plot the frequencies in the right-most panel

for i, filename in enumerate(filenames):

        freqs = frequencys[i]

        for j, f in enumerate(freqs):

            if f is not None:

                ax5.scatter(angles[i], f, marker=markers[j], c=colours[i], s=200)

ax5.scatter(angles[-1], 18.12, marker='x', c=colours[-1], s=200)

aoki_s1_angles = load_matrix('/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/AOKI-DATA/AokiS1manual.csv')[:,0]
aoki_s1_freqs = load_matrix('/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/AOKI-DATA/AokiS1manual.csv')[:,1]/1e3

aoki_s2_angles = load_matrix('/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/AOKI-DATA/AokiS2manual.csv')[:,0]
aoki_s2_freqs = load_matrix('/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/AOKI-DATA/AokiS2manual.csv')[:,1]/1e3

ax5.scatter(aoki_s1_angles, aoki_s1_freqs, s=100, c='k', marker='<', zorder=-1)
ax5.scatter(aoki_s2_angles, aoki_s2_freqs, s=100, c='k', marker='>', zorder=-1)



# plot the lower frequency example

dat = load_matrix(path+'0.035K_186deg_sweep048_up.csv')
field = dat[:, 0]
tau = dat[:, 1]

new_field, new_tau = interp_range(field, tau, 26, 27.9)

fit_tau = savgol_filter(new_tau, tesla_window(new_field, 0.45), 3)

subtracted_tau = new_tau - fit_tau

inv_x, inv_y = invert_x(new_field, subtracted_tau)
fft_x, fft_y = qo_fft(inv_x, inv_y, freq_cut=20000)

ax1.plot(new_field, subtracted_tau*1e7, linewidth=1.4, c='#A32E7E')
ax3.plot(fft_x/1e3, fft_y/np.max(fft_y),linewidth=1.8, c='#A32E7E')


# plot the lower frequency example '#C83A73'

dat = load_matrix(path+'0.035K_159deg_sweep087_down.csv')
field = dat[:, 0]
tau = dat[:, 1]

new_field, new_tau = interp_range(field, tau, 26, 27.9)

fit_tau = savgol_filter(new_tau, tesla_window(new_field, 0.3), 3)

subtracted_tau = new_tau - fit_tau

inv_x, inv_y = invert_x(new_field, subtracted_tau)
fft_x, fft_y = qo_fft(inv_x, inv_y, freq_cut=20000)

ax2.plot(new_field, subtracted_tau*1e7, linewidth=1.4, c='#C83A73')
ax4.plot(fft_x/1e3, fft_y/np.max(fft_y),linewidth=1.8, c='#C83A73')

ax3.set_ybound(0, 1.05)
ax3.set_xbound(0, 13.05)
ax4.set_ybound(0, 1.05)
ax4.set_xbound(0, 20.05)

# ax1.set_yticks([-2,-1,0,1,2])

ax3.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0])
ax4.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0])

ax1.set_xticks([26, 26.5,27,27.5,28])
ax1.set_xticklabels([])
ax2.set_xticks([26, 26.5,27,27.5,28])
ax2.set_xticklabels([26, 26.5,27,27.5,28])



ax5.set_ybound(0, 20)
ax5.set_yticks([0,2,4,6,8,10,12,14,16,18,20])
ax5.set_yticklabels([0,2,4,6,8,10,12,14,16,18,20])
ax5.set_xlim(-8, 90)

publication_plot(ax1, '', r'$\Delta\tau$ (arb.)')
publication_plot(ax2, r'$(\mu_0H)$ (T)', r'$\Delta\tau$ (arb.)')
publication_plot(ax3, r'', r'FFT amplitude (arb.)')
publication_plot(ax4, r'Frequency (kT)', r'FFT amplitude (arb.)')
publication_plot(ax5, r'$\theta$ ($\degree$)', r'Frequency (kT)')


ax5.annotate(r'$T = 30$ mK ', xy=(0.82, 0.05), xycoords='axes fraction',
             ha="center", va="center", fontname='arial', fontsize=20)

ax5.annotate(r'$90 \degree\Rightarrow\mathbf{H}\parallel[100]$ ', xy=(0.95, 0.9), xycoords='axes fraction',
             ha="right", va="center", fontname='arial', fontsize=20)

ax5.annotate(r'$0 \degree\Rightarrow\mathbf{H}\parallel [001]$ ', xy=(0.95, 0.95), xycoords='axes fraction',
             ha="right", va="center", fontname='arial', fontsize=20)

import matplotlib as mpl

# read image file
with mpl.cbook.get_sample_data('/Volumes/GoogleDrive/My Drive/cantilever-Popies.png') as file:
    arr_image = plt.imread(file, format='png')

# Draw image
axin = ax5.inset_axes([-6.5, 8, 40, 20], transform=ax5.transData)  # create new inset axes in data coordinates
axin.imshow(arr_image)
axin.axis('off')

#ax1.annotate(r'$\theta = -6\degree$ ', xy=(0.82, 0.05), xycoords='axes fraction',
#              ha="center", va="center", fontname='arial', fontsize=20, c='#A32E7E')
ax1.annotate(r'$\theta = 2\degree$ ', xy=(0.82, 0.05), xycoords='axes fraction',
             ha="center", va="center", fontname='arial', fontsize=20, c='#A32E7E')

# ax2.annotate(r'$\theta = 21\degree$ ', xy=(0.16, 0.05), xycoords='axes fraction',
#              ha="center", va="center", fontname='arial', fontsize=20, c='#C83A73')

ax2.annotate(r'$\theta = 25\degree$ ', xy=(0.16, 0.05), xycoords='axes fraction',
             ha="center", va="center", fontname='arial', fontsize=20, c='#C83A73')

# fft_ax.set_xbound(0,25.05)

plt.tight_layout(pad=1.5)
plt.savefig('/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/'+'aoki+updated_angles+angles-s6sb.pdf', dpi=300)
# plt.show()




