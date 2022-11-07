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




path = '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/'




filenames = [#path+'0.035K_192deg_sweep306_up.csv',
             #'/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/0.035K_186deg_sweep226_up.csv',
             path+'0.035K_174deg_sweep304_up.csv',
            '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/0.035K_162deg_sweep232_down.csv',
            '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/0.035K_150deg_sweep239_up.csv',
            # '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/0.035K_144deg_sweep250_down.csv',
             path+'0.035K_126deg_sweep299_up.csv',
             path+'0.035K_114deg_sweep296_up.csv',
            '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/0.035K_102deg_sweep289_down.csv',
            '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/0.035K_96deg_sweep283_down.csv'
]



field_ranges = (25.1, 27.9, 0.71)



def tesla_window(x, tesla_wind):
    # print(2 * int(round(0.5 * tesla_wind / x[1] - x[0])) + 1)
    return 2 * int(round(0.5 * tesla_wind / np.abs(x[1] - x[0]))) + 1

fig, a = MakePlot(figsize=(10, 12), gs=True).create()
gs = fig.add_gridspec(3, 5, wspace=0.7)
ax2 = fig.add_subplot(gs[1:, :3])
# gs = fig.add_gridspec(3,10, wspace=45)
ax3 = fig.add_subplot(gs[1:, 3:5])
gs = fig.add_gridspec(3, 5)
ax1 = fig.add_subplot(gs[0,:5])




super_dat = load_matrix('/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/bgsub_super_cut.csv')

field = super_dat[:,0]
tau = super_dat[:,1]

inds = field > 22.75
field = field[inds]
tau = tau[inds]

ax1.plot(field,1e7*tau,linewidth=2.2,c='#F74A4D')#'#F52571')

publication_plot(ax1, r'$\mu_0H$ (T)', r'$\Delta\tau$ (arb.)')

ax1.set_ybound(-3.5,3.5)
ax1.set_yticks([-3,0,3])



xs, inv_xs, ys, inv_ys = [], [], [], []

angles = np.abs(np.array([6, 18, 30,  54,66, 78,84]))#-12,-6,36,


sorted_inds = np.argsort(angles)
filenames = np.array(filenames)[sorted_inds].tolist()
# field_ranges = np.array(field_ranges)[sorted_inds].tolist()
angles = angles[sorted_inds]



fft_files = ['/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/new_ffts/mega1666.dat.fft',
             '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/new_ffts/mega16718.dat.fft',
             '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/new_ffts/mega16830.dat.fft',
             # '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/new_ffts/mega16936.dat.fft',
             '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/new_ffts/mega17054.dat.fft',
             '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/new_ffts/mega17166.dat.fft',
             '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/new_ffts/mega17278.dat.fft',
             '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/new_ffts/mega17384.dat.fft']



markers = ['o', '+', 'v', 'x','4']

# colours = ['#56CEBE',
#            '#60ADBF',
#            '#6A8DC0',
#            '#746CC1',
#            '#7E4BC2',
#            '#882BC3',
#            '#920AC4']

colours = ['#3bcfd4','#6CC0A1','#9CB16D','#E49B1F','#FA6F29','#F74A4D','#F52571','#F20094']#'#CCA239'4th,'#FC9305',6th

# colours = colours[sorted_inds]

# Plot the frequencies in the right-most panel

fft_xs, fft_ys = [], []

grads = []
  # x in data untis, y in axes fraction



offsets_wiggles = [0, .65, 2*.65, 3*0.65-0.1,4*0.65-0.25, 5*0.65-0.25, 6*0.65-0.2]
offsets_ffts = [0, 0.9, 2*0.9, 3*0.9, 4*0.9+.1, 5*0.9+.1, 6*0.9+.1]

for i, filename in enumerate(filenames):


        dat = load_matrix(filename)
        field = dat[:, 0]
        tau = dat[:, 1]

        if field[0] > field[-1]:
            field = np.flip(field)
            tau = np.flip(tau)

        if tau[0] > tau[-1]:
            tau *= -1

        tau -= tau[0]
        tau-=tau[0]

        new_field, new_tau = interp_range(field, tau, 25.1, 27.9)

        new_tau-=new_tau[0]



        fit = np.polyfit(new_field, new_tau,1)

        m = fit[0]
        grads.append(m)

        fit_tau = savgol_filter(new_tau, tesla_window(new_field, 0.7), 3)

        subtracted_tau = new_tau - fit_tau

        inv_x, inv_y = invert_x(new_field, subtracted_tau)
        fft_x, fft_y = qo_fft(inv_x, inv_y, freq_cut=10000)

        fft_xs.append(fft_x)
        fft_ys.append(fft_y)

        mega_fft = np.loadtxt(fft_files[i], skiprows=1, delimiter='\t')

        ax2.plot(new_field, subtracted_tau * 1e6 + offsets_wiggles[i], linewidth=1.4, c=colours[i], label = str(angles[i])+r'$\degree$')
        ax3.plot(mega_fft[:,0] / 1e3, (mega_fft[:,1] / np.max(mega_fft[:,1])) + offsets_ffts[i], linewidth=1.8, c=colours[i])

        np.savetxt('/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/data/s1c/'+str(angles[i])+'.csv',
                   np.array([new_field, subtracted_tau]).T, delimiter=',')

        ax3.annotate(str(angles[i])+r'$\degree$', xy=(7.84, offsets_ffts[i]+0.1), xycoords='data',
                     fontname='arial',fontsize=22,color=colours[i])

        ax2.annotate(str(angles[i]) + r'$\degree$', xy=(27.95, offsets_wiggles[i]), xycoords='data',
                     fontname='arial', fontsize=22, color=colours[i])





ax2.set_xbound(25,28.3)
ax2.set_xticks([25, 26,27,28])
ax2.set_ybound(-0.5, 4.3)

# ax2.set_xbound(25,28)
# ax2.set_xticks([25, 26,27,28])

ax3.set_ybound(-0.02,6.75)
ax3.set_xbound(0,10)





ax3.set_xticks([0,5,10])

# things to do
# squish FFTs, maybe square
# new panel wit figures after squishing FFT
# play with big sweep



# ax3.set_xbound(-15, 90)
#
publication_plot(ax2, r'$\mu_0H$ (T)', r'$\Delta\tau$ (arb.)')
publication_plot(ax3, r'Frequency (kT)', r'FFT amplitude (arb.)')
# publication_plot(axin, r'$\phi$ ($\degree$)', r'$\tau^\prime$ (arb.)',label_fontsize=14,tick_fontsize=12)
# publication_plot(axin, r'Frequency (kT)', r'FFT amplitude (arb.)',label_fontsize=14,tick_fontsize=12)


# change here

ax1.annotate(r'$T = 30$ mK ', xy=(0.38, 0.1), xycoords='axes fraction',
             ha="left", va="center", fontname='arial', fontsize=26)

# ax1.annotate(r'$\phi = 78 \degree$', xy=(0.25, 0.1), xycoords='axes fraction',
#              ha="left", va="center", fontname='arial', fontsize=20)

import matplotlib as mpl

# read image file
with mpl.cbook.get_sample_data('/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/78deg-removebg-preview.png') as file:
    arr_image = plt.imread(file, format='png')

# Draw image
axin = ax1.inset_axes([0.1, 0.06, 0.2, .2])  # create new inset axes in data coordinates
axin.imshow(arr_image)
axin.axis('off')

axin.annotate(r'$\phi = 78 \degree$', xy=(0.25, 0.47), xycoords='axes fraction',
              ha="left", va="center", fontname='arial', fontsize=20)

axin.annotate(r'$b$', xy=(0, 1.2), xycoords='axes fraction',
              ha="center", va="center", fontname='arial', fontsize=20)

axin.annotate(r'$c$', xy=(1.12, 0), xycoords='axes fraction',
              ha="center", va="center", fontname='arial', fontsize=20)







ax1.annotate(r'a', xy=(0.02, 0.9), xycoords='axes fraction',
             ha="left", va="center", fontname='arial', fontweight='bold', fontsize=28)

ax2.annotate(r'b', xy=(0.02*(1+3/5), .95), xycoords='axes fraction',
             ha="left", va="center", fontname='arial', fontweight='bold', fontsize=28)

ax3.annotate(r'c', xy=(0.02*(1+7/5), .95), xycoords='axes fraction',
             ha="left", va="center", fontname='arial', fontweight='bold', fontsize=28)



# fig.add_axes([0,-55,1,1]).axis("off")
plt.tight_layout(pad=.5)
plt.savefig('/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/new_figs/c2b-w-mega-v4.pdf'
            , dpi=300, bbox_inches = "tight")
# plt.show()




