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
'''

/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/VT15/data/14/qos/bgsub_0.3K_up_14deg_15-20T_s001.csv
/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/VT15/data/14/qos/bgsub_0.3K_up_14deg_21-26T_s001.csv
/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/VT15/data/14/qos/bgsub_0.3K_up_14deg_28-32T_s001.csv
/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/VT15/data/14/qos/bgsub_0.3K_up_14deg_32-37T_s001.csv
/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/VT15/data/14/qos/bgsub_0.3K_up_14deg_34-39T_s001.csv
/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/VT15/data/14/qos/bgsub_0.3K_up_14deg_40-44.5T_s001.csv
/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/VT15/data/14/qos/bgsub_inv_0.3K_up_14deg_15-20T_s001.csv
/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/VT15/data/14/qos/bgsub_inv_0.3K_up_14deg_21-26T_s001.csv
/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/VT15/data/14/qos/bgsub_inv_0.3K_up_14deg_28-32T_s001.csv
/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/VT15/data/14/qos/bgsub_inv_0.3K_up_14deg_32-37T_s001.csv
/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/VT15/data/14/qos/bgsub_inv_0.3K_up_14deg_34-39T_s001.csv
/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/VT15/data/14/qos/bgsub_inv_0.3K_up_14deg_40-44.5T_s001.csv

'''

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



fig, a = MakePlot(figsize=(8,10),gs=True).create()
gs = fig.add_gridspec(3,7)
ax3 = fig.add_subplot(gs[0, :3])
ax4 = fig.add_subplot(gs[0, 4:7])
ax5 = fig.add_subplot(gs[1, :3])
ax6 = fig.add_subplot(gs[1, 4:7])
ax1 = fig.add_subplot(gs[2, :3])
ax2 = fig.add_subplot(gs[2, 4:7])

dat_qo_40_45 = load_matrix('/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/VT15/data/14/qos/bgsub_0.3K_up_14deg_40-44.5T_s001.csv')

x = dat_qo_40_45[:,0]
y = dat_qo_40_45[:,1]

x, y = invert_x(x, y)

fft_x, fft_y = qo_fft(x, y,freq_cut=12000)


ax1.plot(x*1e2,y*1e4,lw=1.1,c='#820263')
ax2.plot(fft_x/1e3,fft_y/np.max(fft_y),lw=1.5,c='#820263')

publication_plot(ax1, r'($\mu_0H)^{-1}$ ($\times 10^2$ T$^{-1}$)', '',label_fontsize=20,tick_fontsize=17)

ax2.set_xticks([0,6,12])
publication_plot(ax2, r'Frequency (kT)', r'Amplitude (arb.)',label_fontsize=20,tick_fontsize=17)

ax1.annotate(r'40 T < $\mu_0H$ < 44.5 T', xy=(0.6, 0.15), xycoords='axes fraction',
            ha="center", va="center", fontname='arial', fontsize=16)
# ax3.annotate(r'$\phi = 14\degree$', xy=(0.4, 0.84), xycoords='axes fraction',
#             ha="center", va="center", fontname='arial', fontsize=20)

dat_qo_21_26 = load_matrix('/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/VT15/data/14/qos/bgsub_0.3K_up_14deg_21-26T_s001.csv')

x = dat_qo_21_26[:,0]
y = dat_qo_21_26[:,1]

x, y = invert_x(x, y)

fft_x, fft_y = qo_fft(x, y,freq_cut=12000)


ax3.plot(x*1e2,y*1e4,lw=1.1,c='#00E8FC')
ax4.plot(fft_x/1e3,fft_y/np.max(fft_y),lw=1.5,c='#00E8FC')

publication_plot(ax3, r'', '',label_fontsize=20,tick_fontsize=17)
ax4.set_xticks([0,6,12])
publication_plot(ax4, r'', r'Amplitude (arb.)',label_fontsize=20,tick_fontsize=17)

ax3.annotate(r'21 T < $\mu_0H$ < 26 T', xy=(0.6, 0.13), xycoords='axes fraction',
            ha="center", va="center", fontname='arial', fontsize=16)

ax3.annotate(r'$\phi = 14\degree$', xy=(0.4, 0.84), xycoords='axes fraction',
            ha="center", va="center", fontname='arial', fontsize=16)

dat_qo_32_37 = load_matrix('/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/VT15/data/14/qos/bgsub_0.3K_up_14deg_32-37T_s001.csv')

x = dat_qo_32_37[:,0]
y = dat_qo_32_37[:,1]

x, y = invert_x(x, y)

fft_x, fft_y = qo_fft(x, 1e6*y,freq_cut=12000)


ax5.plot(x*1e2,y*1e4,lw=1.1,c='#E94F37')
ax6.plot(fft_x/1e3,fft_y/np.max(fft_y),lw=1.5,c='#E94F37')

publication_plot(ax5, r'', '',label_fontsize=20,tick_fontsize=17)
ax6.set_xticks([0,6,12])
publication_plot(ax6, r'', r'Amplitude (arb.)',label_fontsize=20,tick_fontsize=17)

ax5.annotate(r'32 T < $\mu_0H$ < 37 T', xy=(0.6, 0.13), xycoords='axes fraction',
            ha="center", va="center", fontname='arial', fontsize=16)

fig.text(0.025,0.5,'Capacitance'+ r'($\times 10^{-4}$ pF)',
         ha='center',va='center', rotation='vertical',fontsize=24, fontname='arial')
#  fig.text(0.51, 0.55, '$\ln(R)$ ', va='center', rotation='vertical',fontsize=24, fontname='arial')


# plt.savefig('/Volumes/GoogleDrive/My Drive/FirstYearReport/Figures/FSVT2-rotator.png', dpi=300)
plt.tight_layout(pad=10)
plt.savefig('/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/VT15/VT15-14degrees-QOs3.png',dpi=300)
plt.show()






