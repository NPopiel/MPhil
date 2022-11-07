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

files.reverse()
angles.reverse()

fit_ranges = [(32.3, 34),
              (33.3, 35),
              (35.3, 37)]
              # (37.1, 38.10)]

small_plot_fields = [32.5, 33.5, 35.3]

fig, a = MakePlot(figsize=(14,9),gs=True).create()

gs = fig.add_gridspec(3,2)
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 1])


# #
B_thresh = 10
ax = ax1
for i, f1 in enumerate(files):


    print(f1)

    field = np.genfromtxt(f1, delimiter=',')[5:, 0]
    volts = 1e3*np.abs(medfilt(np.genfromtxt(f1, delimiter=',')[5:, 1],31))

    volts -= volts[0]

    ax.plot(field, np.abs(volts), lw=2, c=plt.cm.jet(i / len(angles)), label=str(angles[i]))

    if i < 3:

        locs1 = field > fit_ranges[i][0]
        locs2 = field < fit_ranges[i][1]

        locs = locs1 & locs2

        field_fit = field[locs]
        volts_fit = volts[locs]

        f = np.poly1d(np.polyfit(field_fit, volts_fit, 1))

        ax.plot(field, f(field), linestyle='-.',lw=1, c=plt.cm.jet(i / len(angles)), alpha=0.4)

        small_plot_locs = field > 32

        small_field = field[small_plot_locs]
        small_volts = volts[small_plot_locs]

        ax2.plot(small_field, small_volts, lw=2, c=plt.cm.jet(i / len(angles)))

        # ax3.plot(small_field, savgol_filter(small_volts, 125, 3, 1), c=plt.cm.jet(i / len(angles)),lw=2)


'''
bgsub_0.4K_up_67.5deg_38.2-41.4T_s003.csv
bgsub_0.4K_up_60deg_38.2-41.4T_s001.csv
bgsub_0.4K_up_64deg_38.2-41.4T_s002.csv

/Volumes/GoogleDrive/My Drive/FirstYearReport/QOs_VT69/bgsub_fft_0.4K_up_60deg_38.2-41.4T_s001.csv
/Volumes/GoogleDrive/My Drive/FirstYearReport/QOs_VT69/bgsub_fft_0.4K_up_64deg_38.2-41.4T_s002.csv
/Volumes/GoogleDrive/My Drive/FirstYearReport/QOs_VT69/bgsub_fft_0.4K_up_67.5deg_38.2-41.4T_s003.csv

/Volumes/GoogleDrive/My Drive/FirstYearReport/QOs_VT69/bgsub_inv_0.4K_up_60deg_38.2-41.4T_s001.csv
/Volumes/GoogleDrive/My Drive/FirstYearReport/QOs_VT69/bgsub_inv_0.4K_up_64deg_38.2-41.4T_s002.csv
/Volumes/GoogleDrive/My Drive/FirstYearReport/QOs_VT69/bgsub_inv_0.4K_up_67.5deg_38.2-41.4T_s003.csv
'''

qo_files = ['bgsub_0.4K_up_67.5deg_38.2-41.4T_s003.csv']
# 'bgsub_0.4K_up_60deg_38.2-41.4T_s001.csv',
# 'bgsub_0.4K_up_64deg_38.2-41.4T_s002.csv']

qo_files2 = ['bgsub_inv_0.4K_up_67.5deg_32-41.5T_s003.csv',
'bgsub_inv_0.4K_up_60deg_32-41.5T_s001.csv',
'bgsub_inv_0.4K_up_64deg_32-41.5T_s002.csv']

qo_files =  ['bgsub_inv_0.4K_up_67.5deg_38.2-41.4T_s003.csv']
# 'bgsub_inv_0.4K_up_60deg_38.2-41.4T_s001.csv',
# 'bgsub_inv_0.4K_up_64deg_38.2-41.4T_s002.csv']

fft_files = ['bgsub_fft_0.4K_up_67.5deg_38.2-41.4T_s003.csv']
# 'bgsub_fft_0.4K_up_60deg_38.2-41.4T_s001.csv',
# 'bgsub_fft_0.4K_up_64deg_38.2-41.4T_s002.csv']

fft_files2 = ['bgsub_fft_0.4K_up_67.5deg_32-41.5T_s003.csv',
'bgsub_fft_0.4K_up_60deg_32-41.5T_s001.csv',
'bgsub_fft_0.4K_up_64deg_32-41.5T_s002.csv']

# qos = load_matrix('/Volumes/GoogleDrive/My Drive/FirstYearReport/QOs_VT69/bgsub_0.4K_up_64deg_38.2-41.4T_s002.csv')

for i, file in enumerate(qo_files):
    qos = load_matrix('/Volumes/GoogleDrive/My Drive/FirstYearReport/QOs_VT69/' + file)

    ffts = load_matrix('/Volumes/GoogleDrive/My Drive/FirstYearReport/QOs_VT69/' + fft_files2[i])

    field_qo = qos[:,0]
    qo = median_filter(qos[:,1], 43)

    freq = ffts[:,0]
    amp = ffts[:,1]

    amp /= np.max(amp)

    ax3.plot(field_qo, qo,c=plt.cm.jet(i/ len(angles)),lw=2)

    # freq, amp = qo_fft(field_qo, qo,freq_cut=45000)
    #

    ls = freq<20000

    ax4.plot(freq[ls]/1e3, amp[ls], c=plt.cm.jet(i/ len(angles)), lw=2)


# ax3.set_ylim(-5e-7, 5e-7)
publication_plot(ax, '$\mu_0H_0$ (T)', r'$\tau$ (arb.)')
publication_plot(ax2, '$\mu_0H_0$ (T)', r'$\tau$ (arb.)')
ax3.set_ylim(-1e-6, 1e-6)
publication_plot(ax3, '$\mu_0H_0$ (T)', r'$\Delta\tau$ (arb.)')
publication_plot(ax4, 'Frequency (kT)', r'Amplitude (arb.)')


legend = ax.legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 20, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

plt.tight_layout(pad=2)
plt.show()