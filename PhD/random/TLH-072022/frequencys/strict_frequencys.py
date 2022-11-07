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


path = '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/'

filenames = [#path+'0.035K_192deg_sweep306_up.csv',
             #'/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/0.035K_186deg_sweep226_up.csv',
             path+'0.035K_174deg_sweep304_up.csv',
            '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/0.035K_171deg_sweep257_down.csv',
            '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/0.035K_162deg_sweep232_down.csv',
            '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/0.035K_150deg_sweep239_up.csv',
            '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/0.035K_144deg_sweep250_down.csv',
             path+'0.035K_126deg_sweep299_up.csv',
             path+'0.035K_114deg_sweep296_up.csv',
            '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/0.035K_102deg_sweep289_down.csv',
            '/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/S1c/angles/0.035K_96deg_sweep283_down.csv'
]


def tesla_window(x, tesla_wind):
    # print(2 * int(round(0.5 * tesla_wind / x[1] - x[0])) + 1)
    return 2 * int(round(0.5 * tesla_wind / np.abs(x[1] - x[0]))) + 1

fig, a = MakePlot(figsize=(12, 10), gs=True).create()
gs = fig.add_gridspec(3, 2)
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[:, 1])


angles = np.abs(np.array([6, 9, 18, 30, 36, 54,66, 78,84]))#-12,-6,


sorted_inds = np.argsort(angles)
filenames = np.array(filenames)[sorted_inds].tolist()
# field_ranges = np.array(field_ranges)[sorted_inds].tolist()
angles = angles[sorted_inds]

# Original frequencys i pulled out of giga
#
#
# frequencys = [#(3.2, 4, 4.7, 5.2, 7.8),#-12
#               #(4.3, 5.6,6.2, None, None),#-6
#               (3.9, 4.6, 5.1, None, None),#6
#               (4,  4.9, 5.4, 6.1, None),#18 6.1 is maybe
#               (3.5, 4.1, 4.5, 5.3,7.0),#30 7.0
#               (3.3, 3.7, 4, 4.5, 7.9),#367.7
#               (3.1, 3.6, 4.1, None, 7.1),#54 7.1
#               (3.47,4,4.7,None,6.9),#66 ,9.9
#               (3.5,4.6, None, None, 7.7),#78
#               (3.8, 4.85,5.4, None, 7.45)] # 84


# frequencys = [(3.925, 4.579, 5.124, None), # 6
#               (4.007, 4.906, 5.424, None), #18
#               (3.543, 4.061, 4.497, None), # 30
#               (3.299, 3.656, 4.096, 4.564), # 36
#               (3.216, 3.601, None, None), # 54
#               (3.489, 3.980, None, None), # 66
#               (3.789, 4.606, None, None), # 78
#               (3.707, None, None, None)] # 84

frequencys_open_circles = [(4.579, 5.124, None), # 6
              (4.003, 4.565,  None), # 9
              (4.906, 5.424, None), #18
              (4.061, 4.497, None), # 30
              (3.299, 3.656, 4.564), # 36
              (3.601, None, None), # 54
              (3.980, None, None), # 66
              (4.606, None, None), # 78
              (None, None, None)] # 84

frequencys = [3.925,
              5.378,
              4.007,
              3.543,
              4.096,
              3.216,
              3.489,
              3.789,
              3.707]

questionable_frequencys = [(3.244, None, None, None), # 6
                           (4.003, 4.565, 5.378, None), # 9
                           (None, 6.16, None, None), # 18
                           (None, 5.315, None, None), # 30
                           (None, None, None, None), # 36
                           (6.873, None, None, None), # 54
                           (None, None, None, None), # 66
                           (None, None, None, None), # 78
                           (None, 4.852, 2.917, None)]# 84

frequencys = np.array(frequencys)[sorted_inds].tolist()


markers = ['o', '+', 'v', 'x','4']

# colours = ['#56CEBE',
#            '#60ADBF',
#            '#6A8DC0',
#            '#746CC1',
#            '#7E4BC2',
#            '#882BC3',
#            '#920AC4']

colours = ['#3bcfd4','#6CC0A1','#9CB16D','#CCA239','#E49B1F','#FA6F29','#F74A4D','#F52571','#F20094']#'#CCA239'4th,'#FC9305',6th

# colours = colours[sorted_inds]

# Plot the frequencies in the right-most panel


for i, filename in enumerate(filenames):

        freq = frequencys[i]
        ax1.scatter(angles[i], freq, marker='o', c=colours[i], s=245)


        for j, f in enumerate(frequencys_open_circles[i]):

            if f is not None:

                ax1.scatter(angles[i], f, marker='o', c=colours[i], s=245)

        for j, f in enumerate(questionable_frequencys[i]):

            if f is not None:

                ax1.scatter(angles[i], f, marker='o', c=colours[i], s=245)


angles = np.abs(np.array([-6, 0, 6, 12, 15, 21, 27, 33])+9)



sorted_inds = np.argsort(angles)

filenames = np.array(filenames)[sorted_inds].tolist()
# field_ranges = np.array(field_ranges)[sorted_inds].tolist()
angles = angles[sorted_inds]



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

                ax2.scatter(angles[i], f, marker='o', c=colours[i], s=245)

ax2.scatter(angles[-1], 18.12, marker='o', c=colours[-1], s=245)

aoki_s1_angles = load_matrix('/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/AOKI-DATA/AokiS1manual.csv')[:,0]
aoki_s1_freqs = load_matrix('/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/AOKI-DATA/AokiS1manual.csv')[:,1]/1e3

aoki_s2_angles = load_matrix('/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/AOKI-DATA/AokiS2manual.csv')[:,0]
aoki_s2_freqs = load_matrix('/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/AOKI-DATA/AokiS2manual.csv')[:,1]/1e3

ax2.scatter(aoki_s1_angles, aoki_s1_freqs, s=100, c='k', marker='<', zorder=-1)
ax2.scatter(aoki_s2_angles, aoki_s2_freqs, s=100, c='k', marker='>', zorder=-1)



ax1.set_ybound(0,20)
ax1.set_xbound(-2,90)
ax2.set_ybound(0,20)
ax2.set_xbound(-2,90)
ax1.set_yticks([0,5,10,15,20])
ax2.set_yticks([0,5,10,15,20])
ax1.set_yticklabels([0,5,10,15,20])
ax2.set_yticklabels([0,5,10,15,20])


publication_plot(ax1, r'$\phi$ ($\degree$)', r'Frequency (kT)')
publication_plot(ax2, r'$\theta$ ($\degree$)', r'')


ax1.annotate(r'$90 \degree\Rightarrow\mathbf{H}\parallel[010]$ ', xy=(0.435, 0.03), xycoords='axes fraction',
             ha="right", va="center", fontname='arial', fontsize=20)

ax1.annotate(r'$0 \degree\Rightarrow\mathbf{H}\parallel [001]$ ', xy=(0.435, 0.08), xycoords='axes fraction',
             ha="right", va="center", fontname='arial', fontsize=20)

ax2.annotate(r'$90 \degree\Rightarrow\mathbf{H}\parallel[100]$ ', xy=(0.435, 0.03), xycoords='axes fraction',
             ha="right", va="center", fontname='arial', fontsize=20)

ax2.annotate(r'$0 \degree\Rightarrow\mathbf{H}\parallel [001]$ ', xy=(0.435, 0.08), xycoords='axes fraction',
             ha="right", va="center", fontname='arial', fontsize=20)


ax1.annotate(r'a', xy=(0.05, 0.95), xycoords='axes fraction',
             ha="left", va="center", fontname='arial', fontweight='bold', fontsize=28)

ax2.annotate(r'b', xy=(0.05, 0.95), xycoords='axes fraction',
             ha="left", va="center", fontname='arial', fontweight='bold', fontsize=28)




# fig.add_axes([0,-55,1,1]).axis("off")
plt.tight_layout(pad=.5)
plt.savefig('/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/new_figs/freqs-v1-20kT-solid.pdf'
            , dpi=300, bbox_inches = "tight")
plt.show()




