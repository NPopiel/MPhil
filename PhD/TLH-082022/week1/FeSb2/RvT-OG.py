import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
import pandas as pd
import numpy.linalg
from tools.DataFile import DataFile
from tools.MakePlot import *
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from tools.ColorMaps import *
from tools.utils import *


main_path = '/Volumes/GoogleDrive/My Drive/Figures/TLH_08_2022/resistance_scaling/'

VT69_filename = '2020-01-29-FeSb2_v67_v68_v69_00001.dat'
VT16_filename = 'FeSb2_S19Sep_VT16_VT17_VT18.dat'
VT54_filename = '2020-01-29-FeSb2_v52_v53_v54.dat'
VT166_filename = 'VT163-VT166-VT164-BRT-cooldown.dat'
VT154_filename = 'VT154-VT155-VT156-BRT-cooldown.dat'

relevant_columns = ['Temperature (K)',
                    'Magnetic Field (Oe)',
                    'Bridge 1 Resistance (Ohms)',
                    'Bridge 1 Excitation (uA)'
                    'Bridge 2 Resistance (Ohms)',
                    'Bridge 2 Excitation (uA)']


fig, a = MakePlot(gs=True,figsize=(14,9)).create()

gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[:,0])

gs = fig.add_gridspec(5, 2, hspace= 0)
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,1])
ax4 = fig.add_subplot(gs[2,1])
ax5 = fig.add_subplot(gs[3,1])
ax6 = fig.add_subplot(gs[4,1])

dat_VT16 =load_matrix(main_path+VT16_filename)

R_VT16 = np.squeeze(np.array(dat_VT16['Bridge 1 Resistance (Ohms)']))
T_VT16 = np.squeeze(np.array(dat_VT16['Temperature (K)']))

dat_VT166 =load_matrix(main_path+VT166_filename)

R_VT166 = np.squeeze(np.array(dat_VT166['Bridge 2 Resistance (Ohms)']))
T_VT166 = np.squeeze(np.array(dat_VT166['Temperature (K)']))

dat_VT69 =load_matrix(main_path+VT69_filename)

R_VT69 = np.squeeze(np.array(dat_VT69['Bridge 3 Resistance (Ohms)']))
T_VT69 = np.squeeze(np.array(dat_VT69['Temperature (K)']))

dat_VT54 =load_matrix(main_path+VT54_filename)

R_VT54 = np.squeeze(np.array(dat_VT54['Bridge 3 Resistance (Ohms)']))
T_VT54 = np.squeeze(np.array(dat_VT54['Temperature (K)']))

dat_VT154 =load_matrix(main_path+VT154_filename)

R_VT154 = np.squeeze(np.array(dat_VT154['Bridge 1 Resistance (Ohms)']))
T_VT154 = np.squeeze(np.array(dat_VT154['Temperature (K)']))


cmap = select_discrete_cmap('bulbasaur')

ax1.scatter(T_VT16, R_VT16, label='VT16', s=40, c=cmap[0])
ax1.scatter(T_VT69, R_VT69, label='VT69', s=40, c=cmap[7])
ax1.scatter(T_VT54, R_VT54, label='VT54', s=40, c=cmap[4])
ax1.scatter(T_VT154, R_VT154, label='VT154', s=40, c=cmap[3])
ax1.scatter(T_VT166, R_VT166, label='VT166', s=40, c=cmap[5])


legend = ax1.legend(framealpha=0, ncol=1, loc='upper right',
              prop={'size': 24, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())


bbox_args = dict(boxstyle="round", fc="0.8")
arrow_args = dict(width=0.8, headlength=10, headwidth=8,color='k')



# ax1.set_ylim(1e-3, 3e7)
# ax1.set_xlim(1.5,330)

publication_plot(ax1, 'Temperature (K)', r'Resistance ($\Omega$)', x_ax_log=True, y_ax_log=True)

rs = [R_VT16, R_VT69, R_VT54, R_VT154, R_VT166]
ts = [T_VT16, T_VT69, T_VT54, T_VT154, T_VT166]
axs = [ax2, ax3, ax4, ax5, ax6]
cs = [cmap[0], cmap[7], cmap[4], cmap[3], cmap[5]]
ranges = [[(1/300,0.03), (0.11, 0.16)],
          [(1/300,0.03), (0.11, 0.16)],
          [(1/300,0.03), (0.11, 0.16)],
          [(1/300,0.03), (0.11, 0.16)],
          [(1/300,0.03), (0.11, 0.16)]]

plt.subplots_adjust(hspace=1)

yticks = [(-5,0,5), (-5,0,5), (-5,0,5), (-5,0,5), (-5,0,5)]

for i in range(len(rs)):


    y = np.log(rs[i])
    x = 1/ts[i]
    ax = axs[i]

    ax.scatter(x, y, s=40, c=cs[i])


    range1 = ranges[i][0]

    locs12 = x >= range1[0]
    locs2 =  x <= range1[1]

    locs1 = locs12 & locs2

    gap1, c1 = np.polyfit(x[locs1],y[locs1], deg=1)
    x1 = np.linspace(range1[0]-.01,range1[1]+.01,25)
    y1 = gap1*x1 + c1
    ax.plot(x1, y1, linestyle='dashed', linewidth=4.5, c='k')
    print(i)
    print(str(kelvin_2_mev(gap1).round(2)))

    # Fit Delta2, 1/T in [0.11, 0.16] for VT1
    # 1/T in [0.135, 0.2] for VT26
    # 1/T in [0.13, 0.22]

    range2 = ranges[i][1]
    locs2 = np.where(np.logical_and(x>=range2[0], x<=range2[1]))
    gap2, c2 = np.polyfit(x[locs2],y[locs2], deg=1)
    print(str(kelvin_2_mev(gap2).round(2)))
    x2 = np.linspace(range2[0]-.02,range2[1]+.02,25)
    y2 = gap2*x2 + c2
    ax.plot(x2, y2, linestyle='dashed', linewidth=4.5, c='k')

    ax.set_xlim(-.005,0.4)
    ax.set_xticks([0,0.1,0.2,0.3,0.4])
    ax.set_yticks(yticks[i])

    if i == 2:

        publication_plot(ax, r'$\frac{1}{T}$' + r' $(\mathrm{K}^{-1})$', r'')
    else:
        publication_plot(ax, r'', r'')
        ax.set_xticklabels([])

fig.text(0.51, 0.55, '$\ln(R)$ ', va='center', rotation='vertical',fontsize=24, fontname='arial')

ax2.annotate(r'$\Delta_1$ = 13.5 meV', xy=(0.025, -4.5), xycoords='data',
                 ha="left", va="center"
            , fontname='arial', fontsize=16)

ax2.annotate(r'$\Delta_2$ = 3.6 meV', xy=(0.15, 2), xycoords='data',
                 ha="left", va="center"
            , fontname='arial', fontsize=16)

ax3.annotate(r'$\Delta_1$ = 14.0 meV', xy=(0.025, -4.5), xycoords='data',
                 ha="left", va="center"
            , fontname='arial', fontsize=16)
ax3.annotate(r'$\Delta_2$ = 3.8meV', xy=(0.15, 2), xycoords='data',
                 ha="left", va="center"
            , fontname='arial', fontsize=16)

ax4.annotate(r'$\Delta_1$ = 15.8 meV', xy=(0.025, -4.5), xycoords='data',
                 ha="left", va="center"
            , fontname='arial', fontsize=16)
ax4.annotate(r'$\Delta_2$ = 2.5 meV', xy=(0.15, 2), xycoords='data',
                 ha="left", va="center"
            , fontname='arial', fontsize=16)

ax5.annotate(r'$\Delta_1$ = 14.3 meV', xy=(0.025, -4.5), xycoords='data',
                 ha="left", va="center"
            , fontname='arial', fontsize=16)

ax5.annotate(r'$\Delta_2$ = 5.8 meV', xy=(0.15, 2), xycoords='data',
                 ha="left", va="center"
            , fontname='arial', fontsize=16)

ax6.annotate(r'$\Delta_1$ = 14.8 meV', xy=(0.025, -4.5), xycoords='data',
                 ha="left", va="center"
            , fontname='arial', fontsize=16)

ax6.annotate(r'$\Delta_2$ = 6.0 meV', xy=(0.15, 2), xycoords='data',
                 ha="left", va="center"
            , fontname='arial', fontsize=16)


ax2.annotate(r'VT16', xy=(0.9, 0.2), xycoords='axes fraction',
                 ha="center", va="center"
            , fontname='arial', fontsize=20)
ax3.annotate(r'VT69', xy=(0.9, 0.2),xycoords='axes fraction',
                 ha="center", va="center"
            , fontname='arial', fontsize=20)
ax4.annotate(r'VT54', xy=(0.9, 0.2),xycoords='axes fraction',
                 ha="center", va="center"
            , fontname='arial', fontsize=20)
ax5.annotate(r'VT154', xy=(0.9, 0.2),xycoords='axes fraction',
                 ha="center", va="center"
            , fontname='arial', fontsize=20)
ax6.annotate(r'VT166', xy=(0.9, 0.2),xycoords='axes fraction',
                 ha="center", va="center"
            , fontname='arial', fontsize=20)




plt.tight_layout(pad=3)
plt.show()
# plt.savefig(main_path+'VT16+VT69-RvT.png', dpi=300, bbox_inches='tight')