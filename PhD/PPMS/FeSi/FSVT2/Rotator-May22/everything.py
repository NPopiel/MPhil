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

RvTs = [
'/Volumes/GoogleDrive/My Drive/Data/FeSi/Data/FSVT2-BRT-rotator_cooldown-good.dat',
'/Volumes/GoogleDrive/My Drive/Data/FeSi/Data/FSVT2-rotator-warmup.dat']

ivs = ['/Volumes/GoogleDrive/My Drive/Data/FeSi/Data/FSVT2_IV_1p8K_0T.dat',
'/Volumes/GoogleDrive/My Drive/Data/FeSi/Data/FSVT2_IV_1p8K_14T.dat',
'/Volumes/GoogleDrive/My Drive/Data/FeSi/Data/FSVT2_IV_1p8K_14T-90deg.dat',
'/Volumes/GoogleDrive/My Drive/Data/FeSi/Data/FSVT2_IV_1p8K_0Tnum2.dat']

iv_save_names = ['FSVT2-0T-1p8K.csv',
                 'FSVT2-14T-1p8K.csv',
                 'FSVT2-neg14T-1p8K.csv',
                 'FSVT2-0Tb-1p8K.csv']

field_sweeps = ['/Volumes/GoogleDrive/My Drive/Data/FeSi/Data/FSVT2_3p1mA_1.8K_FS-0deg.dat',
'/Volumes/GoogleDrive/My Drive/Data/FeSi/Data/FSVT2_3p1mA_1.8K_FS-15deg.dat',
'/Volumes/GoogleDrive/My Drive/Data/FeSi/Data/FSVT2_3p1mA_1.8K_FS-30deg.dat',
'/Volumes/GoogleDrive/My Drive/Data/FeSi/Data/FSVT2_3p1mA_1.8K_FS-45deg.dat',
'/Volumes/GoogleDrive/My Drive/Data/FeSi/Data/FSVT2_3p1mA_1.8K_FS-60deg.dat',
'/Volumes/GoogleDrive/My Drive/Data/FeSi/Data/FSVT2_3p1mA_1.8K_FS-75deg.dat',
'/Volumes/GoogleDrive/My Drive/Data/FeSi/Data/FSVT2_3p1mA_1.8K_FS-90deg.dat']

angles = [0, 15, 30, 45, 60, 75, 90]

# Plot the cooldown/warmup

labels_RvT = ['Cooling','Warming']
colors = ['#5BC0EB', '#D90368']

fig, a = MakePlot(gs=True).create()

gs = fig.add_gridspec(2,3)
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1:3])
ax3 = fig.add_subplot(gs[1, 1:3])

ax = ax1

for i, sweep in enumerate(RvTs):

    R = np.abs(load_matrix(sweep)['Bridge 2 Resistance (Ohms)'])
    T = load_matrix(sweep)['Temperature (K)']

    ax.plot(T, R, lw=2,label=labels_RvT[i], c=colors[i])


publication_plot(ax, 'Temperature (K)', 'Resistance ($\Omega$)', x_ax_log=True, y_ax_log=True)

ax.annotate(r'FeSi VT2', xy=(0.2, 0.16), xycoords='axes fraction',
            ha="center", va="center", fontname='arial', fontsize=20)
ax.annotate(r'$\mu_0H$ = 0 T', xy=(0.2, 0.1), xycoords='axes fraction',
            ha="center", va="center", fontname='arial', fontsize=20)

legend = ax.legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 20, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())





labels_iv = ['0 T',
             '14 T',
             '-14 T',
             '0 T']

cs = ['#087F8C',
      '#F75C03',
      '#AD91A3',
      '#9E0031']

ax=ax2

for i, iv_dat in enumerate(ivs):

    R = load_matrix(iv_dat)['Bridge 2 Resistance (Ohms)']

    I = load_matrix(iv_dat)['Bridge 2 Excitation (uA)']

    V = R*I/1e6
  
    arr = np.array([I, R]).T

    np.savetxt('/Volumes/GoogleDrive/My Drive/Figures/IV-step/'+iv_save_names[i],
               arr, delimiter=',')

    ax.plot(I/1e3, V, lw=2, label=labels_iv[i], c=cs[i])

publication_plot(ax, 'Current (mA)', 'Voltage (V)')

ax.annotate(r'$T$ = 1.8 K', xy=(0.9, 0.22), xycoords='axes fraction',
    ha="center", va="center", fontname='arial', fontsize=20)


handles, labels = ax.get_legend_handles_labels()

legend = ax.legend(handles[::-1], labels[::-1],framealpha=0, ncol=1, loc='best',
              prop={'size': 20, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())



colors = ['#F94144',
          '#F3722C',
          '#F9C74F',
          '#90BE6D',
          '#43AA8B',
          '#577590',
          '#277DA1']

ax = ax3

for i, sweep in enumerate(field_sweeps):

    R = load_matrix(sweep)['Bridge 2 Resistance (Ohms)'][1:-2]
    B = load_matrix(sweep)['Magnetic Field (Oe)'][1:-2]/1e4

    ax.plot(B, R, lw=2,label=str(angles[i])+r'$\degree$', c=colors[i])


publication_plot(ax, 'Magnetic Field (T)', 'Resistance ($\Omega$)')

ax.annotate(r'$T$ = 1.8 K', xy=(0.1, 0.9), xycoords='axes fraction',
            ha="center", va="center", fontname='arial', fontsize=20)
ax.annotate(r'$I$ = 3.1 mA', xy=(0.1, 0.76), xycoords='axes fraction',
    ha="center", va="center", fontname='arial', fontsize=20)


handles, labels = ax.get_legend_handles_labels()

legend = ax.legend(handles[::-1], labels[::-1],framealpha=0, ncol=2, loc='best',
              prop={'size': 20, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

ax1.text(-0.08, 1.05, 'A', transform=ax1.transAxes,
      fontsize=25, fontweight='bold', fontname='arial', va='top', ha='right')
ax2.text(-0.05, 1.12, 'B', transform=ax2.transAxes,
      fontsize=25, fontweight='bold', fontname='arial', va='top', ha='right')
ax3.text(-0.05, 1.12, 'C', transform=ax3.transAxes,
      fontsize=25, fontweight='bold', fontname='arial', va='top', ha='right')

plt.tight_layout(pad=.5)

plt.savefig('/Volumes/GoogleDrive/My Drive/FirstYearReport/Figures/FSVT2-rotator.png', dpi=300)
plt.show()






