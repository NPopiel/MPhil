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


colours = ['#001219',
           '#005F73',
           '#0A9396',
           '#94D2BD',
           '#E9D8A6',
           '#EE9B00',
           '#CA6702',
           '#BB3E03',
           '#AE2012',
           '#9B2226']

torque_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeGa3-VLS1/'

fig, a = MakePlot(figsize=(12, 12), gs=True).create()
gs = fig.add_gridspec(2, 1)

ax1 = fig.add_subplot(gs[0,:])
ax2 = fig.add_subplot(gs[1,:])


sweeps = ['0.29K_-6.66deg_sweep010_up.csv',
          '0.29K_-3.33deg_sweep017_down.csv',
          '0.29K_0deg_sweep019_up.csv',
          '0.29K_6.66deg_sweep022_up.csv',
          '0.29K_10deg_sweep024_down.csv',
          '0.29K_13.33deg_sweep026_up.csv',
          '0.29K_20deg_sweep033_down.csv',
          '0.29K_26deg_sweep040_down.csv']

lstyles = ['solid', 'dashed', ]

angles = [-6.66, -3.33, 0, 6.66, 10, 13.33, 20, 26.66]

for i, s_name in enumerate(sweeps):


    torque_dat = load_matrix(torque_path + s_name)
    field = torque_dat[:, 0]
    tau = torque_dat[:, 1]

    if field[0] > field[-1]:
        field = np.flip(field)
        tau = np.flip(tau)

    # if tau[-1] < 0:
    #     tau *=-1

    tau -= tau[0]

    if i > 5:

        tau*=14.14/100

    ax1.plot(field, tau*1e5, linewidth=2, c=colours[i], label=str(angles[i])+r'$\degree$')
    ax2.plot(field ** 2, tau * 1e5, linewidth=2, c=colours[i], label=str(angles[i]) + r'$\degree$')

    # ax1.annotate(str(angles[i]) + r'$\degree$', xy=(1.03, i * 0.1125), xycoords='axes fraction',
    #              fontname='arial', fontsize=16, color=colours[i])

publication_plot(ax1, r'$\mu_0H$ (T)', r'$\tau$ (arb.)')
publication_plot(ax2, r'$(\mu_0H)^2$ (T$^2$)', r'$\tau$ (arb.)')

handles, labels = ax1.get_legend_handles_labels()

legend = ax1.legend(handles, labels, framealpha=0, ncol=len(angles), loc='best',
                    prop={'size': 18, 'family': 'arial'},
                    handlelength=0, labelspacing=2.6)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())


plt.show()




