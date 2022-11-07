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


colours = ['#042A2B',
           '#1B4C50',
           '#316E75',
           '#5EB1BF',
           '#96CFDB',
           '#CDEDF6',
           '#DEB49E',
           '#EF7B45',
           '#E46136',
           '#D84727']

torque_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeSb2-VT16/'

fig, a = MakePlot(figsize=(16, 12), gs=True).create()
gs = fig.add_gridspec(2, 2)

ax1 = fig.add_subplot(gs[:,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,1])

def load_torque(path, name):

    torque_dat = load_matrix(path + name)
    field = torque_dat[:, 0]
    tau = torque_dat[:, 1]

    if field[0] > field[-1]:
        field = np.flip(field)
        tau = np.flip(tau)

    # if tau[-1] < 0:
    #     tau *=-1

    tau -= tau[0]

    return field, tau



sweeps = [(None,'0.29K_96.66deg_sweep141_up.csv'),
          ('0.29K_90deg_sweep142_down.csv','0.29K_90deg_sweep143_down.csv'),
          ('0.29K_83.33deg_sweep144_up.csv','0.29K_83.33deg_sweep145_up.csv'),
          ('0.29K_76.66deg_sweep146_down.csv',None),
          ('0.29K_70deg_sweep148_up.csv','0.29K_70deg_sweep149_up.csv'),
          ('0.29K_56.66deg_sweep152_up.csv','0.29K_56.66deg_sweep153_down.csv'),
          ('0.29K_43.33deg_sweep154_up.csv','0.29K_43.33deg_sweep155_up.csv'),
          ('0.29K_30deg_sweep156_up.csv','0.29K_30deg_sweep157_up.csv'),
          ('0.29K_16.66deg_sweep158_up.csv','0.29K_16.66deg_sweep159_down.csv'),
          ('0.29K_3.33deg_sweep160_up.csv',None)]

lstyles = ['solid', 'dashed']

angles = [96.66, 90, 83.33, 76.66, 70, 56.66, 43.33, 30, 16.66, 3.33]

xs, ys = [], []

grads = []

for i, s_name in enumerate(sweeps):

    if s_name[0] is not None:
        field, tau = load_torque(torque_path, s_name[0])

        tau *=-1

        ax1.annotate(str(angles[i]) + r'$\degree$', xy=(0.15, 0.95 - i * 0.65/10), xycoords='axes fraction',
                     fontname='arial', fontsize=24, color=colours[i])

        ax2.plot(field**2, tau*1e5, linewidth=2, c=colours[i], label=str(angles[i])+r'$\degree$')

        fit_locs1 = field > 17
        fit_locs2 = field < 20

        fit_locs = fit_locs1 & fit_locs2

        slope = np.polyfit(field[fit_locs]**2, tau[fit_locs]*1e5, 1)[0]

        ax3.scatter(angles[i], slope*1e3, c=colours[i], s=275)

        ax1.plot(field, tau*1e5, linewidth=2, c=colours[i], label=str(angles[i])+r'$\degree$')

    if s_name[1] is not None:
        field, tau = load_torque(torque_path, s_name[1])

        tau *=-1



        ax2.plot(field ** 2, tau * 1e5, linewidth=2, c=colours[i], label=str(angles[i]) + r'$\degree$')


        ax1.plot(field, tau * 1e5, linewidth=2, c=colours[i])

        if i == 0:
            ax1.annotate(str(angles[i]) + r'$\degree$', xy=(0.15, 0.95 - i * 0.65 / 10), xycoords='axes fraction',
                         fontname='arial', fontsize=24, color=colours[i])
            fit_locs1 = field > 17
            fit_locs2 = field < 20

            fit_locs = fit_locs1 & fit_locs2

            slope = np.polyfit(field[fit_locs] ** 2, tau[fit_locs] * 1e5, 1)[0]

            ax3.scatter(angles[i], slope * 1e3, c=colours[i], s=275)






publication_plot(ax1, r'$\mu_0H$ (T)', r'$\tau$ (arb.)')
publication_plot(ax2, r'$(\mu_0H)^2$ (T$^2$)', r'$\tau$ (arb.)')
publication_plot(ax3, r'$\theta$ ($\degree$)', r'$\tau^\prime$ (arb.)')


# ax1.set_ybound(-6,6)
# ax1.set_xbound(-0.2, 40)

# handles, labels = ax1.get_legend_handles_labels()
#
# legend = ax1.legend(handles, labels, framealpha=0, ncol=len(angles), loc='best',
#                     prop={'size': 18, 'family': 'arial'},
#                     handlelength=0, labelspacing=2.6)
#
# for line,text in zip(legend.get_lines(), legend.get_texts()):
#     text.set_color(line.get_color())

plt.tight_layout(pad=1)

plt.savefig('/Volumes/GoogleDrive/My Drive/Figures/TLH_08_2022/VT16/raw_50V.pdf', dpi=300, bbox_inches='tight')

plt.show()




