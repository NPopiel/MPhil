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


colours = ['#D8A47F',
           #'#EF8354',
           '#EF675F',
           '#EE4B6A',
           '#DF3B57',
           #'#AB495E',
           '#775665',
           '#43646C',
           '#0F7173']

torque_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeGa3-VLS1/'

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

    if tau[-1] < 0:
        tau *=-1

    tau -= tau[0]

    return field, tau



sweeps = [('0.29K_-6.66deg_sweep011_up.csv','0.29K_-6.66deg_sweep010_up.csv'),
          ('0.29K_-3.33deg_sweep017_down.csv','0.29K_-3.33deg_sweep018_down.csv'),
          ('0.29K_6.66deg_sweep022_up.csv','0.29K_6.66deg_sweep023_down.csv'),
          ('0.29K_13.33deg_sweep026_up.csv','0.29K_13.33deg_sweep027_down.csv'),
          ('0.29K_26deg_sweep040_down.csv','0.29K_26deg_sweep041_down.csv'),
          ('0.29K_33.33deg_sweep046_down.csv','0.29K_33.33deg_sweep048_up.csv')]

lstyles = ['solid', 'dashed']

angles = [-7,-3.5,7,14,28,35]

xs, ys = [], []

grads = []

for i, s_name in enumerate(sweeps):


    field, tau = load_torque(torque_path, s_name[0])

    if i < 3:
        tau*=100/14.14

    ax1.annotate(str(angles[i]) + r'$\degree$', xy=(0.89, 0.05 + i * 0.95/6), xycoords='axes fraction',
                 fontname='arial', fontsize=24, color=colours[i])

    ax2.plot(field**2, tau*1e5, linewidth=2, c=colours[i], label=str(angles[i])+r'$\degree$')

    fit_locs1 = field > 17
    fit_locs2 = field < 20

    fit_locs = fit_locs1 & fit_locs2

    slope = np.polyfit(field[fit_locs]**2, tau[fit_locs]*1e5, 1)[0]

    ax3.scatter(angles[i], slope*1e3, c=colours[i], s=275)

    ax1.plot(field, tau*1e5, linewidth=2, c=colours[i], label=str(angles[i])+r'$\degree$')

    field, tau = load_torque(torque_path, s_name[1])
    if i < 3:
        tau*=100/14.14



    ax1.plot(field, tau*1e5, linewidth=2, c=colours[i])




publication_plot(ax1, r'$\mu_0H$ (T)', r'$\tau$ (arb.)')
publication_plot(ax2, r'$(\mu_0H)^2$ (T$^2$)', r'$\tau$ (arb.)')
publication_plot(ax3, r'$\theta$ ($\degree$)', r'$\tau^\prime$ (arb.)')


# ax1.set_ybound(-6,6)
ax1.set_xbound(-0.2, 40)

# handles, labels = ax1.get_legend_handles_labels()
#
# legend = ax1.legend(handles, labels, framealpha=0, ncol=len(angles), loc='best',
#                     prop={'size': 18, 'family': 'arial'},
#                     handlelength=0, labelspacing=2.6)
#
# for line,text in zip(legend.get_lines(), legend.get_texts()):
#     text.set_color(line.get_color())

plt.tight_layout(pad=1)

plt.savefig('/Volumes/GoogleDrive/My Drive/Figures/TLH_08_2022/FeGa3-VLS1/raw_data-og.pdf', dpi=300, bbox_inches='tight')

plt.show()




