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


colours = ['#5F0F40',
           '#6E0C38',
           '#7D092F',
           '#9A031E',
           '#B32520',
           '#CB4721',
           '#FB8B24',
           '#E36414',
           '#AE5E26',
           '#795838',
           '#44524A',
           '#2A4F53',
           '#0F4C5C',
           '#255C6B']

torque_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeSb2-VT54/resistance/'

fig, a = MakePlot(figsize=(16, 12), gs=True).create()
gs = fig.add_gridspec(2, 2)

ax1 = fig.add_subplot(gs[:,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,1])


axin = ax1.inset_axes([0.13, 0.1, 0.4, .2])

def load_torque(path, name, R=False):

    torque_dat = load_matrix(path + name)
    field = torque_dat[:, 0]
    tau = torque_dat[:, 1]

    if field[0] > field[-1]:
        field = np.flip(field)
        tau = np.flip(tau)

    # if tau[-1] < 0:
    #     tau *=-1

    if not R:
        tau -= tau[0]

    return field, tau



sweeps = [('0.29K_-4deg_sweep124_up.csv','0.29K_-4deg_sweep125_down.csv'),
          ('0.29K_8deg_sweep122_up.csv','0.29K_8deg_sweep123_down.csv'),
          ('0.29K_20deg_sweep120_down.csv','0.29K_20deg_sweep121_down.csv'),
          ('0.29K_32deg_sweep118_up.csv','0.29K_32deg_sweep119_up.csv'),
          ('0.29K_40deg_sweep116_down.csv','0.29K_40deg_sweep117_down.csv'),
          ('0.29K_50deg_sweep114_up.csv','0.29K_50deg_sweep115_down.csv'),
          ('0.29K_56.66deg_sweep112_down.csv','0.29K_56.66deg_sweep113_up.csv'),
          ('0.29K_63.33deg_sweep110_down.csv','0.29K_63.33deg_sweep111_down.csv'),
          ('0.29K_70deg_sweep107_up.csv','0.29K_70deg_sweep108_down.csv'),
          ('0.29K_76.66deg_sweep103_up.csv','0.29K_76.66deg_sweep104_down.csv'),
          ('0.29K_83.33deg_sweep101_up.csv','0.29K_83.33deg_sweep102_up.csv'),
          ('0.29K_90deg_sweep099_up.csv','0.29K_90deg_sweep100_up.csv'),
          ('0.29K_96.66deg_sweep096_up.csv','0.29K_96.66deg_sweep097_down.csv'),
          ('0.29K_103.33deg_sweep094_up.csv','0.29K_103.33deg_sweep095_down.csv')]

resistance_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeSb2-VT54/resistance/'

lstyles = ['solid', 'dashed']

angles = [-4, 8, 20, 32, 40, 50, 57, 64, 71, 78, 85, 92, 99, 106]

xs, ys = [], []

grads = []

for i, s_name in enumerate(sweeps):


    field, tau = load_torque(torque_path, s_name[0])

    locs1 = field > 12
    locs2 = field < 28

    locs = locs1 & locs2

    field = field[locs]
    tau = tau[locs]

    # tau *=-1

    ax1.annotate(str(angles[i]) + r'$\degree$', xy=(0.89, 0.05 + i * 0.95/14), xycoords='axes fraction',
                 fontname='arial', fontsize=24, color=colours[i])

    ax2.plot(field**2, tau*1e5, linewidth=2, c=colours[i], label=str(angles[i])+r'$\degree$')

    fit_locs1 = field > 17
    fit_locs2 = field < 20

    fit_locs = fit_locs1 & fit_locs2

    slope = np.polyfit(field[fit_locs]**2, tau[fit_locs]*1e5, 1)[0]

    ax3.scatter(angles[i], slope*1e3, c=colours[i], s=275)

    ax1.plot(field, tau*1e5, linewidth=2, c=colours[i], label=str(angles[i])+r'$\degree$')

    field, tau = load_torque(torque_path, s_name[1])

    locs1 = field > 12
    locs2 = field < 28

    locs = locs1 & locs2

    field = field[locs]
    tau = tau[locs]

    # tau*=-1

    ax1.plot(field, tau*1e5, linewidth=2, c=colours[i])

    field, V = load_torque(resistance_path, s_name[0], R=True)

    axin.plot(field, np.abs(V/1e-5), linewidth=2, c=colours[i])




publication_plot(ax1, r'$\mu_0H$ (T)', r'$\tau$ (arb.)')
publication_plot(ax2, r'$(\mu_0H)^2$ (T$^2$)', r'$\tau$ (arb.)')
publication_plot(ax3, r'$\theta$ ($\degree$)', r'$\tau^\prime$ (arb.)')

publication_plot(axin, r'$\mu_0H$ (T)', r'$R$ ($\Omega$)', label_fontsize=18, tick_fontsize=16)

axin.set_ybound(0, 520)


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

plt.savefig('/Volumes/GoogleDrive/My Drive/Figures/TLH_08_2022/VT54/raw_data-VT54-tohoku-esque.pdf', dpi=300, bbox_inches='tight')

# plt.show()




