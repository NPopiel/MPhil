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


def tesla_window(x, tesla_wind):
    # print(2 * int(round(0.5 * tesla_wind / x[1] - x[0])) + 1)
    return 2 * int(round(0.5 * tesla_wind / np.abs(x[1] - x[0]))) + 1



torque_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week2_torque/VT69/'

conversion_factor = 8811.49035935598 #pF/V
const = 1.5878 # pF

fig, a = MakePlot(figsize=(12, 16), gs=True).create()
gs = fig.add_gridspec(1, 2)

ax1 = fig.add_subplot(gs[:,0])

ax3 = fig.add_subplot(gs[:, 1])

def load_tau(dat, filenumber, upper_threshold=41.5,lower_threshold=.1):

    T = np.array(dat['VT69_X_'+str(filenumber)])
    B = np.array(dat['Field_'+str(filenumber)])


    locs1 = B > lower_threshold
    locs2 = B < upper_threshold

    locs = locs1 & locs2

    T = T[locs]
    B = B[locs]

    return T, B


sweeps = [#'0.35K_90deg_sweep004_down.csv', # ON AH
    'Cambridge_2022_Sep.0', # ON AH
    '0.35K_70deg_sweep009_down.csv', # ON AH,
    '0.35K_50deg_sweep011_up.csv' # AH
          ]

lstyles = ['solid']

angles = [70, 70, 50]

for i, s_name in enumerate(sweeps):


    torque_dat = load_matrix(torque_path + s_name)
    field = torque_dat[:, 0]
    tau = torque_dat[:, 1]

    # locs = np.argwhere(np.isnan(tau))
    #
    # tau = tau[~locs]
    # field = field[~locs]

    if field[0] > field[-1]:
        field = np.flip(field)
        tau = np.flip(tau)

    if tau[-1] < 0:
        tau *= -1



    if tau[0] != np.nan:
        tau -= tau[0]

    ax1.plot(field, tau, linewidth=2, c=colours[i], label=str(angles[i])+r'$\degree$')

    ax3.plot(field ** 2, tau, linewidth=2, c=colours[i], label=str(angles[i]) + r'$\degree$')


ax3.annotate('VT69',xy=(0.1, 0.95), xycoords='axes fraction',
             ha="left", va="center", fontname='arial', fontweight='bold',fontsize=26)

publication_plot(ax1, r'$\mu_0H$ (T)', r'$\tau$ (pF)')
publication_plot(ax3, r'$(\mu_0H)^2$ (T$^2$)', r'$\tau$ (pF)')

handles, labels = ax1.get_legend_handles_labels()

legend = ax1.legend(handles, labels, framealpha=0, ncol=len(angles), loc='best',
                    prop={'size': 18, 'family': 'arial'},
                    handlelength=0, labelspacing=2.6)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())


plt.tight_layout(pad=1)

plt.show()




