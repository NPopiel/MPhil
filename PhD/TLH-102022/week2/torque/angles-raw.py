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


def load_xy(fileandpath, filenumber, up=True,y_root='Coil_X_', x_root = 'Field_', skiprows=9, space_x=False, space_y=False):
    dat  = pd.read_csv(fileandpath, delimiter='\t', skiprows=skiprows)

    if space_x:
        x = np.array(dat[x_root + str(filenumber) + ' '])
    else:
        x = np.array(dat[x_root + str(filenumber)])

    if space_y:
        y = np.array(dat[y_root + str(filenumber) + ' '])
    else:
        y = np.array(dat[y_root + str(filenumber)])

    if not up:
        x = np.flip(x)
        y = np.flip(y)
    return x, y

def load_xy(fileandpath, filenumber, up=True,y_root='Coil_X_', x_root = 'Field_', skiprows=9, space_x=False, space_y=False):
    dat  = pd.read_csv(fileandpath, delimiter='\t', skiprows=skiprows)

    if space_x:
        x = np.array(dat[x_root + str(filenumber) + ' '])
    else:
        x = np.array(dat[x_root + str(filenumber)])

    if space_y:
        y = np.array(dat[y_root + str(filenumber) + ' '])
    else:
        y = np.array(dat[y_root + str(filenumber)])

    if not up:
        x = np.flip(x)
        y = np.flip(y)
    return x, y

resistance_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2022_10_TLH/week2/'

# torque_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeSb2-VT19/torque/'

fig, a = MakePlot(figsize=(12, 8), gs=True).create()
gs = fig.add_gridspec(1,2)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

Vxx_root = 'Vxx/'
Vxy_root = 'Vxy/'

file_root = 'Cambridge_October.'
file_end = '.txt'

filenumbers_up = ['023'
               ]

filenumbers_down = [None]

lstyles = ['solid', 'dashed']

angles = [120]

signal_VT154_up = ['X']
signal_VT154_dn = [None]

signal_VT15_up = ['Y']
signal_VT15_dn = [None]


fig, a = MakePlot(figsize=(16, 8), gs=True).create()
gs = fig.add_gridspec(1,2)

ax1 = fig.add_subplot(gs[:,0])
ax2 = fig.add_subplot(gs[:,1])





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


window = 1971

oneT_window = 741
polyorder = 3




field_root = 'Field_'
VT154_X_root = 'VT154_X_'
VT154_Y_root = 'VT154_Y_'
VT15_X_root = 'VT15_X_'
VT15_Y_root = 'VT15_Y_'

for i, s_num in enumerate(filenumbers_up):

    if signal_VT154_up[i] == 'X':
         VT154_root = VT154_X_root
    else:
         VT154_root = VT154_Y_root

    if signal_VT15_up[i] == 'X':
        VT15_root = VT15_X_root
    else:
        VT15_root = VT15_Y_root

    B, VT154 = load_xy(resistance_path + file_root + str(s_num) + file_end, str(s_num), x_root=field_root,
                       y_root=VT154_root, skiprows=9)
    _, VT15 = load_xy(resistance_path + file_root + str(s_num) + file_end, str(s_num), x_root=field_root,
                      y_root=VT15_root, skiprows=9)

    ax1.plot(B, VT154, linewidth=2, c=colours[i], label=str(angles[i]) + r'$\degree$')
    ax2.plot(B, VT15, linewidth=2, c=colours[i], label=str(angles[i]) + r'$\degree$')

    if filenumbers_down[i] is not None:

        if signal_VT154_dn[i] == 'X':
            VT154_root = VT154_X_root
        else:
            VT154_root = VT154_Y_root

        if signal_VT15_dn[i] == 'X':
            VT15_root = VT15_X_root
        else:
            VT15_root = VT15_Y_root

        B, VT154 = load_xy(resistance_path + file_root + str(filenumbers_down[i]) + file_end, str(filenumbers_down[i]), x_root=field_root,
                           y_root=VT154_root, skiprows=9)
        _, VT15 = load_xy(resistance_path + file_root + str(filenumbers_down[i]) + file_end, str(filenumbers_down[i]), x_root=field_root,
                          y_root=VT15_root, skiprows=9)

        ax1.plot(B, VT154, linewidth=2, c=colours[i], linestyle='dashed')
        ax2.plot(B, VT15, linewidth=2, c=colours[i],  linestyle='dashed')







publication_plot(ax1, r'$\mu_0 H$ (T)', r'VT154 Torque (arb.)')#^\prime
publication_plot(ax2, r'$\mu_0 H$ (T)', r'VT15 Torque (arb.)')#^\prime


ax1.set_xbound(0,42)
# ax1.set_ybound(-.75, 14.5)

ax2.set_xbound(0,42)
# ax2.set_ybound(-1.25, 13)





# legend = ax1.legend(framealpha=0, ncol=1, loc='best',
#               prop={'size': 24, 'family': 'arial'}, handlelength=0)
#
# for line,text in zip(legend.get_lines(), legend.get_texts()):
#     text.set_color(line.get_color())



plt.tight_layout(pad=1)

# plt.savefig(main_path+'X-r2r3-fig.png', dpi=300, bbox_inches='tight')
plt.show()



