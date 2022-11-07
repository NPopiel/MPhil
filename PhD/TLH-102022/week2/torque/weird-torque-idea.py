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

def torque_integrator(interpd_B, interpd_chi, step=3):

    locs = np.arange(0, len(interpd_B), step)

    ints = []

    avg_fields = []
    avg_chi = []

    for l1, l2 in zip(locs, locs[1:]):

        I = np.trapz(interpd_chi[l1:l2], interpd_B[l1:l2])

        ints.append(I)

        avg_fields.append(np.average(interpd_B[l1:l2]))
        avg_chi.append(np.average(interpd_chi[l1:l2]))


    B = np.array(avg_fields)
    M = np.array(ints)
    chi = np.array(avg_chi)

    y = chi*B + M


    locs = np.arange(0,len(B),step)


    avg_fields = []
    avg_tau = []

    for l1, l2 in zip(locs, locs[1:]):

        I = np.trapz(y[l1:l2], B[l1:l2])

        avg_fields.append(np.average(B[l1:l2]))
        avg_tau.append(I)

    H = np.array(avg_fields)
    T = np.array(avg_tau)

    return H, T


main_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2022_10_TLH/week1/ACMS/VT154/'


dat30 = pd.read_csv(main_path + 'Cambridge_October.051_Up.csv')
field_30_up = dat30['Field_051']
coil_X_30_up = dat30['Coil_X_051']
coil_Y_30_up = dat30['Coil_Y_051']
coil_R_30_up = dat30['Coil_R_051']

dat30 = pd.read_csv(main_path + 'Cambridge_October.051_Down.csv')
field_30_dn = dat30['Field_051']
coil_X_30_dn = dat30['Coil_X_051']
coil_Y_30_dn = dat30['Coil_Y_051']
coil_R_30_dn = dat30['Coil_R_051']

field_20_up, coil_Y_20_up = load_xy(main_path+'Cambridge_October.066.txt', '066',x_root='Field_', y_root='Coil_Y_',skiprows=12)
_, coil_X_20_up = load_xy(main_path+'Cambridge_October.066.txt', '066', x_root='Field_',y_root='Coil_X_',skiprows=12)
_, coil_R_20_up = load_xy(main_path+'Cambridge_October.066.txt', '066', x_root='Field_',y_root='Coil_R_',skiprows=12)

field_20_dn, coil_Y_20_dn = load_xy(main_path+'Cambridge_October.067.txt', '067',x_root='Field_', y_root='Coil_Y_',skiprows=12)
_, coil_X_20_dn = load_xy(main_path+'Cambridge_October.067.txt', '067', x_root='Field_',y_root='Coil_X_',skiprows=12)
_, coil_R_20_dn = load_xy(main_path+'Cambridge_October.067.txt', '067', x_root='Field_',y_root='Coil_R_',skiprows=12)

field_15_up, coil_Y_15_up = load_xy(main_path+'Cambridge_October.075.txt', '075',x_root='Field_', y_root='Coil_Y_',skiprows=8)
_, coil_X_15_up = load_xy(main_path+'Cambridge_October.075.txt', '075', x_root='Field_',y_root='Coil_X_',skiprows=8)
_, coil_R_15_up = load_xy(main_path+'Cambridge_October.075.txt', '075', x_root='Field_',y_root='Coil_R_',skiprows=8)

field_15_dn, coil_Y_15_dn = load_xy(main_path+'Cambridge_October.076.txt', '076',x_root='Field_', y_root='Coil_Y_',skiprows=12)
_, coil_X_15_dn = load_xy(main_path+'Cambridge_October.076.txt', '076', x_root='Field_',y_root='Coil_X_',skiprows=12)
_, coil_R_15_dn = load_xy(main_path+'Cambridge_October.076.txt', '076', x_root='Field_',y_root='Coil_R_',skiprows=12)


field_10_up, coil_Y_10_up = load_xy(main_path+'Cambridge_October.078.txt', '078',x_root='Field_', y_root='Coil_Y_',skiprows=14)
_, coil_X_10_up = load_xy(main_path+'Cambridge_October.078.txt', '078', x_root='Field_',y_root='Coil_X_',skiprows=14)
_, coil_R_10_up = load_xy(main_path+'Cambridge_October.078.txt', '078', x_root='Field_',y_root='Coil_R_',skiprows=14)

field_10_dn, coil_Y_10_dn = load_xy(main_path+'Cambridge_October.079.txt', '079',x_root='Field_', y_root='Coil_Y_',skiprows=12)
_, coil_X_10_dn = load_xy(main_path+'Cambridge_October.079.txt', '079', x_root='Field_',y_root='Coil_X_',skiprows=12)
_, coil_R_10_dn = load_xy(main_path+'Cambridge_October.079.txt', '079', x_root='Field_',y_root='Coil_R_',skiprows=12)

field_5_up, coil_Y_5_up = load_xy(main_path+'Cambridge_October.081.txt', '081',x_root='Field_', y_root='Coil_Y_',skiprows=14)
_, coil_X_5_up = load_xy(main_path+'Cambridge_October.081.txt', '081', x_root='Field_',y_root='Coil_X_',skiprows=14)
_, coil_R_5_up = load_xy(main_path+'Cambridge_October.081.txt', '081', x_root='Field_',y_root='Coil_R_',skiprows=14)

field_5_dn, coil_Y_5_dn = load_xy(main_path+'Cambridge_October.082.txt', '082',x_root='Field_', y_root='Coil_Y_',skiprows=12)
_, coil_X_5_dn = load_xy(main_path+'Cambridge_October.082.txt', '082', x_root='Field_',y_root='Coil_X_',skiprows=12)
_, coil_R_5_dn = load_xy(main_path+'Cambridge_October.082.txt', '082', x_root='Field_',y_root='Coil_R_',skiprows=12)

field_0_up, coil_Y_0_up = load_xy(main_path+'Cambridge_October.084.txt', '084',x_root='Field_', y_root='Coil_Y_',skiprows=14)
_, coil_X_0_up = load_xy(main_path+'Cambridge_October.084.txt', '084', x_root='Field_',y_root='Coil_X_',skiprows=14)
_, coil_R_0_up = load_xy(main_path+'Cambridge_October.084.txt', '084', x_root='Field_',y_root='Coil_R_',skiprows=14)

field_0_dn, coil_Y_0_dn = load_xy(main_path+'Cambridge_October.085.txt', '085',x_root='Field_', y_root='Coil_Y_',skiprows=12)
_, coil_X_0_dn = load_xy(main_path+'Cambridge_October.085.txt', '085', x_root='Field_',y_root='Coil_X_',skiprows=12)
_, coil_R_0_dn = load_xy(main_path+'Cambridge_October.085.txt', '085', x_root='Field_',y_root='Coil_R_',skiprows=12)

field_p5_up, coil_Y_p5_up = load_xy(main_path+'Cambridge_October.087.txt', '087',x_root='Field_', y_root='Coil_Y_',skiprows=14)
_, coil_X_p5_up = load_xy(main_path+'Cambridge_October.087.txt', '087', x_root='Field_',y_root='Coil_X_',skiprows=14)
_, coil_R_p5_up = load_xy(main_path+'Cambridge_October.087.txt', '087', x_root='Field_',y_root='Coil_R_',skiprows=14)

field_p5_dn, coil_Y_p5_dn = load_xy(main_path+'Cambridge_October.088.txt', '088',x_root='Field_', y_root='Coil_Y_',skiprows=12)
_, coil_X_p5_dn = load_xy(main_path+'Cambridge_October.088.txt', '088', x_root='Field_',y_root='Coil_X_',skiprows=12)
_, coil_R_p5_dn = load_xy(main_path+'Cambridge_October.088.txt', '088', x_root='Field_',y_root='Coil_R_',skiprows=12)

field_p10_up, coil_Y_p10_up = load_xy(main_path+'Cambridge_October.090.txt', '090',x_root='Field_', y_root='Coil_Y_',skiprows=14)
_, coil_X_p10_up = load_xy(main_path+'Cambridge_October.090.txt', '090', x_root='Field_',y_root='Coil_X_',skiprows=14)
_, coil_R_p10_up = load_xy(main_path+'Cambridge_October.090.txt', '090', x_root='Field_',y_root='Coil_R_',skiprows=14)

field_p10_dn, coil_Y_p10_dn = load_xy(main_path+'Cambridge_October.091.txt', '091',x_root='Field_', y_root='Coil_Y_',skiprows=12)
_, coil_X_p10_dn = load_xy(main_path+'Cambridge_October.091.txt', '091', x_root='Field_',y_root='Coil_X_',skiprows=12)
_, coil_R_p10_dn = load_xy(main_path+'Cambridge_October.091.txt', '091', x_root='Field_',y_root='Coil_R_',skiprows=12)


up_Xs = [coil_X_30_up, coil_X_20_up, coil_X_15_up, coil_X_10_up, coil_X_5_up, coil_X_0_up, coil_X_p5_up, coil_X_p10_up]
dn_Xs = [coil_X_30_dn, coil_X_20_dn, coil_X_15_dn, coil_X_10_dn, coil_X_5_dn, coil_X_0_dn, coil_X_p5_dn, coil_X_p10_dn]

up_Ys = [coil_Y_30_up, coil_Y_20_up, coil_Y_15_up, coil_Y_10_up, coil_Y_5_up, coil_Y_0_up, coil_Y_p5_up, coil_Y_p10_up]
dn_Ys = [coil_Y_30_dn, coil_Y_20_dn, coil_Y_15_dn, coil_Y_10_dn, coil_Y_5_dn, coil_Y_0_dn, coil_Y_p5_dn, coil_Y_p10_dn]

up_Rs = [coil_R_30_up, coil_R_20_up, coil_R_15_up, coil_R_10_up, coil_R_5_up, coil_R_0_up, coil_R_p5_up, coil_R_p10_up]
dn_Rs = [coil_R_30_dn, coil_R_20_dn, coil_R_15_dn, coil_R_10_dn, coil_R_5_dn, coil_R_0_dn, coil_R_p5_dn, coil_R_p10_dn]

up_fields = [field_30_up, field_20_up, field_15_up, field_10_up, field_5_up, field_0_up, field_p5_up, field_p10_up]
dn_fields = [field_30_dn, field_20_dn, field_15_dn, field_10_dn, field_5_dn, field_0_dn, field_p5_dn, field_p10_dn]

angles = [30, 20, 15, 10, 5, 0, -5, -10]


fig, (ax1, ax2, ax3, ax4) = MakePlot(figsize=(18, 8), nrows=1, ncols=4).create()



colours = ['#5F0F40',
           '#7D092F',
           '#9A031E',
           '#CB4721',
           '#E36414',
           '#AE5E26',
           '#945B2F',
           '#795838',
           '#44524A'
           '#0F4C5C']

window = 1971

oneT_window = 741
polyorder = 3

for i, angle in enumerate(angles):

    field_up = up_fields[i]
    X_up = up_Xs[i]
    Y_up = up_Ys[i]
    R_up = up_Rs[i]

    field_up *= -1

    ax1.plot(field_up, Y_up * 1e4, linewidth=2, c=colours[i], label = str(angle) + r' $\degree$')

    field_dn = dn_fields[i]
    X_dn = dn_Xs[i]
    Y_dn = dn_Ys[i]
    R_dn = dn_Rs[i]
    if i != 0:
        last_point = Y_dn[0]
    else:
        last_point = 13.57e-4

    Y_dn = np.flip(Y_dn)

    field_dn *=-1

    field_dn = np.flip(field_dn)

    ax1.plot(field_dn, Y_dn*1e4, linewidth=2,c=colours[i])

    interpd_field = np.linspace(0, 28, 500000)
    interpd_X_up = np.interp(interpd_field, field_dn, Y_dn*1e4)

    H, T = torque_integrator(interpd_field, interpd_X_up, step=5)

    ax3.plot(H, T, linewidth=2, c=colours[i])




publication_plot(ax1, r'$\mu_0 H$ (T)', r'$\chi^\prime$ (arb.)')#^\prime
publication_plot(ax3, r'$\mu_0 H$ (T)', r'$\tilde{\tau}$ (arb.)')#^\prime
ax1.set_xbound(0,35)


ax2.set_xbound(0,29)

colours_torque = ['#3bcfd4','#6CC0A1','#9CB16D','#CCA239','#E49B1F','#FC9305','#FA6F29','#F74A4D','#F52571','#F20094']


resistance_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2022_10_TLH/week2/'

# torque_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeSb2-VT19/torque/'

file_root = 'Cambridge_October.'
file_end = '.txt'

filenumbers_up = ['030', '033', '045'
               ]

filenumbers_down = ['031', '034', '046']

lstyles = ['solid', 'dashed']

angles_torque = np.array([120, 115, 105])

angles_torque = 180 - angles_torque

signal_VT154_up = ['X', 'X',  'X']
signal_VT154_dn = ['X', 'X',  'X']


field_root = 'Field_'
VT154_X_root = 'VT154_X_'
VT154_Y_root = 'VT154_Y_'


for i, s_num in enumerate(filenumbers_up):

    if signal_VT154_up[i] == 'X':
         VT154_root = VT154_X_root
    else:
         VT154_root = VT154_Y_root



    B, VT154 = load_xy(resistance_path + file_root + str(s_num) + file_end, str(s_num), x_root=field_root,
                       y_root=VT154_root, skiprows=9)

    VT154 *=-1

    VT154 -= VT154[0]

    ax2.plot(B, VT154*1e5, linewidth=2, c=colours_torque[i], label=str(angles_torque[i]) + r'$\degree$')

    ax4.plot(B, VT154 * 1e4 / B, linewidth=2, c=colours_torque[i], label=str(angles_torque[i]) + r'$\degree$')



    B, VT154 = load_xy(resistance_path + file_root + str(filenumbers_down[i]) + file_end, str(filenumbers_down[i]), x_root=field_root,
                           y_root=VT154_root, skiprows=9)
    VT154 *=-1
    VT154 = np.flip(VT154)
    B = np.flip(B)

    VT154 -= VT154[0]

    ax2.plot(B, VT154*1e5, linewidth=2, c=colours_torque[i])



publication_plot(ax2, r'$\mu_0 H$ (T)', r'$\tau$ (arb.)')#^\prime

publication_plot(ax4, r'$\mu_0 H$ (T)', r'$\tilde{\chi}$ (arb.)')

ax2.set_xbound(0,43)

legend = ax2.legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 24, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

legend = ax1.legend(framealpha=0, ncol=1, loc='best',
                        prop={'size': 24, 'family': 'arial'}, handlelength=0)

for line, text in zip(legend.get_lines(), legend.get_texts()):
        text.set_color(line.get_color())



ax1.annotate(r'$T = 85$ mK', xy=(0.3, 0.9), xycoords='axes fraction',
              ha="left", va="center", fontname='arial', fontsize=22)


plt.tight_layout(pad=1)
plt.savefig(main_path+'weird-VT154.png', dpi=300, bbox_inches='tight')
plt.show()



