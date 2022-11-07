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


fig, a = MakePlot(figsize=(10, 14), gs=True).create()
gs = fig.add_gridspec(5,2)

ax4 = fig.add_subplot(gs[:2,:])
ax1 = fig.add_subplot(gs[2:,0])
ax2 = fig.add_subplot(gs[2:,1])


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

    ax1.plot(field_up, X_up * 1e4, linewidth=2, c=colours[i], label = str(angle) + r' $\degree$')

    field_dn = dn_fields[i]
    X_dn = dn_Xs[i]
    Y_dn = dn_Ys[i]
    R_dn = dn_Rs[i]
    if i != 0:
        last_point = X_dn[0]
    else:
        last_point = 13.57e-4

    X_dn = np.flip(X_dn)

    field_dn *=-1

    field_dn = np.flip(field_dn)

    ax1.plot(field_dn, X_dn*1e4, linewidth=2,c=colours[i])

    ax1.annotate(str(angles[i]) + r'$\degree$', xy=(33, last_point*1e4), xycoords='data', ha='right', va='center',
                 fontname='arial', fontsize=22, color=colours[i])



publication_plot(ax1, r'$\mu_0 H$ (T)', r'$\chi^\prime$ (arb.)')#^\prime
ax1.set_xbound(0,35)
ax1.set_ybound(-.75, 14.5)

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



    B, VT154 = load_xy(resistance_path + file_root + str(filenumbers_down[i]) + file_end, str(filenumbers_down[i]), x_root=field_root,
                           y_root=VT154_root, skiprows=9)
    VT154 *=-1
    VT154 = np.flip(VT154)
    B = np.flip(B)

    VT154 -= VT154[0]

    ax2.plot(B, VT154*1e5, linewidth=2, c=colours_torque[i])



publication_plot(ax2, r'$\mu_0 H$ (T)', r'$\tau$ (arb.)')#^\prime

ax2.set_xbound(0,43)

legend = ax2.legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 24, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())


import matplotlib as mpl

# read image file
with mpl.cbook.get_sample_data('/Volumes/GoogleDrive/Shared drives/AGE_JRF/Magnet times/TLH_SCM4_Jul22/Figs/78deg-removebg-preview.png') as file:
    arr_image = plt.imread(file, format='png')

# Draw image
axin = ax1.inset_axes([0.1, 0.69, 0.25, 0.25])  # create new inset axes in data coordinates
axin.imshow(arr_image)
axin.axis('off')

axin.annotate(r'$\theta$', xy=(0.32, 0.4), xycoords='axes fraction',
              ha="left", va="center", fontname='arial', fontsize=20)

axin.annotate(r'$a$', xy=(0, 1.2), xycoords='axes fraction',
              ha="center", va="center", fontname='arial', fontsize=20)

axin.annotate(r'$c$', xy=(1.12, 0), xycoords='axes fraction',
              ha="center", va="center", fontname='arial', fontsize=20)


ax1.annotate(r'$T = 85$ mK', xy=(0.3, 0.9), xycoords='axes fraction',
              ha="left", va="center", fontname='arial', fontsize=22)

locs_max = np.array([16.3, 19.9, 21.2, 24.0])
angles_r2_r3 = np.array([30, 20, 15, 10])

r2r3_fields = np.array([40, 37, 25, 23, 21, 20, 19, 18, 28, 30, 30.5, 35, 36, 40])

r2r3_angles = np.array([2,7, 12, 16, 19, 21, 30, 37, 56, 60, 64, 68, 72, 76])

r1r2_fields = np.array([32.5, 25, 11, 10.5, 8, 7.6, 8, 8.5, 10, 10, 12, 15, 22])
r1r2_angles = np.array([2, 7, 12, 16, 28, 38, 45, 55, 60, 70, 75, 82, 90])

ax4.scatter(r2r3_angles, r2r3_fields, c='k', marker = 'x', s=200)
ax4.scatter(r1r2_angles, r1r2_fields, c='k', marker = 'd', s=200)

r1r2_ac_fields = np.array([8.6, 11.6, 11.9, 10.5, 13.5])


for i, l in enumerate(locs_max):

    ax4.scatter(angles_r2_r3[i], l, s=200, c=colours[i])

for i in range(len(r1r2_ac_fields)):
    ax4.scatter(angles[i], r1r2_ac_fields[i], c=colours[i], s=200, marker='D')


torque_angles = np.array([120, 115, 105])

torque_angles = 180 - torque_angles
r1r2_torque_fields = np.array([14.8, 16, 28.5])
r2r3_torque_fields = np.array([20.2, 21.5, 36.5])

for i in range(len(torque_angles)):
    ax4.scatter(torque_angles[i], r1r2_torque_fields[i], facecolor='none', edgecolor=colours_torque[i], s=200, marker='D')
    ax4.scatter(torque_angles[i], r2r3_torque_fields[i], facecolor='none', edgecolor=colours_torque[i], s=200)




publication_plot(ax4, r'$\theta (\degree)$', r'$\mu_0 H$ (T)')#^\prime
ax4.set_xbound(-1,91)
ax4.set_ybound(0, 45)

ax4.set_yticks([0,10,20,30,40])



ax1.annotate('Up',xy=(25.9,10.7), xytext=(25.1, 9.2), xycoords='data', ha='left', va='center',
                 fontname='arial', fontsize=22, color=colours[0],
             arrowprops=dict(facecolor=colours[0], edgecolor=colours[0],width=1, headwidth=5))

ax1.annotate('Down', xy=(18.8, 6.9),xytext=(20, 8.4), xycoords='data', ha='right', va='center',
                 fontname='arial', fontsize=22, color=colours[0],
             arrowprops=dict(facecolor=colours[0],edgecolor=colours[0], width=1, headwidth=5))

plt.tight_layout(pad=1)

plt.savefig(main_path+'complete-VT154.png', dpi=300, bbox_inches='tight')
plt.show()



