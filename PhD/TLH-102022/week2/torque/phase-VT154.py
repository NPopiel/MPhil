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


fig, a = MakePlot(figsize=(8, 6), gs=True).create()
gs = fig.add_gridspec(1,1)


ax4 = fig.add_subplot(gs[0,0])


# colours = ['#001219',
#            '#005F73',
#            '#0A9396',
#            '#94D2BD',
#            '#E9D8A6',
#            '#EE9B00',
#            '#CA6702',
#            '#BB3E03',
#            '#AE2012',
#            '#9B2226']

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

colours = ['#3bcfd4','#6CC0A1','#9CB16D','#CCA239','#E49B1F','#FC9305','#FA6F29','#F74A4D','#F52571','#F20094']

for i in range(len(torque_angles)):
    ax4.scatter(torque_angles[i], r1r2_torque_fields[i], facecolor='none', edgecolor=colours[i], s=200, marker='D')
    ax4.scatter(torque_angles[i], r2r3_torque_fields[i], facecolor='none', edgecolor=colours[i], s=200)




publication_plot(ax4, r'$\theta (\degree)$', r'$\mu_0 H$ (T)')#^\prime
ax4.set_xbound(-1,91)
ax4.set_ybound(0, 45)

ax4.set_yticks([0,10,20,30,40])



# legend = ax1.legend(framealpha=0, ncol=1, loc='best',
#               prop={'size': 24, 'family': 'arial'}, handlelength=0)
#
# for line,text in zip(legend.get_lines(), legend.get_texts()):
#     text.set_color(line.get_color())



plt.tight_layout(pad=1)

# plt.savefig(main_path+'X-r2r3-fig.png', dpi=300, bbox_inches='tight')
plt.show()



